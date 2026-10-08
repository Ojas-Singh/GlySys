//! Single-precision cluster-pair engine for periodic nonbonded dynamics.
//!
//! This is the CPU counterpart of the tiled GPU engine and follows the same
//! physics as [`crate::pbc::PbcForceField`]: Amber Lennard-Jones plus
//! reaction-field electrostatics strictly inside the cutoff with the
//! minimum-image convention, exclusions removed, and 1-4 pairs evaluated as
//! plain-Coulomb exceptions with their Amber scales. Pair arithmetic is f32
//! with f64 block reductions (the OpenMM CPU-platform precision model); the
//! f64 evaluator remains the reference oracle.
//!
//! Atoms are ordered along a Hilbert curve of a fine periodic grid and
//! grouped into clusters of eight. For each cluster the list holds the atoms
//! of later clusters within cutoff + skin of its bounding box, each with an
//! eight-bit interaction mask, so every pair within the cutoff is evaluated
//! once. The kernel keeps a cluster's atoms in SIMD lanes and broadcasts one
//! partner at a time. Lists are rebuilt when any atom moves half the skin.
//!
//! Work is split into one contiguous chunk of clusters per worker with a
//! private force buffer; chunks are combined in a fixed order, so results are
//! reproducible for a given thread count.
//!
//! In PME mode ([`ClusterPairEngine::new_pme`]) regular pairs use the Ewald
//! direct-space term `erfc(alpha r)/r` instead of the reaction field; the
//! long-range remainder comes from [`crate::pme::PmeEngine`]. 1-4 exceptions
//! are plain Coulomb in both modes.
use crate::pbc::{BoxVectors, COULOMB};
use crate::{EnergyError, Result};
use glysys::{ParameterizedSystem, Vec3};
use rayon::prelude::*;
use std::collections::BTreeMap;

const LANES: usize = 8;
const CHUNKS_PER_THREAD: usize = 4;
/// Fewest atoms a parallel task of a per-atom loop takes: these loops are a
/// few nanoseconds per atom, less than the cost of handing out a task.
const PER_ATOM_TASK: usize = 512;
/// Below this many atoms the per-atom loops run on the calling thread:
/// waking the pool for one of them takes longer than the loop itself.
const PARALLEL_ATOMS: usize = 32_768;
/// The same for combining the force buffers, counted in buffer entries.
const PARALLEL_SUMS: usize = 400_000;
const SORT_CELL_ANGSTROM: f64 = 3.0;
const MAX_SORT_BITS: u32 = 7;
/// Polynomial terms of the Ewald direct-space corrections.
const EWALD_TERMS: usize = 16;
/// Largest accepted error of the single-precision Ewald pair force, as a
/// fraction of the pair's bare Coulomb force.
const EWALD_FORCE_TOLERANCE: f64 = 1e-5;

/// Pair electrostatics of the two engine modes.
enum PairElectrostatics {
    ReactionField { solvent_dielectric: f64 },
    Ewald { alpha_per_angstrom: f64 },
}

/// Single-precision Ewald direct-space pair term of the PME mode.
///
/// `erfc(alpha r)/r = 1/r - alpha P_V(t)` and its `(dE/dr)/r` is
/// `alpha^3 P_F(t) - 1/r^3`, with `t = (alpha r)^2` and `P_V`, `P_F` the
/// smooth long-range functions of [`crate::pme`]. The kernel already has
/// `1/r` and `1/r^3`, so it only needs the two corrections: polynomials of
/// degree 15 in `u = 2 r^2/rc^2 - 1` on [-1, 1], interpolated at Chebyshev
/// nodes for this `alpha` and cutoff and evaluated with Estrin's scheme
/// (a short dependency chain, no table lookups). The f32 result carries
/// rounding of a few 1e-7 of the bare Coulomb term, like every other pair
/// quantity in this kernel; `error` records what the fit achieves.
#[derive(Clone, Copy, Debug)]
struct EwaldKernel {
    alpha: f64,
    /// `2/rc^2`: maps `r^2` onto `u + 1`.
    scale: f32,
    /// `alpha^3 P_F` and `alpha P_V` as monomials in `u`.
    force: [f32; EWALD_TERMS],
    energy: [f32; EWALD_TERMS],
    /// Largest errors of the pair force and energy over (0, rc], relative to
    /// the bare Coulomb force and energy of the pair.
    error: [f64; 2],
}

/// Stand-in for the reaction-field instantiations, which never read it.
static NO_EWALD: EwaldKernel = EwaldKernel {
    alpha: 0.,
    scale: 0.,
    force: [0.; EWALD_TERMS],
    energy: [0.; EWALD_TERMS],
    error: [0.; 2],
};

impl EwaldKernel {
    fn new(alpha: f64, cutoff: f64) -> Result<Self> {
        if !alpha.is_finite() || alpha <= 0. {
            return Err(EnergyError::InvalidConfiguration(
                "Ewald coefficient must be positive".into(),
            ));
        }
        let kernel = Self::fit(alpha, cutoff);
        if !kernel.error[0].is_finite() || kernel.error[0] > EWALD_FORCE_TOLERANCE {
            return Err(EnergyError::InvalidConfiguration(format!(
                "alpha * cutoff = {:.2} is too large for the single-precision Ewald kernel",
                alpha * cutoff
            )));
        }
        Ok(kernel)
    }

    /// Interpolate the corrections for one `alpha` and cutoff and measure
    /// the pair terms exactly as the kernel forms them.
    fn fit(alpha: f64, cutoff: f64) -> Self {
        let reach = (alpha * cutoff).powi(2);
        let at = |u: f64| crate::pme::long_range_pair_functions(0.5 * reach * (u + 1.));
        let mut kernel = Self {
            alpha,
            scale: (2. / (cutoff * cutoff)) as f32,
            force: chebyshev_monomials(|u| alpha.powi(3) * at(u).1).map(|c| c as f32),
            energy: chebyshev_monomials(|u| alpha * at(u).0).map(|c| c as f32),
            error: [0.; 2],
        };
        const SAMPLES: usize = 4096;
        for sample in 1..=SAMPLES {
            let r = (cutoff * sample as f64 / SAMPLES as f64) as f32;
            let r2 = r * r;
            let inv_r2 = 1.0 / r2;
            let inv_r = inv_r2.sqrt();
            let u = (r2 * kernel.scale - 1.0).min(1.0);
            let force = ewald_polynomial(&kernel.force, u) - inv_r * inv_r2;
            let energy = inv_r - ewald_polynomial(&kernel.energy, u);
            let exact_r = f64::from(r2).sqrt();
            let (potential, slope) =
                crate::pme::long_range_pair_functions((alpha * exact_r).powi(2));
            let exact_force = alpha.powi(3) * slope - exact_r.powi(-3);
            let exact_energy = 1. / exact_r - alpha * potential;
            kernel.error[0] =
                kernel.error[0].max((f64::from(force) - exact_force).abs() * exact_r.powi(3));
            kernel.error[1] =
                kernel.error[1].max((f64::from(energy) - exact_energy).abs() * exact_r);
        }
        kernel
    }
}

/// Monomial coefficients of the polynomial that interpolates `f` at the
/// `EWALD_TERMS` Chebyshev nodes of [-1, 1].
fn chebyshev_monomials(f: impl Fn(f64) -> f64) -> [f64; EWALD_TERMS] {
    let n = EWALD_TERMS;
    let angle = |k: usize| std::f64::consts::PI * (k as f64 + 0.5) / n as f64;
    let values: [f64; EWALD_TERMS] = std::array::from_fn(|k| f(angle(k).cos()));
    // T_0 = 1, T_1 = u, T_{j+1} = 2 u T_j - T_{j-1}.
    let mut previous = [0f64; EWALD_TERMS];
    let mut current = [0f64; EWALD_TERMS];
    let mut monomials = [0f64; EWALD_TERMS];
    for j in 0..n {
        let sum: f64 = (0..n)
            .map(|k| values[k] * (j as f64 * angle(k)).cos())
            .sum();
        let coefficient = if j == 0 { sum } else { 2. * sum } / n as f64;
        let mut next = [0f64; EWALD_TERMS];
        match j {
            0 => next[0] = 1.,
            1 => next[1] = 1.,
            _ => {
                for k in 0..n {
                    next[k] = if k > 0 { 2. * current[k - 1] } else { 0. } - previous[k];
                }
            }
        }
        for k in 0..n {
            monomials[k] += coefficient * next[k];
        }
        previous = current;
        current = next;
    }
    monomials
}

/// Degree-15 polynomial by Estrin's scheme. The AVX2 kernel performs the
/// same operations in the same order.
#[inline(always)]
fn ewald_polynomial(c: &[f32; EWALD_TERMS], u: f32) -> f32 {
    let u2 = u * u;
    let u4 = u2 * u2;
    let u8 = u4 * u4;
    let p0 = c[0] + c[1] * u;
    let p1 = c[2] + c[3] * u;
    let p2 = c[4] + c[5] * u;
    let p3 = c[6] + c[7] * u;
    let p4 = c[8] + c[9] * u;
    let p5 = c[10] + c[11] * u;
    let p6 = c[12] + c[13] * u;
    let p7 = c[14] + c[15] * u;
    let q0 = p0 + p1 * u2;
    let q1 = p2 + p3 * u2;
    let q2 = p4 + p5 * u2;
    let q3 = p6 + p7 * u2;
    let r0 = q0 + q1 * u4;
    let r1 = q2 + q3 * u4;
    r0 + r1 * u8
}

#[derive(Clone, Copy, Debug)]
struct Exception {
    a: u32,
    b: u32,
    qq: f64,
    epsilon: f64,
    sigma: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Cluster {
    center: [f32; 3],
    per_pair: bool,
    diagonal: [u8; LANES],
}

/// Partners of one cluster (sorted slots) with their lane masks. Kept per
/// cluster and reused across rebuilds, so a rebuild allocates nothing.
#[derive(Clone, Debug, Default)]
struct ClusterList {
    atoms: Vec<u32>,
    masks: Vec<u8>,
}

/// Nonbonded totals of one evaluation (kcal/mol). Gradients are added to the
/// caller's array.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClusterPairTotals {
    pub van_der_waals: f64,
    pub electrostatics: f64,
    /// Pair virial `-sum f.r` over the minimum-image vectors.
    pub virial: f64,
    /// Pair virial split into Lennard-Jones and electrostatic parts.
    pub virial_pair_split: [f64; 2],
}

/// Reusable periodic nonbonded engine for one prepared system.
#[derive(Clone)]
pub struct ClusterPairEngine {
    n: usize,
    cutoff: f64,
    skin: f64,
    krf: f64,
    crf: f64,
    ewald: Option<EwaldKernel>,
    charge: Vec<f32>,
    sigma: Vec<f32>,
    sqrt_epsilon: Vec<f32>,
    molecule: Vec<u32>,
    window: Vec<u64>,
    far: Vec<(u32, u32)>,
    exceptions: Vec<Exception>,
    // List state.
    box_at_build: Option<BoxVectors>,
    reference: Vec<Vec3>,
    order: Vec<u32>,
    clusters: Vec<Cluster>,
    lists: Vec<ClusterList>,
    rebuilds: u64,
    // Per-evaluation scratch, indexed by sorted slot.
    sorted: Vec<[f32; 4]>,
    sorted_lj: Vec<[f32; 2]>,
    buffers: Vec<Vec<[f32; 3]>>,
    plan: WorkPlan,
}

/// How an evaluation is divided among workers. It depends on the lists and
/// on the size of the thread pool only, so it is kept until either changes.
///
/// Every chunk of clusters adds forces into a buffer of its own. A chunk
/// reaches its own atoms and their listed partners, a small part of the
/// system, so the plan records which slots each chunk writes and, for every
/// slot, which chunks write it. Clearing and combining the buffers then cost
/// in proportion to the pair list, not to atoms times chunks.
#[derive(Clone, Debug, Default)]
struct WorkPlan {
    workers: usize,
    /// `rebuilds` of the lists this plan was made for.
    lists: u64,
    /// Chunk `c` evaluates clusters `bounds[c]..bounds[c + 1]`.
    bounds: Vec<usize>,
    /// Sorted slots chunk `c` writes.
    touched: Vec<Vec<u32>>,
    /// Chunks writing slot `s`, ascending: `writers[offsets[s]..offsets[s + 1]]`.
    offsets: Vec<u32>,
    writers: Vec<u32>,
    /// Sorted slot of every atom.
    rank: Vec<u32>,
}

impl ClusterPairEngine {
    pub fn new(
        system: &ParameterizedSystem,
        cutoff: f64,
        skin: f64,
        solvent_dielectric: f64,
    ) -> Result<Self> {
        let electrostatics = PairElectrostatics::ReactionField { solvent_dielectric };
        Self::build(system, cutoff, skin, electrostatics)
    }

    /// PME mode: regular pairs inside the cutoff interact through the Ewald
    /// direct-space term `qq erfc(alpha r)/r`. Pair it with a
    /// [`crate::pme::PmeEngine`] of the same `alpha` for the long-range part
    /// ([`crate::pbc::PbcForceField::evaluate_with_cluster_pme`]).
    pub fn new_pme(
        system: &ParameterizedSystem,
        cutoff: f64,
        skin: f64,
        alpha_per_angstrom: f64,
    ) -> Result<Self> {
        let electrostatics = PairElectrostatics::Ewald { alpha_per_angstrom };
        Self::build(system, cutoff, skin, electrostatics)
    }

    fn build(
        system: &ParameterizedSystem,
        cutoff: f64,
        skin: f64,
        electrostatics: PairElectrostatics,
    ) -> Result<Self> {
        let n = system.atom_count();
        if n == 0 || n >= u32::MAX as usize / 2 {
            return Err(EnergyError::InvalidConfiguration(
                "cluster engine needs between one and 2^31 atoms".into(),
            ));
        }
        if !cutoff.is_finite() || cutoff <= 0. || !skin.is_finite() || skin < 0. {
            return Err(EnergyError::InvalidConfiguration(
                "neighbor cutoff must be positive and skin non-negative".into(),
            ));
        }
        let (krf, crf, ewald) = match electrostatics {
            PairElectrostatics::ReactionField { solvent_dielectric } => {
                if !solvent_dielectric.is_finite() || solvent_dielectric < 1. {
                    return Err(EnergyError::InvalidConfiguration(
                        "reaction-field dielectric must be at least one".into(),
                    ));
                }
                let e = solvent_dielectric;
                let krf = (e - 1.) / (2. * e + 1.) / cutoff.powi(3);
                let crf = 3. * e / (2. * e + 1.) / cutoff;
                (krf, crf, None)
            }
            PairElectrostatics::Ewald { alpha_per_angstrom } => {
                (0., 0., Some(EwaldKernel::new(alpha_per_angstrom, cutoff)?))
            }
        };
        let molecule = molecule_ids(system);
        let mut window = vec![0u64; n];
        let mut far = Vec::new();
        for (atom, set) in system.exclusions().iter().enumerate() {
            for &other in set {
                if molecule[other] != molecule[atom] {
                    return Err(EnergyError::InvalidConfiguration(
                        "exclusion between separate molecules".into(),
                    ));
                }
                let offset = other as i64 - atom as i64 + 32;
                if (0..64).contains(&offset) {
                    window[atom] |= 1 << offset;
                } else {
                    far.push((atom as u32, other as u32));
                }
            }
        }
        // 1-4 pairs are excluded from the pair kernel and evaluated below.
        let mut one_four = BTreeMap::new();
        for (pair, scee, scnb) in system.one_four_pairs() {
            let key = (pair[0].min(pair[1]), pair[0].max(pair[1]));
            one_four.insert(key, (scee, scnb));
        }
        let mut exceptions = Vec::with_capacity(one_four.len());
        for (&(a, b), &(scee, scnb)) in &one_four {
            for (atom, other) in [(a, b), (b, a)] {
                let offset = other as i64 - atom as i64 + 32;
                if (0..64).contains(&offset) {
                    window[atom] |= 1 << offset;
                } else {
                    far.push((atom as u32, other as u32));
                }
            }
            let (first, second) = (&system.atoms()[a], &system.atoms()[b]);
            exceptions.push(Exception {
                a: a as u32,
                b: b as u32,
                qq: COULOMB * first.charge() * second.charge() / scee,
                epsilon: (first.lennard_jones_epsilon() * second.lennard_jones_epsilon())
                    .max(0.)
                    .sqrt()
                    / scnb,
                sigma: first.lennard_jones_radius() + second.lennard_jones_radius(),
            });
        }
        far.sort_unstable();
        far.dedup();
        let atoms = system.atoms();
        Ok(Self {
            n,
            cutoff,
            skin,
            krf,
            crf,
            ewald,
            charge: atoms.iter().map(|a| a.charge() as f32).collect(),
            sigma: atoms
                .iter()
                .map(|a| a.lennard_jones_radius() as f32)
                .collect(),
            sqrt_epsilon: atoms
                .iter()
                .map(|a| a.lennard_jones_epsilon().max(0.).sqrt() as f32)
                .collect(),
            molecule,
            window,
            far,
            exceptions,
            box_at_build: None,
            reference: Vec::new(),
            order: Vec::new(),
            clusters: Vec::new(),
            lists: Vec::new(),
            rebuilds: 0,
            sorted: Vec::new(),
            sorted_lj: Vec::new(),
            buffers: Vec::new(),
            plan: WorkPlan::default(),
        })
    }

    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Ewald coefficient (1/angstrom) of an engine in PME mode; `None` for
    /// reaction field.
    pub fn ewald_alpha(&self) -> Option<f64> {
        self.ewald.map(|ewald| ewald.alpha)
    }

    /// Largest errors of the single-precision Ewald pair force and energy
    /// over all distances up to the cutoff, as fractions of the pair's bare
    /// Coulomb force and energy (PME mode only).
    pub fn ewald_pair_error(&self) -> Option<[f64; 2]> {
        self.ewald.map(|ewald| ewald.error)
    }

    /// Completed list builds since construction.
    pub fn rebuild_count(&self) -> u64 {
        self.rebuilds
    }

    /// Number of listed (cluster, partner) entries; each covers eight pairs.
    pub fn list_len(&self) -> usize {
        self.lists.iter().map(|list| list.atoms.len()).sum()
    }

    /// Force the next evaluation to rebuild its lists.
    pub fn invalidate(&mut self) {
        self.box_at_build = None;
    }

    fn needs_rebuild(&self, wrapped: &[Vec3], box_vec: &BoxVectors) -> bool {
        let Some(built) = self.box_at_build else {
            return true;
        };
        if self.reference.len() != wrapped.len() {
            return true;
        }
        // A pair listed at separation r in the box of the build is now at
        // least r - 2d - |dL| apart, with d the largest displacement and dL
        // the change of the box lengths, so the lists hold while
        // 2d + |dL| stays inside the skin. A barostat changes the box by a
        // part in 10^5 at a time, far less than the skin; the margin on
        // |dL| covers atoms re-imaged across the changed box.
        let change = {
            let (now, then) = (box_vec.as_array(), built.as_array());
            (0..3).map(|d| (now[d] - then[d]).powi(2)).sum::<f64>().sqrt()
        };
        let allowance = 0.5 * (self.skin - 3.0 * change);
        if allowance <= 0. {
            return true;
        }
        let limit = allowance * allowance;
        let moved = |(p, r): (&Vec3, &Vec3)| {
            let d = box_vec.displacement(*p, *r);
            d.x * d.x + d.y * d.y + d.z * d.z > limit
        };
        if wrapped.len() >= PARALLEL_ATOMS {
            wrapped
                .par_iter()
                .zip(self.reference.par_iter())
                .with_min_len(PER_ATOM_TASK)
                .any(moved)
        } else {
            wrapped.iter().zip(self.reference.iter()).any(moved)
        }
    }

    fn rebuild(&mut self, wrapped: &[Vec3], box_vec: &BoxVectors) -> Result<()> {
        if self.cutoff >= 0.5 * box_vec.x.min(box_vec.y).min(box_vec.z) {
            return Err(EnergyError::InvalidConfiguration(
                "cutoff must be below half the shortest box edge".into(),
            ));
        }
        let n = self.n;
        let lengths = box_vec.as_array();
        let longest = lengths.iter().copied().fold(0.0f64, f64::max);
        let bits =
            ((longest / SORT_CELL_ANGSTROM).max(2.0).log2().round() as u32).clamp(1, MAX_SORT_BITS);
        let grid = 1u32 << bits;
        let cell_of = |p: &Vec3| -> [u32; 3] {
            let c = [p.x, p.y, p.z];
            std::array::from_fn(|axis| {
                ((c[axis] / lengths[axis] * f64::from(grid)).floor() as i64)
                    .clamp(0, i64::from(grid) - 1) as u32
            })
        };
        // Deterministic order: Hilbert cell, then atom index.
        let mut keyed: Vec<(u32, u32)> = wrapped
            .par_iter()
            .enumerate()
            .map(|(atom, p)| (hilbert_index(cell_of(p), bits), atom as u32))
            .collect();
        keyed.par_sort_unstable();
        let clusters = n.div_ceil(LANES);
        let mut order = vec![u32::MAX; clusters * LANES];
        for (slot, &(_, atom)) in keyed.iter().enumerate() {
            order[slot] = atom;
        }
        // Sorted slot ranges per Hilbert cell, for neighbor enumeration.
        let cells = 1usize << (3 * bits);
        let mut cell_start = vec![0u32; cells + 1];
        for &(key, _) in &keyed {
            cell_start[key as usize + 1] += 1;
        }
        for cell in 0..cells {
            cell_start[cell + 1] += cell_start[cell];
        }
        let mut cell_key = vec![0u32; cells];
        for x in 0..grid {
            for y in 0..grid {
                for z in 0..grid {
                    let index = ((x * grid + y) * grid + z) as usize;
                    cell_key[index] = hilbert_index([x, y, z], bits);
                }
            }
        }
        let positions: Vec<Vec3> = order
            .iter()
            .map(|&atom| {
                if atom == u32::MAX {
                    Vec3 {
                        x: 0.,
                        y: 0.,
                        z: 0.,
                    }
                } else {
                    wrapped[atom as usize]
                }
            })
            .collect();
        let bounds: Vec<([f64; 3], [f64; 3])> = (0..clusters)
            .into_par_iter()
            .map(|cluster| {
                let anchor = positions[cluster * LANES];
                let mut low = [f64::INFINITY; 3];
                let mut high = [f64::NEG_INFINITY; 3];
                for lane in 0..LANES {
                    if order[cluster * LANES + lane] == u32::MAX {
                        continue;
                    }
                    let d = box_vec.displacement(positions[cluster * LANES + lane], anchor);
                    for (axis, value) in [d.x, d.y, d.z].into_iter().enumerate() {
                        low[axis] = low[axis].min(value);
                        high[axis] = high[axis].max(value);
                    }
                }
                let a = [anchor.x, anchor.y, anchor.z];
                let center = std::array::from_fn(|axis| a[axis] + 0.5 * (low[axis] + high[axis]));
                let half = std::array::from_fn(|axis| 0.5 * (high[axis] - low[axis]));
                (center, half)
            })
            .collect();
        let radius = self.cutoff + self.skin;
        let radius2 = radius * radius;
        let cell_size: [f64; 3] = std::array::from_fn(|axis| lengths[axis] / f64::from(grid));
        let molecule = &self.molecule;
        let window = &self.window;
        let far = &self.far;
        let excluded = |a: u32, b: u32| -> bool {
            let offset = b as i64 - a as i64 + 32;
            if (0..64).contains(&offset) {
                return (window[a as usize] >> offset) & 1 != 0;
            }
            far.binary_search(&(a, b)).is_ok()
        };
        let lengths_f = lengths;
        self.lists.resize_with(clusters, ClusterList::default);
        self.clusters.resize(clusters, Cluster::default());
        let skin = self.skin;
        let cutoff = self.cutoff;
        self.lists
            .par_iter_mut()
            .zip(self.clusters.par_iter_mut())
            .enumerate()
            .for_each_init(Vec::new, |candidates, (cluster, (list, info))| {
                let (center, half) = bounds[cluster];
                list.atoms.clear();
                list.masks.clear();
                candidates.clear();
                // Cells overlapping the box grown by the list radius.
                let mut ranges = [(0i64, 0i64); 3];
                for axis in 0..3 {
                    let low =
                        ((center[axis] - half[axis] - radius) / cell_size[axis]).floor() as i64;
                    let high =
                        ((center[axis] + half[axis] + radius) / cell_size[axis]).floor() as i64;
                    let span = (high - low + 1).min(i64::from(grid));
                    ranges[axis] = (low, low + span - 1);
                }
                let g = i64::from(grid);
                for x in ranges[0].0..=ranges[0].1 {
                    for y in ranges[1].0..=ranges[1].1 {
                        for z in ranges[2].0..=ranges[2].1 {
                            let index = ((x.rem_euclid(g) * g + y.rem_euclid(g)) * g
                                + z.rem_euclid(g)) as usize;
                            let key = cell_key[index] as usize;
                            let first = cell_start[key].max(((cluster + 1) * LANES) as u32);
                            for slot in first..cell_start[key + 1].max(first) {
                                let p = positions[slot as usize];
                                let d = box_vec.displacement(
                                    p,
                                    Vec3 {
                                        x: center[0],
                                        y: center[1],
                                        z: center[2],
                                    },
                                );
                                let gx = (d.x.abs() - half[0]).max(0.);
                                let gy = (d.y.abs() - half[1]).max(0.);
                                let gz = (d.z.abs() - half[2]).max(0.);
                                if gx * gx + gy * gy + gz * gz < radius2 {
                                    candidates.push(slot);
                                }
                            }
                        }
                    }
                }
                // Ascending partner slots keep the kernel's gathers and
                // scatters sequential in memory.
                candidates.sort_unstable();
                let own: [u32; LANES] = std::array::from_fn(|lane| order[cluster * LANES + lane]);
                let own_mol: [u32; LANES] = std::array::from_fn(|lane| {
                    if own[lane] == u32::MAX {
                        u32::MAX
                    } else {
                        molecule[own[lane] as usize]
                    }
                });
                let valid: u8 = (0..LANES).fold(0, |bits, lane| {
                    bits | (u8::from(own[lane] != u32::MAX) << lane)
                });
                for &slot in candidates.iter() {
                    let atom = order[slot as usize];
                    let partner_mol = molecule[atom as usize];
                    let mut mask = valid;
                    for lane in 0..LANES {
                        if own_mol[lane] == partner_mol && excluded(own[lane], atom) {
                            mask &= !(1 << lane);
                        }
                    }
                    if mask != 0 {
                        list.atoms.push(slot);
                        list.masks.push(mask);
                    }
                }
                // Diagonal masks are indexed by the partner lane: bit `l` is
                // set when lane `l` < partner lane interacts with it.
                let mut diagonal = [0u8; LANES];
                for (partner_lane, mask) in diagonal.iter_mut().enumerate() {
                    let partner = own[partner_lane];
                    if partner == u32::MAX {
                        continue;
                    }
                    for lane in 0..partner_lane {
                        if own[lane] != u32::MAX
                            && !(own_mol[lane] == own_mol[partner_lane]
                                && excluded(own[lane], partner))
                        {
                            *mask |= 1 << lane;
                        }
                    }
                }
                let reach: [f64; 3] = std::array::from_fn(|axis| half[axis] + cutoff + 0.5 * skin);
                *info = Cluster {
                    center: center.map(|v| v as f32),
                    per_pair: (0..3).any(|axis| reach[axis] >= 0.5 * lengths_f[axis]),
                    diagonal,
                };
            });
        self.sorted_lj = order
            .iter()
            .map(|&atom| {
                if atom == u32::MAX {
                    [0.0, 0.0]
                } else {
                    [self.sigma[atom as usize], self.sqrt_epsilon[atom as usize]]
                }
            })
            .collect();
        self.order = order;
        self.reference = wrapped.to_vec();
        self.box_at_build = Some(*box_vec);
        self.rebuilds += 1;
        Ok(())
    }

    /// The division of the current lists among `workers` chunks.
    fn work_plan(&self, workers: usize) -> WorkPlan {
        let clusters = self.clusters.len();
        let slots = self.order.len();
        let total_work: usize = self.list_len() + clusters * LANES;
        let mut bounds = Vec::with_capacity(workers + 1);
        bounds.push(0usize);
        let mut acc = 0usize;
        let mut next = 1usize;
        for (index, list) in self.lists.iter().enumerate() {
            acc += list.atoms.len() + LANES;
            while next < workers && acc * workers >= total_work * next {
                bounds.push(index + 1);
                next += 1;
            }
        }
        while bounds.len() <= workers {
            bounds.push(clusters);
        }
        let lists = &self.lists;
        let touched: Vec<Vec<u32>> = (0..workers)
            .into_par_iter()
            .map(|chunk| {
                let mut slots_of_chunk = Vec::new();
                for index in bounds[chunk]..bounds[chunk + 1] {
                    slots_of_chunk.extend((index * LANES..(index + 1) * LANES).map(|s| s as u32));
                    slots_of_chunk.extend_from_slice(&lists[index].atoms);
                }
                slots_of_chunk.sort_unstable();
                slots_of_chunk.dedup();
                slots_of_chunk
            })
            .collect();
        let mut offsets = vec![0u32; slots + 1];
        for chunk in &touched {
            for &slot in chunk {
                offsets[slot as usize + 1] += 1;
            }
        }
        for slot in 0..slots {
            offsets[slot + 1] += offsets[slot];
        }
        let mut writers = vec![0u32; offsets[slots] as usize];
        let mut cursor = offsets.clone();
        for (chunk, slots_of_chunk) in touched.iter().enumerate() {
            for &slot in slots_of_chunk {
                writers[cursor[slot as usize] as usize] = chunk as u32;
                cursor[slot as usize] += 1;
            }
        }
        let mut rank = vec![0u32; self.n];
        for (slot, &atom) in self.order.iter().enumerate() {
            if atom != u32::MAX {
                rank[atom as usize] = slot as u32;
            }
        }
        WorkPlan {
            workers,
            lists: self.rebuilds,
            bounds,
            touched,
            offsets,
            writers,
            rank,
        }
    }

    /// Add pair and 1-4 exception gradients for unwrapped coordinates to
    /// `gradients` and return the nonbonded totals. Energies and virials are
    /// accumulated only when `observables` is set.
    pub fn evaluate_into(
        &mut self,
        coordinates: &[Vec3],
        box_vec: &BoxVectors,
        gradients: &mut [Vec3],
        observables: bool,
    ) -> Result<ClusterPairTotals> {
        if coordinates.len() != self.n || gradients.len() != self.n {
            return Err(EnergyError::CoordinateCount {
                expected: self.n,
                received: coordinates.len(),
            });
        }
        if coordinates
            .iter()
            .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
        {
            return Err(EnergyError::NonFiniteCoordinate);
        }
        let parallel = coordinates.len() >= PARALLEL_ATOMS;
        let wrapped: Vec<Vec3> = if parallel {
            coordinates
                .par_iter()
                .with_min_len(PER_ATOM_TASK)
                .map(|p| box_vec.wrap(*p))
                .collect()
        } else {
            coordinates.iter().map(|p| box_vec.wrap(*p)).collect()
        };
        if self.needs_rebuild(&wrapped, box_vec) {
            self.rebuild(&wrapped, box_vec)?;
        }
        let order = &self.order;
        let charge = &self.charge;
        self.sorted.resize(order.len(), [0.0; 4]);
        let place = |(slot, &atom): (&mut [f32; 4], &u32)| {
            *slot = if atom == u32::MAX {
                [0.0; 4]
            } else {
                let p = wrapped[atom as usize];
                [p.x as f32, p.y as f32, p.z as f32, charge[atom as usize]]
            };
        };
        if parallel {
            self.sorted
                .par_iter_mut()
                .zip(order.par_iter())
                .with_min_len(PER_ATOM_TASK)
                .for_each(place);
        } else {
            self.sorted.iter_mut().zip(order.iter()).for_each(place);
        }
        // One contiguous chunk of clusters per worker, balanced by listed
        // entries; each chunk owns a sorted-slot force buffer.
        // Several chunks per worker let work stealing balance hybrid cores;
        // the count depends only on the pool size, so results remain
        // reproducible for a given thread count. Buffer memory is capped.
        let threads = rayon::current_num_threads().max(1);
        let memory_cap = ((256usize << 20) / (12 * self.order.len().max(1))).max(threads);
        let workers = (CHUNKS_PER_THREAD * threads).min(memory_cap).max(1);
        if self.plan.workers != workers || self.plan.lists != self.rebuilds {
            self.plan = self.work_plan(workers);
        }
        let plan = &self.plan;
        let bounds = &plan.bounds;
        let slots = order.len();
        if self.buffers.len() != workers || self.buffers.first().is_some_and(|b| b.len() != slots) {
            self.buffers = (0..workers).map(|_| vec![[0.0f32; 3]; slots]).collect();
        }
        let params = KernelParams {
            cutoff2: (self.cutoff * self.cutoff) as f32,
            krf: self.krf as f32,
            crf: self.crf as f32,
            box_len: box_vec.as_array().map(|v| v as f32),
            observables,
        };
        let sorted = &self.sorted;
        let sorted_lj = &self.sorted_lj;
        let cluster_list = &self.clusters;
        let lists = &self.lists;
        let ewald = self.ewald;
        let chunk_totals: Vec<[f64; 5]> = self
            .buffers
            .par_iter_mut()
            .enumerate()
            .map(|(chunk, buffer)| {
                for &slot in &plan.touched[chunk] {
                    buffer[slot as usize] = [0.0; 3];
                }
                let mut totals = [0.0f64; 5];
                let range = bounds[chunk]..bounds[chunk + 1];
                for (cluster, data) in cluster_list[range.clone()].iter().enumerate() {
                    let index = range.start + cluster;
                    let (atoms, masks) = (&lists[index].atoms, &lists[index].masks);
                    let result = match &ewald {
                        None => run_cluster(
                            &params, index, data, sorted, sorted_lj, atoms, masks, buffer,
                        ),
                        Some(ewald) => run_cluster_pme(
                            &params, ewald, index, data, sorted, sorted_lj, atoms, masks, buffer,
                        ),
                    };
                    for (total, value) in totals.iter_mut().zip(result) {
                        *total += f64::from(value);
                    }
                }
                totals
            })
            .collect();
        let mut totals = ClusterPairTotals::default();
        for chunk in &chunk_totals {
            totals.van_der_waals += chunk[0];
            totals.electrostatics += chunk[1];
            totals.virial += chunk[2];
            totals.virial_pair_split[0] += chunk[3];
            totals.virial_pair_split[1] += chunk[2] - chunk[3];
        }
        let buffers = &self.buffers;
        // Map sorted slots back to atoms, combining the chunks that wrote a
        // slot in chunk order.
        let combine = |(gradient, &slot): (&mut Vec3, &u32)| {
            let mut sum = [0.0f64; 3];
            let writers =
                plan.offsets[slot as usize] as usize..plan.offsets[slot as usize + 1] as usize;
            for &chunk in &plan.writers[writers] {
                let value = buffers[chunk as usize][slot as usize];
                sum[0] += f64::from(value[0]);
                sum[1] += f64::from(value[1]);
                sum[2] += f64::from(value[2]);
            }
            gradient.x += sum[0];
            gradient.y += sum[1];
            gradient.z += sum[2];
        };
        if plan.writers.len() >= PARALLEL_SUMS {
            gradients
                .par_iter_mut()
                .zip(plan.rank.par_iter())
                .with_min_len(PER_ATOM_TASK)
                .for_each(combine);
        } else {
            gradients.iter_mut().zip(plan.rank.iter()).for_each(combine);
        }
        // 1-4 exceptions in f64; plain Coulomb scaled by 1/scee.
        let cutoff2 = self.cutoff * self.cutoff;
        for exception in &self.exceptions {
            let (a, b) = (exception.a as usize, exception.b as usize);
            let d = box_vec.displacement(wrapped[a], wrapped[b]);
            let r2 = d.x * d.x + d.y * d.y + d.z * d.z;
            if r2 > cutoff2 {
                continue;
            }
            let r = r2.sqrt().max(1e-8);
            let ratio6 = (exception.sigma / r).powi(6);
            let flj = 12. * exception.epsilon * (ratio6 - ratio6 * ratio6) / r;
            let dcoul = -exception.qq / (r * r);
            let fmag = (flj + dcoul) / r;
            gradients[a].x += fmag * d.x;
            gradients[a].y += fmag * d.y;
            gradients[a].z += fmag * d.z;
            gradients[b].x -= fmag * d.x;
            gradients[b].y -= fmag * d.y;
            gradients[b].z -= fmag * d.z;
            if observables {
                totals.van_der_waals += exception.epsilon * (ratio6 * ratio6 - 2. * ratio6);
                totals.electrostatics += exception.qq / r;
                let w = fmag * r2;
                totals.virial -= w;
                let w_lj = (flj / r) * r2;
                totals.virial_pair_split[0] -= w_lj;
                totals.virial_pair_split[1] -= w - w_lj;
            }
        }
        if !totals.van_der_waals.is_finite() || !totals.electrostatics.is_finite() {
            return Err(EnergyError::InvalidConfiguration(
                "nonfinite nonbonded energy".into(),
            ));
        }
        Ok(totals)
    }
}

#[derive(Clone, Copy)]
struct KernelParams {
    cutoff2: f32,
    krf: f32,
    crf: f32,
    box_len: [f32; 3],
    observables: bool,
}

#[allow(clippy::too_many_arguments)]
fn run_cluster(
    params: &KernelParams,
    index: usize,
    cluster: &Cluster,
    sorted: &[[f32; 4]],
    sorted_lj: &[[f32; 2]],
    list_atoms: &[u32],
    list_masks: &[u8],
    buffer: &mut [[f32; 3]],
) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            macro_rules! call {
                ($obs:expr, $per_pair:expr) => {
                    // SAFETY: AVX2 support was detected at runtime.
                    unsafe {
                        avx2::cluster::<$obs, $per_pair, false>(
                            params, &NO_EWALD, index, cluster, sorted, sorted_lj, list_atoms,
                            list_masks, buffer,
                        )
                    }
                };
            }
            return match (params.observables, cluster.per_pair) {
                (false, false) => call!(false, false),
                (false, true) => call!(false, true),
                (true, false) => call!(true, false),
                (true, true) => call!(true, true),
            };
        }
    }
    cluster_kernel(
        params, index, cluster, sorted, sorted_lj, list_atoms, list_masks, buffer,
    )
}

/// [`run_cluster`] for the PME mode.
#[allow(clippy::too_many_arguments)]
fn run_cluster_pme(
    params: &KernelParams,
    ewald: &EwaldKernel,
    index: usize,
    cluster: &Cluster,
    sorted: &[[f32; 4]],
    sorted_lj: &[[f32; 2]],
    list_atoms: &[u32],
    list_masks: &[u8],
    buffer: &mut [[f32; 3]],
) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            macro_rules! call {
                ($obs:expr, $per_pair:expr) => {
                    // SAFETY: AVX2 support was detected at runtime.
                    unsafe {
                        avx2::cluster::<$obs, $per_pair, true>(
                            params, ewald, index, cluster, sorted, sorted_lj, list_atoms,
                            list_masks, buffer,
                        )
                    }
                };
            }
            return match (params.observables, cluster.per_pair) {
                (false, false) => call!(false, false),
                (false, true) => call!(false, true),
                (true, false) => call!(true, false),
                (true, true) => call!(true, true),
            };
        }
    }
    cluster_kernel_mode::<true>(
        params, ewald, index, cluster, sorted, sorted_lj, list_atoms, list_masks, buffer,
    )
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    //! Explicit AVX2 form of [`super::cluster_kernel`]. Every lane performs
    //! the same IEEE operations in the same order as the portable kernel (no
    //! fused multiply-add, ties-to-even rounding), so both produce identical
    //! bits.
    use super::{Cluster, EWALD_TERMS, EwaldKernel, KernelParams, LANES};
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    fn hsum(v: __m256) -> f32 {
        // (0+4, 1+5, 2+6, 3+7), then (0+2, 1+3), then (0+1): `lane_sum`.
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let t = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)))
    }

    #[inline(always)]
    fn near(value: f32, center: f32, length: f32) -> f32 {
        super::nearest_image(value, center, length)
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn cluster<const OBS: bool, const PER_PAIR: bool, const PME: bool>(
        params: &KernelParams,
        ewald: &EwaldKernel,
        index: usize,
        cluster: &Cluster,
        sorted: &[[f32; 4]],
        sorted_lj: &[[f32; 2]],
        list_atoms: &[u32],
        list_masks: &[u8],
        buffer: &mut [[f32; 3]],
    ) -> [f32; 4] {
        let base = index * LANES;
        let c = cluster.center;
        let l = params.box_len;
        let mut lx = [0f32; LANES];
        let mut ly = [0f32; LANES];
        let mut lz = [0f32; LANES];
        let mut lq = [0f32; LANES];
        let mut ls = [0f32; LANES];
        let mut le = [0f32; LANES];
        for lane in 0..LANES {
            let p = sorted[base + lane];
            lx[lane] = near(p[0], c[0], l[0]);
            ly[lane] = near(p[1], c[1], l[1]);
            lz[lane] = near(p[2], c[2], l[2]);
            lq[lane] = (crate::pbc::COULOMB as f32) * p[3];
            ls[lane] = sorted_lj[base + lane][0];
            le[lane] = sorted_lj[base + lane][1];
        }
        // SAFETY: each pointer addresses a live `[f32; LANES]` (8 floats).
        let (xi, yi, zi, qi, si, ei) = unsafe {
            (
                _mm256_loadu_ps(lx.as_ptr()),
                _mm256_loadu_ps(ly.as_ptr()),
                _mm256_loadu_ps(lz.as_ptr()),
                _mm256_loadu_ps(lq.as_ptr()),
                _mm256_loadu_ps(ls.as_ptr()),
                _mm256_loadu_ps(le.as_ptr()),
            )
        };
        let lane_bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
        let zero_i = _mm256_setzero_si256();
        let one = _mm256_set1_ps(1.0);
        let tiny = _mm256_set1_ps(1e-12);
        let cutoff2 = _mm256_set1_ps(params.cutoff2);
        let krf = _mm256_set1_ps(params.krf);
        let krf2 = _mm256_set1_ps(2.0 * params.krf);
        let crf = _mm256_set1_ps(params.crf);
        let twelve = _mm256_set1_ps(12.0);
        let two = _mm256_set1_ps(2.0);
        let lx_v = _mm256_set1_ps(l[0]);
        let ly_v = _mm256_set1_ps(l[1]);
        let lz_v = _mm256_set1_ps(l[2]);
        // Ewald corrections of the PME mode, broadcast once per cluster.
        let ewald_scale = _mm256_set1_ps(ewald.scale);
        let mut ewald_force = [_mm256_setzero_ps(); EWALD_TERMS];
        let mut ewald_energy = [_mm256_setzero_ps(); EWALD_TERMS];
        if PME {
            for term in 0..EWALD_TERMS {
                ewald_force[term] = _mm256_set1_ps(ewald.force[term]);
                if OBS {
                    ewald_energy[term] = _mm256_set1_ps(ewald.energy[term]);
                }
            }
        }
        // `super::ewald_polynomial`, operation for operation.
        macro_rules! polynomial {
            ($c:expr, $u:expr) => {{
                let c = &$c;
                let u = $u;
                let u2 = _mm256_mul_ps(u, u);
                let u4 = _mm256_mul_ps(u2, u2);
                let u8 = _mm256_mul_ps(u4, u4);
                let p0 = _mm256_add_ps(c[0], _mm256_mul_ps(c[1], u));
                let p1 = _mm256_add_ps(c[2], _mm256_mul_ps(c[3], u));
                let p2 = _mm256_add_ps(c[4], _mm256_mul_ps(c[5], u));
                let p3 = _mm256_add_ps(c[6], _mm256_mul_ps(c[7], u));
                let p4 = _mm256_add_ps(c[8], _mm256_mul_ps(c[9], u));
                let p5 = _mm256_add_ps(c[10], _mm256_mul_ps(c[11], u));
                let p6 = _mm256_add_ps(c[12], _mm256_mul_ps(c[13], u));
                let p7 = _mm256_add_ps(c[14], _mm256_mul_ps(c[15], u));
                let q0 = _mm256_add_ps(p0, _mm256_mul_ps(p1, u2));
                let q1 = _mm256_add_ps(p2, _mm256_mul_ps(p3, u2));
                let q2 = _mm256_add_ps(p4, _mm256_mul_ps(p5, u2));
                let q3 = _mm256_add_ps(p6, _mm256_mul_ps(p7, u2));
                let r0 = _mm256_add_ps(q0, _mm256_mul_ps(q1, u4));
                let r1 = _mm256_add_ps(q2, _mm256_mul_ps(q3, u4));
                _mm256_add_ps(r0, _mm256_mul_ps(r1, u8))
            }};
        }
        let mut fx = _mm256_setzero_ps();
        let mut fy = _mm256_setzero_ps();
        let mut fz = _mm256_setzero_ps();
        let mut e_lj = _mm256_setzero_ps();
        let mut e_rf = _mm256_setzero_ps();
        let mut w_all = _mm256_setzero_ps();
        let mut w_lj = _mm256_setzero_ps();
        macro_rules! round {
            ($v:expr) => {
                _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>($v)
            };
        }
        // A macro rather than a closure keeps the body inside this
        // target-feature function so every intrinsic inlines.
        macro_rules! interact {
            ($slot:expr, $mask:expr) => {{
                let slot: usize = $slot;
                let mask: u8 = $mask;
                let p = sorted[slot];
                let xj = _mm256_set1_ps(near(p[0], c[0], l[0]));
                let yj = _mm256_set1_ps(near(p[1], c[1], l[1]));
                let zj = _mm256_set1_ps(near(p[2], c[2], l[2]));
                let qj = _mm256_set1_ps(p[3]);
                let [sj, ej] = sorted_lj[slot];
                let mut dx = _mm256_sub_ps(xi, xj);
                let mut dy = _mm256_sub_ps(yi, yj);
                let mut dz = _mm256_sub_ps(zi, zj);
                if PER_PAIR {
                    dx = _mm256_sub_ps(dx, _mm256_mul_ps(lx_v, round!(_mm256_div_ps(dx, lx_v))));
                    dy = _mm256_sub_ps(dy, _mm256_mul_ps(ly_v, round!(_mm256_div_ps(dy, ly_v))));
                    dz = _mm256_sub_ps(dz, _mm256_mul_ps(lz_v, round!(_mm256_div_ps(dz, lz_v))));
                }
                let r2 = _mm256_max_ps(
                    _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)),
                        _mm256_mul_ps(dz, dz),
                    ),
                    tiny,
                );
                let listed = _mm256_cmpgt_epi32(
                    _mm256_and_si256(_mm256_set1_epi32(i32::from(mask)), lane_bits),
                    zero_i,
                );
                let inside = _mm256_and_ps(
                    _mm256_castsi256_ps(listed),
                    _mm256_cmp_ps::<_CMP_LT_OQ>(r2, cutoff2),
                );
                let keep = _mm256_and_ps(inside, one);
                let inv_r2 = _mm256_div_ps(keep, r2);
                let inv_r = _mm256_sqrt_ps(inv_r2);
                let sigma = _mm256_add_ps(si, _mm256_set1_ps(sj));
                let eps = _mm256_mul_ps(ei, _mm256_set1_ps(ej));
                let s2 = _mm256_mul_ps(_mm256_mul_ps(sigma, sigma), inv_r2);
                let s6 = _mm256_mul_ps(_mm256_mul_ps(s2, s2), s2);
                let qq = _mm256_mul_ps(_mm256_mul_ps(qi, qj), keep);
                let f_lj = _mm256_mul_ps(
                    _mm256_mul_ps(
                        _mm256_mul_ps(twelve, eps),
                        _mm256_sub_ps(s6, _mm256_mul_ps(s6, s6)),
                    ),
                    inv_r2,
                );
                // Masked lanes can lie beyond the cutoff: clamp to the fit.
                let u = if PME {
                    _mm256_min_ps(_mm256_sub_ps(_mm256_mul_ps(r2, ewald_scale), one), one)
                } else {
                    one
                };
                let f_coul = if PME {
                    _mm256_mul_ps(
                        qq,
                        _mm256_sub_ps(polynomial!(ewald_force, u), _mm256_mul_ps(inv_r, inv_r2)),
                    )
                } else {
                    _mm256_mul_ps(qq, _mm256_sub_ps(krf2, _mm256_mul_ps(inv_r, inv_r2)))
                };
                let fmag = _mm256_add_ps(f_lj, f_coul);
                let gx = _mm256_mul_ps(fmag, dx);
                let gy = _mm256_mul_ps(fmag, dy);
                let gz = _mm256_mul_ps(fmag, dz);
                fx = _mm256_add_ps(fx, gx);
                fy = _mm256_add_ps(fy, gy);
                fz = _mm256_add_ps(fz, gz);
                if OBS {
                    e_lj = _mm256_add_ps(
                        e_lj,
                        _mm256_mul_ps(
                            eps,
                            _mm256_sub_ps(_mm256_mul_ps(s6, s6), _mm256_mul_ps(two, s6)),
                        ),
                    );
                    e_rf = if PME {
                        _mm256_add_ps(
                            e_rf,
                            _mm256_mul_ps(qq, _mm256_sub_ps(inv_r, polynomial!(ewald_energy, u))),
                        )
                    } else {
                        _mm256_add_ps(
                            e_rf,
                            _mm256_mul_ps(
                                qq,
                                _mm256_sub_ps(_mm256_add_ps(inv_r, _mm256_mul_ps(krf, r2)), crf),
                            ),
                        )
                    };
                    w_all = _mm256_sub_ps(w_all, _mm256_mul_ps(fmag, r2));
                    w_lj = _mm256_sub_ps(w_lj, _mm256_mul_ps(f_lj, r2));
                }
                let out = &mut buffer[slot];
                out[0] -= hsum(gx);
                out[1] -= hsum(gy);
                out[2] -= hsum(gz);
            }};
        }
        for lane in 1..LANES {
            let mask = cluster.diagonal[lane];
            if mask != 0 {
                interact!(base + lane, mask);
            }
        }
        for entry in 0..list_atoms.len() {
            interact!(list_atoms[entry] as usize, list_masks[entry]);
        }
        let mut out = [[0f32; LANES]; 3];
        // SAFETY: each destination is a live `[f32; LANES]`.
        unsafe {
            _mm256_storeu_ps(out[0].as_mut_ptr(), fx);
            _mm256_storeu_ps(out[1].as_mut_ptr(), fy);
            _mm256_storeu_ps(out[2].as_mut_ptr(), fz);
        }
        for lane in 0..LANES {
            let slot = &mut buffer[base + lane];
            slot[0] += out[0][lane];
            slot[1] += out[1][lane];
            slot[2] += out[2][lane];
        }
        let mut totals = [0f32; 4];
        if OBS {
            let mut lanes = [[0f32; LANES]; 4];
            // SAFETY: each destination is a live `[f32; LANES]`.
            unsafe {
                _mm256_storeu_ps(lanes[0].as_mut_ptr(), e_lj);
                _mm256_storeu_ps(lanes[1].as_mut_ptr(), e_rf);
                _mm256_storeu_ps(lanes[2].as_mut_ptr(), w_all);
                _mm256_storeu_ps(lanes[3].as_mut_ptr(), w_lj);
            }
            for (total, lane_values) in totals.iter_mut().zip(lanes) {
                for value in lane_values {
                    *total += value;
                }
            }
        }
        totals
    }
}

#[inline(always)]
fn nearest_image(value: f32, center: f32, length: f32) -> f32 {
    value - length * ((value - center) / length).round_ties_even()
}

/// One cluster against its diagonal and listed partners. Returns f32 block
/// totals (LJ, electrostatics, virial, LJ virial).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cluster_kernel(
    params: &KernelParams,
    index: usize,
    cluster: &Cluster,
    sorted: &[[f32; 4]],
    sorted_lj: &[[f32; 2]],
    list_atoms: &[u32],
    list_masks: &[u8],
    buffer: &mut [[f32; 3]],
) -> [f32; 4] {
    cluster_kernel_mode::<false>(
        params, &NO_EWALD, index, cluster, sorted, sorted_lj, list_atoms, list_masks, buffer,
    )
}

/// [`cluster_kernel`] with reaction-field (`PME = false`) or Ewald
/// direct-space (`PME = true`) electrostatics.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cluster_kernel_mode<const PME: bool>(
    params: &KernelParams,
    ewald: &EwaldKernel,
    index: usize,
    cluster: &Cluster,
    sorted: &[[f32; 4]],
    sorted_lj: &[[f32; 2]],
    list_atoms: &[u32],
    list_masks: &[u8],
    buffer: &mut [[f32; 3]],
) -> [f32; 4] {
    let base = index * LANES;
    let c = cluster.center;
    let l = params.box_len;
    let mut xi = [0f32; LANES];
    let mut yi = [0f32; LANES];
    let mut zi = [0f32; LANES];
    let mut qi = [0f32; LANES];
    let mut si = [0f32; LANES];
    let mut ei = [0f32; LANES];
    for lane in 0..LANES {
        let p = sorted[base + lane];
        xi[lane] = nearest_image(p[0], c[0], l[0]);
        yi[lane] = nearest_image(p[1], c[1], l[1]);
        zi[lane] = nearest_image(p[2], c[2], l[2]);
        qi[lane] = (COULOMB as f32) * p[3];
        si[lane] = sorted_lj[base + lane][0];
        ei[lane] = sorted_lj[base + lane][1];
    }
    let lanes = Lanes {
        x: xi,
        y: yi,
        z: zi,
        q: qi,
        sigma: si,
        epsilon: ei,
    };
    let mut acc = Accumulators::default();
    macro_rules! walk {
        ($obs:expr, $per_pair:expr) => {{
            for lane in 1..LANES {
                let mask = cluster.diagonal[lane];
                if mask != 0 {
                    interact::<$obs, $per_pair, PME>(
                        params,
                        ewald,
                        cluster,
                        &lanes,
                        &mut acc,
                        sorted,
                        sorted_lj,
                        base + lane,
                        mask,
                        buffer,
                    );
                }
            }
            for entry in 0..list_atoms.len() {
                interact::<$obs, $per_pair, PME>(
                    params,
                    ewald,
                    cluster,
                    &lanes,
                    &mut acc,
                    sorted,
                    sorted_lj,
                    list_atoms[entry] as usize,
                    list_masks[entry],
                    buffer,
                );
            }
        }};
    }
    match (params.observables, cluster.per_pair) {
        (false, false) => walk!(false, false),
        (false, true) => walk!(false, true),
        (true, false) => walk!(true, false),
        (true, true) => walk!(true, true),
    }
    for lane in 0..LANES {
        let out = &mut buffer[base + lane];
        out[0] += acc.fx[lane];
        out[1] += acc.fy[lane];
        out[2] += acc.fz[lane];
    }
    let mut totals = [0f32; 4];
    if params.observables {
        for lane in 0..LANES {
            totals[0] += acc.e_lj[lane];
            totals[1] += acc.e_rf[lane];
            totals[2] += acc.w_all[lane];
            totals[3] += acc.w_lj[lane];
        }
    }
    totals
}

struct Lanes {
    x: [f32; LANES],
    y: [f32; LANES],
    z: [f32; LANES],
    q: [f32; LANES],
    sigma: [f32; LANES],
    epsilon: [f32; LANES],
}

#[derive(Default)]
struct Accumulators {
    fx: [f32; LANES],
    fy: [f32; LANES],
    fz: [f32; LANES],
    e_lj: [f32; LANES],
    e_rf: [f32; LANES],
    w_all: [f32; LANES],
    w_lj: [f32; LANES],
}

/// Fixed pairwise sum of eight lanes; vectorizes and is order-deterministic.
#[inline(always)]
fn lane_sum(v: [f32; LANES]) -> f32 {
    let a = [v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7]];
    let b = [a[0] + a[2], a[1] + a[3]];
    b[0] + b[1]
}

/// The cluster lanes against one partner slot (broadcast). Branch-free so
/// the lane loop compiles to SIMD; `OBS` adds energies and virials, `PME`
/// swaps the reaction field for the Ewald direct-space term.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn interact<const OBS: bool, const PER_PAIR: bool, const PME: bool>(
    params: &KernelParams,
    ewald: &EwaldKernel,
    cluster: &Cluster,
    lanes: &Lanes,
    acc: &mut Accumulators,
    sorted: &[[f32; 4]],
    sorted_lj: &[[f32; 2]],
    slot: usize,
    mask: u8,
    buffer: &mut [[f32; 3]],
) {
    let c = cluster.center;
    let l = params.box_len;
    let krf2 = 2.0 * params.krf;
    let p = sorted[slot];
    let xj = nearest_image(p[0], c[0], l[0]);
    let yj = nearest_image(p[1], c[1], l[1]);
    let zj = nearest_image(p[2], c[2], l[2]);
    let qj = p[3];
    let [sj, ej] = sorted_lj[slot];
    let mut gx = [0f32; LANES];
    let mut gy = [0f32; LANES];
    let mut gz = [0f32; LANES];
    for lane in 0..LANES {
        let mut dx = lanes.x[lane] - xj;
        let mut dy = lanes.y[lane] - yj;
        let mut dz = lanes.z[lane] - zj;
        if PER_PAIR {
            dx -= l[0] * (dx / l[0]).round_ties_even();
            dy -= l[1] * (dy / l[1]).round_ties_even();
            dz -= l[2] * (dz / l[2]).round_ties_even();
        }
        let r2 = (dx * dx + dy * dy + dz * dz).max(1e-12);
        let listed = ((mask >> lane) & 1) != 0;
        let keep = if listed && r2 < params.cutoff2 {
            1.0f32
        } else {
            0.0
        };
        // Masked lanes (including an atom against itself) get zero inverse
        // distances, so every term below vanishes without inf * 0.
        let inv_r2 = keep / r2;
        let inv_r = inv_r2.sqrt();
        let sigma = lanes.sigma[lane] + sj;
        let eps = lanes.epsilon[lane] * ej;
        let s2 = sigma * sigma * inv_r2;
        let s6 = s2 * s2 * s2;
        let qq = lanes.q[lane] * qj * keep;
        let f_lj = 12.0 * eps * (s6 - s6 * s6) * inv_r2;
        // Masked lanes can lie beyond the cutoff: clamp to the fit.
        let u = if PME {
            (r2 * ewald.scale - 1.0).min(1.0)
        } else {
            1.0
        };
        let f_coul = if PME {
            qq * (ewald_polynomial(&ewald.force, u) - inv_r * inv_r2)
        } else {
            qq * (krf2 - inv_r * inv_r2)
        };
        let fmag = f_lj + f_coul;
        gx[lane] = fmag * dx;
        gy[lane] = fmag * dy;
        gz[lane] = fmag * dz;
        if OBS {
            acc.e_lj[lane] += eps * (s6 * s6 - 2.0 * s6);
            acc.e_rf[lane] += if PME {
                qq * (inv_r - ewald_polynomial(&ewald.energy, u))
            } else {
                qq * (inv_r + params.krf * r2 - params.crf)
            };
            acc.w_all[lane] -= fmag * r2;
            acc.w_lj[lane] -= f_lj * r2;
        }
    }
    for lane in 0..LANES {
        acc.fx[lane] += gx[lane];
        acc.fy[lane] += gy[lane];
        acc.fz[lane] += gz[lane];
    }
    let out = &mut buffer[slot];
    out[0] -= lane_sum(gx);
    out[1] -= lane_sum(gy);
    out[2] -= lane_sum(gz);
}

fn molecule_ids(system: &ParameterizedSystem) -> Vec<u32> {
    let mut molecule = vec![0u32; system.atom_count()];
    for group in crate::pbc::molecules(system) {
        let anchor = group[0] as u32;
        for atom in group {
            molecule[atom] = anchor;
        }
    }
    molecule
}

/// Skilling's transposed Hilbert index of a cell in a 2^bits cube (the same
/// ordering as the GPU engine).
fn hilbert_index(cell: [u32; 3], bits: u32) -> u32 {
    let mut x = cell;
    let top = 1u32 << (bits - 1);
    let mut q = top;
    while q > 1 {
        let p = q - 1;
        for i in 0..3 {
            if x[i] & q != 0 {
                x[0] ^= p;
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        q >>= 1;
    }
    x[1] ^= x[0];
    x[2] ^= x[1];
    let mut t = 0;
    q = top;
    while q > 1 {
        if x[2] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    x[0] ^= t;
    x[1] ^= t;
    x[2] ^= t;
    let mut code = 0u32;
    for k in (0..bits).rev() {
        code =
            (code << 3) | (((x[0] >> k) & 1) << 2) | (((x[1] >> k) & 1) << 1) | ((x[2] >> k) & 1);
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::{PbcForceField, PbcNeighborList, ReactionField};

    fn solvated(padding: f64, ions: bool) -> ParameterizedSystem {
        let pdb = include_str!("../../../tests/fixtures/dipeptide.pdb");
        let options = glysys::BuildOptions {
            add_water: true,
            add_ions: ions,
            padding_angstrom: padding,
            ..Default::default()
        };
        glysys::SystemBuilder::new(options)
            .unwrap()
            .prepare_pdb_str(pdb)
            .unwrap()
    }

    fn jitter(coordinates: &mut [Vec3], seed: u64, amplitude: f64) {
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for p in coordinates {
            p.x += amplitude * next();
            p.y += amplitude * next();
            p.z += amplitude * next();
        }
    }

    /// Pair + exception gradients and energies against the f64 reference
    /// evaluator with the bonded contribution subtracted.
    fn compare(system: &ParameterizedSystem, coordinates: &[Vec3], cutoff: f64) {
        let box_vec = BoxVectors::from_system(system).unwrap();
        let field = PbcForceField::new(system, vec![]).unwrap();
        let backend = ReactionField::new(cutoff, 78.5).unwrap();
        let wrapped: Vec<Vec3> = coordinates.iter().map(|p| box_vec.wrap(*p)).collect();
        let list = PbcNeighborList::build(&wrapped, &box_vec, cutoff, 1.5).unwrap();
        let full = field
            .evaluate(coordinates, &box_vec, &list.pairs, &backend, cutoff)
            .unwrap();
        let bonded = field
            .evaluate(coordinates, &box_vec, &[], &backend, cutoff)
            .unwrap();
        let mut engine = ClusterPairEngine::new(system, cutoff, 1.5, 78.5).unwrap();
        let mut gradients = vec![
            Vec3 {
                x: 0.,
                y: 0.,
                z: 0.
            };
            system.atom_count()
        ];
        let totals = engine
            .evaluate_into(coordinates, &box_vec, &mut gradients, true)
            .unwrap();
        let lj = full.components.van_der_waals;
        let rf = full.components.electrostatics;
        assert!(
            (totals.van_der_waals - lj).abs() <= 1e-4 * lj.abs().max(10.),
            "LJ {} vs {lj}",
            totals.van_der_waals
        );
        assert!(
            (totals.electrostatics - rf).abs() <= 1e-4 * rf.abs().max(10.),
            "RF {} vs {rf}",
            totals.electrostatics
        );
        let virial = full.virial_terms[3];
        assert!(
            (totals.virial - virial).abs() <= 1e-3 * virial.abs().max(10.),
            "pair virial {} vs {virial}",
            totals.virial
        );
        let mut error2 = 0.;
        let mut reference2 = 0.;
        for (atom, ((g, f), b)) in gradients
            .iter()
            .zip(&full.gradients)
            .zip(&bonded.gradients)
            .enumerate()
        {
            for (got, want) in [(g.x, f.x - b.x), (g.y, f.y - b.y), (g.z, f.z - b.z)] {
                let delta = got - want;
                assert!(
                    delta.abs() <= 2e-3 + 2e-4 * want.abs(),
                    "atom {atom}: gradient {got} vs {want}"
                );
                error2 += delta * delta;
                reference2 += want * want;
            }
        }
        assert!(
            (error2 / reference2).sqrt() < 2e-5,
            "force RMS {}",
            (error2 / reference2).sqrt()
        );
    }

    #[test]
    fn matches_the_f64_reference_on_solvated_systems() {
        let system = solvated(6.0, false);
        let mut coordinates = system.coordinates();
        compare(&system, &coordinates, 4.0);
        jitter(&mut coordinates, 7, 0.4);
        compare(&system, &coordinates, 4.0);
        let ions = solvated(9.0, true);
        let mut coordinates = ions.coordinates();
        jitter(&mut coordinates, 11, 0.3);
        compare(&ions, &coordinates, 9.0 - 1.0);
    }

    #[test]
    fn simd_and_portable_kernels_agree_bitwise() {
        let system = solvated(6.0, false);
        let box_vec = BoxVectors::from_system(&system).unwrap();
        let mut coordinates = system.coordinates();
        jitter(&mut coordinates, 5, 0.3);
        let mut engine = ClusterPairEngine::new(&system, 4.0, 1.5, 78.5).unwrap();
        let mut g = vec![
            Vec3 {
                x: 0.,
                y: 0.,
                z: 0.
            };
            system.atom_count()
        ];
        engine
            .evaluate_into(&coordinates, &box_vec, &mut g, true)
            .unwrap();
        let params = KernelParams {
            cutoff2: 16.0,
            krf: engine.krf as f32,
            crf: engine.crf as f32,
            box_len: box_vec.as_array().map(|v| v as f32),
            observables: true,
        };
        for per_pair in [false, true] {
            for (index, cluster) in engine.clusters.iter().enumerate() {
                let mut cluster = *cluster;
                cluster.per_pair = per_pair;
                let mut portable = vec![[0f32; 3]; engine.sorted.len()];
                let mut dispatched = portable.clone();
                let list = &engine.lists[index];
                let a = cluster_kernel(
                    &params,
                    index,
                    &cluster,
                    &engine.sorted,
                    &engine.sorted_lj,
                    &list.atoms,
                    &list.masks,
                    &mut portable,
                );
                let b = run_cluster(
                    &params,
                    index,
                    &cluster,
                    &engine.sorted,
                    &engine.sorted_lj,
                    &list.atoms,
                    &list.masks,
                    &mut dispatched,
                );
                assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
                for (x, y) in portable.iter().zip(&dispatched) {
                    assert_eq!(x.map(f32::to_bits), y.map(f32::to_bits));
                }
            }
        }
    }

    #[test]
    fn reuses_lists_until_the_skin_is_crossed_and_is_reproducible() {
        let system = solvated(6.0, false);
        let box_vec = BoxVectors::from_system(&system).unwrap();
        let mut coordinates = system.coordinates();
        jitter(&mut coordinates, 3, 0.2);
        let mut engine = ClusterPairEngine::new(&system, 4.0, 1.5, 78.5).unwrap();
        let mut first = vec![
            Vec3 {
                x: 0.,
                y: 0.,
                z: 0.
            };
            system.atom_count()
        ];
        engine
            .evaluate_into(&coordinates, &box_vec, &mut first, true)
            .unwrap();
        let mut second = vec![
            Vec3 {
                x: 0.,
                y: 0.,
                z: 0.
            };
            system.atom_count()
        ];
        engine
            .evaluate_into(&coordinates, &box_vec, &mut second, true)
            .unwrap();
        assert_eq!(engine.rebuild_count(), 1);
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.x.to_bits(), b.x.to_bits());
            assert_eq!(a.y.to_bits(), b.y.to_bits());
            assert_eq!(a.z.to_bits(), b.z.to_bits());
        }
        coordinates[0].x += 1.0;
        engine
            .evaluate_into(&coordinates, &box_vec, &mut second, false)
            .unwrap();
        assert_eq!(engine.rebuild_count(), 2);
    }

    #[test]
    fn lists_survive_the_small_box_changes_of_a_barostat() {
        // A barostat scales the box and every coordinate by a part in 10^4
        // or less per step. The lists of the old box still hold every pair
        // inside the cutoff, so the forces have to equal those of an engine
        // that builds its lists in the new box.
        let system = solvated(9.0, true);
        let box_vec = BoxVectors::from_system(&system).unwrap();
        let mut coordinates = system.coordinates();
        jitter(&mut coordinates, 5, 0.2);
        let zeros = || {
            vec![
                Vec3 {
                    x: 0.,
                    y: 0.,
                    z: 0.
                };
                system.atom_count()
            ]
        };
        let mut engine = ClusterPairEngine::new(&system, 9.0, 1.5, 78.5).unwrap();
        engine
            .evaluate_into(&coordinates, &box_vec, &mut zeros(), false)
            .unwrap();
        let mut scale = 1.0;
        for _ in 0..5 {
            scale *= 1.0 - 2e-4;
        }
        let small = BoxVectors::new(box_vec.x * scale, box_vec.y * scale, box_vec.z * scale).unwrap();
        let scaled: Vec<Vec3> = coordinates
            .iter()
            .map(|p| Vec3 {
                x: p.x * scale,
                y: p.y * scale,
                z: p.z * scale,
            })
            .collect();
        let mut kept = zeros();
        let kept_totals = engine.evaluate_into(&scaled, &small, &mut kept, true).unwrap();
        assert_eq!(engine.rebuild_count(), 1, "a 0.1% box change kept the lists");
        let mut fresh_engine = ClusterPairEngine::new(&system, 9.0, 1.5, 78.5).unwrap();
        let mut fresh = zeros();
        let fresh_totals = fresh_engine
            .evaluate_into(&scaled, &small, &mut fresh, true)
            .unwrap();
        let scale_of = |values: &[Vec3]| {
            (values.iter().map(|g| g.x * g.x + g.y * g.y + g.z * g.z).sum::<f64>()
                / values.len() as f64)
                .sqrt()
        };
        let difference: Vec<Vec3> = kept
            .iter()
            .zip(&fresh)
            .map(|(a, b)| Vec3 {
                x: a.x - b.x,
                y: a.y - b.y,
                z: a.z - b.z,
            })
            .collect();
        assert!(scale_of(&difference) < 1e-5 * scale_of(&fresh));
        assert!(
            (kept_totals.electrostatics - fresh_totals.electrostatics).abs()
                < 1e-5 * fresh_totals.electrostatics.abs()
        );
        // A change that uses up the skin forces new lists.
        let far = BoxVectors::new(box_vec.x * 0.98, box_vec.y * 0.98, box_vec.z * 0.98).unwrap();
        let shrunk: Vec<Vec3> = coordinates
            .iter()
            .map(|p| Vec3 {
                x: p.x * 0.98,
                y: p.y * 0.98,
                z: p.z * 0.98,
            })
            .collect();
        engine.evaluate_into(&shrunk, &far, &mut zeros(), false).unwrap();
        assert_eq!(engine.rebuild_count(), 2);
    }

    #[test]
    fn hilbert_order_visits_face_neighbours() {
        let bits = 3;
        let side = 1u32 << bits;
        let mut cells = vec![[0u32; 3]; (side * side * side) as usize];
        for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    cells[hilbert_index([x, y, z], bits) as usize] = [x, y, z];
                }
            }
        }
        for pair in cells.windows(2) {
            let distance: u32 = (0..3)
                .map(|axis| pair[0][axis].abs_diff(pair[1][axis]))
                .sum();
            assert_eq!(distance, 1, "{:?} -> {:?}", pair[0], pair[1]);
        }
    }

    /// [`compare`] for the PME mode: direct-space pair and exception terms
    /// against the f64 evaluator with the `erfc` backend.
    fn compare_pme(system: &ParameterizedSystem, coordinates: &[Vec3], cutoff: f64) {
        let box_vec = BoxVectors::from_system(system).unwrap();
        let field = PbcForceField::new(system, vec![]).unwrap();
        let alpha = crate::pme::ewald_coefficient(cutoff, 1e-5).unwrap();
        let backend = crate::pbc::PmeBackend {
            alpha_per_angstrom: alpha,
            grid: [16; 3],
            interpolation_order: 4,
        };
        let wrapped: Vec<Vec3> = coordinates.iter().map(|p| box_vec.wrap(*p)).collect();
        let list = PbcNeighborList::build(&wrapped, &box_vec, cutoff, 1.5).unwrap();
        let full = field
            .evaluate(coordinates, &box_vec, &list.pairs, &backend, cutoff)
            .unwrap();
        let bonded = field
            .evaluate(coordinates, &box_vec, &[], &backend, cutoff)
            .unwrap();
        let mut engine = ClusterPairEngine::new_pme(system, cutoff, 1.5, alpha).unwrap();
        assert_eq!(engine.ewald_alpha(), Some(alpha));
        let mut gradients = vec![
            Vec3 {
                x: 0.,
                y: 0.,
                z: 0.
            };
            system.atom_count()
        ];
        let totals = engine
            .evaluate_into(coordinates, &box_vec, &mut gradients, true)
            .unwrap();
        let lj = full.components.van_der_waals;
        let direct = full.components.electrostatics;
        assert!(
            (totals.van_der_waals - lj).abs() <= 1e-4 * lj.abs().max(10.),
            "LJ {} vs {lj}",
            totals.van_der_waals
        );
        assert!(
            (totals.electrostatics - direct).abs() <= 1e-4 * direct.abs().max(10.),
            "direct space {} vs {direct}",
            totals.electrostatics
        );
        let virial = full.virial_terms[3];
        assert!(
            (totals.virial - virial).abs() <= 1e-3 * virial.abs().max(10.),
            "pair virial {} vs {virial}",
            totals.virial
        );
        let coulomb_virial = full.virial_pair_split[1];
        assert!(
            (totals.virial_pair_split[1] - coulomb_virial).abs()
                <= 1e-3 * coulomb_virial.abs().max(10.),
            "electrostatic virial {} vs {coulomb_virial}",
            totals.virial_pair_split[1]
        );
        let mut error2 = 0.;
        let mut reference2 = 0.;
        for (atom, ((g, f), b)) in gradients
            .iter()
            .zip(&full.gradients)
            .zip(&bonded.gradients)
            .enumerate()
        {
            for (got, want) in [(g.x, f.x - b.x), (g.y, f.y - b.y), (g.z, f.z - b.z)] {
                let delta = got - want;
                assert!(
                    delta.abs() <= 2e-3 + 2e-4 * want.abs(),
                    "atom {atom}: gradient {got} vs {want}"
                );
                error2 += delta * delta;
                reference2 += want * want;
            }
        }
        assert!(
            (error2 / reference2).sqrt() < 2e-5,
            "force RMS {}",
            (error2 / reference2).sqrt()
        );
        // Force-only steps give the same gradients.
        let mut silent = vec![
            Vec3 {
                x: 0.,
                y: 0.,
                z: 0.
            };
            system.atom_count()
        ];
        engine
            .evaluate_into(coordinates, &box_vec, &mut silent, false)
            .unwrap();
        assert_eq!(silent, gradients);
    }

    #[test]
    fn pme_mode_matches_the_f64_reference_on_solvated_systems() {
        let system = solvated(6.0, false);
        let mut coordinates = system.coordinates();
        compare_pme(&system, &coordinates, 4.0);
        jitter(&mut coordinates, 7, 0.4);
        compare_pme(&system, &coordinates, 4.0);
        let ions = solvated(9.0, true);
        let mut coordinates = ions.coordinates();
        jitter(&mut coordinates, 11, 0.3);
        compare_pme(&ions, &coordinates, 9.0 - 1.0);
        // Close to half the box: clusters fall back to per-pair images.
        compare_pme(&ions, &coordinates, 10.0);
    }

    #[test]
    fn pme_simd_and_portable_kernels_agree_bitwise() {
        let system = solvated(6.0, false);
        let box_vec = BoxVectors::from_system(&system).unwrap();
        let mut coordinates = system.coordinates();
        jitter(&mut coordinates, 5, 0.3);
        let alpha = crate::pme::ewald_coefficient(4.0, 1e-5).unwrap();
        let mut engine = ClusterPairEngine::new_pme(&system, 4.0, 1.5, alpha).unwrap();
        let mut g = vec![
            Vec3 {
                x: 0.,
                y: 0.,
                z: 0.
            };
            system.atom_count()
        ];
        engine
            .evaluate_into(&coordinates, &box_vec, &mut g, true)
            .unwrap();
        let ewald = engine.ewald.unwrap();
        for observables in [true, false] {
            let params = KernelParams {
                cutoff2: 16.0,
                krf: 0.0,
                crf: 0.0,
                box_len: box_vec.as_array().map(|v| v as f32),
                observables,
            };
            for per_pair in [false, true] {
                for (index, cluster) in engine.clusters.iter().enumerate() {
                    let mut cluster = *cluster;
                    cluster.per_pair = per_pair;
                    let mut portable = vec![[0f32; 3]; engine.sorted.len()];
                    let mut dispatched = portable.clone();
                    let list = &engine.lists[index];
                    let a = cluster_kernel_mode::<true>(
                        &params,
                        &ewald,
                        index,
                        &cluster,
                        &engine.sorted,
                        &engine.sorted_lj,
                        &list.atoms,
                        &list.masks,
                        &mut portable,
                    );
                    let b = run_cluster_pme(
                        &params,
                        &ewald,
                        index,
                        &cluster,
                        &engine.sorted,
                        &engine.sorted_lj,
                        &list.atoms,
                        &list.masks,
                        &mut dispatched,
                    );
                    assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
                    for (x, y) in portable.iter().zip(&dispatched) {
                        assert_eq!(x.map(f32::to_bits), y.map(f32::to_bits));
                    }
                }
            }
        }
    }

    #[test]
    fn ewald_pair_approximation_is_accurate() {
        // GROMACS-default splitting for common cutoffs, a tighter tolerance,
        // and OpenMM's default (alpha rc = 2.63).
        for (cutoff, rtol) in [
            (9.0, 1e-5),
            (10.0, 1e-5),
            (12.0, 1e-5),
            (9.0, 1e-6),
            (9.0, 2e-4),
        ] {
            let alpha = crate::pme::ewald_coefficient(cutoff, rtol).unwrap();
            let kernel = EwaldKernel::new(alpha, cutoff).unwrap();
            // Recorded errors, relative to the bare Coulomb force and energy
            // (measured: 4e-7 to 7e-7 and 2e-7 to 3e-7).
            assert!(kernel.error[0] < 1e-6, "force error {:e}", kernel.error[0]);
            assert!(kernel.error[1] < 5e-7, "energy error {:e}", kernel.error[1]);
            // The same against erfc directly, on points between the samples
            // the constructor used, with the distance known only as r^2.
            let two_over_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI;
            let (mut worst_force, mut worst_energy, mut worst_at_cutoff) = (0f64, 0f64, 0f64);
            for sample in 0..20_000 {
                let r = (0.5 + (cutoff - 0.5) * (sample as f64 + 0.37) / 20_000.) as f32;
                let r2 = r * r;
                let inv_r2 = 1.0 / r2;
                let inv_r = inv_r2.sqrt();
                let u = (r2 * kernel.scale - 1.0).min(1.0);
                let force = f64::from(ewald_polynomial(&kernel.force, u) - inv_r * inv_r2);
                let energy = f64::from(inv_r - ewald_polynomial(&kernel.energy, u));
                let d = f64::from(r2).sqrt();
                let screened = crate::pme::erfc(alpha * d);
                let gauss = two_over_sqrt_pi * alpha * (-(alpha * d).powi(2)).exp();
                let exact_force = -(screened / d.powi(3) + gauss / (d * d));
                let exact_energy = screened / d;
                let force_error = (force - exact_force).abs() * d.powi(3);
                worst_force = worst_force.max(force_error);
                worst_energy = worst_energy.max((energy - exact_energy).abs() * d);
                if d > cutoff - 1. {
                    worst_at_cutoff = worst_at_cutoff.max(force_error);
                }
            }
            assert!(
                worst_force < 1e-6,
                "rc {cutoff} rtol {rtol}: {worst_force:e}"
            );
            assert!(
                worst_energy < 5e-7,
                "rc {cutoff} rtol {rtol}: {worst_energy:e}"
            );
            assert!(worst_at_cutoff < 1e-6);
        }
        // The fit degrades with alpha * rc and is refused before it is poor.
        assert!(EwaldKernel::new(4.0 / 9.0, 9.0).is_ok());
        assert!(EwaldKernel::new(5.0 / 9.0, 9.0).is_err());
        assert!(EwaldKernel::new(0., 9.0).is_err());
        assert!(EwaldKernel::new(f64::NAN, 9.0).is_err());
        let system = solvated(6.0, false);
        assert!(ClusterPairEngine::new_pme(&system, 4.0, 1.5, 3.0).is_err());
        let engine = ClusterPairEngine::new(&system, 4.0, 1.5, 78.5).unwrap();
        assert_eq!(engine.ewald_alpha(), None);
        assert_eq!(engine.ewald_pair_error(), None);
    }

    #[test]
    fn cluster_pme_matches_the_reference_pme_hamiltonian() {
        let system = solvated(6.0, false);
        let box_vec = BoxVectors::from_system(&system).unwrap();
        let mut coordinates = system.coordinates();
        jitter(&mut coordinates, 13, 0.3);
        let cutoff = 7.0;
        let parameters =
            crate::pme::PmeParameters::for_box(&box_vec, cutoff, 1e-5, 1.0, 4).unwrap();
        let alpha = parameters.alpha_per_angstrom;
        let field = PbcForceField::new(&system, vec![]).unwrap();
        let mut pme = crate::pme::PmeEngine::new(&system, parameters).unwrap();
        let wrapped: Vec<Vec3> = coordinates.iter().map(|p| box_vec.wrap(*p)).collect();
        let list = PbcNeighborList::build(&wrapped, &box_vec, cutoff, 1.5).unwrap();
        let reference = field
            .evaluate_pme(&mut pme, &coordinates, &box_vec, &list.pairs, cutoff)
            .unwrap();
        let mut engine = ClusterPairEngine::new_pme(&system, cutoff, 1.5, alpha).unwrap();
        let fast = field
            .evaluate_with_cluster_pme(
                &mut engine,
                &mut pme,
                &coordinates,
                &box_vec,
                0.,
                false,
                true,
            )
            .unwrap();
        let (want, got) = (reference.components, fast.components);
        assert_eq!(got.bonds, want.bonds);
        assert!((got.van_der_waals - want.van_der_waals).abs() <= 1e-4 * want.van_der_waals.abs());
        assert!(
            (got.electrostatics - want.electrostatics).abs() <= 1e-5 * want.electrostatics.abs(),
            "electrostatics {} vs {}",
            got.electrostatics,
            want.electrostatics
        );
        for (got, want) in [
            (fast.virial, reference.virial),
            (fast.virial_terms[3], reference.virial_terms[3]),
            (fast.virial_pair_split[1], reference.virial_pair_split[1]),
        ] {
            assert!(
                (got - want).abs() <= 1e-3 * want.abs().max(10.),
                "{got} vs {want}"
            );
        }
        let mut error2 = 0.;
        let mut reference2 = 0.;
        for (g, w) in fast.gradients.iter().zip(&reference.gradients) {
            for (got, want) in [(g.x, w.x), (g.y, w.y), (g.z, w.z)] {
                error2 += (got - want).powi(2);
                reference2 += want * want;
            }
        }
        assert!((error2 / reference2).sqrt() < 2e-5);
        // Force-only steps: same gradients, no pair observables.
        let silent = field
            .evaluate_with_cluster_pme(
                &mut engine,
                &mut pme,
                &coordinates,
                &box_vec,
                0.,
                false,
                false,
            )
            .unwrap();
        assert_eq!(silent.gradients, fast.gradients);
        assert_eq!(silent.components.electrostatics, 0.);
        // The two parts run side by side; the result is reproducible.
        let again = field
            .evaluate_with_cluster_pme(
                &mut engine,
                &mut pme,
                &coordinates,
                &box_vec,
                0.,
                false,
                true,
            )
            .unwrap();
        assert_eq!(again.gradients, fast.gradients);
        assert_eq!(again.components, fast.components);
        assert_eq!(again.virial.to_bits(), fast.virial.to_bits());
        // Engines of the other mode or another alpha are refused.
        let mut reaction_field = ClusterPairEngine::new(&system, cutoff, 1.5, 78.5).unwrap();
        let mut other = ClusterPairEngine::new_pme(&system, cutoff, 1.5, 1.1 * alpha).unwrap();
        for wrong in [&mut reaction_field, &mut other] {
            assert!(
                field
                    .evaluate_with_cluster_pme(
                        wrong,
                        &mut pme,
                        &coordinates,
                        &box_vec,
                        0.,
                        false,
                        true
                    )
                    .is_err()
            );
        }
    }
}
