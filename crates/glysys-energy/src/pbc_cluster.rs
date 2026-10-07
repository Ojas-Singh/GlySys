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
use crate::pbc::{BoxVectors, COULOMB};
use crate::{EnergyError, Result};
use glysys::{ParameterizedSystem, Vec3};
use rayon::prelude::*;
use std::collections::BTreeMap;

const LANES: usize = 8;
const CHUNKS_PER_THREAD: usize = 4;
const SORT_CELL_ANGSTROM: f64 = 3.0;
const MAX_SORT_BITS: u32 = 7;

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
}

impl ClusterPairEngine {
    pub fn new(
        system: &ParameterizedSystem,
        cutoff: f64,
        skin: f64,
        solvent_dielectric: f64,
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
        if !solvent_dielectric.is_finite() || solvent_dielectric < 1. {
            return Err(EnergyError::InvalidConfiguration(
                "reaction-field dielectric must be at least one".into(),
            ));
        }
        let e = solvent_dielectric;
        let krf = (e - 1.) / (2. * e + 1.) / cutoff.powi(3);
        let crf = 3. * e / (2. * e + 1.) / cutoff;
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
        })
    }

    pub fn cutoff(&self) -> f64 {
        self.cutoff
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
        if self.box_at_build != Some(*box_vec) || self.reference.len() != wrapped.len() {
            return true;
        }
        let limit = (0.5 * self.skin).powi(2);
        wrapped
            .par_iter()
            .zip(self.reference.par_iter())
            .any(|(p, r)| {
                let d = box_vec.displacement(*p, *r);
                d.x * d.x + d.y * d.y + d.z * d.z > limit
            })
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
        let wrapped: Vec<Vec3> = coordinates.par_iter().map(|p| box_vec.wrap(*p)).collect();
        if self.needs_rebuild(&wrapped, box_vec) {
            self.rebuild(&wrapped, box_vec)?;
        }
        let order = &self.order;
        let charge = &self.charge;
        self.sorted.resize(order.len(), [0.0; 4]);
        self.sorted
            .par_iter_mut()
            .zip(order.par_iter())
            .for_each(|(slot, &atom)| {
                *slot = if atom == u32::MAX {
                    [0.0; 4]
                } else {
                    let p = wrapped[atom as usize];
                    [p.x as f32, p.y as f32, p.z as f32, charge[atom as usize]]
                };
            });
        // One contiguous chunk of clusters per worker, balanced by listed
        // entries; each chunk owns a sorted-slot force buffer.
        // Several chunks per worker let work stealing balance hybrid cores;
        // the count depends only on the pool size, so results remain
        // reproducible for a given thread count. Buffer memory is capped.
        let threads = rayon::current_num_threads().max(1);
        let memory_cap = ((256usize << 20) / (12 * self.order.len().max(1))).max(threads);
        let workers = (CHUNKS_PER_THREAD * threads).min(memory_cap).max(1);
        let clusters = self.clusters.len();
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
        let chunk_totals: Vec<[f64; 5]> = self
            .buffers
            .par_iter_mut()
            .enumerate()
            .map(|(chunk, buffer)| {
                buffer.fill([0.0; 3]);
                let mut totals = [0.0f64; 5];
                let range = bounds[chunk]..bounds[chunk + 1];
                for (cluster, data) in cluster_list[range.clone()].iter().enumerate() {
                    let index = range.start + cluster;
                    let result = run_cluster(
                        &params,
                        index,
                        data,
                        sorted,
                        sorted_lj,
                        &lists[index].atoms,
                        &lists[index].masks,
                        buffer,
                    );
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
        // Map sorted slots back to atoms, combining chunks in order.
        let mut rank = vec![0u32; self.n];
        for (slot, &atom) in order.iter().enumerate() {
            if atom != u32::MAX {
                rank[atom as usize] = slot as u32;
            }
        }
        gradients
            .par_iter_mut()
            .zip(rank.par_iter())
            .for_each(|(gradient, &slot)| {
                let mut sum = [0.0f64; 3];
                for buffer in buffers {
                    let value = buffer[slot as usize];
                    sum[0] += f64::from(value[0]);
                    sum[1] += f64::from(value[1]);
                    sum[2] += f64::from(value[2]);
                }
                gradient.x += sum[0];
                gradient.y += sum[1];
                gradient.z += sum[2];
            });
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
                        avx2::cluster::<$obs, $per_pair>(
                            params, index, cluster, sorted, sorted_lj, list_atoms, list_masks,
                            buffer,
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

#[cfg(target_arch = "x86_64")]
mod avx2 {
    //! Explicit AVX2 form of [`super::cluster_kernel`]. Every lane performs
    //! the same IEEE operations in the same order as the portable kernel (no
    //! fused multiply-add, ties-to-even rounding), so both produce identical
    //! bits.
    use super::{Cluster, KernelParams, LANES};
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
    pub(super) fn cluster<const OBS: bool, const PER_PAIR: bool>(
        params: &KernelParams,
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
                let f_coul = _mm256_mul_ps(qq, _mm256_sub_ps(krf2, _mm256_mul_ps(inv_r, inv_r2)));
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
                    e_rf = _mm256_add_ps(
                        e_rf,
                        _mm256_mul_ps(
                            qq,
                            _mm256_sub_ps(_mm256_add_ps(inv_r, _mm256_mul_ps(krf, r2)), crf),
                        ),
                    );
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
                    interact::<$obs, $per_pair>(
                        params,
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
                interact::<$obs, $per_pair>(
                    params,
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
/// the lane loop compiles to SIMD; `OBS` adds energies and virials.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn interact<const OBS: bool, const PER_PAIR: bool>(
    params: &KernelParams,
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
        let f_coul = qq * (krf2 - inv_r * inv_r2);
        let fmag = f_lj + f_coul;
        gx[lane] = fmag * dx;
        gy[lane] = fmag * dy;
        gz[lane] = fmag * dz;
        if OBS {
            acc.e_lj[lane] += eps * (s6 * s6 - 2.0 * s6);
            acc.e_rf[lane] += qq * (inv_r + params.krf * r2 - params.crf);
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
}
