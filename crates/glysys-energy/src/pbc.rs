//! Periodic explicit-solvent force evaluation with interchangeable electrostatics.
//!
//! Coordinates are integrated in *unwrapped* space so bonded terms never see a
//! box split. Nonbonded work wraps positions into the box once per evaluation
//! and applies the minimum-image convention. The virial accumulates per term
//! from the displacement vectors actually used for each force (minimum-image
//! vectors for pairs, raw separations for covalent terms), which keeps the
//! pressure exact across box crossings and feeds reporting and the Monte
//! Carlo barostat directly.
//!
//! Reaction field is the first electrostatics backend. PME arrives later
//! behind the same [`ElectrostaticsBackend`] trait, reusing these pair lists,
//! exclusions, and 1-4 scales without touching integration or analysis.
use crate::{EnergyComponents, EnergyError, HarmonicRestraint, Result};
use glysys::{ParameterizedSystem, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

pub const COULOMB: f64 = 332.063_713_299;
pub const DEFAULT_CUTOFF_ANGSTROM: f64 = 9.0;
pub const DEFAULT_SKIN_ANGSTROM: f64 = 1.5;
pub const DEFAULT_RF_DIELECTRIC: f64 = 78.5;

/// Orthorhombic periodic box. Triclinic cells are rejected at the boundary.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoxVectors {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl BoxVectors {
    pub fn new(x: f64, y: f64, z: f64) -> Result<Self> {
        if ![x, y, z].iter().all(|v| v.is_finite() && *v > 0.) {
            return Err(EnergyError::InvalidConfiguration(
                "periodic box vectors must be finite and positive".into(),
            ));
        }
        Ok(Self { x, y, z })
    }

    pub fn from_system(system: &ParameterizedSystem) -> Result<Self> {
        let b = system.box_angstrom();
        Self::new(b[0], b[1], b[2])
    }

    pub fn volume(&self) -> f64 {
        self.x * self.y * self.z
    }

    pub fn as_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Wrap one coordinate into `[0, box)`.
    pub fn wrap(&self, mut p: Vec3) -> Vec3 {
        p.x -= self.x * (p.x / self.x).floor();
        p.y -= self.y * (p.y / self.y).floor();
        p.z -= self.z * (p.z / self.z).floor();
        p
    }

    /// Minimum-image displacement `a - b`.
    pub fn displacement(&self, a: Vec3, b: Vec3) -> Vec3 {
        Vec3 {
            x: a.x - b.x - self.x * ((a.x - b.x) / self.x).round(),
            y: a.y - b.y - self.y * ((a.y - b.y) / self.y).round(),
            z: a.z - b.z - self.z * ((a.z - b.z) / self.z).round(),
        }
    }

    /// Whole-molecule wrap: translate each molecule so its first atom is in
    /// the box, keeping bonded groups intact for output and analysis.
    pub fn wrap_molecules(&self, coords: &[Vec3], molecules: &[Vec<usize>]) -> Vec<Vec3> {
        let mut out = coords.to_vec();
        for mol in molecules {
            let Some(&anchor) = mol.first() else { continue };
            let Some(a) = coords.get(anchor) else {
                continue;
            };
            let shift = Vec3 {
                x: -self.x * (a.x / self.x).floor(),
                y: -self.y * (a.y / self.y).floor(),
                z: -self.z * (a.z / self.z).floor(),
            };
            for &i in mol {
                if let Some(p) = out.get_mut(i) {
                    p.x += shift.x;
                    p.y += shift.y;
                    p.z += shift.z;
                }
            }
        }
        out
    }
}

/// One electrostatics evaluation for a pair at distance `r` (angstrom).
/// Input charge product is pre-scaled by [`COULOMB`]; output is
/// `(energy_kcal_mol, dE_dr_kcal_mol_per_angstrom)`.
pub trait ElectrostaticsBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn pair(&self, r: f64, qq: f64) -> (f64, f64);
}

/// Reaction-field electrostatics matching the OpenMM CutoffPeriodic
/// convention (no switching): `E = qq * (1/r + k_rf r^2 - c_rf)` with
/// `k_rf = (1/rc^3)(eps-1)/(2eps+1)` and `c_rf = (1/rc)(3eps)/(2eps+1)`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ReactionField {
    pub cutoff_angstrom: f64,
    pub solvent_dielectric: f64,
}

impl ReactionField {
    pub fn new(cutoff_angstrom: f64, solvent_dielectric: f64) -> Result<Self> {
        if !cutoff_angstrom.is_finite()
            || cutoff_angstrom <= 0.
            || !solvent_dielectric.is_finite()
            || solvent_dielectric < 1.
        {
            return Err(EnergyError::InvalidConfiguration(
                "reaction-field cutoff and dielectric must be positive".into(),
            ));
        }
        Ok(Self {
            cutoff_angstrom,
            solvent_dielectric,
        })
    }

    fn krf(&self) -> f64 {
        let e = self.solvent_dielectric;
        (e - 1.) / (2. * e + 1.) / self.cutoff_angstrom.powi(3)
    }

    fn crf(&self) -> f64 {
        let e = self.solvent_dielectric;
        3. * e / (2. * e + 1.) / self.cutoff_angstrom
    }
}

impl ElectrostaticsBackend for ReactionField {
    fn name(&self) -> &'static str {
        "reaction-field"
    }

    fn pair(&self, r: f64, qq: f64) -> (f64, f64) {
        let (krf, crf) = (self.krf(), self.crf());
        let energy = qq * (1. / r + krf * r * r - crf);
        let derivative = qq * (-1. / (r * r) + 2. * krf * r);
        (energy, derivative)
    }
}

/// Reserved PME insertion point. Constructing an evaluation with this backend
/// fails with a clear error until the mesh implementation lands; pair lists,
/// exclusions, 1-4 scales, and virial plumbing are shared with reaction field.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PmeBackend {
    pub alpha_per_angstrom: f64,
    pub grid: [usize; 3],
    pub interpolation_order: usize,
}

impl ElectrostaticsBackend for PmeBackend {
    fn name(&self) -> &'static str {
        "pme"
    }

    fn pair(&self, _r: f64, _qq: f64) -> (f64, f64) {
        unimplemented!("PME direct/reciprocal kernels are a later milestone")
    }
}

/// Serializable nonbonded configuration. `Pme` is accepted by parsers and
/// rejected at evaluation time so saved protocols stay forward compatible.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NonbondedElectrostatics {
    ReactionField {
        #[serde(default = "default_cutoff")]
        cutoff_angstrom: f64,
        #[serde(default = "default_rf_dielectric")]
        solvent_dielectric: f64,
    },
    Pme {
        alpha_per_angstrom: f64,
        grid: [usize; 3],
        interpolation_order: usize,
    },
}

fn default_cutoff() -> f64 {
    DEFAULT_CUTOFF_ANGSTROM
}

fn default_rf_dielectric() -> f64 {
    DEFAULT_RF_DIELECTRIC
}

impl Default for NonbondedElectrostatics {
    fn default() -> Self {
        Self::ReactionField {
            cutoff_angstrom: DEFAULT_CUTOFF_ANGSTROM,
            solvent_dielectric: DEFAULT_RF_DIELECTRIC,
        }
    }
}

/// Verlet-style PBC pair list with displacement-based rebuild checks.
#[derive(Clone, Debug)]
pub struct PbcNeighborList {
    pub pairs: Vec<(usize, usize)>,
    pub cutoff: f64,
    pub skin: f64,
    box_vectors: BoxVectors,
    reference: Vec<Vec3>,
}

impl PbcNeighborList {
    pub fn build(wrapped: &[Vec3], box_vec: &BoxVectors, cutoff: f64, skin: f64) -> Result<Self> {
        if !cutoff.is_finite() || cutoff <= 0. || !skin.is_finite() || skin < 0. {
            return Err(EnergyError::InvalidConfiguration(
                "neighbor cutoff must be positive and skin non-negative".into(),
            ));
        }
        // The physical cutoff must be below half the shortest edge.  A skin
        // larger than that margin is still valid: the list may conservatively
        // contain every pair in the box, but the evaluator applies the
        // physical cutoff and minimum image exactly.  Rejecting on
        // `cutoff + skin` would make otherwise valid NPT volume moves fail
        // merely because an execution buffer was chosen too generously.
        if cutoff >= 0.5 * box_vec.x.min(box_vec.y).min(box_vec.z) {
            return Err(EnergyError::InvalidConfiguration(
                "cutoff must be below half the shortest box edge".into(),
            ));
        }
        if wrapped.iter().any(|p| {
            !p.x.is_finite()
                || !p.y.is_finite()
                || !p.z.is_finite()
                || p.x.abs() > 1e12
                || p.y.abs() > 1e12
                || p.z.abs() > 1e12
        }) {
            return Err(EnergyError::NonFiniteCoordinate);
        }
        let limit = cutoff + skin;
        // Exact-fit cells: an integer count spanning each box edge so every
        // cell is full and the ±1 stencil matches spatial adjacency through
        // the wrap. Ceil-based counts leave partial edge cells whose atoms
        // sit within range across the boundary yet outside stencil reach,
        // silently dropping pairs (and their forces) from the list.
        let nx = ((box_vec.x / limit).floor() as i32).max(1);
        let ny = ((box_vec.y / limit).floor() as i32).max(1);
        let nz = ((box_vec.z / limit).floor() as i32).max(1);
        let (cx, cy, cz) = (
            box_vec.x / nx as f64,
            box_vec.y / ny as f64,
            box_vec.z / nz as f64,
        );
        let key = |p: Vec3| {
            (
                ((p.x / cx).floor() as i32).clamp(0, nx - 1),
                ((p.y / cy).floor() as i32).clamp(0, ny - 1),
                ((p.z / cz).floor() as i32).clamp(0, nz - 1),
            )
        };
        let mut cells: BTreeMap<(i32, i32, i32), Vec<usize>> = BTreeMap::new();
        for (i, p) in wrapped.iter().copied().enumerate() {
            cells.entry(key(p)).or_default().push(i);
        }
        let mut pairs = Vec::new();
        let at = |ix: i32, iy: i32, iz: i32| {
            cells.get(&(
                ix.rem_euclid(nx.max(1)),
                iy.rem_euclid(ny.max(1)),
                iz.rem_euclid(nz.max(1)),
            ))
        };
        for (cell_key, members) in &cells {
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(other) = at(
                            cell_key.0.saturating_add(dx),
                            cell_key.1.saturating_add(dy),
                            cell_key.2.saturating_add(dz),
                        ) else {
                            continue;
                        };
                        for &i in members {
                            for &j in other {
                                if j <= i {
                                    continue;
                                }
                                let d = box_vec.displacement(wrapped[i], wrapped[j]);
                                if d.x * d.x + d.y * d.y + d.z * d.z <= limit * limit {
                                    pairs.push((i, j));
                                }
                            }
                        }
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        Ok(Self {
            pairs,
            cutoff,
            skin,
            box_vectors: *box_vec,
            reference: wrapped.to_vec(),
        })
    }

    /// A box change invalidates both the cell geometry and the displacement
    /// reference, even when uniformly scaled coordinates happen to have the
    /// same wrapped values.  Callers use this before checking atom motion so
    /// NPT transitions cannot continue with a list built for the old cell.
    pub fn box_changed(&self, box_vec: &BoxVectors) -> bool {
        self.box_vectors != *box_vec
    }

    pub fn needs_rebuild(&self, wrapped: &[Vec3]) -> bool {
        wrapped.len() != self.reference.len()
            || wrapped.iter().zip(&self.reference).any(|(c, r)| {
                let displacement = self.box_vectors.displacement(*c, *r);
                let distance2 = displacement.x * displacement.x
                    + displacement.y * displacement.y
                    + displacement.z * displacement.z;
                distance2 > (self.skin * 0.5).powi(2)
            })
    }
}

/// Water oxygen plus its two hydrogens, in that order.
pub type WaterIndices = [usize; 3];

/// Rigid three-site waters by residue name and O/H/H element pattern.
pub fn classify_waters(system: &ParameterizedSystem) -> Vec<WaterIndices> {
    let mut by_residue: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, atom) in system.atoms().iter().enumerate() {
        by_residue.entry(atom.residue_index()).or_default().push(i);
    }
    let mut waters = Vec::new();
    for (residue, members) in by_residue {
        let name = system.residues()[residue].name();
        if !matches!(name, "HOH" | "WAT" | "TIP3" | "TIP") || members.len() != 3 {
            continue;
        }
        let mut oxygen = None;
        let mut hydrogens = Vec::new();
        for &i in &members {
            match system.atoms()[i].element() {
                8 => oxygen = Some(i),
                1 => hydrogens.push(i),
                _ => {}
            }
        }
        if let (Some(o), [h1, h2]) = (oxygen, hydrogens.as_slice()) {
            waters.push([o, *h1, *h2]);
        }
    }
    waters.sort_unstable();
    waters
}

/// Force-field equilibrium geometry for one rigid water: the two O-H bond
/// lengths and the H-H distance derived from the H-O-H angle term. Targets
/// come from the topology so strained snapshots never pin strain into the
/// constraint manifold (which would pump energy every step).
pub fn water_equilibrium(
    system: &ParameterizedSystem,
    water: WaterIndices,
) -> Result<(f64, f64, f64)> {
    let [o, h1, h2] = water;
    let bond_length = |a: usize, b: usize| {
        system
            .bonds()
            .iter()
            .find(|bond| {
                let [x, y] = bond.atoms();
                (x == a && y == b) || (x == b && y == a)
            })
            .map(|bond| bond.length())
    };
    let (Some(oh1), Some(oh2)) = (bond_length(o, h1), bond_length(o, h2)) else {
        return Err(EnergyError::InvalidConfiguration(
            "rigid water needs O-H bond terms".into(),
        ));
    };
    let theta = system
        .angles()
        .iter()
        .find(|angle| {
            let [a, c, b] = angle.atoms();
            (a == h1 && c == o && b == h2) || (a == h2 && c == o && b == h1)
        })
        .map(|angle| angle.radians());
    let theta = theta.ok_or_else(|| {
        EnergyError::InvalidConfiguration("rigid water needs an H-O-H angle term".into())
    })?;
    if !(oh1 > 0.3 && oh1 < 3. && oh2 > 0.3 && oh2 < 3. && theta > 0.5 && theta < 2.5) {
        return Err(EnergyError::InvalidConfiguration(
            "unphysical water equilibrium geometry".into(),
        ));
    }
    let hh = (oh1 * oh1 + oh2 * oh2 - 2. * oh1 * oh2 * theta.cos()).sqrt();
    Ok((oh1, oh2, hh))
}

/// Connected components over covalent bonds: the whole-molecule grouping used
/// for wrapped output and future per-molecule analysis.
pub fn molecules(system: &ParameterizedSystem) -> Vec<Vec<usize>> {
    let n = system.atom_count();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for bond in system.bonds() {
        let [a, b] = bond.atoms();
        if a < n && b < n {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra] = rb;
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        groups.entry(find(&mut parent, i)).or_default().push(i);
    }
    groups.into_values().collect()
}

pub struct PbcEnergy {
    pub components: EnergyComponents,
    pub gradients: Vec<Vec3>,
    /// Total virial in kcal/mol; feeds pressure reporting and the barostat.
    pub virial: f64,
    /// Per-term virial split (bonds, angles, torsions, pairs), same units.
    /// Restraints are external and excluded from all four.
    pub virial_terms: [f64; 4],
    /// Pair virial split into Lennard-Jones and electrostatic parts.
    pub virial_pair_split: [f64; 2],
}

/// Thread-local result for a deterministic nonbonded reduction. Each worker
/// walks its pair chunk in the canonical sorted order and owns a private
/// gradient buffer; the caller combines chunks in chunk order, so parallel
/// evaluation never needs floating-point atomics or a full pair matrix.
struct PairChunk {
    van_der_waals: f64,
    electrostatics: f64,
    gradients: Vec<Vec3>,
    virial: f64,
    virial_pair_split: [f64; 2],
}

impl PairChunk {
    fn new(atom_count: usize) -> Self {
        Self {
            van_der_waals: 0.,
            electrostatics: 0.,
            gradients: vec![
                Vec3 {
                    x: 0.,
                    y: 0.,
                    z: 0.
                };
                atom_count
            ],
            virial: 0.,
            virial_pair_split: [0., 0.],
        }
    }
}

/// Periodic force field borrowing topology; positions are always unwrapped.
pub struct PbcForceField<'a> {
    system: std::borrow::Cow<'a, ParameterizedSystem>,
    sigma: Vec<f64>,
    epsilon: Vec<f64>,
    charge: Vec<f64>,
    exclusions: Vec<std::collections::BTreeSet<usize>>,
    one_four: HashMap<(usize, usize), (f64, f64)>,
    one_four_atoms: Vec<bool>,
    restraints: Vec<HarmonicRestraint>,
}

fn ordered(first: usize, second: usize) -> (usize, usize) {
    if first < second {
        (first, second)
    } else {
        (second, first)
    }
}

impl<'a> PbcForceField<'a> {
    pub fn new(
        system: &'a ParameterizedSystem,
        restraints: Vec<HarmonicRestraint>,
    ) -> Result<Self> {
        if system.atom_count() == 0 {
            return Err(EnergyError::InvalidConfiguration(
                "periodic force field needs at least one atom".into(),
            ));
        }
        let mut one_four = HashMap::new();
        for torsion in system.dihedrals().iter().filter(|t| !t.is_improper()) {
            let atoms = torsion.atoms();
            one_four.entry(ordered(atoms[0], atoms[3])).or_insert((
                torsion.electrostatic_14_scale(),
                torsion.lennard_jones_14_scale(),
            ));
        }
        let mut one_four_atoms = vec![false; system.atom_count()];
        for &(a, b) in one_four.keys() {
            one_four_atoms[a] = true;
            one_four_atoms[b] = true;
        }
        Ok(Self {
            system: std::borrow::Cow::Borrowed(system),
            sigma: system
                .atoms()
                .iter()
                .map(|a| a.lennard_jones_radius())
                .collect(),
            epsilon: system
                .atoms()
                .iter()
                .map(|a| a.lennard_jones_epsilon())
                .collect(),
            charge: system.atoms().iter().map(|a| a.charge()).collect(),
            exclusions: system.exclusions().to_vec(),
            one_four,
            one_four_atoms,
            restraints,
        })
    }

    pub fn into_owned(self) -> PbcForceField<'static> {
        PbcForceField {
            system: std::borrow::Cow::Owned(self.system.into_owned()),
            sigma: self.sigma,
            epsilon: self.epsilon,
            charge: self.charge,
            exclusions: self.exclusions,
            one_four: self.one_four,
            one_four_atoms: self.one_four_atoms,
            restraints: self.restraints,
        }
    }

    /// OpenMM-compatible homogeneous long-range Lennard-Jones coefficient in
    /// kcal mol^-1 Å^3. The returned correction is `coefficient / volume`.
    /// Parameter classes are counted exactly as OpenMM's dispersion
    /// correction: all particle pairs use Lorentz-Berthelot mixing and no
    /// atom-pair matrix is materialized.
    pub fn dispersion_coefficient(&self, cutoff: f64) -> Result<f64> {
        if !cutoff.is_finite() || cutoff <= 0. {
            return Err(EnergyError::InvalidConfiguration(
                "dispersion correction needs a positive cutoff".into(),
            ));
        }
        let mut classes: BTreeMap<(u64, u64), usize> = BTreeMap::new();
        for (&sigma, &epsilon) in self.sigma.iter().zip(&self.epsilon) {
            if !sigma.is_finite() || !epsilon.is_finite() || sigma < 0. || epsilon < 0. {
                return Err(EnergyError::InvalidConfiguration(
                    "invalid Lennard-Jones parameters for dispersion correction".into(),
                ));
            }
            *classes
                .entry((sigma.to_bits(), epsilon.to_bits()))
                .or_default() += 1;
        }
        let mut sum6 = 0.0;
        let mut sum12 = 0.0;
        let entries: Vec<((f64, f64), usize)> = classes
            .into_iter()
            .map(|((sigma, epsilon), count)| {
                ((f64::from_bits(sigma), f64::from_bits(epsilon)), count)
            })
            .collect();
        for i in 0..entries.len() {
            let (a, na) = entries[i];
            for j in i..entries.len() {
                let (b, nb) = entries[j];
                // GlySys stores Amber Rmin/2 radii and evaluates
                // eps*((Rmin/r)^12 - 2*(Rmin/r)^6). OpenMM's correction is
                // written in sigma, where sigma = Rmin / 2^(1/6).
                let sigma = (a.0 + b.0) / 2f64.powf(1. / 6.);
                let epsilon = (a.1 * b.1).sqrt();
                let pair_count = if i == j {
                    (na * (na + 1) / 2) as f64
                } else {
                    (na * nb) as f64
                };
                let sigma6 = sigma.powi(6);
                sum12 += pair_count * epsilon * sigma6 * sigma6;
                sum6 += pair_count * epsilon * sigma6;
            }
        }
        let n = self.sigma.len() as f64;
        if n == 0. {
            return Ok(0.);
        }
        // OpenMM normalizes the class sums by the number of unordered
        // particle pairs, then multiplies by 8πN².
        let pair_norm = n * (n + 1.0) * 0.5;
        Ok(8.0
            * std::f64::consts::PI
            * n
            * n
            * (sum12 / pair_norm / (9.0 * cutoff.powi(9))
                - sum6 / pair_norm / (3.0 * cutoff.powi(3))))
    }

    /// Evaluate the cutoff Hamiltonian and optionally add the homogeneous
    /// dispersion correction. Existing callers retain the historical
    /// truncated-LJ behavior through `evaluate`.
    pub fn evaluate_with_dispersion(
        &self,
        unwrapped: &[Vec3],
        box_vec: &BoxVectors,
        pairs: &[(usize, usize)],
        backend: &dyn ElectrostaticsBackend,
        cutoff: f64,
        include_dispersion: bool,
    ) -> Result<PbcEnergy> {
        let coefficient = if include_dispersion {
            self.dispersion_coefficient(cutoff)?
        } else {
            0.0
        };
        self.evaluate_with_dispersion_coefficient(
            unwrapped,
            box_vec,
            pairs,
            backend,
            cutoff,
            coefficient,
            include_dispersion,
        )
    }

    /// Evaluate with a caller-supplied homogeneous correction coefficient.
    /// Dynamics sessions precompute this once per prepared topology and use
    /// this entry point in their hot loop; the convenience method above stays
    /// available for one-off scoring calls.
    pub fn evaluate_with_dispersion_coefficient(
        &self,
        unwrapped: &[Vec3],
        box_vec: &BoxVectors,
        pairs: &[(usize, usize)],
        backend: &dyn ElectrostaticsBackend,
        cutoff: f64,
        dispersion_coefficient: f64,
        include_dispersion: bool,
    ) -> Result<PbcEnergy> {
        let mut result = self.evaluate(unwrapped, box_vec, pairs, backend, cutoff)?;
        if include_dispersion {
            if !dispersion_coefficient.is_finite() {
                return Err(EnergyError::InvalidConfiguration(
                    "non-finite dispersion correction coefficient".into(),
                ));
            }
            result.components.dispersion_correction = dispersion_coefficient / box_vec.volume();
        }
        Ok(result)
    }

    pub fn evaluate(
        &self,
        unwrapped: &[Vec3],
        box_vec: &BoxVectors,
        pairs: &[(usize, usize)],
        backend: &dyn ElectrostaticsBackend,
        cutoff: f64,
    ) -> Result<PbcEnergy> {
        let n = self.system.atom_count();
        if unwrapped.len() != n {
            return Err(EnergyError::CoordinateCount {
                expected: n,
                received: unwrapped.len(),
            });
        }
        if unwrapped
            .iter()
            .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
        {
            return Err(EnergyError::NonFiniteCoordinate);
        }
        let mut components = EnergyComponents::default();
        let mut gradients = vec![
            Vec3 {
                x: 0.,
                y: 0.,
                z: 0.
            };
            n
        ];
        // Virial from the displacement vectors actually used for each force:
        // minimum-image vectors for pairs (unwrapped separations can span
        // images and would corrupt the pressure), raw separations for bonds.
        let mut virial_terms = [0.; 4];
        let mut virial_pair_split = [0.; 2];
        // Bonded terms see unwrapped coordinates: never a box split.
        for bond in self.system.bonds() {
            let [a, b] = bond.atoms();
            let dx = unwrapped[a].x - unwrapped[b].x;
            let dy = unwrapped[a].y - unwrapped[b].y;
            let dz = unwrapped[a].z - unwrapped[b].z;
            let r = (dx * dx + dy * dy + dz * dz).sqrt().max(1.0e-8);
            let delta = r - bond.length();
            components.bonds += bond.force() * delta * delta;
            let f = 2. * bond.force() * delta / r;
            gradients[a].x += f * dx;
            gradients[a].y += f * dy;
            gradients[a].z += f * dz;
            gradients[b].x -= f * dx;
            gradients[b].y -= f * dy;
            gradients[b].z -= f * dz;
            // W = sum(r.F) = -sum(r.grad): the minus sign is essential.
            virial_terms[0] -= f * (dx * dx + dy * dy + dz * dz);
        }
        for angle in self.system.angles() {
            let [a, c, b] = angle.atoms();
            let (theta, g) = angle_with_gradient(unwrapped[a], unwrapped[c], unwrapped[b]);
            let delta = theta - angle.radians();
            components.angles += angle.force() * delta * delta;
            let f = 2. * angle.force() * delta;
            for (i, v) in [(a, g.0), (c, g.1), (b, g.2)] {
                gradients[i].x += f * v.x;
                gradients[i].y += f * v.y;
                gradients[i].z += f * v.z;
                // Translation-invariant term: unwrapped coords are exact.
                // Minus sign: virial uses forces, gradients store dE/dx.
                virial_terms[1] -=
                    f * (unwrapped[i].x * v.x + unwrapped[i].y * v.y + unwrapped[i].z * v.z);
            }
        }
        for torsion in self.system.dihedrals() {
            let atoms = torsion.atoms();
            let p = [
                unwrapped[atoms[0]],
                unwrapped[atoms[1]],
                unwrapped[atoms[2]],
                unwrapped[atoms[3]],
            ];
            let (phi, grad) = dihedral_with_gradient(p[0], p[1], p[2], p[3]);
            let arg = torsion.periodicity() as f64 * phi - torsion.phase();
            let energy = torsion.force() * (1. + arg.cos());
            if torsion.is_improper() {
                components.improper_torsions += energy;
            } else {
                components.proper_torsions += energy;
            }
            let f = -(torsion.periodicity() as f64) * torsion.force() * arg.sin();
            for (k, atom) in atoms.iter().enumerate() {
                gradients[*atom].x += f * grad[k].x;
                gradients[*atom].y += f * grad[k].y;
                gradients[*atom].z += f * grad[k].z;
                virial_terms[2] -= f
                    * (unwrapped[*atom].x * grad[k].x
                        + unwrapped[*atom].y * grad[k].y
                        + unwrapped[*atom].z * grad[k].z);
            }
        }
        // Nonbonded pairs use the minimum image of wrapped positions. Every
        // worker receives a sorted contiguous chunk and reduces privately;
        // chunks are folded below in the same order for reproducibility.
        let wrapped: Vec<Vec3> = unwrapped.iter().map(|p| box_vec.wrap(*p)).collect();
        let thread_count = rayon::current_num_threads();
        let chunk_size = pairs
            .len()
            .saturating_add(thread_count.saturating_sub(1))
            .checked_div(thread_count.max(1))
            .unwrap_or(1)
            .max(1);
        let chunks: Vec<PairChunk> = if thread_count > 1 && pairs.len() >= 512 {
            use rayon::prelude::*;
            pairs
                .par_chunks(chunk_size)
                .map(|chunk| self.evaluate_pair_chunk(&wrapped, box_vec, chunk, backend, cutoff))
                .collect()
        } else {
            vec![self.evaluate_pair_chunk(&wrapped, box_vec, pairs, backend, cutoff)]
        };
        for chunk in chunks {
            components.van_der_waals += chunk.van_der_waals;
            components.electrostatics += chunk.electrostatics;
            virial_terms[3] += chunk.virial;
            virial_pair_split[0] += chunk.virial_pair_split[0];
            virial_pair_split[1] += chunk.virial_pair_split[1];
            for (total, local) in gradients.iter_mut().zip(chunk.gradients) {
                total.x += local.x;
                total.y += local.y;
                total.z += local.z;
            }
        }
        for restraint in &self.restraints {
            if let Some(p) = unwrapped.get(restraint.atom) {
                let dx = p.x - restraint.reference.x;
                let dy = p.y - restraint.reference.y;
                let dz = p.z - restraint.reference.z;
                components.restraints += restraint.force * (dx * dx + dy * dy + dz * dz);
                gradients[restraint.atom].x += 2. * restraint.force * dx;
                gradients[restraint.atom].y += 2. * restraint.force * dy;
                gradients[restraint.atom].z += 2. * restraint.force * dz;
            }
        }
        // Restraint forces are external and excluded from the virial by
        // design; angle/torsion/pair virials accumulate inline above.
        let virial = virial_terms.iter().sum();
        Ok(PbcEnergy {
            components,
            gradients,
            virial,
            virial_terms,
            virial_pair_split,
        })
    }

    fn evaluate_pair_chunk(
        &self,
        wrapped: &[Vec3],
        box_vec: &BoxVectors,
        pairs: &[(usize, usize)],
        backend: &dyn ElectrostaticsBackend,
        cutoff: f64,
    ) -> PairChunk {
        let mut result = PairChunk::new(self.system.atom_count());
        let cutoff2 = cutoff * cutoff;
        for &(a, b) in pairs {
            if a >= self.system.atom_count() || b >= self.system.atom_count() {
                continue;
            }
            let pair = ordered(a, b);
            // Most explicit-solvent pairs involve water atoms, which cannot
            // participate in a 1–4 torsion. Avoid their hash lookup while
            // retaining the identical override and summation semantics.
            let scale = if self.one_four_atoms[a] && self.one_four_atoms[b] {
                self.one_four.get(&pair).copied()
            } else {
                None
            };
            if self.exclusions[a].contains(&b) && scale.is_none() {
                continue;
            }
            let (scee, scnb) = scale.unwrap_or((1., 1.));
            let d = box_vec.displacement(wrapped[a], wrapped[b]);
            let r2 = d.x * d.x + d.y * d.y + d.z * d.z;
            // Verlet lists span cutoff plus skin; only pairs inside the
            // cutoff contribute, so rebuilds never move the Hamiltonian.
            if r2 > cutoff2 {
                continue;
            }
            let r = r2.sqrt().max(1.0e-8);
            let sig = self.sigma[a] + self.sigma[b];
            let eps = (self.epsilon[a] * self.epsilon[b]).sqrt() / scnb;
            let ratio6 = (sig / r).powi(6);
            result.van_der_waals += eps * (ratio6 * ratio6 - 2. * ratio6);
            let qq = COULOMB * self.charge[a] * self.charge[b] / scee;
            // 1-4 exceptions bypass the electrostatics backend: OpenMM and
            // Amber evaluate them with plain Coulomb (scaled by 1/scee),
            // while reaction-field screening (and later PME) applies to
            // regular pairs only.
            let (ecoul, decoul_dr) = if scale.is_some() {
                (qq / r, -qq / (r * r))
            } else {
                backend.pair(r, qq)
            };
            result.electrostatics += ecoul;
            let flj = 12. * eps * (ratio6 - ratio6 * ratio6) / r;
            let fmag = (flj + decoul_dr) / r;
            result.gradients[a].x += fmag * d.x;
            result.gradients[a].y += fmag * d.y;
            result.gradients[a].z += fmag * d.z;
            result.gradients[b].x -= fmag * d.x;
            result.gradients[b].y -= fmag * d.y;
            result.gradients[b].z -= fmag * d.z;
            // Pair virial from the minimum-image vector: unwrapped
            // separations can span images and would corrupt the pressure.
            let w_pair = fmag * r2;
            result.virial -= w_pair;
            let w_lj = (flj / r) * r2;
            result.virial_pair_split[0] -= w_lj;
            result.virial_pair_split[1] -= w_pair - w_lj;
        }
        result
    }
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
}

fn norm(v: Vec3) -> f64 {
    (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
}

/// Angle plus analytic Cartesian gradients, matching the implicit engine.
fn angle_with_gradient(a: Vec3, c: Vec3, b: Vec3) -> (f64, (Vec3, Vec3, Vec3)) {
    let u = sub(a, c);
    let v = sub(b, c);
    let ru = norm(u).max(1.0e-8);
    let rv = norm(v).max(1.0e-8);
    let cos_t = ((u.x * v.x + u.y * v.y + u.z * v.z) / (ru * rv)).clamp(-1., 1.);
    let theta = cos_t.acos();
    let sin_t = theta.sin().max(1.0e-8);
    let f = -1. / sin_t;
    let gu = Vec3 {
        x: f * (v.x / (ru * rv) - cos_t * u.x / (ru * ru)),
        y: f * (v.y / (ru * rv) - cos_t * u.y / (ru * ru)),
        z: f * (v.z / (ru * rv) - cos_t * u.z / (ru * ru)),
    };
    let gv = Vec3 {
        x: f * (u.x / (ru * rv) - cos_t * v.x / (rv * rv)),
        y: f * (u.y / (ru * rv) - cos_t * v.y / (rv * rv)),
        z: f * (u.z / (ru * rv) - cos_t * v.z / (rv * rv)),
    };
    let gc = Vec3 {
        x: -gu.x - gv.x,
        y: -gu.y - gv.y,
        z: -gu.z - gv.z,
    };
    (theta, (gu, gc, gv))
}

/// Dihedral value plus exact Cartesian gradients via forward dual numbers.
/// The scalar matches the implicit engine's dihedral; derivatives are exact
/// by construction, so only self-consistency needs testing.
pub(super) fn dihedral_with_gradient(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> (f64, [Vec3; 4]) {
    #[derive(Clone, Copy)]
    struct D {
        v: f64,
        g: [f64; 12],
    }
    impl D {
        fn var(v: f64, k: usize) -> Self {
            let mut g = [0.; 12];
            g[k] = 1.;
            Self { v, g }
        }
        fn cst(v: f64) -> Self {
            Self { v, g: [0.; 12] }
        }
        fn add(self, o: Self) -> Self {
            let mut g = [0.; 12];
            for i in 0..12 {
                g[i] = self.g[i] + o.g[i];
            }
            Self { v: self.v + o.v, g }
        }
        fn sub(self, o: Self) -> Self {
            let mut g = [0.; 12];
            for i in 0..12 {
                g[i] = self.g[i] - o.g[i];
            }
            Self { v: self.v - o.v, g }
        }
        fn mul(self, o: Self) -> Self {
            let mut g = [0.; 12];
            for i in 0..12 {
                g[i] = self.g[i] * o.v + self.v * o.g[i];
            }
            Self { v: self.v * o.v, g }
        }
        fn div(self, o: Self) -> Self {
            let mut g = [0.; 12];
            for i in 0..12 {
                g[i] = (self.g[i] * o.v - self.v * o.g[i]) / (o.v * o.v);
            }
            Self { v: self.v / o.v, g }
        }
        fn sqrt(self) -> Self {
            let r = self.v.sqrt().max(1.0e-16);
            let mut g = [0.; 12];
            for i in 0..12 {
                g[i] = self.g[i] / (2. * r);
            }
            Self { v: r, g }
        }
    }
    #[derive(Clone, Copy)]
    struct V3 {
        x: D,
        y: D,
        z: D,
    }
    let pt = |p: Vec3, k: usize| V3 {
        x: D::var(p.x, k),
        y: D::var(p.y, k + 1),
        z: D::var(p.z, k + 2),
    };
    let (q0, q1, q2, q3) = (pt(p0, 0), pt(p1, 3), pt(p2, 6), pt(p3, 9));
    let sub = |a: V3, b: V3| V3 {
        x: a.x.sub(b.x),
        y: a.y.sub(b.y),
        z: a.z.sub(b.z),
    };
    let cross = |a: V3, b: V3| V3 {
        x: a.y.mul(b.z).sub(a.z.mul(b.y)),
        y: a.z.mul(b.x).sub(a.x.mul(b.z)),
        z: a.x.mul(b.y).sub(a.y.mul(b.x)),
    };
    let dot = |a: V3, b: V3| a.x.mul(b.x).add(a.y.mul(b.y)).add(a.z.mul(b.z));
    let b0 = sub(q1, q0);
    let b1 = sub(q2, q1);
    let b2 = sub(q3, q2);
    let n0 = cross(b0, b1);
    let n1 = cross(b1, b2);
    let r1 = dot(b1, b1).sqrt();
    let u1 = V3 {
        x: b1.x.div(r1),
        y: b1.y.div(r1),
        z: b1.z.div(r1),
    };
    let y = dot(cross(n0, n1), u1);
    let x = dot(n0, n1);
    let denom = x.mul(x).add(y.mul(y));
    let phi_v = y.v.atan2(x.v);
    let mut grad = [Vec3 {
        x: 0.,
        y: 0.,
        z: 0.,
    }; 4];
    for k in 0..12 {
        let dphi = (x.v * y.g[k] - y.v * x.g[k]) / denom.v.max(1.0e-32);
        grad[k / 3].x += if k % 3 == 0 { dphi } else { 0. };
        grad[k / 3].y += if k % 3 == 1 { dphi } else { 0. };
        grad[k / 3].z += if k % 3 == 2 { dphi } else { 0. };
    }
    let _ = D::cst(0.);
    (phi_v, grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    #[test]
    fn wrap_and_minimum_image_round_trip() {
        let b = BoxVectors::new(10., 12., 14.).unwrap();
        let p = v(12.5, -3., 28.);
        let w = b.wrap(p);
        assert!(w.x >= 0. && w.x < 10. && w.y >= 0. && w.y < 12. && w.z >= 0. && w.z < 14.);
        let d = b.displacement(v(9.5, 0., 0.), v(0.5, 0., 0.));
        assert!((d.x + 1.).abs() < 1e-12);
    }

    #[test]
    fn neighbor_skin_may_fill_the_half_box_margin() {
        // A valid physical cutoff can be close to half the box edge.  The
        // execution skin may then cover the complete box; rejecting that
        // conservative list would make an otherwise valid NPT contraction
        // fail before the force evaluator applies the real cutoff.
        let b = BoxVectors::new(20., 20., 20.).unwrap();
        let coords = [v(0., 0., 0.), v(9.5, 0., 0.)];
        let list = PbcNeighborList::build(&coords, &b, 9., 3.).unwrap();
        assert_eq!(list.pairs, vec![(0, 1)]);
        assert_eq!(list.cutoff, 9.);
        assert_eq!(list.skin, 3.);
    }

    #[test]
    fn reaction_field_matches_coulomb_at_short_range() {
        let rf = ReactionField::new(9., 78.5).unwrap();
        // Force correction is O((r/rc)^3): near-Coulomb at short range.
        let (_, analytic) = rf.pair(1., 332.063_713_299);
        assert!((analytic + 332.063_713_299).abs() < 2.);
        let (e_cut, _) = rf.pair(9., 1.);
        // Shifted so the potential is continuous at the cutoff.
        assert!(e_cut.abs() < 1e-9);
    }

    #[test]
    fn reaction_field_force_matches_finite_difference() {
        let rf = ReactionField::new(9., 78.5).unwrap();
        for r in [2., 4., 6.5, 8.9] {
            let (_, analytic) = rf.pair(r, 2.5);
            let (plus, _) = rf.pair(r + 1e-6, 2.5);
            let (minus, _) = rf.pair(r - 1e-6, 2.5);
            assert!(
                (analytic - (plus - minus) / 2e-6).abs() < 1e-6,
                "r={r} analytic={analytic}"
            );
        }
    }

    #[test]
    fn torsion_gradient_matches_finite_difference() {
        let p = [v(0., 0., 0.), v(1.5, 0., 0.), v(2., 1., 0.), v(3., 1., 1.)];
        let h = 1e-6;
        let energy = |q: [Vec3; 4]| {
            let (phi, _) = dihedral_with_gradient(q[0], q[1], q[2], q[3]);
            2. * (1. + (3. * phi - 0.5).cos())
        };
        let (_, grad) = dihedral_with_gradient(p[0], p[1], p[2], p[3]);
        let f = -3. * 2. * (3. * dihedral_with_gradient(p[0], p[1], p[2], p[3]).0 - 0.5).sin();
        for k in 0..4 {
            for axis in 0..3 {
                let mut plus = p;
                let mut minus = p;
                match axis {
                    0 => {
                        plus[k].x += h;
                        minus[k].x -= h;
                    }
                    1 => {
                        plus[k].y += h;
                        minus[k].y -= h;
                    }
                    _ => {
                        plus[k].z += h;
                        minus[k].z -= h;
                    }
                }
                let numeric = (energy(plus) - energy(minus)) / (2. * h);
                let g = [grad[k].x, grad[k].y, grad[k].z][axis];
                assert!((f * g - numeric).abs() < 1e-5, "k={k} axis={axis}");
            }
        }
    }

    #[test]
    fn virial_matches_volume_derivative() {
        // W = -dU/d(ln V): scale coords and box, compare against the
        // analytic virial. Catches sign and image-convention errors.
        let pdb = include_str!("../../../tests/fixtures/dipeptide.pdb");
        let options = glysys::BuildOptions {
            add_water: true,
            add_ions: false,
            padding_angstrom: 6.0,
            ..Default::default()
        };
        let system = glysys::SystemBuilder::new(options)
            .unwrap()
            .prepare_pdb_str(pdb)
            .unwrap();
        let field = PbcForceField::new(&system, vec![]).unwrap();
        let backend = ReactionField::new(4.0, 78.5).unwrap();
        let box_vec = BoxVectors::from_system(&system).unwrap();
        let coords = system.coordinates();
        let wrapped: Vec<Vec3> = coords.iter().map(|p| box_vec.wrap(*p)).collect();
        let pairs = PbcNeighborList::build(&wrapped, &box_vec, 4.0, 1.5).unwrap();
        let base = field
            .evaluate(&coords, &box_vec, &pairs.pairs, &backend, 4.0)
            .unwrap();
        // Keep the perturbation below the width at which a hard-cutoff pair
        // can enter or leave the list.  The reaction-field potential is
        // continuous at the cutoff, but its force is not, so a larger finite
        // difference would measure the cutoff discontinuity rather than the
        // virial at this configuration.
        let h = 1.0e-6;
        let s = 1.0 + h;
        let scaled_box = BoxVectors::new(box_vec.x * s, box_vec.y * s, box_vec.z * s).unwrap();
        let scaled: Vec<Vec3> = coords
            .iter()
            .map(|p| Vec3 {
                x: p.x * s,
                y: p.y * s,
                z: p.z * s,
            })
            .collect();
        let w2: Vec<Vec3> = scaled.iter().map(|p| scaled_box.wrap(*p)).collect();
        let pairs2 = PbcNeighborList::build(&w2, &scaled_box, 4.0, 1.5).unwrap();
        let up = field
            .evaluate(&scaled, &scaled_box, &pairs2.pairs, &backend, 4.0)
            .unwrap();
        // W = -3V dU/dV = -(U(s) - U(1))/ln(s): the 3 counts box
        // dimensionality (pressure acts on three face pairs).
        let s_minus = 1.0 - h;
        let scaled_box_minus = BoxVectors::new(
            box_vec.x * s_minus,
            box_vec.y * s_minus,
            box_vec.z * s_minus,
        )
        .unwrap();
        let scaled_minus: Vec<Vec3> = coords
            .iter()
            .map(|p| Vec3 {
                x: p.x * s_minus,
                y: p.y * s_minus,
                z: p.z * s_minus,
            })
            .collect();
        let wminus: Vec<Vec3> = scaled_minus
            .iter()
            .map(|p| scaled_box_minus.wrap(*p))
            .collect();
        let pairs_minus = PbcNeighborList::build(&wminus, &scaled_box_minus, 4.0, 1.5).unwrap();
        let down = field
            .evaluate(
                &scaled_minus,
                &scaled_box_minus,
                &pairs_minus.pairs,
                &backend,
                4.0,
            )
            .unwrap();
        let numeric = -(up.components.total() - down.components.total()) / (s.ln() - s_minus.ln());
        let analytic = base.virial;
        let diff = (numeric - analytic).abs();
        assert!(
            diff < 0.001 * analytic.abs().max(1.),
            "virial {analytic} vs volume derivative {numeric}"
        );
    }

    #[test]
    fn neighbor_list_respects_minimum_image() {
        let b = BoxVectors::new(10., 10., 10.).unwrap();
        let wrapped = vec![v(0.2, 5., 5.), v(9.8, 5., 5.), v(5., 5., 5.)];
        let list = PbcNeighborList::build(&wrapped, &b, 1.0, 0.2).unwrap();
        // First two atoms are 0.4 apart across the boundary.
        assert!(list.pairs.contains(&(0, 1)));
        assert!(!list.pairs.contains(&(0, 2)));
        // Displacing beyond half the skin flags a rebuild.
        let mut moved = wrapped.clone();
        moved[0].x += 0.5;
        assert!(list.needs_rebuild(&moved));
        assert!(!list.needs_rebuild(&wrapped));
    }

    #[test]
    fn neighbor_rebuild_uses_minimum_image_for_reference_motion_and_box_changes() {
        let old_box = BoxVectors::new(10., 10., 10.).unwrap();
        let wrapped = vec![v(0.2, 5., 5.)];
        let list = PbcNeighborList::build(&wrapped, &old_box, 1.0, 0.4).unwrap();
        // Crossing the periodic boundary by 9.8 Å is a 0.2 Å physical move,
        // so it must not trigger the half-skin rebuild threshold.
        let crossed = vec![v(10.0, 5., 5.)];
        assert!(!list.needs_rebuild(&crossed));
        let new_box = BoxVectors::new(10.1, 10., 10.).unwrap();
        assert!(list.box_changed(&new_box));
        assert!(!list.box_changed(&old_box));
    }
}

#[cfg(test)]
mod parity_probes {
    //! Hand-computed nonbonded probes: exclusions, 1-4 scaling, cutoff
    //! boundary, and minimum image, checked against independently written
    //! Amber formulas (not against the implementation's own helpers).
    use super::*;

    fn v(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    fn small_solvated() -> glysys::ParameterizedSystem {
        let pdb = include_str!("../../../tests/fixtures/dipeptide.pdb");
        let options = glysys::BuildOptions {
            add_water: true,
            add_ions: false,
            padding_angstrom: 6.0,
            ..Default::default()
        };
        glysys::SystemBuilder::new(options)
            .unwrap()
            .prepare_pdb_str(pdb)
            .unwrap()
    }

    const TIP3P_OH: f64 = 0.9572;

    #[test]
    fn excluded_pairs_contribute_nothing() {
        let system = small_solvated();
        let waters = classify_waters(&system);
        assert!(waters.len() >= 2, "need waters for exclusion probes");
        let [o, h1, h2] = waters[0];
        // O-H is 1-2 excluded, H-H is 1-3 excluded.
        assert!(system.exclusions()[o].contains(&h1));
        assert!(system.exclusions()[h1].contains(&h2));
        let field = PbcForceField::new(&system, vec![]).unwrap();
        let box_vec = BoxVectors::from_system(&system).unwrap();
        let coords = system.coordinates();
        let backend = ReactionField::new(4.0, 78.5).unwrap();
        let base = field
            .evaluate(&coords, &box_vec, &[], &backend, 4.0)
            .unwrap();
        assert_eq!(base.components.van_der_waals, 0.);
        assert_eq!(base.components.electrostatics, 0.);
        let probed = field
            .evaluate(&coords, &box_vec, &[(o, h1), (h1, h2)], &backend, 4.0)
            .unwrap();
        assert_eq!(probed.components.van_der_waals, 0.);
        assert_eq!(probed.components.electrostatics, 0.);
        assert_eq!(probed.gradients, base.gradients);
    }

    /// Hand-written Amber 1-4 formulas for one proper torsion's end pair.
    /// 1-4 electrostatics use plain Coulomb (scaled by 1/scee): they bypass
    /// the reaction-field backend exactly as OpenMM exceptions do (verified:
    /// plain sum 124.9516 vs RF-screened 64.0812 on the 40 solute pairs).
    fn hand_14(
        system: &glysys::ParameterizedSystem,
        a: usize,
        b: usize,
        r: f64,
        scee: f64,
        scnb: f64,
    ) -> (f64, f64, f64, f64) {
        let qa = system.atoms()[a].charge();
        let qb = system.atoms()[b].charge();
        let sig =
            system.atoms()[a].lennard_jones_radius() + system.atoms()[b].lennard_jones_radius();
        let eps = (system.atoms()[a].lennard_jones_epsilon()
            * system.atoms()[b].lennard_jones_epsilon())
        .sqrt()
            / scnb;
        let u = (sig / r).powi(6);
        let lj = eps * (u * u - 2. * u);
        // dE/dr for the gradient check below: dE/dr = -eps*(12u^2-12u)/r
        // since du/dr = -6u/r and dE/du = eps*(2u-2).
        let dlj = -eps * (12. * u * u - 12. * u) / r;
        let qq = COULOMB * qa * qb / scee;
        let rf = qq / r;
        let drf = -qq / (r * r);
        (lj, rf, dlj, drf)
    }

    #[test]
    fn one_four_scaling_matches_hand_computation() {
        let system = small_solvated();
        // Exclusions are built by 3-bond BFS, so every 1-4 pair is also
        // exclusion-listed; the evaluator computes it with 1-4 scaling
        // anyway (`excluded && scale.is_some()` branch). That exact branch is
        // what this probe covers.
        let torsion = system
            .dihedrals()
            .iter()
            .find(|t| !t.is_improper())
            .expect("need a proper torsion");
        let ends = torsion.atoms();
        let (a, b) = (ends[0], ends[3]);
        assert!(
            system.exclusions()[a].contains(&b),
            "builder lists 1-4 pairs as excluded-with-scale"
        );
        let scee = torsion.electrostatic_14_scale();
        let scnb = torsion.lennard_jones_14_scale();
        assert!(scee > 1. && scnb > 1., "Amber 1-4 scales expected");
        let field = PbcForceField::new(&system, vec![]).unwrap();
        let box_vec = BoxVectors::from_system(&system).unwrap();
        let coords = system.coordinates();
        let backend = ReactionField::new(4.0, 78.5).unwrap();
        let d = box_vec.displacement(coords[a], coords[b]);
        let r = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
        assert!(r < 4.0, "1-4 pair must be inside the probe cutoff, r={r}");
        let out = field
            .evaluate(&coords, &box_vec, &[(a, b)], &backend, 4.0)
            .unwrap();
        let (lj, rf, dlj, drf) = hand_14(&system, a, b, r, scee, scnb);
        assert!(
            (out.components.van_der_waals - lj).abs() < 1e-9,
            "LJ {} vs {lj}",
            out.components.van_der_waals
        );
        assert!(
            (out.components.electrostatics - rf).abs() < 1e-9,
            "RF {} vs {rf}",
            out.components.electrostatics
        );
        // Gradient check isolates the pair: bonded terms also write into
        // `gradients`, so subtract the pairs-empty baseline first. Stored
        // gradients are dE/dx (the integrator negates for forces).
        let base = field
            .evaluate(&coords, &box_vec, &[], &backend, 4.0)
            .unwrap();
        let fmag = (dlj + drf) / r;
        for (got, want) in [
            (out.gradients[a].x - base.gradients[a].x, fmag * d.x),
            (out.gradients[a].y - base.gradients[a].y, fmag * d.y),
            (out.gradients[a].z - base.gradients[a].z, fmag * d.z),
        ] {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
    }

    #[test]
    fn cutoff_boundary_and_minimum_image() {
        let system = small_solvated();
        let waters = classify_waters(&system);
        let [o1, _, _] = waters[0];
        let w2 = waters[1];
        let field = PbcForceField::new(&system, vec![]).unwrap();
        let box_vec = BoxVectors::from_system(&system).unwrap();
        let cutoff = 4.0;
        let backend = ReactionField::new(cutoff, 78.5).unwrap();
        let base_coords = system.coordinates();
        // Rigidly translate the second water so the O-O separation is an
        // exact prescribed value; bonded terms are untouched.
        let place = |r_target: f64| {
            let mut coords = base_coords.clone();
            let cur = box_vec.displacement(base_coords[w2[0]], base_coords[o1]);
            let cur_r = (cur.x * cur.x + cur.y * cur.y + cur.z * cur.z).sqrt();
            let shift = v(
                base_coords[o1].x + cur.x / cur_r * r_target - base_coords[w2[0]].x,
                base_coords[o1].y + cur.y / cur_r * r_target - base_coords[w2[0]].y,
                base_coords[o1].z + cur.z / cur_r * r_target - base_coords[w2[0]].z,
            );
            for &i in &w2 {
                coords[i] = v(
                    coords[i].x + shift.x,
                    coords[i].y + shift.y,
                    coords[i].z + shift.z,
                );
            }
            coords
        };
        // TIP3P O-O by hand (H atoms excluded from this pair's list).
        let hand_oo = |r: f64| {
            let sig = 2. * system.atoms()[o1].lennard_jones_radius();
            let eps = system.atoms()[o1].lennard_jones_epsilon();
            let u = (sig / r).powi(6);
            let lj = eps * (u * u - 2. * u);
            let qq = COULOMB * system.atoms()[o1].charge() * system.atoms()[w2[0]].charge();
            let rc: f64 = cutoff;
            let diel: f64 = 78.5;
            let krf = (diel - 1.) / (2. * diel + 1.) / rc.powi(3);
            let crf = 3. * diel / (2. * diel + 1.) / rc;
            lj + qq * (1. / r + krf * r * r - crf)
        };
        for (r_target, expect_zero) in [(cutoff - 1e-3, false), (cutoff + 1e-3, true)] {
            let coords = place(r_target);
            let out = field
                .evaluate(&coords, &box_vec, &[(o1, w2[0])], &backend, cutoff)
                .unwrap();
            let got = out.components.van_der_waals + out.components.electrostatics;
            if expect_zero {
                assert_eq!(got, 0., "pair outside cutoff must contribute nothing");
            } else {
                // Verifies the boundary pair is evaluated (not dropped) and
                // matches the hand formula through the wrap.
                assert!(
                    (got - hand_oo(r_target)).abs() < 1e-9,
                    "{got} vs {}",
                    hand_oo(r_target)
                );
            }
        }
        // Minimum image: separate the oxygens by nearly a full box edge along
        // x, so the raw distance is huge and only the wrapped 0.5 A counts.
        let mut coords = base_coords.clone();
        let want = v(
            base_coords[o1].x + box_vec.x - 0.5,
            base_coords[o1].y,
            base_coords[o1].z,
        );
        let shift = v(
            want.x - base_coords[w2[0]].x,
            want.y - base_coords[w2[0]].y,
            want.z - base_coords[w2[0]].z,
        );
        for &i in &w2 {
            coords[i] = v(
                coords[i].x + shift.x,
                coords[i].y + shift.y,
                coords[i].z + shift.z,
            );
        }
        let raw = {
            let d = v(
                coords[w2[0]].x - coords[o1].x,
                coords[w2[0]].y - coords[o1].y,
                coords[w2[0]].z - coords[o1].z,
            );
            (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()
        };
        assert!(raw > box_vec.x - 1., "test setup must span the box");
        let out = field
            .evaluate(&coords, &box_vec, &[(o1, w2[0])], &backend, cutoff)
            .unwrap();
        let got = out.components.van_der_waals + out.components.electrostatics;
        assert!(
            (got - hand_oo(0.5)).abs() < 1e-9,
            "minimum-image energy {got} vs {}",
            hand_oo(0.5)
        );
    }

    #[test]
    fn tip3p_geometry_matches_analytical_targets() {
        // Guard for the hand probes above: fixture waters really are TIP3P.
        let system = small_solvated();
        let coords = system.coordinates();
        // Solvate placement tolerance is ~1e-5 (measured 6.7e-6 max):
        // this only guards that the hand probes below see sane TIP3P water.
        for w in classify_waters(&system).iter().take(8) {
            let d = |a: usize, b: usize| {
                let dx = coords[a].x - coords[b].x;
                let dy = coords[a].y - coords[b].y;
                let dz = coords[a].z - coords[b].z;
                (dx * dx + dy * dy + dz * dz).sqrt()
            };
            assert!((d(w[0], w[1]) - TIP3P_OH).abs() < 1e-4);
            assert!((d(w[0], w[2]) - TIP3P_OH).abs() < 1e-4);
        }
    }

    #[test]
    fn dispersion_correction_has_expected_inverse_volume_dependence() {
        let system = small_solvated();
        let field = PbcForceField::new(&system, vec![]).unwrap();
        let backend = ReactionField::new(4.0, 78.5).unwrap();
        let box_a = BoxVectors::from_system(&system).unwrap();
        let coords = system.coordinates();
        let wrapped_a: Vec<Vec3> = coords.iter().map(|p| box_a.wrap(*p)).collect();
        let pairs_a = PbcNeighborList::build(&wrapped_a, &box_a, 4.0, 1.5).unwrap();
        let a = field
            .evaluate_with_dispersion(&coords, &box_a, &pairs_a.pairs, &backend, 4.0, true)
            .unwrap();

        let scale = 1.07;
        let box_b = BoxVectors::new(box_a.x * scale, box_a.y * scale, box_a.z * scale).unwrap();
        let coords_b: Vec<Vec3> = coords
            .iter()
            .map(|p| Vec3 {
                x: p.x * scale,
                y: p.y * scale,
                z: p.z * scale,
            })
            .collect();
        let wrapped_b: Vec<Vec3> = coords_b.iter().map(|p| box_b.wrap(*p)).collect();
        let pairs_b = PbcNeighborList::build(&wrapped_b, &box_b, 4.0, 1.5).unwrap();
        let b = field
            .evaluate_with_dispersion(&coords_b, &box_b, &pairs_b.pairs, &backend, 4.0, true)
            .unwrap();

        // The homogeneous correction is C/V.  Scaling all coordinates and
        // box vectors leaves the coefficient unchanged, so C recovered from
        // either evaluation must agree independently of the pair energy.
        let ca = a.components.dispersion_correction * box_a.volume();
        let cb = b.components.dispersion_correction * box_b.volume();
        assert!(ca.is_finite() && cb.is_finite());
        assert!((ca - cb).abs() <= 1e-10 * ca.abs().max(1.0));

        // Its volume derivative is -C/V²; this is the pressure contribution
        // used by the MC barostat and provides a dimensional regression
        // independent of the force virial.
        // For a finite interval the exact secant is -C/(V₁V₂), which avoids
        // conflating truncation error with a unit/conversion error.
        let expected = -ca / (box_a.volume() * box_b.volume());
        let finite = (b.components.dispersion_correction - a.components.dispersion_correction)
            / (box_b.volume() - box_a.volume());
        assert!((finite - expected).abs() < 1e-10 * expected.abs().max(1.0));
    }
}
