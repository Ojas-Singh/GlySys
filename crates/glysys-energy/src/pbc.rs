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
            let Some(a) = coords.get(anchor) else { continue };
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
    reference: Vec<Vec3>,
}

impl PbcNeighborList {
    pub fn build(wrapped: &[Vec3], box_vec: &BoxVectors, cutoff: f64, skin: f64) -> Result<Self> {
        if !cutoff.is_finite() || cutoff <= 0. || !skin.is_finite() || skin < 0. {
            return Err(EnergyError::InvalidConfiguration(
                "neighbor cutoff must be positive and skin non-negative".into(),
            ));
        }
        if cutoff + skin >= 0.5 * box_vec.x.min(box_vec.y).min(box_vec.z) {
            return Err(EnergyError::InvalidConfiguration(
                "cutoff plus skin must be below half the shortest box edge".into(),
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
        let (cx, cy, cz) = (box_vec.x / nx as f64, box_vec.y / ny as f64, box_vec.z / nz as f64);
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
                        let Some(other) = at(cell_key.0.saturating_add(dx), cell_key.1.saturating_add(dy), cell_key.2.saturating_add(dz))
                        else {
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
            reference: wrapped.to_vec(),
        })
    }

    pub fn needs_rebuild(&self, wrapped: &[Vec3]) -> bool {
        wrapped.len() != self.reference.len()
            || wrapped
                .iter()
                .zip(&self.reference)
                .any(|(c, r)| {
                    let dx = c.x - r.x;
                    let dy = c.y - r.y;
                    let dz = c.z - r.z;
                    dx * dx + dy * dy + dz * dz > (self.skin * 0.5).powi(2)
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

/// Periodic force field borrowing topology; positions are always unwrapped.
pub struct PbcForceField<'a> {
    system: &'a ParameterizedSystem,
    sigma: Vec<f64>,
    epsilon: Vec<f64>,
    charge: Vec<f64>,
    exclusions: Vec<std::collections::BTreeSet<usize>>,
    one_four: HashMap<(usize, usize), (f64, f64)>,
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
        Ok(Self {
            system,
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
            restraints,
        })
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
        if unwrapped.iter().any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite()) {
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
                virial_terms[1] -= f * (unwrapped[i].x * v.x + unwrapped[i].y * v.y + unwrapped[i].z * v.z);
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
        // Nonbonded pairs use the minimum image of wrapped positions.
        let wrapped: Vec<Vec3> = unwrapped.iter().map(|p| box_vec.wrap(*p)).collect();
        for &(a, b) in pairs {
            if a >= n || b >= n {
                continue;
            }
            let pair = ordered(a, b);
            let scale = self.one_four.get(&pair).copied();
            if self.exclusions[a].contains(&b) && scale.is_none() {
                continue;
            }
            let (scee, scnb) = scale.unwrap_or((1., 1.));
            let d = box_vec.displacement(wrapped[a], wrapped[b]);
            let r2 = d.x * d.x + d.y * d.y + d.z * d.z;
            // Verlet lists span cutoff plus skin; only pairs inside the
            // cutoff contribute, so rebuilds never move the Hamiltonian.
            if r2 > cutoff * cutoff {
                continue;
            }
            let r = r2.sqrt().max(1.0e-8);
            let sig = self.sigma[a] + self.sigma[b];
            let eps = (self.epsilon[a] * self.epsilon[b]).sqrt() / scnb;
            let ratio6 = (sig / r).powi(6);
            components.van_der_waals += eps * (ratio6 * ratio6 - 2. * ratio6);
            let qq = COULOMB * self.charge[a] * self.charge[b] / scee;
            let (ecoul, decoul_dr) = backend.pair(r, qq);
            components.electrostatics += ecoul;
            let flj = 12. * eps * (ratio6 - ratio6 * ratio6) / r;
            let fmag = (flj + decoul_dr) / r;
            gradients[a].x += fmag * d.x;
            gradients[a].y += fmag * d.y;
            gradients[a].z += fmag * d.z;
            gradients[b].x -= fmag * d.x;
            gradients[b].y -= fmag * d.y;
            gradients[b].z -= fmag * d.z;
            // Pair virial from the minimum-image vector: unwrapped
            // separations can span images and would corrupt the pressure.
            let w_pair = fmag * (d.x * d.x + d.y * d.y + d.z * d.z);
            virial_terms[3] -= w_pair;
            // LJ vs electrostatic split for pressure diagnostics.
            let w_lj = (flj / r) * (d.x * d.x + d.y * d.y + d.z * d.z);
            virial_pair_split[0] -= w_lj;
            virial_pair_split[1] -= w_pair - w_lj;
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
fn dihedral_with_gradient(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> (f64, [Vec3; 4]) {
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
            Self {
                v: self.v + o.v,
                g,
            }
        }
        fn sub(self, o: Self) -> Self {
            let mut g = [0.; 12];
            for i in 0..12 {
                g[i] = self.g[i] - o.g[i];
            }
            Self {
                v: self.v - o.v,
                g,
            }
        }
        fn mul(self, o: Self) -> Self {
            let mut g = [0.; 12];
            for i in 0..12 {
                g[i] = self.g[i] * o.v + self.v * o.g[i];
            }
            Self {
                v: self.v * o.v,
                g,
            }
        }
        fn div(self, o: Self) -> Self {
            let mut g = [0.; 12];
            for i in 0..12 {
                g[i] = (self.g[i] * o.v - self.v * o.g[i]) / (o.v * o.v);
            }
            Self {
                v: self.v / o.v,
                g,
            }
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
        z: 0.
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
        let s = 1.0 + 1e-5;
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
        let numeric = -(up.components.total() - base.components.total()) / s.ln();
        let analytic = base.virial;
        let diff = (numeric - analytic).abs();
        assert!(
            diff < 0.05 * analytic.abs().max(1.),
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
}
