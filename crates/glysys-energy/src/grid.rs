//! A tabulated receptor for docking a flexible ligand into a rigid binding site.
//!
//! The ligand-receptor interaction of a prepared system is a sum over atom
//! pairs of the same terms [`crate::EnergyEvaluator`] evaluates: a 12-6
//! Lennard-Jones term from the Amber/GLYCAM radii and well depths and a
//! Coulomb term from the partial charges. With the receptor fixed, the sum
//! over receptor atoms is a function of the ligand atom's position alone, so
//! it can be tabulated once on a regular grid: one potential for the
//! electrostatics (per unit charge) and one Lennard-Jones potential per
//! distinct ligand atom type. A pose then costs one trilinear interpolation
//! per ligand atom instead of a sum over the whole receptor.
//!
//! The grid is a search tool: interpolating a steep repulsive wall is
//! approximate, so a pose found on the grid should be rescored and relaxed
//! with the exact evaluator.

use glysys::{ParameterizedSystem, Vec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{EnergyError, Result};

/// Coulomb constant in kcal Å / (mol e²), as the evaluator uses.
const COULOMB: f64 = 332.063_713_3;
/// Pair distances are floored here before evaluating the potentials.
const MIN_DISTANCE: f64 = 0.4;
/// Grid values are capped, so that interpolation near a clash stays bounded.
const CAP: f64 = 500.0;
/// Receptor atoms beyond this distance do not contribute to the van der Waals grids.
const VDW_CUTOFF: f64 = 20.0;
/// The van der Waals grids hold `asinh(E / VDW_SCALE)`: interpolating that
/// instead of `E` follows the r^-12 repulsive wall far more closely.
const VDW_SCALE: f64 = 1.0;

/// How the grid screens the Coulomb term.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Screening {
    /// `q_i q_j / (eps r)`: the evaluator's own model.
    Constant { dielectric: f64 },
    /// `q_i q_j / (eps(r) r)` with the sigmoidal distance-dependent dielectric
    /// of Mehler and Solmajer (Protein Eng. 4, 903, 1991), which grows from
    /// about 1.4 at contact to 78 in the bulk: a cheap stand-in for solvent screening.
    MehlerSolmajer,
}

impl Screening {
    /// The relative permittivity at distance `r` (angstrom).
    pub fn permittivity(self, r: f64) -> f64 {
        match self {
            Screening::Constant { dielectric } => dielectric,
            Screening::MehlerSolmajer => {
                const A: f64 = -8.5525;
                const EPSILON0: f64 = 78.4;
                const K: f64 = 7.7839;
                const LAMBDA: f64 = 0.003_627;
                let b = EPSILON0 - A;
                A + b / (1.0 + K * (-LAMBDA * b * r).exp())
            }
        }
    }
}

/// Options for [`DockingGrid::build_with`].
#[derive(Debug, Clone, Copy)]
pub struct GridOptions {
    pub spacing: f64,
    pub screening: Screening,
    /// Treat receptor O-H and S-H hydrogens as free to turn about their C-O
    /// axis: each is carried as `ROTAMERS` positions instead of being part of
    /// the tabulated potential, and its energy with the ligand is a
    /// Boltzmann-weighted minimum over the rotamers (at `temperature`).
    pub flexible_hydroxyls: bool,
    /// kT in kcal/mol for the rotamer average.
    pub temperature: f64,
    /// Replace the generic ion-oxygen interaction near a metal with a coordination well.
    pub metal_coordination: Option<MetalCoordination>,
    /// Ceiling (kcal/mol) on the van der Waals energy a ligand atom feels at a grid point, in place of the
    /// default of 500: a soft wall that lets tight crystal contacts cost little.
    pub lj_cap: Option<f64>,
    /// A reward for ligand O and N atoms near receptor N and O atoms.
    pub polar_reward: Option<PolarReward>,
    /// Point sources that are not atoms of the system (for example crystal waters the structure fixer removed).
    pub extra_sources: &'static [ExtraSource],
    /// Soft-core electrostatics: floors on the distance in the Coulomb term (see [`CoulombFloor`]).
    pub coulomb_floor: Option<CoulombFloor>,
    /// Place the Lennard-Jones minimum of every (receptor metal ion, ligand oxygen) pair at the metal-oxygen distance
    /// observed in protein crystal structures ([`metal_oxygen_distance`]) instead of the combined radii of the force
    /// field: the 12-6 ion radii are fitted to hydration free energies and put a coordinating oxygen several kcal/mol up
    /// the repulsive wall at a crystal Ca-O distance.
    pub metal_oxygen_contact: bool,
}

/// Mean metal-oxygen distance (angstrom) in protein metal sites (Harding, Acta Cryst. D 62, 678 (2006): Ca 2.36-2.39,
/// Mg 2.07, Mn 2.19, Zn 2.09, Na 2.41, K 2.84), for the ions it lists; `None` for other elements.
pub fn metal_oxygen_distance(element: u8) -> Option<f64> {
    match element {
        11 => Some(2.41),
        12 => Some(2.07),
        19 => Some(2.84),
        20 => Some(2.38),
        25 => Some(2.19),
        30 => Some(2.09),
        _ => None,
    }
}

/// Floors on the distance at which the receptor's Coulomb potential is evaluated, so that inside them the potential is
/// flat: the electrostatic counterpart of the soft wall. With the van der Waals energy capped (`lj_cap`), a charged
/// ligand atom can otherwise fuse with a receptor atom of opposite charge (a hydroxyl oxygen onto a Ca2+, a hydroxyl
/// hydrogen onto a carboxylate oxygen) for the price of the cap while gaining hundreds of kcal/mol. The floors lie
/// below every legitimate contact distance (a hydrogen-bond H...O is at least about 1.6 A, a metal-oxygen bond about
/// 2.0-2.6 A), so poses that respect the van der Waals radii are scored as before.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoulombFloor {
    /// Floor (angstrom) for every receptor source that is not a metal ion.
    pub general: f64,
    /// Floor (angstrom) for receptor metal ions (the elements of `METALS`).
    pub metal: f64,
}

/// An extra receptor point source: a Lennard-Jones sphere with an optional charge, and whether it counts as a polar
/// (N or O) atom for the [`PolarReward`] field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtraSource {
    pub position: [f64; 3],
    pub charge: f64,
    /// Lennard-Jones radius (Rmin / 2) in angstrom and well depth in kcal/mol.
    pub radius: f64,
    pub epsilon: f64,
    pub polar: bool,
}

/// An empirical reward for ligand O and N atoms near receptor N and O atoms (a hydrogen bond without direction).
///
/// Each ligand O or N gains `depth` kcal/mol times the Gaussian `exp(-(d - distance)^2 / (2 width^2))` summed over
/// the receptor N and O atoms at distance `d`, tabulated on the grid; it is added to the force-field terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolarReward {
    pub depth: f64,
    pub distance: f64,
    pub width: f64,
}

/// An empirical bonus for a ligand oxygen coordinating a receptor metal ion.
///
/// A metal-bound sugar sits in the field of the ion's first shell: the ion
/// repels the neighbouring sugar carbons and hydroxyl hydrogens, and under the
/// distance-dependent screening used for the rest of the receptor those
/// repulsions cancel the attraction of the coordinating oxygens, so a pose held
/// by a calcium in the crystal drifts away from it. This adds a Gaussian of
/// the given `depth` (kcal/mol, a reward) and width `COORD_WIDTH` around
/// `distance` for every (ion, ligand oxygen) pair, on top of the generic terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetalCoordination {
    pub distance: f64,
    pub depth: f64,
}

/// Width (standard deviation) of the coordination bonus, angstrom.
const COORD_WIDTH: f64 = 0.3;
/// Pairs beyond this distance get no bonus.
const COORD_FAR: f64 = 4.4;
/// Elements treated as metal ions.
const METALS: [u8; 10] = [3, 11, 12, 19, 20, 25, 26, 27, 28, 29];

impl GridOptions {
    pub fn new(spacing: f64, screening: Screening) -> Self {
        Self {
            spacing,
            screening,
            flexible_hydroxyls: false,
            temperature: 0.6,
            metal_coordination: None,
            lj_cap: None,
            polar_reward: None,
            extra_sources: &[],
            coulomb_floor: None,
            metal_oxygen_contact: false,
        }
    }
}

/// Rotamers per flexible hydroxyl hydrogen.
const ROTAMERS: usize = 12;
/// Ligand atoms farther than this from a flexible hydrogen are not paired with it.
const FLEX_RANGE: f64 = 10.0;
/// Spacing of the tabulated screened Coulomb kernel used for flexible hydrogens.
const PAIR_STEP: f64 = 0.01;
/// Ligand atoms with a smaller partial charge are not paired with flexible hydrogens.
const MIN_PAIR_CHARGE: f64 = 0.05;

/// A receptor metal ion.
#[derive(Debug, Clone)]
struct Ion {
    position: [f64; 3],
}

/// A ligand oxygen that can coordinate an ion.
#[derive(Debug, Clone)]
struct OxygenProbe {
    /// Position in the ligand atom list.
    index: usize,
}

/// A receptor hydroxyl hydrogen carried as discrete rotamers.
#[derive(Debug, Clone)]
struct FlexibleHydrogen {
    charge: f64,
    positions: [[f64; 3]; ROTAMERS],
    /// Screened Coulomb energy of each rotamer with the rest of the receptor.
    self_energy: [f64; ROTAMERS],
}

#[derive(Debug, Clone)]
pub struct DockingGrid {
    origin: [f64; 3],
    spacing: f64,
    dims: [usize; 3],
    /// Lennard-Jones (radius, well depth) of each ligand atom class.
    classes: Vec<(f64, f64)>,
    /// One grid per class: `asinh(E / VDW_SCALE)` of the van der Waals energy
    /// of a probe of that class.
    vdw: Vec<Vec<f32>>,
    /// The electrostatic potential per unit charge, kcal/mol/e.
    potential: Vec<f32>,
    ligand_class: Vec<usize>,
    ligand_charge: Vec<f64>,
    screening: Screening,
    flexible: Vec<FlexibleHydrogen>,
    temperature: f64,
    coordination: Option<MetalCoordination>,
    ions: Vec<Ion>,
    oxygens: Vec<OxygenProbe>,
    /// `1 / (eps(r) r)` every `PAIR_STEP` angstrom up to `FLEX_RANGE`, for the flexible hydrogens.
    pair_table: Vec<f64>,
    polar: Option<PolarField>,
}

/// The polar reward's grid: the sum of Gaussians around the receptor N and O atoms, and which ligand atoms are O or N.
#[derive(Debug, Clone)]
struct PolarField {
    reward: PolarReward,
    field: Vec<f32>,
    ligand: Vec<bool>,
}

/// A trilinear interpolant's value and gradient (per angstrom).
fn trilinear(
    grid: &[f32],
    dims: [usize; 3],
    spacing: f64,
    f: [f64; 3],
    cell: [usize; 3],
) -> (f64, [f64; 3]) {
    let index = |i: usize, j: usize, k: usize| {
        ((cell[0] + i) * dims[1] + (cell[1] + j)) * dims[2] + (cell[2] + k)
    };
    let c = |i, j, k| f64::from(grid[index(i, j, k)]);
    let (c000, c100, c010, c110) = (c(0, 0, 0), c(1, 0, 0), c(0, 1, 0), c(1, 1, 0));
    let (c001, c101, c011, c111) = (c(0, 0, 1), c(1, 0, 1), c(0, 1, 1), c(1, 1, 1));
    let [fx, fy, fz] = f;
    let (gx, gy, gz) = (1.0 - fx, 1.0 - fy, 1.0 - fz);
    let value = gx * (gy * (gz * c000 + fz * c001) + fy * (gz * c010 + fz * c011))
        + fx * (gy * (gz * c100 + fz * c101) + fy * (gz * c110 + fz * c111));
    let dx = gy * (gz * (c100 - c000) + fz * (c101 - c001))
        + fy * (gz * (c110 - c010) + fz * (c111 - c011));
    let dy = gx * (gz * (c010 - c000) + fz * (c011 - c001))
        + fx * (gz * (c110 - c100) + fz * (c111 - c101));
    let dz = gx * (gy * (c001 - c000) + fy * (c011 - c010))
        + fx * (gy * (c101 - c100) + fy * (c111 - c110));
    (value, [dx / spacing, dy / spacing, dz / spacing])
}

impl DockingGrid {
    /// Tabulate the receptor of `system` (atoms `receptor`, at `coordinates`)
    /// for the ligand atoms `ligand`, over a cube of half-width `half_width`
    /// angstrom around `center`, on a grid of `spacing` angstrom.
    pub fn build(
        system: &ParameterizedSystem,
        coordinates: &[Vec3],
        receptor: &[usize],
        ligand: &[usize],
        center: Vec3,
        half_width: f64,
        spacing: f64,
        screening: Screening,
    ) -> Result<Self> {
        Self::build_with(
            system,
            coordinates,
            receptor,
            ligand,
            center,
            half_width,
            GridOptions::new(spacing, screening),
        )
    }

    pub fn build_with(
        system: &ParameterizedSystem,
        coordinates: &[Vec3],
        receptor: &[usize],
        ligand: &[usize],
        center: Vec3,
        half_width: f64,
        options: GridOptions,
    ) -> Result<Self> {
        let GridOptions {
            spacing,
            screening,
            flexible_hydroxyls,
            temperature,
            metal_coordination,
            lj_cap,
            polar_reward,
            extra_sources,
            coulomb_floor,
            metal_oxygen_contact,
        } = options;
        let lj_ceiling = lj_cap.map_or(CAP, |c| c.min(CAP));
        if spacing <= 0.0 || half_width <= spacing || coordinates.len() != system.atom_count() {
            return Err(EnergyError::InvalidConfiguration(
                "grid needs a positive spacing, a larger half-width and one coordinate per atom"
                    .into(),
            ));
        }
        let atoms = system.atoms();
        // Ligand atom classes: atoms with the same Lennard-Jones parameters share a grid.
        let mut classes: Vec<(f64, f64)> = Vec::new();
        let mut ligand_class = Vec::with_capacity(ligand.len());
        for &i in ligand {
            let key = (
                atoms[i].lennard_jones_radius(),
                atoms[i].lennard_jones_epsilon(),
            );
            let class = classes
                .iter()
                .position(|c| (c.0 - key.0).abs() < 1.0e-9 && (c.1 - key.1).abs() < 1.0e-9)
                .unwrap_or_else(|| {
                    classes.push(key);
                    classes.len() - 1
                });
            ligand_class.push(class);
        }
        // Classes made of oxygen atoms (the GLYCAM and Amber oxygen types have their own Lennard-Jones parameters).
        let mut class_oxygen = vec![true; classes.len()];
        for (&i, &class) in ligand.iter().zip(&ligand_class) {
            class_oxygen[class] &= atoms[i].element() == 8;
        }
        let n = (2.0 * half_width / spacing).ceil() as usize + 1;
        let dims = [n, n, n];
        let origin = [
            center.x - half_width,
            center.y - half_width,
            center.z - half_width,
        ];

        // Receptor hydroxyl hydrogens handled as rotamers (O-H or S-H with one heavy neighbour).
        let mut flexible_atoms = std::collections::BTreeSet::new();
        let mut flexible = Vec::new();
        if flexible_hydroxyls {
            let in_receptor: std::collections::BTreeSet<usize> = receptor.iter().copied().collect();
            let mut neighbors = vec![Vec::new(); atoms.len()];
            for bond in system.bonds() {
                let [a, b] = bond.atoms();
                neighbors[a].push(b);
                neighbors[b].push(a);
            }
            for &h in receptor {
                if atoms[h].element() != 1 || neighbors[h].len() != 1 {
                    continue;
                }
                let o = neighbors[h][0];
                if !matches!(atoms[o].element(), 8 | 16)
                    || neighbors[o].len() != 2
                    || !in_receptor.contains(&o)
                {
                    continue;
                }
                let c = neighbors[o].iter().copied().find(|&n| n != h).unwrap();
                if atoms[c].element() == 1 {
                    continue;
                }
                let (po, pc, ph) = (coordinates[o], coordinates[c], coordinates[h]);
                let axis = [po.x - pc.x, po.y - pc.y, po.z - pc.z];
                let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
                if norm < 1.0e-9 {
                    continue;
                }
                let axis = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
                let rel = [ph.x - po.x, ph.y - po.y, ph.z - po.z];
                let along = rel[0] * axis[0] + rel[1] * axis[1] + rel[2] * axis[2];
                let radial = [
                    rel[0] - along * axis[0],
                    rel[1] - along * axis[1],
                    rel[2] - along * axis[2],
                ];
                let radius =
                    (radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2]).sqrt();
                if radius < 1.0e-9 {
                    continue;
                }
                let e1 = [radial[0] / radius, radial[1] / radius, radial[2] / radius];
                let e2 = [
                    axis[1] * e1[2] - axis[2] * e1[1],
                    axis[2] * e1[0] - axis[0] * e1[2],
                    axis[0] * e1[1] - axis[1] * e1[0],
                ];
                let mut positions = [[0.0; 3]; ROTAMERS];
                for (k, p) in positions.iter_mut().enumerate() {
                    let (sn, cs) = (std::f64::consts::TAU * k as f64 / ROTAMERS as f64).sin_cos();
                    let origin = [po.x, po.y, po.z];
                    for d in 0..3 {
                        p[d] = origin[d] + along * axis[d] + radius * (cs * e1[d] + sn * e2[d]);
                    }
                }
                flexible_atoms.insert(h);
                flexible.push((h, positions));
            }
        }
        let sources: Vec<([f64; 3], f64, f64, f64)> = receptor
            .iter()
            .map(|&j| {
                let c = coordinates[j];
                // A flexible hydrogen is not part of the tabulated potential.
                let charge = if flexible_atoms.contains(&j) {
                    0.0
                } else {
                    atoms[j].charge()
                };
                (
                    [c.x, c.y, c.z],
                    charge,
                    atoms[j].lennard_jones_radius(),
                    atoms[j].lennard_jones_epsilon(),
                )
            })
            .collect();
        let mut sources = sources;
        sources.extend(
            extra_sources
                .iter()
                .map(|e| (e.position, e.charge, e.radius, e.epsilon)),
        );
        let source_count = sources.len();
        // Per source, the distance below which its Coulomb potential stays flat (MIN_DISTANCE without a floor).
        let floors: Vec<f64> = (0..source_count)
            .map(|j| match coulomb_floor {
                None => MIN_DISTANCE,
                Some(floor) => {
                    let metal =
                        j < receptor.len() && METALS.contains(&atoms[receptor[j]].element());
                    (if metal { floor.metal } else { floor.general }).max(MIN_DISTANCE)
                }
            })
            .collect();
        let positions: Vec<[f64; 3]> = sources.iter().map(|s| s.0).collect();
        // Charge per source with the Coulomb constant folded in.
        let charges: Vec<f64> = sources.iter().map(|s| COULOMB * s.1).collect();
        // Per (class, source): the well depth and sigma^6 of the pair, so the
        // inner loop is a few multiply-adds.
        let mut well_depth = vec![0.0f64; classes.len() * source_count];
        let mut sigma6 = vec![0.0f64; classes.len() * source_count];
        for (k, class) in classes.iter().enumerate() {
            for (j, source) in sources.iter().enumerate() {
                well_depth[k * source_count + j] = (class.1 * source.3).sqrt();
                let contact = (metal_oxygen_contact && class_oxygen[k] && j < receptor.len())
                    .then(|| metal_oxygen_distance(atoms[receptor[j]].element()))
                    .flatten();
                sigma6[k * source_count + j] = contact.unwrap_or(class.0 + source.2).powi(6);
            }
        }
        // 1 / (eps(r) r) on a fine table; the sigmoid is too costly per pair.
        const TABLE_STEP: f64 = 0.01;
        let table_len = (400.0 / TABLE_STEP) as usize + 2;
        let screened: Vec<f64> = (0..table_len)
            .map(|i| {
                let r = (i as f64 * TABLE_STEP).max(MIN_DISTANCE);
                1.0 / (screening.permittivity(r) * r)
            })
            .collect();

        let total = n * n * n;
        let mut vdw = vec![vec![0.0f32; total]; classes.len()];
        let mut potential = vec![0.0f32; total];
        // One x-slab per task; each fills its slice of every grid.
        let slab = n * n;
        let mut vdw_slabs: Vec<Vec<&mut [f32]>> = (0..n).map(|_| Vec::new()).collect();
        for grid in &mut vdw {
            for (x, chunk) in grid.chunks_mut(slab).enumerate() {
                vdw_slabs[x].push(chunk);
            }
        }
        let cutoff2 = VDW_CUTOFF * VDW_CUTOFF;
        let class_count = classes.len();
        potential
            .par_chunks_mut(slab)
            .zip(vdw_slabs.into_par_iter())
            .enumerate()
            .for_each(|(x, (potential_slab, mut vdw_slab))| {
                let mut well = vec![0.0f64; class_count];
                for y in 0..n {
                    for z in 0..n {
                        let p = [
                            origin[0] + x as f64 * spacing,
                            origin[1] + y as f64 * spacing,
                            origin[2] + z as f64 * spacing,
                        ];
                        well.iter_mut().for_each(|w| *w = 0.0);
                        let mut phi = 0.0;
                        for (j, q) in positions.iter().enumerate() {
                            let d2 = (p[0] - q[0]).powi(2)
                                + (p[1] - q[1]).powi(2)
                                + (p[2] - q[2]).powi(2);
                            let r = d2.sqrt().max(MIN_DISTANCE);
                            let at = r.max(floors[j]) / TABLE_STEP;
                            let (i, f) = (at as usize, at - (at as usize) as f64);
                            phi += charges[j] * (screened[i] * (1.0 - f) + screened[i + 1] * f);
                            if d2 < cutoff2 {
                                let inv6 = 1.0 / (r * r).powi(3);
                                for k in 0..class_count {
                                    let sig6 = sigma6[k * source_count + j];
                                    well[k] += well_depth[k * source_count + j]
                                        * sig6
                                        * inv6
                                        * (sig6 * inv6 - 2.0);
                                }
                            }
                        }
                        let at = y * n + z;
                        potential_slab[at] = phi.clamp(-CAP, CAP) as f32;
                        for (k, slab_k) in vdw_slab.iter_mut().enumerate() {
                            slab_k[at] = (well[k].min(lj_ceiling) / VDW_SCALE).asinh() as f32;
                        }
                    }
                }
            });
        // Self energy of each rotamer: its Coulomb energy with the rest of the receptor.
        let flexible: Vec<FlexibleHydrogen> = flexible
            .into_iter()
            .map(|(h, positions)| {
                let charge = atoms[h].charge();
                let excluded = &system.exclusions()[h];
                let mut self_energy = [0.0; ROTAMERS];
                for (k, p) in positions.iter().enumerate() {
                    let mut e = 0.0;
                    for &j in receptor {
                        if j == h || excluded.contains(&j) || atoms[j].charge() == 0.0 {
                            continue;
                        }
                        let c = coordinates[j];
                        let r =
                            ((p[0] - c.x).powi(2) + (p[1] - c.y).powi(2) + (p[2] - c.z).powi(2))
                                .sqrt()
                                .max(0.5);
                        e += COULOMB * charge * atoms[j].charge() / (screening.permittivity(r) * r);
                    }
                    self_energy[k] = e;
                }
                FlexibleHydrogen {
                    charge,
                    positions,
                    self_energy,
                }
            })
            // Only hydrogens that can reach the ligand's region matter.
            .filter(|f| {
                let to_center = [
                    f.positions[0][0] - center.x,
                    f.positions[0][1] - center.y,
                    f.positions[0][2] - center.z,
                ];
                (to_center[0].powi(2) + to_center[1].powi(2) + to_center[2].powi(2)).sqrt()
                    < half_width + FLEX_RANGE
            })
            .collect();
        let polar = polar_reward.map(|reward| {
            let sources: Vec<[f64; 3]> = receptor
                .iter()
                .filter(|&&j| matches!(atoms[j].element(), 7 | 8))
                .map(|&j| [coordinates[j].x, coordinates[j].y, coordinates[j].z])
                .chain(extra_sources.iter().filter(|e| e.polar).map(|e| e.position))
                .collect();
            let reach = reward.distance + 4.0 * reward.width;
            let reach2 = reach * reach;
            let mut field = vec![0.0f32; total];
            field
                .par_chunks_mut(slab)
                .enumerate()
                .for_each(|(x, chunk)| {
                    for y in 0..n {
                        for z in 0..n {
                            let p = [
                                origin[0] + x as f64 * spacing,
                                origin[1] + y as f64 * spacing,
                                origin[2] + z as f64 * spacing,
                            ];
                            let mut sum = 0.0;
                            for q in &sources {
                                let (dx, dy, dz) = (p[0] - q[0], p[1] - q[1], p[2] - q[2]);
                                let d2 = dx * dx + dy * dy + dz * dz;
                                if d2 < reach2 {
                                    let u = (d2.sqrt() - reward.distance) / reward.width;
                                    sum += (-0.5 * u * u).exp();
                                }
                            }
                            chunk[y * n + z] = sum as f32;
                        }
                    }
                });
            PolarField {
                reward,
                field,
                ligand: ligand
                    .iter()
                    .map(|&i| matches!(atoms[i].element(), 7 | 8))
                    .collect(),
            }
        });
        let (ions, oxygens) = if metal_coordination.is_some() {
            let ions = receptor
                .iter()
                .filter(|&&j| METALS.contains(&atoms[j].element()))
                .map(|&j| {
                    let c = coordinates[j];
                    Ion {
                        position: [c.x, c.y, c.z],
                    }
                })
                .collect::<Vec<_>>();
            let oxygens = ligand
                .iter()
                .enumerate()
                .filter(|&(_, &i)| atoms[i].element() == 8)
                .map(|(index, _)| OxygenProbe { index })
                .collect::<Vec<_>>();
            (ions, oxygens)
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(Self {
            origin,
            spacing,
            dims,
            classes,
            vdw,
            potential,
            ligand_class,
            ligand_charge: ligand.iter().map(|&i| atoms[i].charge()).collect(),
            screening,
            flexible,
            temperature,
            coordination: metal_coordination,
            ions,
            oxygens,
            pair_table: (0..=((FLEX_RANGE / PAIR_STEP) as usize + 2))
                .map(|i| {
                    let r = (i as f64 * PAIR_STEP).max(0.5);
                    1.0 / (screening.permittivity(r) * r)
                })
                .collect(),
            polar,
        })
    }

    pub fn screening(&self) -> Screening {
        self.screening
    }

    pub fn ligand_atoms(&self) -> usize {
        self.ligand_class.len()
    }

    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    /// Edge length of the tabulated cube, angstrom.
    pub fn extent(&self) -> f64 {
        (self.dims[0] - 1) as f64 * self.spacing
    }

    /// Interaction energy of the ligand atoms at `positions` (in the order
    /// given to `build`) with the receptor, kcal/mol; the gradient with
    /// respect to each position is added to `gradient` when it is given.
    /// A position outside the cube is evaluated at the nearest point of it
    /// plus a quadratic penalty, so a search is pushed back inside.
    pub fn energy_gradient(&self, positions: &[Vec3], mut gradient: Option<&mut [Vec3]>) -> f64 {
        const OUTSIDE: f64 = 10.0;
        let mut total = 0.0;
        for (a, position) in positions.iter().enumerate() {
            let p = [position.x, position.y, position.z];
            let mut inside = [0.0; 3];
            let mut cell = [0usize; 3];
            let mut f = [0.0; 3];
            let mut penalty = 0.0;
            let mut penalty_gradient = [0.0; 3];
            for k in 0..3 {
                let top = (self.dims[k] - 1) as f64 * self.spacing;
                let local = p[k] - self.origin[k];
                let clamped = local.clamp(0.0, top);
                let outside = local - clamped;
                penalty += OUTSIDE * outside * outside;
                penalty_gradient[k] = 2.0 * OUTSIDE * outside;
                inside[k] = clamped;
                let scaled = clamped / self.spacing;
                cell[k] = (scaled.floor() as usize).min(self.dims[k] - 2);
                f[k] = scaled - cell[k] as f64;
            }
            let class = self.ligand_class[a];
            let (compressed, compressed_gradient) =
                trilinear(&self.vdw[class], self.dims, self.spacing, f, cell);
            let well = VDW_SCALE * compressed.sinh();
            let slope = VDW_SCALE * compressed.cosh();
            let well_gradient = [
                slope * compressed_gradient[0],
                slope * compressed_gradient[1],
                slope * compressed_gradient[2],
            ];
            let (phi, phi_gradient) = trilinear(&self.potential, self.dims, self.spacing, f, cell);
            let charge = self.ligand_charge[a];
            let (mut reward, mut reward_gradient) = (0.0, [0.0; 3]);
            if let Some(polar) = &self.polar
                && polar.ligand[a]
            {
                let (value, slope) = trilinear(&polar.field, self.dims, self.spacing, f, cell);
                reward = -polar.reward.depth * value;
                reward_gradient = slope.map(|g| -polar.reward.depth * g);
            }
            total += well + charge * phi + penalty + reward;
            if let Some(gradient) = gradient.as_deref_mut() {
                gradient[a].x += well_gradient[0]
                    + charge * phi_gradient[0]
                    + penalty_gradient[0]
                    + reward_gradient[0];
                gradient[a].y += well_gradient[1]
                    + charge * phi_gradient[1]
                    + penalty_gradient[1]
                    + reward_gradient[1];
                gradient[a].z += well_gradient[2]
                    + charge * phi_gradient[2]
                    + penalty_gradient[2]
                    + reward_gradient[2];
            }
        }
        let flexible = self.flexible_term(positions, gradient.as_deref_mut());
        total + flexible + self.coordination_term(positions, gradient)
    }

    /// The tabulated van der Waals and electrostatic energy of each ligand atom at `positions` (kcal/mol), without
    /// the outside penalty, the polar reward, the flexible hydrogens or the coordination term: a diagnostic of which
    /// atoms carry the interaction energy.
    pub fn atom_terms(&self, positions: &[Vec3]) -> Vec<[f64; 2]> {
        positions
            .iter()
            .enumerate()
            .map(|(a, position)| {
                let p = [position.x, position.y, position.z];
                let mut cell = [0usize; 3];
                let mut f = [0.0; 3];
                for k in 0..3 {
                    let top = (self.dims[k] - 1) as f64 * self.spacing;
                    let scaled = (p[k] - self.origin[k]).clamp(0.0, top) / self.spacing;
                    cell[k] = (scaled.floor() as usize).min(self.dims[k] - 2);
                    f[k] = scaled - cell[k] as f64;
                }
                let (compressed, _) = trilinear(
                    &self.vdw[self.ligand_class[a]],
                    self.dims,
                    self.spacing,
                    f,
                    cell,
                );
                let (phi, _) = trilinear(&self.potential, self.dims, self.spacing, f, cell);
                [VDW_SCALE * compressed.sinh(), self.ligand_charge[a] * phi]
            })
            .collect()
    }

    /// Number of metal ions treated with a coordination well.
    pub fn coordinating_ions(&self) -> usize {
        if self.coordination.is_some() {
            self.ions.len()
        } else {
            0
        }
    }

    /// The coordination bonus summed over every (ion, ligand oxygen) pair within `COORD_FAR`.
    fn coordination_term(&self, positions: &[Vec3], mut gradient: Option<&mut [Vec3]>) -> f64 {
        let Some(model) = self.coordination else {
            return 0.0;
        };
        let mut total = 0.0;
        for ion in &self.ions {
            for oxygen in &self.oxygens {
                let q = &positions[oxygen.index];
                let delta = [
                    q.x - ion.position[0],
                    q.y - ion.position[1],
                    q.z - ion.position[2],
                ];
                let d = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
                if d >= COORD_FAR {
                    continue;
                }
                let offset = (d - model.distance) / COORD_WIDTH;
                let bonus = -model.depth * (-0.5 * offset * offset).exp();
                total += bonus;
                if let Some(gradient) = gradient.as_deref_mut()
                    && d > 1.0e-6
                {
                    // d bonus / d d = -bonus * offset / width
                    let slope = -bonus * offset / COORD_WIDTH;
                    gradient[oxygen.index].x += slope * delta[0] / d;
                    gradient[oxygen.index].y += slope * delta[1] / d;
                    gradient[oxygen.index].z += slope * delta[2] / d;
                }
            }
        }
        total
    }

    /// Number of receptor hydroxyl hydrogens carried as rotamers.
    pub fn flexible_hydrogens(&self) -> usize {
        self.flexible.len()
    }

    /// The Boltzmann-weighted minimum over rotamers of each flexible hydrogen's
    /// energy with the ligand and the receptor, relative to the same quantity
    /// with no ligand (so the term vanishes far from the ligand).
    fn flexible_term(&self, positions: &[Vec3], mut gradient: Option<&mut [Vec3]>) -> f64 {
        if self.flexible.is_empty() {
            return 0.0;
        }
        let kt = self.temperature;
        let range2 = FLEX_RANGE * FLEX_RANGE;
        let mut total = 0.0;
        // Ligand atoms that carry charge, and their bounding box: a hydroxyl
        // farther than the pair range from the box has nothing to pair with.
        let charged: Vec<usize> = (0..positions.len())
            .filter(|&a| self.ligand_charge[a].abs() > MIN_PAIR_CHARGE)
            .collect();
        if charged.is_empty() {
            return 0.0;
        }
        let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for &a in &charged {
            let q = [positions[a].x, positions[a].y, positions[a].z];
            for k in 0..3 {
                low[k] = low[k].min(q[k]);
                high[k] = high[k].max(q[k]);
            }
        }
        for hydrogen in &self.flexible {
            let p = hydrogen.positions[0];
            let outside: f64 = (0..3)
                .map(|k| (low[k] - p[k]).max(p[k] - high[k]).max(0.0).powi(2))
                .sum();
            // The rotamer cone is within 1 A of the first position.
            if outside.sqrt() > FLEX_RANGE + 1.0 {
                continue;
            }
            let mut energies = [0.0; ROTAMERS];
            for (k, p) in hydrogen.positions.iter().enumerate() {
                let mut e = hydrogen.self_energy[k];
                for &a in &charged {
                    let q = &positions[a];
                    let d2 = (p[0] - q.x).powi(2) + (p[1] - q.y).powi(2) + (p[2] - q.z).powi(2);
                    if d2 > range2 {
                        continue;
                    }
                    e += COULOMB * hydrogen.charge * self.ligand_charge[a] * self.kernel(d2.sqrt());
                }
                energies[k] = e;
            }
            let lowest = energies.iter().copied().fold(f64::INFINITY, f64::min);
            let weights: Vec<f64> = energies
                .iter()
                .map(|e| (-(e - lowest) / kt).exp())
                .collect();
            let partition: f64 = weights.iter().sum();
            let free = lowest - kt * partition.ln();
            let reference_lowest = hydrogen
                .self_energy
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let reference: f64 = hydrogen
                .self_energy
                .iter()
                .map(|e| (-(e - reference_lowest) / kt).exp())
                .sum();
            total += free - (reference_lowest - kt * reference.ln());
            if let Some(gradient) = gradient.as_deref_mut() {
                for (k, p) in hydrogen.positions.iter().enumerate() {
                    let weight = weights[k] / partition;
                    if weight < 1.0e-6 {
                        continue;
                    }
                    for &a in &charged {
                        let q = &positions[a];
                        let delta = [q.x - p[0], q.y - p[1], q.z - p[2]];
                        let d2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
                        if d2 > range2 {
                            continue;
                        }
                        let r = d2.sqrt().max(0.5);
                        let slope = COULOMB
                            * hydrogen.charge
                            * self.ligand_charge[a]
                            * self.kernel_slope(d2.sqrt());
                        gradient[a].x += weight * slope * delta[0] / r;
                        gradient[a].y += weight * slope * delta[1] / r;
                        gradient[a].z += weight * slope * delta[2] / r;
                    }
                }
            }
        }
        total
    }

    /// `1 / (eps(r) r)` from the table, linearly interpolated (`r` up to `FLEX_RANGE`).
    fn kernel(&self, r: f64) -> f64 {
        let at = r.max(0.5) / PAIR_STEP;
        let i = at as usize;
        let f = at - i as f64;
        self.pair_table[i] * (1.0 - f) + self.pair_table[i + 1] * f
    }

    /// The derivative of [`Self::kernel`] with respect to `r`.
    fn kernel_slope(&self, r: f64) -> f64 {
        if r < 0.5 {
            return 0.0;
        }
        let i = (r / PAIR_STEP) as usize;
        (self.pair_table[i + 1] - self.pair_table[i]) / PAIR_STEP
    }

    pub fn energy(&self, positions: &[Vec3]) -> f64 {
        self.energy_gradient(positions, None)
    }

    /// The van der Waals and electrostatic parts of [`Self::energy`] (inside the cube only).
    pub fn components(&self, positions: &[Vec3]) -> (f64, f64) {
        let (mut well_total, mut coulomb_total) = (0.0, 0.0);
        for (a, position) in positions.iter().enumerate() {
            let p = [position.x, position.y, position.z];
            let mut cell = [0usize; 3];
            let mut f = [0.0; 3];
            for k in 0..3 {
                let top = (self.dims[k] - 1) as f64 * self.spacing;
                let scaled = (p[k] - self.origin[k]).clamp(0.0, top) / self.spacing;
                cell[k] = (scaled.floor() as usize).min(self.dims[k] - 2);
                f[k] = scaled - cell[k] as f64;
            }
            well_total += VDW_SCALE
                * trilinear(
                    &self.vdw[self.ligand_class[a]],
                    self.dims,
                    self.spacing,
                    f,
                    cell,
                )
                .0
                .sinh();
            coulomb_total += self.ligand_charge[a]
                * trilinear(&self.potential, self.dims, self.spacing, f, cell).0;
        }
        (well_total, coulomb_total)
    }
}

/// The electrostatic energy among the atoms of one molecule with the same
/// distance-dependent screening a [`DockingGrid`] uses for the molecule's
/// interaction with its receptor.
///
/// With one screening function for both, a glycan's internal hydrogen bonds
/// cost the same as the ones it makes with the protein; a constant dielectric
/// for the inside and a sigmoidal one for the outside would favour folding
/// hydroxyls back onto the sugar. Pairs are those the force field does not
/// exclude, with the 1-4 pairs scaled by their `scee`. Pair this with
/// `ParameterizedSystem::without_charges` for the bonded and van der Waals terms.
#[derive(Debug, Clone)]
pub struct ScreenedInternal {
    /// (first, second, 332 q q' / scee)
    pairs: Vec<(usize, usize, f64)>,
    table: Vec<f64>,
}

/// Table spacing and reach of [`ScreenedInternal`], angstrom.
const INTERNAL_STEP: f64 = 0.01;
const INTERNAL_REACH: f64 = 80.0;

impl ScreenedInternal {
    /// `system` is the molecule's own (sub)system.
    pub fn new(system: &ParameterizedSystem, screening: Screening) -> Self {
        let atoms = system.atoms();
        let one_four: std::collections::HashMap<(usize, usize), f64> = system
            .one_four_pairs()
            .into_iter()
            .map(|(pair, scee, _)| ((pair[0].min(pair[1]), pair[0].max(pair[1])), scee))
            .collect();
        let exclusions = system.exclusions();
        let mut pairs = Vec::new();
        for i in 0..atoms.len() {
            for j in i + 1..atoms.len() {
                let scale = one_four.get(&(i, j)).copied();
                if exclusions[i].contains(&j) && scale.is_none() {
                    continue;
                }
                let coefficient =
                    COULOMB * atoms[i].charge() * atoms[j].charge() / scale.unwrap_or(1.0);
                if coefficient != 0.0 {
                    pairs.push((i, j, coefficient));
                }
            }
        }
        let table = (0..=((INTERNAL_REACH / INTERNAL_STEP) as usize + 2))
            .map(|k| {
                let r = (k as f64 * INTERNAL_STEP).max(0.5);
                1.0 / (screening.permittivity(r) * r)
            })
            .collect();
        Self { pairs, table }
    }

    /// The energy of the molecule with its atoms at `positions`; the gradient
    /// with respect to each position is added to `gradient` when it is given.
    pub fn energy_gradient(&self, positions: &[Vec3], mut gradient: Option<&mut [Vec3]>) -> f64 {
        let mut total = 0.0;
        for &(i, j, coefficient) in &self.pairs {
            let delta = [
                positions[j].x - positions[i].x,
                positions[j].y - positions[i].y,
                positions[j].z - positions[i].z,
            ];
            let r = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
            let at = r.max(0.5) / INTERNAL_STEP;
            let k = (at as usize).min(self.table.len() - 2);
            let f = at - k as f64;
            total += coefficient * (self.table[k] * (1.0 - f) + self.table[k + 1] * f);
            if let Some(gradient) = gradient.as_deref_mut()
                && r > 0.5
            {
                let slope = coefficient * (self.table[k + 1] - self.table[k]) / INTERNAL_STEP;
                let scale = slope / r;
                gradient[j].x += scale * delta[0];
                gradient[j].y += scale * delta[1];
                gradient[j].z += scale * delta[2];
                gradient[i].x -= scale * delta[0];
                gradient[i].y -= scale * delta[1];
                gradient[i].z -= scale * delta[2];
            }
        }
        total
    }

    pub fn energy(&self, positions: &[Vec3]) -> f64 {
        self.energy_gradient(positions, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mehler_solmajer_screening_grows_from_contact_to_bulk() {
        let s = Screening::MehlerSolmajer;
        assert!((s.permittivity(0.0) - 1.35).abs() < 0.05);
        assert!(s.permittivity(3.0) > 10.0 && s.permittivity(3.0) < 16.0);
        assert!(s.permittivity(30.0) > 70.0 && s.permittivity(30.0) < 80.0);
    }
}
