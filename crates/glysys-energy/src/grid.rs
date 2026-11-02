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
    /// Directional hydrogen bonding and the unsatisfied-buried-polar penalty (see [`HydrogenBond`]).
    pub hydrogen_bond: Option<HydrogenBond>,
    /// Point sources that are not atoms of the system (for example crystal waters the structure fixer removed).
    pub extra_sources: &'static [ExtraSource],
    /// Soft-core electrostatics: floors on the distance in the Coulomb term (see [`CoulombFloor`]).
    pub coulomb_floor: Option<CoulombFloor>,
    /// Place the Lennard-Jones minimum of every (receptor metal ion, ligand oxygen) pair at the metal-oxygen distance
    /// observed in protein crystal structures ([`metal_oxygen_distance`]) instead of the combined radii of the force
    /// field: the 12-6 ion radii are fitted to hydration free energies and put a coordinating oxygen several kcal/mol up
    /// the repulsive wall at a crystal Ca-O distance.
    pub metal_oxygen_contact: bool,
    /// A reward (kcal/mol per glycan C-H carbon) for CH-pi stacking over the aromatic rings of Phe, Tyr, Trp and His
    /// (see [`ch_pi_weight`]); `None` switches it off.
    pub ch_pi_depth: Option<f64>,
    /// Weights of the tabulated van der Waals and electrostatic terms in the energy (1 and 1 by default), for scoring
    /// functions whose weights are fitted.
    pub term_weights: [f64; 2],
    /// Depths (kcal/mol per unit field) of the contact terms of [`CONTACT_KINDS`]; `None` builds none of their fields,
    /// zeros build them without effect on the energy (to record them for a fit).
    pub contact_depths: Option<[f64; 4]>,
}

/// The contact terms, in the order of `contact_depths` and [`GridTerms::contacts`]: each is a sum of Gaussians
/// `exp(-(d - distance)^2 / (2 width^2))` over a set of receptor atoms, felt by a set of ligand atoms.
/// - `hydrophobic`: receptor carbons bonded to no N or O, felt by ligand carbons (4.0 A, width 0.5);
/// - `cation`: Lys NZ and Arg NE/NH1/NH2, felt by ligand oxygens (3.0 A, width 0.35);
/// - `carboxylate`: Asp OD1/OD2 and Glu OE1/OE2, felt by ligand oxygens (2.7 A, width 0.3);
/// - `oxygen_burial`: every receptor heavy atom that is not a metal, felt by ligand oxygens (4.0 A, width 0.6), a
///   burial count of the glycan's oxygens (a desolvation proxy, whose weight may come out negative).
pub const CONTACT_KINDS: [&str; 4] = ["hydrophobic", "cation", "carboxylate", "oxygen_burial"];
const CONTACT_SHAPES: [(f64, f64); 4] = [(4.0, 0.5), (3.0, 0.35), (2.7, 0.3), (4.0, 0.6)];

/// A sum of Gaussians around `sources` on the grid (zero beyond `distance + 4 width`).
fn gaussian_field(
    sources: &[[f64; 3]],
    distance: f64,
    width: f64,
    origin: [f64; 3],
    spacing: f64,
    n: usize,
) -> Vec<f32> {
    let reach = distance + 4.0 * width;
    let reach2 = reach * reach;
    let mut field = vec![0.0f32; n * n * n];
    field
        .par_chunks_mut(n * n)
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
                    for q in sources {
                        let d2 =
                            (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
                        if d2 < reach2 {
                            let u = (d2.sqrt() - distance) / width;
                            sum += (-0.5 * u * u).exp();
                        }
                    }
                    chunk[y * n + z] = sum as f32;
                }
            }
        });
    field
}

/// The interaction terms of a pose at unit weight (see [`DockingGrid::terms`]): the energy is
/// `w_vdw * vdw + w_elec * elec + polar_depth * polar + ch_pi_depth * ch_pi + coordination_depth * coordination + outside`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct GridTerms {
    pub vdw: f64,
    pub elec: f64,
    /// Minus the polar reward field summed over the ligand O and N atoms (0 when the field was not built).
    pub polar: f64,
    /// Minus the CH-pi field summed over the ligand C-H carbons.
    pub ch_pi: f64,
    /// Minus the metal coordination Gaussians summed over (ion, ligand oxygen) pairs.
    pub coordination: f64,
    /// Minus the directional hydrogen bonds donated by the ligand hydroxyls (0 when the term is off).
    pub hbond_donor: f64,
    /// Minus the directional hydrogen bonds accepted by the ligand hydroxyls (0 when the term is off).
    pub hbond_acceptor: f64,
    /// Buried ligand hydroxyls that found no partner (0 when the term is off).
    pub hbond_unsatisfied: f64,
    /// The quadratic penalty for atoms outside the grid.
    pub outside: f64,
    /// Minus each contact field (see [`CONTACT_KINDS`]) summed over its ligand atoms (0 when not built).
    pub contacts: [f64; 4],
}

/// Geometric weight (0 to 1) of a CH-pi contact for a carbon at `offset` from an aromatic ring centre with unit normal
/// `normal`: 1 when the carbon lies 3.4-4.3 A from the centre within 25 degrees of the normal, tapering smoothly
/// (cosine) to 0 at 3.0 and 4.8 A and at 45 degrees; the usual geometry of a sugar C-H stacked on an aromatic side chain.
pub fn ch_pi_weight(offset: [f64; 3], normal: [f64; 3]) -> f64 {
    let d = (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]).sqrt();
    if !(3.0..4.8).contains(&d) {
        return 0.0;
    }
    // 1 at `full`, 0 at `zero`, a half cosine in between (only called between the two)
    let taper = |x: f64, full: f64, zero: f64| {
        0.5 * (1.0 + (std::f64::consts::PI * (x - full) / (zero - full)).cos())
    };
    let radial = if d < 3.4 {
        taper(d, 3.4, 3.0)
    } else if d > 4.3 {
        taper(d, 4.3, 4.8)
    } else {
        1.0
    };
    let cos = ((offset[0] * normal[0] + offset[1] * normal[1] + offset[2] * normal[2]) / d)
        .abs()
        .min(1.0);
    let angle = cos.acos().to_degrees();
    let angular = if angle <= 25.0 {
        1.0
    } else if angle >= 45.0 {
        0.0
    } else {
        taper(angle, 25.0, 45.0)
    };
    radial * angular
}

/// Aromatic rings of the receptor (Phe, Tyr, Trp, His in their Amber names): centre and unit normal.
fn aromatic_rings(
    system: &ParameterizedSystem,
    coordinates: &[Vec3],
    receptor: &[usize],
) -> Vec<([f64; 3], [f64; 3])> {
    const SIX: [&str; 6] = ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"];
    const TRP6: [&str; 6] = ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"];
    const TRP5: [&str; 5] = ["CG", "CD1", "NE1", "CE2", "CD2"];
    const HIS5: [&str; 5] = ["CG", "ND1", "CD2", "CE1", "NE2"];
    let in_receptor: std::collections::BTreeSet<usize> = receptor.iter().copied().collect();
    let atoms = system.atoms();
    let mut rings = Vec::new();
    for residue in system.residues() {
        let range = residue.atom_range();
        if !in_receptor.contains(&range.start) {
            continue;
        }
        let name = residue.name();
        let sets: Vec<&[&str]> = match name {
            "PHE" | "TYR" | "NPHE" | "CPHE" | "NTYR" | "CTYR" => vec![&SIX],
            "TRP" | "NTRP" | "CTRP" => vec![&TRP6, &TRP5],
            "HIS" | "HID" | "HIE" | "HIP" | "NHID" | "NHIE" | "NHIP" | "CHID" | "CHIE" | "CHIP" => {
                vec![&HIS5]
            }
            _ => continue,
        };
        for set in sets {
            let points: Vec<[f64; 3]> = set
                .iter()
                .filter_map(|n| range.clone().find(|&i| atoms[i].name() == *n))
                .map(|i| [coordinates[i].x, coordinates[i].y, coordinates[i].z])
                .collect();
            if points.len() != set.len() {
                continue;
            }
            let k = points.len() as f64;
            let c = [0, 1, 2].map(|d| points.iter().map(|p| p[d]).sum::<f64>() / k);
            // normal: the cross product of two chords across the ring
            let a = [0, 1, 2].map(|d| points[0][d] - c[d]);
            let b = [0, 1, 2].map(|d| points[points.len() / 2][d] - c[d]);
            let e = [0, 1, 2].map(|d| points[1][d] - c[d]);
            let f = [0, 1, 2].map(|d| points[points.len() / 2 + 1][d] - c[d]);
            let u = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
            let v = [e[0] - f[0], e[1] - f[1], e[2] - f[2]];
            let nrm = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
            if len > 1.0e-9 {
                rings.push((c, [nrm[0] / len, nrm[1] / len, nrm[2] / len]));
            }
        }
    }
    rings
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

/// Directional hydrogen bonding for the ligand's hydroxyls, and a penalty for burying one that finds no
/// partner.
///
/// The electrostatics of this grid is a floored Coulomb potential between point charges, which is blind to
/// the geometry of a hydrogen bond: it cannot tell a hydroxyl donating along its O-H from one that merely
/// has an oxygen nearby, and `oxygen_burial` penalises burial whether or not the buried oxygen is
/// satisfied. Those are the terms a sugar needs to tell one hydroxyl pair from another, which is how most
/// of our mis-ranked poses differ from the right one.
///
/// Evaluated pairwise over the receptor polar atoms near each ligand hydroxyl (a cell list, so the cost is
/// the number of hydroxyls times a small neighbourhood, not a new grid). The geometry is fixed, from the
/// neutron and CSD hydrogen-bond literature; only the three depths are meant to be fitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydrogenBond {
    /// kcal/mol for an ideal bond donated by a ligand hydroxyl
    pub donor_depth: f64,
    /// kcal/mol for an ideal bond accepted by a ligand hydroxyl
    pub acceptor_depth: f64,
    /// kcal/mol for a fully buried hydroxyl with no partner at all
    pub unsatisfied_depth: f64,
}

/// Ideal heavy-atom distance of a ligand O-H donating to a receptor acceptor, angstrom, and its spread.
const HB_DONOR_DISTANCE: (f64, f64) = (2.85, 0.40);
/// Ideal distance from a receptor polar hydrogen to the accepting ligand oxygen, angstrom, and its spread.
const HB_ACCEPTOR_DISTANCE: (f64, f64) = (1.90, 0.35);
/// The ideal approach of a donor to an accepting ligand hydroxyl, as the cosine of the angle between the
/// C->O direction and O->H: a C-O...H angle of 115 degrees makes that cos 65 degrees. With its spread.
const HB_LONE_PAIR: (f64, f64) = (0.4226, 0.35);
/// Beyond this a pair contributes nothing, angstrom.
const HB_CUTOFF: f64 = 4.0;
/// Orientations of a ligand hydroxyl's O-H about its C-O bond, Boltzmann-averaged by the donor term.
const HB_DONOR_ROTAMERS: usize = 12;
/// Burial: receptor heavy atoms count fully within the first distance and not at all beyond the second.
const HB_BURIAL_RANGE: (f64, f64) = (4.0, 6.5);
/// Number of close receptor heavy atoms at which a hydroxyl counts as fully buried.
const HB_BURIAL_FULL: f64 = 12.0;

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
            hydrogen_bond: None,
            extra_sources: &[],
            coulomb_floor: None,
            metal_oxygen_contact: false,
            ch_pi_depth: None,
            term_weights: [1.0, 1.0],
            contact_depths: None,
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
    /// The CH-pi reward's field (the ring weights summed) and its ligand atoms (carbons that carry a hydrogen).
    ch_pi: Option<PolarField>,
    /// Weights of the van der Waals and electrostatic terms.
    term_weights: [f64; 2],
    /// The contact terms' fields and ligand masks, with their depths in `reward.depth`.
    contacts: Vec<PolarField>,
    /// Directional hydrogen bonding (see [`HydrogenBond`]).
    hbond: Option<HbReceptor>,
}

/// A receptor atom that can accept a hydrogen bond, with the direction its lone pairs point (from the
/// heavy atom it hangs on, so a carbonyl oxygen points away from its carbon).
#[derive(Debug, Clone, Copy)]
struct HbAcceptor {
    position: [f64; 3],
    lone_direction: [f64; 3],
}

/// A receptor polar hydrogen and the heavy atom that donates it.
#[derive(Debug, Clone, Copy)]
struct HbDonor {
    hydrogen: [f64; 3],
    donor: [f64; 3],
}

/// A ligand hydroxyl: the oxygen, its hydrogen and the carbon it hangs on, as indices into the pose.
#[derive(Debug, Clone, Copy)]
struct HbHydroxyl {
    oxygen: usize,
    hydrogen: Option<usize>,
    carbon: Option<usize>,
}

/// Everything the hydrogen-bond term needs about the receptor, with a uniform cell list over the polar
/// atoms and the heavy atoms so a pose touches only the neighbourhood of each hydroxyl.
#[derive(Debug, Clone)]
struct HbReceptor {
    model: HydrogenBond,
    acceptors: Vec<HbAcceptor>,
    donors: Vec<HbDonor>,
    heavy: Vec<[f64; 3]>,
    hydroxyls: Vec<HbHydroxyl>,
    cells: HbCells,
}

/// Uniform cells over three point sets (acceptors, donors, heavy atoms) sharing one grid.
#[derive(Debug, Clone)]
struct HbCells {
    origin: [f64; 3],
    size: f64,
    dims: [usize; 3],
    acceptors: Vec<Vec<u32>>,
    donors: Vec<Vec<u32>>,
    heavy: Vec<Vec<u32>>,
}

impl HbCells {
    fn new(sets: [&[[f64; 3]]; 3], size: f64) -> Self {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for set in sets {
            for p in set {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
        if !lo[0].is_finite() {
            lo = [0.0; 3];
            hi = [0.0; 3];
        }
        let dims = [0, 1, 2].map(|k| (((hi[k] - lo[k]) / size).floor() as usize) + 1);
        let count = dims[0] * dims[1] * dims[2];
        let mut cells = Self {
            origin: lo,
            size,
            dims,
            acceptors: vec![Vec::new(); count],
            donors: vec![Vec::new(); count],
            heavy: vec![Vec::new(); count],
        };
        for (which, set) in sets.iter().enumerate() {
            for (i, p) in set.iter().enumerate() {
                let c = cells.cell_of(*p);
                match which {
                    0 => cells.acceptors[c].push(i as u32),
                    1 => cells.donors[c].push(i as u32),
                    _ => cells.heavy[c].push(i as u32),
                }
            }
        }
        cells
    }

    fn cell_of(&self, p: [f64; 3]) -> usize {
        let c = [0, 1, 2].map(|k| {
            (((p[k] - self.origin[k]) / self.size).floor().max(0.0) as usize).min(self.dims[k] - 1)
        });
        (c[0] * self.dims[1] + c[1]) * self.dims[2] + c[2]
    }

    /// Calls `f` with the index of every point of `which` (0 acceptors, 1 donors, 2 heavy) whose cell is
    /// within `radius` of `p`.
    fn near(&self, which: usize, p: [f64; 3], radius: f64, mut f: impl FnMut(usize)) {
        let lo = [0, 1, 2].map(|k| {
            ((p[k] - radius - self.origin[k]) / self.size).floor().max(0.0) as usize
        });
        let hi = [0, 1, 2].map(|k| {
            let v = ((p[k] + radius - self.origin[k]) / self.size).floor();
            if v < 0.0 { None } else { Some((v as usize).min(self.dims[k] - 1)) }
        });
        let (Some(h0), Some(h1), Some(h2)) = (hi[0], hi[1], hi[2]) else {
            return;
        };
        for i in lo[0]..=h0 {
            for j in lo[1]..=h1 {
                for k in lo[2]..=h2 {
                    let c = (i * self.dims[1] + j) * self.dims[2] + k;
                    let list = match which {
                        0 => &self.acceptors[c],
                        1 => &self.donors[c],
                        _ => &self.heavy[c],
                    };
                    for &idx in list {
                        f(idx as usize);
                    }
                }
            }
        }
    }
}

/// The polar reward's grid: the sum of Gaussians around the receptor N and O atoms, and which ligand atoms are O or N.
#[derive(Debug, Clone)]
struct PolarField {
    reward: PolarReward,
    field: Vec<f32>,
    ligand: Vec<bool>,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// The unit vector and the length it came from.
fn unit(a: [f64; 3]) -> ([f64; 3], f64) {
    let n = norm(a).max(1.0e-9);
    ([a[0] / n, a[1] / n, a[2] / n], n)
}

/// 1 below `near`, 0 above `far`, a smooth cosine in between.
fn switch_down(r: f64, near: f64, far: f64) -> f64 {
    if r <= near {
        1.0
    } else if r >= far {
        0.0
    } else {
        0.5 * (1.0 + (std::f64::consts::PI * (r - near) / (far - near)).cos())
    }
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
            hydrogen_bond,
            extra_sources,
            coulomb_floor,
            metal_oxygen_contact,
            ch_pi_depth,
            term_weights,
            contact_depths,
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
        let ch_pi = ch_pi_depth.map(|depth| {
            let rings = aromatic_rings(system, coordinates, receptor);
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
                            let sum: f64 = rings
                                .iter()
                                .map(|(c, normal)| {
                                    ch_pi_weight([p[0] - c[0], p[1] - c[1], p[2] - c[2]], *normal)
                                })
                                .sum();
                            chunk[y * n + z] = sum as f32;
                        }
                    }
                });
            let mut has_hydrogen = vec![false; atoms.len()];
            for bond in system.bonds() {
                let [a, b] = bond.atoms();
                if atoms[a].element() == 1 {
                    has_hydrogen[b] = true;
                }
                if atoms[b].element() == 1 {
                    has_hydrogen[a] = true;
                }
            }
            PolarField {
                reward: PolarReward {
                    depth,
                    distance: 0.0,
                    width: 1.0,
                },
                field,
                ligand: ligand
                    .iter()
                    .map(|&i| atoms[i].element() == 6 && has_hydrogen[i])
                    .collect(),
            }
        });
        let contacts: Vec<PolarField> = match contact_depths {
            None => Vec::new(),
            Some(depths) => {
                let mut bonded_polar = vec![false; atoms.len()];
                for bond in system.bonds() {
                    let [a, b] = bond.atoms();
                    if matches!(atoms[b].element(), 7 | 8) {
                        bonded_polar[a] = true;
                    }
                    if matches!(atoms[a].element(), 7 | 8) {
                        bonded_polar[b] = true;
                    }
                }
                let mut residue_name = vec![""; atoms.len()];
                for residue in system.residues() {
                    for i in residue.atom_range() {
                        residue_name[i] = residue.name();
                    }
                }
                let pick = |keep: &dyn Fn(usize) -> bool| -> Vec<[f64; 3]> {
                    receptor
                        .iter()
                        .copied()
                        .filter(|&j| keep(j))
                        .map(|j| [coordinates[j].x, coordinates[j].y, coordinates[j].z])
                        .collect()
                };
                let cationic = |j: usize| {
                    let r = residue_name[j];
                    let n = atoms[j].name();
                    (r.ends_with("LYS") || r.ends_with("LYN")) && n == "NZ"
                        || r.ends_with("ARG") && matches!(n, "NE" | "NH1" | "NH2")
                };
                let carboxylate = |j: usize| {
                    let r = residue_name[j];
                    let n = atoms[j].name();
                    (r.ends_with("ASP") || r.ends_with("ASH")) && matches!(n, "OD1" | "OD2")
                        || (r.ends_with("GLU") || r.ends_with("GLH")) && matches!(n, "OE1" | "OE2")
                };
                let sources = [
                    pick(&|j| atoms[j].element() == 6 && !bonded_polar[j]),
                    pick(&cationic),
                    pick(&carboxylate),
                    pick(&|j| atoms[j].element() != 1 && !METALS.contains(&atoms[j].element())),
                ];
                (0..4)
                    .map(|k| PolarField {
                        reward: PolarReward {
                            depth: depths[k],
                            distance: CONTACT_SHAPES[k].0,
                            width: CONTACT_SHAPES[k].1,
                        },
                        field: gaussian_field(
                            &sources[k],
                            CONTACT_SHAPES[k].0,
                            CONTACT_SHAPES[k].1,
                            origin,
                            spacing,
                            n,
                        ),
                        ligand: ligand
                            .iter()
                            .map(|&i| {
                                if k == 0 {
                                    atoms[i].element() == 6
                                } else {
                                    atoms[i].element() == 8
                                }
                            })
                            .collect(),
                    })
                    .collect()
            }
        };
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
        // Hydrogen bonding: the receptor's acceptors and polar hydrogens, and the ligand's hydroxyls.
        let hbond = hydrogen_bond.map(|model| {
            let mut attached: Vec<Option<usize>> = vec![None; atoms.len()];
            let mut hydrogens_of: Vec<Vec<usize>> = vec![Vec::new(); atoms.len()];
            let mut carbon_of: Vec<Option<usize>> = vec![None; atoms.len()];
            for bond in system.bonds() {
                let [a, b] = bond.atoms();
                for (x, y) in [(a, b), (b, a)] {
                    if atoms[x].element() == 1 {
                        continue;
                    }
                    if atoms[y].element() == 1 {
                        hydrogens_of[x].push(y);
                    } else {
                        if attached[x].is_none() {
                            attached[x] = Some(y);
                        }
                        if atoms[y].element() == 6 && carbon_of[x].is_none() {
                            carbon_of[x] = Some(y);
                        }
                    }
                }
            }
            let v = |i: usize| [coordinates[i].x, coordinates[i].y, coordinates[i].z];
            let mut acceptors = Vec::new();
            let mut donors = Vec::new();
            let mut heavy = Vec::new();
            for &j in receptor {
                let element = atoms[j].element();
                if element == 1 {
                    // a polar hydrogen is one on a nitrogen or oxygen
                    if let Some(d) = attached[j].or_else(|| {
                        system.bonds().iter().find_map(|b| {
                            let [x, y] = b.atoms();
                            if x == j { Some(y) } else if y == j { Some(x) } else { None }
                        })
                    }) && matches!(atoms[d].element(), 7 | 8)
                    {
                        donors.push(HbDonor { hydrogen: v(j), donor: v(d) });
                    }
                    continue;
                }
                heavy.push(v(j));
                if matches!(element, 7 | 8) && !METALS.contains(&element) {
                    let position = v(j);
                    let lone_direction = match attached[j] {
                        Some(h) => {
                            let d = [position[0] - coordinates[h].x, position[1] - coordinates[h].y, position[2] - coordinates[h].z];
                            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1.0e-9);
                            [d[0] / n, d[1] / n, d[2] / n]
                        }
                        // no heavy neighbour (a lone water oxygen): accept from any side
                        None => [0.0, 0.0, 0.0],
                    };
                    acceptors.push(HbAcceptor { position, lone_direction });
                }
            }
            let hydroxyls = ligand
                .iter()
                .enumerate()
                .filter(|&(_, &i)| atoms[i].element() == 8)
                .map(|(index, &i)| {
                    let place = |target: Option<usize>| {
                        target.and_then(|t| ligand.iter().position(|&l| l == t))
                    };
                    HbHydroxyl {
                        oxygen: index,
                        hydrogen: place(hydrogens_of[i].first().copied()),
                        carbon: place(carbon_of[i]),
                    }
                })
                .collect();
            let acceptor_points: Vec<[f64; 3]> = acceptors.iter().map(|a| a.position).collect();
            let donor_points: Vec<[f64; 3]> = donors.iter().map(|d| d.hydrogen).collect();
            let cells = HbCells::new([&acceptor_points, &donor_points, &heavy], HB_BURIAL_RANGE.1);
            HbReceptor { model, acceptors, donors, heavy, hydroxyls, cells }
        });

        Ok(Self {
            origin,
            spacing,
            dims,
            classes,
            vdw,
            potential,
            ligand_class,
            hbond,
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
            ch_pi,
            term_weights,
            contacts,
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
            if let Some(ch_pi) = &self.ch_pi
                && ch_pi.ligand[a]
            {
                let (value, slope) = trilinear(&ch_pi.field, self.dims, self.spacing, f, cell);
                reward -= ch_pi.reward.depth * value;
                for k in 0..3 {
                    reward_gradient[k] -= ch_pi.reward.depth * slope[k];
                }
            }
            for contact in &self.contacts {
                if contact.ligand[a] && contact.reward.depth != 0.0 {
                    let (value, slope) =
                        trilinear(&contact.field, self.dims, self.spacing, f, cell);
                    reward -= contact.reward.depth * value;
                    for k in 0..3 {
                        reward_gradient[k] -= contact.reward.depth * slope[k];
                    }
                }
            }
            let [wv, we] = self.term_weights;
            total += wv * well + we * charge * phi + penalty + reward;
            if let Some(gradient) = gradient.as_deref_mut() {
                gradient[a].x += wv * well_gradient[0]
                    + we * charge * phi_gradient[0]
                    + penalty_gradient[0]
                    + reward_gradient[0];
                gradient[a].y += wv * well_gradient[1]
                    + we * charge * phi_gradient[1]
                    + penalty_gradient[1]
                    + reward_gradient[1];
                gradient[a].z += wv * well_gradient[2]
                    + we * charge * phi_gradient[2]
                    + penalty_gradient[2]
                    + reward_gradient[2];
            }
        }
        let flexible = self.flexible_term(positions, gradient.as_deref_mut());
        total
            + flexible
            + self.coordination_term(positions, gradient.as_deref_mut())
            + self.hydrogen_bond_term(positions, gradient)
    }

    /// The interaction terms of the pose at unit weight, for fitting the weights of the score; the energy of
    /// [`Self::energy`] is their weighted sum plus the flexible-hydrogen term (not included here).
    pub fn terms(&self, positions: &[Vec3]) -> GridTerms {
        const OUTSIDE: f64 = 10.0;
        let mut t = GridTerms::default();
        for (a, position) in positions.iter().enumerate() {
            let p = [position.x, position.y, position.z];
            let mut cell = [0usize; 3];
            let mut f = [0.0; 3];
            for k in 0..3 {
                let top = (self.dims[k] - 1) as f64 * self.spacing;
                let local = p[k] - self.origin[k];
                let clamped = local.clamp(0.0, top);
                t.outside += OUTSIDE * (local - clamped).powi(2);
                let scaled = clamped / self.spacing;
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
            t.vdw += VDW_SCALE * compressed.sinh();
            let (phi, _) = trilinear(&self.potential, self.dims, self.spacing, f, cell);
            t.elec += self.ligand_charge[a] * phi;
            if let Some(polar) = &self.polar
                && polar.ligand[a]
            {
                t.polar -= trilinear(&polar.field, self.dims, self.spacing, f, cell).0;
            }
            if let Some(ch_pi) = &self.ch_pi
                && ch_pi.ligand[a]
            {
                t.ch_pi -= trilinear(&ch_pi.field, self.dims, self.spacing, f, cell).0;
            }
            for (k, contact) in self.contacts.iter().enumerate() {
                if contact.ligand[a] {
                    t.contacts[k] -= trilinear(&contact.field, self.dims, self.spacing, f, cell).0;
                }
            }
        }
        if let Some(model) = self.coordination {
            for ion in &self.ions {
                for oxygen in &self.oxygens {
                    let q = &positions[oxygen.index];
                    let d = ((q.x - ion.position[0]).powi(2)
                        + (q.y - ion.position[1]).powi(2)
                        + (q.z - ion.position[2]).powi(2))
                    .sqrt();
                    if d < COORD_FAR {
                        let offset = (d - model.distance) / COORD_WIDTH;
                        t.coordination -= (-0.5 * offset * offset).exp();
                    }
                }
            }
        }
        let [donor, acceptor, unsatisfied] = self.hydrogen_bond_parts(positions);
        t.hbond_donor = donor;
        t.hbond_acceptor = acceptor;
        t.hbond_unsatisfied = unsatisfied;
        t
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

    /// Directional hydrogen bonding and the unsatisfied-buried-polar penalty (see [`HydrogenBond`]).
    ///
    /// Per ligand hydroxyl: bonds it donates through its own O-H, bonds it accepts from receptor polar
    /// hydrogens along one of its lone pairs, and, if it is buried and neither happened, a penalty. All
    /// three weights sit in [`HydrogenBond`]; the geometry is fixed (see the constants).
    fn hydrogen_bond_term(&self, positions: &[Vec3], mut gradient: Option<&mut [Vec3]>) -> f64 {
        let Some(hb) = self.hbond.as_ref() else {
            return 0.0;
        };
        let mut total = 0.0;
        for group in &hb.hydroxyls {
            total += self.hydroxyl_term(hb, group, positions);
            // The term separates over hydroxyls, and each one moves with only its own three atoms, so its
            // gradient is three central differences of that hydroxyl alone: exact to the energy by
            // construction, and a handful of neighbour pairs each.
            if let Some(gradient) = gradient.as_deref_mut() {
                let mut moved = [
                    (group.oxygen, positions[group.oxygen]),
                    (group.hydrogen.unwrap_or(group.oxygen), positions[group.hydrogen.unwrap_or(group.oxygen)]),
                    (group.carbon.unwrap_or(group.oxygen), positions[group.carbon.unwrap_or(group.oxygen)]),
                ];
                moved.sort_unstable_by_key(|(i, _)| *i);
                let mut seen = [usize::MAX; 3];
                let mut scratch = positions.to_vec();
                for (slot, &(atom, _)) in moved.iter().enumerate() {
                    if seen.contains(&atom) {
                        continue;
                    }
                    seen[slot] = atom;
                    for axis in 0..3 {
                        let bump = |p: &mut Vec3, d: f64| match axis {
                            0 => p.x += d,
                            1 => p.y += d,
                            _ => p.z += d,
                        };
                        let h = 1.0e-5;
                        bump(&mut scratch[atom], h);
                        let up = self.hydroxyl_term(hb, group, &scratch);
                        bump(&mut scratch[atom], -2.0 * h);
                        let down = self.hydroxyl_term(hb, group, &scratch);
                        bump(&mut scratch[atom], h);
                        let slope = (up - down) / (2.0 * h);
                        match axis {
                            0 => gradient[atom].x += slope,
                            1 => gradient[atom].y += slope,
                            _ => gradient[atom].z += slope,
                        }
                    }
                }
            }
        }
        total
    }

    /// One hydroxyl's hydrogen bonds and its unsatisfied-burial penalty.
    fn hydroxyl_term(&self, hb: &HbReceptor, group: &HbHydroxyl, positions: &[Vec3]) -> f64 {
        let o = positions[group.oxygen];
        let op = [o.x, o.y, o.z];
        let mut total = 0.0;
        // how much partner is within reach of this hydroxyl, whichever way the O-H happens to point
        let mut available = 0.0;
        // bonds the hydroxyl donates: O-H ... A, linear at the hydrogen, into the acceptor's lone pairs.
        //
        // The hydrogen's own position is NOT trusted. A hydroxyl turns freely about its C-O bond, and the
        // builder places the hydrogen where it is farthest from its neighbours, which points it away from
        // the very acceptor it should bond to: over 122 deposited development complexes only 8.6% of the
        // ligand hydroxyls with an acceptor within 3.5 A come out with an O-H...A angle of 120 degrees or
        // more, while 98.4% could reach one by turning. Scoring the hydrogen as built therefore measures
        // the builder, not the pose, which is why a fitted donor depth collapses to zero. Instead the
        // term is the Boltzmann free energy over the orientations of the O-H about the C-O axis, the same
        // treatment the receptor's own flexible hydroxyls get. Smooth, so the central-difference gradient
        // above stays valid; a plain minimum over orientations would not be.
        if let Some(h_index) = group.hydrogen
            && hb.model.donor_depth != 0.0
        {
            let h = positions[h_index];
            let (oh, _) = unit([h.x - o.x, h.y - o.y, h.z - o.z]);
            // the acceptor geometry that does not depend on where the hydrogen points, gathered once
            let mut near: Vec<([f64; 3], f64)> = Vec::new();
            hb.cells.near(0, op, HB_CUTOFF, |k| {
                let a = hb.acceptors[k];
                let d = [a.position[0] - op[0], a.position[1] - op[1], a.position[2] - op[2]];
                let r = norm(d);
                if r >= HB_CUTOFF || r < 1.0e-6 {
                    return;
                }
                let u = [d[0] / r, d[1] / r, d[2] / r];
                let z = (r - HB_DONOR_DISTANCE.0) / HB_DONOR_DISTANCE.1;
                let f = (-0.5 * z * z).exp();
                let c_acc = if norm(a.lone_direction) < 0.5 {
                    1.0
                } else {
                    -dot(u, a.lone_direction)
                };
                if c_acc <= 0.0 {
                    return;
                }
                // whether a partner is there at all, which does not depend on the O-H orientation
                available += f * c_acc * c_acc;
                near.push((u, f * c_acc * c_acc));
            });
            if !near.is_empty() {
                let strength = |direction: [f64; 3]| -> f64 {
                    let mut sum = 0.0;
                    for &(u, weight) in &near {
                        let c_lin = dot(direction, u);
                        if c_lin > 0.0 {
                            sum += weight * c_lin * c_lin;
                        }
                    }
                    -hb.model.donor_depth * sum
                };
                // the cone the hydrogen sweeps about the C-O axis, from the geometry as built
                let cone = group.carbon.and_then(|c| {
                    let p = positions[c];
                    let (axis, axis_len) = unit([o.x - p.x, o.y - p.y, o.z - p.z]);
                    if axis_len < 1.0e-6 {
                        return None;
                    }
                    let along = dot(oh, axis);
                    let radial = [oh[0] - along * axis[0], oh[1] - along * axis[1], oh[2] - along * axis[2]];
                    let (e1, radius) = unit(radial);
                    if radius < 1.0e-6 {
                        return None;
                    }
                    let e2 = [
                        axis[1] * e1[2] - axis[2] * e1[1],
                        axis[2] * e1[0] - axis[0] * e1[2],
                        axis[0] * e1[1] - axis[1] * e1[0],
                    ];
                    Some((axis, along, e1, e2, radius))
                });
                total += match cone {
                    // no carbon, or the O-H lies on the axis: nothing to turn, score it as it stands
                    None => strength(oh),
                    Some((axis, along, e1, e2, radius)) => {
                        let mut energies = [0.0f64; HB_DONOR_ROTAMERS];
                        let mut lowest = f64::INFINITY;
                        for (k, slot) in energies.iter_mut().enumerate() {
                            let t = std::f64::consts::TAU * k as f64 / HB_DONOR_ROTAMERS as f64;
                            let (sn, cs) = t.sin_cos();
                            let direction = [
                                along * axis[0] + radius * (cs * e1[0] + sn * e2[0]),
                                along * axis[1] + radius * (cs * e1[1] + sn * e2[1]),
                                along * axis[2] + radius * (cs * e1[2] + sn * e2[2]),
                            ];
                            *slot = strength(unit(direction).0);
                            lowest = lowest.min(*slot);
                        }
                        let kt = self.temperature.max(1.0e-6);
                        let partition: f64 = energies.iter().map(|e| (-(e - lowest) / kt).exp()).sum();
                        // relative to a free hydroxyl, so an orientation-independent zero stays zero
                        lowest - kt * (partition / HB_DONOR_ROTAMERS as f64).ln()
                    }
                };
            }
        }
        // bonds the hydroxyl accepts: D-H ... O, linear at the receptor hydrogen, along a lone pair of the
        // hydroxyl (the C-O...H angle)
        if hb.model.acceptor_depth != 0.0 {
            let co = group.carbon.map(|c| {
                let p = positions[c];
                unit([o.x - p.x, o.y - p.y, o.z - p.z]).0
            });
            hb.cells.near(1, op, HB_CUTOFF, |k| {
                let donor = hb.donors[k];
                let d = [donor.hydrogen[0] - op[0], donor.hydrogen[1] - op[1], donor.hydrogen[2] - op[2]];
                let r = norm(d);
                if r >= HB_CUTOFF || r < 1.0e-6 {
                    return;
                }
                let u = [d[0] / r, d[1] / r, d[2] / r];
                let z = (r - HB_ACCEPTOR_DISTANCE.0) / HB_ACCEPTOR_DISTANCE.1;
                let f = (-0.5 * z * z).exp();
                let (dh, _) = unit([
                    donor.hydrogen[0] - donor.donor[0],
                    donor.hydrogen[1] - donor.donor[1],
                    donor.hydrogen[2] - donor.donor[2],
                ]);
                let c_lin = -dot(dh, u);
                if c_lin <= 0.0 {
                    return;
                }
                let c_lp = match co {
                    Some(co) => {
                        let c = dot(co, u);
                        let w = (c - HB_LONE_PAIR.0) / HB_LONE_PAIR.1;
                        (-0.5 * w * w).exp()
                    }
                    None => 1.0,
                };
                let shape = f * c_lin * c_lin * c_lp;
                available += shape;
                total -= hb.model.acceptor_depth * shape;
            });
        }
        // a buried hydroxyl that found nothing to bond to
        if hb.model.unsatisfied_depth != 0.0 {
            let mut burial = 0.0;
            hb.cells.near(2, op, HB_BURIAL_RANGE.1, |k| {
                let p = hb.heavy[k];
                let r = norm([p[0] - op[0], p[1] - op[1], p[2] - op[2]]);
                burial += switch_down(r, HB_BURIAL_RANGE.0, HB_BURIAL_RANGE.1);
            });
            let buried = (burial / HB_BURIAL_FULL).min(1.0);
            let missing = (1.0 - available).max(0.0);
            total += hb.model.unsatisfied_depth * buried * missing;
        }
        total
    }

    /// The three hydrogen-bond parts of a pose at unit depth, for fitting.
    fn hydrogen_bond_parts(&self, positions: &[Vec3]) -> [f64; 3] {
        let Some(hb) = self.hbond.as_ref() else {
            return [0.0; 3];
        };
        let unit_model = HbReceptor {
            model: HydrogenBond { donor_depth: 1.0, acceptor_depth: 0.0, unsatisfied_depth: 0.0 },
            ..hb.clone()
        };
        let donor: f64 = unit_model.hydroxyls.iter().map(|g| self.hydroxyl_term(&unit_model, g, positions)).sum();
        let acceptor_model = HbReceptor {
            model: HydrogenBond { donor_depth: 0.0, acceptor_depth: 1.0, unsatisfied_depth: 0.0 },
            ..hb.clone()
        };
        let acceptor: f64 = acceptor_model.hydroxyls.iter().map(|g| self.hydroxyl_term(&acceptor_model, g, positions)).sum();
        // the penalty needs the real satisfaction, so it is taken at the model's own bond depths
        let unsatisfied_model = HbReceptor {
            model: HydrogenBond { unsatisfied_depth: 1.0, ..hb.model },
            ..hb.clone()
        };
        let both: f64 = unsatisfied_model
            .hydroxyls
            .iter()
            .map(|g| self.hydroxyl_term(&unsatisfied_model, g, positions))
            .sum();
        let bonds: f64 = hb.hydroxyls.iter().map(|g| self.hydroxyl_term(hb, g, positions)).sum();
        [donor, acceptor, both - bonds]
    }

    /// Counts behind the hydrogen-bond term: receptor acceptors, receptor polar hydrogens, ligand
    /// hydroxyls, and of those the ones that carry a hydrogen and a carbon.
    pub fn hydrogen_bond_counts(&self) -> [usize; 5] {
        match self.hbond.as_ref() {
            None => [0; 5],
            Some(hb) => [
                hb.acceptors.len(),
                hb.donors.len(),
                hb.hydroxyls.len(),
                hb.hydroxyls.iter().filter(|g| g.hydrogen.is_some()).count(),
                hb.hydroxyls.iter().filter(|g| g.carbon.is_some()).count(),
            ],
        }
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
