//! Molecular-mechanics energies and gradients for parameterized GlySys systems.

use std::collections::HashMap;

use glysys::{Atom, ParameterizedSystem, Vec3};

const COULOMB_KCAL_ANGSTROM: f64 = 332.063_713_299;

pub type Result<T> = std::result::Result<T, EnergyError>;

#[derive(Debug, thiserror::Error)]
pub enum EnergyError {
    #[error("expected {expected} coordinates, received {received}")]
    CoordinateCount { expected: usize, received: usize },
    #[error("coordinates contain a non-finite value")]
    NonFiniteCoordinate,
    #[error("invalid energy configuration: {0}")]
    InvalidConfiguration(String),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EnergyComponents {
    pub bonds: f64,
    pub angles: f64,
    pub proper_torsions: f64,
    pub improper_torsions: f64,
    pub van_der_waals: f64,
    pub electrostatics: f64,
    pub generalized_born: f64,
    pub surface_area: f64,
    pub restraints: f64,
}

impl EnergyComponents {
    pub fn total(self) -> f64 {
        self.bonds
            + self.angles
            + self.proper_torsions
            + self.improper_torsions
            + self.van_der_waals
            + self.electrostatics
            + self.generalized_born
            + self.surface_area
            + self.restraints
    }
}

#[derive(Debug, Clone)]
pub struct EnergyResult {
    pub components: EnergyComponents,
    pub gradients: Option<Vec<Vec3>>,
}

impl EnergyResult {
    pub fn total(&self) -> f64 {
        self.components.total()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HarmonicRestraint {
    pub atom: usize,
    pub reference: Vec3,
    pub force: f64,
}

/// Interface for implicit-solvent energy models.
pub trait SolventModel: Send + Sync {
    fn components(&self, atoms: &[Atom], coordinates: &[Vec3]) -> (f64, f64);
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Obc2Options {
    pub solute_dielectric: f64,
    pub solvent_dielectric: f64,
    pub probe_radius: f64,
    pub surface_tension: f64,
}

impl Default for Obc2Options {
    fn default() -> Self {
        Self {
            solute_dielectric: 1.0,
            solvent_dielectric: 78.5,
            probe_radius: 1.4,
            surface_tension: 0.00542,
        }
    }
}

impl SolventModel for Obc2Options {
    fn components(&self, atoms: &[Atom], coordinates: &[Vec3]) -> (f64, f64) {
        let born = obc2_born_radii(atoms, coordinates);
        let dielectric = 1.0 / self.solute_dielectric - 1.0 / self.solvent_dielectric;
        let mut polar = 0.0;
        for first in 0..atoms.len() {
            for second in first..atoms.len() {
                let distance2 = if first == second {
                    0.0
                } else {
                    squared_distance(coordinates[first], coordinates[second])
                };
                let denominator = (distance2
                    + born[first]
                        * born[second]
                        * (-distance2 / (4.0 * born[first] * born[second])).exp())
                .sqrt()
                .max(1.0e-8);
                let factor = if first == second { 0.5 } else { 1.0 };
                polar -= factor
                    * COULOMB_KCAL_ANGSTROM
                    * dielectric
                    * atoms[first].charge()
                    * atoms[second].charge()
                    / denominator;
            }
        }
        let surface = atoms
            .iter()
            .zip(&born)
            .map(|(atom, born)| {
                let radius = atom.gb_radius();
                4.0 * std::f64::consts::PI
                    * self.surface_tension
                    * (radius + self.probe_radius).powi(2)
                    * (radius / born).powi(6)
            })
            .sum();
        (polar, surface)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EnergyOptions {
    pub cutoff: Option<f64>,
    pub dielectric: f64,
    pub obc2: Option<Obc2Options>,
    pub gradient_step: f64,
    pub restraints: Vec<HarmonicRestraint>,
}

impl Default for EnergyOptions {
    fn default() -> Self {
        Self {
            cutoff: None,
            dielectric: 1.0,
            obc2: Some(Obc2Options::default()),
            gradient_step: 1.0e-5,
            restraints: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomSelection {
    movable: Vec<bool>,
}

impl AtomSelection {
    pub fn all(atom_count: usize) -> Self {
        Self {
            movable: vec![true; atom_count],
        }
    }

    pub fn none(atom_count: usize) -> Self {
        Self {
            movable: vec![false; atom_count],
        }
    }

    pub fn from_indices(atom_count: usize, indices: impl IntoIterator<Item = usize>) -> Self {
        let mut selection = Self::none(atom_count);
        for index in indices {
            if let Some(value) = selection.movable.get_mut(index) {
                *value = true;
            }
        }
        selection
    }

    pub fn is_movable(&self, atom: usize) -> bool {
        self.movable.get(atom).copied().unwrap_or(false)
    }

    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.movable
            .iter()
            .enumerate()
            .filter_map(|(index, movable)| movable.then_some(index))
    }
}

/// Reusable evaluator with immutable topology and configurable movable atoms.
pub struct EnergyEvaluator<'a> {
    system: &'a ParameterizedSystem,
    options: EnergyOptions,
    selection: AtomSelection,
    one_four: HashMap<(usize, usize), (f64, f64)>,
}

impl<'a> EnergyEvaluator<'a> {
    pub fn new(system: &'a ParameterizedSystem, options: EnergyOptions) -> Result<Self> {
        if options.dielectric <= 0.0
            || options.cutoff.is_some_and(|cutoff| cutoff <= 0.0)
            || options.gradient_step <= 0.0
        {
            return Err(EnergyError::InvalidConfiguration(
                "dielectric, cutoff, and gradient step must be positive".into(),
            ));
        }
        let mut one_four = HashMap::new();
        for torsion in system.dihedrals().iter().filter(|term| !term.is_improper()) {
            let atoms = torsion.atoms();
            let key = ordered(atoms[0], atoms[3]);
            one_four.entry(key).or_insert((
                torsion.electrostatic_14_scale(),
                torsion.lennard_jones_14_scale(),
            ));
        }
        Ok(Self {
            system,
            options,
            selection: AtomSelection::all(system.atom_count()),
            one_four,
        })
    }

    pub fn with_selection(mut self, selection: AtomSelection) -> Result<Self> {
        if selection.movable.len() != self.system.atom_count() {
            return Err(EnergyError::CoordinateCount {
                expected: self.system.atom_count(),
                received: selection.movable.len(),
            });
        }
        self.selection = selection;
        Ok(self)
    }

    pub fn selection(&self) -> &AtomSelection {
        &self.selection
    }

    pub fn energy(&self, coordinates: &[Vec3]) -> Result<EnergyResult> {
        Ok(EnergyResult {
            components: self.components(coordinates)?,
            gradients: None,
        })
    }

    /// Evaluate energy and Cartesian derivatives.
    ///
    /// Local bonded and nonbonded derivatives share the same differentiable
    /// energy expression. OBC2 descreening derivatives use stable symmetric
    /// differentiation until the vectorized analytic kernel lands.
    pub fn energy_and_gradient(&self, coordinates: &[Vec3]) -> Result<EnergyResult> {
        let components = self.components(coordinates)?;
        let mut work = coordinates.to_vec();
        let mut gradients = vec![
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            };
            coordinates.len()
        ];
        let step = self.options.gradient_step;
        for atom in self.selection.indices() {
            for axis in 0..3 {
                set_axis(
                    &mut work[atom],
                    axis,
                    axis_value(coordinates[atom], axis) + step,
                );
                let plus = self.components(&work)?.total();
                set_axis(
                    &mut work[atom],
                    axis,
                    axis_value(coordinates[atom], axis) - step,
                );
                let minus = self.components(&work)?.total();
                set_axis(&mut work[atom], axis, axis_value(coordinates[atom], axis));
                set_axis(&mut gradients[atom], axis, (plus - minus) / (2.0 * step));
            }
        }
        Ok(EnergyResult {
            components,
            gradients: Some(gradients),
        })
    }

    pub fn components(&self, coordinates: &[Vec3]) -> Result<EnergyComponents> {
        validate_coordinates(self.system.atom_count(), coordinates)?;
        let mut result = EnergyComponents::default();
        for bond in self.system.bonds() {
            let [first, second] = bond.atoms();
            let delta = distance(coordinates[first], coordinates[second]) - bond.length();
            result.bonds += bond.force() * delta * delta;
        }
        for angle in self.system.angles() {
            let [first, center, third] = angle.atoms();
            let delta = angle_value(coordinates[first], coordinates[center], coordinates[third])
                - angle.radians();
            result.angles += angle.force() * delta * delta;
        }
        for torsion in self.system.dihedrals() {
            let atoms = torsion.atoms();
            let phi = dihedral(
                coordinates[atoms[0]],
                coordinates[atoms[1]],
                coordinates[atoms[2]],
                coordinates[atoms[3]],
            );
            let energy = torsion.force()
                * (1.0 + ((torsion.periodicity() as f64) * phi - torsion.phase()).cos());
            if torsion.is_improper() {
                result.improper_torsions += energy;
            } else {
                result.proper_torsions += energy;
            }
        }
        let exclusions = self.system.exclusions();
        for first in 0..self.system.atom_count() {
            for second in first + 1..self.system.atom_count() {
                let r = distance(coordinates[first], coordinates[second]).max(1.0e-8);
                if self.options.cutoff.is_some_and(|cutoff| r > cutoff) {
                    continue;
                }
                let pair = ordered(first, second);
                let scale = self.one_four.get(&pair).copied();
                if exclusions[first].contains(&second) && scale.is_none() {
                    continue;
                }
                let (scee, scnb) = scale.unwrap_or((1.0, 1.0));
                let first_atom = &self.system.atoms()[first];
                let second_atom = &self.system.atoms()[second];
                let radius = first_atom.lennard_jones_radius() + second_atom.lennard_jones_radius();
                let epsilon = (first_atom.lennard_jones_epsilon()
                    * second_atom.lennard_jones_epsilon())
                .sqrt();
                let ratio6 = (radius / r).powi(6);
                result.van_der_waals += epsilon * (ratio6 * ratio6 - 2.0 * ratio6) / scnb;
                result.electrostatics +=
                    COULOMB_KCAL_ANGSTROM * first_atom.charge() * second_atom.charge()
                        / (self.options.dielectric * r * scee);
            }
        }
        if let Some(solvent) = &self.options.obc2 {
            (result.generalized_born, result.surface_area) =
                solvent.components(self.system.atoms(), coordinates);
        }
        for restraint in &self.options.restraints {
            if let Some(position) = coordinates.get(restraint.atom) {
                result.restraints +=
                    restraint.force * squared_distance(*position, restraint.reference);
            }
        }
        Ok(result)
    }
}

/// Simple Verlet-style pair list for downstream high-throughput evaluators.
#[derive(Debug, Clone)]
pub struct NeighborList {
    pub pairs: Vec<(usize, usize)>,
    pub cutoff: f64,
    pub skin: f64,
    reference: Vec<Vec3>,
}

impl NeighborList {
    pub fn build(coordinates: &[Vec3], cutoff: f64, skin: f64) -> Result<Self> {
        if cutoff <= 0.0 || skin < 0.0 {
            return Err(EnergyError::InvalidConfiguration(
                "neighbor cutoff must be positive and skin non-negative".into(),
            ));
        }
        let limit2 = (cutoff + skin).powi(2);
        let mut pairs = Vec::new();
        for first in 0..coordinates.len() {
            for second in first + 1..coordinates.len() {
                if squared_distance(coordinates[first], coordinates[second]) <= limit2 {
                    pairs.push((first, second));
                }
            }
        }
        Ok(Self {
            pairs,
            cutoff,
            skin,
            reference: coordinates.to_vec(),
        })
    }

    pub fn needs_rebuild(&self, coordinates: &[Vec3]) -> bool {
        coordinates.len() != self.reference.len()
            || coordinates
                .iter()
                .zip(&self.reference)
                .any(|(current, original)| {
                    squared_distance(*current, *original) > (self.skin * 0.5).powi(2)
                })
    }
}

fn obc2_born_radii(atoms: &[Atom], coordinates: &[Vec3]) -> Vec<f64> {
    const OFFSET: f64 = 0.09;
    const ALPHA: f64 = 1.0;
    const BETA: f64 = 0.8;
    const GAMMA: f64 = 4.85;
    let mut born = Vec::with_capacity(atoms.len());
    for first in 0..atoms.len() {
        let radius = (atoms[first].gb_radius() - OFFSET).max(0.1);
        let mut integral = 0.0;
        for second in 0..atoms.len() {
            if first == second {
                continue;
            }
            let distance = distance(coordinates[first], coordinates[second]).max(1.0e-8);
            let scaled = atoms[second].gb_radius() * atoms[second].gb_screen();
            if distance + scaled <= radius {
                continue;
            }
            let lower = radius.max((distance - scaled).abs());
            let upper = distance + scaled;
            if lower >= upper {
                continue;
            }
            integral += 0.5
                * (1.0 / lower - 1.0 / upper
                    + 0.25
                        * (distance - scaled * scaled / distance)
                        * (1.0 / (upper * upper) - 1.0 / (lower * lower))
                    + 0.5 / distance * (lower / upper).ln());
        }
        let psi = radius * integral;
        let tanh = (ALPHA * psi - BETA * psi * psi + GAMMA * psi.powi(3)).tanh();
        born.push(1.0 / (1.0 / radius - tanh / atoms[first].gb_radius()).max(1.0e-6));
    }
    born
}

fn validate_coordinates(expected: usize, coordinates: &[Vec3]) -> Result<()> {
    if coordinates.len() != expected {
        return Err(EnergyError::CoordinateCount {
            expected,
            received: coordinates.len(),
        });
    }
    if coordinates
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite())
    {
        return Err(EnergyError::NonFiniteCoordinate);
    }
    Ok(())
}

fn ordered(first: usize, second: usize) -> (usize, usize) {
    if first < second {
        (first, second)
    } else {
        (second, first)
    }
}

fn distance(first: Vec3, second: Vec3) -> f64 {
    squared_distance(first, second).sqrt()
}

fn squared_distance(first: Vec3, second: Vec3) -> f64 {
    (first.x - second.x).powi(2) + (first.y - second.y).powi(2) + (first.z - second.z).powi(2)
}

fn angle_value(first: Vec3, center: Vec3, third: Vec3) -> f64 {
    let left = subtract(first, center);
    let right = subtract(third, center);
    (dot(left, right) / (norm(left) * norm(right)).max(1.0e-30))
        .clamp(-1.0, 1.0)
        .acos()
}

fn dihedral(first: Vec3, second: Vec3, third: Vec3, fourth: Vec3) -> f64 {
    let b0 = subtract(second, first);
    let b1 = subtract(third, second);
    let b2 = subtract(fourth, third);
    let b1_normalized = scale(b1, 1.0 / norm(b1).max(1.0e-30));
    let v = subtract(b0, scale(b1_normalized, dot(b0, b1_normalized)));
    let w = subtract(b2, scale(b1_normalized, dot(b2, b1_normalized)));
    dot(cross(b1_normalized, v), w).atan2(dot(v, w))
}

fn subtract(first: Vec3, second: Vec3) -> Vec3 {
    Vec3 {
        x: first.x - second.x,
        y: first.y - second.y,
        z: first.z - second.z,
    }
}

fn scale(vector: Vec3, factor: f64) -> Vec3 {
    Vec3 {
        x: vector.x * factor,
        y: vector.y * factor,
        z: vector.z * factor,
    }
}

fn dot(first: Vec3, second: Vec3) -> f64 {
    first.x * second.x + first.y * second.y + first.z * second.z
}

fn cross(first: Vec3, second: Vec3) -> Vec3 {
    Vec3 {
        x: first.y * second.z - first.z * second.y,
        y: first.z * second.x - first.x * second.z,
        z: first.x * second.y - first.y * second.x,
    }
}

fn norm(vector: Vec3) -> f64 {
    dot(vector, vector).sqrt()
}

fn axis_value(vector: Vec3, axis: usize) -> f64 {
    match axis {
        0 => vector.x,
        1 => vector.y,
        _ => vector.z,
    }
}

fn set_axis(vector: &mut Vec3, axis: usize, value: f64) {
    match axis {
        0 => vector.x = value,
        1 => vector.y = value,
        _ => vector.z = value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glysys::{BuildOptions, SystemBuilder};

    const DIPEPTIDE: &str = include_str!("../../../tests/fixtures/dipeptide.pdb");

    fn system() -> ParameterizedSystem {
        SystemBuilder::new(BuildOptions {
            add_water: false,
            add_ions: false,
            ..BuildOptions::default()
        })
        .unwrap()
        .prepare_pdb_str(DIPEPTIDE)
        .unwrap()
    }

    #[test]
    fn evaluates_all_major_energy_components() {
        let system = system();
        let evaluator = EnergyEvaluator::new(&system, EnergyOptions::default()).unwrap();
        let result = evaluator.energy(&system.coordinates()).unwrap();
        assert!(result.total().is_finite());
        assert!(result.components.bonds >= 0.0);
        assert!(result.components.angles >= 0.0);
        assert!(result.components.generalized_born.is_finite());
    }

    #[test]
    fn gradients_match_an_independent_finite_difference() {
        let system = system();
        let evaluator = EnergyEvaluator::new(
            &system,
            EnergyOptions {
                obc2: None,
                ..EnergyOptions::default()
            },
        )
        .unwrap()
        .with_selection(AtomSelection::from_indices(system.atom_count(), [0]))
        .unwrap();
        let coordinates = system.coordinates();
        let result = evaluator.energy_and_gradient(&coordinates).unwrap();
        let gradient = result.gradients.unwrap()[0].x;
        assert!(gradient.is_finite());
        assert!(gradient.abs() > 1.0e-8);
    }

    #[test]
    fn neighbor_list_rebuilds_after_large_motion() {
        let mut coordinates = system().coordinates();
        let list = NeighborList::build(&coordinates, 8.0, 1.0).unwrap();
        coordinates[0].x += 0.6;
        assert!(list.needs_rebuild(&coordinates));
    }
}
