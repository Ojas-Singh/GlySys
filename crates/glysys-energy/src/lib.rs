//! Molecular-mechanics energies and gradients for parameterized GlySys systems.

use std::collections::{BTreeMap, HashMap};

use glysys::{Atom, ParameterizedSystem, Vec3};
use pulp::{Arch, Simd, WithSimd};

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

/// The energy quantity used by a structure-selection workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyScoringMode {
    /// All bonded, nonbonded, restraint, and optional implicit-solvent terms.
    Full,
    /// Protein--glycan cross Lennard-Jones and Coulomb terms only.
    ProteinGlycanInteraction,
}

/// Cross nonbonded terms between two disjoint atom groups, in kcal/mol.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InteractionEnergyComponents {
    pub van_der_waals: f64,
    pub electrostatics: f64,
}

impl InteractionEnergyComponents {
    pub fn total(self) -> f64 {
        self.van_der_waals + self.electrostatics
    }
}

/// A reusable atom-group mask for energy decomposition and fixed/movable
/// selections. Masks are deliberately topology-bound and cannot be resized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomGroupMask {
    members: Vec<bool>,
}

impl AtomGroupMask {
    pub fn none(atom_count: usize) -> Self {
        Self {
            members: vec![false; atom_count],
        }
    }

    pub fn from_indices(atom_count: usize, indices: impl IntoIterator<Item = usize>) -> Self {
        let mut result = Self::none(atom_count);
        for index in indices {
            if let Some(member) = result.members.get_mut(index) {
                *member = true;
            }
        }
        result
    }

    pub fn contains(&self, atom: usize) -> bool {
        self.members.get(atom).copied().unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        !self.members.iter().any(|member| *member)
    }

    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.members
            .iter()
            .enumerate()
            .filter_map(|(index, member)| member.then_some(index))
    }
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
    active_terms_only: bool,
}

struct InteractionKernel<'a> {
    radii2: &'a [f64],
    sigmas: &'a [f64],
    epsilons: &'a [f64],
    charges: &'a [f64],
}

impl WithSimd for InteractionKernel<'_> {
    type Output = InteractionEnergyComponents;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (radii, radii_tail) = S::as_simd_f64s(self.radii2);
        let (sigmas, sigmas_tail) = S::as_simd_f64s(self.sigmas);
        let (epsilons, epsilons_tail) = S::as_simd_f64s(self.epsilons);
        let (charges, charges_tail) = S::as_simd_f64s(self.charges);
        let mut vdw = simd.splat_f64s(0.0);
        let mut coulomb = simd.splat_f64s(0.0);
        let two = simd.splat_f64s(2.0);
        for (((radius2, sigma), epsilon), charge) in
            radii.iter().zip(sigmas).zip(epsilons).zip(charges)
        {
            let radius = simd.sqrt_f64s(*radius2);
            let ratio = simd.div_f64s(*sigma, radius);
            let ratio2 = simd.mul_f64s(ratio, ratio);
            let ratio6 = simd.mul_f64s(simd.mul_f64s(ratio2, ratio2), ratio2);
            let shape = simd.sub_f64s(simd.mul_f64s(ratio6, ratio6), simd.mul_f64s(two, ratio6));
            vdw = simd.add_f64s(vdw, simd.mul_f64s(*epsilon, shape));
            coulomb = simd.add_f64s(coulomb, simd.div_f64s(*charge, radius));
        }
        let mut result = InteractionEnergyComponents {
            van_der_waals: simd.reduce_sum_f64s(vdw),
            electrostatics: simd.reduce_sum_f64s(coulomb),
        };
        for (((radius2, sigma), epsilon), charge) in radii_tail
            .iter()
            .zip(sigmas_tail)
            .zip(epsilons_tail)
            .zip(charges_tail)
        {
            let radius = radius2.sqrt();
            let ratio6 = (sigma / radius).powi(6);
            result.van_der_waals += epsilon * (ratio6 * ratio6 - 2.0 * ratio6);
            result.electrostatics += charge / radius;
        }
        result
    }
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
            active_terms_only: false,
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

    /// Restrict both values and gradients to terms touching selected atoms.
    /// Fixed-only contributions are constant during local minimization and
    /// can be omitted without changing its trajectory.
    pub fn with_active_terms(mut self, selection: AtomSelection) -> Result<Self> {
        self = self.with_selection(selection)?;
        self.active_terms_only = true;
        Ok(self)
    }

    pub fn selection(&self) -> &AtomSelection {
        &self.selection
    }

    fn nonbonded_pairs(&self, coordinates: &[Vec3]) -> Vec<(usize, usize)> {
        if self.active_terms_only {
            selected_nonbonded_pairs(coordinates, self.options.cutoff, &self.selection)
        } else {
            nonbonded_pairs(coordinates, self.options.cutoff)
        }
    }

    /// Calculate only cross nonbonded interactions between two disjoint atom
    /// groups. This mirrors the Cookbook interaction objective: bonded,
    /// internal, restraint, and implicit-solvent terms are intentionally not
    /// part of the result.
    pub fn interaction_energy(
        &self,
        coordinates: &[Vec3],
        first_group: &AtomGroupMask,
        second_group: &AtomGroupMask,
    ) -> Result<InteractionEnergyComponents> {
        validate_coordinates(self.system.atom_count(), coordinates)?;
        if first_group.len() != self.system.atom_count()
            || second_group.len() != self.system.atom_count()
            || first_group
                .indices()
                .any(|index| second_group.contains(index))
        {
            return Err(EnergyError::InvalidConfiguration(
                "interaction groups must be disjoint masks matching the system atom count".into(),
            ));
        }
        let exclusions = self.system.exclusions();
        let mut radii2 = Vec::new();
        let mut sigmas = Vec::new();
        let mut epsilons = Vec::new();
        let mut charges = Vec::new();
        for (first, second) in
            cross_nonbonded_pairs(coordinates, self.options.cutoff, first_group, second_group)
        {
            let pair = ordered(first, second);
            let scale = self.one_four.get(&pair).copied();
            if exclusions[first].contains(&second) && scale.is_none() {
                continue;
            }
            let (scee, scnb) = scale.unwrap_or((1.0, 1.0));
            let first_atom = &self.system.atoms()[first];
            let second_atom = &self.system.atoms()[second];
            let sigma = first_atom.lennard_jones_radius() + second_atom.lennard_jones_radius();
            let epsilon =
                (first_atom.lennard_jones_epsilon() * second_atom.lennard_jones_epsilon()).sqrt();
            radii2.push(squared_distance(coordinates[first], coordinates[second]).max(1.0e-16));
            sigmas.push(sigma);
            epsilons.push(epsilon / scnb);
            charges.push(
                COULOMB_KCAL_ANGSTROM * first_atom.charge() * second_atom.charge()
                    / (self.options.dielectric * scee),
            );
        }
        Ok(Arch::new().dispatch(InteractionKernel {
            radii2: &radii2,
            sigmas: &sigmas,
            epsilons: &epsilons,
            charges: &charges,
        }))
    }

    pub fn energy(&self, coordinates: &[Vec3]) -> Result<EnergyResult> {
        Ok(EnergyResult {
            components: self.components(coordinates)?,
            gradients: None,
        })
    }

    /// Evaluate energy and Cartesian derivatives.
    ///
    /// Bonds, angles, nonbonded interactions, and restraints use closed-form
    /// Cartesian derivatives. Periodic torsions and OBC2 descreening use
    /// forward automatic differentiation.
    pub fn energy_and_gradient(&self, coordinates: &[Vec3]) -> Result<EnergyResult> {
        let components = self.components(coordinates)?;
        let mut gradients = self.analytic_gradient(coordinates)?;
        for (gradient, residual) in gradients
            .iter_mut()
            .zip(self.residual_gradient(coordinates))
        {
            gradient.x += residual.x;
            gradient.y += residual.y;
            gradient.z += residual.z;
        }
        Ok(EnergyResult {
            components,
            gradients: Some(gradients),
        })
    }

    fn analytic_gradient(&self, coordinates: &[Vec3]) -> Result<Vec<Vec3>> {
        validate_coordinates(self.system.atom_count(), coordinates)?;
        let mut gradient = vec![
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            };
            coordinates.len()
        ];
        for bond in self.system.bonds() {
            let [first, second] = bond.atoms();
            if self.active_terms_only
                && !self.selection.is_movable(first)
                && !self.selection.is_movable(second)
            {
                continue;
            }
            let vector = subtract(coordinates[first], coordinates[second]);
            let radius = norm(vector).max(1.0e-12);
            let derivative = 2.0 * bond.force() * (radius - bond.length()) / radius;
            add_scaled(&mut gradient[first], vector, derivative);
            add_scaled(&mut gradient[second], vector, -derivative);
        }
        for angle in self.system.angles() {
            let [first, center, third] = angle.atoms();
            if self.active_terms_only
                && ![first, center, third]
                    .into_iter()
                    .any(|atom| self.selection.is_movable(atom))
            {
                continue;
            }
            let left = subtract(coordinates[first], coordinates[center]);
            let right = subtract(coordinates[third], coordinates[center]);
            let left_norm = norm(left).max(1.0e-12);
            let right_norm = norm(right).max(1.0e-12);
            let cosine = (dot(left, right) / (left_norm * right_norm)).clamp(-1.0, 1.0);
            let theta = cosine.acos();
            let sine = (1.0 - cosine * cosine).sqrt().max(1.0e-12);
            let factor = 2.0 * angle.force() * (theta - angle.radians()) / sine;
            let first_derivative = subtract(
                scale(left, cosine / (left_norm * left_norm)),
                scale(right, 1.0 / (left_norm * right_norm)),
            );
            let third_derivative = subtract(
                scale(right, cosine / (right_norm * right_norm)),
                scale(left, 1.0 / (left_norm * right_norm)),
            );
            add_scaled(&mut gradient[first], first_derivative, factor);
            add_scaled(&mut gradient[third], third_derivative, factor);
            add_scaled(&mut gradient[center], first_derivative, -factor);
            add_scaled(&mut gradient[center], third_derivative, -factor);
        }
        let exclusions = self.system.exclusions();
        for (first, second) in self.nonbonded_pairs(coordinates) {
            let vector = subtract(coordinates[first], coordinates[second]);
            let radius = norm(vector).max(1.0e-8);
            let pair = ordered(first, second);
            let scale_14 = self.one_four.get(&pair).copied();
            if exclusions[first].contains(&second) && scale_14.is_none() {
                continue;
            }
            let (scee, scnb) = scale_14.unwrap_or((1.0, 1.0));
            let first_atom = &self.system.atoms()[first];
            let second_atom = &self.system.atoms()[second];
            let sigma = first_atom.lennard_jones_radius() + second_atom.lennard_jones_radius();
            let epsilon =
                (first_atom.lennard_jones_epsilon() * second_atom.lennard_jones_epsilon()).sqrt();
            let ratio6 = (sigma / radius).powi(6);
            let coulomb = COULOMB_KCAL_ANGSTROM * first_atom.charge() * second_atom.charge()
                / (self.options.dielectric * scee * radius);
            let derivative =
                12.0 * epsilon * (ratio6 - ratio6 * ratio6) / (scnb * radius) - coulomb / radius;
            add_scaled(&mut gradient[first], vector, derivative / radius);
            add_scaled(&mut gradient[second], vector, -derivative / radius);
        }
        for restraint in &self.options.restraints {
            if let Some(position) = coordinates.get(restraint.atom) {
                let vector = subtract(*position, restraint.reference);
                add_scaled(&mut gradient[restraint.atom], vector, 2.0 * restraint.force);
            }
        }
        for (index, value) in gradient.iter_mut().enumerate() {
            if !self.selection.is_movable(index) {
                *value = Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                };
            }
        }
        Ok(gradient)
    }

    fn residual_gradient(&self, coordinates: &[Vec3]) -> Vec<Vec3> {
        let selected = self.selection.indices().collect::<Vec<_>>();
        let mut gradients = vec![
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            };
            coordinates.len()
        ];
        // Torsions are local four-atom terms. Differentiate each in its own
        // fixed 12-dimensional space instead of allocating 3*N derivatives
        // for every operation in every torsion.
        for torsion in self.system.dihedrals() {
            let atoms = torsion.atoms();
            if self.active_terms_only
                && !atoms
                    .into_iter()
                    .any(|atom| self.selection.is_movable(atom))
            {
                continue;
            }
            let points = atoms.map(|atom| {
                let point = coordinates[atom];
                let local = atoms
                    .iter()
                    .position(|candidate| *candidate == atom)
                    .unwrap();
                DualVec3 {
                    x: Dual::coordinate(point.x, 12, Some(local * 3)),
                    y: Dual::coordinate(point.y, 12, Some(local * 3 + 1)),
                    z: Dual::coordinate(point.z, 12, Some(local * 3 + 2)),
                }
            });
            let phi = dual_dihedral(&points[0], &points[1], &points[2], &points[3]);
            let argument = phi
                .scale(torsion.periodicity() as f64)
                .add_constant(-torsion.phase());
            let term = argument.cos().add_constant(1.0).scale(torsion.force());
            for (local, atom) in atoms.iter().enumerate() {
                if self.selection.is_movable(*atom) {
                    gradients[*atom].x += term.gradient[local * 3];
                    gradients[*atom].y += term.gradient[local * 3 + 1];
                    gradients[*atom].z += term.gradient[local * 3 + 2];
                }
            }
        }
        if let Some(options) = &self.options.obc2 {
            let dimension = selected.len() * 3;
            let offsets = selected
                .iter()
                .enumerate()
                .map(|(offset, atom)| (*atom, offset * 3))
                .collect::<HashMap<_, _>>();
            let points = coordinates
                .iter()
                .enumerate()
                .map(|(atom, point)| {
                    let offset = offsets.get(&atom).copied();
                    DualVec3 {
                        x: Dual::coordinate(point.x, dimension, offset),
                        y: Dual::coordinate(point.y, dimension, offset.map(|value| value + 1)),
                        z: Dual::coordinate(point.z, dimension, offset.map(|value| value + 2)),
                    }
                })
                .collect::<Vec<_>>();
            let total = dual_obc2(self.system.atoms(), &points, options, dimension);
            for (offset, atom) in selected.iter().enumerate() {
                gradients[*atom].x += total.gradient[offset * 3];
                gradients[*atom].y += total.gradient[offset * 3 + 1];
                gradients[*atom].z += total.gradient[offset * 3 + 2];
            }
        }
        gradients
    }

    pub fn components(&self, coordinates: &[Vec3]) -> Result<EnergyComponents> {
        validate_coordinates(self.system.atom_count(), coordinates)?;
        let mut result = EnergyComponents::default();
        for bond in self.system.bonds() {
            let [first, second] = bond.atoms();
            if self.active_terms_only
                && !self.selection.is_movable(first)
                && !self.selection.is_movable(second)
            {
                continue;
            }
            let delta = distance(coordinates[first], coordinates[second]) - bond.length();
            result.bonds += bond.force() * delta * delta;
        }
        for angle in self.system.angles() {
            let [first, center, third] = angle.atoms();
            if self.active_terms_only
                && ![first, center, third]
                    .into_iter()
                    .any(|atom| self.selection.is_movable(atom))
            {
                continue;
            }
            let delta = angle_value(coordinates[first], coordinates[center], coordinates[third])
                - angle.radians();
            result.angles += angle.force() * delta * delta;
        }
        for torsion in self.system.dihedrals() {
            let atoms = torsion.atoms();
            if self.active_terms_only
                && !atoms
                    .into_iter()
                    .any(|atom| self.selection.is_movable(atom))
            {
                continue;
            }
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
        for (first, second) in self.nonbonded_pairs(coordinates) {
            let r = distance(coordinates[first], coordinates[second]).max(1.0e-8);
            let pair = ordered(first, second);
            let scale = self.one_four.get(&pair).copied();
            if exclusions[first].contains(&second) && scale.is_none() {
                continue;
            }
            let (scee, scnb) = scale.unwrap_or((1.0, 1.0));
            let first_atom = &self.system.atoms()[first];
            let second_atom = &self.system.atoms()[second];
            let radius = first_atom.lennard_jones_radius() + second_atom.lennard_jones_radius();
            let epsilon =
                (first_atom.lennard_jones_epsilon() * second_atom.lennard_jones_epsilon()).sqrt();
            let ratio6 = (radius / r).powi(6);
            result.van_der_waals += epsilon * (ratio6 * ratio6 - 2.0 * ratio6) / scnb;
            result.electrostatics +=
                COULOMB_KCAL_ANGSTROM * first_atom.charge() * second_atom.charge()
                    / (self.options.dielectric * r * scee);
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

#[derive(Clone)]
struct Dual {
    value: f64,
    gradient: Vec<f64>,
}

impl Dual {
    fn constant(value: f64, dimension: usize) -> Self {
        Self {
            value,
            gradient: vec![0.0; dimension],
        }
    }

    fn coordinate(value: f64, dimension: usize, offset: Option<usize>) -> Self {
        let mut result = Self::constant(value, dimension);
        if let Some(offset) = offset {
            result.gradient[offset] = 1.0;
        }
        result
    }

    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            gradient: self
                .gradient
                .into_iter()
                .zip(other.gradient)
                .map(|(first, second)| first + second)
                .collect(),
        }
    }

    fn sub(self, other: Self) -> Self {
        self.add(other.scale(-1.0))
    }

    fn add_constant(mut self, value: f64) -> Self {
        self.value += value;
        self
    }

    fn scale(mut self, factor: f64) -> Self {
        self.value *= factor;
        for value in &mut self.gradient {
            *value *= factor;
        }
        self
    }

    fn mul(self, other: Self) -> Self {
        let first_value = self.value;
        let second_value = other.value;
        Self {
            value: first_value * second_value,
            gradient: self
                .gradient
                .into_iter()
                .zip(other.gradient)
                .map(|(first, second)| first * second_value + second * first_value)
                .collect(),
        }
    }

    fn reciprocal(self) -> Self {
        let value = 1.0 / self.value;
        let factor = -value * value;
        Self {
            value,
            gradient: self
                .gradient
                .into_iter()
                .map(|gradient| gradient * factor)
                .collect(),
        }
    }

    fn div(self, other: Self) -> Self {
        self.mul(other.reciprocal())
    }

    fn sqrt(self) -> Self {
        let value = self.value.sqrt();
        let factor = 0.5 / value.max(1.0e-30);
        Self {
            value,
            gradient: self
                .gradient
                .into_iter()
                .map(|gradient| gradient * factor)
                .collect(),
        }
    }

    fn exp(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            gradient: self
                .gradient
                .into_iter()
                .map(|gradient| gradient * value)
                .collect(),
        }
    }

    fn ln(self) -> Self {
        let value = self.value.ln();
        let factor = 1.0 / self.value;
        Self {
            value,
            gradient: self
                .gradient
                .into_iter()
                .map(|gradient| gradient * factor)
                .collect(),
        }
    }

    fn tanh(self) -> Self {
        let value = self.value.tanh();
        let factor = 1.0 - value * value;
        Self {
            value,
            gradient: self
                .gradient
                .into_iter()
                .map(|gradient| gradient * factor)
                .collect(),
        }
    }

    fn cos(self) -> Self {
        let factor = -self.value.sin();
        Self {
            value: self.value.cos(),
            gradient: self
                .gradient
                .into_iter()
                .map(|gradient| gradient * factor)
                .collect(),
        }
    }

    fn atan2(self, x: Self) -> Self {
        let denominator = self.value * self.value + x.value * x.value;
        let y_value = self.value;
        let x_value = x.value;
        Self {
            value: y_value.atan2(x_value),
            gradient: self
                .gradient
                .into_iter()
                .zip(x.gradient)
                .map(|(dy, dx)| (x_value * dy - y_value * dx) / denominator.max(1.0e-30))
                .collect(),
        }
    }

    fn powi(self, exponent: usize) -> Self {
        if exponent == 0 {
            return Self::constant(1.0, self.gradient.len());
        }
        let value = self.value.powi(exponent as i32);
        let factor = exponent as f64 * self.value.powi(exponent as i32 - 1);
        Self {
            value,
            gradient: self
                .gradient
                .into_iter()
                .map(|gradient| gradient * factor)
                .collect(),
        }
    }

    fn abs(self) -> Self {
        if self.value < 0.0 {
            self.scale(-1.0)
        } else {
            self
        }
    }

    fn floor(self, minimum: f64) -> Self {
        if self.value < minimum {
            Self::constant(minimum, self.gradient.len())
        } else {
            self
        }
    }
}

#[derive(Clone)]
struct DualVec3 {
    x: Dual,
    y: Dual,
    z: Dual,
}

fn dual_subtract(first: &DualVec3, second: &DualVec3) -> DualVec3 {
    DualVec3 {
        x: first.x.clone().sub(second.x.clone()),
        y: first.y.clone().sub(second.y.clone()),
        z: first.z.clone().sub(second.z.clone()),
    }
}

fn dual_scale(vector: &DualVec3, factor: Dual) -> DualVec3 {
    DualVec3 {
        x: vector.x.clone().mul(factor.clone()),
        y: vector.y.clone().mul(factor.clone()),
        z: vector.z.clone().mul(factor),
    }
}

fn dual_dot(first: &DualVec3, second: &DualVec3) -> Dual {
    first
        .x
        .clone()
        .mul(second.x.clone())
        .add(first.y.clone().mul(second.y.clone()))
        .add(first.z.clone().mul(second.z.clone()))
}

fn dual_cross(first: &DualVec3, second: &DualVec3) -> DualVec3 {
    DualVec3 {
        x: first
            .y
            .clone()
            .mul(second.z.clone())
            .sub(first.z.clone().mul(second.y.clone())),
        y: first
            .z
            .clone()
            .mul(second.x.clone())
            .sub(first.x.clone().mul(second.z.clone())),
        z: first
            .x
            .clone()
            .mul(second.y.clone())
            .sub(first.y.clone().mul(second.x.clone())),
    }
}

fn dual_dihedral(first: &DualVec3, second: &DualVec3, third: &DualVec3, fourth: &DualVec3) -> Dual {
    let b0 = dual_subtract(second, first);
    let b1 = dual_subtract(third, second);
    let b2 = dual_subtract(fourth, third);
    let inverse_norm = dual_dot(&b1, &b1).sqrt().floor(1.0e-30).reciprocal();
    let normalized = dual_scale(&b1, inverse_norm);
    let v = dual_subtract(&b0, &dual_scale(&normalized, dual_dot(&b0, &normalized)));
    let w = dual_subtract(&b2, &dual_scale(&normalized, dual_dot(&b2, &normalized)));
    dual_dot(&dual_cross(&normalized, &v), &w).atan2(dual_dot(&v, &w))
}

fn dual_squared_distance(first: &DualVec3, second: &DualVec3) -> Dual {
    let difference = dual_subtract(first, second);
    dual_dot(&difference, &difference)
}

fn dual_obc2(
    atoms: &[Atom],
    coordinates: &[DualVec3],
    options: &Obc2Options,
    dimension: usize,
) -> Dual {
    const OFFSET: f64 = 0.09;
    const ALPHA: f64 = 1.0;
    const BETA: f64 = 0.8;
    const GAMMA: f64 = 4.85;
    let mut born = Vec::with_capacity(atoms.len());
    for first in 0..atoms.len() {
        let radius = (atoms[first].gb_radius() - OFFSET).max(0.1);
        let mut integral = Dual::constant(0.0, dimension);
        for second in 0..atoms.len() {
            if first == second {
                continue;
            }
            let distance = dual_squared_distance(&coordinates[first], &coordinates[second])
                .sqrt()
                .floor(1.0e-8);
            let scaled = atoms[second].gb_radius() * atoms[second].gb_screen();
            if distance.value + scaled <= radius {
                continue;
            }
            let candidate = distance.clone().add_constant(-scaled).abs();
            let lower = if candidate.value < radius {
                Dual::constant(radius, dimension)
            } else {
                candidate
            };
            let upper = distance.clone().add_constant(scaled);
            if lower.value >= upper.value {
                continue;
            }
            let inverse_lower = lower.clone().reciprocal();
            let inverse_upper = upper.clone().reciprocal();
            let distance_term = distance
                .clone()
                .sub(Dual::constant(scaled * scaled, dimension).div(distance.clone()));
            let inverse_square_delta = inverse_upper
                .clone()
                .powi(2)
                .sub(inverse_lower.clone().powi(2));
            let logarithm = lower.clone().div(upper).ln();
            let term = inverse_lower
                .sub(inverse_upper)
                .add(distance_term.mul(inverse_square_delta).scale(0.25))
                .add(logarithm.div(distance).scale(0.5))
                .scale(0.5);
            integral = integral.add(term);
        }
        let psi = integral.scale(radius);
        let transformed = psi
            .clone()
            .scale(ALPHA)
            .sub(psi.clone().powi(2).scale(BETA))
            .add(psi.powi(3).scale(GAMMA))
            .tanh();
        let denominator = Dual::constant(1.0 / radius, dimension)
            .sub(transformed.scale(1.0 / atoms[first].gb_radius()))
            .floor(1.0e-6);
        born.push(denominator.reciprocal());
    }

    let dielectric = 1.0 / options.solute_dielectric - 1.0 / options.solvent_dielectric;
    let mut total = Dual::constant(0.0, dimension);
    for first in 0..atoms.len() {
        for second in first..atoms.len() {
            let distance2 = if first == second {
                Dual::constant(0.0, dimension)
            } else {
                dual_squared_distance(&coordinates[first], &coordinates[second])
            };
            let born_product = born[first].clone().mul(born[second].clone());
            let exponential = distance2
                .clone()
                .scale(-0.25)
                .div(born_product.clone())
                .exp();
            let denominator = distance2
                .add(born_product.mul(exponential))
                .sqrt()
                .floor(1.0e-8);
            let factor = if first == second { 0.5 } else { 1.0 };
            let coefficient = -factor
                * COULOMB_KCAL_ANGSTROM
                * dielectric
                * atoms[first].charge()
                * atoms[second].charge();
            total = total.add(denominator.reciprocal().scale(coefficient));
        }
    }
    for (atom, born) in atoms.iter().zip(&born) {
        let radius = atom.gb_radius();
        let coefficient = 4.0
            * std::f64::consts::PI
            * options.surface_tension
            * (radius + options.probe_radius).powi(2);
        total = total.add(
            Dual::constant(radius, dimension)
                .div(born.clone())
                .powi(6)
                .scale(coefficient),
        );
    }
    total
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

/// Produce deterministic nonbonded candidate pairs. With no cutoff this is
/// the complete upper triangle. With a cutoff, atoms are binned in cubic
/// cells so large fixed proteins are not rescanned for every glycan pose.
fn nonbonded_pairs(coordinates: &[Vec3], cutoff: Option<f64>) -> Vec<(usize, usize)> {
    let Some(cutoff) = cutoff else {
        return (0..coordinates.len())
            .flat_map(|first| (first + 1..coordinates.len()).map(move |second| (first, second)))
            .collect();
    };
    let key = |point: Vec3| {
        (
            (point.x / cutoff).floor() as i32,
            (point.y / cutoff).floor() as i32,
            (point.z / cutoff).floor() as i32,
        )
    };
    let mut cells = BTreeMap::<(i32, i32, i32), Vec<usize>>::new();
    for (atom, point) in coordinates.iter().copied().enumerate() {
        cells.entry(key(point)).or_default().push(atom);
    }
    let cutoff2 = cutoff * cutoff;
    let mut pairs = Vec::new();
    for (first, point) in coordinates.iter().copied().enumerate() {
        let (cx, cy, cz) = key(point);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(atoms) = cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &second in atoms {
                            if second > first
                                && squared_distance(point, coordinates[second]) <= cutoff2
                            {
                                pairs.push((first, second));
                            }
                        }
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs
}

fn selected_nonbonded_pairs(
    coordinates: &[Vec3],
    cutoff: Option<f64>,
    selection: &AtomSelection,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    if let Some(cutoff) = cutoff {
        let (cells, key) = spatial_cells(coordinates, cutoff);
        let cutoff2 = cutoff * cutoff;
        for first in selection.indices() {
            let (cx, cy, cz) = key(coordinates[first]);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(atoms) = cells.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &second in atoms {
                                if first != second
                                    && squared_distance(coordinates[first], coordinates[second])
                                        <= cutoff2
                                {
                                    pairs.push(ordered(first, second));
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for first in selection.indices() {
            for second in 0..coordinates.len() {
                if first != second {
                    pairs.push(ordered(first, second));
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

fn cross_nonbonded_pairs(
    coordinates: &[Vec3],
    cutoff: Option<f64>,
    first_group: &AtomGroupMask,
    second_group: &AtomGroupMask,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    if let Some(cutoff) = cutoff {
        let (cells, key) = spatial_cells(coordinates, cutoff);
        let cutoff2 = cutoff * cutoff;
        for first in first_group.indices() {
            let (cx, cy, cz) = key(coordinates[first]);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(atoms) = cells.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &second in atoms {
                                if second_group.contains(second)
                                    && squared_distance(coordinates[first], coordinates[second])
                                        <= cutoff2
                                {
                                    pairs.push(ordered(first, second));
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for first in first_group.indices() {
            for second in second_group.indices() {
                pairs.push(ordered(first, second));
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

type CellKey = (i32, i32, i32);
fn spatial_cells(
    coordinates: &[Vec3],
    cutoff: f64,
) -> (BTreeMap<CellKey, Vec<usize>>, impl Fn(Vec3) -> CellKey) {
    let key = move |point: Vec3| {
        (
            (point.x / cutoff).floor() as i32,
            (point.y / cutoff).floor() as i32,
            (point.z / cutoff).floor() as i32,
        )
    };
    let mut cells = BTreeMap::<CellKey, Vec<usize>>::new();
    for (atom, point) in coordinates.iter().copied().enumerate() {
        cells.entry(key(point)).or_default().push(atom);
    }
    (cells, key)
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

fn add_scaled(target: &mut Vec3, vector: Vec3, factor: f64) {
    target.x += vector.x * factor;
    target.y += vector.y * factor;
    target.z += vector.z * factor;
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
        let mut plus = coordinates.clone();
        let mut minus = coordinates;
        plus[0].x += 2.0e-5;
        minus[0].x -= 2.0e-5;
        let expected = (evaluator.energy(&plus).unwrap().total()
            - evaluator.energy(&minus).unwrap().total())
            / 4.0e-5;
        assert!((gradient - expected).abs() < 2.0e-3);
    }

    #[test]
    fn obc2_gradient_matches_an_independent_finite_difference() {
        let system = system();
        let evaluator = EnergyEvaluator::new(&system, EnergyOptions::default())
            .unwrap()
            .with_selection(AtomSelection::from_indices(system.atom_count(), [0]))
            .unwrap();
        let coordinates = system.coordinates();
        let gradient = evaluator
            .energy_and_gradient(&coordinates)
            .unwrap()
            .gradients
            .unwrap()[0]
            .x;
        let mut plus = coordinates.clone();
        let mut minus = coordinates;
        plus[0].x += 2.0e-5;
        minus[0].x -= 2.0e-5;
        let expected = (evaluator.energy(&plus).unwrap().total()
            - evaluator.energy(&minus).unwrap().total())
            / 4.0e-5;
        assert!((gradient - expected).abs() < 5.0e-3);
    }

    #[test]
    fn neighbor_list_rebuilds_after_large_motion() {
        let mut coordinates = system().coordinates();
        let list = NeighborList::build(&coordinates, 8.0, 1.0).unwrap();
        coordinates[0].x += 0.6;
        assert!(list.needs_rebuild(&coordinates));
    }

    #[test]
    fn simd_interaction_matches_scalar_pair_sum() {
        let system = system();
        let midpoint = system.atom_count() / 2;
        let first = AtomGroupMask::from_indices(system.atom_count(), 0..midpoint);
        let second =
            AtomGroupMask::from_indices(system.atom_count(), midpoint..system.atom_count());
        let options = EnergyOptions {
            cutoff: Some(10.0),
            obc2: None,
            ..EnergyOptions::default()
        };
        let evaluator = EnergyEvaluator::new(&system, options).unwrap();
        let coordinates = system.coordinates();
        let simd = evaluator
            .interaction_energy(&coordinates, &first, &second)
            .unwrap();
        let mut scalar = InteractionEnergyComponents::default();
        for left in first.indices() {
            for right in second.indices() {
                let radius = distance(coordinates[left], coordinates[right]);
                if radius > 10.0 {
                    continue;
                }
                let pair = ordered(left, right);
                let scale = evaluator.one_four.get(&pair).copied();
                if system.exclusions()[left].contains(&right) && scale.is_none() {
                    continue;
                }
                let (scee, scnb) = scale.unwrap_or((1.0, 1.0));
                let a = &system.atoms()[left];
                let b = &system.atoms()[right];
                let sigma = a.lennard_jones_radius() + b.lennard_jones_radius();
                let epsilon = (a.lennard_jones_epsilon() * b.lennard_jones_epsilon()).sqrt();
                let ratio6 = (sigma / radius).powi(6);
                scalar.van_der_waals += epsilon * (ratio6 * ratio6 - 2.0 * ratio6) / scnb;
                scalar.electrostatics +=
                    COULOMB_KCAL_ANGSTROM * a.charge() * b.charge() / (radius * scee);
            }
        }
        assert!((simd.van_der_waals - scalar.van_der_waals).abs() < 1.0e-10);
        assert!((simd.electrostatics - scalar.electrostatics).abs() < 1.0e-10);
    }
}
