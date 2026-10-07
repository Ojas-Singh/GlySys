//! Molecular-mechanics energies and gradients for parameterized GlySys systems.

pub mod geometry;
pub mod hydration;
pub mod implicit_cluster;
mod obc2;
pub mod pbc;
pub mod pbc_cluster;
pub mod prior;
pub mod scoring;

/// Number of workers available to the shared CPU evaluator. Native builds
/// use the Rayon pool; portable browser WASM reports one unless the caller
/// initializes a shared-memory Rayon pool explicitly.
pub fn cpu_thread_count() -> usize {
    rayon::current_num_threads()
}

use std::collections::{BTreeMap, HashMap};

use glysys::{Atom, ParameterizedSystem, Vec3};
use pulp::{Arch, Simd, WithSimd};
use rayon::prelude::*;

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
    /// Homogeneous long-range Lennard-Jones correction used by periodic
    /// constant-pressure protocols. It is kept separate so reports can
    /// distinguish the cutoff energy from its volume-dependent correction.
    #[serde(default)]
    pub dispersion_correction: f64,
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

/// One parameterized torsion contribution, retained for downstream
/// per-linkage diagnostics. Multiple Fourier terms on the same four atoms
/// are returned separately and must be summed by the caller when desired.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TorsionEnergyContribution {
    pub atoms: [usize; 4],
    pub energy: f64,
    pub improper: bool,
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
            + self.dispersion_correction
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
        let (polar, surface) = if atoms.len() >= 128 {
            let polar_rows: Vec<f64> = (0..atoms.len())
                .into_par_iter()
                .map(|first| {
                    let mut row = 0.0;
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
                        row -= factor
                            * COULOMB_KCAL_ANGSTROM
                            * dielectric
                            * atoms[first].charge()
                            * atoms[second].charge()
                            / denominator;
                    }
                    row
                })
                .collect();
            let surface_rows: Vec<f64> = atoms
                .par_iter()
                .zip(&born)
                .map(|(atom, born)| {
                    let radius = atom.gb_radius();
                    4.0 * std::f64::consts::PI
                        * self.surface_tension
                        * (radius + self.probe_radius).powi(2)
                        * (radius / born).powi(6)
                })
                .collect();
            (polar_rows.into_iter().sum(), surface_rows.into_iter().sum())
        } else {
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
        };
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
    system: std::borrow::Cow<'a, ParameterizedSystem>,
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
        let one_four = system
            .one_four_pairs()
            .into_iter()
            .map(|(pair, scee, scnb)| (ordered(pair[0], pair[1]), (scee, scnb)))
            .collect::<HashMap<_, _>>();
        Ok(Self {
            system: std::borrow::Cow::Borrowed(system),
            options,
            selection: AtomSelection::all(system.atom_count()),
            one_four,
            active_terms_only: false,
        })
    }

    /// Retain prepared topology and lookup tables for a long-lived job.
    /// A borrowed topology is cloned once; subsequent evaluations reuse it.
    pub fn into_owned(self) -> EnergyEvaluator<'static> {
        EnergyEvaluator {
            system: std::borrow::Cow::Owned(self.system.into_owned()),
            options: self.options,
            selection: self.selection,
            one_four: self.one_four,
            active_terms_only: self.active_terms_only,
        }
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

    /// Visit interacting nonbonded pairs in ascending (first, second) order
    /// with their (scee, scnb) 1-4 scale factors; excluded pairs that are not
    /// 1-4 pairs are skipped. Without a cutoff every pair is a candidate: walk
    /// the upper triangle directly, since allocating it (N²/2 pairs) exhausts
    /// memory for large systems, notably under wasm32, and classify pairs by
    /// marking each row's exclusions instead of per-pair set lookups.
    fn for_each_nonbonded_pair(
        &self,
        coordinates: &[Vec3],
        mut visit: impl FnMut(usize, usize, (f64, f64)),
    ) {
        let exclusions = self.system.exclusions();
        if self.options.cutoff.is_some() || self.active_terms_only {
            for (first, second) in self.nonbonded_pairs(coordinates) {
                let scale = self.one_four.get(&ordered(first, second)).copied();
                if exclusions[first].contains(&second) && scale.is_none() {
                    continue;
                }
                visit(first, second, scale.unwrap_or((1.0, 1.0)));
            }
            return;
        }
        let n = coordinates.len();
        let mut rows = vec![Vec::new(); n];
        for (&(first, second), &scale) in &self.one_four {
            rows[first].push((second, scale));
        }
        const EXCLUDED: u8 = 1;
        const SCALED: u8 = 2;
        let mut marks = vec![0u8; n];
        let mut scales = vec![(1.0, 1.0); n];
        for first in 0..n {
            for &second in exclusions[first].range(first + 1..) {
                marks[second] = EXCLUDED;
            }
            for &(second, scale) in &rows[first] {
                marks[second] = SCALED;
                scales[second] = scale;
            }
            for second in first + 1..n {
                match marks[second] {
                    0 => visit(first, second, (1.0, 1.0)),
                    SCALED => visit(first, second, scales[second]),
                    _ => {}
                }
            }
            for &second in exclusions[first].range(first + 1..) {
                marks[second] = 0;
            }
            for &(second, _) in &rows[first] {
                marks[second] = 0;
            }
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
        self.interaction_energy_with_pair_count(coordinates, first_group, second_group)
            .map(|(energy, _)| energy)
    }

    /// Cross interaction energy together with the number of force-field
    /// pairs that survived cutoff and exclusions.
    pub fn interaction_energy_with_pair_count(
        &self,
        coordinates: &[Vec3],
        first_group: &AtomGroupMask,
        second_group: &AtomGroupMask,
    ) -> Result<(InteractionEnergyComponents, usize)> {
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
        let pair_count = radii2.len();
        Ok((
            Arch::new().dispatch(InteractionKernel {
                radii2: &radii2,
                sigmas: &sigmas,
                epsilons: &epsilons,
                charges: &charges,
            }),
            pair_count,
        ))
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
        let gradients = self.gradient_only(coordinates)?;
        Ok(EnergyResult {
            components,
            gradients: Some(gradients),
        })
    }

    /// Evaluate only the Cartesian gradient. Dynamics uses this on ordinary
    /// integration steps and requests full energy components at report and
    /// checkpoint boundaries, avoiding a second OBC2 all-pairs energy pass.
    pub fn gradient_only(&self, coordinates: &[Vec3]) -> Result<Vec<Vec3>> {
        validate_coordinates(self.system.atom_count(), coordinates)?;
        let mut gradients = self.analytic_gradient(coordinates, &[1.0; 9])?;
        for (gradient, residual) in gradients
            .iter_mut()
            .zip(self.residual_gradient(coordinates, &[1.0; 9]))
        {
            gradient.x += residual.x;
            gradient.y += residual.y;
            gradient.z += residual.z;
        }
        Ok(gradients)
    }

    /// The f32 implicit-solvent engine matching this evaluator's no-cutoff
    /// nonbonded and OBC2 terms, for dynamics force steps; `None` when a
    /// cutoff, term selection or vacuum model makes the reference path apply.
    pub fn implicit_pair_engine(&self) -> Result<Option<implicit_cluster::ImplicitPairEngine>> {
        let n = self.system.atom_count();
        let Some(obc2) = &self.options.obc2 else {
            return Ok(None);
        };
        if self.options.cutoff.is_some()
            || self.active_terms_only
            || (0..n).any(|atom| !self.selection.is_movable(atom))
        {
            return Ok(None);
        }
        implicit_cluster::ImplicitPairEngine::new(
            &self.system,
            obc2.clone(),
            self.options.dielectric,
        )
        .map(Some)
    }

    /// Bonded, torsion and restraint gradient only (f64), the complement of
    /// [`implicit_cluster::ImplicitPairEngine::gradient_into`].
    pub fn bonded_gradient(&self, coordinates: &[Vec3]) -> Result<Vec<Vec3>> {
        self.weighted_gradient(coordinates, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    }

    /// Gradient for dynamics integration steps: the same f64 physics as
    /// [`Self::gradient_only`], with the no-cutoff Lennard-Jones/Coulomb term
    /// evaluated row-parallel. Each atom owns its full row (every partner in
    /// index order), so the result is independent of the thread count but
    /// differs from the serial upper-triangle walk at rounding level. Cutoff,
    /// active-term and partial-selection evaluators use the serial path.
    pub fn gradient_parallel(&self, coordinates: &[Vec3]) -> Result<Vec<Vec3>> {
        let n = self.system.atom_count();
        if self.options.cutoff.is_some()
            || self.active_terms_only
            || (0..n).any(|atom| !self.selection.is_movable(atom))
        {
            return self.gradient_only(coordinates);
        }
        validate_coordinates(n, coordinates)?;
        let mut gradients =
            self.analytic_gradient(coordinates, &[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])?;
        let atoms = self.system.atoms();
        let charge: Vec<f64> = atoms.iter().map(|a| a.charge()).collect();
        let sigma: Vec<f64> = atoms.iter().map(|a| a.lennard_jones_radius()).collect();
        let epsilon: Vec<f64> = atoms.iter().map(|a| a.lennard_jones_epsilon()).collect();
        let mut scaled_rows = vec![Vec::new(); n];
        for (&(first, second), &scale) in &self.one_four {
            scaled_rows[first].push((second, scale));
            scaled_rows[second].push((first, scale));
        }
        let exclusions = self.system.exclusions();
        let dielectric = self.options.dielectric;
        const EXCLUDED: u8 = 1;
        const SCALED: u8 = 2;
        let rows: Vec<Vec3> = (0..n)
            .into_par_iter()
            .map_init(
                || (vec![0u8; n], vec![(1.0, 1.0); n]),
                |(marks, scales), first| {
                    for &second in &exclusions[first] {
                        marks[second] = EXCLUDED;
                    }
                    for &(second, scale) in &scaled_rows[first] {
                        marks[second] = SCALED;
                        scales[second] = scale;
                    }
                    marks[first] = EXCLUDED;
                    let mut row = Vec3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    };
                    for second in 0..n {
                        let (scee, scnb) = match marks[second] {
                            0 => (1.0, 1.0),
                            SCALED => scales[second],
                            _ => continue,
                        };
                        let vector = subtract(coordinates[first], coordinates[second]);
                        let radius = norm(vector).max(1.0e-8);
                        let ratio6 = ((sigma[first] + sigma[second]) / radius).powi(6);
                        let coulomb = COULOMB_KCAL_ANGSTROM * charge[first] * charge[second]
                            / (dielectric * scee * radius);
                        let derivative = 12.0
                            * (epsilon[first] * epsilon[second]).sqrt()
                            * (ratio6 - ratio6 * ratio6)
                            / (scnb * radius)
                            - coulomb / radius;
                        add_scaled(&mut row, vector, derivative / radius);
                    }
                    for &second in &exclusions[first] {
                        marks[second] = 0;
                    }
                    for &(second, _) in &scaled_rows[first] {
                        marks[second] = 0;
                    }
                    marks[first] = 0;
                    row
                },
            )
            .collect();
        for (gradient, (row, residual)) in gradients.iter_mut().zip(
            rows.into_iter()
                .zip(self.residual_gradient(coordinates, &[1.0; 9])),
        ) {
            add_scaled(gradient, row, 1.0);
            add_scaled(gradient, residual, 1.0);
        }
        Ok(gradients)
    }

    /// Analytic derivative of an explicitly weighted component sum.
    pub fn weighted_gradient(&self, coordinates: &[Vec3], weights: [f64; 9]) -> Result<Vec<Vec3>> {
        if weights.iter().any(|w| !w.is_finite()) {
            return Err(EnergyError::InvalidConfiguration(
                "nonfinite term weight".into(),
            ));
        }
        let mut gradient = self.analytic_gradient(coordinates, &weights)?;
        for (g, r) in gradient
            .iter_mut()
            .zip(self.residual_gradient(coordinates, &weights))
        {
            add_scaled(g, r, 1.);
        }
        Ok(gradient)
    }
    pub fn interaction_gradient(
        &self,
        coordinates: &[Vec3],
        first: &AtomGroupMask,
        second: &AtomGroupMask,
        weights: [f64; 2],
    ) -> Result<Vec<Vec3>> {
        self.interaction_energy(coordinates, first, second)?;
        let mut gradients = vec![
            Vec3 {
                x: 0.,
                y: 0.,
                z: 0.
            };
            coordinates.len()
        ];
        for (a, b) in cross_nonbonded_pairs(coordinates, self.options.cutoff, first, second) {
            let scales = self.one_four.get(&ordered(a, b)).copied();
            if self.system.exclusions()[a].contains(&b) && scales.is_none() {
                continue;
            }
            let (scee, scnb) = scales.unwrap_or((1., 1.));
            let aa = &self.system.atoms()[a];
            let ab = &self.system.atoms()[b];
            let delta = subtract(coordinates[a], coordinates[b]);
            let d = norm(delta).max(1e-8);
            let epsilon = (aa.lennard_jones_epsilon() * ab.lennard_jones_epsilon()).sqrt();
            let r6 = ((aa.lennard_jones_radius() + ab.lennard_jones_radius()) / d).powi(6);
            let coulomb = COULOMB_KCAL_ANGSTROM * aa.charge() * ab.charge()
                / (self.options.dielectric * scee * d);
            let factor = (weights[0] * 12. * epsilon * (r6 - r6 * r6) / (scnb * d)
                - weights[1] * coulomb / d)
                / d;
            if self.selection.is_movable(a) {
                add_scaled(&mut gradients[a], delta, factor);
            }
            if self.selection.is_movable(b) {
                add_scaled(&mut gradients[b], delta, -factor);
            }
        }
        Ok(gradients)
    }

    fn analytic_gradient(&self, coordinates: &[Vec3], weights: &[f64; 9]) -> Result<Vec<Vec3>> {
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
            let derivative = weights[0] * 2.0 * bond.force() * (radius - bond.length()) / radius;
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
            let factor = weights[1] * 2.0 * angle.force() * (theta - angle.radians()) / sine;
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
        let atoms = self.system.atoms();
        let apply_pair = |first: usize, second: usize, (scee, scnb): (f64, f64)| {
            if self.active_terms_only
                && !self.selection.is_movable(first)
                && !self.selection.is_movable(second)
            {
                return;
            }
            let vector = subtract(coordinates[first], coordinates[second]);
            let radius = norm(vector).max(1.0e-8);
            let first_atom = &atoms[first];
            let second_atom = &atoms[second];
            let sigma = first_atom.lennard_jones_radius() + second_atom.lennard_jones_radius();
            let epsilon =
                (first_atom.lennard_jones_epsilon() * second_atom.lennard_jones_epsilon()).sqrt();
            let ratio6 = (sigma / radius).powi(6);
            let coulomb = COULOMB_KCAL_ANGSTROM * first_atom.charge() * second_atom.charge()
                / (self.options.dielectric * scee * radius);
            let derivative = weights[4] * 12.0 * epsilon * (ratio6 - ratio6 * ratio6)
                / (scnb * radius)
                - weights[5] * coulomb / radius;
            add_scaled(&mut gradient[first], vector, derivative / radius);
            add_scaled(&mut gradient[second], vector, -derivative / radius);
        };
        if weights[4] != 0.0 || weights[5] != 0.0 {
            self.for_each_nonbonded_pair(coordinates, apply_pair);
        }
        for restraint in &self.options.restraints {
            if let Some(position) = coordinates.get(restraint.atom) {
                let vector = subtract(*position, restraint.reference);
                add_scaled(
                    &mut gradient[restraint.atom],
                    vector,
                    weights[8] * 2.0 * restraint.force,
                );
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

    fn residual_gradient(&self, coordinates: &[Vec3], weights: &[f64; 9]) -> Vec<Vec3> {
        let mut gradients = vec![
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            };
            coordinates.len()
        ];
        // Torsions are local four-atom terms. Use the fixed-size derivative
        // kernel shared with the periodic evaluator instead of allocating a
        // heap-backed derivative vector for every scalar operation.
        for torsion in self.system.dihedrals() {
            let atoms = torsion.atoms();
            if self.active_terms_only
                && !atoms
                    .into_iter()
                    .any(|atom| self.selection.is_movable(atom))
            {
                continue;
            }
            let points = atoms.map(|atom| coordinates[atom]);
            let (phi, phi_gradient) =
                pbc::dihedral_with_gradient(points[0], points[1], points[2], points[3]);
            let periodicity = torsion.periodicity() as f64;
            let derivative = -torsion.force()
                * periodicity
                * (periodicity * phi - torsion.phase()).sin()
                * weights[if torsion.is_improper() { 3 } else { 2 }];
            for (local, atom) in atoms.iter().enumerate() {
                if self.selection.is_movable(*atom) {
                    add_scaled(&mut gradients[*atom], phi_gradient[local], derivative);
                }
            }
        }
        if let Some(options) = self
            .options
            .obc2
            .as_ref()
            .filter(|_| weights[6] != 0.0 || weights[7] != 0.0)
        {
            if weights[6] == weights[7] {
                let values = obc2::gradient(self.system.atoms(), coordinates, options);
                for (i, value) in values.into_iter().enumerate() {
                    if self.selection.is_movable(i) {
                        add_scaled(&mut gradients[i], value, weights[6]);
                    }
                }
            } else {
                let mut polar = options.clone();
                polar.surface_tension = 0.;
                let mut surface = options.clone();
                surface.solute_dielectric = surface.solvent_dielectric;
                for (opts, weight) in [(&polar, weights[6]), (&surface, weights[7])] {
                    if weight != 0. {
                        for (i, value) in obc2::gradient(self.system.atoms(), coordinates, opts)
                            .into_iter()
                            .enumerate()
                        {
                            if self.selection.is_movable(i) {
                                add_scaled(&mut gradients[i], value, weight);
                            }
                        }
                    }
                }
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
        self.for_each_nonbonded_pair(coordinates, |first, second, (scee, scnb)| {
            let r = distance(coordinates[first], coordinates[second]).max(1.0e-8);
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
        });
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

    /// Return each proper/improper torsion term without collapsing distinct
    /// Fourier terms. This is a diagnostic API used to attribute
    /// glycosidic-bond torsions in ReGlyco reports.
    pub fn torsion_energy_contributions(
        &self,
        coordinates: &[Vec3],
    ) -> Result<Vec<TorsionEnergyContribution>> {
        validate_coordinates(self.system.atom_count(), coordinates)?;
        Ok(self
            .system
            .dihedrals()
            .iter()
            .map(|torsion| {
                let atoms = torsion.atoms();
                let phi = dihedral(
                    coordinates[atoms[0]],
                    coordinates[atoms[1]],
                    coordinates[atoms[2]],
                    coordinates[atoms[3]],
                );
                TorsionEnergyContribution {
                    atoms,
                    energy: torsion.force()
                        * (1.0 + ((torsion.periodicity() as f64) * phi - torsion.phase()).cos()),
                    improper: torsion.is_improper(),
                }
            })
            .collect())
    }
}

#[cfg(test)]
#[derive(Clone)]
struct Dual {
    value: f64,
    gradient: Vec<f64>,
}

#[cfg(test)]
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

    #[cfg(test)]
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

    #[cfg(test)]
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

    #[cfg(test)]
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

    #[cfg(test)]
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

    #[cfg(test)]
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

    #[cfg(test)]
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

#[cfg(test)]
#[derive(Clone)]
struct DualVec3 {
    x: Dual,
    y: Dual,
    z: Dual,
}

#[cfg(test)]
fn dual_subtract(first: &DualVec3, second: &DualVec3) -> DualVec3 {
    DualVec3 {
        x: first.x.clone().sub(second.x.clone()),
        y: first.y.clone().sub(second.y.clone()),
        z: first.z.clone().sub(second.z.clone()),
    }
}

#[cfg(test)]
fn dual_dot(first: &DualVec3, second: &DualVec3) -> Dual {
    first
        .x
        .clone()
        .mul(second.x.clone())
        .add(first.y.clone().mul(second.y.clone()))
        .add(first.z.clone().mul(second.z.clone()))
}

#[cfg(test)]
fn dual_squared_distance(first: &DualVec3, second: &DualVec3) -> Dual {
    let difference = dual_subtract(first, second);
    dual_dot(&difference, &difference)
}

#[cfg(test)]
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
            let scaled = (atoms[second].gb_radius() - OFFSET).max(0.1) * atoms[second].gb_screen();
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
    if atoms.len() >= 128 {
        return (0..atoms.len())
            .into_par_iter()
            .map(|first| obc2_born_radius(first, atoms, coordinates))
            .collect();
    }
    (0..atoms.len())
        .map(|first| obc2_born_radius(first, atoms, coordinates))
        .collect()
}

fn obc2_born_radius(first: usize, atoms: &[Atom], coordinates: &[Vec3]) -> f64 {
    const OFFSET: f64 = 0.09;
    const ALPHA: f64 = 1.0;
    const BETA: f64 = 0.8;
    const GAMMA: f64 = 4.85;
    let radius = (atoms[first].gb_radius() - OFFSET).max(0.1);
    let mut integral = 0.0;
    for second in 0..atoms.len() {
        if first == second {
            continue;
        }
        let distance = distance(coordinates[first], coordinates[second]).max(1.0e-8);
        let scaled = (atoms[second].gb_radius() - OFFSET).max(0.1) * atoms[second].gb_screen();
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
    1.0 / (1.0 / radius - tanh / atoms[first].gb_radius()).max(1.0e-6)
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
    let b0 = subtract(first, second);
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
    fn analytic_obc2_matches_full_forward_ad() {
        let system = system();
        let coordinates = system.coordinates();
        let n = coordinates.len();
        let points = coordinates
            .iter()
            .enumerate()
            .map(|(i, p)| DualVec3 {
                x: Dual::coordinate(p.x, 3 * n, Some(3 * i)),
                y: Dual::coordinate(p.y, 3 * n, Some(3 * i + 1)),
                z: Dual::coordinate(p.z, 3 * n, Some(3 * i + 2)),
            })
            .collect::<Vec<_>>();
        let options = Obc2Options::default();
        let reference = dual_obc2(system.atoms(), &points, &options, 3 * n);
        let actual = obc2::gradient(system.atoms(), &coordinates, &options);
        for (i, g) in actual.iter().enumerate() {
            for (axis, value) in [g.x, g.y, g.z].into_iter().enumerate() {
                let expected = reference.gradient[3 * i + axis];
                assert!(
                    (value - expected).abs() < 1e-8 * (1.0 + expected.abs()),
                    "atom={i}, axis={axis}, actual={value}, expected={expected}"
                );
            }
        }
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
