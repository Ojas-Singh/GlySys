//! Molecular dynamics: implicit-solvent Langevin plus explicit-water PBC.
//! Distances Å, time ps, energy kcal/mol.
pub mod accumulators;
pub mod analysis;
pub mod explicit;
pub mod minimization;
pub mod resident_rng;
pub mod session;
pub mod settle;
use glysys::{ParameterizedSystem, Vec3};
use glysys_energy::{EnergyEvaluator, EnergyOptions, Obc2Options};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub(crate) const KB: f64 = 0.00198720425864083;
pub const ACCEL: f64 = 418.4; // (kcal/mol/Å)/amu -> Å/ps²
/// Bar per kcal/mol/Å³, for pressure reporting from the virial.  The name is
/// deliberately explicit: this is **bar**, not kilobar.
pub const BAR_PER_KCAL_MOL_A3: f64 = 4184.0 * 1e25 / 6.02214076e23;
/// Compatibility alias for older in-tree parity probes.  New code should use
/// [`BAR_PER_KCAL_MOL_A3`].
#[allow(dead_code)]
pub(crate) const KBAR_PER_KCAL_MOL_A3: f64 = BAR_PER_KCAL_MOL_A3;
pub const MODEL_VERSION: &str = "obc2-baoab-v1";
/// Explicit-water PBC model: TIP3P, cutoff plus reaction field, velocity
/// Verlet NVE / BAOAB NVT / Monte Carlo barostat NPT, SETTLE waters.
pub const EXPLICIT_MODEL_VERSION: &str = "tip3p-rf-md-v1";
/// Corrected constant-pressure model. The v1 model remains readable for
/// historical NVE/NVT checkpoints, while new NPT/dispersion runs use v2.
pub const EXPLICIT_NPT_MODEL_VERSION: &str = "tip3p-rf-md-npt-v2";
/// Browser requests retain a bounded schedule so an untrusted page cannot
/// accidentally enqueue an effectively unending job. Native/HPC sessions
/// use the larger bound below and still rely on checkpoints and signals for
/// operational control.
pub const MAX_BROWSER_STEPS: usize = 10_000_000;
pub const MAX_NATIVE_STEPS: usize = 2_000_000_000;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid simulation: {0}")]
    Invalid(String),
    #[error(transparent)]
    Energy(#[from] glysys_energy::EnergyError),
    #[error(transparent)]
    Optimization(#[from] glysys_opt::OptimizationError),
}
pub type Result<T> = std::result::Result<T, Error>;
/// Solvent treatment for a run. `implicit` is the original OBC2 path and
/// stays byte-for-byte compatible; `explicit` requires a solvated preparation
/// with a periodic box and runs the PBC engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SolventModel {
    #[default]
    Implicit,
    Explicit,
}

/// Ensemble per simulation segment. NVE exists so force, constraint, PBC,
/// and neighbor-list errors show up as energy drift instead of hiding behind
/// a thermostat.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Ensemble {
    Nve,
    #[default]
    Nvt,
    Npt,
}

/// A normalized execution segment. New browser requests use three explicit
/// stages; the legacy two-segment fields remain valid input adapters.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SimulationStage {
    pub id: String,
    pub ensemble: Ensemble,
    pub steps: usize,
    #[serde(default)]
    pub barostat_adaptation: bool,
}

/// Rigid-water treatment. `none` integrates flexible waters at small
/// timesteps; `settle` constrains O-H and H-H distances.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConstraintModel {
    #[default]
    None,
    Settle,
}

/// Electrostatics model selected by a simulation protocol.  Reaction field is
/// the validated browser/native model today.  PME is represented explicitly so
/// an imported GROMACS recipe cannot be mistaken for reaction field while the
/// reciprocal-space implementation is being completed behind the same engine
/// session.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ElectrostaticsModel {
    #[default]
    ReactionField,
    Pme,
}

/// Pressure-coupling algorithm.  The legacy Monte Carlo barostat remains the
/// default for old JSON protocols; Parrinello–Rahman is an explicit model
/// choice and is rejected by drivers until its constrained stress path is
/// validated against the GOTW reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PressureCoupling {
    #[default]
    MonteCarlo,
    None,
    ParrinelloRahman,
}

/// Thermostat for explicit NVT/NPT segments. Langevin matches OpenMM's
/// LangevinMiddle ordering for parity runs; v-rescale enforces the target
/// temperature exactly and is the production default with constraints.
/// SETTLE carries the position-constraint impulse into the half-step
/// velocity, and the closed-form RATTLE projection is applied after kicks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Thermostat {
    Langevin,
    #[default]
    VRescale,
    NoseHoover,
}

/// One temperature-coupling group retained from an imported native protocol.
/// The group is part of the protocol identity even while the corresponding
/// Nose–Hoover implementation is still behind its validation gate.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ThermostatGroup {
    pub name: String,
    pub tau_ps: f64,
    pub reference_temperature_k: f64,
}

fn default_pressure_bar() -> f64 {
    1.0
}

fn default_barostat_interval() -> usize {
    25
}

fn default_pressure_estimator() -> String {
    "atomic-virial".into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "camelCase")]
pub struct SimulationProtocol {
    pub temperature_k: f64,
    pub timestep_fs: f64,
    pub friction_per_ps: f64,
    pub equilibration_steps: usize,
    pub production_steps: usize,
    pub save_every: usize,
    pub minimization_iterations: usize,
    #[serde(with = "u64_string")]
    pub seed: u64,
    /// Solvent treatment. Defaults to the original implicit behavior.
    pub solvent: SolventModel,
    /// Ensemble for the equilibration and production segments.
    pub equilibration_ensemble: Ensemble,
    pub production_ensemble: Ensemble,
    /// Target pressure for NPT segments, in bar.
    #[serde(default = "default_pressure_bar")]
    pub pressure_bar: f64,
    /// Water constraints for explicit runs.
    pub constraints: ConstraintModel,
    /// Thermostat for explicit NVT/NPT segments.
    pub thermostat: Thermostat,
    /// Electrostatics selected for explicit periodic simulations.
    #[serde(default)]
    pub electrostatics: ElectrostaticsModel,
    /// Pressure coupling selected for NPT segments. Old protocols deserialize
    /// as the validated Monte Carlo barostat.
    #[serde(default)]
    pub pressure_coupling: PressureCoupling,
    /// Nonbonded cutoff in angstrom for explicit runs (9-10 recommended).
    pub cutoff_angstrom: Option<f64>,
    /// Reaction-field solvent dielectric for explicit runs.
    pub rf_dielectric: Option<f64>,
    /// Monte Carlo barostat attempt interval in steps.
    #[serde(default = "default_barostat_interval")]
    pub barostat_interval: usize,
    /// Harmonic solute restraint strength in kcal/mol/A^2 (0 disables).
    #[serde(default)]
    pub restraint_force: f64,
    /// Enable OpenMM-compatible homogeneous long-range LJ correction. Old
    /// protocols omit this field and retain truncated-LJ behavior.
    #[serde(default)]
    pub dispersion_correction: bool,
    /// Optional normalized stage list. When absent, the legacy equilibration
    /// and production fields define two stages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stages: Option<Vec<SimulationStage>>,
    /// Temperature-coupling groups from an imported native recipe. Empty
    /// means that the legacy single-temperature setting is in use.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub thermostat_groups: Vec<ThermostatGroup>,
    /// Nose–Hoover/Parrinello–Rahman coupling time, in ps, when supplied by a
    /// native recipe. It remains metadata until its driver is validated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pressure_tau_ps: Option<f64>,
    /// Isotropic compressibility in bar^-1, retained per coupling group.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pressure_compressibility_bar_inverse: Vec<f64>,
    /// Center-of-mass removal mode from a native recipe (for example
    /// `linear`). Empty means no imported COM policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub com_mode: Option<String>,
    /// Center-of-mass groups from a native recipe.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub com_groups: Vec<String>,
}
impl Default for SimulationProtocol {
    fn default() -> Self {
        Self {
            temperature_k: 300.,
            timestep_fs: 0.5,
            friction_per_ps: 1.,
            equilibration_steps: 2000,
            production_steps: 18000,
            save_every: 100,
            minimization_iterations: 500,
            seed: 0,
            solvent: SolventModel::Implicit,
            equilibration_ensemble: Ensemble::Nvt,
            production_ensemble: Ensemble::Nvt,
            pressure_bar: 1.0,
            constraints: ConstraintModel::None,
            thermostat: Thermostat::VRescale,
            electrostatics: ElectrostaticsModel::ReactionField,
            pressure_coupling: PressureCoupling::MonteCarlo,
            cutoff_angstrom: None,
            rf_dielectric: None,
            barostat_interval: 25,
            restraint_force: 0.0,
            dispersion_correction: false,
            stages: None,
            thermostat_groups: Vec::new(),
            pressure_tau_ps: None,
            pressure_compressibility_bar_inverse: Vec::new(),
            com_mode: None,
            com_groups: Vec::new(),
        }
    }
}

impl SimulationProtocol {
    /// Return the execution stages without mutating the compatibility fields.
    pub fn execution_stages(&self) -> Vec<SimulationStage> {
        self.stages.clone().unwrap_or_else(|| {
            let mut stages = Vec::with_capacity(2);
            if self.equilibration_steps > 0 {
                stages.push(SimulationStage {
                    id: "equilibration".into(),
                    ensemble: self.equilibration_ensemble,
                    steps: self.equilibration_steps,
                    barostat_adaptation: self.equilibration_ensemble == Ensemble::Npt,
                });
            }
            stages.push(SimulationStage {
                id: "production".into(),
                ensemble: self.production_ensemble,
                steps: self.production_steps,
                barostat_adaptation: false,
            });
            stages
        })
    }

    pub fn total_steps(&self) -> usize {
        self.execution_stages()
            .iter()
            .map(|stage| stage.steps)
            .sum()
    }

    /// `(stage index, id, ensemble, adaptation enabled, local step)` for a
    /// global committed step. The final stage is returned at the endpoint.
    pub fn stage_info(&self, step: usize) -> (usize, String, Ensemble, bool, usize) {
        let stages = self.execution_stages();
        let mut start = 0usize;
        for (index, stage) in stages.iter().enumerate() {
            let end = start.saturating_add(stage.steps);
            if step < end || (step == end && index + 1 == stages.len()) {
                return (
                    index,
                    stage.id.clone(),
                    stage.ensemble,
                    stage.barostat_adaptation,
                    step.saturating_sub(start).min(stage.steps),
                );
            }
            start = end;
        }
        let index = stages.len().saturating_sub(1);
        let stage = &stages[index];
        (
            index,
            stage.id.clone(),
            stage.ensemble,
            stage.barostat_adaptation,
            stage.steps,
        )
    }

    pub fn has_npt(&self) -> bool {
        self.execution_stages()
            .iter()
            .any(|stage| stage.ensemble == Ensemble::Npt)
    }

    /// True when a committed step is the end of one stage (including the
    /// final stage).  Stage-boundary frames are retained even when they do
    /// not coincide with the regular `save_every` cadence.
    pub fn is_stage_boundary(&self, step: usize) -> bool {
        let mut offset = 0usize;
        for stage in self.execution_stages() {
            offset = offset.saturating_add(stage.steps);
            if step == offset {
                return true;
            }
        }
        false
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulationState {
    pub schema_version: u32,
    pub model_version: String,
    pub system_fingerprint: String,
    pub protocol: SimulationProtocol,
    pub step: usize,
    pub coordinates: Vec<Vec3>,
    pub reference_coordinates: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub gradient: Vec<Vec3>,
    pub potential_energy: f64,
    #[serde(with = "u64_string")]
    pub rng_state: u64,
    /// Optional per-atom thermostat stream shared by CPU fallback and GPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resident_rng: Option<resident_rng::ResidentThermostatRng>,
    pub integrator_phase: String,
    /// Periodic box in angstrom; zeros for the nonperiodic implicit path.
    #[serde(default)]
    pub box_angstrom: [f64; 3],
    /// `sum(r_unwrapped . F)` in kcal/mol; feeds pressure reporting.
    #[serde(default)]
    pub virial_kcal_mol: f64,
    /// Instantaneous pressure in bar.  `pressure_estimator` identifies
    /// whether this is the molecular finite-difference probe or the cheaper
    /// atomic pair-virial diagnostic.
    #[serde(default)]
    pub pressure_bar: f64,
    #[serde(default = "default_pressure_estimator")]
    pub pressure_estimator: String,
    /// Independent Monte Carlo barostat stream; thermostat noise keeps the
    /// main `rng_state` reproducible across barostat schedule changes.
    #[serde(default)]
    pub barostat_rng: u64,
    /// Absolute volume proposal width in Å³. Zero means no NPT stage.
    #[serde(default)]
    pub barostat_volume_width: f64,
    #[serde(default)]
    pub barostat_attempts: u64,
    #[serde(default)]
    pub barostat_accepts: u64,
    #[serde(default)]
    pub barostat_window_attempts: u32,
    #[serde(default)]
    pub barostat_window_accepts: u32,
    #[serde(default)]
    pub barostat_step_counter: usize,
    #[serde(default)]
    pub barostat_frozen: bool,
    /// On-the-fly water occupancy grid over production frames, if enabled.
    #[serde(default)]
    pub water_occupancy: Option<accumulators::WaterOccupancy>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrajectoryFrame {
    pub step: usize,
    pub time_ps: f64,
    pub segment: String,
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub temperature_k: f64,
    pub coordinates: Vec<Vec3>,
    /// Periodic box at this frame; zeros for implicit runs.
    #[serde(default)]
    pub box_angstrom: [f64; 3],
    /// Instantaneous pressure in bar; zero for non-NPT segments.
    #[serde(default)]
    pub pressure_bar: f64,
    #[serde(default = "default_pressure_estimator")]
    pub pressure_estimator: String,
    /// System density in g/mL; zero for implicit runs.
    #[serde(default)]
    pub density_g_ml: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrajectoryChunk {
    pub first_step: usize,
    pub last_step: usize,
    pub frames: Vec<TrajectoryFrame>,
}
mod u64_string {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(
        value: &u64,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<u64, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Value {
            Text(String),
            Number(u64),
        }
        match Value::deserialize(deserializer)? {
            Value::Text(s) => s.parse().map_err(serde::de::Error::custom),
            Value::Number(n) => Ok(n),
        }
    }
}
pub(crate) fn invalid(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}

pub(crate) fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    ((x >> 11) as f64 + 0.5) / 9007199254740992.
}

pub(crate) fn normal(state: &mut u64) -> f64 {
    (-2. * uniform(state).ln()).sqrt() * (std::f64::consts::TAU * uniform(state)).cos()
}
fn add(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
    }
}
fn scale(a: Vec3, s: f64) -> Vec3 {
    Vec3 {
        x: a.x * s,
        y: a.y * s,
        z: a.z * s,
    }
}
fn norm2(a: Vec3) -> f64 {
    a.x * a.x + a.y * a.y + a.z * a.z
}
fn finite(a: &Vec3) -> bool {
    a.x.is_finite() && a.y.is_finite() && a.z.is_finite()
}
/// Advance the same RNG stream as CPU BAOAB, packing independent normal triples.
pub fn normal_noise(state: &mut u64, count: usize) -> Vec<[f32; 4]> {
    (0..count)
        .map(|_| {
            [
                normal(state) as f32,
                normal(state) as f32,
                normal(state) as f32,
                0.,
            ]
        })
        .collect()
}
fn random_velocity(state: &mut u64, s: f64) -> Vec3 {
    Vec3 {
        x: normal(state) * s,
        y: normal(state) * s,
        z: normal(state) * s,
    }
}
fn fingerprint(system: &ParameterizedSystem) -> String {
    format!(
        "{:x}",
        Sha256::digest(format!("{MODEL_VERSION}:{system:?}").as_bytes())
    )
}
impl SimulationProtocol {
    /// Validate a simulation request for native/HPC use. Native runs may
    /// represent hundreds of nanoseconds, so this uses the larger operational
    /// limit while preserving every physical setting check.
    pub fn validate(&self) -> Result<()> {
        self.validate_for_native()
    }

    /// Explicit spelling for adapters that want to document that the larger
    /// native schedule bound is intentional.
    pub fn validate_for_native(&self) -> Result<()> {
        self.validate_with_limit(MAX_NATIVE_STEPS)
    }

    /// Validate an untrusted browser request. The browser adapter must call
    /// this before constructing a session so a page cannot enqueue an
    /// accidental multi-hundred-nanosecond job.
    pub fn validate_for_browser(&self) -> Result<()> {
        self.validate_with_limit(MAX_BROWSER_STEPS)
    }

    fn validate_with_limit(&self, max_steps: usize) -> Result<()> {
        let legacy_schedule_valid = self.stages.is_some()
            || (self.production_steps > 0
                && self
                    .equilibration_steps
                    .checked_add(self.production_steps)
                    .is_some_and(|s| s <= max_steps));
        if !self.temperature_k.is_finite()
            || self.temperature_k <= 0.
            || !self.timestep_fs.is_finite()
            || self.timestep_fs <= 0.
            || self.timestep_fs > 2.0
            || !self.friction_per_ps.is_finite()
            || self.friction_per_ps < 0.
            || self.save_every == 0
            || !legacy_schedule_valid
            || !self.pressure_bar.is_finite()
            || self.pressure_bar < 0.
            || self.barostat_interval == 0
            || !self.restraint_force.is_finite()
            || self.restraint_force < 0.
            || self
                .cutoff_angstrom
                .is_some_and(|c| !c.is_finite() || c <= 0.)
            || self.rf_dielectric.is_some_and(|d| !d.is_finite() || d < 1.)
            || self.thermostat_groups.iter().any(|group| {
                group.name.trim().is_empty()
                    || !group.tau_ps.is_finite()
                    || group.tau_ps <= 0.
                    || !group.reference_temperature_k.is_finite()
                    || group.reference_temperature_k <= 0.
            })
            || self
                .pressure_tau_ps
                .is_some_and(|tau| !tau.is_finite() || tau <= 0.)
            || self
                .pressure_compressibility_bar_inverse
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.)
            || self
                .com_mode
                .as_ref()
                .is_some_and(|mode| mode.trim().is_empty())
            || self.com_groups.iter().any(|group| group.trim().is_empty())
        {
            return Err(invalid(
                "positive temperature, timestep ≤2 fs, finite friction/pressure/restraints and bounded step counts required",
            ));
        }
        let stages = self.execution_stages();
        if stages.is_empty()
            || stages.iter().any(|stage| {
                stage.id.trim().is_empty()
                    || stage.steps == 0
                    || stage.steps > max_steps
                    || (stage.ensemble == Ensemble::Npt && self.pressure_bar <= 0.)
            })
            || stages
                .iter()
                .map(|stage| stage.steps)
                .try_fold(0usize, usize::checked_add)
                .is_none_or(|total| total > max_steps)
            || {
                let mut ids = std::collections::BTreeSet::new();
                stages.iter().any(|stage| !ids.insert(stage.id.as_str()))
            }
        {
            return Err(invalid(
                "stages need unique nonempty ids, positive bounded steps, and positive NPT pressure",
            ));
        }
        if self.has_npt() && self.friction_per_ps <= 0. {
            return Err(invalid("NPT stages require nonzero Langevin friction"));
        }
        if self.solvent != SolventModel::Explicit
            && (self.electrostatics != ElectrostaticsModel::ReactionField
                || self.pressure_coupling != PressureCoupling::MonteCarlo)
        {
            return Err(invalid(
                "PME and explicit pressure-coupling choices require solvent=explicit",
            ));
        }
        if self.pressure_coupling == PressureCoupling::None && self.has_npt() {
            return Err(invalid("NPT stages require a pressure-coupling algorithm"));
        }
        if self.pressure_coupling == PressureCoupling::ParrinelloRahman && !self.has_npt() {
            return Err(invalid("Parrinello-Rahman is only valid for an NPT stage"));
        }
        Ok(())
    }
}
/// Transactional BAOAB: an error leaves the committed state and RNG untouched.
pub fn step_with<F>(state: &mut SimulationState, masses: &[f64], mut evaluate: F) -> Result<()>
where
    F: FnMut(&[Vec3]) -> Result<(f64, Vec<Vec3>)>,
{
    state.protocol.validate()?;
    if masses.len() != state.coordinates.len()
        || state.velocities.len() != masses.len()
        || state.gradient.len() != masses.len()
        || masses.iter().any(|m| !m.is_finite() || *m <= 0.)
    {
        return Err(invalid("invalid masses or state dimensions"));
    }
    if let Some(rng) = &state.resident_rng {
        rng.validate(masses.len())?;
    }
    let mut next = state.clone();
    let dt = next.protocol.timestep_fs * 0.001;
    let decay = (-next.protocol.friction_per_ps * dt).exp();
    for (i, mass) in masses.iter().enumerate() {
        next.velocities[i] = add(
            next.velocities[i],
            scale(next.gradient[i], -0.5 * dt * ACCEL / mass),
        );
        next.coordinates[i] = add(next.coordinates[i], scale(next.velocities[i], 0.5 * dt));
        let sigma = ((1. - decay * decay) * KB * next.protocol.temperature_k * ACCEL / mass).sqrt();
        next.velocities[i] = add(
            scale(next.velocities[i], decay),
            if let Some(rng) = &mut next.resident_rng {
                scale(rng.normal3(i), sigma)
            } else {
                random_velocity(&mut next.rng_state, sigma)
            },
        );
        next.coordinates[i] = add(next.coordinates[i], scale(next.velocities[i], 0.5 * dt));
    }
    let (energy, gradient) = evaluate(&next.coordinates)?;
    if !energy.is_finite() || gradient.len() != masses.len() || gradient.iter().any(|g| !finite(g))
    {
        return Err(invalid(
            "nonfinite energy or gradient; previous checkpoint retained",
        ));
    }
    for (i, mass) in masses.iter().enumerate() {
        next.velocities[i] = add(
            next.velocities[i],
            scale(gradient[i], -0.5 * dt * ACCEL / mass),
        );
        let displacement = add(next.coordinates[i], scale(state.coordinates[i], -1.));
        if !finite(&next.coordinates[i]) || !finite(&next.velocities[i]) || norm2(displacement) > 1.
        {
            return Err(invalid(
                "unstable integration (>1 Å per step); previous checkpoint retained",
            ));
        }
    }
    next.gradient = gradient;
    next.potential_energy = energy;
    next.step += 1;
    *state = next;
    Ok(())
}
pub struct CpuSimulation<'a> {
    evaluator: EnergyEvaluator<'a>,
    masses: Vec<f64>,
    pub state: SimulationState,
}
impl<'a> CpuSimulation<'a> {
    pub fn into_owned(self) -> CpuSimulation<'static> {
        CpuSimulation {
            evaluator: self.evaluator.into_owned(),
            masses: self.masses,
            state: self.state,
        }
    }

    pub fn new(system: &'a ParameterizedSystem, protocol: SimulationProtocol) -> Result<Self> {
        Self::new_with_progress(system, protocol, |_, _, _, _, _| {})
    }

    /// Construct the implicit-solvent driver with the same progress contract
    /// as the explicit path.  This keeps the browser status useful for the
    /// default OBC2 workflow while preserving the compatibility constructor.
    pub fn new_with_progress<F>(
        system: &'a ParameterizedSystem,
        protocol: SimulationProtocol,
        progress: F,
    ) -> Result<Self>
    where
        F: FnMut(&str, usize, usize, f64, f64),
    {
        Self::construct(system, protocol, None, progress)
    }
    /// Finalize the same physical state after an external evaluator has
    /// completed the shared preparation minimizer. Constraint initialization
    /// and thermal velocities remain identical to the synchronous constructor.
    pub fn from_minimized(
        system: &'a ParameterizedSystem,
        protocol: SimulationProtocol,
        coordinates: Vec<Vec3>,
    ) -> Result<Self> {
        Self::construct(system, protocol, Some(coordinates), |_, _, _, _, _| {})
    }
    fn construct<F>(
        system: &'a ParameterizedSystem,
        protocol: SimulationProtocol,
        minimized: Option<Vec<Vec3>>,
        mut progress: F,
    ) -> Result<Self>
    where
        F: FnMut(&str, usize, usize, f64, f64),
    {
        protocol.validate()?;
        if system.atoms().is_empty()
            || system
                .atoms()
                .iter()
                .any(|a| !a.mass().is_finite() || a.mass() <= 0.)
        {
            return Err(invalid("finite positive masses required"));
        }
        if system.atoms().iter().any(|a| {
            matches!(
                system.residues()[a.residue_index()].name(),
                "HOH" | "WAT" | "TIP3"
            )
        }) {
            return Err(invalid(
                "initial dynamics supports implicit solvent, without explicit waters",
            ));
        }
        let evaluator = EnergyEvaluator::new(
            system,
            EnergyOptions {
                obc2: Some(Obc2Options::default()),
                ..Default::default()
            },
        )?;
        progress("parameterize", 1, 1, 0., 0.);
        let externally_minimized = minimized.is_some();
        let mut coordinates = minimized.unwrap_or_else(|| system.coordinates());
        if coordinates.len() != system.atom_count()
            || coordinates
                .iter()
                .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
        {
            return Err(invalid("invalid minimized coordinates"));
        }
        if protocol.minimization_iterations > 0 && !externally_minimized {
            let flat: Vec<_> = coordinates.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
            let mut optimizer = minimization::PreparationMinimizer::new(
                &flat,
                protocol.minimization_iterations,
                false,
            )?;
            while let Some(p) = optimizer.request() {
                let coords: Vec<_> = p
                    .chunks_exact(3)
                    .map(|p| Vec3 {
                        x: p[0],
                        y: p[1],
                        z: p[2],
                    })
                    .collect();
                let energy = evaluator.energy_and_gradient(&coords)?;
                let value = energy.total();
                let gradient = energy
                    .gradients
                    .unwrap()
                    .iter()
                    .flat_map(|p| [p.x, p.y, p.z])
                    .collect();
                let submitted = optimizer.submit(value, gradient)?;
                for event in submitted {
                    progress(
                        event.stage,
                        event.completed,
                        event.total,
                        event.energy,
                        event.max_gradient,
                    );
                }
            }
            let outcome = optimizer
                .outcome()
                .ok_or_else(|| invalid("L-BFGS stopped without a completed optimization state"))?;
            coordinates = outcome
                .chunks_exact(3)
                .map(|p| Vec3 {
                    x: p[0],
                    y: p[1],
                    z: p[2],
                })
                .collect();
        }
        let e = evaluator.energy_and_gradient(&coordinates)?;
        let potential_energy = e.total();
        let masses: Vec<_> = system.atoms().iter().map(|a| a.mass()).collect();
        let mut rng = protocol.seed;
        let velocities = masses
            .iter()
            .map(|m| random_velocity(&mut rng, (KB * protocol.temperature_k * ACCEL / m).sqrt()))
            .collect();
        let resident_rng = Some(resident_rng::ResidentThermostatRng::seeded(
            protocol.seed,
            masses.len(),
        ));
        let state = SimulationState {
            schema_version: 1,
            model_version: MODEL_VERSION.into(),
            system_fingerprint: fingerprint(system),
            protocol,
            step: 0,
            reference_coordinates: coordinates.clone(),
            coordinates,
            velocities,
            gradient: e.gradients.unwrap(),
            potential_energy,
            rng_state: rng,
            resident_rng,
            integrator_phase: "ready".into(),
            box_angstrom: [0., 0., 0.],
            virial_kcal_mol: 0.,
            pressure_bar: 0.,
            pressure_estimator: default_pressure_estimator(),
            barostat_rng: 0,
            barostat_volume_width: 0.,
            barostat_attempts: 0,
            barostat_accepts: 0,
            barostat_window_attempts: 0,
            barostat_window_accepts: 0,
            barostat_step_counter: 0,
            barostat_frozen: false,
            water_occupancy: None,
        };
        Ok(Self {
            evaluator,
            masses,
            state,
        })
    }
    pub fn restore(system: &'a ParameterizedSystem, state: SimulationState) -> Result<Self> {
        state.protocol.validate()?;
        if let Some(rng) = &state.resident_rng {
            rng.validate(system.atom_count())?;
        }
        if state.schema_version != 1
            || state.model_version != MODEL_VERSION
            || state.system_fingerprint != fingerprint(system)
            || state.integrator_phase != "ready"
            || state
                .coordinates
                .iter()
                .chain(&state.velocities)
                .chain(&state.gradient)
                .any(|v| !finite(v))
            || !state.potential_energy.is_finite()
            || state.reference_coordinates.len() != system.atom_count()
            || state.reference_coordinates.iter().any(|v| !finite(v))
            || state.coordinates.len() != system.atom_count()
            || state.velocities.len() != system.atom_count()
            || state.gradient.len() != system.atom_count()
            || state.step > state.protocol.total_steps()
        {
            return Err(invalid("incompatible or invalid simulation checkpoint"));
        }
        let evaluator = EnergyEvaluator::new(
            system,
            EnergyOptions {
                obc2: Some(Obc2Options::default()),
                ..Default::default()
            },
        )?;
        let reference = evaluator.energy_and_gradient(&state.coordinates)?;
        if (reference.total() - state.potential_energy).abs()
            > 1e-3 + 1e-4 * reference.total().abs()
            || reference
                .gradients
                .as_ref()
                .unwrap()
                .iter()
                .zip(&state.gradient)
                .any(|(a, b)| {
                    norm2(add(*a, scale(*b, -1.))).sqrt() > 1e-3 + 1e-3 * norm2(*a).sqrt()
                })
        {
            return Err(invalid("checkpoint forces do not match prepared chemistry"));
        }
        Ok(Self {
            evaluator,
            masses: system.atoms().iter().map(|a| a.mass()).collect(),
            state,
        })
    }
    pub fn advance(&mut self, steps: usize) -> Result<TrajectoryChunk> {
        let first_step = self.state.step;
        let end = (self.state.step + steps.min(100)).min(self.state.protocol.total_steps());
        let mut frames = Vec::new();
        while self.state.step < end {
            step_with(&mut self.state, &self.masses, |p| {
                let e = self.evaluator.energy_and_gradient(p)?;
                Ok((e.total(), e.gradients.unwrap()))
            })?;
            if self
                .state
                .step
                .is_multiple_of(self.state.protocol.save_every)
                || self.state.protocol.is_stage_boundary(self.state.step)
                || self.state.step == self.state.protocol.total_steps()
            {
                frames.push(self.frame());
            }
        }
        Ok(TrajectoryChunk {
            first_step,
            last_step: self.state.step,
            frames,
        })
    }
    pub fn frame(&self) -> TrajectoryFrame {
        frame_from_state(&self.state, &self.masses)
    }
}
pub fn frame_from_state(state: &SimulationState, masses: &[f64]) -> TrajectoryFrame {
    let kinetic_energy = masses
        .iter()
        .zip(&state.velocities)
        .map(|(m, v)| m * norm2(*v) / (2. * ACCEL))
        .sum::<f64>();
    TrajectoryFrame {
        step: state.step,
        time_ps: state.step as f64 * state.protocol.timestep_fs * 0.001,
        segment: state.protocol.stage_info(state.step).1,
        potential_energy: state.potential_energy,
        kinetic_energy,
        temperature_k: 2. * kinetic_energy / (3. * masses.len() as f64 * KB),
        coordinates: state.coordinates.clone(),
        box_angstrom: state.box_angstrom,
        pressure_bar: state.pressure_bar,
        pressure_estimator: state.pressure_estimator.clone(),
        density_g_ml: 0.,
    }
}

/// Materialize a periodic trajectory frame in the primary box while keeping
/// each covalent molecule intact.  Simulation coordinates remain unwrapped so
/// bonded terms and constraint solvers never see an artificial box split; this
/// conversion is only for viewers, exports, and frame-level analysis.
pub fn frame_from_state_wrapped(
    state: &SimulationState,
    masses: &[f64],
    molecules: &[Vec<usize>],
) -> TrajectoryFrame {
    let mut frame = frame_from_state(state, masses);
    if state
        .box_angstrom
        .iter()
        .all(|length| length.is_finite() && *length > 0.0)
    {
        if let Ok(box_vectors) = glysys_energy::pbc::BoxVectors::new(
            state.box_angstrom[0],
            state.box_angstrom[1],
            state.box_angstrom[2],
        ) {
            frame.coordinates = box_vectors.wrap_molecules(&frame.coordinates, molecules);
        }
    }
    frame
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_protocol_json_defaults_new_capability_metadata() {
        let protocol: SimulationProtocol = serde_json::from_str("{}").unwrap();
        assert_eq!(protocol.electrostatics, ElectrostaticsModel::ReactionField);
        assert_eq!(protocol.pressure_coupling, PressureCoupling::MonteCarlo);
        assert!(protocol.thermostat_groups.is_empty());
        assert!(protocol.pressure_tau_ps.is_none());
        assert!(protocol.com_mode.is_none());
        protocol.validate().unwrap();
    }

    #[test]
    fn native_schedule_bound_allows_long_runs_but_browser_bound_does_not() {
        let mut protocol = SimulationProtocol {
            equilibration_steps: 0,
            production_steps: MAX_BROWSER_STEPS + 1,
            ..Default::default()
        };
        protocol.validate().unwrap();
        assert!(protocol.validate_for_browser().is_err());
        protocol.production_steps = MAX_NATIVE_STEPS + 1;
        assert!(protocol.validate().is_err());
    }

    fn toy() -> SimulationState {
        SimulationState {
            schema_version: 1,
            model_version: MODEL_VERSION.into(),
            system_fingerprint: "toy".into(),
            protocol: SimulationProtocol {
                friction_per_ps: 0.,
                minimization_iterations: 0,
                ..Default::default()
            },
            step: 0,
            coordinates: vec![Vec3 {
                x: 0.1,
                y: 0.,
                z: 0.,
            }],
            reference_coordinates: vec![Vec3 {
                x: 0.1,
                y: 0.,
                z: 0.,
            }],
            velocities: vec![Vec3 {
                x: 0.,
                y: 0.,
                z: 0.,
            }],
            gradient: vec![Vec3 {
                x: 0.1,
                y: 0.,
                z: 0.,
            }],
            potential_energy: 0.005,
            rng_state: 0,
            resident_rng: None,
            integrator_phase: "ready".into(),
            box_angstrom: [0., 0., 0.],
            virial_kcal_mol: 0.,
            pressure_bar: 0.,
            pressure_estimator: default_pressure_estimator(),
            barostat_rng: 0,
            barostat_volume_width: 0.,
            barostat_attempts: 0,
            barostat_accepts: 0,
            barostat_window_attempts: 0,
            barostat_window_accepts: 0,
            barostat_step_counter: 0,
            barostat_frozen: false,
            water_occupancy: None,
        }
    }
    #[test]
    fn harmonic_nve_energy_and_checkpoint_replay() {
        let mut s = toy();
        let force = |p: &[Vec3]| Ok((0.5 * norm2(p[0]), vec![p[0]]));
        for _ in 0..10000 {
            step_with(&mut s, &[12.], force).unwrap();
            let total = s.potential_energy + 12. * norm2(s.velocities[0]) / (2. * ACCEL);
            assert!((total - 0.005).abs() < 1e-7);
        }
        let mut restored: SimulationState =
            serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        for _ in 0..50 {
            step_with(&mut s, &[12.], force).unwrap();
            step_with(&mut restored, &[12.], force).unwrap();
        }
        assert_eq!(s.coordinates, restored.coordinates);
        assert_eq!(s.velocities, restored.velocities);
    }
    #[test]
    fn failed_step_preserves_rng_and_coordinates() {
        let mut s = toy();
        let before = serde_json::to_string(&s).unwrap();
        assert!(step_with(&mut s, &[12.], |_| Err(invalid("device failure"))).is_err());
        assert_eq!(before, serde_json::to_string(&s).unwrap());
    }

    #[test]
    fn periodic_frame_wraps_a_whole_molecule() {
        let mut s = toy();
        s.box_angstrom = [10., 10., 10.];
        s.coordinates = vec![
            Vec3 {
                x: 10.2,
                y: -0.2,
                z: 5.,
            },
            Vec3 {
                x: 10.9,
                y: -0.1,
                z: 5.,
            },
        ];
        s.velocities.push(Vec3 {
            x: 0.,
            y: 0.,
            z: 0.,
        });
        s.gradient.push(Vec3 {
            x: 0.,
            y: 0.,
            z: 0.,
        });
        let frame = frame_from_state_wrapped(&s, &[12., 1.], &[vec![0, 1]]);
        assert!((frame.coordinates[0].x - 0.2).abs() < 1e-12);
        assert!((frame.coordinates[0].y - 9.8).abs() < 1e-12);
        // Both atoms receive the same translation; bonded geometry does not
        // get split at the periodic boundary.
        assert!((frame.coordinates[1].x - 0.9).abs() < 1e-12);
        assert!((frame.coordinates[1].y - 9.9).abs() < 1e-12);
    }

    #[test]
    fn thermostat_recovers_kinetic_temperature() {
        let mut s = toy();
        s.protocol.friction_per_ps = 10.;
        s.gradient[0] = Vec3 {
            x: 0.,
            y: 0.,
            z: 0.,
        };
        let mut sum = 0.;
        for i in 0..200000 {
            step_with(&mut s, &[12.], |_| {
                Ok((
                    0.,
                    vec![Vec3 {
                        x: 0.,
                        y: 0.,
                        z: 0.,
                    }],
                ))
            })
            .unwrap();
            if i >= 10000 {
                sum += 12. * norm2(s.velocities[0]) / (3. * ACCEL * KB);
            }
        }
        assert!((sum / 190000. - 300.).abs() < 20.);
    }
}
