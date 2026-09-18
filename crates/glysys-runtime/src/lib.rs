//! Backend-neutral execution sessions.
//!
//! The runtime is the only layer that decides whether a typed workload uses
//! the CPU reference implementation or a resident GPU evaluator. Physics
//! remains in `glysys-dynamics` and `glysys-energy`; this crate contains
//! lifecycle, fallback, diagnostics, and serialization contracts shared by
//! native and browser adapters.

use glysys::{ParameterizedSystem, Vec3};
use glysys_dynamics::explicit::ExplicitSimulation;
use glysys_dynamics::{
    CpuSimulation, Ensemble, SimulationProtocol, SimulationState, SolventModel, TrajectoryChunk,
};
use glysys_energy::hydration::{
    HydrationField, HydrationRequest, PhysicalProbe, ProbeScore, WaterPose,
};
use glysys_energy::pbc::NonbondedElectrostatics;
use glysys_energy::scoring::{
    EvaluationRequest, EvaluationResult, PoseBatch, PreparedEvaluator, PreparedScene, ScoreModel,
};
use glysys_gpu::hydration::ResidentWaterProbe;
use glysys_gpu::pbc::ResidentPbc;
use glysys_gpu::scoring::PreparedGpuEvaluator;
use glysys_gpu::steric::ResidentSteric;
use glysys_gpu::{GpuContext, GpuContextOptions, MemoryProfile};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const RUNTIME_SCHEMA_VERSION: u32 = 2;
pub const CHECKPOINT_SCHEMA_VERSION: u32 = 3;
pub const WORKER_PROTOCOL_VERSION: u32 = 2;

/// High-level backend preference. `Auto` may fall back to the CPU reference;
/// `Gpu` reports an unavailable/capacity error instead of silently changing
/// backend, while correctness checks remain independent of this selector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendPreference {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ValidationMode {
    #[default]
    None,
    CpuReference,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryPolicy {
    #[default]
    Adaptive,
    LowMemory,
    Explicit,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ExecutionOptions {
    pub backend: BackendPreference,
    pub memory_policy: MemoryPolicy,
    pub memory_budget_bytes: Option<u64>,
    pub cpu_thread_limit: Option<usize>,
    pub validation: ValidationMode,
    pub max_submission_steps: usize,
    pub cancellation_enabled: bool,
}

impl Default for ExecutionOptions {
    fn default() -> Self {
        Self {
            backend: BackendPreference::Auto,
            memory_policy: MemoryPolicy::Adaptive,
            memory_budget_bytes: None,
            cpu_thread_limit: None,
            validation: ValidationMode::None,
            max_submission_steps: 128,
            cancellation_enabled: true,
        }
    }
}

impl ExecutionOptions {
    pub fn gpu_memory_profile(&self) -> MemoryProfile {
        match self.memory_policy {
            MemoryPolicy::Adaptive => MemoryProfile::Adaptive,
            MemoryPolicy::LowMemory => MemoryProfile::LowMemory,
            MemoryPolicy::Explicit => {
                MemoryProfile::Explicit(self.memory_budget_bytes.unwrap_or(256 * 1024 * 1024))
            }
        }
    }

    pub fn context_options(&self, label: impl Into<String>) -> GpuContextOptions {
        GpuContextOptions {
            memory_profile: self.gpu_memory_profile(),
            label: label.into(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutionDiagnostics {
    pub requested_backend: BackendPreference,
    pub actual_backend: String,
    pub adapter_identity: Option<String>,
    pub cpu_variant: Option<String>,
    pub cpu_threads: Option<usize>,
    pub allocation_budget_bytes: u64,
    pub peak_allocation_bytes: u64,
    pub stage_timings_ms: BTreeMap<String, f64>,
    pub fallback_reason: Option<String>,
    pub validation_mode: ValidationMode,
    pub reference_call_count: u64,
}

impl ExecutionDiagnostics {
    fn cpu(options: &ExecutionOptions, fallback_reason: Option<String>) -> Self {
        Self {
            requested_backend: options.backend,
            actual_backend: "CPU".into(),
            cpu_variant: Some("reference".into()),
            cpu_threads: options.cpu_thread_limit,
            allocation_budget_bytes: options.gpu_memory_profile().budget(),
            fallback_reason,
            validation_mode: options.validation,
            ..Self::default()
        }
    }

    fn gpu(options: &ExecutionOptions, context: &GpuContext) -> Self {
        let info = context.adapter_info();
        let stats = context.allocation_stats();
        Self {
            requested_backend: options.backend,
            actual_backend: "GPU".into(),
            adapter_identity: Some(format!(
                "{} ({:?}/0x{:x}/0x{:x})",
                info.name, info.backend, info.vendor, info.device
            )),
            allocation_budget_bytes: stats.budget,
            peak_allocation_bytes: stats.peak_bytes,
            validation_mode: options.validation,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SessionErrorKind {
    InvalidInput,
    UnsupportedModel,
    UnavailableDevice,
    Capacity,
    DeviceLoss,
    Nonfinite,
    Cancellation,
    Storage,
    Output,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionError {
    pub kind: SessionErrorKind,
    pub message: String,
    pub cpu_fallback_valid: bool,
}

impl SessionError {
    pub fn new(
        kind: SessionErrorKind,
        message: impl Into<String>,
        cpu_fallback_valid: bool,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            cpu_fallback_valid,
        }
    }

    fn from_gpu(error: glysys_gpu::device::Error) -> Self {
        let kind = match error {
            glysys_gpu::device::Error::Unavailable(_) => SessionErrorKind::UnavailableDevice,
            glysys_gpu::device::Error::Capacity => SessionErrorKind::Capacity,
            glysys_gpu::device::Error::Nonfinite => SessionErrorKind::Nonfinite,
            glysys_gpu::device::Error::Input(_) => SessionErrorKind::InvalidInput,
            glysys_gpu::device::Error::Execution(ref message) => {
                let lower = message.to_ascii_lowercase();
                if lower.contains("device") || lower.contains("lost") {
                    SessionErrorKind::DeviceLoss
                } else {
                    SessionErrorKind::Output
                }
            }
        };
        let fallback = matches!(
            kind,
            SessionErrorKind::UnavailableDevice
                | SessionErrorKind::Capacity
                | SessionErrorKind::DeviceLoss
                | SessionErrorKind::Nonfinite
        );
        Self::new(kind, error.to_string(), fallback)
    }
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", format_args!("{:?}", self.kind), self.message)
    }
}

impl std::error::Error for SessionError {}

impl From<glysys_dynamics::Error> for SessionError {
    fn from(error: glysys_dynamics::Error) -> Self {
        Self::new(SessionErrorKind::InvalidInput, error.to_string(), false)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdvanceRequest {
    pub steps: usize,
    #[serde(default)]
    pub include_checkpoint: bool,
    #[serde(default)]
    pub max_frames: Option<usize>,
}

impl Default for AdvanceRequest {
    fn default() -> Self {
        Self {
            steps: 1,
            include_checkpoint: false,
            max_frames: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeCheckpoint {
    pub schema_version: u32,
    pub runtime_schema_version: u32,
    pub state: SimulationState,
    pub diagnostics: ExecutionDiagnostics,
}

impl RuntimeCheckpoint {
    pub fn from_state(state: SimulationState, diagnostics: ExecutionDiagnostics) -> Self {
        Self {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            runtime_schema_version: RUNTIME_SCHEMA_VERSION,
            state,
            diagnostics,
        }
    }

    pub fn decode(json: &str) -> Result<Self, SessionError> {
        // Inspect the envelope before deserializing the full state. This is
        // deliberate: a v1/v2 checkpoint contains a top-level SimulationState
        // and cannot be parsed as the v3 runtime envelope. Report a migration
        // error instead of leaking a serde "missing field" diagnostic.
        let value: serde_json::Value = serde_json::from_str(json).map_err(|error| {
            SessionError::new(
                SessionErrorKind::InvalidInput,
                format!("invalid runtime checkpoint: {error}"),
                false,
            )
        })?;
        let object = value.as_object().ok_or_else(|| {
            SessionError::new(
                SessionErrorKind::InvalidInput,
                format!("checkpoint requires runtime schema v{RUNTIME_SCHEMA_VERSION} and checkpoint schema v{CHECKPOINT_SCHEMA_VERSION}"),
                false,
            )
        })?;
        let schema_version = object
            .get("schemaVersion")
            .or_else(|| object.get("schema_version"))
            .and_then(serde_json::Value::as_u64);
        let runtime_schema_version = object
            .get("runtimeSchemaVersion")
            .or_else(|| object.get("runtime_schema_version"))
            .and_then(serde_json::Value::as_u64);
        if object.get("state").is_none()
            || schema_version != Some(CHECKPOINT_SCHEMA_VERSION as u64)
            || runtime_schema_version != Some(RUNTIME_SCHEMA_VERSION as u64)
        {
            return Err(SessionError::new(
                SessionErrorKind::InvalidInput,
                format!(
                    "unsupported checkpoint schema (found {:?}, runtime {:?}); GlySys requires v{} (runtime v{})",
                    schema_version,
                    runtime_schema_version,
                    CHECKPOINT_SCHEMA_VERSION,
                    RUNTIME_SCHEMA_VERSION
                ),
                false,
            ));
        }
        let checkpoint: Self = serde_json::from_value(value).map_err(|error| {
            SessionError::new(
                SessionErrorKind::InvalidInput,
                format!("invalid runtime checkpoint: {error}"),
                false,
            )
        })?;
        Ok(checkpoint)
    }

    pub fn encode(&self) -> Result<String, SessionError> {
        serde_json::to_string(self)
            .map_err(|error| SessionError::new(SessionErrorKind::Output, error.to_string(), false))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdvanceResult {
    pub chunk: TrajectoryChunk,
    pub checkpoint: Option<RuntimeCheckpoint>,
    pub diagnostics: ExecutionDiagnostics,
}

enum SimulationDriver {
    ExplicitCpu(ExplicitSimulation<'static>),
    ImplicitCpu(CpuSimulation<'static>),
    ExplicitGpu {
        simulation: ExplicitSimulation<'static>,
        gpu: ResidentPbc,
    },
}

impl SimulationDriver {
    fn state(&self) -> &SimulationState {
        match self {
            Self::ExplicitCpu(sim) => &sim.state,
            Self::ImplicitCpu(sim) => &sim.state,
            Self::ExplicitGpu { simulation, .. } => &simulation.state,
        }
    }
}

fn restore_cpu_driver(
    system: &ParameterizedSystem,
    state: &SimulationState,
) -> Result<SimulationDriver, SessionError> {
    if state.protocol.solvent == SolventModel::Explicit {
        Ok(SimulationDriver::ExplicitCpu(
            ExplicitSimulation::restore(system, state.clone())?.into_owned(),
        ))
    } else {
        Ok(SimulationDriver::ImplicitCpu(
            CpuSimulation::restore(system, state.clone())?.into_owned(),
        ))
    }
}

/// Shared high-level simulation session. The CPU core is synchronous; the
/// async method is the adapter boundary needed for resident GPU submission.
pub struct SimulationSession {
    system: ParameterizedSystem,
    driver: SimulationDriver,
    options: ExecutionOptions,
    diagnostics: ExecutionDiagnostics,
    cancelled: bool,
    gpu_advances: u64,
    cpu_advances: u64,
}

fn gpu_compatible(protocol: &SimulationProtocol) -> bool {
    protocol.solvent == SolventModel::Explicit
        && protocol.constraints == glysys_dynamics::ConstraintModel::Settle
        && protocol.thermostat == glysys_dynamics::Thermostat::Langevin
        && !protocol.has_npt()
        && protocol.restraint_force == 0.0
        && !protocol.dispersion_correction
}

fn gpu_total(
    result: &glysys_gpu::pbc::EnergyResult,
) -> Result<(f64, Vec<Vec3>, f64), SessionError> {
    let gradients = result
        .gradients
        .as_ref()
        .ok_or_else(|| SessionError::new(SessionErrorKind::Output, "GPU omitted gradients", true))?
        .iter()
        .map(|g| Vec3 {
            x: g[0] as f64,
            y: g[1] as f64,
            z: g[2] as f64,
        })
        .collect::<Vec<_>>();
    let virial = result
        .virial
        .ok_or_else(|| SessionError::new(SessionErrorKind::Output, "GPU omitted virial", true))?;
    let total = result.lj
        + result.rf
        + result.bonds
        + result.angles
        + result.proper_torsions
        + result.improper_torsions
        + result.dispersion_correction;
    if !total.is_finite() || !virial.is_finite() {
        return Err(SessionError::new(
            SessionErrorKind::Nonfinite,
            "GPU returned nonfinite energy",
            true,
        ));
    }
    Ok((total, gradients, virial))
}

impl SimulationSession {
    pub async fn new(
        system: ParameterizedSystem,
        protocol: SimulationProtocol,
        options: ExecutionOptions,
    ) -> Result<Self, SessionError> {
        Self::new_with_context(system, protocol, options, None).await
    }

    /// Construct a simulation using a coordinator-owned GPU context. The
    /// context is optional for native callers, but browser workers pass their
    /// one per-run handle so typed sessions share one adapter/device/queue.
    pub async fn new_with_context(
        system: ParameterizedSystem,
        protocol: SimulationProtocol,
        options: ExecutionOptions,
        shared_context: Option<GpuContext>,
    ) -> Result<Self, SessionError> {
        protocol.validate_for_native()?;
        let requested = options.backend;
        let cpu_driver = || -> Result<SimulationDriver, SessionError> {
            if protocol.solvent == SolventModel::Explicit {
                Ok(SimulationDriver::ExplicitCpu(
                    ExplicitSimulation::new(&system, protocol.clone())?.into_owned(),
                ))
            } else {
                Ok(SimulationDriver::ImplicitCpu(
                    CpuSimulation::new(&system, protocol.clone())?.into_owned(),
                ))
            }
        };
        if requested == BackendPreference::Cpu || !gpu_compatible(&protocol) {
            let reason = (requested != BackendPreference::Cpu).then(|| {
                "requested GPU is unavailable for this protocol; using the CPU reference session".into()
            });
            let driver = cpu_driver()?;
            return Ok(Self {
                system,
                driver,
                diagnostics: ExecutionDiagnostics::cpu(&options, reason),
                options,
                cancelled: false,
                gpu_advances: 0,
                cpu_advances: 0,
            });
        }

        // Build the CPU state once because it owns the validated preparation
        // and initial forces. The resident evaluator receives the same state;
        // no CPU reference calls occur during normal GPU advancement.
        let cpu = cpu_driver()?;
        let SimulationDriver::ExplicitCpu(simulation) = cpu else {
            unreachable!("GPU compatibility implies explicit solvent");
        };
        let cutoff = protocol.cutoff_angstrom.unwrap_or(9.0);
        let electro = NonbondedElectrostatics::ReactionField {
            cutoff_angstrom: cutoff,
            solvent_dielectric: protocol.rf_dielectric.unwrap_or(78.5),
        };
        let packing = glysys_gpu::pbc::PbcPacking::new(&system, cutoff, 1.5)
            .map_err(|e| SessionError::new(SessionErrorKind::InvalidInput, e.to_string(), false))?;
        let max_pairs = simulation
            .pair_count()
            .saturating_mul(2)
            .saturating_add(1024)
            .max(4096)
            .min(u32::MAX as usize) as u32;
        let context_result = match shared_context {
            Some(context) => Ok(context),
            None => GpuContext::new(options.context_options("GlySys simulation")).await,
        };
        let context = match context_result {
            Ok(context) => context,
            Err(error) if requested == BackendPreference::Auto => {
                return Ok(Self {
                    system,
                    driver: SimulationDriver::ExplicitCpu(simulation),
                    diagnostics: ExecutionDiagnostics::cpu(
                        &options,
                        Some(format!("GPU context unavailable: {error}")),
                    ),
                    options,
                    cancelled: false,
                    gpu_advances: 0,
                    cpu_advances: 0,
                });
            }
            Err(error) => return Err(SessionError::from_gpu(error)),
        };
        let mut gpu = match context.create_pbc(&packing, &electro, max_pairs).await {
            Ok(gpu) => gpu,
            Err(error) if requested == BackendPreference::Auto => {
                return Ok(Self {
                    system,
                    driver: SimulationDriver::ExplicitCpu(simulation),
                    diagnostics: ExecutionDiagnostics::cpu(&options, Some(error.to_string())),
                    options,
                    cancelled: false,
                    gpu_advances: 0,
                    cpu_advances: 0,
                });
            }
            Err(error) => return Err(SessionError::from_gpu(error)),
        };
        let box_xyz = simulation.state.box_angstrom.map(|value| value as f32);
        if let Err(error) = gpu.initialize_dynamics(
            &simulation.state.coordinates,
            &simulation.state.velocities,
            box_xyz,
            (protocol.timestep_fs * 0.001) as f32,
        ) {
            let error = SessionError::from_gpu(error);
            if requested == BackendPreference::Auto && error.cpu_fallback_valid {
                return Ok(Self {
                    system,
                    driver: SimulationDriver::ExplicitCpu(simulation),
                    diagnostics: ExecutionDiagnostics::cpu(&options, Some(error.message)),
                    options,
                    cancelled: false,
                    gpu_advances: 0,
                    cpu_advances: 0,
                });
            }
            return Err(error);
        }
        let initial = match gpu.energy_and_forces(true).await {
            Ok(initial) => initial,
            Err(error) if requested == BackendPreference::Auto => {
                return Ok(Self {
                    system,
                    driver: SimulationDriver::ExplicitCpu(simulation),
                    diagnostics: ExecutionDiagnostics::cpu(
                        &options,
                        Some(SessionError::from_gpu(error).message),
                    ),
                    options,
                    cancelled: false,
                    gpu_advances: 0,
                    cpu_advances: 0,
                });
            }
            Err(error) => return Err(SessionError::from_gpu(error)),
        };
        let (energy, gradients, virial) = match gpu_total(&initial) {
            Ok(values) => values,
            Err(error) if requested == BackendPreference::Auto && error.cpu_fallback_valid => {
                return Ok(Self {
                    system,
                    driver: SimulationDriver::ExplicitCpu(simulation),
                    diagnostics: ExecutionDiagnostics::cpu(&options, Some(error.message)),
                    options,
                    cancelled: false,
                    gpu_advances: 0,
                    cpu_advances: 0,
                });
            }
            Err(error) => return Err(error),
        };
        let mut simulation = simulation;
        let coords = simulation.state.coordinates.clone();
        let velocities = simulation.state.velocities.clone();
        simulation.install_evaluated_state(coords, velocities, 0, energy, gradients, virial)?;
        Ok(Self {
            system,
            driver: SimulationDriver::ExplicitGpu { simulation, gpu },
            diagnostics: ExecutionDiagnostics::gpu(&options, &context),
            options,
            cancelled: false,
            gpu_advances: 1,
            cpu_advances: 0,
        })
    }

    pub async fn from_checkpoint(
        system: ParameterizedSystem,
        checkpoint: RuntimeCheckpoint,
        options: ExecutionOptions,
    ) -> Result<Self, SessionError> {
        Self::from_checkpoint_with_context(system, checkpoint, options, None).await
    }

    /// Restore a checkpoint on an existing coordinator context when supplied.
    /// Device buffers are always recreated from the canonical host state.
    pub async fn from_checkpoint_with_context(
        system: ParameterizedSystem,
        checkpoint: RuntimeCheckpoint,
        options: ExecutionOptions,
        shared_context: Option<GpuContext>,
    ) -> Result<Self, SessionError> {
        if checkpoint.schema_version != CHECKPOINT_SCHEMA_VERSION
            || checkpoint.runtime_schema_version != RUNTIME_SCHEMA_VERSION
        {
            return Err(SessionError::new(
                SessionErrorKind::InvalidInput,
                "checkpoint schema is not supported by runtime v2",
                false,
            ));
        }
        let state = checkpoint.state;
        let protocol = state.protocol.clone();
        if options.backend == BackendPreference::Cpu
            || protocol.solvent != SolventModel::Explicit
            || !gpu_compatible(&protocol)
        {
            let reason = (options.backend != BackendPreference::Cpu).then(|| {
                "checkpoint restored on the CPU reference session because this protocol is not GPU-compatible".into()
            });
            let driver = restore_cpu_driver(&system, &state)?;
            return Ok(Self {
                system,
                driver,
                options: options.clone(),
                diagnostics: ExecutionDiagnostics::cpu(&options, reason),
                cancelled: false,
                gpu_advances: 0,
                cpu_advances: 0,
            });
        }

        // Recreate GPU resources from the canonical host checkpoint. GPU
        // buffers are deliberately not part of the serialized format, so a
        // restart never depends on the device that wrote the checkpoint.
        let cpu = restore_cpu_driver(&system, &state)?;
        let SimulationDriver::ExplicitCpu(simulation) = cpu else {
            unreachable!("GPU-compatible checkpoint must be explicit solvent");
        };
        let cutoff = protocol.cutoff_angstrom.unwrap_or(9.0);
        let electro = NonbondedElectrostatics::ReactionField {
            cutoff_angstrom: cutoff,
            solvent_dielectric: protocol.rf_dielectric.unwrap_or(78.5),
        };
        let packing =
            glysys_gpu::pbc::PbcPacking::new_with_box(&system, state.box_angstrom, cutoff, 1.5)
                .map_err(|e| {
                    SessionError::new(SessionErrorKind::InvalidInput, e.to_string(), false)
                })?;
        let max_pairs = simulation
            .pair_count()
            .saturating_mul(2)
            .saturating_add(1024)
            .max(4096)
            .min(u32::MAX as usize) as u32;
        let context_result = match shared_context {
            Some(context) => Ok(context),
            None => GpuContext::new(options.context_options("GlySys simulation restart")).await,
        };
        let context = match context_result {
            Ok(context) => context,
            Err(error) if options.backend == BackendPreference::Auto => {
                return Ok(Self {
                    system,
                    driver: SimulationDriver::ExplicitCpu(simulation),
                    options: options.clone(),
                    diagnostics: ExecutionDiagnostics::cpu(
                        &options,
                        Some(format!(
                            "GPU context unavailable while restoring checkpoint: {error}"
                        )),
                    ),
                    cancelled: false,
                    gpu_advances: 0,
                    cpu_advances: 0,
                });
            }
            Err(error) => return Err(SessionError::from_gpu(error)),
        };
        let mut gpu = match context.create_pbc(&packing, &electro, max_pairs).await {
            Ok(gpu) => gpu,
            Err(error) if options.backend == BackendPreference::Auto => {
                return Ok(Self {
                    system,
                    driver: SimulationDriver::ExplicitCpu(simulation),
                    options: options.clone(),
                    diagnostics: ExecutionDiagnostics::cpu(&options, Some(error.to_string())),
                    cancelled: false,
                    gpu_advances: 0,
                    cpu_advances: 0,
                });
            }
            Err(error) => return Err(SessionError::from_gpu(error)),
        };
        let box_xyz = state.box_angstrom.map(|value| value as f32);
        let dt_ps = (protocol.timestep_fs * 0.001) as f32;
        if let Some(rng) = state.resident_rng.as_ref() {
            if let Err(error) = gpu.set_timestep(dt_ps) {
                let error = SessionError::from_gpu(error);
                if options.backend == BackendPreference::Auto && error.cpu_fallback_valid {
                    return Ok(Self {
                        system,
                        driver: SimulationDriver::ExplicitCpu(simulation),
                        options: options.clone(),
                        diagnostics: ExecutionDiagnostics::cpu(&options, Some(error.message)),
                        cancelled: false,
                        gpu_advances: 0,
                        cpu_advances: 0,
                    });
                }
                return Err(error);
            }
            if let Err(error) = gpu.set_dynamics_state(
                &glysys_gpu::pbc::ResidentDynamicsState {
                    coordinates: state.coordinates.clone(),
                    velocities: state.velocities.clone(),
                    rng_words: rng.words.clone(),
                },
                box_xyz,
                true,
            ) {
                let error = SessionError::from_gpu(error);
                if options.backend == BackendPreference::Auto && error.cpu_fallback_valid {
                    return Ok(Self {
                        system,
                        driver: SimulationDriver::ExplicitCpu(simulation),
                        options: options.clone(),
                        diagnostics: ExecutionDiagnostics::cpu(&options, Some(error.message)),
                        cancelled: false,
                        gpu_advances: 0,
                        cpu_advances: 0,
                    });
                }
                return Err(error);
            }
        } else {
            if let Err(error) =
                gpu.initialize_dynamics(&state.coordinates, &state.velocities, box_xyz, dt_ps)
            {
                let error = SessionError::from_gpu(error);
                if options.backend == BackendPreference::Auto && error.cpu_fallback_valid {
                    return Ok(Self {
                        system,
                        driver: SimulationDriver::ExplicitCpu(simulation),
                        options: options.clone(),
                        diagnostics: ExecutionDiagnostics::cpu(&options, Some(error.message)),
                        cancelled: false,
                        gpu_advances: 0,
                        cpu_advances: 0,
                    });
                }
                return Err(error);
            }
        }
        let initial = match gpu.energy_and_forces(true).await {
            Ok(initial) => initial,
            Err(error) if options.backend == BackendPreference::Auto => {
                return Ok(Self {
                    system,
                    driver: SimulationDriver::ExplicitCpu(simulation),
                    options: options.clone(),
                    diagnostics: ExecutionDiagnostics::cpu(&options, Some(error.to_string())),
                    cancelled: false,
                    gpu_advances: 0,
                    cpu_advances: 0,
                });
            }
            Err(error) => return Err(SessionError::from_gpu(error)),
        };
        let (energy, gradients, virial) = match gpu_total(&initial) {
            Ok(values) => values,
            Err(error)
                if options.backend == BackendPreference::Auto && error.cpu_fallback_valid =>
            {
                return Ok(Self {
                    system,
                    driver: SimulationDriver::ExplicitCpu(simulation),
                    options: options.clone(),
                    diagnostics: ExecutionDiagnostics::cpu(&options, Some(error.message)),
                    cancelled: false,
                    gpu_advances: 0,
                    cpu_advances: 0,
                });
            }
            Err(error) => return Err(error),
        };
        let mut simulation = simulation;
        simulation.install_evaluated_state(
            state.coordinates.clone(),
            state.velocities.clone(),
            state.step,
            energy,
            gradients,
            virial,
        )?;
        Ok(Self {
            system,
            driver: SimulationDriver::ExplicitGpu { simulation, gpu },
            diagnostics: ExecutionDiagnostics::gpu(&options, &context),
            options,
            cancelled: false,
            gpu_advances: 1,
            cpu_advances: 0,
        })
    }

    pub fn state(&self) -> &SimulationState {
        self.driver.state()
    }

    pub fn checkpoint(&self) -> Result<RuntimeCheckpoint, SessionError> {
        Ok(RuntimeCheckpoint::from_state(
            self.state().clone(),
            self.diagnostics.clone(),
        ))
    }

    pub fn diagnostics(&self) -> &ExecutionDiagnostics {
        &self.diagnostics
    }

    /// Roll back a failed resident GPU job to its last host state and switch
    /// only this typed session to the CPU reference implementation. The
    /// caller decides whether the error is recoverable (normally `Auto`).
    pub fn fallback_to_cpu(&mut self, reason: impl Into<String>) -> Result<(), SessionError> {
        if matches!(
            &self.driver,
            SimulationDriver::ExplicitCpu(_) | SimulationDriver::ImplicitCpu(_)
        ) {
            return Ok(());
        }
        let state = self.state().clone();
        let simulation = if state.protocol.solvent == SolventModel::Explicit {
            SimulationDriver::ExplicitCpu(
                ExplicitSimulation::restore(&self.system, state)?.into_owned(),
            )
        } else {
            SimulationDriver::ImplicitCpu(CpuSimulation::restore(&self.system, state)?.into_owned())
        };
        self.driver = simulation;
        self.diagnostics.actual_backend = if self.gpu_advances > 0 {
            "Mixed".into()
        } else {
            "CPU".into()
        };
        self.diagnostics.cpu_variant = Some("reference".into());
        self.diagnostics.fallback_reason = Some(reason.into());
        Ok(())
    }

    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    fn advance_cpu_steps(&mut self, steps: usize) -> Result<TrajectoryChunk, SessionError> {
        match &mut self.driver {
            SimulationDriver::ExplicitCpu(sim) => {
                let result = sim.advance(steps).map_err(Into::into);
                if result.is_ok() {
                    self.cpu_advances = self.cpu_advances.saturating_add(1);
                }
                result
            }
            SimulationDriver::ImplicitCpu(sim) => {
                let result = sim.advance(steps).map_err(Into::into);
                if result.is_ok() {
                    self.cpu_advances = self.cpu_advances.saturating_add(1);
                }
                result
            }
            SimulationDriver::ExplicitGpu { .. } => Err(SessionError::new(
                SessionErrorKind::Output,
                "CPU advance requested while the GPU driver is active",
                false,
            )),
        }
    }

    fn gpu_step_count(&self, requested: usize) -> usize {
        let SimulationDriver::ExplicitGpu { simulation, .. } = &self.driver else {
            return requested;
        };
        let stage_index = simulation
            .state
            .protocol
            .stage_info(simulation.state.step)
            .0;
        let (_, _, _, _, local_step) = simulation.state.protocol.stage_info(simulation.state.step);
        let stage = &simulation.state.protocol.execution_stages()[stage_index];
        requested
            .min(stage.steps.saturating_sub(local_step).max(1))
            .min(128)
    }

    async fn advance_gpu_steps(&mut self, count: usize) -> Result<TrajectoryChunk, SessionError> {
        let SimulationDriver::ExplicitGpu { simulation, gpu } = &mut self.driver else {
            return Err(SessionError::new(
                SessionErrorKind::Output,
                "GPU advance requested while the CPU driver is active",
                false,
            ));
        };
        let first = simulation.state.step;
        let (_, _, ensemble, _, _) = simulation.state.protocol.stage_info(first);
        match ensemble {
            Ensemble::Nve => gpu.dynamics_steps(count).await,
            Ensemble::Nvt => {
                gpu.dynamics_steps_nvt(
                    count,
                    simulation.state.protocol.temperature_k as f32,
                    simulation.state.protocol.friction_per_ps as f32,
                )
                .await
            }
            Ensemble::Npt => Err(glysys_gpu::device::Error::Input(
                "GPU NPT is not enabled in this cleanup",
            )),
        }
        .map_err(SessionError::from_gpu)?;
        let box_xyz = simulation.state.box_angstrom.map(|value| value as f32);
        let (resident, energy) = gpu
            .read_dynamics_snapshot(box_xyz)
            .await
            .map_err(SessionError::from_gpu)?;
        let (potential, gradients, virial) = gpu_total(&energy)?;
        let next_step = first + count;
        simulation.install_evaluated_state(
            resident.coordinates,
            resident.velocities,
            next_step,
            potential,
            gradients,
            virial,
        )?;
        if let Some(rng) = &mut simulation.state.resident_rng {
            if resident.rng_words.len() == rng.words.len() {
                rng.words.clone_from(&resident.rng_words);
            }
        }
        self.gpu_advances = self.gpu_advances.saturating_add(1);
        let save = next_step.is_multiple_of(simulation.state.protocol.save_every)
            || simulation.state.protocol.is_stage_boundary(next_step)
            || next_step == simulation.state.protocol.total_steps();
        Ok(TrajectoryChunk {
            first_step: first,
            last_step: next_step,
            frames: save.then(|| simulation.frame()).into_iter().collect(),
        })
    }

    pub async fn advance(
        &mut self,
        request: AdvanceRequest,
    ) -> Result<AdvanceResult, SessionError> {
        if self.cancelled {
            return Err(SessionError::new(
                SessionErrorKind::Cancellation,
                "simulation cancelled",
                false,
            ));
        }
        if request.steps == 0 {
            return Err(SessionError::new(
                SessionErrorKind::InvalidInput,
                "advance steps must be positive",
                false,
            ));
        }
        let limit = request.steps.min(self.options.max_submission_steps.max(1));
        let gpu_active = matches!(&self.driver, SimulationDriver::ExplicitGpu { .. });
        let chunk = if !gpu_active {
            self.advance_cpu_steps(limit)?
        } else {
            let gpu_steps = self.gpu_step_count(limit);
            match self.advance_gpu_steps(gpu_steps).await {
                Ok(chunk) => chunk,
                Err(error)
                    if self.options.backend == BackendPreference::Auto
                        && error.cpu_fallback_valid =>
                {
                    // The resident device may have partially executed the
                    // failed dispatch.  `fallback_to_cpu` restores from the
                    // last host-committed state, so retrying these bounded
                    // steps cannot duplicate or skip stochastic work.
                    let reason = error.message.clone();
                    self.fallback_to_cpu(reason)?;
                    self.advance_cpu_steps(gpu_steps)?
                }
                Err(error) => return Err(error),
            }
        };
        let mut chunk = chunk;
        if let Some(max_frames) = request.max_frames {
            chunk.frames.truncate(max_frames);
        }
        let checkpoint = request
            .include_checkpoint
            .then(|| self.checkpoint())
            .transpose()?;
        Ok(AdvanceResult {
            chunk,
            checkpoint,
            diagnostics: self.diagnostics.clone(),
        })
    }
}

/// Preparation lifecycle wrapper. The actual chemistry/minimization remains
/// in `glysys-dynamics`; this typed session gives all adapters one ownership
/// and diagnostics contract without exposing UI or filesystem types.
pub struct PreparationSession {
    system: ParameterizedSystem,
    diagnostics: ExecutionDiagnostics,
    context: Option<GpuContext>,
}

impl PreparationSession {
    pub fn new(system: ParameterizedSystem, options: ExecutionOptions) -> Self {
        Self::new_with_context(system, options, None)
    }

    pub fn new_with_context(
        system: ParameterizedSystem,
        options: ExecutionOptions,
        context: Option<GpuContext>,
    ) -> Self {
        Self {
            system,
            diagnostics: ExecutionDiagnostics::cpu(&options, None),
            context,
        }
    }

    pub fn system(&self) -> &ParameterizedSystem {
        &self.system
    }

    pub fn into_system(self) -> ParameterizedSystem {
        self.system
    }

    pub fn diagnostics(&self) -> &ExecutionDiagnostics {
        &self.diagnostics
    }

    pub fn take_context(&mut self) -> Option<GpuContext> {
        self.context.take()
    }
}

enum HydrationDriver {
    Cpu(PhysicalProbe),
    Gpu {
        probe: PhysicalProbe,
        resident: ResidentWaterProbe,
    },
}

/// Whole-surface hydration session. Probe/GC scientific behavior is still
/// implemented by `glysys-energy`; this adapter only owns backend selection
/// and resident probe resources.
pub struct HydrationSession {
    driver: HydrationDriver,
    diagnostics: ExecutionDiagnostics,
}

impl HydrationSession {
    pub async fn new(
        system: &ParameterizedSystem,
        options: ExecutionOptions,
    ) -> Result<Self, SessionError> {
        Self::new_with_context(system, options, None).await
    }

    pub async fn new_with_context(
        system: &ParameterizedSystem,
        options: ExecutionOptions,
        shared_context: Option<GpuContext>,
    ) -> Result<Self, SessionError> {
        let probe = PhysicalProbe::new(system)
            .map_err(|e| SessionError::new(SessionErrorKind::InvalidInput, e.to_string(), false))?;
        if options.backend == BackendPreference::Cpu {
            return Ok(Self {
                driver: HydrationDriver::Cpu(probe),
                diagnostics: ExecutionDiagnostics::cpu(&options, None),
            });
        }
        let context_result = match shared_context {
            Some(context) => Ok(context),
            None => GpuContext::new(options.context_options("GlySys hydration")).await,
        };
        let context = match context_result {
            Ok(context) => context,
            Err(error) if options.backend == BackendPreference::Auto => {
                return Ok(Self {
                    driver: HydrationDriver::Cpu(probe),
                    diagnostics: ExecutionDiagnostics::cpu(&options, Some(error.to_string())),
                });
            }
            Err(error) => return Err(SessionError::from_gpu(error)),
        };
        let resident = match context.create_hydration(&probe, 4096).await {
            Ok(resident) => resident,
            Err(error) if options.backend == BackendPreference::Auto => {
                return Ok(Self {
                    driver: HydrationDriver::Cpu(probe),
                    diagnostics: ExecutionDiagnostics::cpu(&options, Some(error.to_string())),
                });
            }
            Err(error) => return Err(SessionError::from_gpu(error)),
        };
        Ok(Self {
            driver: HydrationDriver::Gpu { probe, resident },
            diagnostics: ExecutionDiagnostics::gpu(&options, &context),
        })
    }

    pub fn diagnostics(&self) -> &ExecutionDiagnostics {
        &self.diagnostics
    }

    pub fn capacity(&self) -> usize {
        match &self.driver {
            HydrationDriver::Cpu(_) => 1,
            HydrationDriver::Gpu { resident, .. } => resident.capacity.max(1) as usize,
        }
    }

    /// Evaluate a bounded batch of rigid probe poses. This is the shared
    /// hydration primitive used by browser tiling and native callers; it
    /// keeps backend selection and fallback in the typed session instead of
    /// making each adapter own a resident water-probe evaluator.
    pub async fn evaluate_poses(
        &mut self,
        poses: &[WaterPose],
        cutoff: Option<f64>,
    ) -> Result<Vec<Option<ProbeScore>>, SessionError> {
        if poses.is_empty() {
            return Ok(Vec::new());
        }
        match &mut self.driver {
            HydrationDriver::Cpu(probe) => {
                self.diagnostics.reference_call_count = self
                    .diagnostics
                    .reference_call_count
                    .saturating_add(poses.len() as u64);
                Ok(poses
                    .iter()
                    .map(|pose| probe.score(*pose, cutoff))
                    .collect())
            }
            HydrationDriver::Gpu { probe, resident } => {
                let capacity = resident.capacity.max(1);
                let mut values = Vec::with_capacity(poses.len());
                for chunk in poses.chunks(capacity) {
                    match resident.evaluate(chunk, cutoff).await {
                        Ok(scores) => values.extend(scores),
                        Err(error)
                            if self.diagnostics.requested_backend == BackendPreference::Auto =>
                        {
                            let fallback_probe = probe.clone();
                            let error = SessionError::from_gpu(error);
                            self.driver = HydrationDriver::Cpu(fallback_probe);
                            self.diagnostics.actual_backend = if values.is_empty() {
                                "CPU".into()
                            } else {
                                "Mixed".into()
                            };
                            self.diagnostics.cpu_variant = Some("reference".into());
                            self.diagnostics.fallback_reason = Some(error.message);
                            if let HydrationDriver::Cpu(probe) = &self.driver {
                                self.diagnostics.reference_call_count = self
                                    .diagnostics
                                    .reference_call_count
                                    .saturating_add(poses.len() as u64);
                                return Ok(poses
                                    .iter()
                                    .map(|pose| probe.score(*pose, cutoff))
                                    .collect());
                            }
                            unreachable!("hydration fallback did not install CPU probe");
                        }
                        Err(error) => return Err(SessionError::from_gpu(error)),
                    }
                }
                Ok(values)
            }
        }
    }

    pub async fn predict(
        &mut self,
        request: &HydrationRequest,
    ) -> Result<HydrationField, SessionError> {
        let dims = PhysicalProbe::dimensions(request)
            .map_err(|e| SessionError::new(SessionErrorKind::InvalidInput, e.to_string(), false))?;
        let points = dims.iter().product::<usize>();
        let mut best = vec![None; points];
        for index in 0..points {
            let poses = PhysicalProbe::poses(request, dims, index);
            let scores = self.evaluate_poses(&poses, request.cutoff).await?;
            for (pose, score) in poses.into_iter().zip(scores.into_iter().flatten()) {
                if best[index]
                    .as_ref()
                    .is_none_or(|(_, previous): &(WaterPose, ProbeScore)| {
                        score.total() < previous.total()
                    })
                {
                    best[index] = Some((pose, score));
                }
            }
        }
        let backend = self.diagnostics.actual_backend.clone();
        let field = if glysys_energy::hydration::is_gc_request(request) {
            let tiles = vec![glysys_energy::hydration::TileGrid {
                offset: [0, 0, 0],
                dimensions: dims,
                energies: best
                    .iter()
                    .map(|v| v.as_ref().map(|(_, score)| score.total()))
                    .collect(),
            }];
            let candidates = best.iter().filter_map(|v| *v).collect();
            match &self.driver {
                HydrationDriver::Cpu(probe) | HydrationDriver::Gpu { probe, .. } => {
                    probe.finish_tiled_gc(request, dims, tiles, candidates, &backend)
                }
            }
        } else {
            Ok(match &self.driver {
                HydrationDriver::Cpu(probe) | HydrationDriver::Gpu { probe, .. } => {
                    probe.finish(request, dims, best, &backend)
                }
            })
        }
        .map_err(|e| SessionError::new(SessionErrorKind::InvalidInput, e.to_string(), false))?;
        Ok(field)
    }
}

/// Scoring session backed by the versioned scoring contracts. CPU and GPU
/// evaluators expose the same request/result types; unsupported GPU terms are
/// reported as a scoped capability failure.
pub struct ScoringSession {
    cpu: PreparedEvaluator,
    gpu: Option<PreparedGpuEvaluator>,
    diagnostics: ExecutionDiagnostics,
    gpu_batches: u64,
}

impl ScoringSession {
    pub async fn new(
        scene: PreparedScene,
        model: ScoreModel,
        options: ExecutionOptions,
    ) -> Result<Self, SessionError> {
        Self::new_with_context(scene, model, options, None).await
    }

    pub async fn new_with_context(
        scene: PreparedScene,
        model: ScoreModel,
        options: ExecutionOptions,
        shared_context: Option<GpuContext>,
    ) -> Result<Self, SessionError> {
        let cpu = PreparedEvaluator::new(scene.clone(), model.clone())
            .map_err(|e| SessionError::new(SessionErrorKind::InvalidInput, e.to_string(), false))?;
        if options.backend == BackendPreference::Cpu {
            return Ok(Self {
                cpu,
                gpu: None,
                diagnostics: ExecutionDiagnostics::cpu(&options, None),
                gpu_batches: 0,
            });
        }
        let context_result = match shared_context {
            Some(context) => Ok(context),
            None => GpuContext::new(options.context_options("GlySys scoring")).await,
        };
        let context = match context_result {
            Ok(context) => context,
            Err(error) if options.backend == BackendPreference::Auto => {
                return Ok(Self {
                    cpu,
                    gpu: None,
                    diagnostics: ExecutionDiagnostics::cpu(&options, Some(error.to_string())),
                    gpu_batches: 0,
                });
            }
            Err(error) => return Err(SessionError::from_gpu(error)),
        };
        match context.create_scoring(scene, model, 64).await {
            Ok(gpu) => Ok(Self {
                cpu,
                gpu: Some(gpu),
                diagnostics: ExecutionDiagnostics::gpu(&options, &context),
                gpu_batches: 0,
            }),
            Err(error) if options.backend == BackendPreference::Auto => Ok(Self {
                cpu,
                gpu: None,
                diagnostics: ExecutionDiagnostics::cpu(&options, Some(error.to_string())),
                gpu_batches: 0,
            }),
            Err(error) => Err(SessionError::from_gpu(error)),
        }
    }

    pub async fn evaluate(
        &mut self,
        batch: &PoseBatch,
        request: &EvaluationRequest,
    ) -> Result<Vec<EvaluationResult>, SessionError> {
        if let Some(gpu) = &mut self.gpu {
            match gpu.evaluate(batch, request).await {
                Ok(result) => {
                    self.gpu_batches = self.gpu_batches.saturating_add(1);
                    return Ok(result);
                }
                Err(error) if self.diagnostics.requested_backend == BackendPreference::Auto => {
                    self.gpu = None;
                    self.diagnostics.actual_backend = if self.gpu_batches == 0 {
                        "CPU".into()
                    } else {
                        "Mixed".into()
                    };
                    self.diagnostics.fallback_reason = Some(error.to_string());
                }
                Err(error) => return Err(SessionError::from_gpu(error)),
            }
        }
        let result = self
            .cpu
            .evaluate(batch, request)
            .map_err(|e| SessionError::new(SessionErrorKind::InvalidInput, e.to_string(), false))?;
        self.diagnostics.reference_call_count = self
            .diagnostics
            .reference_call_count
            .saturating_add(batch.poses.len() as u64);
        Ok(result)
    }

    pub fn diagnostics(&self) -> &ExecutionDiagnostics {
        &self.diagnostics
    }
}

/// Shared steric session. ReGlyco supplies its traversal-sensitive CPU policy;
/// this type owns only the resident GPU evaluator and its scoped diagnostics.
pub struct StericSession {
    resident: ResidentSteric,
    diagnostics: ExecutionDiagnostics,
}

impl StericSession {
    pub async fn new(
        context: &GpuContext,
        library: &glysys_gpu::steric::AttachmentLibrary,
        capacity: u32,
    ) -> Result<Self, SessionError> {
        let resident = context
            .create_steric(library, capacity)
            .await
            .map_err(SessionError::from_gpu)?;
        let mut options = ExecutionOptions::default();
        options.backend = BackendPreference::Gpu;
        Ok(Self {
            resident,
            diagnostics: ExecutionDiagnostics::gpu(&options, context),
        })
    }

    pub async fn evaluate(
        &mut self,
        genes: &[[u32; 4]],
        cutoff: f32,
    ) -> Result<Vec<f32>, SessionError> {
        self.resident
            .evaluate(genes, cutoff)
            .await
            .map_err(SessionError::from_gpu)
    }

    pub fn capacity(&self) -> u32 {
        self.resident.capacity
    }
    pub fn diagnostics(&self) -> &ExecutionDiagnostics {
        &self.diagnostics
    }
}

/// Common lifecycle façade used by native and browser adapters. Typed session
/// constructors remain available above so scientific outputs stay explicit.
pub struct ExecutionSession {
    simulation: SimulationSession,
}

impl ExecutionSession {
    /// Construct the shared simulation façade.  `new_simulation` remains the
    /// explicit spelling used by adapters that create more than one typed
    /// workload, while this short constructor keeps the common session API
    /// uniform for native callers.
    pub async fn new(
        system: ParameterizedSystem,
        protocol: SimulationProtocol,
        options: ExecutionOptions,
    ) -> Result<Self, SessionError> {
        Self::new_simulation(system, protocol, options).await
    }

    pub async fn new_simulation(
        system: ParameterizedSystem,
        protocol: SimulationProtocol,
        options: ExecutionOptions,
    ) -> Result<Self, SessionError> {
        Self::new_simulation_with_context(system, protocol, options, None).await
    }

    pub async fn new_simulation_with_context(
        system: ParameterizedSystem,
        protocol: SimulationProtocol,
        options: ExecutionOptions,
        context: Option<GpuContext>,
    ) -> Result<Self, SessionError> {
        Ok(Self {
            simulation: SimulationSession::new_with_context(system, protocol, options, context)
                .await?,
        })
    }

    pub async fn advance(
        &mut self,
        request: AdvanceRequest,
    ) -> Result<AdvanceResult, SessionError> {
        self.simulation.advance(request).await
    }

    pub async fn from_checkpoint(
        system: ParameterizedSystem,
        checkpoint: RuntimeCheckpoint,
        options: ExecutionOptions,
    ) -> Result<Self, SessionError> {
        Self::from_checkpoint_with_context(system, checkpoint, options, None).await
    }

    pub async fn from_checkpoint_with_context(
        system: ParameterizedSystem,
        checkpoint: RuntimeCheckpoint,
        options: ExecutionOptions,
        context: Option<GpuContext>,
    ) -> Result<Self, SessionError> {
        Ok(Self {
            simulation: SimulationSession::from_checkpoint_with_context(
                system, checkpoint, options, context,
            )
            .await?,
        })
    }

    pub fn state(&self) -> &SimulationState {
        self.simulation.state()
    }
    pub fn checkpoint(&self) -> Result<RuntimeCheckpoint, SessionError> {
        self.simulation.checkpoint()
    }
    pub fn diagnostics(&self) -> &ExecutionDiagnostics {
        self.simulation.diagnostics()
    }
    pub fn cancel(&mut self) {
        self.simulation.cancel();
    }
    pub fn fallback_to_cpu(&mut self, reason: impl Into<String>) -> Result<(), SessionError> {
        self.simulation.fallback_to_cpu(reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_schema_rejects_legacy_records() {
        let json = r#"{"schemaVersion":1,"runtimeSchemaVersion":1,"state":{},"diagnostics":{}}"#;
        let error = RuntimeCheckpoint::decode(json).unwrap_err();
        assert_eq!(error.kind, SessionErrorKind::InvalidInput);
        assert!(error.message.contains("requires v3"));
    }

    #[test]
    fn checkpoint_schema_rejects_raw_simulation_state() {
        // Older browser/native writers serialized SimulationState directly.
        // A missing runtime envelope must be a migration error, even when the
        // payload happens to contain fields that look like a checkpoint.
        let json = r#"{"schemaVersion":1,"modelVersion":"obc2-baoab-v1","step":0}"#;
        let error = RuntimeCheckpoint::decode(json).unwrap_err();
        assert_eq!(error.kind, SessionErrorKind::InvalidInput);
        assert!(error.message.contains("requires v3"));
    }

    #[test]
    fn execution_options_map_to_explicit_budget() {
        let options = ExecutionOptions {
            memory_policy: MemoryPolicy::Explicit,
            memory_budget_bytes: Some(1024),
            ..Default::default()
        };
        assert_eq!(options.gpu_memory_profile().budget(), 1024);
    }

    #[test]
    fn cpu_runtime_session_matches_direct_reference_session() {
        pollster::block_on(async {
            let system = glysys::SystemBuilder::new(glysys::BuildOptions {
                add_water: false,
                add_ions: false,
                ..Default::default()
            })
            .unwrap()
            .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))
            .unwrap();
            let protocol = SimulationProtocol {
                solvent: SolventModel::Implicit,
                constraints: glysys_dynamics::ConstraintModel::None,
                equilibration_ensemble: Ensemble::Nve,
                production_ensemble: Ensemble::Nve,
                equilibration_steps: 0,
                production_steps: 4,
                minimization_iterations: 0,
                timestep_fs: 0.1,
                save_every: 1,
                seed: 17,
                ..Default::default()
            };
            let mut direct = CpuSimulation::new(&system, protocol.clone())
                .unwrap()
                .into_owned();
            let mut runtime = SimulationSession::new(
                system.clone(),
                protocol,
                ExecutionOptions {
                    backend: BackendPreference::Cpu,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            let expected = direct.advance(3).unwrap();
            let actual = runtime
                .advance(AdvanceRequest {
                    steps: 3,
                    ..Default::default()
                })
                .await
                .unwrap();
            assert_eq!(actual.chunk.frames.len(), expected.frames.len());
            for (actual, expected) in actual.chunk.frames.iter().zip(&expected.frames) {
                assert_eq!(actual.step, expected.step);
                assert_eq!(actual.coordinates, expected.coordinates);
                assert_eq!(actual.box_angstrom, expected.box_angstrom);
                assert_eq!(actual.potential_energy, expected.potential_energy);
            }
            assert_eq!(runtime.state().step, direct.state.step);
            assert_eq!(runtime.state().coordinates, direct.state.coordinates);
            assert_eq!(runtime.state().velocities, direct.state.velocities);
            assert_eq!(runtime.state().gradient, direct.state.gradient);
            assert_eq!(runtime.state().rng_state, direct.state.rng_state);
            match (&runtime.state().resident_rng, &direct.state.resident_rng) {
                (Some(actual), Some(expected)) => {
                    assert_eq!(actual.version, expected.version);
                    assert_eq!(actual.words, expected.words);
                }
                (None, None) => {}
                _ => panic!("runtime and direct sessions disagree on resident RNG"),
            }
            assert_eq!(
                runtime.state().potential_energy,
                direct.state.potential_energy
            );

            let encoded = runtime.checkpoint().unwrap().encode().unwrap();
            let decoded = RuntimeCheckpoint::decode(&encoded).unwrap();
            assert_eq!(decoded.schema_version, CHECKPOINT_SCHEMA_VERSION);
            assert_eq!(decoded.runtime_schema_version, RUNTIME_SCHEMA_VERSION);
            assert_eq!(decoded.state.step, runtime.state().step);
            assert_eq!(decoded.state.coordinates, runtime.state().coordinates);
            assert_eq!(decoded.state.velocities, runtime.state().velocities);
            assert_eq!(decoded.state.rng_state, runtime.state().rng_state);
        });
    }
}
