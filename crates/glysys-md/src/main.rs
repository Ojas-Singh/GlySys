use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use glysys::ParameterizedSystem;
use glysys_dynamics::{Ensemble, SimulationProtocol, SolventModel};
use glysys_runtime::{
    AdvanceRequest, BackendPreference, ExecutionDiagnostics, ExecutionOptions, ExecutionSession,
    MemoryPolicy, RuntimeCheckpoint,
};

mod bundle;
mod gromacs;

const NATIVE_GPU_EXPLICIT_BATCH_STEPS: usize = 2500;
const NATIVE_GPU_IMPLICIT_BATCH_STEPS: usize = 2500;
const NATIVE_CPU_BATCH_STEPS: usize = 100;

#[derive(Debug, Clone, Copy, ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Auto,
    Cpu,
    Vulkan,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MemoryMode {
    /// Size resident allocations from the workload and the adapter limits.
    Adaptive,
    /// Retain the historical 256 MiB aggregate cap.
    LowMemory,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImplicitGpuLanes {
    #[value(name = "4")]
    Four,
    #[value(name = "8")]
    Eight,
    #[value(name = "16")]
    Sixteen,
    #[value(name = "32")]
    ThirtyTwo,
    #[value(name = "64")]
    SixtyFour,
    #[value(name = "128")]
    OneTwentyEight,
}

impl ImplicitGpuLanes {
    fn count(self) -> usize {
        match self {
            Self::Four => 4,
            Self::Eight => 8,
            Self::Sixteen => 16,
            Self::ThirtyTwo => 32,
            Self::SixtyFour => 64,
            Self::OneTwentyEight => 128,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImplicitGpuPacketSteps {
    #[value(name = "8")]
    Eight,
    #[value(name = "16")]
    Sixteen,
    #[value(name = "32")]
    ThirtyTwo,
    #[value(name = "64")]
    SixtyFour,
    #[value(name = "128")]
    OneTwentyEight,
}

impl ImplicitGpuPacketSteps {
    fn count(self) -> usize {
        match self {
            Self::Eight => 8,
            Self::Sixteen => 16,
            Self::ThirtyTwo => 32,
            Self::SixtyFour => 64,
            Self::OneTwentyEight => 128,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "glysys-md",
    version,
    about = "Run reproducible GlySys molecular dynamics"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run a prepared snapshot from step zero.
    Run(RunArgs),
    /// Resume a run from its JSON checkpoint.
    Resume(ResumeArgs),
    /// Print a small timing record for a prepared snapshot.
    Benchmark(BenchmarkArgs),
    /// List native adapters visible to wgpu.
    Devices,
    /// Resolve a supported GROMACS .mdp into a versioned GlySys protocol.
    ResolveMdp(ResolveMdpArgs),
    /// Verify a generated GROMACS .gro/.top pair against a lossless snapshot.
    VerifyGromacs(VerifyGromacsArgs),
}

#[derive(Debug, Args, Clone)]
struct ResolveMdpArgs {
    /// Supported GOTW .mdp file.
    #[arg(long)]
    mdp: PathBuf,
    /// Optional path for the resolved protocol JSON. stdout is always a
    /// machine-readable audit record.
    #[arg(long)]
    protocol_out: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct VerifyGromacsArgs {
    /// Prepared directory or system.snapshot.json file.
    #[arg(short, long)]
    input: PathBuf,
    /// GROMACS topology; defaults to input/system.top for a directory.
    #[arg(long)]
    topology: Option<PathBuf>,
    /// GROMACS coordinates; defaults to input/system.gro for a directory.
    #[arg(long)]
    coordinates: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct CommonArgs {
    /// Prepared directory or `system.snapshot.json` file.
    #[arg(short, long)]
    input: PathBuf,
    /// JSON SimulationProtocol. If omitted, an explicit protocol is inferred
    /// for a solvated snapshot and an implicit protocol for a dry snapshot.
    #[arg(long)]
    protocol: Option<PathBuf>,
    /// Resolve a supported GROMACS .mdp file into the versioned GlySys
    /// protocol contract. This records PME/Nose-Hoover/Parrinello-Rahman as
    /// requested choices; the run stops explicitly until those kernels pass
    /// their validation gates.
    #[arg(long, conflicts_with = "protocol")]
    gromacs_mdp: Option<PathBuf>,
    /// Execution backend. Auto prefers a compatible native resident GPU and
    /// falls back to CPU with the reason recorded in run.json.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,
    /// Rayon worker count for this process. Use one disjoint process per
    /// replica in Slurm rather than a shared whole-node pool.
    #[arg(long)]
    threads: Option<usize>,
    /// Implicit GPU all-pairs tile size (lanes cooperating per target atom).
    // Native default selected from the RX 7800 XT 1CRN lane sweep; the shared
    // runtime/browser default remains unchanged.
    #[arg(long, value_enum, default_value_t = ImplicitGpuLanes::Sixteen)]
    implicit_gpu_lanes_per_target: ImplicitGpuLanes,
    /// Implicit GPU LF-middle steps encoded in one bounded packet.
    // 32 steps had the best median among the measured 8/16/32/64/128 packets.
    #[arg(long, value_enum, default_value_t = ImplicitGpuPacketSteps::ThirtyTwo)]
    implicit_gpu_packet_steps: ImplicitGpuPacketSteps,
    /// Opt in to cooperative 64-lane explicit pair evaluation with a bounded
    /// fixed-stride native neighbor layout. This keeps the established CSR
    /// path available for browser compatibility checks and side-by-side runs.
    #[arg(long)]
    explicit_gpu_tiled_nonbonded: bool,
    /// Explicit GPU Verlet skin in angstroms. Skin changes neighbor-list
    /// rebuild frequency only; it does not change the physical cutoff.
    #[arg(long, default_value_t = 1.5)]
    explicit_gpu_neighbor_skin_angstrom: f64,
    /// Override preparation minimization iterations for a short benchmark or
    /// a deliberately minimized production setup.
    #[arg(long)]
    minimization_iterations: Option<usize>,
    /// Override the protocol seed for independent HPC replicas. A resume
    /// always uses the checkpoint's persisted random streams.
    #[arg(long)]
    seed: Option<u64>,
    /// GPU allocation policy. Adaptive is the default and does not impose a
    /// synthetic 256 MiB cap; device limits and allocation failures still
    /// control the usable size.
    #[arg(long, value_enum, default_value_t = MemoryMode::Adaptive)]
    gpu_memory: MemoryMode,
    /// Optional explicit aggregate GPU budget in MiB. It is useful for a
    /// reproducible benchmark or a known shared-device allocation.
    #[arg(long, conflicts_with = "gpu_memory")]
    gpu_memory_mib: Option<u64>,
}

#[derive(Debug, Args, Clone)]
struct RunArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Output directory for checkpoint, trajectory, and run metadata.
    #[arg(short, long)]
    output: PathBuf,
    /// Override the normalized protocol's total step count for a short test.
    #[arg(long)]
    steps: Option<usize>,
    /// Write scalar observables to observables.jsonl at this step interval,
    /// without generating additional coordinate frames.
    #[arg(long)]
    observables_every: Option<usize>,
}

#[derive(Debug, Args, Clone)]
struct ResumeArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Existing run directory containing `checkpoint.json`.
    #[arg(short, long)]
    output: PathBuf,
    /// Write scalar observables to observables.jsonl at this step interval.
    #[arg(long)]
    observables_every: Option<usize>,
}

#[derive(Debug, Args, Clone)]
struct BenchmarkArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Fixed-step production window (mutually exclusive with --seconds).
    #[arg(long, conflicts_with = "seconds")]
    steps: Option<usize>,
    /// Wall-clock production duration per repeat; GPU work is synchronized by
    /// the completed advance/readback before this timer is stopped.
    #[arg(long, conflicts_with = "steps")]
    seconds: Option<f64>,
    /// Completed warmup steps excluded from the measured window.
    #[arg(long, default_value_t = 0)]
    warmup_steps: usize,
    #[arg(long, default_value_t = 1)]
    repeats: usize,
    #[arg(long, value_enum, default_value_t = BenchmarkTiming::Off)]
    profile_timing: BenchmarkTiming,
    /// Canonical runtime checkpoint used to seed each repeat.
    #[arg(long)]
    starting_checkpoint: Option<PathBuf>,
    /// Optional final runtime checkpoint (requires exactly one repeat).
    #[arg(long)]
    ending_checkpoint: Option<PathBuf>,
    /// Optional JSON result destination; stdout always receives the result.
    #[arg(long)]
    json_output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum BenchmarkTiming {
    Off,
    Host,
    Gpu,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Devices => devices(),
        Command::Run(args) => run(args),
        Command::Resume(args) => resume(args),
        Command::Benchmark(args) => benchmark(args),
        Command::ResolveMdp(args) => resolve_mdp(args),
        Command::VerifyGromacs(args) => verify_gromacs(args),
    }
}

fn resolve_mdp(args: ResolveMdpArgs) -> Result<()> {
    let resolved = gromacs::parse(&args.mdp)?;
    let protocol = resolved.to_protocol()?;
    let audit = serde_json::json!({
        "resolved": resolved,
        "protocol": protocol,
        "runtimeStatus": "requested model is retained; PME/Nose-Hoover/Parrinello-Rahman drivers remain capability-gated",
    });
    let bytes = serde_json::to_vec_pretty(&audit)?;
    if let Some(path) = args.protocol_out {
        fs::write(path, serde_json::to_vec_pretty(&audit["protocol"])?)?;
    }
    println!("{}", String::from_utf8(bytes).expect("serde JSON is UTF-8"));
    Ok(())
}

fn verify_gromacs(args: VerifyGromacsArgs) -> Result<()> {
    let system = load_system(&args.input)?;
    let directory = args
        .input
        .is_dir()
        .then(|| args.input.clone())
        .or_else(|| args.input.parent().map(Path::to_path_buf));
    let topology = args.topology.unwrap_or_else(|| {
        directory
            .as_ref()
            .map(|dir| dir.join("system.top"))
            .unwrap_or_else(|| PathBuf::from("system.top"))
    });
    let coordinates = args.coordinates.unwrap_or_else(|| {
        directory
            .as_ref()
            .map(|dir| dir.join("system.gro"))
            .unwrap_or_else(|| PathBuf::from("system.gro"))
    });
    let report = bundle::verify_bundle(&system, &topology, &coordinates)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn devices() -> Result<()> {
    let adapters = glysys_gpu::adapter_report();
    if adapters.is_empty() {
        println!("[]");
    } else {
        println!("{}", serde_json::to_string_pretty(&adapters)?);
    }
    Ok(())
}

fn snapshot_path(input: &Path) -> PathBuf {
    if input.is_dir() {
        input.join("system.snapshot.json")
    } else {
        input.to_path_buf()
    }
}

fn load_system(input: &Path) -> Result<ParameterizedSystem> {
    let path = snapshot_path(input);
    let json = fs::read_to_string(&path)
        .with_context(|| format!("reading prepared snapshot {}", path.display()))?;
    ParameterizedSystem::from_snapshot_json(&json)
        .map_err(|e| anyhow::anyhow!(e.to_string()))
        .with_context(|| format!("validating prepared snapshot {}", path.display()))
}

fn default_protocol(system: &ParameterizedSystem) -> SimulationProtocol {
    let explicit = system.report().waters > 0 && system.box_angstrom().iter().all(|v| *v > 0.0);
    SimulationProtocol {
        solvent: if explicit {
            SolventModel::Explicit
        } else {
            SolventModel::Implicit
        },
        constraints: if explicit {
            glysys_dynamics::ConstraintModel::Settle
        } else {
            glysys_dynamics::ConstraintModel::None
        },
        timestep_fs: if explicit { 2.0 } else { 1.0 },
        cutoff_angstrom: explicit.then_some(9.0),
        rf_dielectric: explicit.then_some(78.5),
        thermostat: if explicit {
            glysys_dynamics::Thermostat::Langevin
        } else {
            glysys_dynamics::Thermostat::VRescale
        },
        electrostatics: glysys_dynamics::ElectrostaticsModel::ReactionField,
        pressure_coupling: glysys_dynamics::PressureCoupling::MonteCarlo,
        equilibration_ensemble: Ensemble::Nvt,
        production_ensemble: Ensemble::Nvt,
        ..SimulationProtocol::default()
    }
}

fn load_protocol(
    path: Option<&Path>,
    gromacs_mdp: Option<&Path>,
    system: &ParameterizedSystem,
) -> Result<SimulationProtocol> {
    if let Some(path) = gromacs_mdp {
        if path.extension().and_then(|s| s.to_str()) != Some("mdp") {
            bail!("--gromacs-mdp must point to a .mdp file");
        }
        let resolved = gromacs::parse(path)?;
        eprintln!(
            "resolved GROMACS recipe: coulombtype={} tcoupl={} pcoupl={} cutoff={:.3} A steps={}",
            resolved.coulomb_type,
            resolved.thermostat,
            resolved.pressure_coupling,
            resolved.cutoff_angstrom,
            resolved.steps
        );
        return resolved.to_protocol();
    }
    let Some(path) = path else {
        return Ok(default_protocol(system));
    };
    let json =
        fs::read_to_string(path).with_context(|| format!("reading protocol {}", path.display()))?;
    let protocol: SimulationProtocol = serde_json::from_str(&json)
        .with_context(|| format!("parsing protocol {}", path.display()))?;
    protocol
        .validate()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    Ok(protocol)
}

fn memory_policy(mode: MemoryMode, mib: Option<u64>) -> Result<(MemoryPolicy, Option<u64>)> {
    if let Some(mib) = mib {
        let bytes = mib
            .checked_mul(1024 * 1024)
            .ok_or_else(|| anyhow::anyhow!("--gpu-memory-mib is too large"))?;
        if bytes == 0 {
            bail!("--gpu-memory-mib must be positive");
        }
        return Ok((MemoryPolicy::Explicit, Some(bytes)));
    }
    Ok((
        match mode {
            MemoryMode::Adaptive => MemoryPolicy::Adaptive,
            MemoryMode::LowMemory => MemoryPolicy::LowMemory,
        },
        None,
    ))
}

fn apply_cli_overrides(
    mut protocol: SimulationProtocol,
    minimization_iterations: Option<usize>,
    seed: Option<u64>,
) -> SimulationProtocol {
    if let Some(iterations) = minimization_iterations {
        protocol.minimization_iterations = iterations;
    }
    if let Some(seed) = seed {
        protocol.seed = seed;
    }
    protocol
}

fn configure_threads(threads: Option<usize>) -> Result<usize> {
    let requested = threads.unwrap_or_else(|| rayon::current_num_threads());
    if requested == 0 {
        bail!("--threads must be positive");
    }
    if threads.is_some() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(requested)
            .build_global()
            .map_err(|e| anyhow::anyhow!("initializing Rayon thread pool: {e}"))?;
    }
    Ok(rayon::current_num_threads())
}

fn choose_backend(requested: Backend, _protocol: &SimulationProtocol) -> Result<&'static str> {
    match requested {
        Backend::Cpu => Ok("cpu"),
        Backend::Auto => Ok("auto"),
        Backend::Vulkan => Ok("vulkan"),
    }
}

fn execution_options(
    backend: Backend,
    memory: MemoryPolicy,
    threads: usize,
    implicit_gpu_lanes_per_target: usize,
    implicit_gpu_packet_steps: usize,
    explicit_gpu_tiled_nonbonded: bool,
    explicit_gpu_neighbor_skin_angstrom: f64,
) -> ExecutionOptions {
    ExecutionOptions {
        backend: match backend {
            Backend::Auto => BackendPreference::Auto,
            Backend::Cpu => BackendPreference::Cpu,
            Backend::Vulkan => BackendPreference::Gpu,
        },
        memory_policy: memory,
        cpu_thread_limit: Some(threads),
        // Native GPU runs are split at checkpoint/frame boundaries below.
        // Permit a full explicit 5 ps checkpoint interval in one resident
        // session advance; the GPU adapter bounds actual encoded work.
        max_submission_steps: NATIVE_GPU_EXPLICIT_BATCH_STEPS,
        implicit_gpu_lanes_per_target,
        implicit_gpu_packet_steps,
        explicit_gpu_tiled_nonbonded,
        explicit_gpu_neighbor_skin_angstrom,
        ..ExecutionOptions::default()
    }
}

fn preferred_batch_steps(session: &ExecutionSession) -> usize {
    if session.diagnostics().actual_backend != "GPU" {
        return NATIVE_CPU_BATCH_STEPS;
    }
    match session.state().protocol.solvent {
        SolventModel::Explicit => NATIVE_GPU_EXPLICIT_BATCH_STEPS,
        SolventModel::Implicit
            if session.state().protocol.langevin_discretization
                == glysys_dynamics::LangevinDiscretization::LfMiddle =>
        {
            NATIVE_GPU_IMPLICIT_BATCH_STEPS
        }
        SolventModel::Implicit => glysys_gpu::dynamics::MAX_STEPS,
    }
}

fn timed_benchmark_batch_steps(
    preferred: usize,
    elapsed_seconds: f64,
    duration_seconds: f64,
    measured_steps_per_second: Option<f64>,
) -> usize {
    let remaining_seconds = (duration_seconds - elapsed_seconds).max(0.0);
    let target = measured_steps_per_second
        .filter(|rate| rate.is_finite() && *rate > 0.0)
        .map(|rate| (rate * remaining_seconds * 0.9).floor() as usize)
        // Use a deliberately small first packet: a CPU explicit step can be
        // hundreds of milliseconds on this system, so a fixed 64-step probe
        // would make a nominal 5-second window overshoot by 10+ seconds.
        .unwrap_or(8);
    target.max(1).min(preferred.max(1))
}

fn scheduled_advance_limit(
    protocol: &SimulationProtocol,
    current_step: usize,
    preferred_steps: usize,
    checkpoint_interval: usize,
    observables_every: Option<usize>,
) -> usize {
    let remaining = protocol.total_steps().saturating_sub(current_step);
    if remaining == 0 {
        return 0;
    }
    let save_every = protocol.save_every.max(1);
    let checkpoint_interval = checkpoint_interval.max(1);
    let until_frame = save_every - current_step % save_every;
    let until_checkpoint = checkpoint_interval - current_step % checkpoint_interval;
    let until_observable = observables_every
        .filter(|interval| *interval > 0)
        .map(|interval| interval - current_step % interval)
        .unwrap_or(remaining);
    let mut stage_end = 0usize;
    let until_stage = protocol
        .execution_stages()
        .iter()
        .filter_map(|stage| {
            stage_end = stage_end.saturating_add(stage.steps);
            if stage_end > current_step {
                Some(stage_end - current_step)
            } else {
                None
            }
        })
        .min()
        .unwrap_or(remaining);
    preferred_steps
        .max(1)
        .min(remaining)
        .min(until_frame)
        .min(until_checkpoint)
        .min(until_observable)
        .min(until_stage)
}

fn checkpoint_due_at(
    protocol: &SimulationProtocol,
    completed_step: usize,
    interval: usize,
) -> bool {
    completed_step > 0
        && (completed_step.is_multiple_of(interval.max(1))
            || protocol.is_stage_boundary(completed_step)
            || completed_step == protocol.total_steps())
}

fn construct(
    system: ParameterizedSystem,
    protocol: SimulationProtocol,
    options: ExecutionOptions,
) -> Result<ExecutionSession> {
    pollster::block_on(ExecutionSession::new_simulation(system, protocol, options))
        .map_err(|e| anyhow::anyhow!(e.to_string()))
}

fn write_metadata(
    output: &Path,
    threads: usize,
    session: &ExecutionSession,
    started: Instant,
    simulation_seconds: f64,
    simulation_ns: f64,
    observables_every: Option<usize>,
    checkpoint_write_count: u64,
    setup_and_minimization_seconds: f64,
    file_output_seconds: f64,
) -> Result<()> {
    let state = session.state();
    let mut diagnostics = session.diagnostics().clone();
    diagnostics.checkpoint_write_count = checkpoint_write_count;
    diagnostics
        .stage_timings_ms
        .insert("fileOutput".into(), file_output_seconds * 1000.0);
    diagnostics.stage_timings_ms.insert(
        "setupAndMinimization".into(),
        setup_and_minimization_seconds * 1000.0,
    );
    let metadata = serde_json::json!({
        "schemaVersion": glysys_runtime::RUNTIME_SCHEMA_VERSION,
        "runtimeSchemaVersion": glysys_runtime::RUNTIME_SCHEMA_VERSION,
        "checkpointSchemaVersion": glysys_runtime::CHECKPOINT_SCHEMA_VERSION,
        "workerProtocolVersion": glysys_runtime::WORKER_PROTOCOL_VERSION,
        "backend": diagnostics.actual_backend,
        "requestedBackend": diagnostics.requested_backend,
        "adapter": diagnostics.adapter_identity,
        "allocationBudgetBytes": diagnostics.allocation_budget_bytes,
        "peakAllocationBytes": diagnostics.peak_allocation_bytes,
        "validationMode": diagnostics.validation_mode,
        "referenceCallCount": diagnostics.reference_call_count,
        "threads": threads,
        "steps": state.step,
        "totalSteps": state.protocol.total_steps(),
        "simulatedPs": state.step as f64 * state.protocol.timestep_fs * 0.001,
        "wallSeconds": started.elapsed().as_secs_f64(),
        "simulationSeconds": simulation_seconds,
        "setupAndMinimizationSeconds": setup_and_minimization_seconds,
        "productionIntegrationAndQueueWaitSeconds": diagnostics.stage_timings_ms.get("productionIntegrationAndQueueWait").copied().unwrap_or(0.0) / 1000.0,
        "readbackAndDecodeSeconds": diagnostics.stage_timings_ms.get("readbackAndDecode").copied().unwrap_or(0.0) / 1000.0,
        "fileOutputSeconds": file_output_seconds,
        "simulationNs": simulation_ns,
        "nsPerDay": simulation_ns / simulation_seconds.max(f64::MIN_POSITIVE) * 86400.0,
        "modelVersion": state.model_version,
        "velocityConvention": state.velocity_convention,
        "observablesEvery": observables_every,
        "protocol": state.protocol,
        "fallbackReason": diagnostics.fallback_reason,
        "diagnostics": diagnostics,
    });
    fs::write(
        output.join("run.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    Ok(())
}

fn write_checkpoint(output: &Path, checkpoint: &RuntimeCheckpoint) -> Result<()> {
    let temporary = output.join("checkpoint.json.tmp");
    fs::write(&temporary, serde_json::to_vec(checkpoint)?)?;
    fs::rename(temporary, output.join("checkpoint.json"))?;
    Ok(())
}

fn run_loop(
    mut session: ExecutionSession,
    system: &ParameterizedSystem,
    output: &Path,
    threads: usize,
    started: Instant,
    requested_backend: &str,
    observables_every: Option<usize>,
) -> Result<()> {
    if observables_every == Some(0) {
        bail!("--observables-every must be positive");
    }
    fs::create_dir_all(output)?;
    let trajectory = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output.join("trajectory.jsonl"))?;
    let mut trajectory = BufWriter::new(trajectory);
    let mut observables = observables_every
        .map(|_| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(output.join("observables.jsonl"))
                .map(BufWriter::new)
                .with_context(|| "opening observables.jsonl")
        })
        .transpose()?;
    let mut initial_checkpoint = session
        .checkpoint()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let mut checkpoint_write_count = 0u64;
    let mut file_output_seconds = 0.0;
    let initial_file_output = Instant::now();
    if initial_checkpoint.state.step == 0 {
        // The live checkpoint is replaced periodically for restart. Preserve
        // the minimized step-zero state separately for reproducible external
        // references (for example, an independent OpenMM trajectory).
        checkpoint_write_count += 1;
        initial_checkpoint.diagnostics.checkpoint_write_count = checkpoint_write_count;
        fs::write(
            output.join("step-zero-reference-checkpoint.json"),
            serde_json::to_vec(&initial_checkpoint)?,
        )?;
    }
    checkpoint_write_count += 1;
    initial_checkpoint.diagnostics.checkpoint_write_count = checkpoint_write_count;
    write_checkpoint(output, &initial_checkpoint)?;
    let initial_file_output_seconds = initial_file_output.elapsed().as_secs_f64();
    file_output_seconds += initial_file_output_seconds;
    // Keep restart I/O off the per-100-step submission cadence. Five
    // simulated picoseconds bounds recovery loss while avoiding thousands of
    // multi-megabyte checkpoint rewrites in a nanosecond qualification run.
    let checkpoint_interval =
        ((5.0 / (session.state().protocol.timestep_fs * 0.001)).round() as usize).max(1);
    let masses: Vec<_> = system.atoms().iter().map(|atom| atom.mass()).collect();
    let starting_step = session.current_step();
    let simulation_started = Instant::now();
    let mut last_progress = simulation_started;
    while session.current_step() < session.state().protocol.total_steps() {
        let current_step = session.current_step();
        // Ask for no more than the next observable/checkpoint/stage boundary.
        // A backend may return a partial advance; all decisions below use the
        // committed endpoint rather than this requested endpoint.
        let requested_steps = scheduled_advance_limit(
            &session.state().protocol,
            current_step,
            preferred_batch_steps(&session),
            checkpoint_interval,
            observables_every,
        );
        let expected_endpoint = current_step.saturating_add(requested_steps);
        let observable_target = observables_every.is_some_and(|interval| {
            expected_endpoint.is_multiple_of(interval)
                || expected_endpoint == session.state().protocol.total_steps()
        });
        let full_state_target = expected_endpoint
            .is_multiple_of(session.state().protocol.save_every.max(1))
            || expected_endpoint.is_multiple_of(checkpoint_interval)
            || session
                .state()
                .protocol
                .is_stage_boundary(expected_endpoint)
            || expected_endpoint == session.state().protocol.total_steps();
        let scalar_only = observable_target && !full_state_target;
        let (completed_step, frames, scalar_observation) = if scalar_only {
            let observation = match pollster::block_on(
                session.advance_with_scalar_observation(requested_steps),
            ) {
                Ok(observation) => observation,
                Err(error) if requested_backend == "auto" && error.cpu_fallback_valid => {
                    session
                        .fallback_to_cpu(error.to_string())
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    continue;
                }
                Err(error) => return Err(anyhow::anyhow!(error.to_string())),
            };
            (observation.step, Vec::new(), Some(observation))
        } else {
            let result = match pollster::block_on(session.advance(AdvanceRequest {
                steps: requested_steps,
                include_checkpoint: false,
                max_frames: None,
            })) {
                Ok(result) => result,
                Err(error) if requested_backend == "auto" && error.cpu_fallback_valid => {
                    session
                        .fallback_to_cpu(error.to_string())
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    continue;
                }
                Err(error) => return Err(anyhow::anyhow!(error.to_string())),
            };
            (result.chunk.last_step, result.chunk.frames, None)
        };
        let has_frames = !frames.is_empty();
        let frame_output_started = Instant::now();
        for frame in frames {
            serde_json::to_writer(&mut trajectory, &frame)?;
            trajectory.write_all(b"\n")?;
        }
        if has_frames {
            trajectory.flush()?;
        }
        file_output_seconds += frame_output_started.elapsed().as_secs_f64();
        let observable_due = observables_every.is_some_and(|interval| {
            completed_step > current_step
                && (completed_step.is_multiple_of(interval)
                    || completed_step == session.state().protocol.total_steps())
        });
        if observable_due {
            let (potential, kinetic, temperature) = scalar_observation.map_or_else(
                || {
                    let kinetic = glysys_dynamics::explicit::kinetic_energy(
                        &masses,
                        &session.state().velocities,
                    );
                    let dof = if session.state().degrees_of_freedom == 0 {
                        3 * masses.len()
                    } else {
                        session.state().degrees_of_freedom
                    };
                    (
                        session.state().potential_energy,
                        kinetic,
                        glysys_dynamics::explicit::kinetic_temperature(kinetic, dof),
                    )
                },
                |observation| {
                    (
                        observation.potential_energy_kcal_mol,
                        observation.kinetic_energy_kcal_mol,
                        observation.temperature_k,
                    )
                },
            );
            let sample = serde_json::json!({
                "step": completed_step,
                "timePs": completed_step as f64 * session.state().protocol.timestep_fs * 0.001,
                "segment": session.state().protocol.stage_info(completed_step).1,
                "potentialEnergyKcalMol": potential,
                "kineticEnergyKcalMol": kinetic,
                "temperatureK": temperature,
                "backend": session.diagnostics().actual_backend,
                "fallbackReason": session.diagnostics().fallback_reason,
            });
            if let Some(writer) = observables.as_mut() {
                let output_started = Instant::now();
                serde_json::to_writer(&mut *writer, &sample)?;
                writer.write_all(b"\n")?;
                writer.flush()?;
                file_output_seconds += output_started.elapsed().as_secs_f64();
            }
        }
        let checkpoint_due = completed_step > current_step
            && checkpoint_due_at(
                &session.state().protocol,
                completed_step,
                checkpoint_interval,
            );
        if checkpoint_due {
            let mut checkpoint = session
                .checkpoint()
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            checkpoint_write_count += 1;
            checkpoint.diagnostics.checkpoint_write_count = checkpoint_write_count;
            let output_started = Instant::now();
            write_checkpoint(output, &checkpoint)?;
            file_output_seconds += output_started.elapsed().as_secs_f64();
        }
        if checkpoint_due || last_progress.elapsed() >= std::time::Duration::from_secs(1) {
            eprintln!(
                "step={}/{} backend={} threads={threads}",
                session.current_step(),
                session.state().protocol.total_steps(),
                session.diagnostics().actual_backend
            );
            last_progress = Instant::now();
        }
    }
    let simulation_seconds = simulation_started.elapsed().as_secs_f64();
    let simulation_ns = (session.state().step - starting_step) as f64
        * session.state().protocol.timestep_fs
        * 1.0e-6;
    write_metadata(
        output,
        threads,
        &session,
        started,
        simulation_seconds,
        simulation_ns,
        observables_every,
        checkpoint_write_count,
        (simulation_started.duration_since(started).as_secs_f64() - initial_file_output_seconds)
            .max(0.0),
        file_output_seconds,
    )?;
    Ok(())
}

fn run(args: RunArgs) -> Result<()> {
    if args.output.join("checkpoint.json").exists()
        || args.output.join("trajectory.jsonl").exists()
        || args
            .output
            .join("step-zero-reference-checkpoint.json")
            .exists()
    {
        bail!(
            "output directory {} already contains a run; use 'resume' or choose a new directory",
            args.output.display()
        );
    }
    let system = load_system(&args.common.input)?;
    let mut protocol = apply_cli_overrides(
        load_protocol(
            args.common.protocol.as_deref(),
            args.common.gromacs_mdp.as_deref(),
            &system,
        )?,
        args.common.minimization_iterations,
        args.common.seed,
    );
    if let Some(steps) = args.steps {
        protocol.production_steps = steps;
        protocol.equilibration_steps = 0;
        protocol.stages = None;
    }
    if let Some(mdp) = args.common.gromacs_mdp.as_deref() {
        let resolved = gromacs::parse(mdp)?;
        let audit = serde_json::json!({
            "resolved": resolved,
            "protocol": protocol,
            "runtimeStatus": "requested model is retained; PME/Nose-Hoover/Parrinello-Rahman drivers remain capability-gated",
        });
        fs::create_dir_all(&args.output)?;
        fs::write(
            args.output.join("resolved-mdp.json"),
            serde_json::to_vec_pretty(&audit)?,
        )?;
    }
    let _selection = choose_backend(args.common.backend, &protocol)?;
    let threads = configure_threads(args.common.threads)?;
    let (memory, budget) = memory_policy(args.common.gpu_memory, args.common.gpu_memory_mib)?;
    let mut options = execution_options(
        args.common.backend,
        memory,
        threads,
        args.common.implicit_gpu_lanes_per_target.count(),
        args.common.implicit_gpu_packet_steps.count(),
        args.common.explicit_gpu_tiled_nonbonded,
        args.common.explicit_gpu_neighbor_skin_angstrom,
    );
    options.memory_budget_bytes = budget;
    let started = Instant::now();
    let session = construct(system.clone(), protocol, options)?;
    run_loop(
        session,
        &system,
        &args.output,
        threads,
        started,
        match args.common.backend {
            Backend::Auto => "auto",
            Backend::Cpu => "cpu",
            Backend::Vulkan => "vulkan",
        },
        args.observables_every,
    )
}

fn resume(args: ResumeArgs) -> Result<()> {
    if !args.output.join("checkpoint.json").is_file() {
        bail!(
            "resume output directory has no checkpoint.json: {}",
            args.output.display()
        );
    }
    let system = load_system(&args.common.input)?;
    let checkpoint = fs::read_to_string(args.output.join("checkpoint.json"))
        .with_context(|| format!("reading checkpoint in {}", args.output.display()))?;
    let runtime_checkpoint =
        RuntimeCheckpoint::decode(&checkpoint).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let state = runtime_checkpoint.state.clone();
    let _selection = choose_backend(args.common.backend, &state.protocol)?;
    let threads = configure_threads(args.common.threads)?;
    let (memory, budget) = memory_policy(args.common.gpu_memory, args.common.gpu_memory_mib)?;
    let mut options = execution_options(
        args.common.backend,
        memory,
        threads,
        args.common.implicit_gpu_lanes_per_target.count(),
        args.common.implicit_gpu_packet_steps.count(),
        args.common.explicit_gpu_tiled_nonbonded,
        args.common.explicit_gpu_neighbor_skin_angstrom,
    );
    options.memory_budget_bytes = budget;
    let started = Instant::now();
    let session = pollster::block_on(ExecutionSession::from_checkpoint(
        system.clone(),
        runtime_checkpoint,
        options,
    ))
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    run_loop(
        session,
        &system,
        &args.output,
        threads,
        started,
        match args.common.backend {
            Backend::Auto => "auto",
            Backend::Cpu => "cpu",
            Backend::Vulkan => "vulkan",
        },
        args.observables_every,
    )
}

fn extend_benchmark_schedule(protocol: &mut SimulationProtocol) -> Result<()> {
    protocol.save_every = usize::MAX;
    if let Some(stages) = &mut protocol.stages {
        if stages.is_empty() {
            bail!("benchmark protocol has an empty explicit stage list");
        }
        let prefix_steps = stages[..stages.len() - 1]
            .iter()
            .try_fold(0usize, |sum, stage| sum.checked_add(stage.steps))
            .ok_or_else(|| anyhow::anyhow!("benchmark stage count overflow"))?;
        if prefix_steps >= glysys_dynamics::MAX_NATIVE_STEPS {
            bail!("benchmark warmup stages leave no room for a timed production stage");
        }
        stages.last_mut().unwrap().steps = glysys_dynamics::MAX_NATIVE_STEPS - prefix_steps;
    } else {
        if protocol.equilibration_steps >= glysys_dynamics::MAX_NATIVE_STEPS {
            bail!("benchmark equilibration leaves no room for production steps");
        }
        protocol.production_steps =
            glysys_dynamics::MAX_NATIVE_STEPS - protocol.equilibration_steps;
    }
    protocol
        .validate_for_native()
        .map_err(|e| anyhow::anyhow!(e.to_string()))
}

fn advance_exact_steps(session: &mut ExecutionSession, requested: usize) -> Result<usize> {
    let mut remaining = requested;
    let mut completed = 0usize;
    while remaining > 0 {
        let before = session.state().step;
        if before >= session.state().protocol.total_steps() {
            bail!("benchmark schedule ended before completing its requested window");
        }
        let count = remaining
            .min(preferred_batch_steps(session))
            .min(session.state().protocol.total_steps() - before);
        let result = pollster::block_on(session.advance(AdvanceRequest {
            steps: count,
            include_checkpoint: false,
            max_frames: Some(0),
        }))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let advanced = result.chunk.last_step.saturating_sub(before);
        if advanced == 0 {
            bail!("benchmark session made no progress at step {before}");
        }
        completed += advanced;
        remaining = remaining.saturating_sub(advanced);
    }
    Ok(completed)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return 0.0;
    }
    if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    } else {
        values[values.len() / 2]
    }
}

fn diagnostics_between(
    after: &ExecutionDiagnostics,
    before: &ExecutionDiagnostics,
) -> ExecutionDiagnostics {
    let mut delta = after.clone();
    for (name, value) in &mut delta.stage_timings_ms {
        *value = (*value - before.stage_timings_ms.get(name).copied().unwrap_or(0.0)).max(0.0);
    }
    delta.submission_count = after
        .submission_count
        .saturating_sub(before.submission_count);
    delta.full_state_readback_count = after
        .full_state_readback_count
        .saturating_sub(before.full_state_readback_count);
    delta.full_state_readback_bytes = after
        .full_state_readback_bytes
        .saturating_sub(before.full_state_readback_bytes);
    delta.scalar_readback_count = after
        .scalar_readback_count
        .saturating_sub(before.scalar_readback_count);
    delta.scalar_readback_bytes = after
        .scalar_readback_bytes
        .saturating_sub(before.scalar_readback_bytes);
    delta.checkpoint_write_count = after
        .checkpoint_write_count
        .saturating_sub(before.checkpoint_write_count);
    delta.reference_call_count = after
        .reference_call_count
        .saturating_sub(before.reference_call_count);
    delta.neighbor_rebuild_count =
        match (after.neighbor_rebuild_count, before.neighbor_rebuild_count) {
            (Some(after), Some(before)) => Some(after.saturating_sub(before)),
            (after, None) => after,
            (None, _) => None,
        };
    delta
}

fn benchmark(args: BenchmarkArgs) -> Result<()> {
    if args.repeats == 0 {
        bail!("--repeats must be positive");
    }
    if args.ending_checkpoint.is_some() && args.repeats != 1 {
        bail!("--ending-checkpoint requires --repeats 1");
    }
    if args
        .seconds
        .is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0)
    {
        bail!("--seconds must be finite and positive");
    }
    if args.steps == Some(0) {
        bail!("--steps must be positive");
    }
    let system = load_system(&args.common.input)?;
    let mut checkpoint = args
        .starting_checkpoint
        .as_ref()
        .map(|path| {
            let json = fs::read_to_string(path)
                .with_context(|| format!("reading starting checkpoint {}", path.display()))?;
            RuntimeCheckpoint::decode(&json).map_err(|e| anyhow::anyhow!(e.to_string()))
        })
        .transpose()?;
    let mut protocol = if let Some(checkpoint) = &checkpoint {
        checkpoint.state.protocol.clone()
    } else {
        apply_cli_overrides(
            load_protocol(
                args.common.protocol.as_deref(),
                args.common.gromacs_mdp.as_deref(),
                &system,
            )?,
            args.common.minimization_iterations,
            args.common.seed,
        )
    };
    extend_benchmark_schedule(&mut protocol)?;
    if let Some(checkpoint) = &mut checkpoint {
        // Keep the caller's canonical coordinates, velocities, RNG and
        // model identity, but extend only this in-memory schedule so warmup
        // and timed windows cannot be truncated by a short source protocol.
        checkpoint.state.protocol = protocol.clone();
    }
    let _selection = choose_backend(args.common.backend, &protocol)?;
    let threads = configure_threads(args.common.threads)?;
    let (memory, budget) = memory_policy(args.common.gpu_memory, args.common.gpu_memory_mib)?;
    let mut options = execution_options(
        args.common.backend,
        memory,
        threads,
        args.common.implicit_gpu_lanes_per_target.count(),
        args.common.implicit_gpu_packet_steps.count(),
        args.common.explicit_gpu_tiled_nonbonded,
        args.common.explicit_gpu_neighbor_skin_angstrom,
    );
    options.memory_budget_bytes = budget;
    options.profile_gpu_timing = matches!(args.profile_timing, BenchmarkTiming::Gpu);

    let initial_setup = Instant::now();
    let canonical = if let Some(checkpoint) = checkpoint {
        checkpoint
    } else {
        let initial = construct(system.clone(), protocol, options.clone())?;
        initial
            .checkpoint()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
    };
    let initial_setup_seconds = initial_setup.elapsed().as_secs_f64();
    let fixed_steps = args.steps.unwrap_or(100);
    let mut records = Vec::with_capacity(args.repeats);
    let mut total_restore_seconds = 0.0;
    for repeat in 0..args.repeats {
        let restore_started = Instant::now();
        let mut session = pollster::block_on(ExecutionSession::from_checkpoint(
            system.clone(),
            canonical.clone(),
            options.clone(),
        ))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let restore_seconds = restore_started.elapsed().as_secs_f64();
        total_restore_seconds += restore_seconds;
        let warmup_completed = advance_exact_steps(&mut session, args.warmup_steps)?;
        let diagnostics_before_measurement = session.diagnostics().clone();
        let initial_step = session.state().step;
        let simulation_started = Instant::now();
        let measured_steps = if let Some(seconds) = args.seconds {
            let duration = std::time::Duration::from_secs_f64(seconds);
            let mut completed = 0usize;
            let mut measured_steps_per_second = None;
            while simulation_started.elapsed() < duration {
                let before = session.state().step;
                if before >= session.state().protocol.total_steps() {
                    bail!("starting checkpoint schedule ended during timed benchmark");
                }
                let elapsed_seconds = simulation_started.elapsed().as_secs_f64();
                let count = timed_benchmark_batch_steps(
                    preferred_batch_steps(&session)
                        .min(session.state().protocol.total_steps() - before),
                    elapsed_seconds,
                    args.seconds.unwrap_or_default(),
                    measured_steps_per_second,
                );
                let packet_started = Instant::now();
                let result = pollster::block_on(session.advance(AdvanceRequest {
                    steps: count,
                    include_checkpoint: false,
                    max_frames: Some(0),
                }))
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let advanced = result.chunk.last_step.saturating_sub(before);
                if advanced == 0 {
                    bail!("timed benchmark made no progress at step {before}");
                }
                completed += advanced;
                let packet_seconds = packet_started.elapsed().as_secs_f64();
                if packet_seconds > 0.0 {
                    let observed = advanced as f64 / packet_seconds;
                    measured_steps_per_second = Some(match measured_steps_per_second {
                        Some(previous) => 0.5 * previous + 0.5 * observed,
                        None => observed,
                    });
                }
            }
            completed
        } else {
            advance_exact_steps(&mut session, fixed_steps)?
        };
        if let Some(path) = args.ending_checkpoint.as_ref() {
            let checkpoint = session
                .checkpoint()
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, serde_json::to_vec(&checkpoint)?)?;
        }
        let simulation_seconds = simulation_started.elapsed().as_secs_f64();
        let simulated_ps = measured_steps as f64 * session.state().protocol.timestep_fs * 0.001;
        let measured_diagnostics =
            diagnostics_between(session.diagnostics(), &diagnostics_before_measurement);
        records.push(serde_json::json!({
            "repeat": repeat + 1,
            "backend": session.diagnostics().actual_backend,
            "fallbackReason": session.diagnostics().fallback_reason,
            "startStep": initial_step,
            "warmupSteps": warmup_completed,
            "steps": measured_steps,
            "timestepFs": session.state().protocol.timestep_fs,
            "simulatedPs": simulated_ps,
            "restoreSeconds": restore_seconds,
            "simulationSeconds": simulation_seconds,
            "stepsPerSecond": measured_steps as f64 / simulation_seconds.max(f64::MIN_POSITIVE),
            "psPerDay": simulated_ps / simulation_seconds.max(f64::MIN_POSITIVE) * 86400.0,
            "nsPerDay": simulated_ps / 1000.0 / simulation_seconds.max(f64::MIN_POSITIVE) * 86400.0,
            "diagnostics": measured_diagnostics,
        }));
    }
    let mut rates: Vec<f64> = records
        .iter()
        .filter_map(|record| record["nsPerDay"].as_f64())
        .collect();
    let median_ns_per_day = median(&mut rates);
    let result = serde_json::json!({
        "schemaVersion": 1,
        "atoms": system.atom_count(),
        "threads": threads,
        "requestedBackend": args.common.backend,
        "warmupSteps": args.warmup_steps,
        "fixedSteps": args.steps,
        "durationSeconds": args.seconds,
        "repeats": records,
        "medianNsPerDay": median_ns_per_day,
        "initialSetupSeconds": initial_setup_seconds,
        "meanRestoreSeconds": total_restore_seconds / args.repeats as f64,
        "profileTiming": args.profile_timing,
        "gpuStageTimingStatus": records.iter().find_map(|record| {
            record["diagnostics"]["gpuStageTimingStatus"].as_str()
        }).unwrap_or("not-requested"),
        "startingCheckpoint": args.starting_checkpoint,
        "protocol": canonical.state.protocol,
        "modelVersion": canonical.state.model_version,
        "velocityConvention": canonical.state.velocity_convention,
        "outputPolicy": "coordinate frames suppressed; full-state synchronization occurs at each native advance boundary",
    });
    let json = serde_json::to_vec_pretty(&result)?;
    if let Some(path) = args.json_output {
        fs::write(&path, &json)
            .with_context(|| format!("writing benchmark JSON {}", path.display()))?;
    }
    println!("{}", String::from_utf8(json).expect("JSON is UTF-8"));
    Ok(())
}

#[cfg(test)]
mod scheduling_tests {
    use super::*;

    #[test]
    fn duration_benchmark_uses_a_short_probe_then_adapts_to_remaining_time() {
        assert_eq!(timed_benchmark_batch_steps(2500, 0.0, 5.0, None), 8);
        assert_eq!(
            timed_benchmark_batch_steps(2500, 0.0, 5.0, Some(100.0)),
            450
        );
        assert_eq!(
            timed_benchmark_batch_steps(2500, 4.5, 5.0, Some(1000.0)),
            450
        );
        assert_eq!(
            timed_benchmark_batch_steps(128, 4.5, 5.0, Some(1000.0)),
            128
        );
        assert_eq!(timed_benchmark_batch_steps(2500, 4.5, 5.0, Some(0.0)), 8);
    }

    #[test]
    fn artificial_128_step_backend_hits_exact_checkpoint_and_frame_steps() {
        let protocol = SimulationProtocol {
            equilibration_steps: 0,
            production_steps: 5000,
            save_every: 2500,
            ..Default::default()
        };
        let mut step = 0usize;
        let mut checkpoints = Vec::new();
        while step < protocol.total_steps() {
            let request = scheduled_advance_limit(&protocol, step, 128, 2500, None);
            assert!(request <= 128, "artificial backend cap must be respected");
            step += request;
            if checkpoint_due_at(&protocol, step, 2500) {
                checkpoints.push(step);
            }
        }
        assert_eq!(checkpoints, [2500, 5000]);
    }

    #[test]
    fn partial_backend_advances_do_not_checkpoint_before_actual_boundary() {
        let protocol = SimulationProtocol {
            equilibration_steps: 0,
            production_steps: 2600,
            save_every: 500,
            ..Default::default()
        };
        assert_eq!(scheduled_advance_limit(&protocol, 2499, 128, 2500, None), 1);
        assert!(!checkpoint_due_at(&protocol, 2499, 2500));
        assert!(checkpoint_due_at(&protocol, 2500, 2500));
        assert!(checkpoint_due_at(&protocol, 2600, 2500));

        let mut step = 2464usize;
        let mut reached = Vec::new();
        while step < 2600 {
            let requested = scheduled_advance_limit(&protocol, step, 128, 2500, None);
            // Simulate a backend that may return a shorter committed chunk.
            step += requested.min(17);
            if checkpoint_due_at(&protocol, step, 2500) {
                reached.push(step);
            }
        }
        assert_eq!(reached, [2500, 2600]);
    }

    #[test]
    fn stage_boundary_limits_scheduling_and_resume_before_boundary() {
        let protocol = SimulationProtocol {
            equilibration_steps: 0,
            production_steps: 3000,
            save_every: 1000,
            stages: Some(vec![
                glysys_dynamics::SimulationStage {
                    id: "stage-a".into(),
                    ensemble: Ensemble::Nvt,
                    steps: 127,
                    barostat_adaptation: false,
                },
                glysys_dynamics::SimulationStage {
                    id: "stage-b".into(),
                    ensemble: Ensemble::Nvt,
                    steps: 2873,
                    barostat_adaptation: false,
                },
            ]),
            ..Default::default()
        };
        assert_eq!(scheduled_advance_limit(&protocol, 126, 128, 2500, None), 1);
        assert!(checkpoint_due_at(&protocol, 127, 2500));
        assert_eq!(scheduled_advance_limit(&protocol, 2999, 128, 2500, None), 1);
        assert!(checkpoint_due_at(&protocol, 3000, 2500));
    }

    #[test]
    fn scalar_observable_interval_is_an_exact_advance_boundary() {
        let protocol = SimulationProtocol {
            equilibration_steps: 0,
            production_steps: 2600,
            save_every: 2500,
            ..Default::default()
        };
        assert_eq!(
            scheduled_advance_limit(&protocol, 95, 128, 2500, Some(100)),
            5
        );
        assert_eq!(
            scheduled_advance_limit(&protocol, 100, 128, 2500, Some(100)),
            100
        );
    }

    #[test]
    fn benchmark_schedule_extends_only_the_in_memory_production_window() {
        let mut protocol = SimulationProtocol {
            solvent: SolventModel::Implicit,
            constraints: glysys_dynamics::ConstraintModel::HBonds,
            thermostat: glysys_dynamics::Thermostat::Langevin,
            langevin_discretization: glysys_dynamics::LangevinDiscretization::LfMiddle,
            timestep_fs: 2.0,
            equilibration_steps: 4000,
            production_steps: 150_000,
            save_every: 2500,
            ..Default::default()
        };
        let saved = protocol.clone();
        extend_benchmark_schedule(&mut protocol).unwrap();
        assert_eq!(protocol.equilibration_steps, 4000);
        assert_eq!(
            protocol.production_steps,
            glysys_dynamics::MAX_NATIVE_STEPS - 4000
        );
        assert_eq!(protocol.save_every, usize::MAX);
        assert_eq!(saved.production_steps, 150_000);
        assert_eq!(saved.save_every, 2500);
    }
}
