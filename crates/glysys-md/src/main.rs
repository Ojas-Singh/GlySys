use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use glysys::ParameterizedSystem;
use glysys_dynamics::{Ensemble, SimulationProtocol, SolventModel};
use glysys_runtime::{
    AdvanceRequest, BackendPreference, ExecutionOptions, ExecutionSession, MemoryPolicy,
    RuntimeCheckpoint,
};

mod bundle;
mod gromacs;

#[derive(Debug, Clone, Copy, ValueEnum)]
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
}

#[derive(Debug, Args, Clone)]
struct ResumeArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Existing run directory containing `checkpoint.json`.
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Debug, Args, Clone)]
struct BenchmarkArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, default_value_t = 100)]
    steps: usize,
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

fn execution_options(backend: Backend, memory: MemoryPolicy, threads: usize) -> ExecutionOptions {
    ExecutionOptions {
        backend: match backend {
            Backend::Auto => BackendPreference::Auto,
            Backend::Cpu => BackendPreference::Cpu,
            Backend::Vulkan => BackendPreference::Gpu,
        },
        memory_policy: memory,
        cpu_thread_limit: Some(threads),
        ..ExecutionOptions::default()
    }
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
) -> Result<()> {
    let state = session.state();
    let diagnostics = session.diagnostics();
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
        "modelVersion": state.model_version,
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
    output: &Path,
    threads: usize,
    started: Instant,
    requested_backend: &str,
) -> Result<()> {
    fs::create_dir_all(output)?;
    let trajectory = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output.join("trajectory.jsonl"))?;
    let mut trajectory = BufWriter::new(trajectory);
    write_checkpoint(
        output,
        &session
            .checkpoint()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
    )?;
    while session.state().step < session.state().protocol.total_steps() {
        let remaining = session.state().protocol.total_steps() - session.state().step;
        let result = match pollster::block_on(session.advance(AdvanceRequest {
            steps: remaining.min(100),
            include_checkpoint: true,
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
        for frame in result.chunk.frames {
            serde_json::to_writer(&mut trajectory, &frame)?;
            trajectory.write_all(b"\n")?;
        }
        trajectory.flush()?;
        if let Some(checkpoint) = result.checkpoint {
            write_checkpoint(output, &checkpoint)?;
        }
        eprintln!(
            "step={}/{} backend={} threads={threads}",
            session.state().step,
            session.state().protocol.total_steps(),
            session.diagnostics().actual_backend
        );
    }
    write_metadata(output, threads, &session, started)?;
    Ok(())
}

fn run(args: RunArgs) -> Result<()> {
    if args.output.join("checkpoint.json").exists() || args.output.join("trajectory.jsonl").exists()
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
    let mut options = execution_options(args.common.backend, memory, threads);
    options.memory_budget_bytes = budget;
    let started = Instant::now();
    let session = construct(system, protocol, options)?;
    run_loop(
        session,
        &args.output,
        threads,
        started,
        match args.common.backend {
            Backend::Auto => "auto",
            Backend::Cpu => "cpu",
            Backend::Vulkan => "vulkan",
        },
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
    let mut options = execution_options(args.common.backend, memory, threads);
    options.memory_budget_bytes = budget;
    let started = Instant::now();
    let session = pollster::block_on(ExecutionSession::from_checkpoint(
        system,
        runtime_checkpoint,
        options,
    ))
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    run_loop(
        session,
        &args.output,
        threads,
        started,
        match args.common.backend {
            Backend::Auto => "auto",
            Backend::Cpu => "cpu",
            Backend::Vulkan => "vulkan",
        },
    )
}

fn benchmark(args: BenchmarkArgs) -> Result<()> {
    let system = load_system(&args.common.input)?;
    let protocol = apply_cli_overrides(
        load_protocol(
            args.common.protocol.as_deref(),
            args.common.gromacs_mdp.as_deref(),
            &system,
        )?,
        args.common.minimization_iterations,
        args.common.seed,
    );
    let _selection = choose_backend(args.common.backend, &protocol)?;
    let threads = configure_threads(args.common.threads)?;
    let (memory, budget) = memory_policy(args.common.gpu_memory, args.common.gpu_memory_mib)?;
    let mut options = execution_options(args.common.backend, memory, threads);
    options.memory_budget_bytes = budget;
    let started = Instant::now();
    let mut session = construct(system.clone(), protocol, options)?;
    let setup_seconds = started.elapsed().as_secs_f64();
    let simulation_started = Instant::now();
    let mut remaining = args.steps;
    while remaining > 0 {
        let count = remaining.min(100);
        pollster::block_on(session.advance(AdvanceRequest {
            steps: count,
            include_checkpoint: false,
            max_frames: Some(0),
        }))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        remaining -= count;
    }
    let seconds = simulation_started.elapsed().as_secs_f64();
    let ps = args.steps as f64 * session.state().protocol.timestep_fs * 0.001;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "backend": session.diagnostics().actual_backend,
            "fallbackReason": session.diagnostics().fallback_reason,
            "threads": threads,
            "atoms": system.atom_count(),
            "steps": args.steps,
            "simulatedPs": ps,
            "setupSeconds": setup_seconds,
            "simulationSeconds": seconds,
            "stepsPerSecond": args.steps as f64 / seconds.max(f64::MIN_POSITIVE),
            "psPerDay": ps / seconds.max(f64::MIN_POSITIVE) * 86400.0,
        }))?
    );
    Ok(())
}
