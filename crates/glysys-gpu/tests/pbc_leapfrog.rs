//! The resident leap-frog integrator with Nose-Hoover and Parrinello-Rahman
//! coupling against `ExplicitSimulation::leapfrog_step`: both start from the
//! same positions, half-step velocities and forces, and must stay together
//! step for step while single precision allows it, with the same thermostat
//! frictions, pressures and box.
use glysys::{BuildOptions, ParameterizedSystem, SystemBuilder, Vec3};
use glysys_dynamics::coupling::CouplingPlan;
use glysys_dynamics::explicit::{ExplicitSimulation, pme_parameters, rf_backend};
use glysys_dynamics::{
    ConstraintModel, ElectrostaticsModel, Ensemble, PressureCoupling, SimulationProtocol,
    SimulationStage, SolventModel, Thermostat, ThermostatGroup,
};
use glysys_energy::pbc::NonbondedElectrostatics;
use glysys_gpu::pbc::{PbcKernel, PbcPacking, ResidentPbc};
use glysys_gpu::pbc_leapfrog::{LeapfrogCoupling, LeapfrogVariables};
use glysys_gpu::{GpuContext, GpuContextOptions};
use std::sync::{Mutex, OnceLock};

const CUTOFF: f64 = 6.0;
const SKIN: f64 = 1.5;
const KB: f64 = 0.00198720425864083;

// Device-owning tests of one binary run one at a time (see tests/pbc.rs).
static GPU_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn gpu_test_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU_TEST_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn solvated_dipeptide() -> ParameterizedSystem {
    SystemBuilder::new(BuildOptions {
        add_water: true,
        add_ions: false,
        padding_angstrom: 9.0,
        ..Default::default()
    })
    .unwrap()
    .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))
    .unwrap()
}

fn protocol(electrostatics: ElectrostaticsModel, ensemble: Ensemble) -> SimulationProtocol {
    let group = |name: &str| ThermostatGroup {
        name: name.into(),
        tau_ps: 0.5,
        reference_temperature_k: 300.0,
    };
    SimulationProtocol {
        temperature_k: 300.0,
        timestep_fs: 2.0,
        solvent: SolventModel::Explicit,
        constraints: ConstraintModel::Settle,
        thermostat: Thermostat::NoseHoover,
        electrostatics,
        pressure_coupling: if ensemble == Ensemble::Npt {
            PressureCoupling::ParrinelloRahman
        } else {
            PressureCoupling::None
        },
        cutoff_angstrom: Some(CUTOFF),
        dispersion_correction: true,
        equilibration_steps: 0,
        production_steps: 10_000,
        equilibration_ensemble: ensemble,
        production_ensemble: ensemble,
        stages: Some(vec![SimulationStage {
            id: "production".into(),
            ensemble,
            steps: 10_000,
            barostat_adaptation: false,
        }]),
        thermostat_groups: vec![group("WAT"), group("System_&_!WAT")],
        pressure_tau_ps: Some(2.0),
        pressure_compressibility_bar_inverse: vec![4.5e-5],
        com_mode: Some("linear".into()),
        com_groups: vec!["WAT".into(), "System_&_!WAT".into()],
        // short intervals, so a short run meets every kind of step
        temperature_coupling_interval: Some(5),
        pressure_coupling_interval: Some(4),
        com_removal_interval: Some(10),
        minimization_iterations: 200,
        save_every: 1_000_000,
        seed: 11,
        ..SimulationProtocol::default()
    }
}

/// The CPU coupling plan as the numbers the resident integrator takes.
fn coupling(plan: &CouplingPlan, masses: &[f64], ensemble: Ensemble, tail: f64) -> LeapfrogCoupling {
    let mut group_bits = plan.group_of_atom.clone();
    let mut com_mass = [0.0; 2];
    for (index, group) in plan.com_groups.iter().enumerate() {
        for &atom in group {
            group_bits[atom] |= (index as u32) << 1;
            com_mass[index] += masses[atom];
        }
    }
    LeapfrogCoupling {
        group_bits,
        inverse_q: [0, 1].map(|g| plan.groups[g].inverse_mass()),
        reference_temperature_k: [0, 1].map(|g| plan.groups[g].reference_temperature_k),
        degrees_of_freedom: [0, 1].map(|g| plan.groups[g].degrees_of_freedom),
        com_mass,
        timestep_ps: 0.002,
        thermostat: ensemble != Ensemble::Nve,
        barostat: ensemble == Ensemble::Npt,
        temperature_interval: plan.temperature_interval,
        pressure_interval: plan.pressure_interval,
        com_interval: plan.com_interval,
        barostat_coefficient: 4.0 * std::f64::consts::PI.powi(2) * plan.compressibility_per_bar
            / (3.0 * plan.pressure_tau_ps * plan.pressure_tau_ps),
        reference_pressure_bar: plan.reference_pressure_bar,
        dispersion_pressure_coefficient: tail,
        box_change_allowance: 0.05,
    }
}

struct Pair {
    cpu: ExplicitSimulation<'static>,
    gpu: ResidentPbc,
    plan: CouplingPlan,
}

/// A CPU simulation at step 0 and a resident evaluator holding the same
/// state, forces included.
async fn start(electrostatics: ElectrostaticsModel, ensemble: Ensemble) -> Pair {
    let system = solvated_dipeptide();
    let protocol = protocol(electrostatics, ensemble);
    let cpu = ExplicitSimulation::new(&system, protocol.clone()).unwrap().into_owned();
    let plan = cpu.coupling_plan().unwrap().clone();
    let backend = match electrostatics {
        ElectrostaticsModel::Pme => {
            let p = pme_parameters(&system, &protocol).unwrap();
            NonbondedElectrostatics::Pme {
                alpha_per_angstrom: p.alpha_per_angstrom,
                grid: p.grid,
                interpolation_order: p.interpolation_order,
            }
        }
        ElectrostaticsModel::ReactionField => {
            let rf = rf_backend(&protocol).unwrap();
            NonbondedElectrostatics::ReactionField {
                cutoff_angstrom: rf.cutoff_angstrom,
                solvent_dielectric: rf.solvent_dielectric,
            }
        }
    };
    let packing = PbcPacking::new(&system, CUTOFF, SKIN).unwrap();
    let context = GpuContext::new(GpuContextOptions::default()).await.unwrap();
    let mut gpu = ResidentPbc::with_context_kernel(&context, &packing, &backend, 1 << 20, PbcKernel::Tiles)
        .await
        .unwrap();
    gpu.configure_leapfrog(coupling(
        &plan,
        cpu.masses(),
        ensemble,
        cpu.dispersion_pressure_coefficient(),
    ))
    .await
    .unwrap();
    gpu.set_coordinates(&cpu.state.coordinates, cpu.state.box_angstrom.map(|v| v as f32), true);
    gpu.set_velocities(&cpu.state.velocities).unwrap();
    gpu.energy_and_forces(true).await.unwrap();
    gpu.set_leapfrog_variables(&LeapfrogVariables {
        box_angstrom: cpu.state.box_angstrom,
        ..LeapfrogVariables::default()
    })
    .unwrap();
    Pair { cpu, gpu, plan }
}

fn largest_distance(a: &[Vec3], b: &[Vec3]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, q)| ((p.x - q.x).powi(2) + (p.y - q.y).powi(2) + (p.z - q.z).powi(2)).sqrt())
        .fold(0.0, f64::max)
}

/// Advance both engines to `step` and return the resident variables and the
/// largest distance between the two sets of positions.
async fn advance_to(pair: &mut Pair, done: &mut usize, step: usize) -> (LeapfrogVariables, f64) {
    pair.gpu
        .dynamics_steps_leapfrog(*done as u64, step - *done)
        .await
        .unwrap();
    assert_eq!(pair.gpu.dynamics_status().await.unwrap(), None);
    pair.cpu.advance(step - *done).unwrap();
    *done = step;
    let variables = pair.gpu.read_leapfrog_variables().await.unwrap();
    let state = pair
        .gpu
        .read_dynamics_checkpoint(variables.box_angstrom.map(|v| v as f32))
        .await
        .unwrap();
    let distance = largest_distance(&state.coordinates, &pair.cpu.state.coordinates);
    (variables, distance)
}

fn follows_in_nvt(electrostatics: ElectrostaticsModel) {
    let _guard = gpu_test_guard();
    pollster::block_on(async {
        let mut pair = start(electrostatics, Ensemble::Nvt).await;
        let mut done = 0;
        // steps 1 and 6 advance the thermostats; step 10 removes the
        // center-of-mass motion
        for step in [1, 2, 7, 12] {
            let (variables, distance) = advance_to(&mut pair, &mut done, step).await;
            assert!(distance < 2e-4 * step as f64, "{electrostatics:?} step {step}: {distance:e} A apart");
            let coupling = pair.cpu.state.coupling.as_ref().unwrap();
            for group in 0..2 {
                let (cpu, gpu) = (
                    coupling.thermostat_velocity[group],
                    variables.thermostat_velocity[group],
                );
                assert!(
                    (cpu - gpu).abs() < 2e-4 * cpu.abs().max(1.0),
                    "{electrostatics:?} step {step} group {group}: friction {cpu} on the CPU, {gpu} on the device"
                );
            }
            assert_eq!(variables.box_angstrom.map(|b| b as f32), pair.cpu.state.box_angstrom.map(|b| b as f32));
        }
    });
}

fn follows_in_npt(electrostatics: ElectrostaticsModel) {
    let _guard = gpu_test_guard();
    pollster::block_on(async {
        let mut pair = start(electrostatics, Ensemble::Npt).await;
        let start_box = pair.cpu.state.box_angstrom;
        let mut done = 0;
        // the pressures of steps 0, 4 and 8 drive the box on steps 1, 5 and 9
        for step in [1, 2, 6, 10, 13] {
            let (variables, distance) = advance_to(&mut pair, &mut done, step).await;
            assert!(distance < 3e-4 * step as f64, "{electrostatics:?} step {step}: {distance:e} A apart");
            let coupling = pair.cpu.state.coupling.as_ref().unwrap();
            for axis in 0..3 {
                let (cpu, gpu) = (pair.cpu.state.box_angstrom[axis], variables.box_angstrom[axis]);
                assert!((cpu - gpu).abs() < 2e-5 * cpu, "{electrostatics:?} step {step}: box {cpu} / {gpu}");
                let (cpu, gpu) = (coupling.box_velocity[axis], variables.box_velocity[axis]);
                assert!(
                    (cpu - gpu).abs() < 2e-3 * cpu.abs().max(1e-2),
                    "{electrostatics:?} step {step}: box velocity {cpu} / {gpu}"
                );
            }
        }
        // Step 12 was a pressure step: the device's group kinetic energies
        // are those of the velocities it stored.
        let variables = pair.gpu.read_leapfrog_variables().await.unwrap();
        let state = pair
            .gpu
            .read_dynamics_checkpoint(variables.box_angstrom.map(|v| v as f32))
            .await
            .unwrap();
        let mut expected = [0.0; 2];
        for ((v, mass), group) in state
            .velocities
            .iter()
            .zip(pair.cpu.masses())
            .zip(&pair.plan.group_of_atom)
        {
            expected[*group as usize] += mass * (v.x * v.x + v.y * v.y + v.z * v.z) / (2.0 * 418.4);
        }
        for group in 0..2 {
            let got = variables.kinetic_energy[group];
            assert!(
                (got - expected[group]).abs() < 2e-5 * expected[group],
                "group {group}: kinetic energy {got} on the device, {} from its velocities",
                expected[group]
            );
            let temperature = 2.0 * got / (pair.plan.groups[group].degrees_of_freedom * KB);
            assert!(temperature > 50.0 && temperature < 1000.0, "group {group} at {temperature} K");
        }
        // The box has moved, and the two engines agree on the last pressure
        // the barostat used (that of step 12).
        assert!((pair.cpu.state.box_angstrom[0] - start_box[0]).abs() > 1e-5);
        let (variables, _) = advance_to(&mut pair, &mut done, 14).await;
        let coupling = pair.cpu.state.coupling.as_ref().unwrap();
        assert!(
            (coupling.pressure_bar - variables.pressure_bar).abs() < 5.0 + 2e-3 * coupling.pressure_bar.abs(),
            "{electrostatics:?}: pressure {} bar on the CPU, {} on the device",
            coupling.pressure_bar,
            variables.pressure_bar
        );
    });
}

#[test]
fn reaction_field_nvt_follows_the_cpu_integrator() {
    follows_in_nvt(ElectrostaticsModel::ReactionField);
}

#[test]
fn reaction_field_npt_follows_the_cpu_integrator() {
    follows_in_npt(ElectrostaticsModel::ReactionField);
}
