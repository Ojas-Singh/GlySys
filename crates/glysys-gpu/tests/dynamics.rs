use glysys::{BuildOptions, SystemBuilder};
use glysys_dynamics::{CpuSimulation, SimulationProtocol, normal_noise};
#[test]
fn gpu_baoab_matches_cpu_steps_and_force_reference() {
    pollster::block_on(async {
        let system = SystemBuilder::new(BuildOptions {
            add_water: false,
            add_ions: false,
            ..Default::default()
        })
        .unwrap()
        .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))
        .unwrap();
        for friction in [0., 1.] {
            let protocol = SimulationProtocol {
                minimization_iterations: 30,
                equilibration_steps: 0,
                production_steps: 20,
                friction_per_ps: friction,
                ..Default::default()
            };
            let mut cpu = CpuSimulation::new(&system, protocol).unwrap();
            let start = cpu.state.clone();
            let mut rng = start.rng_state;
            let noise = normal_noise(&mut rng, system.atom_count() * 20);
            let mut gpu = glysys_gpu::dynamics::ResidentDynamics::new(&system)
                .await
                .unwrap();
            let output = gpu
                .advance(
                    &start.coordinates,
                    &start.velocities,
                    &noise,
                    20,
                    0.0005,
                    300.,
                    friction,
                )
                .await
                .unwrap();
            cpu.advance(20).unwrap();
            assert_eq!(rng, cpu.state.rng_state);
            for (a, b) in output.coordinates.iter().zip(&cpu.state.coordinates) {
                for (x, y) in [(a.x, b.x), (a.y, b.y), (a.z, b.z)] {
                    assert!((x - y).abs() < 0.001, "coordinate {x} {y}");
                }
            }
            let reference = glysys_energy::EnergyEvaluator::new(
                &system,
                glysys_energy::EnergyOptions {
                    obc2: Some(Default::default()),
                    ..Default::default()
                },
            )
            .unwrap()
            .energy_and_gradient(&output.coordinates)
            .unwrap();
            let energy: f64 = output.components[..9].iter().map(|x| *x as f64).sum();
            assert!((energy - reference.total()).abs() < 1e-3 + 1e-4 * reference.total().abs());
            for (a, b) in output.gradients.iter().zip(reference.gradients.unwrap()) {
                for (x, y) in [(a.x, b.x), (a.y, b.y), (a.z, b.z)] {
                    assert!((x - y).abs() < 1e-3 + 1e-3 * y.abs(), "gradient {x} {y}");
                }
            }
        }
    });
}
