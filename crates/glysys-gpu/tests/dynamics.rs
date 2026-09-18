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
            // Explicitly exercise the compatibility adapter for older restarts.
            cpu.state.resident_rng = None;
            let start = cpu.state.clone();
            let mut rng = start.rng_state;
            let noise = normal_noise(&mut rng, system.atom_count() * 20);
            let context = glysys_gpu::GpuContext::new(glysys_gpu::GpuContextOptions::default())
                .await
                .unwrap();
            let mut gpu = glysys_gpu::dynamics::ResidentDynamics::with_context(&system, &context)
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

#[test]
fn resident_baoab_preserves_rng_across_batches_and_restart() {
    pollster::block_on(async {
        let system = SystemBuilder::new(BuildOptions {
            add_water: false,
            add_ions: false,
            ..Default::default()
        })
        .unwrap()
        .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))
        .unwrap();
        let mut cpu = CpuSimulation::new(
            &system,
            SimulationProtocol {
                minimization_iterations: 30,
                equilibration_steps: 0,
                production_steps: 32,
                ..Default::default()
            },
        )
        .unwrap();
        let start = cpu.state.clone();
        let context = glysys_gpu::GpuContext::new(glysys_gpu::GpuContextOptions::default())
            .await
            .unwrap();
        let mut gpu = glysys_gpu::dynamics::ResidentDynamics::with_context(&system, &context)
            .await
            .unwrap();
        let first = gpu
            .advance_resident(
                &start.coordinates,
                &start.velocities,
                &start.resident_rng.as_ref().unwrap().words,
                8,
                0.0005,
                300.,
                1.,
            )
            .await
            .unwrap();
        let resumed_context = glysys_gpu::GpuContext::new(glysys_gpu::GpuContextOptions::default())
            .await
            .unwrap();
        let mut resumed =
            glysys_gpu::dynamics::ResidentDynamics::with_context(&system, &resumed_context)
                .await
                .unwrap();
        let reference = resumed
            .advance_resident(
                &first.coordinates,
                &first.velocities,
                first.rng_words.as_ref().unwrap(),
                24,
                0.0005,
                300.,
                1.,
            )
            .await
            .unwrap();
        // Supplied host state is intentionally stale: a resident session must
        // neither reinstall it nor recompute its starting force each batch.
        let output = gpu
            .advance_resident(
                &start.coordinates,
                &start.velocities,
                &start.resident_rng.as_ref().unwrap().words,
                24,
                0.0005,
                300.,
                1.,
            )
            .await
            .unwrap();
        cpu.advance(32).unwrap();
        assert_eq!(
            output.rng_words.as_ref().unwrap(),
            &cpu.state.resident_rng.as_ref().unwrap().words
        );
        assert_eq!(output.rng_words, reference.rng_words);
        for ((a, b), r) in output
            .coordinates
            .iter()
            .zip(&cpu.state.coordinates)
            .zip(&reference.coordinates)
        {
            for (x, y, z) in [(a.x, b.x, r.x), (a.y, b.y, r.y), (a.z, b.z, r.z)] {
                assert!((x - y).abs() < 1e-3, "CPU {x} {y}");
                assert!((x - z).abs() < 1e-4, "restart {x} {z}");
            }
        }
    });
}
