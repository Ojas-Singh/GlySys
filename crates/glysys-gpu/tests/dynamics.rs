use glysys::{BuildOptions, SystemBuilder};
use glysys_dynamics::{CpuSimulation, SimulationProtocol, normal_noise};
use wgpu::util::DeviceExt;

#[test]
fn resident_gpu_thermostat_noise_has_unit_gaussian_variance() {
    pollster::block_on(async {
        const SEEDS: usize = 65_536;
        let words = glysys_dynamics::resident_rng::ResidentThermostatRng::seeded(17, SEEDS).words;
        let context = glysys_gpu::GpuContext::new(glysys_gpu::GpuContextOptions::default())
            .await
            .unwrap();
        let device = context.device();
        let queue = context.queue();
        let seed_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("thermostat RNG test seeds"),
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("thermostat RNG test samples"),
            size: (SEEDS * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("thermostat RNG test readback"),
            size: (SEEDS * 16) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let source = format!(
            "{}\n\
             @group(0) @binding(0) var<storage, read> seeds: array<u32>;\n\
             @group(0) @binding(1) var<storage, read_write> samples: array<vec4<f32>>;\n\
             @compute @workgroup_size(64)\n\
             fn sample_noise(@builtin(global_invocation_id) id: vec3<u32>) {{\n\
               if (id.x < arrayLength(&seeds)) {{ samples[id.x] = rng_normal3(seeds[id.x]); }}\n\
             }}",
            include_str!("../src/resident_rng.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("thermostat RNG sampling test"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("thermostat RNG sampling test"),
            layout: None,
            module: &shader,
            entry_point: Some("sample_noise"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thermostat RNG sampling test"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: seed_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((SEEDS as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, (SEEDS * 16) as u64);
        queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::PollType::Wait).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, f32>(&mapped);
        let mut sum = 0.0f64;
        let mut square = 0.0f64;
        let mut fourth = 0.0f64;
        let mut count = 0.0f64;
        for triple in values.chunks_exact(4) {
            for value in &triple[..3] {
                let value = f64::from(*value);
                sum += value;
                square += value * value;
                fourth += value.powi(4);
                count += 1.0;
            }
        }
        drop(mapped);
        let mean = sum / count;
        let variance = square / count - mean * mean;
        let fourth_moment = fourth / count;
        assert!(mean.abs() < 0.01, "GPU noise mean {mean}");
        assert!(
            (variance - 1.0).abs() < 0.015,
            "GPU noise variance {variance}"
        );
        assert!(
            (fourth_moment - 3.0).abs() < 0.10,
            "GPU noise fourth moment {fourth_moment}"
        );
    });
}

#[test]
fn resident_gpu_free_ou_thermostat_samples_the_requested_temperature() {
    pollster::block_on(async {
        const SEEDS: usize = 1_024;
        const STEPS: u32 = 10_000;
        const BURN_IN: u32 = 1_000;
        const KB: f32 = 0.00198720425864083;
        const ACCEL: f32 = 418.4;
        const TARGET_K: f32 = 300.0;
        let words = glysys_dynamics::resident_rng::ResidentThermostatRng::seeded(41, SEEDS).words;
        let context = glysys_gpu::GpuContext::new(glysys_gpu::GpuContextOptions::default())
            .await
            .unwrap();
        let device = context.device();
        let queue = context.queue();
        let seed_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("free OU test seeds"),
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_size = (SEEDS * 2 * 16) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("free OU test statistics"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("free OU test readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let source = format!(
            "{}\n\
             @group(0) @binding(0) var<storage, read> seeds: array<u32>;\n\
             @group(0) @binding(1) var<storage, read_write> stats: array<vec4<f32>>;\n\
             @compute @workgroup_size(64)\n\
             fn sample_ou(@builtin(global_invocation_id) id: vec3<u32>) {{\n\
               if (id.x >= arrayLength(&seeds)) {{ return; }}\n\
               let decay = exp(-0.001);\n\
               let sigma = sqrt((1.0 - decay * decay) * 0.00198720425864083 * 300.0 * 418.4);\n\
               var seed = seeds[id.x];\n\
               var velocity = vec3<f32>(0.0);\n\
               var normal_sum = 0.0; var normal_square = 0.0; var normal_fourth = 0.0;\n\
               var velocity_square = 0.0;\n\
               for (var step = 0u; step < 10000u; step++) {{\n\
                 let draw = rng_normal3(seed);\n\
                 seed = bitcast<u32>(draw.w);\n\
                 velocity = decay * velocity + sigma * draw.xyz;\n\
                 if (step >= 1000u) {{\n\
                   normal_sum += draw.x + draw.y + draw.z;\n\
                   normal_square += dot(draw.xyz, draw.xyz);\n\
                   normal_fourth += pow(draw.x, 4.0) + pow(draw.y, 4.0) + pow(draw.z, 4.0);\n\
                   velocity_square += dot(velocity, velocity);\n\
                 }}\n\
               }}\n\
               stats[2u * id.x] = vec4<f32>(normal_sum, normal_square, normal_fourth, 0.0);\n\
               stats[2u * id.x + 1u] = vec4<f32>(velocity_square, 0.0, 0.0, 0.0);\n\
             }}",
            include_str!("../src/resident_rng.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("free OU thermostat test"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("free OU thermostat test"),
            layout: None,
            module: &shader,
            entry_point: Some("sample_ou"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("free OU thermostat test"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: seed_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((SEEDS as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_size);
        queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::PollType::Wait).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, f32>(&mapped);
        let mut normal_sum = 0.0f64;
        let mut normal_square = 0.0f64;
        let mut normal_fourth = 0.0f64;
        let mut velocity_square = 0.0f64;
        for atom in 0..SEEDS {
            normal_sum += f64::from(values[atom * 8]);
            normal_square += f64::from(values[atom * 8 + 1]);
            normal_fourth += f64::from(values[atom * 8 + 2]);
            velocity_square += f64::from(values[atom * 8 + 4]);
        }
        drop(mapped);
        let samples = (SEEDS * (STEPS - BURN_IN) as usize * 3) as f64;
        let mean = normal_sum / samples;
        let variance = normal_square / samples - mean * mean;
        let fourth_moment = normal_fourth / samples;
        let temperature = velocity_square / samples / (f64::from(KB) * f64::from(ACCEL));
        eprintln!(
            "free OU GPU: mean={mean:.5}, variance={variance:.5}, fourth={fourth_moment:.5}, T={temperature:.3} K"
        );
        assert!(mean.abs() < 0.01, "GPU OU noise mean {mean}");
        assert!(
            (variance - 1.0).abs() < 0.015,
            "GPU OU noise variance {variance}"
        );
        assert!(
            (fourth_moment - 3.0).abs() < 0.10,
            "GPU OU noise fourth moment {fourth_moment}"
        );
        assert!(
            (temperature - f64::from(TARGET_K)).abs() < 8.0,
            "free-particle GPU OU temperature {temperature} K"
        );
    });
}

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

#[test]
fn resident_lf_middle_projects_implicit_xh_constraints_at_two_femtoseconds() {
    pollster::block_on(async {
        let system = SystemBuilder::new(BuildOptions {
            add_water: false,
            add_ions: false,
            ..Default::default()
        })
        .unwrap()
        .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))
        .unwrap();
        let protocol = SimulationProtocol {
            minimization_iterations: 30,
            equilibration_steps: 0,
            production_steps: 8,
            save_every: 2,
            seed: 29,
            solvent: glysys_dynamics::SolventModel::Implicit,
            constraints: glysys_dynamics::ConstraintModel::HBonds,
            thermostat: glysys_dynamics::Thermostat::Langevin,
            langevin_discretization: glysys_dynamics::LangevinDiscretization::LfMiddle,
            timestep_fs: 2.0,
            ..Default::default()
        };
        let mut cpu = CpuSimulation::new(&system, protocol).unwrap();
        let start = cpu.state.clone();
        let context = glysys_gpu::GpuContext::new(glysys_gpu::GpuContextOptions::default())
            .await
            .unwrap();
        let mut gpu = glysys_gpu::dynamics::ResidentDynamics::with_context(&system, &context)
            .await
            .unwrap();
        let rng = start.resident_rng.as_ref().unwrap().words.clone();
        gpu.initialize_resident(&start.coordinates, &start.velocities, &rng)
            .await
            .unwrap();
        let output = gpu
            .advance_resident_lf_middle(
                &start.coordinates,
                &start.velocities,
                &rng,
                4,
                0.002,
                300.0,
                1.0,
            )
            .await
            .unwrap();
        cpu.advance(4).unwrap();
        assert_eq!(output.coordinates.len(), system.atom_count());
        assert_eq!(
            output.rng_words.as_ref().unwrap().len(),
            system.atom_count()
        );
        assert_ne!(output.rng_words.as_ref().unwrap(), &rng);
        assert_eq!(
            output.rng_words.as_ref().unwrap(),
            &cpu.state.resident_rng.as_ref().unwrap().words,
            "the persistent per-atom RNG streams must agree across CPU and GPU"
        );
        let max_position_delta = output
            .coordinates
            .iter()
            .zip(&cpu.state.coordinates)
            .map(|(gpu, cpu)| {
                ((gpu.x - cpu.x).powi(2) + (gpu.y - cpu.y).powi(2) + (gpu.z - cpu.z).powi(2)).sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(
            max_position_delta < 0.05,
            "CPU/GPU 2 fs trajectory diverged by {max_position_delta} A after four steps"
        );
        assert!(
            output
                .gradients
                .iter()
                .all(|g| { g.x.is_finite() && g.y.is_finite() && g.z.is_finite() })
        );
        for bond in system.bonds() {
            let [a, b] = bond.atoms();
            if system.atoms()[a].element() == 1 || system.atoms()[b].element() == 1 {
                let delta = [
                    output.coordinates[a].x - output.coordinates[b].x,
                    output.coordinates[a].y - output.coordinates[b].y,
                    output.coordinates[a].z - output.coordinates[b].z,
                ];
                let distance =
                    (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
                let relative = (distance - bond.length()).abs() / bond.length();
                assert!(relative <= 2e-5, "X-H constraint residual {relative:e}");
            }
        }
    });
}
