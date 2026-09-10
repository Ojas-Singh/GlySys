//! BAOAB dispatches share resident force-field coordinates and gradients.
use crate::{
    device::{BUFFER_BUDGET, Config, Error, ResidentEvaluator},
    topology::PreparedTopology,
};
use glysys::{ParameterizedSystem, Vec3};
use glysys_energy::{EnergyOptions, Obc2Options};
use wgpu::util::DeviceExt;
pub const MAX_STEPS: usize = 20;
pub struct DynamicsBatch {
    pub coordinates: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub gradients: Vec<Vec3>,
    pub components: [f32; 12],
}
pub struct ResidentDynamics {
    energy: ResidentEvaluator,
    topology: PreparedTopology,
    velocities: wgpu::Buffer,
    noise: wgpu::Buffer,
    uniform: wgpu::Buffer,
    status: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind: wgpu::BindGroup,
    pipelines: [wgpu::ComputePipeline; 2],
    masses: Vec<f64>,
}
impl ResidentDynamics {
    pub async fn new(system: &ParameterizedSystem) -> Result<Self, Error> {
        let n = system.atom_count();
        if n == 0
            || system
                .atoms()
                .iter()
                .any(|a| !a.mass().is_finite() || a.mass() <= 0.)
        {
            return Err(Error::Input("positive atom masses required"));
        }
        let topology = PreparedTopology::new(
            system,
            &EnergyOptions {
                obc2: Some(Obc2Options::default()),
                ..Default::default()
            },
            &vec![0; n],
        )?;
        let energy = ResidentEvaluator::new(topology.view(), 1).await?;
        let existing = energy.buffers.iter().map(|b| b.size()).sum::<u64>() + energy.staging.size();
        let additional = n as u64 * (16 + 16 * MAX_STEPS as u64 + 96) + MAX_STEPS as u64 * 256 + 80;
        if existing + additional > BUFFER_BUDGET
            || n as u64 * 16 * MAX_STEPS as u64
                > energy.device.limits().max_storage_buffer_binding_size as u64
        {
            return Err(Error::Capacity);
        }
        let device = &energy.device;
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let alloc = |size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let velocities = alloc(
            n as u64 * 16,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let noise = alloc(
            n as u64 * 16 * MAX_STEPS as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let uniform = alloc(
            MAX_STEPS as u64 * 256,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let status = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0; 16],
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = alloc(
            n as u64 * 96 + 64,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let entries: Vec<_> = (0..6)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding == 3 || binding == 4,
                        }
                    },
                    has_dynamic_offset: binding == 0,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let buffers = [
            &uniform,
            &energy.buffers[2],
            &velocities,
            &energy.buffers[7],
            &noise,
            &status,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: if i == 0 {
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: b,
                        offset: 0,
                        size: std::num::NonZeroU64::new(32),
                    })
                } else {
                    b.as_entire_binding()
                },
            })
            .collect();
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BAOAB"),
            source: wgpu::ShaderSource::Wgsl(include_str!("dynamics.wgsl").into()),
        });
        let pipelines = ["before_force", "after_force"].map(|name| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let validation = device.pop_error_scope().await;
        let allocation = device.pop_error_scope().await;
        if let Some(e) = validation {
            return Err(Error::Execution(e.to_string()));
        }
        if allocation.is_some() {
            return Err(Error::Capacity);
        }
        Ok(Self {
            energy,
            topology,
            velocities,
            noise,
            uniform,
            status,
            staging,
            bind,
            pipelines,
            masses: system.atoms().iter().map(|a| a.mass()).collect(),
        })
    }
    pub async fn advance(
        &mut self,
        coordinates: &[Vec3],
        velocities: &[Vec3],
        noise: &[[f32; 4]],
        steps: usize,
        dt_ps: f64,
        temperature: f64,
        friction: f64,
    ) -> Result<DynamicsBatch, Error> {
        let n = self.masses.len();
        if steps == 0
            || steps > MAX_STEPS
            || coordinates.len() != n
            || velocities.len() != n
            || noise.len() != n * steps
            || !dt_ps.is_finite()
            || dt_ps <= 0.
            || dt_ps > 0.0005
            || !temperature.is_finite()
            || temperature <= 0.
            || !friction.is_finite()
            || friction < 0.
        {
            return Err(Error::Input("invalid dynamics batch"));
        }
        let p = self.topology.coordinates(coordinates, &vec![true; n])?;
        let v: Vec<[f32; 4]> = velocities
            .iter()
            .zip(&self.masses)
            .map(|(v, m)| [v.x as f32, v.y as f32, v.z as f32, (1. / m) as f32])
            .collect();
        if v.iter()
            .flatten()
            .chain(noise.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err(Error::Nonfinite);
        }
        let config = Config {
            size: [n as u32, 1, 0, 1],
            energy: [0., 1., 1., 0.],
            solvent: [1., 78.5, 1.4, 0.00542],
            spare: [0.; 4],
        };
        // Establish starting force before the first B half-kick.
        self.energy.evaluate(config, &p, true).await?;
        let decay = (-friction * dt_ps).exp();
        let mut uniforms = vec![0u32; MAX_STEPS * 64];
        for i in 0..steps {
            uniforms[i * 64] = n as u32;
            uniforms[i * 64 + 1] = (i * n) as u32;
            uniforms[i * 64 + 4] = (dt_ps as f32).to_bits();
            uniforms[i * 64 + 5] = (decay as f32).to_bits();
            uniforms[i * 64 + 6] =
                (((1. - decay * decay) * 0.00198720425864083 * temperature * 418.4) as f32)
                    .to_bits();
        }
        let device = &self.energy.device;
        let queue = &self.energy.queue;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        queue.write_buffer(&self.velocities, 0, bytemuck::cast_slice(&v));
        queue.write_buffer(&self.noise, 0, bytemuck::cast_slice(noise));
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&uniforms));
        queue.write_buffer(&self.status, 0, &[0; 16]);
        let mut encoder = device.create_command_encoder(&Default::default());
        for step in 0..steps {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("BAO"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines[0]);
                pass.set_bind_group(0, &self.bind, &[(step * 256) as u32]);
                pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
            }
            for i in 0..4 {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("OBC2 forces"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.energy.pipelines[i]);
                pass.set_bind_group(0, &self.energy.bind_group, &[]);
                pass.dispatch_workgroups(if i == 3 { 1 } else { (n as u32).div_ceil(64) }, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("B"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines[1]);
                pass.set_bind_group(0, &self.bind, &[(step * 256) as u32]);
                pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
            }
        }
        let block = n as u64 * 16;
        encoder.copy_buffer_to_buffer(&self.energy.buffers[2], 0, &self.staging, 0, block);
        encoder.copy_buffer_to_buffer(&self.velocities, 0, &self.staging, block, block);
        encoder.copy_buffer_to_buffer(
            &self.energy.buffers[7],
            0,
            &self.staging,
            block * 2,
            n as u64 * 64 + 48,
        );
        encoder.copy_buffer_to_buffer(&self.status, 0, &self.staging, n as u64 * 96 + 48, 4);
        queue.submit([encoder.finish()]);
        if let Some(e) = device.pop_error_scope().await {
            return Err(Error::Execution(e.to_string()));
        }
        let slice = self.staging.slice(..n as u64 * 96 + 52);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        #[cfg(not(target_arch = "wasm32"))]
        device
            .poll(wgpu::PollType::Wait)
            .map_err(|e| Error::Execution(e.to_string()))?;
        rx.await
            .map_err(|e| Error::Execution(e.to_string()))?
            .map_err(|e| Error::Execution(e.to_string()))?;
        let mapped = slice.get_mapped_range();
        let bytes = mapped.to_vec();
        drop(mapped);
        self.staging.unmap();
        let invalid = u32::from_le_bytes(bytes[n * 96 + 48..n * 96 + 52].try_into().unwrap());
        if invalid != 0 {
            return Err(Error::Execution(
                "unstable dynamics step; restore checkpoint".into(),
            ));
        }
        let floats = bytemuck::cast_slice::<u8, f32>(&bytes[..n * 96 + 48]);
        if floats.iter().any(|x| !x.is_finite()) {
            return Err(Error::Nonfinite);
        }
        let coords = floats[..n * 4]
            .chunks_exact(4)
            .map(|p| Vec3 {
                x: p[0] as f64 + self.topology.origin.x,
                y: p[1] as f64 + self.topology.origin.y,
                z: p[2] as f64 + self.topology.origin.z,
            })
            .collect();
        let velocities = floats[n * 4..n * 8]
            .chunks_exact(4)
            .map(|p| Vec3 {
                x: p[0] as f64,
                y: p[1] as f64,
                z: p[2] as f64,
            })
            .collect();
        let gradients = floats[n * 20..n * 24]
            .chunks_exact(4)
            .map(|p| Vec3 {
                x: p[0] as f64,
                y: p[1] as f64,
                z: p[2] as f64,
            })
            .collect();
        let components = floats[n * 24..n * 24 + 12].try_into().unwrap();
        Ok(DynamicsBatch {
            coordinates: coords,
            velocities,
            gradients,
            components,
        })
    }
}
