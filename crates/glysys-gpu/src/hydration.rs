//! Resident, bounded rigid-water probe batches. No atom-pair matrices.
use crate::context::{AllocationReservation, GpuContext};
use crate::device::Error;
use glysys::Vec3;
use glysys_energy::hydration::{PhysicalProbe, ProbeScore, WaterPose};
use wgpu::util::DeviceExt;
pub struct ResidentWaterProbe {
    pub(crate) _context: GpuContext,
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: Vec<wgpu::Buffer>,
    staging: wgpu::Buffer,
    bind: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    origin: Vec3,
    atoms: u32,
    pub capacity: usize,
    _allocation: AllocationReservation,
}
impl ResidentWaterProbe {
    pub async fn with_context(
        context: &GpuContext,
        probe: &PhysicalProbe,
        requested: usize,
    ) -> Result<Self, Error> {
        if probe.atoms.is_empty() || requested == 0 {
            return Err(Error::Input("empty water-probe batch"));
        }
        let device = context.device().clone();
        let queue = context.queue().clone();
        let origin = probe.atoms[0].position;
        let data: Vec<[f32; 8]> = probe
            .atoms
            .iter()
            .map(|a| {
                [
                    (a.position.x - origin.x) as f32,
                    (a.position.y - origin.y) as f32,
                    (a.position.z - origin.z) as f32,
                    0.,
                    a.charge as f32,
                    a.radius as f32,
                    a.epsilon as f32,
                    0.,
                ]
            })
            .collect();
        let fixed = data.len() as u64 * 32;
        let limits = device.limits();
        let limit = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size as u64);
        let budget = context.memory_profile().budget();
        if fixed > budget || fixed > limit {
            return Err(Error::Capacity);
        }
        let capacity = requested
            .min(((budget.saturating_sub(fixed).saturating_sub(16)) / 80) as usize)
            .min((limit / 48) as usize)
            .min(limits.max_compute_workgroups_per_dimension as usize);
        if capacity == 0 {
            return Err(Error::Capacity);
        }
        let reservation = context.reserve(
            fixed
                .checked_add(capacity as u64 * 48)
                .and_then(|v| v.checked_add(capacity as u64 * 16))
                .and_then(|v| v.checked_add(capacity as u64 * 16))
                .ok_or(Error::Capacity)?,
        )?;
        crate::push_error_scope(&device, wgpu::ErrorFilter::OutOfMemory);
        crate::push_error_scope(&device, wgpu::ErrorFilter::Validation);
        let buffer = |size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let buffers = vec![
            buffer(
                16,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            buffer(
                capacity as u64 * 48,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            buffer(
                capacity as u64 * 16,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
        ];
        let staging = buffer(
            capacity as u64 * 16,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let entries: Vec<_> = (0..4)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 3,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TIP3P physical water probe"),
            source: wgpu::ShaderSource::Wgsl(include_str!("hydration.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("evaluate"),
            compilation_options: Default::default(),
            cache: None,
        });
        context.record_pipeline("hydration.evaluate");
        let validation = crate::pop_error_scope(&device).await;
        let allocation = crate::pop_error_scope(&device).await;
        if let Some(e) = validation {
            return Err(Error::Execution(e.to_string()));
        }
        if allocation.is_some() {
            return Err(Error::Capacity);
        }
        Ok(Self {
            _context: context.clone(),
            device,
            queue,
            buffers,
            staging,
            bind,
            pipeline,
            origin,
            atoms: data.len() as u32,
            capacity,
            _allocation: reservation,
        })
    }
    pub async fn evaluate(
        &mut self,
        poses: &[WaterPose],
        cutoff: Option<f64>,
    ) -> Result<Vec<Option<ProbeScore>>, Error> {
        if poses.is_empty() {
            return Ok(Vec::new());
        }
        if poses.len() > self.capacity || cutoff.is_some_and(|c| !c.is_finite() || c <= 0.) {
            return Err(Error::Input("water probe size or cutoff"));
        }
        let data: Vec<[f32; 12]> = poses
            .iter()
            .map(|p| {
                let mut v = [0.; 12];
                for (i, p) in [p.oxygen, p.hydrogens[0], p.hydrogens[1]]
                    .into_iter()
                    .enumerate()
                {
                    v[4 * i] = (p.x - self.origin.x) as f32;
                    v[4 * i + 1] = (p.y - self.origin.y) as f32;
                    v[4 * i + 2] = (p.z - self.origin.z) as f32;
                }
                v
            })
            .collect();
        if data.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Error::Nonfinite);
        }
        let config = [
            self.atoms,
            poses.len() as u32,
            (cutoff.unwrap_or(0.) as f32).to_bits(),
            0,
        ];
        crate::push_error_scope(&self.device, wgpu::ErrorFilter::Validation);
        self.queue
            .write_buffer(&self.buffers[0], 0, bytemuck::cast_slice(&config));
        self.queue
            .write_buffer(&self.buffers[2], 0, bytemuck::cast_slice(&data));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(poses.len() as u32, 1, 1);
        }
        let bytes = poses.len() as u64 * 16;
        encoder.copy_buffer_to_buffer(&self.buffers[3], 0, &self.staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        if let Some(e) = crate::pop_error_scope(&self.device).await {
            return Err(Error::Execution(e.to_string()));
        }
        let slice = self.staging.slice(..bytes);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::Wait)
            .map_err(|e| Error::Execution(e.to_string()))?;
        rx.await
            .map_err(|e| Error::Execution(e.to_string()))?
            .map_err(|e| Error::Execution(e.to_string()))?;
        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped).to_vec();
        drop(mapped);
        self.staging.unmap();
        if values.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Error::Nonfinite);
        }
        Ok(values
            .into_iter()
            .map(|v| {
                (v[2] == 0.).then_some(ProbeScore {
                    lennard_jones: v[0] as f64,
                    electrostatics: v[1] as f64,
                })
            })
            .collect())
    }
}
