//! Worker-owned resident compute resources. This module never starts Rayon work.
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

pub const BUFFER_BUDGET: u64 = 256 * 1024 * 1024;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Atom {
    pub ff: [f32; 4],
    pub more: [f32; 4],
    pub ranges: [u32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Term {
    pub ids: [u32; 4],
    pub parameters: [f32; 4],
    pub reference: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Special {
    pub other: u32,
    pub scee: f32,
    pub scnb: f32,
    pub spare: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Config {
    pub size: [u32; 4],
    pub energy: [f32; 4],
    pub solvent: [f32; 4],
    pub spare: [f32; 4],
}
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("WebGPU unavailable: {0}")]
    Unavailable(String),
    #[error("GPU buffer budget or device limit exceeded")]
    Capacity,
    #[error("Invalid GPU input: {0}")]
    Input(&'static str),
    #[error("GPU execution failed: {0}")]
    Execution(String),
    #[error("GPU returned nonfinite values")]
    Nonfinite,
}
/// Packed immutable topology. Coordinates and restraint references must share a centered origin.
#[derive(Clone, Copy)]
pub struct Topology<'a> {
    pub atoms: &'a [Atom],
    pub terms: &'a [Term],
    pub incidence: &'a [[u32; 2]],
    pub specials: &'a [Special],
}
/// A single owner serializes dispatches and reuses all device and readback buffers.
pub struct ResidentEvaluator {
    // Keep the browser GPU instance alive for every pending map/dispatch.
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    pub adapter_info: wgpu::AdapterInfo,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) buffers: Vec<wgpu::Buffer>,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) pipelines: Vec<wgpu::ComputePipeline>,
    pub(crate) staging: wgpu::Buffer,
    pub(crate) atoms: u32,
    capacity: u32,
}
pub struct BatchResult {
    /// Bond, angle, proper, improper, LJ, electrostatic, GB, SA, restraint, pair count, padding.
    pub components: Vec<[f32; 12]>,
    pub gradients: Option<Vec<[f32; 4]>>,
}
impl ResidentEvaluator {
    pub async fn new(topology: Topology<'_>, requested_batch: u32) -> Result<Self, Error> {
        Self::with_budget(topology, requested_batch, BUFFER_BUDGET).await
    }
    pub async fn with_budget(topology: Topology<'_>, requested_batch: u32, budget: u64) -> Result<Self, Error> {
        let mut batch = requested_batch;
        loop {
            match Self::allocate(topology, batch, budget.min(BUFFER_BUDGET)).await {
                Err(Error::Capacity) if batch > 1 => batch = (batch / 2).max(1),
                result => return result,
            }
        }
    }
    async fn allocate(topology: Topology<'_>, requested_batch: u32, budget: u64) -> Result<Self, Error> {
        if topology.atoms.is_empty() || requested_batch == 0 {
            return Err(Error::Input("empty topology or batch"));
        }
        let atoms = u32::try_from(topology.atoms.len()).map_err(|_| Error::Capacity)?;
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|e| Error::Unavailable(e.to_string()))?;
        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("GlySys compute"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| Error::Unavailable(e.to_string()))?;
        let fixed: [&[u8]; 4] = [
            bytemuck::cast_slice(topology.atoms),
            bytemuck::cast_slice(topology.terms),
            bytemuck::cast_slice(topology.incidence),
            bytemuck::cast_slice(topology.specials),
        ];
        let limits = device.limits();
        let max_buffer = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size as u64);
        if fixed.iter().any(|x| x.len() as u64 > max_buffer) {
            return Err(Error::Capacity);
        }
        let fixed_bytes: u64 = fixed.iter().map(|x| (x.len() as u64).max(48)).sum();
        let mut capacity = requested_batch;
        loop {
            let count = u64::from(atoms) * u64::from(capacity);
            let output = count * 64 + u64::from(capacity) * 48;
            let staging = count * 16 + u64::from(capacity) * 48;
            if count.div_ceil(64) <= u64::from(limits.max_compute_workgroups_per_dimension)
                && capacity <= limits.max_compute_workgroups_per_dimension
                && output <= max_buffer
                && staging <= limits.max_buffer_size
                && fixed_bytes + count * 32 + output + staging + 64 <= budget
            {
                break;
            }
            if capacity == 1 {
                return Err(Error::Capacity);
            }
            capacity = (capacity / 2).max(1);
        }
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let count = u64::from(atoms) * u64::from(capacity);
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let immutable = |data: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("immutable topology"),
                contents: if data.is_empty() { &[0; 48] } else { data },
                usage: storage,
            })
        };
        let buffers = vec![
            buffer(
                "config",
                64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            immutable(fixed[0]),
            buffer(
                "coordinates",
                count * 16,
                storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            ),
            immutable(fixed[1]),
            immutable(fixed[2]),
            immutable(fixed[3]),
            buffer("Born scratch", count * 16, storage),
            buffer(
                "energies and gradients",
                count * 64 + u64::from(capacity) * 48,
                storage | wgpu::BufferUsages::COPY_SRC,
            ),
        ];
        let staging = buffer(
            "readback",
            count * 16 + u64::from(capacity) * 48,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let entries: Vec<_> = (0..8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 6,
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GlySys energies"),
            source: wgpu::ShaderSource::Wgsl(include_str!("energy.wgsl").into()),
        });
        let pipelines = ["born_radii", "born_adjoint", "evaluate", "reduce", "evaluate"]
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions { constants: &[("COMPUTE_GRADIENTS", if index == 4 { 0.0 } else { 1.0 })], ..Default::default() },
                    cache: None,
                })
            })
            .collect();
        let validation = device.pop_error_scope().await;
        let allocation = device.pop_error_scope().await;
        if let Some(e) = validation {
            return Err(Error::Execution(e.to_string()));
        }
        if allocation.is_some() {
            drop(pipelines);
            drop(bind_group);
            drop(buffers);
            drop(staging);
            drop(queue);
            drop(device);
            if capacity == 1 {
                return Err(Error::Capacity);
            }
            return Box::pin(Self::new(topology, (capacity / 2).max(1))).await;
        }
        Ok(Self {
            _instance: instance,
            _adapter: adapter,
            adapter_info,
            device,
            queue,
            buffers,
            bind_group,
            pipelines,
            staging,
            atoms,
            capacity,
        })
    }
    pub fn batch_capacity(&self) -> u32 {
        self.capacity
    }
    pub async fn evaluate(
        &mut self,
        config: Config,
        coordinates: &[[f32; 4]],
        gradients: bool,
    ) -> Result<BatchResult, Error> {
        let [atoms, batch, mode, active] = config.size;
        if mode > 1
            || active > 1
            || config
                .energy
                .iter()
                .chain(config.solvent.iter())
                .any(|x| !x.is_finite())
            || config.energy[0] < 0.
            || config.energy[1] <= 0.
            || ![0., 1.].contains(&config.energy[2])
            || config.solvent[0] <= 0.
            || config.solvent[1] <= 0.
            || config.solvent[2] < 0.
            || config.solvent[3] < 0.
            || (mode == 1 && config.energy[2] != 0.)
        {
            return Err(Error::Input("unsupported energy configuration"));
        }
        if atoms != self.atoms
            || batch == 0
            || batch > self.capacity
            || coordinates.len() != atoms as usize * batch as usize
        {
            return Err(Error::Input("coordinate dimensions"));
        }
        if coordinates.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::Input("nonfinite coordinate"));
        }
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        self.queue
            .write_buffer(&self.buffers[0], 0, bytemuck::bytes_of(&config));
        self.queue
            .write_buffer(&self.buffers[2], 0, bytemuck::cast_slice(coordinates));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for i in 0..4 {
            if i == 1 && !gradients { continue; }
            let pipeline = &self.pipelines[if i == 2 && !gradients { 4 } else { i }];
            if i < 2 && (config.energy[2] == 0.0 || config.size[2] != 0) {
                continue;
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(
                if i == 3 {
                    batch
                } else {
                    (atoms * batch).div_ceil(64)
                },
                1,
                1,
            );
        }
        let count = u64::from(atoms) * u64::from(batch);
        let summary = u64::from(batch) * 48;
        encoder.copy_buffer_to_buffer(&self.buffers[7], count * 64, &self.staging, 0, summary);
        if gradients {
            encoder.copy_buffer_to_buffer(
                &self.buffers[7],
                count * 48,
                &self.staging,
                summary,
                count * 16,
            );
        }
        self.queue.submit([encoder.finish()]);
        if let Some(e) = self.device.pop_error_scope().await {
            return Err(Error::Execution(e.to_string()));
        }
        let bytes = summary + if gradients { count * 16 } else { 0 };
        let slice = self.staging.slice(..bytes);
        let (sender, receiver) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::Wait)
            .map_err(|e| Error::Execution(e.to_string()))?;
        receiver
            .await
            .map_err(|e| Error::Execution(e.to_string()))?
            .map_err(|e| Error::Execution(e.to_string()))?;
        let mapped = slice.get_mapped_range();
        let components =
            bytemuck::cast_slice::<u8, [f32; 12]>(&mapped[..summary as usize]).to_vec();
        let gradients = gradients
            .then(|| bytemuck::cast_slice::<u8, [f32; 4]>(&mapped[summary as usize..]).to_vec());
        drop(mapped);
        self.staging.unmap();
        if components.iter().flatten().any(|v| !v.is_finite())
            || gradients
                .as_ref()
                .is_some_and(|g| g.iter().flatten().any(|v| !v.is_finite()))
        {
            return Err(Error::Nonfinite);
        }
        Ok(BatchResult {
            components,
            gradients,
        })
    }
}
