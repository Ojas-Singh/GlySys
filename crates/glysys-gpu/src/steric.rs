//! Resident attachment library and batched traversal-compatible steric screening.
use crate::device::{BUFFER_BUDGET, Error};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AttachmentPose {
    pub bounds: [u32; 4],
    pub indices: [u32; 4],
    pub b: [f32; 4],
    pub link: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ReceptorUpdate {
    pub value: [f32; 4],
    pub indices: [u32; 4],
}
pub struct AttachmentLibrary {
    pub protein: Vec<[f32; 4]>,
    pub poses: Vec<AttachmentPose>,
    pub coordinates: Vec<[f32; 4]>,
    pub updates: Vec<ReceptorUpdate>,
    pub sites: u32,
    pub candidate_atoms: u32,
}
// Sorted streams retain Cookbook's original receptor atom traversal order.
fn receptor_cells(library: &AttachmentLibrary) -> Vec<[u32; 4]> {
    let flexible: std::collections::BTreeSet<_> = library.updates.iter().map(|u| u.indices[0]).collect();
    let mut cells = std::collections::BTreeMap::<[i32;3], Vec<u32>>::new();
    let safe = library.protein.iter().chain(&library.coordinates).all(|p| p[..3].iter().all(|v| v.is_finite() && v.abs() < 100_000.));
    for (i,p) in library.protein.iter().enumerate() {
        if !flexible.contains(&(i as u32)) { cells.entry([p[0],p[1],p[2]].map(|x| (x / 3.4).floor() as i32)).or_default().push(i as u32); }
    }
    let mut packed = vec![[0;4];1+2*cells.len()];
    packed[0]=[cells.len() as u32,0,flexible.len() as u32,u32::from(safe)];
    for (i,(key,atoms)) in cells.into_iter().enumerate() {
        packed[1+2*i]=[key[0] as u32,key[1] as u32,key[2] as u32,packed.len() as u32];
        packed[2+2*i]=[atoms.len() as u32,0,0,0];
        packed.extend(atoms.into_iter().map(|atom| [atom,0,0,0]));
    }
    packed[0][1]=packed.len() as u32;
    packed.extend(flexible.into_iter().map(|atom| [atom,0,0,0]));
    packed
}
pub struct ResidentSteric {
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: Vec<wgpu::Buffer>,
    staging: wgpu::Buffer,
    bind: wgpu::BindGroup,
    pipelines: Vec<wgpu::ComputePipeline>,
    sites: u32,
    atoms: u32,
    protein: u32,
    pub capacity: u32,
    poses: Vec<AttachmentPose>,
}
impl ResidentSteric {
    pub async fn new(library: &AttachmentLibrary, requested: u32) -> Result<Self, Error> {
        Self::with_budget(library, requested, BUFFER_BUDGET).await
    }
    pub async fn with_budget(library: &AttachmentLibrary, requested: u32, budget: u64) -> Result<Self, Error> {
        let mut batch = requested;
        loop {
            match Self::allocate(library, batch, budget.min(BUFFER_BUDGET)).await {
                Err(Error::Capacity) if batch > 1 => batch = (batch / 2).max(1),
                result => return result,
            }
        }
    }
    async fn allocate(library: &AttachmentLibrary, requested: u32, budget: u64) -> Result<Self, Error> {
        if library.sites == 0 || library.candidate_atoms == 0 || requested == 0 {
            return Err(Error::Input("empty attachment library"));
        }
        for pose in &library.poses {
            if pose.bounds[0]
                .checked_add(pose.bounds[1])
                .is_none_or(|end| end as usize > library.coordinates.len())
                || pose.bounds[2] > pose.bounds[1]
                || pose.bounds[3] >= pose.bounds[1]
                || pose.indices[0] >= pose.bounds[1]
                || pose.indices[1] > pose.indices[2]
                || pose.indices[2] as usize > library.updates.len()
            {
                return Err(Error::Input("invalid attachment library ranges"));
            }
        }
        if library
            .updates
            .iter()
            .any(|u| u.indices[0] as usize >= library.protein.len())
        {
            return Err(Error::Input("invalid receptor update"));
        }
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .map_err(|e| Error::Unavailable(e.to_string()))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("GlySys steric"),
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Unavailable(e.to_string()))?;
        let grid = receptor_cells(library);
        let data: [&[u8]; 5] = [
            bytemuck::cast_slice(&library.protein),
            bytemuck::cast_slice(&library.poses),
            bytemuck::cast_slice(&library.coordinates),
            bytemuck::cast_slice(&library.updates),
            bytemuck::cast_slice(&grid),
        ];
        let fixed = data.iter().map(|v| v.len().max(32) as u64).sum::<u64>() + 32;
        let each = u64::from(library.candidate_atoms) * 16 + u64::from(library.sites) * 24;
        let limit = u64::from(device.limits().max_storage_buffer_binding_size)
            .min(device.limits().max_buffer_size);
        if fixed >= budget || data.iter().any(|v| v.len() as u64 > limit) {
            return Err(Error::Capacity);
        }
        let capacity = requested.min(
            ((budget - fixed) / each)
                .min(limit / (u64::from(library.candidate_atoms) * 16))
                .min(limit / (u64::from(library.sites) * 16)) as u32,
        );
        if capacity == 0 {
            return Err(Error::Capacity);
        }
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let alloc = |name, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let immutable = |bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: if bytes.is_empty() { &[0; 32] } else { bytes },
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let buffers = vec![
            alloc(
                "config",
                32,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            immutable(data[0]),
            immutable(data[1]),
            immutable(data[2]),
            alloc(
                "genes",
                u64::from(capacity) * u64::from(library.sites) * 16,
                storage | wgpu::BufferUsages::COPY_DST,
            ),
            alloc(
                "transforms",
                u64::from(capacity) * u64::from(library.candidate_atoms) * 16,
                storage,
            ),
            alloc(
                "scores",
                u64::from(capacity) * u64::from(library.sites) * 4,
                storage | wgpu::BufferUsages::COPY_SRC,
            ),
            immutable(data[3]),
            immutable(data[4]),
        ];
        let staging = alloc(
            "scores readback",
            u64::from(capacity) * u64::from(library.sites) * 4,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let entries = (0..9)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 5 && binding != 6,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attachment transforms and sterics"),
            source: wgpu::ShaderSource::Wgsl(include_str!("steric.wgsl").into()),
        });
        let pipelines = ["transform", "evaluate"]
            .iter()
            .map(|name| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
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
            return Err(Error::Capacity);
        }
        Ok(Self {
            _instance: instance,
            device,
            queue,
            buffers,
            staging,
            bind,
            pipelines,
            sites: library.sites,
            atoms: library.candidate_atoms,
            protein: library.protein.len() as u32,
            capacity,
            poses: library.poses.clone(),
        })
    }
    pub async fn evaluate(&mut self, genes: &[[u32; 4]], cutoff: f32) -> Result<Vec<f32>, Error> {
        if genes.is_empty()
            || !genes.len().is_multiple_of(self.sites as usize)
            || genes.len() > self.sites as usize * self.capacity as usize
            || !cutoff.is_finite()
            || cutoff <= 0.
        {
            return Err(Error::Input("invalid steric batch"));
        }
        for gene in genes {
            let pose = self
                .poses
                .get(gene[0] as usize)
                .ok_or(Error::Input("unknown attachment pose"))?;
            if !f32::from_bits(gene[1]).is_finite()
                || !f32::from_bits(gene[2]).is_finite()
                || gene[3]
                    .checked_add(pose.bounds[1])
                    .is_none_or(|end| end > self.atoms)
            {
                return Err(Error::Input("invalid attachment coordinates"));
            }
        }
        let count = genes.len() as u32;
        let config = [
            self.sites,
            count / self.sites,
            self.atoms,
            self.protein,
            (cutoff * cutoff).to_bits(),
            0,
            0,
            0,
        ];
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        self.queue
            .write_buffer(&self.buffers[0], 0, bytemuck::cast_slice(&config));
        self.queue
            .write_buffer(&self.buffers[4], 0, bytemuck::cast_slice(genes));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for pipeline in &self.pipelines {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        let bytes = u64::from(count) * 4;
        encoder.copy_buffer_to_buffer(&self.buffers[6], 0, &self.staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        if let Some(e) = self.device.pop_error_scope().await {
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
        let view = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        self.staging.unmap();
        if result.iter().any(|v| !v.is_finite()) {
            return Err(Error::Nonfinite);
        }
        Ok(result)
    }
}
