//! Resident implicit-solvent dynamics share force-field coordinates/gradients.
use crate::{
    context::{AllocationReservation, GpuContext},
    device::{Config, Error, ResidentEvaluator},
    topology::PreparedTopology,
};
use bytemuck::Zeroable;
use glysys::{ParameterizedSystem, Vec3};
use glysys_energy::{EnergyOptions, Obc2Options};
use std::collections::{BTreeMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;
use wgpu::util::DeviceExt;
pub const MAX_STEPS: usize = 128;
/// LF-middle integration emits multiple force and constraint passes per step.
/// Keep those packets smaller than the legacy BAOAB batch to bound command
/// encoder and driver memory on native adapters.
pub const LF_MIDDLE_PACKET_STEPS: usize = 16;
pub const LF_MIDDLE_MAX_PACKET_STEPS: usize = 128;
const LEGACY_NOISE_STEPS: usize = 20;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConstraintGroup {
    start: u32,
    count: u32,
    padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuConstraint {
    atoms: [u32; 4],
    parameters: [f32; 4],
}

fn implicit_hbond_groups(
    system: &ParameterizedSystem,
) -> (Vec<ConstraintGroup>, Vec<GpuConstraint>) {
    let mut groups = BTreeMap::<usize, Vec<(usize, f64)>>::new();
    let mut seen = HashSet::new();
    for bond in system.bonds() {
        let [a, b] = bond.atoms();
        let (parent, hydrogen) = match (
            system.atoms()[a].element() == 1,
            system.atoms()[b].element() == 1,
        ) {
            (true, false) => (b, a),
            (false, true) => (a, b),
            _ => continue,
        };
        if seen.insert((parent.min(hydrogen), parent.max(hydrogen))) {
            groups
                .entry(parent)
                .or_default()
                .push((hydrogen, bond.length()));
        }
    }
    let mut headers = Vec::with_capacity(groups.len().max(1));
    let mut constraints = Vec::new();
    let mut constrained_atoms = HashSet::new();
    for (parent, hydrogens) in groups {
        constrained_atoms.insert(parent);
        headers.push(ConstraintGroup {
            start: constraints.len() as u32,
            count: hydrogens.len() as u32,
            padding: [parent as u32, 0],
        });
        for (hydrogen, target) in hydrogens {
            constrained_atoms.insert(hydrogen);
            constraints.push(GpuConstraint {
                atoms: [parent as u32, hydrogen as u32, 0, 0],
                parameters: [
                    target as f32,
                    (1.0 / system.atoms()[parent].mass()) as f32,
                    (1.0 / system.atoms()[hydrogen].mass()) as f32,
                    0.0,
                ],
            });
        }
    }
    for atom in 0..system.atom_count() {
        if !constrained_atoms.contains(&atom) {
            headers.push(ConstraintGroup {
                start: 0,
                count: 0,
                padding: [atom as u32, 0],
            });
        }
    }
    if constraints.is_empty() {
        constraints.push(GpuConstraint::zeroed());
    }
    (headers, constraints)
}

pub struct DynamicsBatch {
    pub coordinates: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub gradients: Vec<Vec3>,
    pub components: [f32; 12],
    pub rng_words: Option<Vec<u32>>,
    pub host_enqueue_wait_ms: f64,
    pub host_readback_ms: f64,
}

/// A resident LF-middle advance with thermodynamic scalars only. The device
/// coordinates, velocities, force buffers, and RNG stream remain resident.
pub struct DynamicsObservation {
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub host_enqueue_wait_ms: f64,
    pub host_readback_ms: f64,
    pub readback_bytes: u64,
}

enum LfMiddleReadback {
    Snapshot(DynamicsBatch),
    Observation(DynamicsObservation),
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
    pipelines: Vec<wgpu::ComputePipeline>,
    _constraint_groups: wgpu::Buffer,
    _constraints: wgpu::Buffer,
    _trial_coordinates: wgpu::Buffer,
    _observation_masses: wgpu::Buffer,
    observation_output: wgpu::Buffer,
    observation_bind: wgpu::BindGroup,
    observation_pipeline: wgpu::ComputePipeline,
    constraint_group_count: u32,
    md_lanes_per_target: usize,
    md_pipeline_start: usize,
    packet_steps: usize,
    dense_special_lookup_base: Option<u32>,
    masses: Vec<f64>,
    resident_initialized: bool,
    gpu_stage_timings_ms: std::sync::Mutex<BTreeMap<String, f64>>,
    _allocation: AllocationReservation,
}
impl ResidentDynamics {
    /// Construct implicit resident dynamics on a shared GPU context.
    pub async fn with_context(
        system: &ParameterizedSystem,
        context: &GpuContext,
    ) -> Result<Self, Error> {
        Self::with_context_and_lanes(system, context, 8).await
    }

    /// Construct implicit dynamics with the selected cooperative all-pairs
    /// lane count. Values 4, 8, and 16 are separately compiled variants.
    pub async fn with_context_and_lanes(
        system: &ParameterizedSystem,
        context: &GpuContext,
        md_lanes_per_target: usize,
    ) -> Result<Self, Error> {
        Self::with_context_tuning(system, context, md_lanes_per_target, LF_MIDDLE_PACKET_STEPS)
            .await
    }

    /// Construct implicit dynamics with measured lane and packet variants.
    pub async fn with_context_tuning(
        system: &ParameterizedSystem,
        context: &GpuContext,
        md_lanes_per_target: usize,
        packet_steps: usize,
    ) -> Result<Self, Error> {
        let md_pipeline_start = match md_lanes_per_target {
            4 => 5,
            8 => 8,
            16 => 11,
            32 => 14,
            64 => 17,
            128 => 20,
            _ => {
                return Err(Error::Input(
                    "implicit MD lanes per target must be 4, 8, 16, 32, 64, or 128",
                ));
            }
        };
        if ![8, 16, 32, 64, LF_MIDDLE_MAX_PACKET_STEPS].contains(&packet_steps) {
            return Err(Error::Input(
                "implicit LF-middle packet must contain 8, 16, 32, 64, or 128 steps",
            ));
        }
        let n = system.atom_count();
        if n == 0
            || system
                .atoms()
                .iter()
                .any(|a| !a.mass().is_finite() || a.mass() <= 0.)
        {
            return Err(Error::Input("positive atom masses required"));
        }
        let mut topology = PreparedTopology::new(
            system,
            &EnergyOptions {
                obc2: Some(Obc2Options::default()),
                ..Default::default()
            },
            &vec![0; n],
        )?;
        let dense_lookup_limit = context
            .memory_profile()
            .budget()
            .min(u64::from(
                context.device().limits().max_storage_buffer_binding_size,
            ))
            .min(64 * 1024 * 1024);
        topology.append_dense_special_lookup(dense_lookup_limit)?;
        let dense_special_lookup_base = topology.dense_special_lookup_base();
        let (groups, constraints) = implicit_hbond_groups(system);
        let energy = ResidentEvaluator::with_context(context, topology.view(), 1).await?;
        let existing = energy.buffers.iter().map(|b| b.size()).sum::<u64>() + energy.staging.size();
        let additional = n as u64 * (16 + 16 * LEGACY_NOISE_STEPS as u64 + 64 + 16 + 4)
            + MAX_STEPS as u64 * 256
            + groups.len() as u64 * 16
            + constraints.len() as u64 * 32
            + 96;
        if existing
            .checked_add(additional)
            .is_none_or(|bytes| bytes > context.memory_profile().budget())
            || n as u64 * 16 * LEGACY_NOISE_STEPS as u64
                > energy.device.limits().max_storage_buffer_binding_size as u64
        {
            return Err(Error::Capacity);
        }
        let reservation = context.reserve(additional)?;
        let device = &energy.device;
        crate::push_error_scope(&device, wgpu::ErrorFilter::OutOfMemory);
        crate::push_error_scope(&device, wgpu::ErrorFilter::Validation);
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
            n as u64 * 16 * LEGACY_NOISE_STEPS as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
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
            n as u64 * 64 + 64,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let entries: Vec<_> = (0..9)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: matches!(binding, 3 | 6 | 7),
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
        let group_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("implicit H-bond constraint groups"),
            contents: bytemuck::cast_slice(&groups),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let constraint_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("implicit H-bond constraints"),
            contents: bytemuck::cast_slice(&constraints),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let trial_coordinates = alloc(
            n as u64 * 32,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let observation_masses: Vec<f32> = system
            .atoms()
            .iter()
            .map(|atom| atom.mass() as f32)
            .collect();
        let observation_masses = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("implicit dynamics observation masses"),
            contents: bytemuck::cast_slice(&observation_masses),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let observation_output = alloc(
            16,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let buffers = [
            &uniform,
            &energy.buffers[2],
            &velocities,
            &energy.buffers[7],
            &noise,
            &status,
            &group_buffer,
            &constraint_buffer,
            &trial_coordinates,
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
        let observation_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("implicit dynamics scalar observation layout"),
                entries: &(0..4)
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: binding != 3,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            });
        let observation_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("implicit dynamics scalar observation pipeline layout"),
                bind_group_layouts: &[&observation_layout],
                push_constant_ranges: &[],
            });
        let observation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("implicit dynamics scalar observations"),
            source: wgpu::ShaderSource::Wgsl(include_str!("dynamics_observables.wgsl").into()),
        });
        let observation_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("reduce_dynamics_observation"),
                layout: Some(&observation_pipeline_layout),
                module: &observation_shader,
                entry_point: Some("reduce_dynamics_observation"),
                compilation_options: Default::default(),
                cache: None,
            });
        let observation_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("implicit dynamics scalar observation"),
            layout: &observation_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: velocities.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: observation_masses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: energy.buffers[7].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: observation_output.as_entire_binding(),
                },
            ],
        });
        context.record_pipeline("dynamics.reduce_observation_scalars");
        // Per-stage error scopes so a driver rejection names the exact
        // kernel: shader-module failures (WGSL/Tint) and per-pipeline
        // failures (backend translation, e.g. Metal library creation) need
        // different fixes, and the adapter identity matters for both.
        let adapter = crate::adapter::describe(&energy.adapter_info);
        crate::push_error_scope(&device, wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BAOAB"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("resident_rng.wgsl"),
                    include_str!("dynamics.wgsl")
                )
                .into(),
            ),
        });
        if let Some(e) = crate::pop_error_scope(&device).await {
            return Err(Error::Execution(format!(
                "GlySys dynamics shader rejected on {adapter}: {e}"
            )));
        }
        let mut pipelines = Vec::with_capacity(11);
        for name in [
            "before_force",
            "after_force",
            "lf_kick_velocity_projection",
            "lf_drift_thermostat_snapshot",
            "lf_position_projection",
        ] {
            crate::push_error_scope(&device, wgpu::ErrorFilter::Validation);
            pipelines.push(
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                }),
            );
            if let Some(e) = crate::pop_error_scope(&device).await {
                return Err(Error::Execution(format!(
                    "GlySys dynamics pipeline '{name}' rejected on {adapter}: {e}"
                )));
            }
        }
        for name in [
            "dynamics.before_force",
            "dynamics.after_force",
            "dynamics.lf_kick_velocity_projection",
            "dynamics.lf_drift_thermostat_snapshot",
            "dynamics.lf_position_projection",
        ] {
            context.record_pipeline(name);
        }
        let validation = crate::pop_error_scope(&device).await;
        let allocation = crate::pop_error_scope(&device).await;
        if let Some(e) = validation {
            return Err(Error::Execution(e.to_string()));
        }
        if allocation.is_some() {
            return Err(Error::Capacity);
        }
        let constraint_group_count = groups.len() as u32;
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
            _constraint_groups: group_buffer,
            _constraints: constraint_buffer,
            _trial_coordinates: trial_coordinates,
            _observation_masses: observation_masses,
            observation_output,
            observation_bind,
            observation_pipeline,
            constraint_group_count,
            md_lanes_per_target,
            md_pipeline_start,
            packet_steps,
            dense_special_lookup_base,
            resident_initialized: false,
            gpu_stage_timings_ms: std::sync::Mutex::new(BTreeMap::new()),
            masses: system.atoms().iter().map(|a| a.mass()).collect(),
            _allocation: reservation,
        })
    }

    pub fn take_gpu_stage_timings_ms(&self) -> BTreeMap<String, f64> {
        self.gpu_stage_timings_ms
            .lock()
            .map(|mut timings| std::mem::take(&mut *timings))
            .unwrap_or_default()
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
        self.resident_initialized = false;
        self.advance_internal(
            coordinates,
            velocities,
            noise,
            None,
            steps,
            dt_ps,
            temperature,
            friction,
        )
        .await
    }
    /// Coordinates, starting forces and stochastic state are installed once.
    /// Later batches use the resident state, without host noise uploads.
    pub async fn advance_resident(
        &mut self,
        coordinates: &[Vec3],
        velocities: &[Vec3],
        rng: &[u32],
        steps: usize,
        dt_ps: f64,
        temperature: f64,
        friction: f64,
    ) -> Result<DynamicsBatch, Error> {
        self.advance_internal(
            coordinates,
            velocities,
            &[],
            Some(rng),
            steps,
            dt_ps,
            temperature,
            friction,
        )
        .await
    }

    /// Advance constrained implicit NVT with the OpenMM LF-middle operator
    /// sequence. Coordinates, velocities, and the persistent RNG stay on the
    /// device between calls; a complete committed snapshot is returned here.
    pub async fn advance_resident_lf_middle(
        &mut self,
        coordinates: &[Vec3],
        velocities: &[Vec3],
        rng: &[u32],
        steps: usize,
        dt_ps: f64,
        temperature: f64,
        friction: f64,
    ) -> Result<DynamicsBatch, Error> {
        let result = self
            .advance_lf_middle_inner(
                coordinates,
                velocities,
                rng,
                steps,
                dt_ps,
                temperature,
                friction,
                false,
            )
            .await;
        if result.is_err() {
            self.resident_initialized = false;
        }
        match result? {
            LfMiddleReadback::Snapshot(batch) => Ok(batch),
            LfMiddleReadback::Observation(_) => unreachable!("snapshot readback was requested"),
        }
    }

    /// Advance the same constrained LF-middle path while returning only the
    /// on-device potential and kinetic energy reductions.
    pub async fn advance_resident_lf_middle_observation(
        &mut self,
        coordinates: &[Vec3],
        velocities: &[Vec3],
        rng: &[u32],
        steps: usize,
        dt_ps: f64,
        temperature: f64,
        friction: f64,
    ) -> Result<DynamicsObservation, Error> {
        let result = self
            .advance_lf_middle_inner(
                coordinates,
                velocities,
                rng,
                steps,
                dt_ps,
                temperature,
                friction,
                true,
            )
            .await;
        if result.is_err() {
            self.resident_initialized = false;
        }
        match result? {
            LfMiddleReadback::Observation(observation) => Ok(observation),
            LfMiddleReadback::Snapshot(_) => unreachable!("scalar readback was requested"),
        }
    }

    async fn advance_lf_middle_inner(
        &mut self,
        coordinates: &[Vec3],
        velocities: &[Vec3],
        rng: &[u32],
        steps: usize,
        dt_ps: f64,
        temperature: f64,
        friction: f64,
        scalar_observation: bool,
    ) -> Result<LfMiddleReadback, Error> {
        let n = self.masses.len();
        if steps == 0
            || coordinates.len() != n
            || velocities.len() != n
            || rng.len() != n
            || rng.contains(&0)
            || !dt_ps.is_finite()
            || dt_ps <= 0.
            || dt_ps > 0.002
            || !temperature.is_finite()
            || temperature <= 0.
            || !friction.is_finite()
            || friction < 0.
            || coordinates
                .iter()
                .chain(velocities)
                .any(|v| !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite())
        {
            return Err(Error::Input("invalid LF-middle dynamics batch"));
        }
        if !self.resident_initialized {
            self.initialize_resident(coordinates, velocities, rng)
                .await?;
        }

        let integration_started = Instant::now();
        let decay = (-friction * dt_ps).exp();
        let sigma2 = (1. - decay * decay) * 0.00198720425864083 * temperature * 418.4;
        let device = &self.energy.device;
        let queue = &self.energy.queue;
        crate::push_error_scope(device, wgpu::ErrorFilter::Validation);
        queue.write_buffer(&self.status, 0, &[0; 16]);
        let atoms = (n as u32).div_ceil(64);
        let workgroup_size = if self.md_lanes_per_target >= 64 {
            128
        } else {
            64
        };
        let targets_per_workgroup = workgroup_size / self.md_lanes_per_target as u32;
        let md_targets = (n as u32).div_ceil(targets_per_workgroup);
        let groups = self.constraint_group_count.div_ceil(64);
        let mut remaining = steps;
        while remaining > 0 {
            let batch = remaining.min(self.packet_steps);
            crate::push_error_scope(device, wgpu::ErrorFilter::OutOfMemory);
            #[cfg(not(target_arch = "wasm32"))]
            let timestamp_query_count = if self.energy._context.gpu_timestamps_enabled() {
                u32::try_from(batch.checked_mul(12).ok_or(Error::Capacity)?)
                    .map_err(|_| Error::Capacity)?
            } else {
                0
            };
            #[cfg(target_arch = "wasm32")]
            let timestamp_query_count = 0u32;
            let timestamp_query_set = (timestamp_query_count > 0).then(|| {
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("GlySys implicit dynamics stage timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: timestamp_query_count,
                })
            });
            let timestamp_bytes = u64::from(timestamp_query_count) * 8;
            let timestamp_resolve = (timestamp_query_count > 0).then(|| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GlySys implicit timestamp resolve"),
                    size: timestamp_bytes,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            });
            let timestamp_readback = (timestamp_query_count > 0).then(|| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GlySys implicit timestamp readback"),
                    size: timestamp_bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            });
            let mut timestamp_stages = Vec::<(&'static str, u32, u32)>::new();
            let mut uniforms = vec![0u32; batch * 64];
            for i in 0..batch {
                uniforms[i * 64] = n as u32;
                uniforms[i * 64 + 3] = self.constraint_group_count;
                uniforms[i * 64 + 4] = (dt_ps as f32).to_bits();
                uniforms[i * 64 + 5] = (decay as f32).to_bits();
                uniforms[i * 64 + 6] = (sigma2 as f32).to_bits();
            }
            queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&uniforms));
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident LF-middle bounded packet"),
            });
            for step_index in 0..batch {
                let offset = (step_index * 256) as u32;
                let stages = [
                    (
                        2,
                        "LF full force kick and velocity projection",
                        groups,
                        "velocityKickProjection",
                    ),
                    (
                        3,
                        "LF OU, split drift, and trial snapshot",
                        atoms,
                        "thermostatAndDrift",
                    ),
                    (
                        4,
                        "LF position projection and velocity correction",
                        groups,
                        "positionProjection",
                    ),
                    (
                        self.md_pipeline_start,
                        "OBC2 tiled Born radii",
                        md_targets,
                        "bornRadii",
                    ),
                    (
                        self.md_pipeline_start + 1,
                        "OBC2 tiled Born adjoints",
                        md_targets,
                        "bornAdjoints",
                    ),
                    (
                        self.md_pipeline_start + 2,
                        "OBC2 tiled per-atom forces",
                        md_targets,
                        "directForces",
                    ),
                ];
                for (stage_index, (pipeline_index, label, workgroups, stage_name)) in
                    stages.into_iter().enumerate()
                {
                    let timestamp_writes = timestamp_query_set.as_ref().map(|query_set| {
                        let query = ((step_index * 6 + stage_index) * 2) as u32;
                        wgpu::ComputePassTimestampWrites {
                            query_set,
                            beginning_of_pass_write_index: Some(query),
                            end_of_pass_write_index: Some(query + 1),
                        }
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(label),
                        timestamp_writes,
                    });
                    if pipeline_index < self.pipelines.len() {
                        pass.set_pipeline(&self.pipelines[pipeline_index]);
                        pass.set_bind_group(0, &self.bind, &[offset]);
                    } else {
                        pass.set_pipeline(&self.energy.pipeline_set.pipelines[pipeline_index]);
                        pass.set_bind_group(0, &self.energy.bind_group, &[]);
                    }
                    pass.dispatch_workgroups(workgroups, 1, 1);
                    drop(pass);
                    timestamp_stages.push((
                        stage_name,
                        ((step_index * 6 + stage_index) * 2) as u32,
                        ((step_index * 6 + stage_index) * 2 + 1) as u32,
                    ));
                }
            }
            if let (Some(query_set), Some(resolve), Some(readback)) = (
                timestamp_query_set.as_ref(),
                timestamp_resolve.as_ref(),
                timestamp_readback.as_ref(),
            ) {
                encoder.resolve_query_set(query_set, 0..timestamp_query_count, resolve, 0);
                encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, timestamp_bytes);
            }
            queue.submit([encoder.finish()]);
            remaining -= batch;
            #[cfg(not(target_arch = "wasm32"))]
            if let Err(error) = device.poll(wgpu::PollType::Wait) {
                let _ = crate::pop_error_scope(device).await;
                let _ = crate::pop_error_scope(device).await;
                return Err(Error::Execution(error.to_string()));
            }
            if let Some(error) = crate::pop_error_scope(device).await {
                let _ = crate::pop_error_scope(device).await;
                return Err(Error::Execution(format!(
                    "LF-middle GPU packet failed: {error}"
                )));
            }
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(readback) = timestamp_readback.as_ref() {
                let slice = readback.slice(0..timestamp_bytes);
                let (tx, rx) = futures_channel::oneshot::channel();
                slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });
                device
                    .poll(wgpu::PollType::Wait)
                    .map_err(|error| Error::Execution(error.to_string()))?;
                rx.await
                    .map_err(|error| Error::Execution(error.to_string()))?
                    .map_err(|error| Error::Execution(error.to_string()))?;
                let mapped = slice.get_mapped_range();
                let values: &[u64] = bytemuck::cast_slice(&mapped);
                let period_ns =
                    self.energy
                        ._context
                        .gpu_timestamp_period_ns()
                        .ok_or_else(|| {
                            Error::Execution("GPU timestamp period is unavailable".into())
                        })?;
                let mut accumulated = self
                    .gpu_stage_timings_ms
                    .lock()
                    .map_err(|_| Error::Execution("GPU timing accumulator was poisoned".into()))?;
                for (stage, begin, end) in &timestamp_stages {
                    let duration_ns = values[*end as usize].wrapping_sub(values[*begin as usize])
                        as f64
                        * period_ns;
                    *accumulated.entry((*stage).to_owned()).or_default() +=
                        duration_ns / 1_000_000.0;
                }
                drop(accumulated);
                drop(mapped);
                readback.unmap();
            }
        }
        let host_enqueue_wait_ms = integration_started.elapsed().as_secs_f64() * 1000.0;

        let readback_started = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("resident LF-middle committed snapshot"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("OBC2 LF-middle committed energy reduction"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.energy.pipeline_set.pipelines[3]);
            pass.set_bind_group(0, &self.energy.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        if scalar_observation {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("OBC2 LF-middle scalar observation reduction"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.observation_pipeline);
                pass.set_bind_group(0, &self.observation_bind, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&self.observation_output, 0, &self.staging, 0, 16);
            encoder.copy_buffer_to_buffer(&self.status, 0, &self.staging, 16, 4);
            queue.submit([encoder.finish()]);
            let result = async {
                if let Some(error) = crate::pop_error_scope(device).await {
                    return Err(Error::Execution(error.to_string()));
                }
                let slice = self.staging.slice(..20);
                let (tx, rx) = futures_channel::oneshot::channel();
                slice.map_async(wgpu::MapMode::Read, move |value| {
                    let _ = tx.send(value);
                });
                #[cfg(not(target_arch = "wasm32"))]
                device
                    .poll(wgpu::PollType::Wait)
                    .map_err(|error| Error::Execution(error.to_string()))?;
                rx.await
                    .map_err(|error| Error::Execution(error.to_string()))?
                    .map_err(|error| Error::Execution(error.to_string()))?;
                let mapped = slice.get_mapped_range();
                let bytes = mapped.to_vec();
                drop(mapped);
                self.staging.unmap();
                let invalid = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
                if invalid != 0 {
                    return Err(Error::Execution(
                        "LF-middle constraint projection did not converge; restore checkpoint"
                            .into(),
                    ));
                }
                let values: &[f32] = bytemuck::cast_slice(&bytes[..16]);
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(Error::Nonfinite);
                }
                Ok(LfMiddleReadback::Observation(DynamicsObservation {
                    potential_energy: values[0] as f64,
                    kinetic_energy: values[1] as f64,
                    host_enqueue_wait_ms,
                    host_readback_ms: readback_started.elapsed().as_secs_f64() * 1000.0,
                    readback_bytes: 20,
                }))
            }
            .await;
            if result.is_err() {
                self.resident_initialized = false;
            }
            return result;
        }
        let block = n as u64 * 16;
        encoder.copy_buffer_to_buffer(&self.energy.buffers[2], 0, &self.staging, 0, block);
        encoder.copy_buffer_to_buffer(&self.velocities, 0, &self.staging, block, block);
        encoder.copy_buffer_to_buffer(
            &self.energy.buffers[7],
            block * 3,
            &self.staging,
            block * 2,
            block + 48,
        );
        encoder.copy_buffer_to_buffer(&self.status, 0, &self.staging, n as u64 * 48 + 48, 4);
        encoder.copy_buffer_to_buffer(&self.noise, 0, &self.staging, n as u64 * 48 + 52, block);
        queue.submit([encoder.finish()]);
        let result = async {
            if let Some(e) = crate::pop_error_scope(device).await {
                return Err(Error::Execution(e.to_string()));
            }
            let mapped_len = n as u64 * 64 + 52;
            let slice = self.staging.slice(..mapped_len);
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
            let invalid = u32::from_le_bytes(bytes[n * 48 + 48..n * 48 + 52].try_into().unwrap());
            if invalid != 0 {
                return Err(Error::Execution(
                    "LF-middle constraint projection did not converge; restore checkpoint".into(),
                ));
            }
            let floats = bytemuck::cast_slice::<u8, f32>(&bytes[..n * 48 + 48]);
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
            let gradients = floats[n * 8..n * 12]
                .chunks_exact(4)
                .map(|p| Vec3 {
                    x: p[0] as f64,
                    y: p[1] as f64,
                    z: p[2] as f64,
                })
                .collect();
            let components = floats[n * 12..n * 12 + 12].try_into().unwrap();
            let rng_words = bytemuck::cast_slice::<u8, u32>(&bytes[n * 48 + 52..])
                .chunks_exact(4)
                .map(|p| p[3])
                .collect();
            Ok(LfMiddleReadback::Snapshot(DynamicsBatch {
                coordinates: coords,
                velocities,
                gradients,
                components,
                rng_words: Some(rng_words),
                host_enqueue_wait_ms,
                host_readback_ms: readback_started.elapsed().as_secs_f64() * 1000.0,
            }))
        }
        .await;
        if result.is_err() {
            self.resident_initialized = false;
        }
        result
    }

    /// Evaluate and install an implicit-solvent starting state without
    /// advancing the integrator. This lets runtime sessions report the
    /// canonical initial step while keeping force evaluation and thermostat
    /// state resident on the device.
    pub async fn initialize_resident(
        &mut self,
        coordinates: &[Vec3],
        velocities: &[Vec3],
        rng: &[u32],
    ) -> Result<DynamicsBatch, Error> {
        let n = self.masses.len();
        if coordinates.len() != n
            || velocities.len() != n
            || rng.len() != n
            || rng.contains(&0)
            || coordinates
                .iter()
                .chain(velocities)
                .any(|v| !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite())
        {
            return Err(Error::Input("invalid implicit dynamics initial state"));
        }
        let positions = self.topology.coordinates(coordinates, &vec![true; n])?;
        let velocity_data: Vec<[f32; 4]> = velocities
            .iter()
            .zip(&self.masses)
            .map(|(v, m)| [v.x as f32, v.y as f32, v.z as f32, (1. / m) as f32])
            .collect();
        if velocity_data.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::Nonfinite);
        }
        let config = Config {
            size: [n as u32, 1, 0, 1],
            energy: [0., 1., 1., 0.],
            solvent: [1., 78.5, 1.4, 0.00542],
            spare: self
                .dense_special_lookup_base
                .map_or([0.; 4], |base| [f32::from_bits(base), 1.0, 0.0, 0.0]),
        };
        let result = self.energy.evaluate(config, &positions, true).await?;
        let gradients = result
            .gradients
            .ok_or(Error::Execution(
                "implicit initial gradients omitted".into(),
            ))?
            .into_iter()
            .map(|g| Vec3 {
                x: g[0] as f64,
                y: g[1] as f64,
                z: g[2] as f64,
            })
            .collect::<Vec<_>>();
        if gradients
            .iter()
            .any(|g| !g.x.is_finite() || !g.y.is_finite() || !g.z.is_finite())
        {
            return Err(Error::Nonfinite);
        }
        let words: Vec<[u32; 4]> = rng.iter().map(|&word| [0, 0, 0, word]).collect();
        self.energy
            .queue
            .write_buffer(&self.velocities, 0, bytemuck::cast_slice(&velocity_data));
        self.energy
            .queue
            .write_buffer(&self.noise, 0, bytemuck::cast_slice(&words));
        self.resident_initialized = true;
        Ok(DynamicsBatch {
            coordinates: coordinates.to_vec(),
            velocities: velocities.to_vec(),
            gradients,
            components: result.components[0],
            rng_words: Some(rng.to_vec()),
            host_enqueue_wait_ms: 0.0,
            host_readback_ms: 0.0,
        })
    }

    async fn advance_internal(
        &mut self,
        coordinates: &[Vec3],
        velocities: &[Vec3],
        noise: &[[f32; 4]],
        rng: Option<&[u32]>,
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
            || (rng.is_none() && (noise.len() != n * steps || steps > LEGACY_NOISE_STEPS))
            || rng.is_some_and(|r| r.len() != n || r.contains(&0))
            || !dt_ps.is_finite()
            || dt_ps <= 0.
            || dt_ps > 0.001
            || !temperature.is_finite()
            || temperature <= 0.
            || !friction.is_finite()
            || friction < 0.
        {
            return Err(Error::Input("invalid dynamics batch"));
        }
        if rng.is_none() || !self.resident_initialized {
            let p = self.topology.coordinates(coordinates, &vec![true; n])?;
            let v: Vec<[f32; 4]> = velocities
                .iter()
                .zip(&self.masses)
                .map(|(v, m)| [v.x as f32, v.y as f32, v.z as f32, (1. / m) as f32])
                .collect();
            if v.iter().flatten().any(|v| !v.is_finite()) {
                return Err(Error::Nonfinite);
            }
            let config = Config {
                size: [n as u32, 1, 0, 1],
                energy: [0., 1., 1., 0.],
                solvent: [1., 78.5, 1.4, 0.00542],
                spare: [0.; 4],
            };
            self.energy.evaluate(config, &p, true).await?;
            self.energy
                .queue
                .write_buffer(&self.velocities, 0, bytemuck::cast_slice(&v));
            if let Some(rng) = rng {
                let words: Vec<[u32; 4]> = rng.iter().map(|&word| [0, 0, 0, word]).collect();
                self.energy
                    .queue
                    .write_buffer(&self.noise, 0, bytemuck::cast_slice(&words));
            }
        }
        if rng.is_none() && noise.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::Nonfinite);
        }
        let integration_started = Instant::now();
        let decay = (-friction * dt_ps).exp();
        let mut uniforms = vec![0u32; MAX_STEPS * 64];
        for i in 0..steps {
            uniforms[i * 64] = n as u32;
            uniforms[i * 64 + 1] = if rng.is_some() { 0 } else { (i * n) as u32 };
            uniforms[i * 64 + 2] = u32::from(rng.is_some());
            uniforms[i * 64 + 4] = (dt_ps as f32).to_bits();
            uniforms[i * 64 + 5] = (decay as f32).to_bits();
            uniforms[i * 64 + 6] =
                (((1. - decay * decay) * 0.00198720425864083 * temperature * 418.4) as f32)
                    .to_bits();
        }
        let device = &self.energy.device;
        let queue = &self.energy.queue;
        crate::push_error_scope(&device, wgpu::ErrorFilter::Validation);
        if rng.is_none() {
            queue.write_buffer(&self.noise, 0, bytemuck::cast_slice(noise));
        }
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
                pass.set_pipeline(&self.energy.pipeline_set.pipelines[i]);
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
            block * 3,
            &self.staging,
            block * 2,
            block + 48,
        );
        encoder.copy_buffer_to_buffer(&self.status, 0, &self.staging, n as u64 * 48 + 48, 4);
        if rng.is_some() {
            encoder.copy_buffer_to_buffer(&self.noise, 0, &self.staging, n as u64 * 48 + 52, block);
        }
        queue.submit([encoder.finish()]);
        let host_enqueue_wait_ms = integration_started.elapsed().as_secs_f64() * 1000.0;
        let readback_started = Instant::now();
        if let Some(e) = crate::pop_error_scope(&device).await {
            return Err(Error::Execution(e.to_string()));
        }
        let slice = self
            .staging
            .slice(..n as u64 * if rng.is_some() { 64 } else { 48 } + 52);
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
        let invalid = u32::from_le_bytes(bytes[n * 48 + 48..n * 48 + 52].try_into().unwrap());
        if invalid != 0 {
            return Err(Error::Execution(
                "unstable dynamics step; restore checkpoint".into(),
            ));
        }
        let floats = bytemuck::cast_slice::<u8, f32>(&bytes[..n * 48 + 48]);
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
        let gradients = floats[n * 8..n * 12]
            .chunks_exact(4)
            .map(|p| Vec3 {
                x: p[0] as f64,
                y: p[1] as f64,
                z: p[2] as f64,
            })
            .collect();
        let components = floats[n * 12..n * 12 + 12].try_into().unwrap();
        let rng_words = rng.map(|_| {
            bytemuck::cast_slice::<u8, u32>(&bytes[n * 48 + 52..])
                .chunks_exact(4)
                .map(|p| p[3])
                .collect()
        });
        self.resident_initialized = rng.is_some();
        Ok(DynamicsBatch {
            coordinates: coords,
            velocities,
            gradients,
            components,
            rng_words,
            host_enqueue_wait_ms,
            host_readback_ms: readback_started.elapsed().as_secs_f64() * 1000.0,
        })
    }
}
