//! Shared device ownership for all GlySys GPU workloads.
//!
//! A [`GpuContext`] is created once by the coordinating worker/process and is
//! passed to typed resident evaluators.  Workload modules only allocate their
//! own buffers and pipelines; adapter and device creation, limits, and the
//! aggregate allocation ledger live here.

use crate::device::{Error, MemoryProfile, high_performance_adapter_options};
use crate::{
    dynamics::ResidentDynamics,
    hydration::ResidentWaterProbe,
    scoring::PreparedGpuEvaluator,
    steric::{AttachmentLibrary, ResidentSteric},
};
use glysys::ParameterizedSystem;
use glysys_energy::{
    hydration::PhysicalProbe,
    pbc::NonbondedElectrostatics,
    scoring::{PreparedScene, ScoreModel},
};
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

/// Options used when creating a shared GPU context.
#[derive(Clone, Debug)]
pub struct GpuContextOptions {
    pub memory_profile: MemoryProfile,
    pub label: String,
    /// Request optional native Vulkan timestamp queries for diagnostic runs.
    /// The default stays disabled so browser and production paths are unchanged.
    pub gpu_timestamps: bool,
    /// Opt in to the cooperative explicit PBC pair kernel. Browser and shared
    /// runtime callers keep the serial-per-atom kernel unless requested.
    pub pbc_tiled_nonbonded: bool,
}

impl Default for GpuContextOptions {
    fn default() -> Self {
        Self {
            memory_profile: MemoryProfile::Adaptive,
            label: "GlySys GPU context".into(),
            gpu_timestamps: false,
            pbc_tiled_nonbonded: false,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct AllocationStats {
    pub budget: u64,
    pub used_bytes: u64,
    pub peak_bytes: u64,
    pub reservations: u64,
}

/// Cached shader/layout/pipeline resources shared by repeated workloads.
pub struct PipelineSet {
    pub bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pub pipeline_layout: Arc<wgpu::PipelineLayout>,
    pub shaders: Vec<Arc<wgpu::ShaderModule>>,
    pub pipelines: Vec<Arc<wgpu::ComputePipeline>>,
}

#[derive(Default)]
struct PipelineCache {
    sets: HashMap<String, Arc<PipelineSet>>,
}

#[derive(Debug)]
struct AllocationLedger {
    stats: AllocationStats,
}

impl AllocationLedger {
    fn new(budget: u64) -> Self {
        Self {
            stats: AllocationStats {
                budget,
                ..AllocationStats::default()
            },
        }
    }

    fn reserve(&mut self, bytes: u64) -> Result<(), Error> {
        if bytes > self.stats.budget
            || self
                .stats
                .used_bytes
                .checked_add(bytes)
                .is_none_or(|next| next > self.stats.budget)
        {
            return Err(Error::Capacity);
        }
        self.stats.used_bytes += bytes;
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.used_bytes);
        self.stats.reservations += 1;
        Ok(())
    }

    fn release(&mut self, bytes: u64) {
        self.stats.used_bytes = self.stats.used_bytes.saturating_sub(bytes);
    }
}

/// An RAII reservation in the shared allocation ledger.
pub struct AllocationReservation {
    ledger: Arc<Mutex<AllocationLedger>>,
    bytes: u64,
}

impl AllocationReservation {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for AllocationReservation {
    fn drop(&mut self) {
        if let Ok(mut ledger) = self.ledger.lock() {
            ledger.release(self.bytes);
        }
    }
}

struct GpuContextInner {
    // Keeping the instance and adapter in the same owner as the device is
    // required by wgpu on some platforms while asynchronous work is pending.
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: wgpu::Limits,
    ledger: Arc<Mutex<AllocationLedger>>,
    pipelines: Mutex<BTreeSet<String>>,
    pipeline_cache: Mutex<PipelineCache>,
    memory_profile: MemoryProfile,
    gpu_timestamps_enabled: bool,
    pbc_tiled_nonbonded: bool,
}

/// Shared, coordinator-owned GPU state. Cloning this value only clones a
/// handle; it never requests a second adapter or device.
#[derive(Clone)]
pub struct GpuContext(Arc<GpuContextInner>);

impl GpuContext {
    pub async fn new(options: GpuContextOptions) -> Result<Self, Error> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&high_performance_adapter_options())
            .await
            .map_err(|e| Error::Unavailable(e.to_string()))?;
        let adapter_info = adapter.get_info();
        let limits = adapter.limits();
        let timestamp_features =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        let gpu_timestamps_enabled =
            options.gpu_timestamps && adapter.features().contains(timestamp_features);
        let required_features = if gpu_timestamps_enabled {
            timestamp_features
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(&options.label),
                required_features,
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| Error::Unavailable(e.to_string()))?;
        #[cfg(target_arch = "wasm32")]
        device.on_uncaptured_error(Box::new(|error| {
            web_sys::console::error_1(&wasm_bindgen::JsValue::from_str(&format!(
                "GlySys WebGPU uncaptured error: {error}"
            )));
        }));
        Ok(Self(Arc::new(GpuContextInner {
            _instance: instance,
            _adapter: adapter,
            adapter_info,
            device,
            queue,
            limits,
            ledger: Arc::new(Mutex::new(AllocationLedger::new(
                options.memory_profile.budget(),
            ))),
            pipelines: Mutex::new(BTreeSet::new()),
            pipeline_cache: Mutex::new(PipelineCache::default()),
            memory_profile: options.memory_profile,
            gpu_timestamps_enabled,
            pbc_tiled_nonbonded: options.pbc_tiled_nonbonded,
        })))
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.0.adapter_info
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.0.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.0.queue
    }

    pub fn limits(&self) -> &wgpu::Limits {
        &self.0.limits
    }

    pub fn memory_profile(&self) -> MemoryProfile {
        self.0.memory_profile
    }

    pub fn gpu_timestamps_enabled(&self) -> bool {
        self.0.gpu_timestamps_enabled
    }

    pub fn pbc_tiled_nonbonded_enabled(&self) -> bool {
        self.0.pbc_tiled_nonbonded
    }

    pub fn gpu_timestamp_period_ns(&self) -> Option<f64> {
        let period = self.0.queue.get_timestamp_period();
        (self.0.gpu_timestamps_enabled && period.is_finite() && period > 0.0)
            .then_some(f64::from(period))
    }

    /// Reserve aggregate bytes before creating a workload's buffers. The
    /// reservation must be retained by that workload for its lifetime.
    pub fn reserve(&self, bytes: u64) -> Result<AllocationReservation, Error> {
        let mut ledger = self.0.ledger.lock().map_err(|_| Error::Capacity)?;
        ledger.reserve(bytes)?;
        Ok(AllocationReservation {
            ledger: Arc::clone(&self.0.ledger),
            bytes,
        })
    }

    pub fn allocation_stats(&self) -> AllocationStats {
        self.0
            .ledger
            .lock()
            .map(|ledger| ledger.stats.clone())
            .unwrap_or_else(|_| AllocationStats {
                budget: self.memory_profile().budget(),
                ..AllocationStats::default()
            })
    }

    /// Record a compiled entry point for diagnostics and future pipeline
    /// caching. Compilation itself remains in the typed workload module so
    /// its bind layout stays explicit and reviewable.
    pub fn record_pipeline(&self, label: impl Into<String>) {
        if let Ok(mut pipelines) = self.0.pipelines.lock() {
            pipelines.insert(label.into());
        }
    }

    pub fn pipeline_labels(&self) -> Vec<String> {
        self.0
            .pipelines
            .lock()
            .map(|pipelines| pipelines.iter().cloned().collect())
            .unwrap_or_default()
    }

    fn cached_pipeline_set(
        &self,
        key: &str,
        create: impl FnOnce() -> Result<PipelineSet, Error>,
    ) -> Result<(Arc<PipelineSet>, bool), Error> {
        if let Some(set) = self
            .0
            .pipeline_cache
            .lock()
            .map_err(|_| Error::Execution("pipeline cache lock poisoned".into()))?
            .sets
            .get(key)
        {
            return Ok((Arc::clone(set), false));
        }
        let set = Arc::new(create()?);
        let mut cache = self
            .0
            .pipeline_cache
            .lock()
            .map_err(|_| Error::Execution("pipeline cache lock poisoned".into()))?;
        let set = Arc::clone(cache.sets.entry(key.to_string()).or_insert(set));
        Ok((set, true))
    }

    /// Compile and retain the canonical steric shader and its two pipelines.
    pub fn steric_pipeline_set(&self) -> Result<(Arc<PipelineSet>, bool), Error> {
        self.cached_pipeline_set("steric.v1", || {
            let device = self.device();
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
            let bind_group_layout = Arc::new(device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("steric bind group"),
                    entries: &entries,
                },
            ));
            let pipeline_layout = Arc::new(device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: Some("steric pipeline"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                },
            ));
            let shader = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("attachment transforms and sterics"),
                source: wgpu::ShaderSource::Wgsl(include_str!("steric.wgsl").into()),
            }));
            let pipelines = ["transform", "evaluate"]
                .iter()
                .map(|name| {
                    Arc::new(
                        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(name),
                            layout: Some(&pipeline_layout),
                            module: &shader,
                            entry_point: Some(name),
                            compilation_options: Default::default(),
                            cache: None,
                        }),
                    )
                })
                .collect();
            for name in ["steric.transform", "steric.evaluate"] {
                self.record_pipeline(name);
            }
            Ok(PipelineSet {
                bind_group_layout,
                pipeline_layout,
                shaders: vec![shader],
                pipelines,
            })
        })
    }

    /// Compile and retain the canonical energy shader variants and pipelines.
    pub fn energy_pipeline_set(&self) -> Result<(Arc<PipelineSet>, bool), Error> {
        self.cached_pipeline_set("energy.v1", || {
            let device = self.device();
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
            let bind_group_layout = Arc::new(device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("energy bind group"),
                    entries: &entries,
                },
            ));
            let pipeline_layout = Arc::new(device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: Some("energy pipeline"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                },
            ));
            fn energy_source(gradients: bool, md_lanes: usize, md_workgroup: usize) -> String {
                let source = include_str!("energy.wgsl")
                    .replacen(
                        "const COMPUTE_GRADIENTS: bool = true;",
                        &format!("const COMPUTE_GRADIENTS: bool = {gradients};"),
                        1,
                    )
                    .replace(
                        "const MD_LANES_PER_TARGET:u32=8u;",
                        &format!("const MD_LANES_PER_TARGET:u32={md_lanes}u;"),
                    )
                    .replace(
                        "const MD_WORKGROUP_SIZE:u32=64u;",
                        &format!("const MD_WORKGROUP_SIZE:u32={md_workgroup}u;"),
                    );
                if md_workgroup == 128 {
                    source
                        .replace("array<f32,64>", "array<f32,128>")
                        .replace("array<vec4<f32>,64>", "array<vec4<f32>,128>")
                } else {
                    source
                }
            }
            let shader = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("GlySys energies"),
                source: wgpu::ShaderSource::Wgsl(energy_source(true, 8, 64).into()),
            }));
            let shader_nograd =
                Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("GlySys energies (score-only)"),
                    source: wgpu::ShaderSource::Wgsl(energy_source(false, 8, 64).into()),
                }));
            let shader_md4 = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("GlySys energies (MD 4 lanes per target)"),
                source: wgpu::ShaderSource::Wgsl(energy_source(true, 4, 64).into()),
            }));
            let shader_md16 = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("GlySys energies (MD 16 lanes per target)"),
                source: wgpu::ShaderSource::Wgsl(energy_source(true, 16, 64).into()),
            }));
            let shader_md32 = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("GlySys energies (MD 32 lanes per target)"),
                source: wgpu::ShaderSource::Wgsl(energy_source(true, 32, 64).into()),
            }));
            let shader_md64 = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("GlySys energies (MD 64 lanes per target)"),
                source: wgpu::ShaderSource::Wgsl(energy_source(true, 64, 128).into()),
            }));
            let shader_md128 =
                Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("GlySys energies (MD 128 lanes per target)"),
                    source: wgpu::ShaderSource::Wgsl(energy_source(true, 128, 128).into()),
                }));
            let specs = [
                (&shader, "born_radii", true),
                (&shader, "born_adjoint", true),
                (&shader, "evaluate", true),
                (&shader, "reduce", true),
                (&shader_nograd, "evaluate", false),
                (&shader_md4, "born_radii_md", true),
                (&shader_md4, "born_adjoint_md", true),
                (&shader_md4, "evaluate_md", true),
                (&shader, "born_radii_md", true),
                (&shader, "born_adjoint_md", true),
                (&shader, "evaluate_md", true),
                (&shader_md16, "born_radii_md", true),
                (&shader_md16, "born_adjoint_md", true),
                (&shader_md16, "evaluate_md", true),
                (&shader_md32, "born_radii_md", true),
                (&shader_md32, "born_adjoint_md", true),
                (&shader_md32, "evaluate_md", true),
                (&shader_md64, "born_radii_md", true),
                (&shader_md64, "born_adjoint_md", true),
                (&shader_md64, "evaluate_md", true),
                (&shader_md128, "born_radii_md", true),
                (&shader_md128, "born_adjoint_md", true),
                (&shader_md128, "evaluate_md", true),
            ];
            let pipelines = specs
                .iter()
                .map(|(module, entry, _)| {
                    Arc::new(
                        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(entry),
                            layout: Some(&pipeline_layout),
                            module,
                            entry_point: Some(*entry),
                            compilation_options: Default::default(),
                            cache: None,
                        }),
                    )
                })
                .collect();
            for (_, entry, gradients) in specs {
                self.record_pipeline(format!("energy.{entry}.gradients={gradients}"));
            }
            Ok(PipelineSet {
                bind_group_layout,
                pipeline_layout,
                shaders: vec![
                    shader,
                    shader_nograd,
                    shader_md4,
                    shader_md16,
                    shader_md32,
                    shader_md64,
                    shader_md128,
                ],
                pipelines,
            })
        })
    }

    /// Typed workload factories keep device ownership in the coordinator and
    /// make it impossible for a resident evaluator to silently create a
    /// second adapter. Each returned session retains a lightweight handle to
    /// this context while owning only its workload buffers.
    pub async fn create_dynamics(
        &self,
        system: &ParameterizedSystem,
    ) -> Result<ResidentDynamics, Error> {
        self.create_dynamics_with_lanes(system, 8).await
    }

    /// Create implicit resident dynamics with a selectable all-pairs tile.
    /// Supported lane counts are 4, 8, and 16 per target atom.
    pub async fn create_dynamics_with_lanes(
        &self,
        system: &ParameterizedSystem,
        lanes_per_target: usize,
    ) -> Result<ResidentDynamics, Error> {
        self.create_dynamics_with_tuning(
            system,
            lanes_per_target,
            crate::dynamics::LF_MIDDLE_PACKET_STEPS,
        )
        .await
    }

    /// Create implicit resident dynamics with selected all-pairs lanes and
    /// bounded integration packet size.
    pub async fn create_dynamics_with_tuning(
        &self,
        system: &ParameterizedSystem,
        lanes_per_target: usize,
        packet_steps: usize,
    ) -> Result<ResidentDynamics, Error> {
        ResidentDynamics::with_context_tuning(system, self, lanes_per_target, packet_steps).await
    }

    pub async fn create_hydration(
        &self,
        probe: &PhysicalProbe,
        capacity: usize,
    ) -> Result<ResidentWaterProbe, Error> {
        ResidentWaterProbe::with_context(self, probe, capacity).await
    }

    pub async fn create_steric(
        &self,
        library: &AttachmentLibrary,
        capacity: u32,
    ) -> Result<ResidentSteric, Error> {
        ResidentSteric::with_context(self, library, capacity).await
    }

    pub async fn create_scoring(
        &self,
        scene: PreparedScene,
        model: ScoreModel,
        capacity: u32,
    ) -> Result<PreparedGpuEvaluator, Error> {
        PreparedGpuEvaluator::with_context(self, scene, model, capacity).await
    }

    pub async fn create_pbc(
        &self,
        packing: &crate::pbc::PbcPacking,
        backend: &NonbondedElectrostatics,
        max_pairs: u32,
    ) -> Result<crate::pbc::ResidentPbc, Error> {
        crate::pbc::ResidentPbc::with_context_variant(
            self,
            packing,
            backend,
            max_pairs,
            self.pbc_tiled_nonbonded_enabled(),
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_reservations_are_bounded_and_released() {
        let ledger = Arc::new(Mutex::new(AllocationLedger::new(16)));
        let reservation = {
            let mut state = ledger.lock().unwrap();
            state.reserve(8).unwrap();
            AllocationReservation {
                ledger: Arc::clone(&ledger),
                bytes: 8,
            }
        };
        assert_eq!(ledger.lock().unwrap().stats.used_bytes, 8);
        assert!(ledger.lock().unwrap().reserve(9).is_err());
        drop(reservation);
        assert_eq!(ledger.lock().unwrap().stats.used_bytes, 0);
    }
}
