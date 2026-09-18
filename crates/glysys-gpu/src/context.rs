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
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

/// Options used when creating a shared GPU context.
#[derive(Clone, Debug)]
pub struct GpuContextOptions {
    pub memory_profile: MemoryProfile,
    pub label: String,
}

impl Default for GpuContextOptions {
    fn default() -> Self {
        Self {
            memory_profile: MemoryProfile::Adaptive,
            label: "GlySys GPU context".into(),
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
    memory_profile: MemoryProfile,
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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(&options.label),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| Error::Unavailable(e.to_string()))?;
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
            memory_profile: options.memory_profile,
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

    /// Typed workload factories keep device ownership in the coordinator and
    /// make it impossible for a resident evaluator to silently create a
    /// second adapter. Each returned session retains a lightweight handle to
    /// this context while owning only its workload buffers.
    pub async fn create_dynamics(
        &self,
        system: &ParameterizedSystem,
    ) -> Result<ResidentDynamics, Error> {
        ResidentDynamics::with_context(system, self).await
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
        crate::pbc::ResidentPbc::with_context(self, packing, backend, max_pairs).await
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
