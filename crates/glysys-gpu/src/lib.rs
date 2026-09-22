//! Optional resident WebGPU backend. GPU handles stay on the invoking worker.
pub(crate) mod adapter;
pub mod context;
pub mod device;
pub mod pbc;
pub mod topology;

#[cfg(test)]
mod tests {
    #[test]
    fn pbc_shader_is_valid_portable_wgsl() {
        let source = format!(
            "{}\n{}",
            include_str!("resident_rng.wgsl"),
            include_str!("pbc.wgsl")
        );
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn shader_is_valid_portable_wgsl() {
        let source = include_str!("energy.wgsl");
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod execution_tests {
    use super::device::*;
    use crate::{GpuContext, GpuContextOptions};
    // Explicit opt-in: a software Vulkan adapter is useful for correctness, never performance evidence.
    #[test]
    #[ignore = "requires a Vulkan adapter; run explicitly for numerical validation"]
    fn two_atom_coulomb_energy_and_gradient() {
        pollster::block_on(async {
            let context = GpuContext::new(GpuContextOptions::default()).await.unwrap();
            let atoms = [
                Atom {
                    ff: [1., 0., 0., 1.5],
                    more: [0.8, 1., 0., 0.],
                    ranges: [0; 4],
                },
                Atom {
                    ff: [-1., 0., 0., 1.5],
                    more: [0.8, 2., 0., 0.],
                    ranges: [0; 4],
                },
            ];
            let mut gpu = ResidentEvaluator::with_context(
                &context,
                Topology {
                    atoms: &atoms,
                    terms: &[],
                    incidence: &[],
                    specials: &[],
                },
                2,
            )
            .await
            .unwrap();
            let config = Config {
                size: [2, 2, 1, 0],
                energy: [0., 1., 0., 0.],
                solvent: [1., 78.5, 1.4, 0.00542],
                spare: [0.; 4],
            };
            let result = gpu
                .evaluate(
                    config,
                    &[
                        [-1., 0., 0., 1.],
                        [1., 0., 0., 1.],
                        [-2., 0., 0., 1.],
                        [2., 0., 0., 1.],
                    ],
                    true,
                )
                .await
                .unwrap();
            assert!((result.components[0][5] + 332.063713299 / 2.).abs() < 0.001);
            assert!((result.components[1][5] + 332.063713299 / 4.).abs() < 0.001);
            let g = result.gradients.unwrap();
            assert!((g[0][0] + 332.063713299 / 4.).abs() < 0.001);
            assert!((g[1][0] - 332.063713299 / 4.).abs() < 0.001);
            // Reuse resident allocations with a smaller batch and scores-only readback.
            let result = gpu
                .evaluate(
                    Config {
                        size: [2, 1, 1, 0],
                        ..config
                    },
                    &[[-1., 0., 0., 1.], [1., 0., 0., 1.]],
                    false,
                )
                .await
                .unwrap();
            assert!(result.gradients.is_none());
            assert!((result.components[0][5] + 332.063713299 / 2.).abs() < 0.001);
        });
    }
}

pub mod steric;

pub mod scoring;

pub mod hydration;

pub mod dynamics;

/// WebGPU error scopes are useful on native backends, but Safari's WebGPU
/// implementation has returned a non-`GPUError` object from
/// `popErrorScope()`. wgpu 26 currently performs an internal `dyn_into().unwrap()`
/// for that value, which aborts the WASM worker as an opaque `unreachable`
/// trap. Browser shader/device errors are still surfaced by the device and
/// subsequent operation result paths; avoid the incompatible scope promise on
/// wasm until wgpu provides a fallible conversion.
pub(crate) fn push_error_scope(device: &wgpu::Device, filter: wgpu::ErrorFilter) {
    #[cfg(not(target_arch = "wasm32"))]
    device.push_error_scope(filter);
    #[cfg(target_arch = "wasm32")]
    let _ = (device, filter);
}

pub(crate) async fn pop_error_scope(device: &wgpu::Device) -> Option<wgpu::Error> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        device.pop_error_scope().await
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = device;
        None
    }
}

pub use context::{AllocationReservation, AllocationStats, GpuContext, GpuContextOptions};
pub use device::{ADAPTIVE_MEMORY_BUDGET, LOW_MEMORY_BUDGET, MemoryProfile};

#[derive(Clone, Debug, serde::Serialize)]
pub struct GpuAdapterInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub vendor: u32,
    pub device: u32,
    pub features: String,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_compute_workgroup_size: [u32; 3],
    pub max_compute_invocations_per_workgroup: u32,
}

/// Enumerate adapters visible to native wgpu. This is diagnostic only; the
/// returned limits and adapter type must still be checked for a workload.
#[cfg(not(target_arch = "wasm32"))]
pub fn adapter_report() -> Vec<GpuAdapterInfo> {
    wgpu::Instance::default()
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .map(|adapter| {
            let info = adapter.get_info();
            let limits = adapter.limits();
            GpuAdapterInfo {
                name: info.name,
                backend: format!("{:?}", info.backend),
                device_type: format!("{:?}", info.device_type),
                vendor: info.vendor,
                device: info.device,
                features: format!("{:?}", adapter.features()),
                max_buffer_size: limits.max_buffer_size,
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size as u64,
                max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
                max_compute_workgroup_size: [
                    limits.max_compute_workgroup_size_x,
                    limits.max_compute_workgroup_size_y,
                    limits.max_compute_workgroup_size_z,
                ],
                max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            }
        })
        .collect()
}

#[cfg(target_arch = "wasm32")]
pub fn adapter_report() -> Vec<GpuAdapterInfo> {
    Vec::new()
}
