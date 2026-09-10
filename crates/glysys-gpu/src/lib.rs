//! Optional resident WebGPU backend. GPU handles stay on the invoking worker.
pub mod device;
pub mod topology;

#[cfg(test)]
mod tests {
    #[test]
    fn shader_is_valid_portable_wgsl() {
        let source = include_str!("energy.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
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
    // Explicit opt-in: a software Vulkan adapter is useful for correctness, never performance evidence.
    #[test]
    #[ignore = "requires a Vulkan adapter; run explicitly for numerical validation"]
    fn two_atom_coulomb_energy_and_gradient() {
        pollster::block_on(async {
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
            let mut gpu = ResidentEvaluator::new(
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
