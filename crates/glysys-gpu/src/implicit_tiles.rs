//! Host side of the tiled OBC2 force pass (`implicit_tiles.wgsl`) used by
//! [`crate::dynamics::ResidentDynamics`] in its LF-middle loop.
use crate::device::ResidentEvaluator;

/// Entry points with the global bindings each one statically uses (checked
/// against naga in the unit tests). Pipelines use automatic layouts.
pub(crate) const KERNELS: [(&str, &[u32]); 7] = [
    ("born_tiles", &[0, 1, 2, 8]),
    ("born_finish", &[0, 1, 6, 8]),
    ("adjoint_tiles", &[0, 1, 2, 6, 8]),
    ("adjoint_finish", &[0, 1, 6, 8]),
    ("gb_tiles", &[0, 1, 2, 5, 6, 8]),
    ("bonded_terms", &[0, 2, 3, 8, 9]),
    ("finalize_forces", &[0, 1, 6, 7, 8]),
];

/// Stages in dispatch order.
pub(crate) const STAGES: usize = 7;

const WIDE: u32 = 32_768;

pub(crate) struct ImplicitTiles {
    _acc: wgpu::Buffer,
    _uniform: wgpu::Buffer,
    /// (pipeline, bind group) for force-only and energy variants of each kernel.
    kernels: Vec<[(wgpu::ComputePipeline, wgpu::BindGroup); 2]>,
    atoms: u32,
    terms: u32,
}

impl ImplicitTiles {
    pub fn allocation_bytes(atoms: u32) -> u64 {
        4 * (8 * u64::from(atoms) + 18) + 16
    }

    pub fn new(evaluator: &ResidentEvaluator, atoms: u32, terms: u32) -> Self {
        use wgpu::util::DeviceExt;
        let device = &evaluator.device;
        let acc = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("implicit tiles accumulators"),
            size: Self::allocation_bytes(atoms) - 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("implicit tiles config"),
            contents: bytemuck::cast_slice(&[terms, 0u32, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GlySys implicit tiles"),
            source: wgpu::ShaderSource::Wgsl(include_str!("implicit_tiles.wgsl").into()),
        });
        let resource = |binding: u32| -> wgpu::BindingResource<'_> {
            match binding {
                0 => evaluator.buffers[0].as_entire_binding(),
                1 => evaluator.buffers[1].as_entire_binding(),
                2 => evaluator.buffers[2].as_entire_binding(),
                3 => evaluator.buffers[3].as_entire_binding(),
                5 => evaluator.buffers[5].as_entire_binding(),
                6 => evaluator.buffers[6].as_entire_binding(),
                7 => evaluator.buffers[7].as_entire_binding(),
                8 => acc.as_entire_binding(),
                9 => uniform.as_entire_binding(),
                _ => unreachable!("implicit_tiles.wgsl declares bindings 0-3 and 5-9"),
            }
        };
        let kernels = KERNELS
            .iter()
            .map(|(entry, bindings)| {
                [false, true].map(|energy| {
                    let constants: &[(&str, f64)] = if energy {
                        &[("COMPUTE_ENERGY", 1.0)]
                    } else {
                        &[]
                    };
                    let pipeline =
                        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(entry),
                            layout: None,
                            module: &shader,
                            entry_point: Some(entry),
                            compilation_options: wgpu::PipelineCompilationOptions {
                                constants,
                                zero_initialize_workgroup_memory: false,
                            },
                            cache: None,
                        });
                    let entries: Vec<_> = bindings
                        .iter()
                        .map(|&binding| wgpu::BindGroupEntry {
                            binding,
                            resource: resource(binding),
                        })
                        .collect();
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(entry),
                        layout: &pipeline.get_bind_group_layout(0),
                        entries: &entries,
                    });
                    (pipeline, bind_group)
                })
            })
            .collect();
        Self {
            _acc: acc,
            _uniform: uniform,
            kernels,
            atoms,
            terms,
        }
    }

    /// Encode the force pass (pair tiles, bonded terms, finalize) after the
    /// Born radii and adjoints have been computed for the current positions.
    pub fn encode(&self, pass: &mut wgpu::ComputePass<'_>, energy: bool, stage: usize) {
        let blocks = self.atoms.div_ceil(32);
        let terms = self.terms.div_ceil(64).max(1);
        let (pipeline, bind_group) = &self.kernels[stage][usize::from(energy)];
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        match stage {
            0 | 2 | 4 => pass.dispatch_workgroups(blocks, blocks, 1),
            5 => pass.dispatch_workgroups(terms.min(WIDE), terms.div_ceil(WIDE), 1),
            _ => pass.dispatch_workgroups(self.atoms.div_ceil(64), 1, 1),
        }
    }

    pub fn stage_name(stage: usize) -> &'static str {
        match stage {
            0 | 1 => "bornRadii",
            2 | 3 => "bornAdjoints",
            4 => "pairForces",
            5 => "bondedForces",
            _ => "forceFinalize",
        }
    }

    /// Whether `atoms` fits the two-dimensional tile dispatch.
    pub fn supports(atoms: u32, limits: &wgpu::Limits) -> bool {
        atoms.div_ceil(32) <= limits.max_compute_workgroups_per_dimension
            && atoms.div_ceil(64) <= limits.max_compute_workgroups_per_dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates_and_binding_table_matches_static_use() {
        let source = include_str!("implicit_tiles.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        for (index, entry) in module.entry_points.iter().enumerate() {
            let function = info.get_entry_point(index);
            let mut used: Vec<u32> = module
                .global_variables
                .iter()
                .filter(|(handle, _)| !function[*handle].is_empty())
                .filter_map(|(_, variable)| variable.binding.as_ref().map(|b| b.binding))
                .collect();
            used.sort_unstable();
            let declared = KERNELS
                .iter()
                .find(|(name, _)| *name == entry.name)
                .unwrap_or_else(|| panic!("{} missing from KERNELS", entry.name))
                .1;
            assert_eq!(used, declared, "{} binding table", entry.name);
        }
        assert_eq!(module.entry_points.len(), KERNELS.len());
    }
}
