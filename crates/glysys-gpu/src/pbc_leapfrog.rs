//! Host side of the resident leap-frog integrator with Nose-Hoover and
//! Parrinello-Rahman coupling (`pbc_leapfrog.wgsl`).
//!
//! The coupling set-up arrives as plain numbers ([`LeapfrogCoupling`]) from
//! `glysys-dynamics::coupling::CouplingPlan`, and the kernels of every step
//! are chosen here from the step number alone ([`StepPlan`]), the way
//! `ExplicitSimulation::leapfrog_step` decides them on the CPU.
use crate::device::Error;

/// Coupling of one prepared system and protocol stage. At most two
/// temperature groups and two center-of-mass groups (water and the rest).
#[derive(Clone, Debug)]
pub struct LeapfrogCoupling {
    /// Per atom: bit 0 is the temperature group, bit 1 the center-of-mass
    /// group.
    pub group_bits: Vec<u32>,
    /// `1/Q` of the two thermostats, 1/(K ps²); zero for an unused group.
    pub inverse_q: [f64; 2],
    pub reference_temperature_k: [f64; 2],
    /// Kinetic degrees of freedom of the two temperature groups.
    pub degrees_of_freedom: [f64; 2],
    /// Total mass of the two center-of-mass groups, amu; zero when unused.
    pub com_mass: [f64; 2],
    pub timestep_ps: f64,
    /// False for a constant-energy stage.
    pub thermostat: bool,
    pub barostat: bool,
    pub temperature_interval: usize,
    pub pressure_interval: usize,
    /// Zero leaves center-of-mass motion alone.
    pub com_interval: usize,
    /// `4 π² β / (3 τ_p²)`, 1/(bar ps²).
    pub barostat_coefficient: f64,
    pub reference_pressure_bar: f64,
    /// Long-range dispersion term of the pressure, kcal Å³/mol: the pressure
    /// gains this over the squared volume.
    pub dispersion_pressure_coefficient: f64,
    /// Box change (Å) a pair list tolerates before it is rebuilt; taken from
    /// the neighbor skin.
    pub box_change_allowance: f64,
}

/// Thermostat and barostat variables: uploaded at the start of a segment and
/// read back at its observation points.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LeapfrogVariables {
    pub box_angstrom: [f64; 3],
    pub box_velocity: [f64; 3],
    pub thermostat_velocity: [f64; 2],
    pub thermostat_position: [f64; 2],
    /// Pressure of the last pressure step, bar, and whether the barostat has
    /// yet to use it.
    pub pressure_bar: f64,
    pub pressure_pending: bool,
    /// Kinetic energy of the temperature groups after the last step that
    /// measured it, kcal/mol.
    pub kinetic_energy: [f64; 2],
}

/// What one step does, from its number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StepPlan {
    pub thermostat: bool,
    pub barostat: bool,
    pub pressure: bool,
    pub com: bool,
    /// The forces at the end of this step must come with their virial.
    pub virial_next: bool,
}

fn acts_on(step: u64, interval: usize) -> bool {
    interval <= 1 || step % interval as u64 == 1
}

impl LeapfrogCoupling {
    pub(crate) fn plan(&self, step: u64) -> StepPlan {
        let pressure_interval = self.pressure_interval.max(1) as u64;
        StepPlan {
            thermostat: self.thermostat && acts_on(step, self.temperature_interval),
            barostat: self.barostat && acts_on(step, self.pressure_interval),
            pressure: self.barostat && step % pressure_interval == 0,
            com: self.com_interval > 0 && step % self.com_interval as u64 == 0,
            virial_next: self.barostat && (step + 1) % pressure_interval == 0,
        }
    }

    fn validate(&self, atoms: u32) -> Result<(), Error> {
        let finite = |values: &[f64]| values.iter().all(|v| v.is_finite() && *v >= 0.0);
        if self.group_bits.len() != atoms as usize
            || self.group_bits.iter().any(|bits| *bits > 3)
            || !finite(&self.inverse_q)
            || !finite(&self.reference_temperature_k)
            || !finite(&self.degrees_of_freedom)
            || !finite(&self.com_mass)
            || !(self.timestep_ps.is_finite() && self.timestep_ps > 0.0)
            || self.temperature_interval == 0
            || self.pressure_interval == 0
            || !finite(&[self.barostat_coefficient, self.box_change_allowance])
            || !self.reference_pressure_bar.is_finite()
            || !self.dispersion_pressure_coefficient.is_finite()
        {
            return Err(Error::Input("leap-frog coupling"));
        }
        Ok(())
    }
}

/// Entry points with the global bindings each one statically uses.
pub(crate) const KERNELS: [(&str, &[u32]); 8] = [
    ("reduce_partial", &[0, 1, 2, 3, 4, 5]),
    ("reduce_final", &[0, 5]),
    ("couple", &[0, 9]),
    ("kick_drift", &[0, 1, 2, 3, 4]),
    ("pressure", &[0, 3]),
    ("scale", &[0, 1]),
    ("apply_box", &[0, 6, 7, 8, 9]),
    ("remove_com", &[0, 2, 4]),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeapfrogKernel {
    ReducePartialOld,
    ReduceFinalOld,
    ReducePartialNew,
    ReduceFinalNew,
    CoupleThermostat,
    CoupleBarostat,
    CoupleBoth,
    KickDrift,
    KickDriftCoupled,
    Pressure,
    Scale,
    ApplyBox,
    RemoveCom,
}

impl LeapfrogKernel {
    pub(crate) const ALL: [LeapfrogKernel; 13] = [
        Self::ReducePartialOld,
        Self::ReduceFinalOld,
        Self::ReducePartialNew,
        Self::ReduceFinalNew,
        Self::CoupleThermostat,
        Self::CoupleBarostat,
        Self::CoupleBoth,
        Self::KickDrift,
        Self::KickDriftCoupled,
        Self::Pressure,
        Self::Scale,
        Self::ApplyBox,
        Self::RemoveCom,
    ];

    fn entry(self) -> &'static str {
        match self {
            Self::ReducePartialOld | Self::ReducePartialNew => "reduce_partial",
            Self::ReduceFinalOld | Self::ReduceFinalNew => "reduce_final",
            Self::CoupleThermostat | Self::CoupleBarostat | Self::CoupleBoth => "couple",
            Self::KickDrift | Self::KickDriftCoupled => "kick_drift",
            Self::Pressure => "pressure",
            Self::Scale => "scale",
            Self::ApplyBox => "apply_box",
            Self::RemoveCom => "remove_com",
        }
    }

    fn constants(self) -> &'static [(&'static str, f64)] {
        match self {
            Self::ReducePartialNew | Self::ReduceFinalNew => &[("REDUCE_NEW", 1.0)],
            Self::CoupleThermostat => &[("THERMOSTAT", 1.0)],
            Self::CoupleBarostat => &[("BAROSTAT", 1.0)],
            Self::CoupleBoth => &[("THERMOSTAT", 1.0), ("BAROSTAT", 1.0)],
            Self::KickDriftCoupled => &[("COUPLED", 1.0)],
            _ => &[],
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn stage_name(self) -> &'static str {
        match self {
            Self::KickDrift | Self::KickDriftCoupled => "integration",
            _ => "coupling",
        }
    }

    pub(crate) fn job(self) -> usize {
        Self::ALL.iter().position(|kernel| *kernel == self).unwrap()
    }

    pub(crate) fn from_job(job: usize) -> Self {
        Self::ALL[job]
    }
}

/// Atoms summed by one workgroup of the first reduction level
/// (`PARTIAL_ATOMS` in the shader).
const PARTIAL_ATOMS: u32 = 1024;
/// Bytes of the `Coupling` struct: sixteen vec4.
const COUPLING_BYTES: u64 = 256;

/// Buffers of [`crate::pbc::ResidentPbc`] and its engines that the
/// integrator reads or writes.
pub(crate) struct LeapfrogShared<'a> {
    pub sys: &'a wgpu::Buffer,
    pub state: &'a wgpu::Buffer,
    pub out: &'a wgpu::Buffer,
    pub aux: &'a wgpu::Buffer,
    pub pbc_config: &'a wgpu::Buffer,
    pub tile_config: &'a wgpu::Buffer,
    /// Uniform of the PME mesh and the vec4 element holding its box.
    pub mesh_config: Option<(&'a wgpu::Buffer, u32)>,
    pub status_word: u32,
    pub rebuild_word: u32,
    pub cutoff: f32,
}

pub(crate) struct LeapfrogEngine {
    pub coupling: LeapfrogCoupling,
    n: u32,
    partial_groups: u32,
    status_word: u32,
    rebuild_word: u32,
    cutoff: f32,
    pub cs: wgpu::Buffer,
    _groups: wgpu::Buffer,
    _partials: wgpu::Buffer,
    _mesh_placeholder: Option<wgpu::Buffer>,
    pipelines: Vec<wgpu::ComputePipeline>,
    bind_groups: Vec<wgpu::BindGroup>,
}

impl LeapfrogEngine {
    pub fn allocation_bytes(atoms: u32) -> u64 {
        COUPLING_BYTES + 4 * u64::from(atoms) + 48 * u64::from(atoms.div_ceil(PARTIAL_ATOMS)) + 64
    }

    pub fn new(
        device: &wgpu::Device,
        shared: LeapfrogShared<'_>,
        coupling: LeapfrogCoupling,
        atoms: u32,
    ) -> Result<Self, Error> {
        use wgpu::util::DeviceExt;
        coupling.validate(atoms)?;
        let storage = wgpu::BufferUsages::STORAGE;
        let partial_groups = atoms.div_ceil(PARTIAL_ATOMS).max(1);
        let cs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbc leapfrog coupling"),
            size: COUPLING_BYTES,
            usage: storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let groups = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbc leapfrog groups"),
            contents: bytemuck::cast_slice(&coupling.group_bits),
            usage: storage,
        });
        let partials = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbc leapfrog partials"),
            size: 48 * u64::from(partial_groups),
            usage: storage,
            mapped_at_creation: false,
        });
        let mesh_placeholder = shared.mesh_config.is_none().then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbc leapfrog mesh placeholder"),
                size: 64,
                usage: storage,
                mapped_at_creation: false,
            })
        });
        let (mesh_config, mesh_box_element) = match (&shared.mesh_config, &mesh_placeholder) {
            (Some((buffer, element)), _) => (*buffer, *element),
            (None, Some(buffer)) => (buffer, 1),
            (None, None) => unreachable!("a placeholder stands in for a missing mesh"),
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GlySys PBC leapfrog"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pbc_leapfrog.wgsl").into()),
        });
        let resource = |binding: u32| -> wgpu::BindingResource<'_> {
            match binding {
                0 => cs.as_entire_binding(),
                1 => shared.sys.as_entire_binding(),
                2 => shared.state.as_entire_binding(),
                3 => shared.out.as_entire_binding(),
                4 => groups.as_entire_binding(),
                5 => partials.as_entire_binding(),
                6 => shared.pbc_config.as_entire_binding(),
                7 => shared.tile_config.as_entire_binding(),
                8 => mesh_config.as_entire_binding(),
                9 => shared.aux.as_entire_binding(),
                _ => unreachable!("pbc_leapfrog.wgsl declares bindings 0..=9"),
            }
        };
        let mut pipelines = Vec::with_capacity(LeapfrogKernel::ALL.len());
        let mut bind_groups = Vec::with_capacity(LeapfrogKernel::ALL.len());
        for kernel in LeapfrogKernel::ALL {
            let mut constants = kernel.constants().to_vec();
            if kernel == LeapfrogKernel::ApplyBox {
                constants.push(("MESH_BOX_ELEMENT", f64::from(mesh_box_element)));
            }
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(kernel.entry()),
                layout: None,
                module: &shader,
                entry_point: Some(kernel.entry()),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });
            let bindings = KERNELS
                .iter()
                .find(|(name, _)| *name == kernel.entry())
                .map(|(_, bindings)| *bindings)
                .expect("every leapfrog kernel has a binding list");
            let entries: Vec<_> = bindings
                .iter()
                .map(|&binding| wgpu::BindGroupEntry {
                    binding,
                    resource: resource(binding),
                })
                .collect();
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(kernel.entry()),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            }));
            pipelines.push(pipeline);
        }
        Ok(Self {
            coupling,
            n: atoms,
            partial_groups,
            status_word: shared.status_word,
            rebuild_word: shared.rebuild_word,
            cutoff: shared.cutoff,
            cs,
            _groups: groups,
            _partials: partials,
            _mesh_placeholder: mesh_placeholder,
            pipelines,
            bind_groups,
        })
    }

    pub fn kernel(&self, kernel: LeapfrogKernel) -> (&wgpu::ComputePipeline, &wgpu::BindGroup) {
        let index = kernel.job();
        (&self.pipelines[index], &self.bind_groups[index])
    }

    pub fn groups(&self, kernel: LeapfrogKernel) -> u32 {
        match kernel {
            LeapfrogKernel::ReducePartialOld | LeapfrogKernel::ReducePartialNew => {
                self.partial_groups
            }
            LeapfrogKernel::KickDrift
            | LeapfrogKernel::KickDriftCoupled
            | LeapfrogKernel::Scale
            | LeapfrogKernel::RemoveCom => self.n.div_ceil(64).max(1),
            _ => 1,
        }
    }

    /// Write the constants and the variables of a segment.
    pub fn upload(&self, queue: &wgpu::Queue, variables: &LeapfrogVariables, rebuild_count: u32) {
        let c = &self.coupling;
        let f = |v: f64| (v as f32).to_bits();
        let per_dof = |dof: f64| {
            if dof > 0.0 {
                2.0 / (dof * 0.00198720425864083)
            } else {
                0.0
            }
        };
        let inverse = |mass: f64| if mass > 0.0 { 1.0 / mass } else { 0.0 };
        let v = variables;
        let words: [u32; 64] = [
            // sizes
            self.n,
            self.partial_groups,
            self.status_word,
            self.rebuild_word,
            // box
            f(v.box_angstrom[0]),
            f(v.box_angstrom[1]),
            f(v.box_angstrom[2]),
            self.cutoff.to_bits(),
            // box velocity
            f(v.box_velocity[0]),
            f(v.box_velocity[1]),
            f(v.box_velocity[2]),
            0,
            // xi
            f(v.thermostat_velocity[0]),
            f(v.thermostat_velocity[1]),
            f(v.thermostat_position[0]),
            f(v.thermostat_position[1]),
            // factor
            0,
            0,
            0,
            0,
            // drag; w: a pressure is stored
            0,
            0,
            0,
            f(if v.pressure_pending { 1.0 } else { 0.0 }),
            // kinetic old
            f(v.kinetic_energy[0]),
            f(v.kinetic_energy[1]),
            0,
            0,
            // kinetic new
            f(v.kinetic_energy[0]),
            f(v.kinetic_energy[1]),
            0,
            0,
            // momenta
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            // pressure
            f(v.pressure_bar),
            0,
            0,
            0,
            // inverse_q
            f(c.inverse_q[0]),
            f(c.inverse_q[1]),
            f(c.reference_temperature_k[0]),
            f(c.reference_temperature_k[1]),
            // per_dof
            f(per_dof(c.degrees_of_freedom[0])),
            f(per_dof(c.degrees_of_freedom[1])),
            f(c.temperature_interval as f64 * c.timestep_ps),
            f(c.pressure_interval as f64 * c.timestep_ps),
            // barostat
            f(c.barostat_coefficient),
            f(c.reference_pressure_bar),
            f(c.dispersion_pressure_coefficient),
            f(c.timestep_ps),
            // com
            f(inverse(c.com_mass[0])),
            f(inverse(c.com_mass[1])),
            0,
            0,
            // list
            0,
            f(c.box_change_allowance),
            rebuild_count,
            0,
        ];
        queue.write_buffer(&self.cs, 0, bytemuck::cast_slice(&words));
    }

    pub fn decode(bytes: &[u8]) -> LeapfrogVariables {
        let words: &[f32] = bytemuck::cast_slice(bytes);
        let at = |element: usize, lane: usize| f64::from(words[4 * element + lane]);
        LeapfrogVariables {
            box_angstrom: [at(1, 0), at(1, 1), at(1, 2)],
            box_velocity: [at(2, 0), at(2, 1), at(2, 2)],
            thermostat_velocity: [at(3, 0), at(3, 1)],
            thermostat_position: [at(3, 2), at(3, 3)],
            pressure_bar: at(10, 0),
            pressure_pending: at(5, 3) != 0.0,
            kinetic_energy: [at(7, 0), at(7, 1)],
        }
    }

    pub const fn coupling_bytes() -> u64 {
        COUPLING_BYTES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates_and_binding_table_matches_static_use() {
        let source = include_str!("pbc_leapfrog.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("pbc_leapfrog.wgsl validates");
        for (index, entry) in module.entry_points.iter().enumerate() {
            let mut used: Vec<u32> = module
                .global_variables
                .iter()
                .filter(|(handle, _)| !info.get_entry_point(index)[*handle].is_empty())
                .filter_map(|(_, variable)| variable.binding.as_ref().map(|b| b.binding))
                .collect();
            used.sort_unstable();
            let listed = KERNELS
                .iter()
                .find(|(name, _)| *name == entry.name)
                .map(|(_, bindings)| bindings.to_vec())
                .unwrap_or_else(|| panic!("{} is missing from KERNELS", entry.name));
            assert_eq!(used, listed, "bindings of {}", entry.name);
            let storage = used.len();
            assert!(storage <= 8, "{} binds {storage} storage buffers", entry.name);
        }
        assert_eq!(module.entry_points.len(), KERNELS.len());
    }

    fn coupling() -> LeapfrogCoupling {
        LeapfrogCoupling {
            group_bits: vec![0; 3],
            inverse_q: [0.13, 0.0],
            reference_temperature_k: [300.0, 0.0],
            degrees_of_freedom: [6.0, 0.0],
            com_mass: [18.0, 0.0],
            timestep_ps: 0.002,
            thermostat: true,
            barostat: true,
            temperature_interval: 25,
            pressure_interval: 10,
            com_interval: 100,
            barostat_coefficient: 1e-6,
            reference_pressure_bar: 1.0,
            dispersion_pressure_coefficient: 0.0,
            box_change_allowance: 0.05,
        }
    }

    #[test]
    fn steps_follow_the_cpu_schedule() {
        let c = coupling();
        // The pressure of every tenth step drives the barostat on the next.
        assert!(c.plan(0).pressure && !c.plan(0).barostat && c.plan(0).com);
        assert!(c.plan(1).barostat && c.plan(1).thermostat && !c.plan(1).pressure);
        assert!(c.plan(9).virial_next && !c.plan(9).pressure);
        assert!(c.plan(10).pressure && c.plan(11).barostat && !c.plan(11).thermostat);
        assert!(c.plan(26).thermostat && !c.plan(26).barostat);
        assert!(c.plan(100).com && !c.plan(50).com);
        let mut nvt = coupling();
        nvt.barostat = false;
        assert!(!nvt.plan(10).pressure && !nvt.plan(11).barostat && !nvt.plan(9).virial_next);
    }
}
