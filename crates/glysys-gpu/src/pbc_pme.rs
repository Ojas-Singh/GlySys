//! Host side of the reciprocal-space particle-mesh Ewald kernels
//! (`pbc_pme.wgsl`): mesh sizing, the constant tables, the pipelines and the
//! dispatch shapes. The mesh shares the coordinate, pair-gradient and
//! accumulator buffers of [`crate::pbc::ResidentPbc`]; see the shader for the
//! algorithm.
use crate::device::Error;
use crate::pbc_tiles::wide_groups;

/// Smallest and largest mesh size per axis. A mesh line is transformed inside
/// one workgroup, so it has to fit one of the two buffers of its memory.
const MIN_POINTS: u32 = 16;
const MAX_POINTS: u32 = 256;
/// Elements of one workgroup buffer, `LINE` in the shader: a workgroup
/// transforms `GROUP_ELEMENTS / K` neighbouring lines of `K` points.
const GROUP_ELEMENTS: u32 = 256;
/// Bytes of the uniform: grid and atom count, box and alpha, log2 of the grid.
const UNIFORM_BYTES: u64 = 48;
/// Entries of the twiddle table, `exp(-2 pi i j / 256)`.
const TWIDDLES: usize = MAX_POINTS as usize;

/// Byte offset in [`PmeMesh::uniform`] of the box lengths: three consecutive
/// `f32` (x, y, z, in angstrom). Every box-dependent quantity of the kernels
/// is derived from these twelve bytes at each dispatch, so a device kernel may
/// rewrite them through a storage binding between evaluations.
pub const BOX_OFFSET_BYTES: u64 = 16;

/// Entry points with the global bindings each one statically uses. Automatic
/// pipeline layouts are derived from the shader, so a bind group must supply
/// exactly these bindings (checked against naga in the unit tests).
pub(crate) const KERNELS: [(&str, &[u32]); 7] = [
    ("spread", &[0, 1, 2]),
    ("fft_x_forward", &[0, 3, 4, 6]),
    ("fft_y", &[0, 4, 6]),
    ("fft_z", &[0, 4, 6, 7]),
    ("fft_z_energy", &[0, 4, 6, 7, 9]),
    ("fft_x_inverse", &[0, 4, 5, 6]),
    ("gather", &[0, 1, 5, 8]),
];

/// Pipelines in creation order. `FftYForward` and `FftYInverse` are one entry
/// point with different override constants; the two convolutions are separate
/// entry points so that the force-only one does not bind the accumulators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PmeKernel {
    /// Charges onto the fixed-point mesh.
    Spread,
    /// Real-to-complex transform along x; clears the charge mesh.
    FftXForward,
    FftYForward,
    /// Forward z, influence function, inverse z.
    Convolve,
    /// [`Self::Convolve`] that also adds the reciprocal energy and virial to
    /// the accumulators.
    ConvolveEnergy,
    FftYInverse,
    /// Complex-to-real transform along x onto the potential mesh.
    FftXInverse,
    /// Adds every atom's reciprocal gradient to its pair gradient.
    Gather,
}

impl PmeKernel {
    const ALL: [PmeKernel; 8] = [
        Self::Spread,
        Self::FftXForward,
        Self::FftYForward,
        Self::Convolve,
        Self::ConvolveEnergy,
        Self::FftYInverse,
        Self::FftXInverse,
        Self::Gather,
    ];

    const FORCES: [PmeKernel; 7] = [
        Self::Spread,
        Self::FftXForward,
        Self::FftYForward,
        Self::Convolve,
        Self::FftYInverse,
        Self::FftXInverse,
        Self::Gather,
    ];

    const ENERGY: [PmeKernel; 7] = [
        Self::Spread,
        Self::FftXForward,
        Self::FftYForward,
        Self::ConvolveEnergy,
        Self::FftYInverse,
        Self::FftXInverse,
        Self::Gather,
    ];

    fn entry(self) -> &'static str {
        match self {
            Self::Spread => "spread",
            Self::FftXForward => "fft_x_forward",
            Self::FftYForward | Self::FftYInverse => "fft_y",
            Self::Convolve => "fft_z",
            Self::ConvolveEnergy => "fft_z_energy",
            Self::FftXInverse => "fft_x_inverse",
            Self::Gather => "gather",
        }
    }

    fn constants(self) -> &'static [(&'static str, f64)] {
        match self {
            Self::ConvolveEnergy => &[("COMPUTE_ENERGY", 1.0)],
            Self::FftYInverse => &[("INVERSE", 1.0)],
            _ => &[],
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|kernel| *kernel == self).unwrap()
    }

    /// Stable code of the kernel for a dispatcher's job table.
    pub fn job(self) -> usize {
        self.index()
    }

    pub fn from_job(job: usize) -> Self {
        Self::ALL[job]
    }

    /// Short name for timing reports.
    pub fn stage_name(self) -> &'static str {
        match self {
            Self::Spread => "pmeSpread",
            Self::FftXForward => "pmeFftXForward",
            Self::FftYForward => "pmeFftYForward",
            Self::Convolve | Self::ConvolveEnergy => "pmeConvolve",
            Self::FftYInverse => "pmeFftYInverse",
            Self::FftXInverse => "pmeFftXInverse",
            Self::Gather => "pmeGather",
        }
    }
}

/// Mesh size for a box: per axis the smallest power of two that is at least
/// `box / spacing`, no less than 16 and no more than 256.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PmeSizing {
    pub n: u32,
    pub grid: [u32; 3],
}

impl PmeSizing {
    pub fn new(n: u32, box_xyz: [f64; 3], fourier_spacing_angstrom: f64) -> Result<Self, Error> {
        if !fourier_spacing_angstrom.is_finite() || fourier_spacing_angstrom <= 0.0 {
            return Err(Error::Input("Fourier spacing must be positive"));
        }
        let mut grid = [0u32; 3];
        for (points, length) in grid.iter_mut().zip(box_xyz) {
            if !length.is_finite() || length <= 0.0 {
                return Err(Error::Input("box lengths must be finite and positive"));
            }
            // The small tolerance keeps an exact fit (32 A at 1 A) from
            // rounding up to the next size, as in `PmeParameters::for_box`.
            let needed = length / fourier_spacing_angstrom;
            let needed = (needed - 1e-9 * needed.max(1.0)).ceil();
            if !needed.is_finite() || needed > f64::from(MAX_POINTS) {
                return Err(Error::Capacity);
            }
            *points = (needed as u32).max(MIN_POINTS).next_power_of_two();
        }
        Self::with_grid(n, grid)
    }

    /// Sizing for a given mesh: powers of two from 16 to 256 per axis.
    pub fn with_grid(n: u32, grid: [u32; 3]) -> Result<Self, Error> {
        if grid
            .iter()
            .any(|points| !points.is_power_of_two() || *points < MIN_POINTS)
        {
            return Err(Error::Input(
                "PME mesh sizes must be powers of two, at least 16",
            ));
        }
        if grid.iter().any(|points| *points > MAX_POINTS) {
            return Err(Error::Capacity);
        }
        // The fixed-point accumulators of the tile engine index 6 words per
        // atom with u32.
        if n == 0 || n >= 1 << 24 {
            return Err(Error::Capacity);
        }
        Ok(Self { n, grid })
    }

    /// Points of the real meshes.
    fn points(&self) -> u64 {
        self.grid.iter().map(|points| u64::from(*points)).product()
    }

    /// Elements of the half spectrum (kx <= Kx/2).
    fn spectrum_points(&self) -> u64 {
        u64::from(self.grid[0] / 2 + 1) * u64::from(self.grid[1]) * u64::from(self.grid[2])
    }

    fn moduli_len(&self) -> u64 {
        self.grid.iter().map(|points| u64::from(*points)).sum()
    }

    /// Bytes of every buffer [`PmeMesh::new`] creates.
    pub fn allocation_bytes(&self) -> u64 {
        UNIFORM_BYTES
            + 4 * self.points()
            + 8 * self.spectrum_points()
            + 4 * self.points()
            + 8 * TWIDDLES as u64
            + 4 * self.moduli_len()
    }

    /// The largest single buffer, to compare with the device's storage
    /// binding limit.
    pub fn largest_binding_bytes(&self) -> u64 {
        (8 * self.spectrum_points()).max(4 * self.points())
    }

    /// Workgroup counts `(x, y)` of a kernel's direct dispatch `(x, y, 1)`:
    /// 16 atoms per workgroup, or the neighbouring mesh lines that fit the
    /// workgroup's memory.
    pub fn groups(&self, kernel: PmeKernel) -> (u32, u32) {
        let [kx, ky, kz] = self.grid;
        // `lines_per_group` of the shader.
        let lines = |points: u32, available: u32| (GROUP_ELEMENTS / points).min(available);
        match kernel {
            // Four lanes per atom.
            PmeKernel::Spread | PmeKernel::Gather => wide_groups(self.n.div_ceil(16).max(1)),
            // One complex line is the two real lines z and z + 1.
            PmeKernel::FftXForward | PmeKernel::FftXInverse => (kz / 2 / lines(kx, kz / 2), ky),
            PmeKernel::FftYForward | PmeKernel::FftYInverse => (kz / lines(ky, kz), kx / 2 + 1),
            PmeKernel::Convolve | PmeKernel::ConvolveEnergy => (ky / lines(kz, ky), kx / 2 + 1),
        }
    }
}

/// `1/|b(m)|^2` of the order-4 cardinal B-spline on `points` mesh points
/// (Essmann et al.), the reciprocal of `spline_moduli` in
/// `glysys_energy::pme`.
fn inverse_moduli(points: u32) -> Vec<f32> {
    // The order-4 spline at the mesh points: weights of points 1, 2 and 3.
    const THETA: [f64; 3] = [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0];
    let raw: Vec<f64> = (0..points)
        .map(|m| {
            let (mut re, mut im) = (0.0, 0.0);
            for (k, weight) in THETA.iter().enumerate() {
                let arg =
                    2.0 * std::f64::consts::PI * f64::from(m * (k as u32 + 1)) / f64::from(points);
                re += weight * arg.cos();
                im += weight * arg.sin();
            }
            re * re + im * im
        })
        .collect();
    let size = points as usize;
    (0..size)
        .map(|m| {
            // An even order has no zero on the mesh; the CPU engine's
            // fallback is kept so the two cannot drift apart.
            let modulus = if raw[m] < 1e-7 {
                0.5 * (raw[(m + size - 1) % size] + raw[(m + 1) % size])
            } else {
                raw[m]
            };
            (1.0 / modulus) as f32
        })
        .collect()
}

/// `exp(-2 pi i j / 256)`. A transform of `K` points reads every `256 / K`th
/// entry; the inverse conjugates.
fn twiddles() -> Vec<[f32; 2]> {
    (0..TWIDDLES)
        .map(|j| {
            let angle = -2.0 * std::f64::consts::PI * j as f64 / TWIDDLES as f64;
            [angle.cos() as f32, angle.sin() as f32]
        })
        .collect()
}

/// Buffers owned by the caller.
pub struct PmeShared<'a> {
    /// `array<vec4<f32>>`: `sys[2i] = (charge e, sigma, epsilon, mass)`,
    /// `sys[2i + 1] = (x, y, z, 0)` in angstrom, not wrapped. Read only.
    pub sys: &'a wgpu::Buffer,
    /// `array<vec4<f32>>`, one per atom: `(dE/dx, dE/dy, dE/dz, 0)`. The
    /// reciprocal gradient is added to it.
    pub pair_grad: &'a wgpu::Buffer,
    /// `array<atomic<u32>>` fixed-point accumulators of the tile engine. Only
    /// the energy convolution binds it.
    pub acc: &'a wgpu::Buffer,
}

/// The mesh, tables and pipelines of one system.
///
/// One evaluation is the kernels of [`Self::chain`] dispatched in order with
/// the shapes of [`Self::groups`]. Between the first and the last of them the
/// positions in `sys` and the box in the uniform must not change (spread and
/// gather both derive the spline weights from them), and no second
/// evaluation may start: there is one mesh. The gather has to come after the
/// pair kernel has stored `pair_grad` for the evaluation, since it adds to it.
///
/// Limits: mesh sizes are powers of two from 16 to 256; the charge on one
/// mesh point has to stay below 32 e in magnitude (32-bit fixed point with
/// scale 2^26); atoms without charge or with a non-finite position are left
/// out, silently.
pub struct PmeMesh {
    sizing: PmeSizing,
    alpha: f32,
    uniform: wgpu::Buffer,
    _charge: wgpu::Buffer,
    _spectrum: wgpu::Buffer,
    _potential: wgpu::Buffer,
    _twiddle: wgpu::Buffer,
    _moduli: wgpu::Buffer,
    pipelines: Vec<wgpu::ComputePipeline>,
    bind_groups: Vec<wgpu::BindGroup>,
}

impl PmeMesh {
    /// Create the mesh for a sizing. The caller checks
    /// [`PmeSizing::allocation_bytes`] against its budget and calls
    /// [`Self::upload`] with the box before the first evaluation.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sizing: PmeSizing,
        shared: PmeShared<'_>,
        alpha_per_angstrom: f32,
    ) -> Self {
        use wgpu::util::DeviceExt;
        let storage = wgpu::BufferUsages::STORAGE;
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbc pme config"),
            size: UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | storage,
            mapped_at_creation: false,
        });
        // wgpu creates buffers zeroed: the charge mesh starts empty and every
        // evaluation leaves it empty again.
        let charge = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbc pme charge mesh"),
            size: 4 * sizing.points(),
            usage: storage,
            mapped_at_creation: false,
        });
        let spectrum = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbc pme half spectrum"),
            size: 8 * sizing.spectrum_points(),
            usage: storage,
            mapped_at_creation: false,
        });
        let potential = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbc pme potential mesh"),
            size: 4 * sizing.points(),
            usage: storage,
            mapped_at_creation: false,
        });
        let twiddle = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbc pme twiddles"),
            contents: bytemuck::cast_slice(&twiddles()),
            usage: storage,
        });
        let moduli_data: Vec<f32> = sizing
            .grid
            .iter()
            .flat_map(|points| inverse_moduli(*points))
            .collect();
        let moduli = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbc pme spline moduli"),
            contents: bytemuck::cast_slice(&moduli_data),
            usage: storage,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GlySys PBC PME"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pbc_pme.wgsl").into()),
        });
        let resource = |binding: u32| -> wgpu::BindingResource<'_> {
            match binding {
                0 => uniform.as_entire_binding(),
                1 => shared.sys.as_entire_binding(),
                2 | 3 => charge.as_entire_binding(),
                4 => spectrum.as_entire_binding(),
                5 => potential.as_entire_binding(),
                6 => twiddle.as_entire_binding(),
                7 => moduli.as_entire_binding(),
                8 => shared.pair_grad.as_entire_binding(),
                9 => shared.acc.as_entire_binding(),
                _ => unreachable!("pbc_pme.wgsl declares bindings 0..=9"),
            }
        };
        let mut pipelines = Vec::with_capacity(PmeKernel::ALL.len());
        let mut bind_groups = Vec::with_capacity(PmeKernel::ALL.len());
        for kernel in PmeKernel::ALL {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(kernel.stage_name()),
                layout: None,
                module: &shader,
                entry_point: Some(kernel.entry()),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: kernel.constants(),
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });
            let bindings = KERNELS
                .iter()
                .find(|(name, _)| *name == kernel.entry())
                .map(|(_, bindings)| *bindings)
                .expect("every PME kernel has a binding list");
            let entries: Vec<_> = bindings
                .iter()
                .map(|&binding| wgpu::BindGroupEntry {
                    binding,
                    resource: resource(binding),
                })
                .collect();
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(kernel.stage_name()),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            pipelines.push(pipeline);
            bind_groups.push(bind_group);
        }
        let mesh = Self {
            sizing,
            alpha: alpha_per_angstrom,
            uniform,
            _charge: charge,
            _spectrum: spectrum,
            _potential: potential,
            _twiddle: twiddle,
            _moduli: moduli,
            pipelines,
            bind_groups,
        };
        // Everything but the box, so the uniform is never left half written.
        queue.write_buffer(
            &mesh.uniform,
            0,
            bytemuck::cast_slice(&mesh.words([0.0; 3])),
        );
        mesh
    }

    pub fn sizing(&self) -> &PmeSizing {
        &self.sizing
    }

    fn words(&self, box_xyz: [f32; 3]) -> [u32; 12] {
        let [kx, ky, kz] = self.sizing.grid;
        [
            kx,
            ky,
            kz,
            self.sizing.n,
            box_xyz[0].to_bits(),
            box_xyz[1].to_bits(),
            box_xyz[2].to_bits(),
            self.alpha.to_bits(),
            kx.trailing_zeros(),
            ky.trailing_zeros(),
            kz.trailing_zeros(),
            0,
        ]
    }

    /// Write the uniform for a box (grid, atom count, alpha, box). Called at
    /// start and whenever the host changes the box; the mesh size and alpha
    /// stay fixed, as in GROMACS.
    pub fn upload(&self, queue: &wgpu::Queue, box_xyz: [f32; 3]) {
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&self.words(box_xyz)));
    }

    /// The uniform buffer (`UNIFORM | COPY_DST | STORAGE`), with the box
    /// lengths at [`BOX_OFFSET_BYTES`].
    pub fn uniform(&self) -> &wgpu::Buffer {
        &self.uniform
    }

    /// Kernels of one evaluation in dispatch order: for a force-only step,
    /// or for a step that also adds the reciprocal energy and virial to the
    /// accumulators.
    pub fn chain(energy: bool) -> &'static [PmeKernel] {
        if energy {
            &PmeKernel::ENERGY
        } else {
            &PmeKernel::FORCES
        }
    }

    pub fn kernel(&self, kernel: PmeKernel) -> (&wgpu::ComputePipeline, &wgpu::BindGroup) {
        let index = kernel.index();
        (&self.pipelines[index], &self.bind_groups[index])
    }

    /// Workgroup counts `(x, y)` of the direct dispatch `(x, y, 1)`.
    pub fn groups(&self, kernel: PmeKernel) -> (u32, u32) {
        self.sizing.groups(kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates_and_binding_table_matches_static_use() {
        let source = include_str!("pbc_pme.wgsl");
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
            let storage = used.iter().filter(|binding| **binding != 0).count();
            assert!(
                storage <= 8,
                "{} exceeds eight storage bindings",
                entry.name
            );
        }
        assert_eq!(module.entry_points.len(), KERNELS.len());
        // The force-only chain never binds the accumulators.
        for kernel in PmeMesh::chain(false) {
            let bindings = KERNELS
                .iter()
                .find(|(name, _)| *name == kernel.entry())
                .unwrap()
                .1;
            assert!(!bindings.contains(&9), "{kernel:?} binds the accumulators");
        }
        for kernel in PmeKernel::ALL {
            assert!(KERNELS.iter().any(|(name, _)| *name == kernel.entry()));
        }
    }

    /// The host's dispatch shapes assume the shader's buffer size.
    #[test]
    fn group_size_matches_the_shader() {
        let source = include_str!("pbc_pme.wgsl");
        assert!(source.contains(&format!("const LINE: u32 = {GROUP_ELEMENTS}u;")));
        assert!(GROUP_ELEMENTS >= MAX_POINTS && GROUP_ELEMENTS.is_power_of_two());
    }

    /// Workgroup memory stays far below the 16 KiB every device grants.
    #[test]
    fn workgroup_memory_is_small() {
        let source = include_str!("pbc_pme.wgsl");
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let bytes: u32 = module
            .global_variables
            .iter()
            .filter(|(_, variable)| variable.space == naga::AddressSpace::WorkGroup)
            .map(|(_, variable)| layouter[variable.ty].size)
            .sum();
        assert!(bytes <= 8 * 1024, "{bytes} bytes of workgroup memory");
    }

    #[test]
    fn sizing_picks_powers_of_two_within_limits() {
        let sizing = PmeSizing::new(9_000, [40.0, 44.0, 52.0], 1.2).unwrap();
        assert_eq!(sizing.grid, [64, 64, 64]);
        let loose = PmeSizing::new(9_000, [40.0, 44.0, 52.0], 1.5).unwrap();
        assert_eq!(loose.grid, [32, 32, 64]);
        // An exact fit does not round up; a tiny box gets the minimum.
        assert_eq!(
            PmeSizing::new(10, [32.0, 12.0, 256.0], 1.0).unwrap().grid,
            [32, 16, 256]
        );
        assert!(matches!(
            PmeSizing::new(10, [300.0, 30.0, 30.0], 1.0),
            Err(Error::Capacity)
        ));
        assert!(matches!(
            PmeSizing::with_grid(10, [48, 64, 64]),
            Err(Error::Input(_))
        ));
        assert!(matches!(
            PmeSizing::with_grid(10, [512, 64, 64]),
            Err(Error::Capacity)
        ));
        assert_eq!(
            sizing.allocation_bytes(),
            48 + 4 * 64 * 64 * 64 + 8 * 33 * 64 * 64 + 4 * 64 * 64 * 64 + 8 * 256 + 4 * 192
        );
    }

    #[test]
    fn dispatch_shapes_cover_the_mesh() {
        let sizing = PmeSizing::with_grid(9_001, [32, 64, 128]).unwrap();
        assert_eq!(sizing.groups(PmeKernel::Spread), (563, 1));
        assert_eq!(sizing.groups(PmeKernel::Gather), (563, 1));
        // Every transform covers each line of its mesh exactly once.
        for grid in [[32, 64, 128], [16, 16, 16], [256, 16, 256], [64, 64, 64]] {
            let sizing = PmeSizing::with_grid(100, grid).unwrap();
            let [kx, ky, kz] = grid;
            let per_group = |points: u32, available: u32| (GROUP_ELEMENTS / points).min(available);
            let (x, y) = sizing.groups(PmeKernel::FftXForward);
            assert_eq!(sizing.groups(PmeKernel::FftXInverse), (x, y));
            assert_eq!(x * per_group(kx, kz / 2) * 2, kz, "{grid:?}");
            assert_eq!(y, ky);
            let (x, y) = sizing.groups(PmeKernel::FftYForward);
            assert_eq!(sizing.groups(PmeKernel::FftYInverse), (x, y));
            assert_eq!((x * per_group(ky, kz), y), (kz, kx / 2 + 1), "{grid:?}");
            let (x, y) = sizing.groups(PmeKernel::Convolve);
            assert_eq!(sizing.groups(PmeKernel::ConvolveEnergy), (x, y));
            assert_eq!((x * per_group(kz, ky), y), (ky, kx / 2 + 1), "{grid:?}");
        }
        let wide = PmeSizing::with_grid(3_000_000, [64, 64, 64]).unwrap();
        assert_eq!(wide.groups(PmeKernel::Spread), (32_768, 6));
        assert_eq!(PmeMesh::chain(false).len(), 7);
        assert_eq!(PmeMesh::chain(true)[3], PmeKernel::ConvolveEnergy);
    }

    /// The moduli are those of the CPU engine: at the origin the spline sums
    /// to one, and at the Nyquist index its transform is 1/3.
    #[test]
    fn tables_match_their_closed_forms() {
        let moduli = inverse_moduli(64);
        assert!((moduli[0] - 1.0).abs() < 1e-6);
        assert!((moduli[32] - 9.0).abs() < 1e-5);
        assert!((moduli[5] - moduli[59]).abs() < 1e-6);
        let table = twiddles();
        assert_eq!(table[0], [1.0, 0.0]);
        assert!((table[64][0]).abs() < 1e-7 && (table[64][1] + 1.0).abs() < 1e-7);
        assert!((table[128][0] + 1.0).abs() < 1e-7);
    }
}
