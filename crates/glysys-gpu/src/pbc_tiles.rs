//! Host side of the tiled explicit nonbonded engine (`pbc_tiles.wgsl`).
//!
//! The engine replaces the neighbor maintenance, pair, and bonded passes of
//! [`crate::pbc::ResidentPbc`] while sharing its coordinate, velocity,
//! metadata, and output buffers, so the validated SETTLE/RATTLE integrator
//! kernels run unchanged on top of it. See the shader for the algorithm.
use crate::device::Error;
use crate::pbc::PbcPacking;
use std::collections::HashSet;

/// Coulomb constant in kcal Å / (mol e²), identical to the shader constant.
const COULOMB: f64 = 332.063713299;
/// Target edge of the Hilbert sort cells. About three atoms per cell in water
/// keeps 32 consecutive atoms spatially compact.
const SORT_CELL_ANGSTROM: f64 = 3.0;
/// At most 128 sort cells per axis (2^21 buckets).
const MAX_SORT_BITS: u32 = 7;
/// Atoms diffuse far less than the skin between rebuilds, so the spatial order
/// is refreshed on a fixed cadence of resident steps (about 0.1-0.25 A of
/// water diffusion); block boxes and tiles are rebuilt whenever the skin is
/// crossed. The cadence only affects performance, never the physics.
pub(crate) const RESORT_STEPS: u32 = 64;
pub(crate) const STAGES: u64 = 4;
const BLOCK: u32 = 32;
const WIDE: u32 = 32_768;

/// Entry points with the global bindings each one statically uses. Automatic
/// pipeline layouts are derived from the shader, so a bind group must supply
/// exactly these bindings (checked against naga in the unit tests).
pub(crate) const KERNELS: [(&str, &[u32]); 16] = [
    ("check_rebuild", &[0, 1, 2, 5]),
    ("prepare_args", &[0, 2, 11]),
    ("bucket_clear", &[0, 2, 5]),
    ("bucket_count", &[0, 1, 5]),
    ("scan_local", &[0, 5]),
    ("scan_partials", &[0, 5]),
    ("scan_add", &[0, 5]),
    ("bucket_scatter", &[0, 5]),
    ("bucket_order", &[0, 5]),
    ("block_bounds", &[0, 1, 5, 6]),
    ("tile_build", &[0, 3, 5, 6]),
    ("far_exclusions", &[0, 3, 5, 6]),
    ("finish_rebuild", &[0, 2, 5]),
    ("nb_tiles", &[0, 1, 2, 3, 6, 7, 12, 13]),
    ("bonded_terms", &[0, 1, 2, 3, 7, 9, 10]),
    ("finalize_forces", &[0, 2, 7, 8, 12]),
];

/// Pipelines in creation order. Variants of one entry point differ only in
/// their override constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TileKernel {
    CheckRebuild,
    PrepareArgs,
    BucketClear,
    BucketCount,
    ScanLocal,
    ScanPartials,
    ScanAdd,
    BucketScatter,
    BucketOrder,
    BlockBounds,
    TileBuild,
    FarExclusions,
    FinishRebuild,
    PairForces,
    PairEnergy,
    BondedDynamicsForces,
    BondedDynamicsEnergy,
    BondedAllForces,
    BondedAllEnergy,
    Finalize,
}

impl TileKernel {
    const ALL: [TileKernel; 20] = [
        Self::CheckRebuild,
        Self::PrepareArgs,
        Self::BucketClear,
        Self::BucketCount,
        Self::ScanLocal,
        Self::ScanPartials,
        Self::ScanAdd,
        Self::BucketScatter,
        Self::BucketOrder,
        Self::BlockBounds,
        Self::TileBuild,
        Self::FarExclusions,
        Self::FinishRebuild,
        Self::PairForces,
        Self::PairEnergy,
        Self::BondedDynamicsForces,
        Self::BondedDynamicsEnergy,
        Self::BondedAllForces,
        Self::BondedAllEnergy,
        Self::Finalize,
    ];

    fn entry(self) -> &'static str {
        match self {
            Self::CheckRebuild => "check_rebuild",
            Self::PrepareArgs => "prepare_args",
            Self::BucketClear => "bucket_clear",
            Self::BucketCount => "bucket_count",
            Self::ScanLocal => "scan_local",
            Self::ScanPartials => "scan_partials",
            Self::ScanAdd => "scan_add",
            Self::BucketScatter => "bucket_scatter",
            Self::BucketOrder => "bucket_order",
            Self::BlockBounds => "block_bounds",
            Self::TileBuild => "tile_build",
            Self::FarExclusions => "far_exclusions",
            Self::FinishRebuild => "finish_rebuild",
            Self::PairForces | Self::PairEnergy => "nb_tiles",
            Self::BondedDynamicsForces
            | Self::BondedDynamicsEnergy
            | Self::BondedAllForces
            | Self::BondedAllEnergy => "bonded_terms",
            Self::Finalize => "finalize_forces",
        }
    }

    fn constants(self) -> &'static [(&'static str, f64)] {
        match self {
            Self::PairEnergy => &[("COMPUTE_ENERGY", 1.0)],
            Self::BondedDynamicsForces => &[("INCLUDE_CONSTRAINED", 0.0)],
            Self::BondedDynamicsEnergy => &[("COMPUTE_ENERGY", 1.0), ("INCLUDE_CONSTRAINED", 0.0)],
            Self::BondedAllEnergy => &[("COMPUTE_ENERGY", 1.0)],
            _ => &[],
        }
    }

    /// Indirect argument slot for list-rebuild stages, written on the device
    /// from the rebuild flag so an unchanged list costs no host round trip.
    pub(crate) fn rebuild_stage(self) -> Option<u64> {
        Some(match self {
            Self::BlockBounds => 0,
            Self::TileBuild => 1,
            Self::FarExclusions => 2,
            Self::FinishRebuild => 3,
            _ => return None,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn stage_name(self) -> &'static str {
        match self {
            Self::CheckRebuild | Self::PrepareArgs => "neighborCheck",
            Self::BucketClear
            | Self::BucketCount
            | Self::ScanLocal
            | Self::ScanPartials
            | Self::ScanAdd
            | Self::BucketScatter
            | Self::BucketOrder => "neighborSort",
            Self::BlockBounds => "neighborBounds",
            Self::TileBuild | Self::FarExclusions => "neighborTiles",
            Self::FinishRebuild => "neighborFinish",
            Self::PairForces | Self::PairEnergy => "nonbondedForces",
            Self::BondedDynamicsForces
            | Self::BondedDynamicsEnergy
            | Self::BondedAllForces
            | Self::BondedAllEnergy => "bondedForces",
            Self::Finalize => "forceFinalize",
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|kernel| *kernel == self).unwrap()
    }

    /// Stable job code used by the resident dispatcher.
    pub(crate) fn job(self) -> usize {
        self.index()
    }

    pub(crate) fn from_job(job: usize) -> Self {
        Self::ALL[job]
    }

    /// Displacement check, dispatch preparation and the rebuild stages.
    pub(crate) const REBUILD: [TileKernel; 6] = [
        Self::CheckRebuild,
        Self::PrepareArgs,
        Self::BlockBounds,
        Self::TileBuild,
        Self::FarExclusions,
        Self::FinishRebuild,
    ];

    /// Spatial resort, dispatched directly ahead of [`Self::REBUILD`].
    pub(crate) const RESORT: [TileKernel; 7] = [
        Self::BucketClear,
        Self::BucketCount,
        Self::ScanLocal,
        Self::ScanPartials,
        Self::ScanAdd,
        Self::BucketScatter,
        Self::BucketOrder,
    ];
}

/// Topology data for the tiled engine, built once from the prepared system.
pub(crate) struct TilePacking {
    /// Per-atom LJ/charge/molecule words, specials ranges, then exceptions.
    pub atoms: Vec<[f32; 4]>,
    /// Bonds (1 vec4), angles (2) and torsions (2), constrained terms last.
    pub terms: Vec<[f32; 4]>,
    pub exceptions: u32,
    pub bonds: u32,
    pub free_bonds: u32,
    pub angles: u32,
    pub free_angles: u32,
    pub torsions: u32,
    pub angle_offset: u32,
    pub torsion_offset: u32,
    pub exception_offset: u32,
    /// Directed exclusion pairs more than 32 indices apart.
    pub far_exclusions: u32,
    pub far_offset: u32,
}

impl TilePacking {
    /// Derive the engine topology from the shared PBC packing, so the tiled
    /// and historical kernels see bit-identical parameters and constraints.
    pub fn new(packing: &PbcPacking) -> Result<Self, Error> {
        let n = packing.params.len();
        if n >= 1 << 24 {
            return Err(Error::Capacity);
        }
        let mut molecule = vec![0u32; n];
        for group in &packing.molecules {
            let anchor = u32::try_from(group[0]).map_err(|_| Error::Capacity)?;
            for &atom in group {
                molecule[atom] = anchor;
            }
        }
        let mut atoms = Vec::with_capacity(2 * n);
        for (index, p) in packing.params.iter().enumerate() {
            atoms.push([
                p[1],
                (f64::from(p[2]).max(0.0)).sqrt() as f32,
                p[0],
                f32::from_bits(molecule[index]),
            ]);
        }
        // Specials range and a bitmask of the partners within 32 indices of
        // the atom (bit k is atom index - 32 + k). Topological exclusions are
        // almost always that local; the remaining directed pairs are listed
        // separately for the far-exclusion pass.
        let mut far_pairs = Vec::new();
        for (atom, range) in packing.ranges.iter().enumerate() {
            let partners = &packing.specials[range[0] as usize..range[1] as usize];
            let mut window = 0u64;
            for special in partners {
                let offset = i64::from(special.other) - atom as i64 + 32;
                if (0..64).contains(&offset) {
                    window |= 1 << offset;
                } else {
                    far_pairs.push([atom as u32, special.other]);
                }
            }
            atoms.push([
                f32::from_bits(range[0]),
                f32::from_bits(range[1]),
                f32::from_bits(window as u32),
                f32::from_bits((window >> 32) as u32),
            ]);
        }
        // Exclusions and 1-4 pairs are always intramolecular; the shader
        // uses the molecule test to skip the specials search otherwise.
        let exception_offset = u32::try_from(atoms.len()).map_err(|_| Error::Capacity)?;
        let mut exceptions = 0u32;
        for (a, range) in packing.ranges.iter().enumerate() {
            for special in &packing.specials[range[0] as usize..range[1] as usize] {
                let b = special.other as usize;
                if b >= n || molecule[a] != molecule[b] {
                    return Err(Error::Input("exclusion between separate molecules"));
                }
                if special.spare == 0 || b <= a {
                    continue;
                }
                let (pa, pb) = (packing.params[a], packing.params[b]);
                let eps =
                    (f64::from(pa[2]) * f64::from(pb[2])).max(0.0).sqrt() / f64::from(special.scnb);
                atoms.push([
                    f32::from_bits(a as u32),
                    f32::from_bits(b as u32),
                    (COULOMB * f64::from(pa[0]) * f64::from(pb[0]) / f64::from(special.scee))
                        as f32,
                    eps as f32,
                ]);
                atoms.push([pa[1] + pb[1], 0.0, 0.0, 0.0]);
                exceptions += 1;
            }
        }
        let far_offset = u32::try_from(atoms.len()).map_err(|_| Error::Capacity)?;
        let far_exclusions = u32::try_from(far_pairs.len()).map_err(|_| Error::Capacity)?;
        for [atom, other] in far_pairs {
            atoms.push([f32::from_bits(atom), f32::from_bits(other), 0.0, 0.0]);
        }

        // Constraint sets enforced by the resident SETTLE/SHAKE kernels.
        let [_, angle_start, torsion_start, water_start, constraint_end] = packing.bonded_offsets;
        let water_count = packing.bonded_counts[3] as usize;
        let mut water_of = vec![usize::MAX; n];
        let mut constrained = HashSet::new();
        let pair = |a: usize, b: usize| (a.min(b), a.max(b));
        for w in 0..water_count {
            let head = packing.bonded[water_start + 2 * w];
            let [o, h1, h2] = [head[0] as usize, head[1] as usize, head[2] as usize];
            for atom in [o, h1, h2] {
                water_of[atom] = w;
            }
            constrained.extend([pair(o, h1), pair(o, h2), pair(h1, h2)]);
        }
        for t in &packing.bonded[water_start + 2 * water_count..constraint_end] {
            constrained.insert(pair(t[0] as usize, t[1] as usize));
        }
        let mut terms = Vec::new();
        let mut rigid = Vec::new();
        for t in &packing.bonded[..angle_start] {
            if constrained.contains(&pair(t[0] as usize, t[1] as usize)) {
                rigid.push(*t);
            } else {
                terms.push(*t);
            }
        }
        let free_bonds = terms.len() as u32;
        terms.append(&mut rigid);
        let bonds = terms.len() as u32;
        let angle_offset = terms.len() as u32;
        let mut free = Vec::new();
        let mut rigid = Vec::new();
        for k in (angle_start..torsion_start).step_by(2) {
            let head = packing.bonded[k];
            let [a, c, b] = [head[0] as usize, head[1] as usize, head[2] as usize];
            let in_one_water = water_of[a] != usize::MAX
                && water_of[a] == water_of[b]
                && water_of[a] == water_of[c];
            let packed = [head, packing.bonded[k + 1]];
            if in_one_water {
                rigid.push(packed);
            } else {
                free.push(packed);
            }
        }
        let free_angles = free.len() as u32;
        let angles = (free.len() + rigid.len()) as u32;
        terms.extend(free.into_iter().chain(rigid).flatten());
        let torsion_offset = terms.len() as u32;
        terms.extend_from_slice(&packing.bonded[torsion_start..water_start]);
        let torsions = ((water_start - torsion_start) / 2) as u32;
        Ok(Self {
            atoms,
            terms,
            exceptions,
            bonds,
            free_bonds,
            angles,
            free_angles,
            torsions,
            angle_offset,
            torsion_offset,
            exception_offset,
            far_exclusions,
            far_offset,
        })
    }
}

/// Word offsets inside the `work` buffer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WorkLayout {
    pub order: u32,
    pub bucket_of: u32,
    pub counts: u32,
    pub starts: u32,
    pub partials: u32,
    pub block_tiles: u32,
    pub tile_atoms: u32,
    pub tile_masks: u32,
    pub reference: u32,
    pub counters: u32,
    pub words: u64,
}

impl WorkLayout {
    fn new(n: u32, blocks: u32, buckets: u32, scan_groups: u32, capacity: u32) -> Option<Self> {
        let mut cursor = 0u64;
        let mut take = |words: u64| -> Option<u32> {
            let at = u32::try_from(cursor).ok()?;
            cursor = cursor.checked_add(words)?;
            Some(at)
        };
        let order = take(u64::from(blocks) * u64::from(BLOCK))?;
        let bucket_of = take(u64::from(n))?;
        let counts = take(u64::from(buckets))?;
        let starts = take(u64::from(buckets) + 1)?;
        let partials = take(u64::from(scan_groups) + 1)?;
        let block_tiles = take(2 * u64::from(blocks))?;
        let tile_atoms = take(u64::from(capacity) * u64::from(BLOCK))?;
        let tile_masks = take(u64::from(capacity) * u64::from(BLOCK))?;
        let reference = take(3 * u64::from(n))?;
        let counters = take(4)?;
        u32::try_from(cursor).ok()?;
        Some(Self {
            order,
            bucket_of,
            counts,
            starts,
            partials,
            block_tiles,
            tile_atoms,
            tile_masks,
            reference,
            counters,
            words: cursor,
        })
    }
}

/// Static sizing of the engine for one prepared system and box.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TileSizing {
    pub n: u32,
    pub blocks: u32,
    pub bits: u32,
    pub buckets: u32,
    pub scan_groups: u32,
    /// Tiles reserved per block (the diagonal tile included).
    pub stride: u32,
    pub capacity: u32,
    pub layout: WorkLayout,
}

impl TileSizing {
    pub fn new(n: u32, box_xyz: [f64; 3], list_radius: f64) -> Result<Self, Error> {
        let blocks = n.div_ceil(BLOCK).max(1);
        let longest = box_xyz.iter().copied().fold(0.0f64, f64::max);
        let bits =
            ((longest / SORT_CELL_ANGSTROM).max(2.0).log2().round() as u32).clamp(1, MAX_SORT_BITS);
        let buckets = 1u32 << (3 * bits);
        let scan_groups = buckets.div_ceil(1024);
        // Tiles per block: the diagonal tile plus the atoms of other blocks
        // inside the list radius of a block box. Bound the box edge at 1.5x
        // a compact 32-atom cube and count the Steiner volume of that box
        // grown by the radius, with headroom for density fluctuations.
        let volume = box_xyz.iter().product::<f64>();
        let density = (f64::from(n) / volume).max(1e-6);
        let edge = 1.5 * (f64::from(BLOCK) / density).cbrt();
        let r = list_radius;
        let steiner = edge.powi(3)
            + 6.0 * edge * edge * r
            + 3.0 * std::f64::consts::PI * edge * r * r
            + 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        let atoms_in_reach = (density * steiner).min(f64::from(n));
        let per_block = 3.0 + 1.5 * (atoms_in_reach + 64.0) / f64::from(BLOCK);
        let stride = per_block.ceil().min(f64::from(blocks) + 1.0).max(2.0) as u32;
        let capacity = u32::try_from(u64::from(stride) * u64::from(blocks))
            .ok()
            .filter(|tiles| *tiles <= u32::MAX / (2 * BLOCK))
            .ok_or(Error::Capacity)?;
        let layout =
            WorkLayout::new(n, blocks, buckets, scan_groups, capacity).ok_or(Error::Capacity)?;
        Ok(Self {
            n,
            blocks,
            bits,
            buckets,
            scan_groups,
            stride,
            capacity,
            layout,
        })
    }

    /// Sort-grid uniform words: cells per axis, bits.
    pub fn grid(&self) -> [u32; 2] {
        [1 << self.bits, self.bits]
    }

    pub fn work_bytes(&self) -> u64 {
        self.layout.words * 4
    }

    pub fn accumulator_words(&self) -> u64 {
        6 * u64::from(self.n) + 2 * 9
    }
}

/// Wide dispatch shape for kernels that index workgroups with
/// `group.x + group.y * 32768`.
pub(crate) fn wide_groups(groups: u32) -> (u32, u32) {
    if groups <= WIDE {
        (groups, 1)
    } else {
        (WIDE, groups.div_ceil(WIDE))
    }
}

pub(crate) struct TileEngine {
    pub sizing: TileSizing,
    pub packing_counts: [u32; 6],
    uniform: wgpu::Buffer,
    pub work: wgpu::Buffer,
    pub args: wgpu::Buffer,
    _atoms: wgpu::Buffer,
    _blocks: wgpu::Buffer,
    _acc: wgpu::Buffer,
    _terms: wgpu::Buffer,
    _pair_grad: wgpu::Buffer,
    pipelines: Vec<wgpu::ComputePipeline>,
    bind_groups: Vec<wgpu::BindGroup>,
    cutoff: f32,
    krf: f32,
    crf: f32,
    list_radius: f32,
    status_word: u32,
    rebuild_word: u32,
    offsets: [u32; 4],
    far: [u32; 2],
    resort_pending: std::sync::atomic::AtomicBool,
    steps_since_resort: std::sync::atomic::AtomicU32,
}

/// Legacy `ResidentPbc` buffers shared with the engine.
pub(crate) struct SharedBuffers<'a> {
    pub sys: &'a wgpu::Buffer,
    pub aux: &'a wgpu::Buffer,
    pub specials: &'a wgpu::Buffer,
    pub out: &'a wgpu::Buffer,
    pub bcoords: &'a wgpu::Buffer,
}

impl TileEngine {
    /// Bytes allocated by [`Self::new`] for a sizing and packing.
    pub fn allocation_bytes(sizing: &TileSizing, packing: &TilePacking) -> u64 {
        sizing.work_bytes()
            + 32 * u64::from(sizing.blocks)
            + 512 * u64::from(sizing.blocks)
            + 16 * u64::from(sizing.n.div_ceil(4))
            + 4 * sizing.accumulator_words()
            + 16 * u64::from(sizing.n)
            + 16 * packing.atoms.len() as u64
            + 16 * packing.terms.len().max(1) as u64
            + 12 * STAGES
            + 256
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sizing: TileSizing,
        packing: &TilePacking,
        shared: SharedBuffers<'_>,
        electro: [f32; 4],
        list_radius: f32,
        status_word: u32,
        rebuild_word: u32,
    ) -> Self {
        use wgpu::util::DeviceExt;
        let storage = wgpu::BufferUsages::STORAGE;
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbc tiles config"),
            size: 160,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let work = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbc tiles work"),
            size: sizing.work_bytes(),
            usage: storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let blocks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbc tiles blocks"),
            size: 32 * u64::from(sizing.blocks)
                + 16 * 32 * u64::from(sizing.blocks)
                + 4 * u64::from(sizing.n.div_ceil(4)) * 4,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let acc = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbc tiles accumulators"),
            size: 4 * sizing.accumulator_words(),
            usage: storage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pair_grad = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbc tiles pair gradients"),
            size: 16 * u64::from(sizing.n),
            usage: storage,
            mapped_at_creation: false,
        });
        let atoms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbc tiles atoms"),
            contents: bytemuck::cast_slice(&packing.atoms),
            usage: storage,
        });
        let terms_data: Vec<[f32; 4]> = if packing.terms.is_empty() {
            vec![[0.0; 4]]
        } else {
            packing.terms.clone()
        };
        let terms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbc tiles terms"),
            contents: bytemuck::cast_slice(&terms_data),
            usage: storage,
        });
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbc tiles dispatch"),
            size: 12 * STAGES,
            usage: storage | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        // Unused order slots in the final partial block stay invalid; every
        // resort rewrites only the first `n` slots.
        let order = vec![u32::MAX; (sizing.blocks * BLOCK) as usize];
        queue.write_buffer(
            &work,
            u64::from(sizing.layout.order) * 4,
            bytemuck::cast_slice(&order),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GlySys PBC tiles"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pbc_tiles.wgsl").into()),
        });
        let resource = |binding: u32| -> wgpu::BindingResource<'_> {
            match binding {
                0 => uniform.as_entire_binding(),
                1 => shared.sys.as_entire_binding(),
                2 => shared.aux.as_entire_binding(),
                3 => atoms.as_entire_binding(),
                4 => shared.specials.as_entire_binding(),
                5 => work.as_entire_binding(),
                6 => blocks.as_entire_binding(),
                7 => acc.as_entire_binding(),
                8 => shared.out.as_entire_binding(),
                9 => shared.bcoords.as_entire_binding(),
                10 => terms.as_entire_binding(),
                11 => args.as_entire_binding(),
                12 => pair_grad.as_entire_binding(),
                13 => work.as_entire_binding(),
                _ => unreachable!("pbc_tiles.wgsl declares bindings 0..=13"),
            }
        };
        let mut pipelines = Vec::with_capacity(TileKernel::ALL.len());
        let mut bind_groups = Vec::with_capacity(TileKernel::ALL.len());
        for kernel in TileKernel::ALL {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(kernel.entry()),
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
                .expect("every tile kernel has a binding list");
            let entries: Vec<_> = bindings
                .iter()
                .map(|&binding| wgpu::BindGroupEntry {
                    binding,
                    resource: resource(binding),
                })
                .collect();
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(kernel.entry()),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            pipelines.push(pipeline);
            bind_groups.push(bind_group);
        }
        Self {
            sizing,
            packing_counts: [
                packing.exceptions,
                packing.bonds,
                packing.angles,
                packing.torsions,
                packing.free_bonds,
                packing.free_angles,
            ],
            uniform,
            work,
            args,
            _atoms: atoms,
            _blocks: blocks,
            _acc: acc,
            _terms: terms,
            _pair_grad: pair_grad,
            pipelines,
            bind_groups,
            cutoff: electro[1],
            krf: electro[2],
            crf: electro[3],
            list_radius,
            status_word,
            rebuild_word,
            offsets: [
                0,
                packing.angle_offset,
                packing.torsion_offset,
                packing.exception_offset,
            ],
            far: [packing.far_exclusions, packing.far_offset],
            resort_pending: std::sync::atomic::AtomicBool::new(true),
            steps_since_resort: std::sync::atomic::AtomicU32::new(0),
        }
    }

    pub fn kernel(&self, kernel: TileKernel) -> (&wgpu::ComputePipeline, &wgpu::BindGroup) {
        let index = kernel.index();
        (&self.pipelines[index], &self.bind_groups[index])
    }

    /// Workgroup count for a direct (non-rebuild) dispatch.
    pub fn groups(&self, kernel: TileKernel) -> u32 {
        match kernel {
            TileKernel::CheckRebuild | TileKernel::Finalize => self.sizing.n.div_ceil(128).max(1),
            TileKernel::PrepareArgs => 1,
            TileKernel::PairForces | TileKernel::PairEnergy => self.sizing.blocks,
            TileKernel::BondedDynamicsForces | TileKernel::BondedDynamicsEnergy => {
                self.terms(false).div_ceil(64).max(1)
            }
            TileKernel::BondedAllForces | TileKernel::BondedAllEnergy => {
                self.terms(true).div_ceil(64).max(1)
            }
            TileKernel::BucketClear | TileKernel::BucketOrder => {
                self.sizing.buckets.div_ceil(128).max(1)
            }
            TileKernel::BucketCount | TileKernel::BucketScatter => {
                self.sizing.n.div_ceil(128).max(1)
            }
            TileKernel::ScanLocal | TileKernel::ScanAdd => self.sizing.scan_groups.max(1),
            TileKernel::ScanPartials => 1,
            _ => unreachable!("rebuild stages use indirect dispatch"),
        }
    }

    fn terms(&self, include_constrained: bool) -> u32 {
        let [exceptions, bonds, angles, torsions, free_bonds, free_angles] = self.packing_counts;
        if include_constrained {
            exceptions + bonds + angles + torsions
        } else {
            exceptions + free_bonds + free_angles + torsions
        }
    }

    /// Whether the next encoded step must resort atoms: after a full
    /// coordinate upload and then every [`RESORT_STEPS`] steps.
    pub fn take_resort(&self) -> bool {
        use std::sync::atomic::Ordering;
        let due = self.steps_since_resort.fetch_add(1, Ordering::Relaxed) + 1 >= RESORT_STEPS;
        let pending = self.resort_pending.swap(false, Ordering::Relaxed);
        if due || pending {
            self.steps_since_resort.store(0, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Rewrite the engine uniform for the current box; `resort` schedules a
    /// spatial resort before the next evaluation.
    pub fn upload(&self, queue: &wgpu::Queue, box_xyz: [f32; 3], resort: bool) {
        let s = &self.sizing;
        let l = &s.layout;
        let grid = s.grid();
        let half_skin = 0.5 * (self.list_radius - self.cutoff);
        let [exceptions, bonds, angles, torsions, free_bonds, free_angles] = self.packing_counts;
        let words: [u32; 40] = [
            s.n,
            s.blocks,
            s.stride,
            self.far[1],
            box_xyz[0].to_bits(),
            box_xyz[1].to_bits(),
            box_xyz[2].to_bits(),
            self.cutoff.to_bits(),
            self.krf.to_bits(),
            self.crf.to_bits(),
            self.list_radius.to_bits(),
            half_skin.to_bits(),
            grid[0],
            grid[1],
            0,
            s.buckets,
            l.order,
            l.bucket_of,
            l.counts,
            l.starts,
            l.partials,
            l.block_tiles,
            l.tile_atoms,
            l.tile_masks,
            l.reference,
            l.counters,
            self.status_word,
            self.rebuild_word,
            exceptions,
            bonds,
            angles,
            torsions,
            free_bonds,
            free_angles,
            s.scan_groups,
            self.far[0],
            self.offsets[0],
            self.offsets[1],
            self.offsets[2],
            self.offsets[3],
        ];
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&words));
        if resort {
            self.resort_pending
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates_and_binding_table_matches_static_use() {
        let source = include_str!("pbc_tiles.wgsl");
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
    }

    #[test]
    fn sizing_fits_dense_water_and_tiny_boxes() {
        let water = TileSizing::new(12_132, [49.4, 49.4, 49.4], 10.5).unwrap();
        assert_eq!(water.blocks, 380);
        assert!(water.stride >= 80, "stride {}", water.stride);
        assert_eq!(water.capacity, water.stride * water.blocks);
        assert!(water.buckets >= 16 * 16 * 16);
        let tiny = TileSizing::new(5, [20.0, 20.0, 20.0], 6.0).unwrap();
        assert_eq!(tiny.blocks, 1);
        assert!(tiny.capacity >= 2);
    }

    #[test]
    fn wide_groups_cover_large_dispatches() {
        assert_eq!(wide_groups(10), (10, 1));
        assert_eq!(wide_groups(WIDE + 1), (WIDE, 2));
    }
}
