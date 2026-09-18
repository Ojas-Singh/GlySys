//! Resident periodic-boundary nonbonded evaluator (CutoffPeriodic-style).
//!
//! Canonical GPU counterpart of `glysys-energy::pbc`: exact-fit cells,
//! Verlet pair enumeration within cutoff+skin, and Amber Lennard-Jones plus
//! reaction-field electrostatics evaluated strictly inside the cutoff with
//! the minimum-image convention. The WGSL (`pbc.wgsl`) is shared verbatim
//! between the browser laboratory engine and the native validation harness.
//!
//! Binding budget: one uniform plus five storage bindings, within the
//! WebGPU-guaranteed eight per shader stage. Per-atom parameters and
//! coordinates share one interleaved region; cell heads, linked-list links,
//! specials ranges, and the pair counter share one u32 region; per-thread
//! partials, gradients, and totals share one float region.
//!
//! PME seam: electrostatics arrive as [`NonbondedElectrostatics`]; only the
//! reaction-field variant dispatches today. A PME backend adds a reciprocal
//! kernel and uniform variant while reusing these lists, traversal,
//! packing, and (later) integration — never a second pair-list
//! implementation.
use crate::context::{AllocationReservation, GpuContext};
use crate::device::{Error, Special};
use glysys::{ParameterizedSystem, Vec3};
use glysys_energy::pbc::{
    BoxVectors, NonbondedElectrostatics, classify_waters, molecules, water_equilibrium,
};
use std::collections::BTreeMap;
use wgpu::util::DeviceExt;

/// Minimal PBC packing: per-atom parameters plus exclusion/1-4 specials.
/// Mirrors `topology::PreparedTopology` conventions (scee == 0 skips).
pub struct PbcPacking {
    pub params: Vec<[f32; 4]>,
    pub ranges: Vec<[u32; 2]>,
    pub specials: Vec<Special>,
    pub bonded: Vec<[f32; 4]>,
    pub bonded_offsets: [usize; 5],
    pub bonded_counts: [u32; 4],
    pub adjacency_offset: u32,
    pub molecules: Vec<Vec<usize>>,
    pub bonded_coords: Vec<[f32; 4]>,
    pub solute_constraint_count: u32,
    pub dims: [u32; 4],
    pub limit: f64,
    pub cutoff: f64,
}

impl PbcPacking {
    pub fn new(system: &ParameterizedSystem, cutoff: f64, skin: f64) -> Result<Self, Error> {
        Self::new_with_box(system, system.box_angstrom(), cutoff, skin)
    }

    /// Build the same topology/parameter packing for an explicitly supplied
    /// orthorhombic box.  NPT changes the active cell grid as the box expands
    /// or contracts; keeping the box as an input lets the resident execution
    /// layer rebuild that derived grid without mutating the prepared
    /// chemistry or counting the proposal as an MC rejection.
    pub fn new_with_box(
        system: &ParameterizedSystem,
        box_xyz: [f64; 3],
        cutoff: f64,
        skin: f64,
    ) -> Result<Self, Error> {
        let n = system.atom_count();
        if n == 0 || n > u32::MAX as usize {
            return Err(Error::Input("atom count"));
        }
        if !cutoff.is_finite() || cutoff <= 0. || !skin.is_finite() || skin < 0. {
            return Err(Error::Input("cutoff/skin"));
        }
        let b = box_xyz;
        if !b.iter().all(|v| v.is_finite() && *v > 0.) {
            return Err(Error::Input("periodic box"));
        }
        if cutoff >= 0.5 * b[0].min(b[1]).min(b[2]) {
            return Err(Error::Input("cutoff below half box edge"));
        }
        // Exact-fit cells, mirroring PbcNeighborList::build (the
        // neighbor-list exact-match test guards this duplication).
        let limit = cutoff + skin;
        let nx = ((b[0] / limit).floor() as u32).max(1);
        let ny = ((b[1] / limit).floor() as u32).max(1);
        let nz = ((b[2] / limit).floor() as u32).max(1);
        let params = system
            .atoms()
            .iter()
            .map(|a| {
                [
                    a.charge() as f32,
                    a.lennard_jones_radius() as f32,
                    a.lennard_jones_epsilon() as f32,
                    a.mass() as f32,
                ]
            })
            .collect();
        // (scee, scnb, is_exception): exclusions carry (0, 0, false) and are
        // skipped; 1-4 pairs carry Amber scales plus spare = 1 so the shader
        // evaluates them with plain Coulomb (OpenMM exception convention).
        // 1-4 entries overwrite exclusions with `insert`: every 1-4 pair is
        // also exclusion-listed, and `or_insert` would silently drop the
        // scaled interaction.
        let mut exceptions: Vec<BTreeMap<usize, (f32, f32, bool)>> = system
            .exclusions()
            .iter()
            .map(|set| set.iter().map(|i| (*i, (0., 0., false))).collect())
            .collect();
        for t in system.dihedrals().iter().filter(|t| !t.is_improper()) {
            let ids = t.atoms();
            let key = (ids[0].min(ids[3]), ids[0].max(ids[3]));
            // Both directions: each endpoint's thread evaluates its half of
            // the pair, so a one-sided entry would compute one half scaled
            // and the other half as a regular pair.
            for (first, second) in [(key.0, key.1), (key.1, key.0)] {
                exceptions[first].insert(
                    second,
                    (
                        t.electrostatic_14_scale() as f32,
                        t.lennard_jones_14_scale() as f32,
                        true,
                    ),
                );
            }
        }
        let mut ranges = Vec::with_capacity(n);
        let mut specials = Vec::new();
        for map in exceptions {
            let start = u32::try_from(specials.len()).map_err(|_| Error::Capacity)?;
            for (other, (scee, scnb, is_exception)) in map {
                specials.push(Special {
                    other: other as u32,
                    scee,
                    scnb,
                    spare: u32::from(is_exception),
                });
            }
            let end = u32::try_from(specials.len()).map_err(|_| Error::Capacity)?;
            ranges.push([start, end]);
        }
        let mut bonded = Vec::new();
        let bonds = bonded.len();
        for bond in system.bonds() {
            let [a, b] = bond.atoms();
            bonded.push([
                a as f32,
                b as f32,
                bond.force() as f32,
                bond.length() as f32,
            ]);
        }
        let angles = bonded.len();
        for angle in system.angles() {
            let [a, c, b] = angle.atoms();
            bonded.push([a as f32, c as f32, b as f32, angle.force() as f32]);
            bonded.push([angle.radians() as f32, 0., 0., 0.]);
        }
        let torsions = bonded.len();
        for torsion in system.dihedrals() {
            let a = torsion.atoms();
            bonded.push([a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32]);
            bonded.push([
                torsion.force() as f32,
                torsion.periodicity() as f32,
                torsion.phase() as f32,
                f32::from(torsion.is_improper()),
            ]);
        }
        let restraints = bonded.len();
        let waters_at = bonded.len();
        let waters = classify_waters(system);
        for water in &waters {
            let (oh1, oh2, hh) = water_equilibrium(system, *water)
                .map_err(|_| Error::Input("unsupported rigid-water topology"))?;
            if (oh1 - oh2).abs() > 1e-8 {
                return Err(Error::Input(
                    "GPU SETTLE requires an isosceles water geometry",
                ));
            }
            let [o, h1, h2] = *water;
            if o >= system.atoms().len() || h1 >= system.atoms().len() || h2 >= system.atoms().len()
            {
                return Err(Error::Input("rigid-water atom index out of range"));
            }
            let mh1 = system.atoms()[h1].mass();
            let mh2 = system.atoms()[h2].mass();
            if (mh1 - mh2).abs() > 1e-8 {
                return Err(Error::Input("GPU SETTLE requires equal hydrogen masses"));
            }
            bonded.push([o as f32, h1 as f32, h2 as f32, oh1 as f32]);
            bonded.push([
                oh2 as f32,
                hh as f32,
                system.atoms()[o].mass() as f32,
                mh1 as f32,
            ]);
        }
        let solute_at = bonded.len();
        let in_water: std::collections::HashSet<usize> = waters.iter().flatten().copied().collect();
        for bond in system.bonds() {
            let [a, b] = bond.atoms();
            if !in_water.contains(&a)
                && !in_water.contains(&b)
                && (system.atoms()[a].element() == 1 || system.atoms()[b].element() == 1)
            {
                bonded.push([a as f32, b as f32, bond.length() as f32, 0.]);
            }
        }
        let ends = bonded.len();
        let water_count = ((solute_at - waters_at) / 2)
            .try_into()
            .map_err(|_| Error::Capacity)?;
        let solute_constraint_count = (ends - solute_at).try_into().map_err(|_| Error::Capacity)?;
        let molecules = molecules(system);
        let raw = system.coordinates();
        let mut bonded_coords = vec![[0f32; 4]; n];
        for group in &molecules {
            let mut center = [0f64; 3];
            for &atom in group {
                center[0] += raw[atom].x;
                center[1] += raw[atom].y;
                center[2] += raw[atom].z;
            }
            let count = group.len() as f64;
            let anchor = group[0];
            for &atom in group {
                bonded_coords[atom] = [
                    (raw[atom].x - center[0] / count) as f32,
                    (raw[atom].y - center[1] / count) as f32,
                    (raw[atom].z - center[2] / count) as f32,
                    f32::from_bits(anchor as u32),
                ];
            }
        }
        let bonded_offsets = [bonds, angles, torsions, restraints, ends];
        let bonded_counts = [
            (angles - bonds) as u32 / 1,
            (torsions - angles) as u32 / 2,
            (restraints - torsions) as u32 / 2,
            water_count,
        ];
        // Stable incident-term lists preserve each atom's original summation
        // order without scanning every bonded term for every atom.
        let adjacency_offset = u32::try_from(bonded.len()).map_err(|_| Error::Capacity)?;
        if adjacency_offset > (1 << 24) {
            return Err(Error::Capacity);
        }
        let mut incident: Vec<[Vec<u32>; 3]> = (0..n)
            .map(|_| std::array::from_fn(|_| Vec::new()))
            .collect();
        for (k, term) in system.bonds().iter().enumerate() {
            for i in term.atoms() {
                incident[i][0].push(k as u32);
            }
        }
        for (k, term) in system.angles().iter().enumerate() {
            for i in term.atoms() {
                incident[i][1].push(k as u32);
            }
        }
        for (k, term) in system.dihedrals().iter().enumerate() {
            for i in term.atoms() {
                incident[i][2].push(k as u32);
            }
        }
        let mut indices = Vec::new();
        for terms in incident {
            let mut ranges = [0u32; 4];
            for (kind, list) in terms.into_iter().enumerate() {
                ranges[kind] = u32::try_from(indices.len()).map_err(|_| Error::Capacity)?;
                indices.extend(list);
            }
            ranges[3] = u32::try_from(indices.len()).map_err(|_| Error::Capacity)?;
            bonded.push(ranges.map(f32::from_bits));
        }
        for chunk in indices.chunks(4) {
            let mut packed = [0.; 4];
            for (slot, &index) in packed.iter_mut().zip(chunk) {
                *slot = f32::from_bits(index);
            }
            bonded.push(packed);
        }
        Ok(Self {
            params,
            ranges,
            specials,
            bonded,
            bonded_offsets,
            bonded_counts,
            adjacency_offset,
            molecules,
            bonded_coords,
            solute_constraint_count,
            dims: [n as u32, nx, ny, nz],
            limit,
            cutoff,
        })
    }

    pub fn cell_count(&self) -> u32 {
        self.dims[1] * self.dims[2] * self.dims[3]
    }
}

/// Reaction-field uniform parameters (method tag 0) or an explicit
/// rejection for PME. Host-side f64 math, cast once, documented in the
/// validation report.
fn electrostatics_uniform(backend: &NonbondedElectrostatics) -> Result<[f32; 4], Error> {
    match backend {
        NonbondedElectrostatics::ReactionField {
            cutoff_angstrom,
            solvent_dielectric,
        } => {
            let (rc, e) = (*cutoff_angstrom, *solvent_dielectric);
            if !rc.is_finite() || rc <= 0. || !e.is_finite() || e < 1. {
                return Err(Error::Input("reaction-field parameters"));
            }
            let krf = (e - 1.) / (2. * e + 1.) / rc.powi(3);
            let crf = 3. * e / (2. * e + 1.) / rc;
            Ok([0., rc as f32, krf as f32, crf as f32])
        }
        NonbondedElectrostatics::Pme { .. } => Err(Error::Input("PME not yet implemented on GPU")),
    }
}

/// Wrap coordinates into the box and pack as f32 (matches the CPU evaluate
/// convention of wrapping internally before minimum-image traversal).
pub fn wrap_coords_f32(coords: &[Vec3], box_vec: &BoxVectors) -> Vec<[f32; 4]> {
    coords
        .iter()
        .map(|p| {
            let w = box_vec.wrap(*p);
            [w.x as f32, w.y as f32, w.z as f32, 0.]
        })
        .collect()
}

/// Pack resident coordinates without wrapping. Bonded terms require intact
/// molecules; cell assignment and minimum-image nonbonded displacement wrap
/// on the GPU.
pub fn pack_coords_f32(coords: &[Vec3]) -> Vec<[f32; 4]> {
    coords
        .iter()
        .map(|p| [p.x as f32, p.y as f32, p.z as f32, 0.])
        .collect()
}

pub struct NeighborResult {
    /// Pairs within cutoff+skin (sort on the host before comparing).
    pub pairs: Vec<(u32, u32)>,
    /// Full atomic counter (exceeds `pairs.len()` on overflow).
    pub count: u32,
}

#[derive(Debug)]
pub struct EnergyResult {
    pub lj: f64,
    pub rf: f64,
    /// Filled by the caller for the homogeneous long-range LJ correction;
    /// the resident pair kernels intentionally keep this volume-only term
    /// separate from coordinate forces.
    pub dispersion_correction: f64,
    pub evaluated_pairs: f64,
    pub neighbor_overflow: bool,
    pub bonds: f64,
    pub angles: f64,
    pub proper_torsions: f64,
    pub improper_torsions: f64,
    pub bonded_partial: Vec<[f32; 4]>,
    pub gradients: Option<Vec<[f32; 4]>>,
    pub virial: Option<f64>,
    /// Intermolecular pair virial only. Internal molecular interactions are
    /// excluded because molecule-preserving volume moves leave them
    /// unchanged. Present when gradients/virial observables were requested.
    pub pair_virial: Option<f64>,
}

/// A serializable sample of the resident integrator state.  The random words
/// are part of the state, rather than an implementation detail: preserving
/// them makes a checkpoint/restart continue the same Langevin stream.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResidentDynamicsState {
    pub coordinates: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub rng_words: Vec<u32>,
}

fn dynamics_error(value: u32) -> Option<&'static str> {
    match value {
        0 => None,
        1 => Some("SETTLE numerical branch"),
        2 => Some("GPU pair-buffer capacity"),
        3 => Some("GPU neighbor-list capacity"),
        _ => Some("GPU dynamics numerical error"),
    }
}

pub struct ResidentPbc {
    pub adapter_info: wgpu::AdapterInfo,
    pub(crate) _context: GpuContext,
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: Vec<wgpu::Buffer>,
    bind_group: wgpu::BindGroup,
    pipelines: Vec<wgpu::ComputePipeline>,
    staging: [wgpu::Buffer; 3],
    staging_busy: [std::sync::atomic::AtomicBool; 3],
    staging_bytes: u64,
    n: u32,
    dims: [u32; 4],
    ncells: u32,
    max_pairs: u32,
    electro: [f32; 4],
    limit: f32,
    bonded_counts: [u32; 4],
    adjacency_offset: u32,
    molecules: Vec<Vec<usize>>,
    water_count: u32,
    solute_constraint_count: u32,
    dt_ps: f32,
    params: Vec<[f32; 4]>,
    srange_image: Vec<u32>,
    _allocation: AllocationReservation,
}

fn workgroups(n: u32) -> u32 {
    n.div_ceil(64).max(1)
}

/// Bytes needed for the combined resident dynamics snapshot.  Keep this
/// calculation next to the allocation code so a new readback field cannot
/// silently make the first frame exceed the staging slot.
fn dynamics_snapshot_staging_bytes(atom_count: u64) -> Option<u64> {
    atom_count
        .checked_mul(80)
        .and_then(|bytes| bytes.checked_add(64))
}

#[cfg(test)]
mod allocation_tests {
    use super::dynamics_snapshot_staging_bytes;

    #[test]
    fn snapshot_staging_covers_combined_state_and_observables() {
        for atoms in [1, 3, 2246, 50_000] {
            let atoms = atoms as u64;
            let per_atom = atoms * 16;
            let required = 5 * per_atom + 36; // sys, velocities, gradients, bonded, totals, status
            assert!(dynamics_snapshot_staging_bytes(atoms).unwrap() >= required);
        }
    }
}

struct ReadbackLease<'a> {
    buffer: &'a wgpu::Buffer,
    busy: &'a std::sync::atomic::AtomicBool,
    mapped: bool,
}
impl Drop for ReadbackLease<'_> {
    fn drop(&mut self) {
        if self.mapped {
            self.buffer.unmap();
        }
        self.busy.store(false, std::sync::atomic::Ordering::Release);
    }
}

impl ResidentPbc {
    /// Return whether a new orthorhombic box can use the resident cell
    /// allocation without changing its dimensions. A volume move that
    /// crosses a cell-count boundary must rebuild the derived evaluator;
    /// treating it as an ordinary Monte Carlo reject would bias the volume
    /// distribution.
    pub fn supports_box(&self, box_xyz: [f32; 3]) -> bool {
        if !box_xyz
            .iter()
            .all(|length| length.is_finite() && *length > 0.0)
            || self.electro[1] >= 0.5 * box_xyz[0].min(box_xyz[1]).min(box_xyz[2])
        {
            return false;
        }
        let limit = self.limit as f64;
        [box_xyz[0] as f64, box_xyz[1] as f64, box_xyz[2] as f64]
            .into_iter()
            .zip(self.dims[1..].iter().copied())
            .all(|(length, cells)| (length / limit).floor().max(1.0) as u32 == cells)
    }

    /// Current active cell dimensions.  The allocated buffers are larger
    /// than the active grid in some devices; callers must use this value when
    /// deciding whether a box transition requires a rebuild.
    pub fn cell_dimensions(&self) -> [u32; 3] {
        [self.dims[1], self.dims[2], self.dims[3]]
    }

    /// Construct a periodic evaluator on an existing shared context.
    pub async fn with_context(
        context: &GpuContext,
        packing: &PbcPacking,
        backend: &NonbondedElectrostatics,
        max_pairs: u32,
    ) -> Result<Self, Error> {
        Self::with_budget_context(
            context,
            packing,
            backend,
            max_pairs,
            context.memory_profile().budget(),
        )
        .await
    }

    async fn with_budget_context(
        context: &GpuContext,
        packing: &PbcPacking,
        backend: &NonbondedElectrostatics,
        max_pairs: u32,
        budget: u64,
    ) -> Result<Self, Error> {
        let electro = electrostatics_uniform(backend)?;
        // The packing cutoff and the backend cutoff must agree: cells are
        // sized for one limit, physics evaluated at one cutoff.
        if (electro[1] as f64 - packing.cutoff).abs() > 1e-6 {
            return Err(Error::Input("backend cutoff must match packing cutoff"));
        }
        let n = packing.dims[0];
        let ncells = packing.cell_count();
        if n == 0 || ncells == 0 || max_pairs == 0 {
            return Err(Error::Input("empty packing"));
        }
        let adapter_info = context.adapter_info().clone();
        let device = context.device().clone();
        let queue = context.queue().clone();
        let limits = device.limits();
        let max_buffer = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size as u64);
        let n64 = u64::from(n);
        let pairs_bytes = u64::from(max_pairs).checked_mul(8).ok_or(Error::Capacity)?;
        let sys_bytes = n64.checked_mul(32).ok_or(Error::Capacity)?;
        // Cell heads, links, two range words/atom, pair counter, and a
        // single atomic numerical-status flag.
        let meta_words = u64::from(ncells)
            .checked_add(8u64.checked_mul(n64).ok_or(Error::Capacity)?)
            .and_then(|v| v.checked_add(5))
            .and_then(|v| v.checked_add(u64::from(workgroups(n))))
            .ok_or(Error::Capacity)?;
        let meta_bytes = meta_words.checked_mul(4).ok_or(Error::Capacity)?;
        let out_bytes = 6u64
            .checked_mul(n64)
            .and_then(|v| v.checked_add(2))
            .and_then(|v| v.checked_mul(16))
            .ok_or(Error::Capacity)?;
        let bonded_bytes = (packing.bonded.len() as u64)
            .checked_mul(16)
            .ok_or(Error::Capacity)?;
        let bonded_coord_bytes = n64.checked_mul(16).ok_or(Error::Capacity)?;
        let state_bytes = 2u64
            .checked_mul(n64)
            .and_then(|v| v.checked_mul(16))
            .ok_or(Error::Capacity)?;
        let special_bytes = (packing.specials.len() as u64)
            .checked_mul(16)
            .ok_or(Error::Capacity)?;
        let heap = sys_bytes
            .checked_add(meta_bytes)
            .and_then(|v| v.checked_add(special_bytes))
            .and_then(|v| v.checked_add(pairs_bytes))
            .and_then(|v| v.checked_add(bonded_bytes))
            .and_then(|v| v.checked_add(bonded_coord_bytes))
            .and_then(|v| v.checked_add(state_bytes))
            .and_then(|v| v.checked_add(64 * 4))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or(Error::Capacity)?;
        // A dynamics snapshot packs 5 per-atom regions plus a 32-byte
        // scalar block and a 4-byte status word in one transfer.  Keep a
        // little alignment headroom: the previous `+20` calculation was
        // four bytes smaller than that exact layout for every atom count,
        // so the first snapshot of an otherwise valid run returned a
        // misleading `Error::Capacity` before any physics was read back.
        let staging_bytes = dynamics_snapshot_staging_bytes(n64)
            .ok_or(Error::Capacity)?
            .max(65536);
        if heap
            .checked_add(3 * staging_bytes)
            .is_none_or(|bytes| bytes > budget)
            || pairs_bytes > max_buffer
            || out_bytes > max_buffer
            || sys_bytes > max_buffer
            || workgroups(n) > limits.max_compute_workgroups_per_dimension
        {
            return Err(Error::Capacity);
        }
        let reservation =
            context.reserve(heap.checked_add(3 * staging_bytes).ok_or(Error::Capacity)?)?;
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let storage = wgpu::BufferUsages::STORAGE;
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let immutable = |label: &str, data: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if data.is_empty() { &[0; 48] } else { data },
                usage: storage,
            })
        };
        let buffers = vec![
            buffer(
                "pbc config",
                112,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            buffer(
                "pbc sys",
                sys_bytes,
                storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            ),
            buffer(
                "pbc meta",
                meta_bytes,
                storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            ),
            immutable("pbc specials", bytemuck::cast_slice(&packing.specials)),
            buffer(
                "pbc pairs",
                pairs_bytes,
                storage | wgpu::BufferUsages::COPY_SRC,
            ),
            buffer("pbc out", out_bytes, storage | wgpu::BufferUsages::COPY_SRC),
            immutable("pbc bonded", bytemuck::cast_slice(&packing.bonded)),
            buffer(
                "pbc bonded coords",
                bonded_coord_bytes,
                storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            ),
            buffer(
                "pbc dynamics state",
                state_bytes,
                storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            ),
        ];
        let entries: Vec<_> = (0..9)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding == 3 || binding == 6,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GlySys PBC"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("resident_rng.wgsl"),
                    include_str!("pbc.wgsl")
                )
                .into(),
            ),
        });
        let pipelines = [
            "insert_atoms",
            "count_neighbors",
            "eval",
            "reduce",
            "bonded_energy",
            "integrate_first",
            "settle",
            "kick_second",
            "rattle",
            "refresh_bonded_coords",
            "clear_meta",
            "integrate_first_half",
            "langevin_ou",
            "integrate_second_half",
            "check_neighbors",
            "scan_neighbors",
            "scan_neighbor_blocks",
            "apply_neighbor_offsets",
            "fill_neighbors",
            "sort_neighbors",
            "finish_neighbors",
        ]
        .iter()
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        })
        .collect();
        for entry in [
            "insert_atoms",
            "count_neighbors",
            "eval",
            "reduce",
            "bonded_energy",
            "integrate_first",
            "settle",
            "kick_second",
            "rattle",
            "refresh_bonded_coords",
            "clear_meta",
            "integrate_first_half",
            "langevin_ou",
            "integrate_second_half",
            "check_neighbors",
            "scan_neighbors",
            "scan_neighbor_blocks",
            "apply_neighbor_offsets",
            "fill_neighbors",
            "sort_neighbors",
            "finish_neighbors",
        ] {
            context.record_pipeline(format!("pbc.{entry}"));
        }
        let staging = std::array::from_fn(|_| {
            buffer(
                "pbc readback",
                staging_bytes,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            )
        });
        let validation = device.pop_error_scope().await;
        let allocation = device.pop_error_scope().await;
        if let Some(e) = validation {
            return Err(Error::Execution(e.to_string()));
        }
        if allocation.is_some() {
            return Err(Error::Capacity);
        }
        // Static specials-range image, uploaded with every coordinate update
        // (small: 2 u32 per atom).
        let mut srange_image = vec![0u32; 2 * n as usize];
        for (i, r) in packing.ranges.iter().enumerate() {
            srange_image[2 * i] = r[0];
            srange_image[2 * i + 1] = r[1];
        }
        Ok(Self {
            adapter_info,
            _context: context.clone(),
            device,
            queue,
            buffers,
            bind_group,
            pipelines,
            staging,
            staging_busy: std::array::from_fn(|_| std::sync::atomic::AtomicBool::new(false)),
            staging_bytes,
            n,
            dims: packing.dims,
            ncells,
            max_pairs,
            electro,
            limit: packing.limit as f32,
            bonded_counts: packing.bonded_counts,
            adjacency_offset: packing.adjacency_offset,
            molecules: packing.molecules.clone(),
            water_count: packing.bonded_counts[3],
            solute_constraint_count: packing.solute_constraint_count,
            dt_ps: f64::NAN as f32,
            params: packing.params.clone(),
            srange_image,
            _allocation: reservation,
        })
    }

    fn meta_srange_off(&self) -> u64 {
        (u64::from(self.ncells) + u64::from(self.n)) * 4
    }
    fn meta_count_off(&self) -> u64 {
        (u64::from(self.ncells) + 3 * u64::from(self.n)) * 4
    }
    fn meta_status_off(&self) -> u64 {
        self.meta_count_off() + 4
    }

    /// Upload interleaved params/coords, static specials ranges, a cleared
    /// cell-head table, and a zeroed pair counter. The uniform (dims, box,
    /// electrostatics, flags) is rewritten every call: box and coordinates
    /// change per frame, so nothing is cached across calls.
    pub fn set_coordinates(&self, unwrapped: &[Vec3], box_xyz: [f32; 3], gradients: bool) {
        assert_eq!(unwrapped.len() as u32, self.n);
        let mut config = [0u8; 112];
        for (i, v) in self.dims.iter().enumerate() {
            config[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in [box_xyz[0], box_xyz[1], box_xyz[2], self.limit]
            .iter()
            .enumerate()
        {
            config[16 + 4 * i..20 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in self.electro.iter().enumerate() {
            config[32 + 4 * i..36 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        let misc = [
            if gradients { 1f32 } else { 0. },
            self.max_pairs as f32,
            self.ncells as f32,
            0.,
        ];
        for (i, v) in misc.iter().enumerate() {
            config[48 + 4 * i..52 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in self.bonded_counts.iter().enumerate() {
            config[64 + 4 * i..68 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        let dynamic = [
            self.water_count as f32,
            self.solute_constraint_count as f32,
            self.dt_ps,
            0.,
        ];
        for (i, v) in dynamic.iter().enumerate() {
            config[80 + 4 * i..84 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        // NVE defaults to a full drift interval and a harmless temperature;
        // the NVT entry point overwrites these three scalar fields before its
        // dispatch without touching resident coordinates or velocities.
        let thermo = [300.0f32, 1.0, self.adjacency_offset as f32, 0.0];
        for (i, v) in thermo.iter().enumerate() {
            config[96 + 4 * i..100 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        self.queue.write_buffer(&self.buffers[0], 0, &config);
        let mut sys = Vec::with_capacity(unwrapped.len() * 2);
        for (p, c) in self.params.iter().zip(unwrapped.iter()) {
            sys.push(*p);
            // Centering before the f32 cast reduces cancellation in bonded
            // coordinate differences. It is a rigid translation: GPU cells
            // wrap the centered coordinate and minimum image is unchanged.
            sys.push([
                (c.x - 0.5 * box_xyz[0] as f64) as f32,
                (c.y - 0.5 * box_xyz[1] as f64) as f32,
                (c.z - 0.5 * box_xyz[2] as f64) as f32,
                0.,
            ]);
        }
        self.queue
            .write_buffer(&self.buffers[1], 0, bytemuck::cast_slice(&sys));
        let mut bonded_coords = vec![[0f32; 4]; unwrapped.len()];
        for group in &self.molecules {
            let mut center = [0f64; 3];
            for &atom in group {
                center[0] += unwrapped[atom].x;
                center[1] += unwrapped[atom].y;
                center[2] += unwrapped[atom].z;
            }
            let count = group.len() as f64;
            let anchor = group[0];
            for &atom in group {
                bonded_coords[atom] = [
                    (unwrapped[atom].x - center[0] / count) as f32,
                    (unwrapped[atom].y - center[1] / count) as f32,
                    (unwrapped[atom].z - center[2] / count) as f32,
                    f32::from_bits(anchor as u32),
                ];
            }
        }
        self.queue
            .write_buffer(&self.buffers[7], 0, bytemuck::cast_slice(&bonded_coords));
        self.queue.write_buffer(
            &self.buffers[2],
            self.meta_srange_off(),
            bytemuck::cast_slice(&self.srange_image),
        );
        self.queue
            .write_buffer(&self.buffers[2], 0, &vec![0xFFu8; self.ncells as usize * 4]);
        self.queue
            .write_buffer(&self.buffers[2], self.meta_count_off(), &[0; 4]);
        // A coordinate upload starts a fresh evaluation session.  Dynamics
        // itself deliberately leaves the numerical-status word sticky until
        // the host polls it at a bounded checkpoint.
        self.queue
            .write_buffer(&self.buffers[2], self.meta_status_off(), &[0; 4]);
        self.queue.write_buffer(
            &self.buffers[2],
            self.meta_status_off() + 4,
            &1u32.to_le_bytes(),
        );
    }

    /// Configure the resident integrator timestep in picoseconds. The value
    /// is copied into the next coordinate upload's dynamic uniform; changing
    /// it does not touch resident positions, velocities, or force buffers.
    pub fn set_timestep(&mut self, dt_ps: f32) -> Result<(), Error> {
        if !dt_ps.is_finite() || dt_ps <= 0.0 {
            return Err(Error::Input(
                "dynamics timestep must be finite and positive",
            ));
        }
        self.dt_ps = dt_ps;
        Ok(())
    }

    /// Upload initial velocities for the resident dynamics state. Positions
    /// are uploaded separately with [`Self::set_coordinates`], so callers can
    /// initialize forces first and then start the asynchronous step loop.
    pub fn set_velocities(&self, velocities: &[Vec3]) -> Result<(), Error> {
        let rng_words: Vec<u32> = (0..self.n as usize)
            .map(|i| {
                // Per-atom state words make the OU thermostat reproducible
                // without a host RNG upload on every step. A nonzero seed is
                // derived from the atom index; checkpoint readback preserves
                // the evolving bit pattern in the resident buffer.
                0x9E37_79B9u32.wrapping_add((i as u32).wrapping_mul(0x6D2B_79F5))
            })
            .collect();
        self.set_velocities_with_rng(velocities, &rng_words)
    }

    /// Upload velocities together with the exact per-atom RNG words from a
    /// checkpoint.  This is deliberately separate from [`Self::set_velocities`]
    /// so a fresh run retains the stable seed convention while a restart can
    /// continue its original stochastic stream.
    pub fn set_velocities_with_rng(
        &self,
        velocities: &[Vec3],
        rng_words: &[u32],
    ) -> Result<(), Error> {
        if velocities.len() as u32 != self.n || rng_words.len() as u32 != self.n {
            return Err(Error::Input("velocity/RNG count"));
        }
        if velocities
            .iter()
            .any(|v| !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite())
        {
            return Err(Error::Input("non-finite velocity"));
        }
        let packed: Vec<[f32; 4]> = velocities
            .iter()
            .zip(rng_words)
            .map(|(v, &seed)| [v.x as f32, v.y as f32, v.z as f32, f32::from_bits(seed)])
            .collect();
        self.queue
            .write_buffer(&self.buffers[8], 0, bytemuck::cast_slice(&packed));
        Ok(())
    }

    /// Initialize a GPU-resident constrained dynamics loop. The first force
    /// evaluation remains an explicit call to [`Self::energy_and_forces`]; it
    /// is needed to seed the force buffer that the first Verlet kick consumes.
    pub fn initialize_dynamics(
        &mut self,
        coordinates: &[Vec3],
        velocities: &[Vec3],
        box_xyz: [f32; 3],
        dt_ps: f32,
    ) -> Result<(), Error> {
        self.set_timestep(dt_ps)?;
        self.set_coordinates(coordinates, box_xyz, true);
        self.set_velocities(velocities)
    }

    /// Restore coordinates, velocities, and stochastic state from a resident
    /// checkpoint.  The force buffer is intentionally not serialized here:
    /// call [`Self::energy_and_forces`] once after restoring, then resume the
    /// resident step loop.  No trajectory coordinate is reconstructed on the
    /// CPU between subsequent steps.
    pub fn set_dynamics_state(
        &self,
        state: &ResidentDynamicsState,
        box_xyz: [f32; 3],
        gradients: bool,
    ) -> Result<(), Error> {
        if state.coordinates.len() as u32 != self.n {
            return Err(Error::Input("checkpoint coordinate count"));
        }
        self.set_coordinates(&state.coordinates, box_xyz, gradients);
        self.set_velocities_with_rng(&state.velocities, &state.rng_words)
    }

    fn write_nvt_uniforms(&self, temperature_k: f32, friction_per_ps: f32, drift_multiplier: f32) {
        self.queue
            .write_buffer(&self.buffers[0], 92, &friction_per_ps.to_le_bytes());
        self.queue
            .write_buffer(&self.buffers[0], 96, &temperature_k.to_le_bytes());
        self.queue
            .write_buffer(&self.buffers[0], 100, &drift_multiplier.to_le_bytes());
    }

    /// Record a chain of compute passes into one encoder with a single
    /// submit. Besides fewer round trips, this keeps pass ordering explicit
    /// for software Vulkan, where back-to-back submits without an intervening
    /// transfer have been observed to stall.
    fn dispatch_chain(&self, jobs: &[(usize, u32)]) {
        self.dispatch_repeated(jobs, 1);
    }

    fn dispatch_repeated(&self, jobs: &[(usize, u32)], steps: usize) {
        let mut expanded = Vec::with_capacity(jobs.len() + 10);
        for &(pipeline, groups) in jobs {
            if pipeline == 0 {
                let n = workgroups(self.n);
                expanded.extend_from_slice(&[
                    (14, n),
                    (10, workgroups(self.n.max(self.ncells))),
                    (0, n),
                    (1, n),
                    (15, n),
                    (16, 1),
                    (17, n),
                    (18, n),
                    (19, n),
                    (20, 1),
                ]);
            } else if pipeline != 10 {
                expanded.push((pipeline, groups));
            }
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..steps {
            for &(pipeline, groups) in &expanded {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines[pipeline]);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
    }

    async fn readback(&self, src: usize, offset: u64, size: u64) -> Result<Vec<u8>, Error> {
        let mut out = Vec::with_capacity(usize::try_from(size).map_err(|_| Error::Capacity)?);
        let mut copied = 0;
        while copied < size {
            let chunk = (size - copied).min(self.staging_bytes);
            out.extend_from_slice(
                &self
                    .readback_ranges(&[(src, offset + copied, chunk)])
                    .await?,
            );
            copied += chunk;
        }
        Ok(out)
    }

    /// Copy a bounded snapshot in one submission/map. The source buffers all
    /// refer to the same queue boundary, including status and stochastic state.
    async fn readback_ranges(&self, ranges: &[(usize, u64, u64)]) -> Result<Vec<u8>, Error> {
        use std::sync::atomic::Ordering;
        let size: u64 = ranges.iter().map(|r| r.2).sum();
        if size > self.staging_bytes {
            return Err(Error::Capacity);
        }
        let slot = self
            .staging_busy
            .iter()
            .position(|busy| {
                busy.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
            })
            .ok_or_else(|| Error::Execution("all three readback slots are occupied".into()))?;
        let mut lease = ReadbackLease {
            buffer: &self.staging[slot],
            busy: &self.staging_busy[slot],
            mapped: false,
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MD snapshot"),
            });
        let mut destination = 0;
        for &(source, offset, bytes) in ranges {
            encoder.copy_buffer_to_buffer(
                &self.buffers[source],
                offset,
                lease.buffer,
                destination,
                bytes,
            );
            destination += bytes;
        }
        self.queue.submit(Some(encoder.finish()));
        let slice = lease.buffer.slice(0..size);
        let (tx, rx) = futures_channel::oneshot::channel();
        lease.mapped = true;
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::Wait)
            .map_err(|e| Error::Execution(e.to_string()))?;
        if let Err(error) = rx.await.map_err(|e| Error::Execution(e.to_string()))? {
            lease.mapped = false;
            return Err(Error::Execution(error.to_string()));
        }
        let mapped = slice.get_mapped_range();
        let out = mapped.to_vec();
        drop(mapped);
        lease.buffer.unmap();
        lease.mapped = false;
        Ok(out)
    }

    /// One asynchronous transfer for status, checkpoint and force observables.
    pub async fn read_dynamics_snapshot(
        &self,
        box_xyz: [f32; 3],
    ) -> Result<(ResidentDynamicsState, EnergyResult), Error> {
        let n = u64::from(self.n) * 16;
        let bytes = self
            .readback_ranges(&[
                (1, 0, 2 * n),
                (8, 0, n),
                (5, n, n),
                (5, 3 * n, n),
                (5, 4 * n, 32),
                (2, self.meta_status_off(), 4),
            ])
            .await?;
        let status = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
        if let Some(error) = dynamics_error(status) {
            return Err(Error::Execution(error.into()));
        }
        let n = n as usize;
        let state = self.decode_checkpoint(&bytes[..2 * n], &bytes[2 * n..3 * n], box_xyz);
        let energy = Self::decode_observables(
            &bytes[5 * n..5 * n + 32],
            Some(&bytes[3 * n..4 * n]),
            &bytes[4 * n..5 * n],
        );
        Ok((state, energy))
    }

    /// Build cells and enumerate pairs within cutoff+skin.
    pub async fn neighbor_list(&self) -> Result<NeighborResult, Error> {
        self.queue
            .write_buffer(&self.buffers[2], self.meta_status_off(), &[0; 4]);
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        self.dispatch_chain(&[(0, workgroups(self.n))]);
        let count_bytes = self.readback(2, self.meta_count_off(), 4).await?;
        let directed = u32::from_le_bytes(count_bytes[0..4].try_into().unwrap());
        if let Some(e) = self.device.pop_error_scope().await {
            return Err(Error::Execution(e.to_string()));
        }
        if directed > 2 * self.max_pairs {
            return Err(Error::Capacity);
        }
        let pair_bytes = if directed > 0 {
            self.readback(4, 0, u64::from(directed) * 4).await?
        } else {
            Vec::new()
        };
        let offset_bytes = self
            .readback(
                2,
                (u64::from(self.ncells) + 7 * u64::from(self.n) + 4) * 4,
                (u64::from(self.n) + 1) * 4,
            )
            .await?;
        let offsets: &[u32] = bytemuck::cast_slice(&offset_bytes);
        let indices: &[u32] = bytemuck::cast_slice(&pair_bytes);
        let mut pairs = Vec::new();
        for a in 0..self.n {
            for &b in &indices[offsets[a as usize] as usize..offsets[a as usize + 1] as usize] {
                if b > a {
                    pairs.push((a, b));
                }
            }
        }
        Ok(NeighborResult {
            count: pairs.len() as u32,
            pairs,
        })
    }

    pub async fn neighbor_rebuild_count(&self) -> Result<u32, Error> {
        let bytes = self.readback(2, self.meta_status_off() + 8, 4).await?;
        Ok(u32::from_le_bytes(
            bytes[..4]
                .try_into()
                .map_err(|_| Error::Input("neighbor counter"))?,
        ))
    }

    /// Build cells, evaluate nonbonded energy/forces, reduce totals.
    pub async fn energy_and_forces(&self, gradients: bool) -> Result<EnergyResult, Error> {
        // The resident dynamics loop keeps gradients enabled, but score-only
        // callers must be able to omit the gradient/adjoint work without
        // re-uploading coordinates.  Rewrite only the flag in the existing
        // uniform; queue ordering makes it visible before this dispatch.
        self.queue.write_buffer(
            &self.buffers[0],
            48,
            &(if gradients { 1.0f32 } else { 0.0f32 }).to_le_bytes(),
        );
        self.queue
            .write_buffer(&self.buffers[2], self.meta_status_off(), &[0; 4]);
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let n = workgroups(self.n);
        self.dispatch_chain(&[
            (10, workgroups(self.n.max(self.ncells))),
            (0, n),
            (2, n),
            (4, n),
            (3, 1),
        ]);
        let result = self.read_dynamics_observables(gradients).await;
        if let Some(e) = self.device.pop_error_scope().await {
            return Err(Error::Execution(e.to_string()));
        }
        result
    }

    /// Extract the last resident force evaluation without recomputing it on
    /// either backend. The fourth gradient lane carries the pair/bond virial.
    pub async fn read_dynamics_observables(&self, gradients: bool) -> Result<EnergyResult, Error> {
        let total_bytes = self.readback(5, 4 * u64::from(self.n) * 16, 32).await?;
        let grad_bytes = if gradients {
            Some(
                self.readback(5, u64::from(self.n) * 16, u64::from(self.n) * 16)
                    .await?,
            )
        } else {
            None
        };
        let partial_bytes = self
            .readback(5, 3 * u64::from(self.n) * 16, u64::from(self.n) * 16)
            .await?;
        Ok(Self::decode_observables(
            &total_bytes,
            grad_bytes.as_deref(),
            &partial_bytes,
        ))
    }
    fn decode_observables(
        total_bytes: &[u8],
        grad_bytes: Option<&[u8]>,
        partial_bytes: &[u8],
    ) -> EnergyResult {
        let t: &[f32] = bytemuck::cast_slice(total_bytes);
        let overflow = t[3] > 0.5;
        let grad = grad_bytes.map(|bytes| {
            bytemuck::cast_slice::<u8, f32>(bytes)
                .chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect()
        });
        let partial: &[f32] = bytemuck::cast_slice(partial_bytes);
        let mut bonded_totals = [0f64; 4];
        for c in partial.chunks_exact(4) {
            for (total, value) in bonded_totals.iter_mut().zip(c) {
                *total += *value as f64;
            }
        }
        let virial = grad
            .as_ref()
            .map(|g: &Vec<[f32; 4]>| g.iter().map(|v| v[3] as f64).sum());
        let pair_virial = grad.as_ref().map(|_| t[4] as f64);
        EnergyResult {
            lj: t[0] as f64,
            rf: t[1] as f64,
            dispersion_correction: 0.,
            evaluated_pairs: t[2] as f64,
            neighbor_overflow: overflow,
            bonds: bonded_totals[0],
            angles: bonded_totals[1],
            proper_torsions: bonded_totals[2],
            improper_torsions: bonded_totals[3],
            bonded_partial: partial
                .chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect(),
            gradients: grad,
            virial,
            pair_virial,
        }
    }

    /// Advance one resident velocity-Verlet step. The pass order is:
    ///
    /// `kick/drift -> SETTLE/SHAKE -> refresh bonded frame -> cells/pairs ->
    /// forces -> kick -> RATTLE`.
    ///
    /// No coordinates, velocities, or forces are read back. A later explicit
    /// [`Self::read_dynamics_state`] call can sample a checkpoint or frame.
    pub async fn dynamics_step(&self) -> Result<(), Error> {
        self.dynamics_steps(1).await
    }

    /// Encode a bounded sequence without intermediate host synchronization.
    /// Callers split at output and protocol boundaries before using this API.
    pub async fn dynamics_steps(&self, steps: usize) -> Result<(), Error> {
        if !(1..=128).contains(&steps) {
            return Err(Error::Input("dynamics batch must contain 1–128 steps"));
        }
        if !self.dt_ps.is_finite() || self.dt_ps <= 0.0 {
            return Err(Error::Input("dynamics timestep is not configured"));
        }
        self.queue
            .write_buffer(&self.buffers[0], 100, &1.0f32.to_le_bytes());
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let groups = workgroups(self.n.max(self.ncells));
        self.dispatch_repeated(
            &[
                (10, groups),            // clear resident cells/counter
                (5, workgroups(self.n)), // first half kick + drift
                (6, groups),             // analytic SETTLE + SHAKE
                (9, workgroups(self.n)), // update molecule-centered bonded data
                (0, workgroups(self.n)), // cell insertion
                (2, workgroups(self.n)), // nonbonded force
                (4, workgroups(self.n)), // bonded force/energy
                (3, 1),                  // deterministic reduction
                (7, workgroups(self.n)), // second half kick
                (8, groups),             // analytic water RATTLE + solute RATTLE
            ],
            steps,
        );
        if let Some(e) = self.device.pop_error_scope().await {
            return Err(Error::Execution(e.to_string()));
        }
        Ok(())
    }

    /// Advance one resident BAOAB Langevin step. The OU thermostat is
    /// counter-based and lives in the WGSL state word, while temperature and
    /// friction are scalar uniforms supplied once per dispatch. This keeps
    /// the full coordinate/velocity/force path on the device; only explicit
    /// checkpoint reads cross the API boundary.
    pub async fn dynamics_step_nvt(
        &self,
        temperature_k: f32,
        friction_per_ps: f32,
    ) -> Result<(), Error> {
        self.dynamics_steps_nvt(1, temperature_k, friction_per_ps)
            .await
    }

    pub async fn dynamics_steps_nvt(
        &self,
        steps: usize,
        temperature_k: f32,
        friction_per_ps: f32,
    ) -> Result<(), Error> {
        if !(1..=128).contains(&steps) {
            return Err(Error::Input("dynamics batch must contain 1–128 steps"));
        }
        if !self.dt_ps.is_finite() || self.dt_ps <= 0.0 {
            return Err(Error::Input("dynamics timestep is not configured"));
        }
        if !temperature_k.is_finite() || temperature_k <= 0.0 {
            return Err(Error::Input("NVT temperature must be finite and positive"));
        }
        if !friction_per_ps.is_finite() || friction_per_ps < 0.0 {
            return Err(Error::Input("NVT friction must be finite and non-negative"));
        }
        self.write_nvt_uniforms(temperature_k, friction_per_ps, 2.0);
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let groups = workgroups(self.n.max(self.ncells));
        self.dispatch_repeated(
            &[
                (10, groups),
                (11, workgroups(self.n)), // first A/2
                (6, groups),              // SETTLE/SHAKE
                (12, workgroups(self.n)), // OU thermostat
                (13, workgroups(self.n)), // second A/2
                (6, groups),              // SETTLE/SHAKE
                (9, workgroups(self.n)),  // bonded frame
                (0, workgroups(self.n)),  // cells
                (2, workgroups(self.n)),  // forces
                (4, workgroups(self.n)),  // bonded forces
                (3, 1),                   // reduction
                (7, workgroups(self.n)),  // second B/2
                (8, groups),              // RATTLE
            ],
            steps,
        );
        if let Some(e) = self.device.pop_error_scope().await {
            return Err(Error::Execution(e.to_string()));
        }
        Ok(())
    }

    /// Check the sticky status flag written by the resident shader chain.
    /// This is an infrequent checkpoint/fallback poll, not part of the hot
    /// step loop. A set flag means that one dispatch in the current batch
    /// encountered a constraint or bounded-neighbourhood error; the caller
    /// must restore the last committed CPU checkpoint rather than exporting
    /// resident coordinates.
    pub async fn dynamics_status(&self) -> Result<Option<&'static str>, Error> {
        let bytes = self.readback(2, self.meta_status_off(), 4).await?;
        let value = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        Ok(dynamics_error(value))
    }

    /// Read the resident coordinates, velocities, and per-atom RNG words. The
    /// GPU keeps coordinates centered by half a box length to improve f32
    /// precision; this method restores the caller's coordinate convention.
    pub async fn read_dynamics_checkpoint(
        &self,
        box_xyz: [f32; 3],
    ) -> Result<ResidentDynamicsState, Error> {
        let sys_bytes = u64::from(self.n) * 2 * 16;
        let coord_bytes = self.readback(1, 0, sys_bytes).await?;
        let velocity_bytes = self.readback(8, 0, u64::from(self.n) * 16).await?;
        Ok(self.decode_checkpoint(&coord_bytes, &velocity_bytes, box_xyz))
    }
    fn decode_checkpoint(
        &self,
        coord_bytes: &[u8],
        velocity_bytes: &[u8],
        box_xyz: [f32; 3],
    ) -> ResidentDynamicsState {
        let sys: &[f32] = bytemuck::cast_slice(&coord_bytes);
        let vel: &[f32] = bytemuck::cast_slice(&velocity_bytes);
        let mut coordinates = Vec::with_capacity(self.n as usize);
        let mut velocities = Vec::with_capacity(self.n as usize);
        let mut rng_words = Vec::with_capacity(self.n as usize);
        for i in 0..self.n as usize {
            let p = &sys[8 * i + 4..8 * i + 7];
            coordinates.push(Vec3 {
                x: p[0] as f64 + 0.5 * box_xyz[0] as f64,
                y: p[1] as f64 + 0.5 * box_xyz[1] as f64,
                z: p[2] as f64 + 0.5 * box_xyz[2] as f64,
            });
            let v = &vel[4 * i..4 * i + 4];
            velocities.push(Vec3 {
                x: v[0] as f64,
                y: v[1] as f64,
                z: v[2] as f64,
            });
            rng_words.push(v[3].to_bits());
        }
        ResidentDynamicsState {
            coordinates,
            velocities,
            rng_words,
        }
    }

    /// Convenience readback for trajectory frames that do not need the RNG
    /// words. Checkpoint callers should use [`Self::read_dynamics_checkpoint`]
    /// instead.
    pub async fn read_dynamics_state(
        &self,
        box_xyz: [f32; 3],
    ) -> Result<(Vec<Vec3>, Vec<Vec3>), Error> {
        let state = self.read_dynamics_checkpoint(box_xyz).await?;
        Ok((state.coordinates, state.velocities))
    }
}
