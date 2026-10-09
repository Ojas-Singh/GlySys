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
use crate::pbc_leapfrog::{
    LeapfrogCoupling, LeapfrogEngine, LeapfrogKernel, LeapfrogShared, LeapfrogVariables,
};
use crate::pbc_tiles::{SharedBuffers, TileEngine, TileKernel, TilePacking, TileSizing};
use glysys::{ParameterizedSystem, Vec3};
use glysys_energy::pbc::{
    BoxVectors, NonbondedElectrostatics, classify_waters, molecules, water_equilibrium,
};
use glysys_energy::pbc_cluster::{EwaldPairPolynomials, ewald_pair_polynomials};
use std::collections::BTreeMap;
use wgpu::util::DeviceExt;

// Keep each native/WebGPU command buffer bounded. A 64-step encoded chain
// triggered wgpu out-of-memory on the RX 7800 XT. Matched 5-second samples
// favored eight-step packets slightly over sixteen, while thirty-two was
// slower, so retain the smaller measured winner.
pub const MAX_ENCODED_DYNAMICS_STEPS: usize = 8;
const MAX_SOLUTE_CONSTRAINT_COMPONENT_BONDS: usize = 8;
pub const TILED_NEIGHBORS_PER_ATOM: u32 = 640;
const INDIRECT_NEIGHBOR_DISPATCH: u32 = u32::MAX;
/// Job codes at or above this value select a tiled-engine pipeline.
const TILE_JOB: usize = 1000;
/// Job codes of the leap-frog coupling kernels.
const LEAPFROG_JOB: usize = 2000;
/// SETTLE/SHAKE that also records the constraint virial.
const SETTLE_VIRIAL: usize = 30;

/// Explicit nonbonded implementation of a resident evaluator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PbcKernel {
    /// Morton-sorted 32-atom blocks with half-shell tiles, separate 1-4 and
    /// per-term bonded passes, and deterministic fixed-point accumulation.
    #[default]
    Tiles,
    /// Historical per-atom CSR Verlet list evaluated by one invocation per atom.
    Csr,
    /// Historical fixed 640-entry rows evaluated by 64 cooperating lanes.
    FixedRows,
}

impl PbcKernel {
    /// Diagnostic label recorded with run provenance.
    pub fn label(self) -> &'static str {
        match self {
            Self::Tiles => "block-tiles-32",
            Self::Csr => "serial-per-atom",
            Self::FixedRows => "cooperative-64-lane",
        }
    }
}

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
    pub solute_constraint_component_offset: u32,
    pub solute_constraint_component_count: u32,
    pub dims: [u32; 4],
    pub limit: f64,
    pub cutoff: f64,
    /// Orthorhombic box the packing was built for (Å).
    pub box_angstrom: [f64; 3],
    /// Size the tile lists for this many times the present density. A
    /// barostat that compresses the box needs the room.
    pub tile_density_headroom: f64,
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
        for (pair, scee, scnb) in system.one_four_pairs() {
            // Both directions: each endpoint's thread evaluates its half of
            // the pair, so a one-sided entry would compute one half scaled
            // and the other half as a regular pair.
            for (first, second) in [(pair[0], pair[1]), (pair[1], pair[0])] {
                exceptions[first].insert(second, (scee as f32, scnb as f32, true));
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
        let mut solute_constraints = Vec::<[f32; 4]>::new();
        let mut solute_constraint_edges = Vec::<[u32; 2]>::new();
        for bond in system.bonds() {
            let [a, b] = bond.atoms();
            if !in_water.contains(&a)
                && !in_water.contains(&b)
                && (system.atoms()[a].element() == 1 || system.atoms()[b].element() == 1)
            {
                solute_constraint_edges.push([a as u32, b as u32]);
                solute_constraints.push([a as f32, b as f32, bond.length() as f32, 0.]);
            }
        }
        let constraint_components = solute_constraint_components(&solute_constraint_edges, n)?;
        let mut constraint_component_ranges = Vec::with_capacity(constraint_components.len());
        let mut local_constraint_offset = 0usize;
        for component in &constraint_components {
            constraint_component_ranges.push((
                u32::try_from(solute_at + local_constraint_offset).map_err(|_| Error::Capacity)?,
                u32::try_from(component.len()).map_err(|_| Error::Capacity)?,
            ));
            for &constraint in component {
                bonded.push(solute_constraints[constraint]);
            }
            local_constraint_offset += component.len();
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
        let solute_constraint_component_offset =
            u32::try_from(bonded.len()).map_err(|_| Error::Capacity)?;
        for (first_bond, bond_count) in &constraint_component_ranges {
            bonded.push([
                f32::from_bits(*first_bond),
                f32::from_bits(*bond_count),
                0.0,
                0.0,
            ]);
        }
        let solute_constraint_component_count =
            u32::try_from(constraint_component_ranges.len()).map_err(|_| Error::Capacity)?;
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
            solute_constraint_component_offset,
            solute_constraint_component_count,
            dims: [n as u32, nx, ny, nz],
            limit,
            cutoff,
            box_angstrom: b,
            tile_density_headroom: 1.0,
        })
    }

    pub fn cell_count(&self) -> u32 {
        self.dims[1] * self.dims[2] * self.dims[3]
    }
}

/// Reaction-field uniform parameters (method tag 0), or method tag 1 with
/// the Ewald coefficient and the fitted direct-space pair polynomials for
/// PME. Host-side f64 math, cast once, documented in the validation report.
/// The PME variant carries no cutoff of its own: it uses the packing's.
fn electrostatics_uniform(
    backend: &NonbondedElectrostatics,
    cutoff: f64,
) -> Result<([f32; 4], Option<EwaldPairPolynomials>), Error> {
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
            Ok(([0., rc as f32, krf as f32, crf as f32], None))
        }
        NonbondedElectrostatics::Pme {
            alpha_per_angstrom,
            interpolation_order,
            ..
        } => {
            if *interpolation_order != 4 {
                return Err(Error::Input("GPU PME interpolates with order 4 only"));
            }
            let ewald = ewald_pair_polynomials(*alpha_per_angstrom, cutoff)
                .map_err(|_| Error::Input("Ewald coefficient and cutoff"))?;
            Ok((
                [1., cutoff as f32, *alpha_per_angstrom as f32, 0.],
                Some(ewald),
            ))
        }
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

/// Thermodynamic scalars read from a resident explicit GPU state. Coordinates,
/// velocities, RNG, and force arrays stay on the device.
#[derive(Clone, Copy, Debug)]
pub struct DynamicsScalars {
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub neighbor_rebuild_count: u64,
    pub readback_bytes: u64,
}

/// A serializable sample of the resident integrator state.  The random words
/// are part of the state, rather than an implementation detail: preserving
/// them makes a checkpoint/restart continue the same Langevin stream.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResidentDynamicsState {
    pub coordinates: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub rng_words: Vec<u32>,
    #[serde(default)]
    pub neighbor_rebuild_count: u64,
}

fn dynamics_error(value: u32) -> Option<String> {
    let constraint_kind = value & 0xf000_0000;
    if constraint_kind == 0x5000_0000 || constraint_kind == 0x6000_0000 {
        // The shader packs the index of the constraint group (a heavy atom
        // with its hydrogens) into 12 bits and the relative residual, in
        // parts per million, into 16: both saturate.
        let group = (value >> 16) & 0x0fff;
        let encoded = value & 0xffff;
        let residual = encoded as f32 / 1_000_000.0;
        let kind = if constraint_kind == 0x5000_0000 {
            "position"
        } else {
            "velocity"
        };
        let group = if group == 0x0fff {
            "4095 or beyond".to_string()
        } else {
            group.to_string()
        };
        let residual = if encoded == 0xffff {
            format!("{residual:.1e} or more")
        } else {
            format!("{residual:.1e}")
        };
        return Some(format!(
            "GPU solute {kind} constraints failed in constraint group {group} (relative error {residual})"
        ));
    }
    match value {
        0 => None,
        1 => Some("SETTLE numerical branch".into()),
        2 => Some("GPU pair-buffer capacity".into()),
        3 => Some("GPU neighbor-list capacity".into()),
        5 => Some("GPU solute position-constraint projection failed to converge".into()),
        6 => Some("GPU solute velocity-constraint projection failed to converge".into()),
        7 => Some("GPU SETTLE position projection failed its geometry check".into()),
        8 => Some("GPU SETTLE velocity projection failed its constraint check".into()),
        9 => Some("the box shrank below twice the cutoff".into()),
        10 => Some("the barostat changed the box by more than 10% in one coupling step".into()),
        _ => Some("GPU dynamics numerical error".into()),
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
    solute_constraint_component_offset: u32,
    solute_constraint_component_count: u32,
    tiled_nonbonded: bool,
    kernel: PbcKernel,
    tiles: Option<TileEngine>,
    leapfrog: Option<LeapfrogEngine>,
    /// PME: self energy (kcal/mol) and the coefficient of the uniform
    /// background energy of a charged box (kcal Å³/mol, over the volume).
    ewald_constants: Option<[f64; 2]>,
    box_xyz: std::sync::Mutex<[f32; 3]>,
    gpu_stage_timings_ms: std::sync::Mutex<BTreeMap<String, f64>>,
    dt_ps: f32,
    params: Vec<[f32; 4]>,
    srange_image: Vec<u32>,
    _allocation: AllocationReservation,
}

fn workgroups(n: u32) -> u32 {
    n.div_ceil(64).max(1)
}

#[cfg(not(target_arch = "wasm32"))]
fn pbc_stage_name(pipeline: usize) -> &'static str {
    match pipeline {
        0 => "neighborInsert",
        1 => "neighborCount",
        2 => "nonbondedForces",
        3 => "reductions",
        4 | 9 => "bondedForces",
        6..=8 | SETTLE_VIRIAL => "constraints",
        10 => "neighborClear",
        14 => "neighborCheck",
        15 => "neighborScan",
        16 => "neighborBlockScan",
        17 => "neighborOffsetApply",
        18 => "neighborFill",
        19 => "neighborSort",
        20 => "neighborFinish",
        25 => "neighborDispatchPrepare",
        26 => "neighborSort",
        27 => "neighborFillFixed",
        28 => "neighborSortFixed",
        29 => "dynamicsScalarReduction",
        _ => "integration",
    }
}

fn solute_constraint_components(
    edges: &[[u32; 2]],
    atom_count: usize,
) -> Result<Vec<Vec<usize>>, Error> {
    let mut atom_edges = vec![Vec::<usize>::new(); atom_count];
    for (edge_index, [a, b]) in edges.iter().copied().enumerate() {
        let (a, b) = (a as usize, b as usize);
        if a >= atom_count || b >= atom_count {
            return Err(Error::Input("solute constraint atom index"));
        }
        atom_edges[a].push(edge_index);
        atom_edges[b].push(edge_index);
    }
    let mut visited = vec![false; edges.len()];
    let mut components = Vec::new();
    for first in 0..edges.len() {
        if visited[first] {
            continue;
        }
        let mut stack = vec![first];
        let mut component = Vec::new();
        while let Some(edge_index) = stack.pop() {
            if visited[edge_index] {
                continue;
            }
            visited[edge_index] = true;
            component.push(edge_index);
            for atom in edges[edge_index] {
                for &neighbor in &atom_edges[atom as usize] {
                    if !visited[neighbor] {
                        stack.push(neighbor);
                    }
                }
            }
        }
        component.sort_unstable();
        if component.len() > MAX_SOLUTE_CONSTRAINT_COMPONENT_BONDS {
            return Err(Error::Input(
                "GPU solute constraint component exceeds eight bonds",
            ));
        }
        components.push(component);
    }
    Ok(components)
}

#[cfg(test)]
mod constraint_component_tests {
    use super::solute_constraint_components;

    #[test]
    fn shares_edges_only_within_connected_components_and_preserves_order() {
        let components =
            solute_constraint_components(&[[0, 1], [0, 2], [3, 4], [5, 6], [5, 7]], 8).unwrap();
        assert_eq!(components, vec![vec![0, 1], vec![2], vec![3, 4]]);
    }

    #[test]
    fn rejects_oversized_components_for_the_gpu_solver() {
        let edges: Vec<_> = (1..=9).map(|atom| [0, atom]).collect();
        assert!(solute_constraint_components(&edges, 10).is_err());
    }
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
        if self.tiles.is_some() {
            return true;
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
        Self::with_context_kernel(context, packing, backend, max_pairs, PbcKernel::Tiles).await
    }

    /// Construct a periodic evaluator and optionally select the cooperative
    /// fixed-row pair kernel instead of the default implementation.
    pub async fn with_context_variant(
        context: &GpuContext,
        packing: &PbcPacking,
        backend: &NonbondedElectrostatics,
        max_pairs: u32,
        tiled_nonbonded: bool,
    ) -> Result<Self, Error> {
        let kernel = if tiled_nonbonded {
            PbcKernel::FixedRows
        } else {
            PbcKernel::Tiles
        };
        Self::with_context_kernel(context, packing, backend, max_pairs, kernel).await
    }

    /// Construct a periodic evaluator with an explicit nonbonded kernel.
    /// `max_pairs` bounds the historical neighbor lists and is ignored by
    /// [`PbcKernel::Tiles`], which sizes its tile lists from the packing.
    pub async fn with_context_kernel(
        context: &GpuContext,
        packing: &PbcPacking,
        backend: &NonbondedElectrostatics,
        max_pairs: u32,
        kernel: PbcKernel,
    ) -> Result<Self, Error> {
        Self::with_budget_context(
            context,
            packing,
            backend,
            max_pairs,
            context.memory_profile().budget(),
            kernel,
        )
        .await
    }

    /// The nonbonded kernel this evaluator runs.
    pub fn kernel(&self) -> PbcKernel {
        self.kernel
    }

    async fn with_budget_context(
        context: &GpuContext,
        packing: &PbcPacking,
        backend: &NonbondedElectrostatics,
        max_pairs: u32,
        budget: u64,
        kernel: PbcKernel,
    ) -> Result<Self, Error> {
        let tiled_nonbonded = kernel == PbcKernel::FixedRows;
        let (electro, ewald) = electrostatics_uniform(backend, packing.cutoff)?;
        // The packing cutoff and the backend cutoff must agree: cells are
        // sized for one limit, physics evaluated at one cutoff.
        if (electro[1] as f64 - packing.cutoff).abs() > 1e-6 {
            return Err(Error::Input("backend cutoff must match packing cutoff"));
        }
        if ewald.is_some() && kernel != PbcKernel::Tiles {
            return Err(Error::Input("GPU PME needs the tiled pair kernel"));
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
        let pairs_bytes = match kernel {
            PbcKernel::FixedRows => n64
                .checked_mul(u64::from(TILED_NEIGHBORS_PER_ATOM))
                .and_then(|words| words.checked_mul(4))
                .ok_or(Error::Capacity)?,
            PbcKernel::Csr => u64::from(max_pairs).checked_mul(8).ok_or(Error::Capacity)?,
            // The historical pair kernels are never dispatched; keep a
            // minimal buffer so the shared bind group stays valid.
            PbcKernel::Tiles => 64,
        };
        let tile_plan = if kernel == PbcKernel::Tiles {
            let tile_packing = TilePacking::new_for(packing, ewald.is_some())?;
            let shrink = packing.tile_density_headroom.max(1.0).cbrt();
            let sizing =
                TileSizing::new(n, packing.box_angstrom.map(|b| b / shrink), packing.limit)?;
            if n > 4_000_000 {
                return Err(Error::Capacity);
            }
            Some((tile_packing, sizing))
        } else {
            None
        };
        let tile_bytes = tile_plan.as_ref().map_or(0, |(tile_packing, sizing)| {
            TileEngine::allocation_bytes(sizing, tile_packing)
        });
        let sys_bytes = n64.checked_mul(32).ok_or(Error::Capacity)?;
        // Cell heads, links, two range words/atom, pair counter, and a
        // single atomic numerical-status flag.
        let meta_words = u64::from(ncells)
            .checked_add(8u64.checked_mul(n64).ok_or(Error::Capacity)?)
            .and_then(|v| v.checked_add(5))
            .and_then(|v| v.checked_add(u64::from(workgroups(n))))
            .and_then(|v| v.checked_add(30))
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
            .and_then(|v| v.checked_add(tile_bytes))
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
            || if tiled_nonbonded {
                n > limits.max_compute_workgroups_per_dimension
            } else {
                workgroups(n) > limits.max_compute_workgroups_per_dimension
            }
        {
            return Err(Error::Capacity);
        }
        let reservation =
            context.reserve(heap.checked_add(3 * staging_bytes).ok_or(Error::Capacity)?)?;
        crate::push_error_scope(&device, wgpu::ErrorFilter::OutOfMemory);
        crate::push_error_scope(&device, wgpu::ErrorFilter::Validation);
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
            // A barostat kernel rewrites the box through a storage binding.
            buffer(
                "pbc config",
                128,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | storage,
            ),
            buffer(
                "pbc sys",
                sys_bytes,
                storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            ),
            buffer(
                "pbc meta",
                meta_bytes,
                storage
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::INDIRECT,
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
        let mut pipelines: Vec<wgpu::ComputePipeline> = [
            "insert_atoms",
            "count_neighbors",
            if tiled_nonbonded {
                "eval_tiled_fixed"
            } else {
                "eval"
            },
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
            "lf_full_kick",
            "lf_first_half_drift",
            "lf_save_start",
            "lf_second_half_drift",
            "prepare_neighbor_dispatch",
            "sort_neighbors_tiled",
            "fill_neighbors_fixed",
            "sort_neighbors_fixed",
            "reduce_dynamics_scalars",
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
        debug_assert_eq!(pipelines.len(), SETTLE_VIRIAL);
        pipelines.push(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("settle with constraint virial"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("settle"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("CONSTRAINT_VIRIAL", 1.0)],
                    ..Default::default()
                },
                cache: None,
            }),
        );
        for entry in [
            "insert_atoms",
            "count_neighbors",
            if tiled_nonbonded {
                "eval_tiled_fixed"
            } else {
                "eval"
            },
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
            "lf_full_kick",
            "lf_first_half_drift",
            "lf_save_start",
            "lf_second_half_drift",
            "prepare_neighbor_dispatch",
            "sort_neighbors_tiled",
            "fill_neighbors_fixed",
            "sort_neighbors_fixed",
            "reduce_dynamics_scalars",
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
        let tiles = tile_plan.map(|(tile_packing, sizing)| {
            let status_word = ncells + 3 * n + 1;
            for (kernel, _) in crate::pbc_tiles::KERNELS {
                context.record_pipeline(format!("pbc_tiles.{kernel}"));
            }
            TileEngine::new(
                &device,
                &queue,
                sizing,
                &tile_packing,
                SharedBuffers {
                    sys: &buffers[1],
                    aux: &buffers[2],
                    specials: &buffers[3],
                    out: &buffers[5],
                    bcoords: &buffers[7],
                },
                electro,
                ewald,
                packing.limit as f32,
                status_word,
                status_word + 1,
            )
        });
        let validation = crate::pop_error_scope(&device).await;
        let allocation = crate::pop_error_scope(&device).await;
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
            solute_constraint_component_offset: packing.solute_constraint_component_offset,
            solute_constraint_component_count: packing.solute_constraint_component_count,
            tiled_nonbonded,
            kernel,
            tiles,
            leapfrog: None,
            ewald_constants: ewald.map(|ewald| {
                let alpha = ewald.alpha_per_angstrom;
                let charges = packing.params.iter().map(|p| f64::from(p[0]));
                let squares: f64 = charges.clone().map(|q| q * q).sum();
                let net: f64 = charges.sum();
                const COULOMB: f64 = 332.063713299;
                [
                    -COULOMB * alpha / std::f64::consts::PI.sqrt() * squares,
                    -COULOMB * std::f64::consts::PI * net * net / (2.0 * alpha * alpha),
                ]
            }),
            box_xyz: std::sync::Mutex::new(packing.box_angstrom.map(|v| v as f32)),
            gpu_stage_timings_ms: std::sync::Mutex::new(BTreeMap::new()),
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

    fn eval_dispatch_groups(&self) -> u32 {
        if self.tiled_nonbonded {
            self.n
        } else {
            workgroups(self.n)
        }
    }

    /// Upload interleaved params/coords, static specials ranges, a cleared
    /// cell-head table, and a zeroed pair counter. The uniform (dims, box,
    /// electrostatics, flags) is rewritten every call: box and coordinates
    /// change per frame, so nothing is cached across calls.
    pub fn set_coordinates(&self, unwrapped: &[Vec3], box_xyz: [f32; 3], gradients: bool) {
        assert_eq!(unwrapped.len() as u32, self.n);
        self.set_coordinates_from(box_xyz, gradients, true, |i| {
            [unwrapped[i].x, unwrapped[i].y, unwrapped[i].z]
        });
    }

    /// Upload f64 xyz coordinates without first materializing a `Vec<Vec3>`.
    /// This is used by the minimizer, whose working coordinates are already a
    /// flat slice, to avoid one large allocation and an extra full-system copy
    /// on every energy/gradient evaluation.
    pub fn set_flat_coordinates_f64(&self, unwrapped: &[f64], box_xyz: [f32; 3], gradients: bool) {
        assert_eq!(unwrapped.len(), self.n as usize * 3);
        self.set_coordinates_from(box_xyz, gradients, true, |i| {
            let offset = 3 * i;
            [
                unwrapped[offset],
                unwrapped[offset + 1],
                unwrapped[offset + 2],
            ]
        });
    }

    /// Upload minimization trial coordinates while retaining the cached
    /// Verlet list. The subsequent force evaluation checks displacement
    /// against the list's reference coordinates and rebuilds when the skin is
    /// crossed. Use only after an initial successful force evaluation on this
    /// evaluator, with the same topology and periodic box.
    pub fn set_flat_coordinates_f64_reusing_neighbors(
        &self,
        unwrapped: &[f64],
        box_xyz: [f32; 3],
        gradients: bool,
    ) {
        assert_eq!(unwrapped.len(), self.n as usize * 3);
        self.set_coordinates_from(box_xyz, gradients, false, |i| {
            let offset = 3 * i;
            [
                unwrapped[offset],
                unwrapped[offset + 1],
                unwrapped[offset + 2],
            ]
        });
    }

    fn set_coordinates_from(
        &self,
        box_xyz: [f32; 3],
        gradients: bool,
        reset_neighbor_search: bool,
        mut coordinate: impl FnMut(usize) -> [f64; 3],
    ) {
        if let Ok(mut current) = self.box_xyz.lock() {
            *current = box_xyz;
        }
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
            f32::from_bits(self.solute_constraint_component_offset),
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
        let thermo = [
            300.0f32,
            1.0,
            self.adjacency_offset as f32,
            f32::from_bits(self.solute_constraint_component_count),
        ];
        for (i, v) in thermo.iter().enumerate() {
            config[96 + 4 * i..100 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        self.queue.write_buffer(&self.buffers[0], 0, &config);
        let atom_count = self.n as usize;
        let mut sys = Vec::with_capacity(atom_count * 2);
        for (index, p) in self.params.iter().enumerate() {
            let c = coordinate(index);
            sys.push(*p);
            // Centering before the f32 cast reduces cancellation in bonded
            // coordinate differences. It is a rigid translation: GPU cells
            // wrap the centered coordinate and minimum image is unchanged.
            sys.push([
                (c[0] - 0.5 * box_xyz[0] as f64) as f32,
                (c[1] - 0.5 * box_xyz[1] as f64) as f32,
                (c[2] - 0.5 * box_xyz[2] as f64) as f32,
                0.,
            ]);
        }
        self.queue
            .write_buffer(&self.buffers[1], 0, bytemuck::cast_slice(&sys));
        let mut bonded_coords = vec![[0f32; 4]; atom_count];
        for group in &self.molecules {
            let mut center = [0f64; 3];
            for &atom in group {
                let c = coordinate(atom);
                center[0] += c[0];
                center[1] += c[1];
                center[2] += c[2];
            }
            let count = group.len() as f64;
            let anchor = group[0];
            for &atom in group {
                let c = coordinate(atom);
                bonded_coords[atom] = [
                    (c[0] - center[0] / count) as f32,
                    (c[1] - center[1] / count) as f32,
                    (c[2] - center[2] / count) as f32,
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
        // A coordinate upload starts a fresh evaluation session.  Dynamics
        // itself deliberately leaves the numerical-status word sticky until
        // the host polls it at a bounded checkpoint.
        self.queue
            .write_buffer(&self.buffers[2], self.meta_status_off(), &[0; 4]);
        if let Some(tiles) = &self.tiles {
            tiles.upload(&self.queue, box_xyz, reset_neighbor_search);
        }
        if reset_neighbor_search {
            self.queue
                .write_buffer(&self.buffers[2], 0, &vec![0xFFu8; self.ncells as usize * 4]);
            self.queue
                .write_buffer(&self.buffers[2], self.meta_count_off(), &[0; 4]);
            self.queue.write_buffer(
                &self.buffers[2],
                self.meta_status_off() + 4,
                &1u32.to_le_bytes(),
            );
        }
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

    /// Initialize a fresh resident run with the stochastic stream already
    /// recorded in its step-zero checkpoint.  The convenience initializer
    /// above retains its historical per-atom seed convention for direct GPU
    /// callers; runtime-managed simulations must use this method so a fresh
    /// run and a resume from step zero start from the same RNG state.
    pub fn initialize_dynamics_with_rng(
        &mut self,
        coordinates: &[Vec3],
        velocities: &[Vec3],
        box_xyz: [f32; 3],
        dt_ps: f32,
        rng_words: &[u32],
    ) -> Result<(), Error> {
        self.set_timestep(dt_ps)?;
        self.set_coordinates(coordinates, box_xyz, true);
        self.set_velocities_with_rng(velocities, rng_words)
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
    /// transfer have been observed to stall. A chain is a static evaluation:
    /// tiled force passes include every bonded term and publish energies.
    fn dispatch_chain(&self, jobs: &[(usize, u32)]) {
        self.dispatch_repeated(jobs, 1, true, true);
    }

    /// Encode `steps` repetitions of `jobs` and submit them together. With
    /// the tiled engine only the final repetition evaluates energies (when
    /// `final_energy`), and dynamics skips bonded terms held rigid by
    /// constraints unless `all_terms`.
    fn dispatch_repeated(
        &self,
        jobs: &[(usize, u32)],
        steps: usize,
        final_energy: bool,
        all_terms: bool,
    ) -> wgpu::SubmissionIndex {
        let mut variants: [Option<Vec<(usize, u32)>>; 4] = Default::default();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // One pass per submission: wgpu still orders every dispatch that
        // touches a writable storage buffer, without per-pass overhead.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        for step in 0..steps {
            let energy = final_energy && step + 1 == steps;
            let resort = self.tiles.as_ref().is_some_and(TileEngine::take_resort);
            let variant = usize::from(energy) + 2 * usize::from(resort);
            let expanded = variants[variant]
                .get_or_insert_with(|| self.expanded_jobs(jobs, energy, all_terms, resort));
            for &(pipeline, groups) in expanded.iter() {
                self.encode_job(&mut pass, pipeline, groups);
            }
        }
        drop(pass);
        self.queue.submit(Some(encoder.finish()))
    }

    fn encode_job(&self, pass: &mut wgpu::ComputePass<'_>, pipeline: usize, groups: u32) {
        if pipeline >= LEAPFROG_JOB {
            let engine = self
                .leapfrog
                .as_ref()
                .expect("leap-frog jobs are only planned for a configured integrator");
            let (compute, bind_group) =
                engine.kernel(LeapfrogKernel::from_job(pipeline - LEAPFROG_JOB));
            pass.set_pipeline(compute);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            return;
        }
        if pipeline >= TILE_JOB {
            let tiles = self
                .tiles
                .as_ref()
                .expect("tiled jobs are only expanded for the tiled engine");
            let kernel = TileKernel::from_job(pipeline - TILE_JOB);
            let (compute, bind_group) = tiles.kernel(kernel);
            pass.set_pipeline(compute);
            pass.set_bind_group(0, bind_group, &[]);
            if let Some(stage) = kernel.rebuild_stage() {
                pass.dispatch_workgroups_indirect(&tiles.args, 12 * stage);
            } else {
                let (x, y) = crate::pbc_tiles::wide_groups(groups);
                pass.dispatch_workgroups(x, y, 1);
            }
            return;
        }
        pass.set_pipeline(&self.pipelines[pipeline]);
        pass.set_bind_group(0, &self.bind_group, &[]);
        if groups == INDIRECT_NEIGHBOR_DISPATCH {
            pass.dispatch_workgroups_indirect(
                &self.buffers[2],
                self.neighbor_indirect_offset(pipeline),
            );
        } else {
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn stage_name(&self, pipeline: usize) -> &'static str {
        if pipeline >= LEAPFROG_JOB {
            LeapfrogKernel::from_job(pipeline - LEAPFROG_JOB).stage_name()
        } else if pipeline >= TILE_JOB {
            TileKernel::from_job(pipeline - TILE_JOB).stage_name()
        } else {
            pbc_stage_name(pipeline)
        }
    }

    fn tile_job(&self, kernel: TileKernel) -> (usize, u32) {
        let tiles = self.tiles.as_ref().expect("tiled engine");
        let groups = if kernel.rebuild_stage().is_some() {
            INDIRECT_NEIGHBOR_DISPATCH
        } else {
            tiles.groups(kernel)
        };
        (TILE_JOB + kernel.job(), groups)
    }

    fn expanded_jobs(
        &self,
        jobs: &[(usize, u32)],
        energy: bool,
        all_terms: bool,
        resort: bool,
    ) -> Vec<(usize, u32)> {
        let mut expanded = Vec::with_capacity(jobs.len() + 12);
        if self.tiles.is_some() {
            for &(pipeline, groups) in jobs {
                match pipeline {
                    0 => {
                        if resort {
                            for kernel in TileKernel::RESORT {
                                expanded.push(self.tile_job(kernel));
                            }
                        }
                        for kernel in TileKernel::REBUILD {
                            expanded.push(self.tile_job(kernel));
                        }
                    }
                    2 => expanded.push(self.tile_job(if energy {
                        TileKernel::PairEnergy
                    } else {
                        TileKernel::PairForces
                    })),
                    4 => {
                        expanded.push(self.tile_job(match (all_terms, energy) {
                            (true, true) => TileKernel::BondedAllEnergy,
                            (true, false) => TileKernel::BondedAllForces,
                            (false, true) => TileKernel::BondedDynamicsEnergy,
                            (false, false) => TileKernel::BondedDynamicsForces,
                        }));
                        expanded.push(self.tile_job(TileKernel::Finalize));
                    }
                    // Cell clearing and per-atom partial reductions belong to
                    // the historical kernels; the tiled engine publishes
                    // totals from its finalize pass.
                    3 | 10 => {}
                    _ => expanded.push((pipeline, groups)),
                }
            }
            return expanded;
        }
        for &(pipeline, groups) in jobs {
            if pipeline == 0 {
                let n = workgroups(self.n);
                expanded.extend_from_slice(&[
                    (14, n),
                    (25, 1),
                    (10, INDIRECT_NEIGHBOR_DISPATCH),
                    (0, INDIRECT_NEIGHBOR_DISPATCH),
                ]);
                if self.tiled_nonbonded {
                    expanded.extend_from_slice(&[
                        (27, INDIRECT_NEIGHBOR_DISPATCH),
                        (28, INDIRECT_NEIGHBOR_DISPATCH),
                        (20, INDIRECT_NEIGHBOR_DISPATCH),
                    ]);
                } else {
                    expanded.extend_from_slice(&[
                        (1, INDIRECT_NEIGHBOR_DISPATCH),
                        (15, INDIRECT_NEIGHBOR_DISPATCH),
                        (16, INDIRECT_NEIGHBOR_DISPATCH),
                        (17, INDIRECT_NEIGHBOR_DISPATCH),
                        (18, INDIRECT_NEIGHBOR_DISPATCH),
                        (19, INDIRECT_NEIGHBOR_DISPATCH),
                        (20, INDIRECT_NEIGHBOR_DISPATCH),
                    ]);
                }
            } else if pipeline != 10 {
                expanded.push((pipeline, groups));
            }
        }
        // WebGPU forbids using one buffer as both a writable storage binding
        // and indirect-dispatch arguments in the same synchronization scope.
        // The historical neighbor dispatch arguments live in `aux`, which is
        // also the PBC metadata storage binding, so these kernels use bounded
        // direct dispatches on every target; each stage already checks the
        // rebuild/status flags. (Native wgpu used to hide the conflict behind
        // its indirect-validation copy, which the context now disables.) The
        // tiled engine keeps its arguments in a separate buffer.
        for (pipeline, groups) in &mut expanded {
            if *groups != INDIRECT_NEIGHBOR_DISPATCH {
                continue;
            }
            *groups = match *pipeline {
                10 => workgroups(self.n.max(self.ncells)),
                16 | 20 => 1,
                26 | 28 => self.n,
                _ => workgroups(self.n),
            };
        }
        expanded
    }

    fn neighbor_indirect_offset(&self, pipeline: usize) -> u64 {
        let slot = match pipeline {
            10 => 0u64,
            0 => 1,
            1 => 2,
            15 => 3,
            16 => 4,
            17 => 5,
            18 => 6,
            19 => 7,
            20 => 8,
            26 => 9,
            27 => 6,
            28 => 9,
            _ => unreachable!("only neighbor rebuild passes use indirect dispatch"),
        };
        let base =
            u64::from(self.ncells) + 8 * u64::from(self.n) + 5 + u64::from(workgroups(self.n));
        (base + 3 * slot) * std::mem::size_of::<u32>() as u64
    }

    pub fn take_gpu_stage_timings_ms(&self) -> BTreeMap<String, f64> {
        self.gpu_stage_timings_ms
            .lock()
            .map(|mut timings| std::mem::take(&mut *timings))
            .unwrap_or_default()
    }

    /// Submit a dynamics advance in bounded packets. Up to two packets are in
    /// flight so command encoding overlaps device execution; only the final
    /// step of the advance publishes energies for the observers.
    async fn dispatch_dynamics_bounded(
        &self,
        jobs: &[(usize, u32)],
        steps: usize,
    ) -> Result<(), Error> {
        let mut remaining = steps;
        #[cfg(not(target_arch = "wasm32"))]
        let mut previous: Option<wgpu::SubmissionIndex> = None;
        while remaining > 0 {
            let batch = remaining.min(MAX_ENCODED_DYNAMICS_STEPS);
            let last = batch == remaining;
            #[cfg(not(target_arch = "wasm32"))]
            if self._context.gpu_timestamps_enabled() {
                self.dispatch_repeated_profiled(jobs, batch, last).await?;
            } else {
                let index = self.dispatch_repeated(jobs, batch, last, true);
                if let Some(earlier) = previous.replace(index) {
                    self.device
                        .poll(wgpu::PollType::WaitForSubmissionIndex(earlier))
                        .map_err(|e| Error::Execution(e.to_string()))?;
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                self.dispatch_repeated(jobs, batch, last, true);
            }
            remaining -= batch;
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn dispatch_repeated_profiled(
        &self,
        jobs: &[(usize, u32)],
        steps: usize,
        final_energy: bool,
    ) -> Result<(), Error> {
        let resorts: Vec<bool> = (0..steps)
            .map(|_| self.tiles.as_ref().is_some_and(TileEngine::take_resort))
            .collect();
        let plans: Vec<Vec<(usize, u32)>> = resorts
            .iter()
            .enumerate()
            .map(|(step, &resort)| {
                self.expanded_jobs(jobs, final_energy && step + 1 == steps, true, resort)
            })
            .collect();
        self.dispatch_plans_profiled(&plans).await
    }

    /// Encode every job of `plans` in its own timestamped pass, submit, and
    /// add the per-stage device times to the accumulator.
    #[cfg(not(target_arch = "wasm32"))]
    async fn dispatch_plans_profiled(&self, plans: &[Vec<(usize, u32)>]) -> Result<(), Error> {
        let query_count = plans
            .iter()
            .map(Vec::len)
            .sum::<usize>()
            .checked_mul(2)
            .and_then(|queries| u32::try_from(queries).ok())
            .filter(|count| *count > 0)
            .ok_or(Error::Capacity)?;
        let byte_size = u64::from(query_count) * std::mem::size_of::<u64>() as u64;
        let query_set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GlySys explicit dynamics stage timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: query_count,
        });
        let resolved = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlySys timestamp resolve"),
            size: byte_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlySys timestamp readback"),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GlySys profiled explicit dynamics"),
            });
        let mut query_index = 0u32;
        let mut stages = Vec::with_capacity(query_count as usize / 2);
        for expanded in plans {
            for &(pipeline, groups) in expanded {
                let begin = query_index;
                let end = begin + 1;
                query_index += 2;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GlySys profiled PBC dynamics stage"),
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &query_set,
                        beginning_of_pass_write_index: Some(begin),
                        end_of_pass_write_index: Some(end),
                    }),
                });
                self.encode_job(&mut pass, pipeline, groups);
                drop(pass);
                stages.push((self.stage_name(pipeline), begin, end));
            }
        }
        encoder.resolve_query_set(&query_set, 0..query_count, &resolved, 0);
        encoder.copy_buffer_to_buffer(&resolved, 0, &readback, 0, byte_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(0..byte_size);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait)
            .map_err(|error| Error::Execution(error.to_string()))?;
        rx.await
            .map_err(|error| Error::Execution(error.to_string()))?
            .map_err(|error| Error::Execution(error.to_string()))?;
        let mapped = slice.get_mapped_range();
        let values: &[u64] = bytemuck::cast_slice(&mapped);
        let period_ns = self
            ._context
            .gpu_timestamp_period_ns()
            .ok_or_else(|| Error::Execution("GPU timestamp period is unavailable".into()))?;
        let mut elapsed_ms = BTreeMap::<String, f64>::new();
        for (stage, begin, end) in stages {
            let duration_ns =
                values[end as usize].wrapping_sub(values[begin as usize]) as f64 * period_ns;
            *elapsed_ms.entry(stage.to_owned()).or_default() += duration_ns / 1_000_000.0;
        }
        drop(mapped);
        readback.unmap();
        let mut accumulated = self
            .gpu_stage_timings_ms
            .lock()
            .map_err(|_| Error::Execution("GPU timing accumulator was poisoned".into()))?;
        for (stage, milliseconds) in elapsed_ms {
            *accumulated.entry(stage).or_default() += milliseconds;
        }
        Ok(())
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
        let sources: Vec<_> = ranges
            .iter()
            .map(|&(source, offset, bytes)| (&self.buffers[source], offset, bytes))
            .collect();
        self.readback_from(&sources).await
    }

    async fn readback_buffer(
        &self,
        buffer: &wgpu::Buffer,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, Error> {
        self.readback_from(&[(buffer, offset, size)]).await
    }

    async fn readback_from(&self, ranges: &[(&wgpu::Buffer, u64, u64)]) -> Result<Vec<u8>, Error> {
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
            encoder.copy_buffer_to_buffer(source, offset, lease.buffer, destination, bytes);
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
                (2, self.meta_status_off(), 12),
            ])
            .await?;
        let status =
            u32::from_le_bytes(bytes[bytes.len() - 12..bytes.len() - 8].try_into().unwrap());
        if let Some(error) = dynamics_error(status) {
            return Err(Error::Execution(error));
        }
        let n = n as usize;
        let mut state = self.decode_checkpoint(&bytes[..2 * n], &bytes[2 * n..3 * n], box_xyz);
        state.neighbor_rebuild_count =
            u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap()) as u64;
        let energy = self.with_ewald_constants(Self::decode_observables(
            &bytes[5 * n..5 * n + 32],
            Some(&bytes[3 * n..4 * n]),
            &bytes[4 * n..5 * n],
        ));
        Ok((state, energy))
    }

    /// Reduce potential and kinetic energies on-device, then transfer only
    /// those two scalars, the status word, and the neighbor rebuild counter.
    pub async fn read_dynamics_scalars(&self) -> Result<DynamicsScalars, Error> {
        self.dispatch_chain(&[(29, 1)]);
        let scalar_offset = (4 * u64::from(self.n) + 2) * 16;
        let bytes = self
            .readback_ranges(&[(5, scalar_offset, 16), (2, self.meta_status_off(), 12)])
            .await?;
        let status = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        if let Some(error) = dynamics_error(status) {
            return Err(Error::Execution(error));
        }
        let values: &[f32] = bytemuck::cast_slice(&bytes[..16]);
        if !values[0].is_finite() || !values[1].is_finite() {
            return Err(Error::Nonfinite);
        }
        Ok(DynamicsScalars {
            potential_energy: values[0] as f64,
            kinetic_energy: values[1] as f64,
            neighbor_rebuild_count: u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as u64,
            readback_bytes: 28,
        })
    }

    /// Build cells and enumerate pairs within cutoff+skin.
    pub async fn neighbor_list(&self) -> Result<NeighborResult, Error> {
        if self.tiles.is_some() {
            return self.tile_neighbor_list().await;
        }
        self.queue
            .write_buffer(&self.buffers[2], self.meta_status_off(), &[0; 4]);
        crate::push_error_scope(&self.device, wgpu::ErrorFilter::Validation);
        self.dispatch_chain(&[(0, workgroups(self.n))]);
        let count_bytes = self.readback(2, self.meta_count_off(), 4).await?;
        let directed = u32::from_le_bytes(count_bytes[0..4].try_into().unwrap());
        let status_bytes = self.readback(2, self.meta_status_off(), 4).await?;
        let status = u32::from_le_bytes(status_bytes[0..4].try_into().unwrap());
        if let Some(e) = crate::pop_error_scope(&self.device).await {
            return Err(Error::Execution(e.to_string()));
        }
        if status != 0 {
            return Err(if status == 2 || status == 3 {
                Error::Capacity
            } else {
                Error::Execution(
                    dynamics_error(status).unwrap_or_else(|| "GPU neighbor-list failure".into()),
                )
            });
        }
        if directed > 2 * self.max_pairs {
            return Err(Error::Capacity);
        }
        let pair_bytes = if self.tiled_nonbonded {
            self.readback(
                4,
                0,
                u64::from(self.n) * u64::from(TILED_NEIGHBORS_PER_ATOM) * 4,
            )
            .await?
        } else if directed > 0 {
            self.readback(4, 0, u64::from(directed) * 4).await?
        } else {
            Vec::new()
        };
        if self.tiled_nonbonded {
            let count_bytes = self
                .readback(
                    2,
                    (u64::from(self.ncells) + 6 * u64::from(self.n) + 4) * 4,
                    u64::from(self.n) * 4,
                )
                .await?;
            let counts: &[u32] = bytemuck::cast_slice(&count_bytes);
            let indices: &[u32] = bytemuck::cast_slice(&pair_bytes);
            let mut pairs = Vec::new();
            for a in 0..self.n as usize {
                let start = a * TILED_NEIGHBORS_PER_ATOM as usize;
                for &b in &indices[start..start + counts[a] as usize] {
                    if b as usize > a {
                        pairs.push((a as u32, b));
                    }
                }
            }
            pairs.sort_unstable();
            return Ok(NeighborResult {
                count: pairs.len() as u32,
                pairs,
            });
        }
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

    /// Pairs within cutoff + skin covered by the tiled engine's current
    /// lists, recovered from the tiles and filtered by the same f32
    /// minimum-image distance the list build uses. Exclusion masks are
    /// ignored so the result matches the historical list semantics.
    async fn tile_neighbor_list(&self) -> Result<NeighborResult, Error> {
        let tiles = self.tiles.as_ref().expect("tiled engine");
        self.queue
            .write_buffer(&self.buffers[2], self.meta_status_off(), &[0; 4]);
        crate::push_error_scope(&self.device, wgpu::ErrorFilter::Validation);
        self.dispatch_chain(&[(0, workgroups(self.n))]);
        let status_bytes = self.readback(2, self.meta_status_off(), 4).await?;
        let status = u32::from_le_bytes(status_bytes[0..4].try_into().unwrap());
        if let Some(e) = crate::pop_error_scope(&self.device).await {
            return Err(Error::Execution(e.to_string()));
        }
        if status != 0 {
            return Err(if status == 2 || status == 3 {
                Error::Capacity
            } else {
                Error::Execution(
                    dynamics_error(status).unwrap_or_else(|| "GPU neighbor-list failure".into()),
                )
            });
        }
        let layout = tiles.sizing.layout;
        let blocks = u64::from(tiles.sizing.blocks);
        let order_bytes = self
            .readback_work(u64::from(layout.order) * 4, blocks * 32 * 4)
            .await?;
        let tile_bytes = self
            .readback_work(u64::from(layout.block_tiles) * 4, blocks * 2 * 4)
            .await?;
        let order: Vec<u32> = bytemuck::cast_slice(&order_bytes).to_vec();
        let block_tiles: Vec<u32> = bytemuck::cast_slice(&tile_bytes).to_vec();
        let used = tiles.sizing.capacity;
        let atom_bytes = self
            .readback_work(u64::from(layout.tile_atoms) * 4, u64::from(used) * 32 * 4)
            .await?;
        let tile_atoms: &[u32] = bytemuck::cast_slice(&atom_bytes);
        let sys_bytes = self.readback(1, 0, u64::from(self.n) * 32).await?;
        let sys: &[f32] = bytemuck::cast_slice(&sys_bytes);
        let box_xyz = *self.box_xyz.lock().map_err(|_| Error::Input("box lock"))?;
        let limit2 = self.limit * self.limit;
        let position = |atom: u32| {
            let i = 8 * atom as usize + 4;
            [sys[i], sys[i + 1], sys[i + 2]]
        };
        let within = |a: u32, b: u32| {
            let (pa, pb) = (position(a), position(b));
            let mut r2 = 0.0f32;
            for axis in 0..3 {
                let d = pa[axis] - pb[axis];
                let d = d - box_xyz[axis] * (d / box_xyz[axis]).round_ties_even();
                r2 += d * d;
            }
            r2 <= limit2
        };
        let mut pairs = Vec::new();
        for (block, range) in block_tiles.chunks_exact(2).enumerate() {
            for tile in range[0]..range[0] + range[1] {
                for row in 0..32usize {
                    let a = order[32 * block + row];
                    if a == u32::MAX {
                        continue;
                    }
                    for slot in 0..32usize {
                        let b = tile_atoms[32 * tile as usize + slot];
                        if b == u32::MAX || (tile == range[0] && slot <= row) {
                            continue;
                        }
                        if within(a, b) {
                            pairs.push((a.min(b), a.max(b)));
                        }
                    }
                }
            }
        }
        // Each pair is listed from both of its blocks.
        pairs.sort_unstable();
        pairs.dedup();
        Ok(NeighborResult {
            count: pairs.len() as u32,
            pairs,
        })
    }

    /// Tiled-engine list occupancy as `(tiles in use, tile capacity, blocks,
    /// largest block)` after the most recent rebuild; `None` for the
    /// historical kernels.
    pub async fn tile_occupancy(&self) -> Result<Option<(u32, u32, u32, u32)>, Error> {
        let Some(tiles) = &self.tiles else {
            return Ok(None);
        };
        let blocks = u64::from(tiles.sizing.blocks);
        let bytes = self
            .readback_work(u64::from(tiles.sizing.layout.block_tiles) * 4, blocks * 8)
            .await?;
        let ranges: &[u32] = bytemuck::cast_slice(&bytes);
        let used = ranges.chunks_exact(2).map(|range| range[1]).sum();
        let largest = ranges
            .chunks_exact(2)
            .map(|range| range[1])
            .max()
            .unwrap_or(0);
        Ok(Some((
            used,
            tiles.sizing.capacity,
            tiles.sizing.blocks,
            largest,
        )))
    }

    async fn readback_work(&self, offset: u64, size: u64) -> Result<Vec<u8>, Error> {
        let tiles = self.tiles.as_ref().expect("tiled engine");
        let mut out = Vec::with_capacity(usize::try_from(size).map_err(|_| Error::Capacity)?);
        let mut copied = 0;
        while copied < size {
            let chunk = (size - copied).min(self.staging_bytes);
            out.extend_from_slice(
                &self
                    .readback_buffer(&tiles.work, offset + copied, chunk)
                    .await?,
            );
            copied += chunk;
        }
        Ok(out)
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
        crate::push_error_scope(&self.device, wgpu::ErrorFilter::Validation);
        let n = workgroups(self.n);
        self.dispatch_chain(&[
            (10, workgroups(self.n.max(self.ncells))),
            (0, n),
            (2, self.eval_dispatch_groups()),
            (4, n),
            (3, 1),
        ]);
        let result = self.read_dynamics_observables(gradients).await;
        if let Some(e) = crate::pop_error_scope(&self.device).await {
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
        Ok(self.with_ewald_constants(Self::decode_observables(
            &total_bytes,
            grad_bytes.as_deref(),
            &partial_bytes,
        )))
    }

    /// Add the position-independent parts of the Ewald sum: the self energy
    /// and, for a charged box, the uniform background (energy and virial).
    fn with_ewald_constants(&self, mut result: EnergyResult) -> EnergyResult {
        if let Some([self_energy, background]) = self.ewald_constants {
            let volume = self
                .box_xyz
                .lock()
                .map(|b| f64::from(b[0]) * f64::from(b[1]) * f64::from(b[2]))
                .unwrap_or(f64::NAN);
            let background = background / volume;
            result.rf += self_energy + background;
            if let Some(virial) = &mut result.virial {
                *virial += 3.0 * background;
            }
        }
        result
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

    async fn reduce_energy_once(&self) -> Result<(), Error> {
        if self.tiles.is_some() {
            return Ok(());
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self._context.gpu_timestamps_enabled() {
            return self.dispatch_repeated_profiled(&[(3, 1)], 1, false).await;
        }
        // Force kernels overwrite the per-atom energy partials on every step,
        // but no integrator consumes the scalar reduction. Reduce only once
        // at the returned advance boundary, where snapshots may observe it.
        self.dispatch_chain(&[(3, 1)]);
        Ok(())
    }

    /// Encode a bounded sequence without intermediate host synchronization.
    /// Callers split at output and protocol boundaries before using this API.
    pub async fn dynamics_steps(&self, steps: usize) -> Result<(), Error> {
        if steps == 0 {
            return Err(Error::Input(
                "dynamics batch must contain at least one step",
            ));
        }
        if !self.dt_ps.is_finite() || self.dt_ps <= 0.0 {
            return Err(Error::Input("dynamics timestep is not configured"));
        }
        self.queue
            .write_buffer(&self.buffers[0], 100, &1.0f32.to_le_bytes());
        crate::push_error_scope(&self.device, wgpu::ErrorFilter::Validation);
        let groups = workgroups(self.n.max(self.ncells));
        self.dispatch_dynamics_bounded(
            &[
                (10, groups),                     // clear resident cells/counter
                (5, workgroups(self.n)),          // first half kick + drift
                (6, groups),                      // analytic SETTLE + SHAKE
                (9, workgroups(self.n)),          // update molecule-centered bonded data
                (0, workgroups(self.n)),          // cell insertion
                (2, self.eval_dispatch_groups()), // nonbonded force
                (4, workgroups(self.n)),          // bonded force/energy
                (7, workgroups(self.n)),          // second half kick
                (8, groups),                      // analytic water RATTLE + solute RATTLE
            ],
            steps,
        )
        .await?;
        self.reduce_energy_once().await?;
        if let Some(e) = crate::pop_error_scope(&self.device).await {
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
        if steps == 0 {
            return Err(Error::Input(
                "dynamics batch must contain at least one step",
            ));
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
        crate::push_error_scope(&self.device, wgpu::ErrorFilter::Validation);
        let groups = workgroups(self.n.max(self.ncells));
        self.dispatch_dynamics_bounded(
            &[
                (10, groups),
                (11, workgroups(self.n)),         // first A/2
                (6, groups),                      // SETTLE/SHAKE
                (12, workgroups(self.n)),         // OU thermostat
                (13, workgroups(self.n)),         // second A/2
                (6, groups),                      // SETTLE/SHAKE
                (9, workgroups(self.n)),          // bonded frame
                (0, workgroups(self.n)),          // cells
                (2, self.eval_dispatch_groups()), // forces
                (4, workgroups(self.n)),          // bonded forces
                (7, workgroups(self.n)),          // second B/2
                (8, groups),                      // RATTLE
            ],
            steps,
        )
        .await?;
        self.reduce_energy_once().await?;
        if let Some(e) = crate::pop_error_scope(&self.device).await {
            return Err(Error::Execution(e.to_string()));
        }
        Ok(())
    }

    /// Advance constrained explicit NVT using the OpenMM LF-middle sequence:
    /// full kick and RATTLE, half drift, OU, half drift, SETTLE/SHAKE position
    /// projection with velocity correction, then force refresh. There is no
    /// final half-kick/RATTLE, preserving LF-middle's centered velocities.
    pub async fn dynamics_steps_lf_middle(
        &self,
        steps: usize,
        temperature_k: f32,
        friction_per_ps: f32,
    ) -> Result<(), Error> {
        if steps == 0 {
            return Err(Error::Input(
                "LF-middle batch must contain at least one step",
            ));
        }
        if !self.dt_ps.is_finite() || self.dt_ps <= 0.0 || self.dt_ps > 0.002 {
            return Err(Error::Input("LF-middle timestep must be in (0, 2 fs]"));
        }
        if !temperature_k.is_finite() || temperature_k <= 0.0 {
            return Err(Error::Input("NVT temperature must be finite and positive"));
        }
        if !friction_per_ps.is_finite() || friction_per_ps < 0.0 {
            return Err(Error::Input("NVT friction must be finite and non-negative"));
        }
        self.write_nvt_uniforms(temperature_k, friction_per_ps, 1.0);
        self.queue
            .write_buffer(&self.buffers[2], self.meta_status_off(), &[0; 4]);
        crate::push_error_scope(&self.device, wgpu::ErrorFilter::Validation);
        let atoms = workgroups(self.n);
        let groups = workgroups(self.n.max(self.ncells));
        self.dispatch_dynamics_bounded(
            &[
                (23, atoms),                      // preserve the start-of-step coordinates for SETTLE
                (21, atoms),                      // full force kick
                (8, groups),                      // velocity projection at current coordinates
                (22, atoms),                      // first half drift
                (12, atoms),                      // Ornstein-Uhlenbeck thermostat
                (24, atoms), // second half drift without replacing start coordinates
                (6, groups), // SETTLE/SHAKE plus velocity correction
                (9, atoms),  // molecule-centered bonded coordinates
                (0, atoms),  // neighbor maintenance
                (2, self.eval_dispatch_groups()), // nonbonded forces at the new coordinates
                (4, atoms),  // bonded forces
            ],
            steps,
        )
        .await?;
        self.reduce_energy_once().await?;
        if let Some(e) = crate::pop_error_scope(&self.device).await {
            return Err(Error::Execution(e.to_string()));
        }
        Ok(())
    }

    /// Set up the leap-frog integrator with Nose-Hoover and Parrinello-Rahman
    /// coupling for the next segment. The resident coordinates, velocities
    /// and forces are untouched; call [`Self::set_leapfrog_variables`] next.
    pub async fn configure_leapfrog(&mut self, coupling: LeapfrogCoupling) -> Result<(), Error> {
        let Some(tiles) = &self.tiles else {
            return Err(Error::Input("GPU leap-frog needs the tiled pair kernel"));
        };
        let status_word = self.ncells + 3 * self.n + 1;
        crate::push_error_scope(&self.device, wgpu::ErrorFilter::Validation);
        let engine = LeapfrogEngine::new(
            &self.device,
            LeapfrogShared {
                sys: &self.buffers[1],
                state: &self.buffers[8],
                out: &self.buffers[5],
                aux: &self.buffers[2],
                pbc_config: &self.buffers[0],
                tile_config: &tiles.uniform,
                mesh_config: None,
                status_word,
                rebuild_word: status_word + 1,
                cutoff: self.electro[1],
            },
            coupling,
            self.n,
        );
        if let Some(error) = crate::pop_error_scope(&self.device).await {
            return Err(Error::Execution(error.to_string()));
        }
        let engine = engine?;
        self.set_timestep(engine.coupling.timestep_ps as f32)?;
        // A barostat changes the box between list rebuilds; its allowance
        // comes out of the skin. The next coordinate upload applies it.
        if let Some(tiles) = &mut self.tiles {
            tiles.skin_margin = engine.coupling.box_change_allowance as f32;
        }
        self.leapfrog = Some(engine);
        Ok(())
    }

    /// Choose the couplings of the coming steps: none for a constant-energy
    /// stage, the thermostat for NVT, both for NPT. Leaving NPT stops the
    /// box. Call between advances, after [`Self::read_leapfrog_variables`].
    pub fn set_leapfrog_ensemble(&mut self, thermostat: bool, barostat: bool) -> Result<(), Error> {
        let Some(engine) = &mut self.leapfrog else {
            return Err(Error::Input("leap-frog integrator is not configured"));
        };
        if engine.coupling.barostat && !barostat {
            // box velocity; this step's drag and the pending-pressure flag
            self.queue.write_buffer(&engine.cs, 32, &[0; 16]);
            self.queue.write_buffer(&engine.cs, 80, &[0; 16]);
        }
        engine.coupling.thermostat = thermostat;
        engine.coupling.barostat = barostat;
        Ok(())
    }

    /// Upload the thermostat and barostat variables, with the box, for the
    /// configured leap-frog integrator. The box must be the one of the last
    /// coordinate upload.
    pub fn set_leapfrog_variables(&self, variables: &LeapfrogVariables) -> Result<(), Error> {
        let Some(engine) = &self.leapfrog else {
            return Err(Error::Input("leap-frog integrator is not configured"));
        };
        // The device compares its rebuild counter with the one seen at the
        // last box change; starting from an impossible value makes the first
        // comparison a mismatch, which is the safe side.
        engine.upload(&self.queue, variables, u32::MAX);
        Ok(())
    }

    /// Thermostat and barostat variables and the current box.
    pub async fn read_leapfrog_variables(&self) -> Result<LeapfrogVariables, Error> {
        let Some(engine) = &self.leapfrog else {
            return Err(Error::Input("leap-frog integrator is not configured"));
        };
        let bytes = self
            .readback_buffer(&engine.cs, 0, LeapfrogEngine::coupling_bytes())
            .await?;
        let variables = LeapfrogEngine::decode(&bytes);
        if let Ok(mut current) = self.box_xyz.lock() {
            *current = variables.box_angstrom.map(|v| v as f32);
        }
        Ok(variables)
    }

    fn leapfrog_jobs(&self, engine: &LeapfrogEngine, step: u64) -> Vec<(usize, u32)> {
        let plan = engine.coupling.plan(step);
        let atoms = workgroups(self.n);
        let groups = workgroups(self.n.max(self.ncells));
        let job = |kernel: LeapfrogKernel| (LEAPFROG_JOB + kernel.job(), engine.groups(kernel));
        let mut jobs = Vec::with_capacity(16);
        if plan.thermostat || plan.pressure {
            jobs.push(job(LeapfrogKernel::ReducePartialOld));
            jobs.push(job(LeapfrogKernel::ReduceFinalOld));
        }
        match (plan.thermostat, plan.barostat) {
            (true, true) => jobs.push(job(LeapfrogKernel::CoupleBoth)),
            (true, false) => jobs.push(job(LeapfrogKernel::CoupleThermostat)),
            (false, true) => jobs.push(job(LeapfrogKernel::CoupleBarostat)),
            (false, false) => {}
        }
        jobs.push(job(if plan.thermostat || plan.barostat {
            LeapfrogKernel::KickDriftCoupled
        } else {
            LeapfrogKernel::KickDrift
        }));
        // SETTLE/SHAKE with the constraint displacement added to velocities
        jobs.push((if plan.pressure { SETTLE_VIRIAL } else { 6 }, groups));
        if plan.pressure || plan.com {
            jobs.push(job(LeapfrogKernel::ReducePartialNew));
            jobs.push(job(LeapfrogKernel::ReduceFinalNew));
        }
        if plan.pressure {
            jobs.push(job(LeapfrogKernel::Pressure));
        }
        if plan.barostat {
            jobs.push(job(LeapfrogKernel::Scale));
            jobs.push(job(LeapfrogKernel::ApplyBox));
        }
        if plan.com {
            jobs.push(job(LeapfrogKernel::RemoveCom));
        }
        jobs.extend_from_slice(&[
            (9, atoms),                       // molecule-centered bonded coordinates
            (0, atoms),                       // neighbor maintenance
            (2, self.eval_dispatch_groups()), // nonbonded forces at the new coordinates
            (4, atoms),                       // bonded forces
        ]);
        jobs
    }

    /// Advance `steps` leap-frog steps, the first of which is step
    /// `first_step` of the segment's schedule (thermostat, barostat and
    /// center-of-mass removal act on fixed step numbers). The forces of the
    /// resident state must come from an evaluation with energies when
    /// `first_step` is a pressure step. The final step publishes energies.
    pub async fn dynamics_steps_leapfrog(&self, first_step: u64, steps: usize) -> Result<(), Error> {
        let Some(engine) = &self.leapfrog else {
            return Err(Error::Input("leap-frog integrator is not configured"));
        };
        if steps == 0 {
            return Err(Error::Input(
                "leap-frog batch must contain at least one step",
            ));
        }
        if !self.dt_ps.is_finite() || self.dt_ps <= 0.0 {
            return Err(Error::Input("dynamics timestep is not configured"));
        }
        // SETTLE/SHAKE along the bond directions of the start of the step,
        // with the whole displacement added to the velocities.
        self.write_nvt_uniforms(300.0, 0.0, 1.0);
        self.queue
            .write_buffer(&self.buffers[2], self.meta_status_off(), &[0; 4]);
        crate::push_error_scope(&self.device, wgpu::ErrorFilter::Validation);
        let mut done = 0usize;
        #[cfg(not(target_arch = "wasm32"))]
        let mut previous: Option<wgpu::SubmissionIndex> = None;
        while done < steps {
            let batch = (steps - done).min(MAX_ENCODED_DYNAMICS_STEPS);
            let plans: Vec<Vec<(usize, u32)>> = (0..batch)
                .map(|offset| {
                    let step = first_step + (done + offset) as u64;
                    let last = done + offset + 1 == steps;
                    let energy = last || engine.coupling.plan(step).virial_next;
                    let resort = self.tiles.as_ref().is_some_and(TileEngine::take_resort);
                    self.expanded_jobs(&self.leapfrog_jobs(engine, step), energy, true, resort)
                })
                .collect();
            #[cfg(not(target_arch = "wasm32"))]
            if self._context.gpu_timestamps_enabled() {
                self.dispatch_plans_profiled(&plans).await?;
                done += batch;
                continue;
            }
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for &(pipeline, groups) in plans.iter().flatten() {
                self.encode_job(&mut pass, pipeline, groups);
            }
            drop(pass);
            let index = self.queue.submit(Some(encoder.finish()));
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(earlier) = previous.replace(index) {
                self.device
                    .poll(wgpu::PollType::WaitForSubmissionIndex(earlier))
                    .map_err(|e| Error::Execution(e.to_string()))?;
            }
            #[cfg(target_arch = "wasm32")]
            let _ = index;
            done += batch;
        }
        if let Some(e) = crate::pop_error_scope(&self.device).await {
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
    pub async fn dynamics_status(&self) -> Result<Option<String>, Error> {
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
            neighbor_rebuild_count: 0,
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

#[cfg(test)]
mod status_tests {
    use super::dynamics_error;

    #[test]
    fn constraint_failures_name_their_group_and_say_when_a_field_is_saturated() {
        assert_eq!(dynamics_error(0), None);
        assert_eq!(
            dynamics_error(0x5000_0000 | (149 << 16) | 1234).unwrap(),
            "GPU solute position constraints failed in constraint group 149 (relative error 1.2e-3)"
        );
        assert_eq!(
            dynamics_error(0x5000_0000 | (149 << 16) | 0xffff).unwrap(),
            "GPU solute position constraints failed in constraint group 149 (relative error 6.6e-2 or more)"
        );
        assert_eq!(
            dynamics_error(0x6000_0000 | (0x0fff << 16) | 30).unwrap(),
            "GPU solute velocity constraints failed in constraint group 4095 or beyond (relative error 3.0e-5)"
        );
    }
}
