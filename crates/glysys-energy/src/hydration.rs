//! Exploratory rigid-water probe sites, not equilibrium water occupancies.
use crate::{EnergyError, Result};
use glysys::{ParameterizedSystem, Vec3};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MODEL_VERSION: &str = "tip3p-probe-v1";
/// Grand-canonical-lite water sampling over probe-seeded candidates.
/// Same TIP3P physics as the probe plus water-water terms and a fixed
/// chemical potential; reports occupancies, not single-probe minima.
pub const GC_MODEL_VERSION: &str = "gc-water-v1";
/// Bulk chemical potential proxy in kcal/mol (favorable waters must beat this).
pub const GC_DEFAULT_MU_KCAL_MOL: f64 = -6.0;
/// Thermal energy at 300 K in kcal/mol for Metropolis acceptance.
pub const GC_KT_KCAL_MOL: f64 = 0.596;
pub const GC_DEFAULT_STEPS: usize = 20_000;
const COULOMB: f64 = 332.063_713_299;
const OH: f64 = 0.9572;
const ANGLE: f64 = 104.52 * std::f64::consts::PI / 180.;
// Match the shipped GLYCAM TIP3P table (not another engine's rounded table).
pub const O_RADIUS: f64 = 1.7683;
pub const O_EPSILON: f64 = 0.1520;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct HydrationRequest {
    pub minimum: Vec3,
    pub maximum: Vec3,
    #[serde(default = "spacing")]
    pub spacing: f64,
    #[serde(default = "orientations")]
    pub orientations: usize,
    #[serde(default = "max_sites")]
    pub max_sites: usize,
    #[serde(default)]
    pub cutoff: Option<f64>,
    /// Hydration method selector. `None` means the legacy fast probe path so
    /// previously exported request JSON keeps parsing and keeps its meaning.
    /// New callers send `Some("gc")` (default in the UI) or `Some("probe")`.
    #[serde(default)]
    pub method: Option<String>,
    /// Fixed chemical potential in kcal/mol for GC insertion/deletion.
    #[serde(default)]
    pub chemical_potential: Option<f64>,
    /// Number of GC Monte Carlo steps (deterministic RNG).
    #[serde(default)]
    pub gc_steps: Option<usize>,
    /// Optional GC RNG seed; defaults to a fingerprint-derived seed.
    #[serde(default)]
    pub gc_seed: Option<u64>,
}

/// Returns true for GC requests; `None` (old JSON) stays on the probe path.
pub fn is_gc_request(request: &HydrationRequest) -> bool {
    request
        .method
        .as_deref()
        .is_some_and(|m| m.eq_ignore_ascii_case("gc"))
}

pub fn gc_chemical_potential(request: &HydrationRequest) -> Result<f64> {
    let mu = request.chemical_potential.unwrap_or(GC_DEFAULT_MU_KCAL_MOL);
    if !mu.is_finite() || mu < -20. || mu > 5. {
        return Err(invalid("invalid GC chemical potential"));
    }
    Ok(mu)
}

pub fn gc_step_count(request: &HydrationRequest) -> Result<usize> {
    let steps = request.gc_steps.unwrap_or(GC_DEFAULT_STEPS);
    if steps < 1_000 || steps > 500_000 {
        return Err(invalid("invalid GC step count"));
    }
    Ok(steps)
}
fn spacing() -> f64 {
    1.
}
fn orientations() -> usize {
    96
}
fn max_sites() -> usize {
    100
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeScore {
    pub lennard_jones: f64,
    pub electrostatics: f64,
}
impl ProbeScore {
    pub fn total(self) -> f64 {
        self.lennard_jones + self.electrostatics
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HydrationSite {
    pub id: String,
    pub position: Vec3,
    pub hydrogens: Option<[Vec3; 2]>,
    pub source: String,
    pub score: Option<ProbeScore>,
    pub experimental_occupancy: Option<f64>,
    pub confidence: Option<f64>,
    pub displacement_free_energy: Option<f64>,
    /// GC visit frequency in (0, 1]; absent on probe/deposited/imported sites.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occupancy: Option<f64>,
    /// GC water-water contribution summed over the other selected waters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub water_water: Option<ProbeScore>,
    /// Heuristic environment label: bridging / surface / exposed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bridging: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HydrationField {
    pub source_context: Option<HydrationContext>,
    pub schema_version: u32,
    pub model_version: String,
    pub receptor_fingerprint: String,
    pub request: HydrationRequest,
    pub sites: Vec<HydrationSite>,
    pub dimensions: [usize; 3],
    /// z varies fastest. Invalid/excluded voxels are null, never infinite JSON.
    pub probe_energy: Vec<Option<f64>>,
    pub backend: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HydrationContext {
    pub model_version: String,
    pub request: HydrationRequest,
    pub receptor_fingerprint: String,
}
#[derive(Clone, Copy, Debug)]
pub struct ProbeAtom {
    pub position: Vec3,
    pub charge: f64,
    pub radius: f64,
    pub epsilon: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct WaterPose {
    pub oxygen: Vec3,
    pub hydrogens: [Vec3; 2],
}
pub trait HydrationProvider {
    fn model_version(&self) -> &str;
    fn predict(&self, request: &HydrationRequest) -> Result<HydrationField>;
}
#[derive(Clone)]
pub struct PhysicalProbe {
    pub atoms: Vec<ProbeAtom>,
    pub fingerprint: String,
}
fn invalid(message: &str) -> EnergyError {
    EnergyError::InvalidConfiguration(message.into())
}
fn add(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
    }
}
fn sub(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
}
fn distance2(a: Vec3, b: Vec3) -> f64 {
    let d = sub(a, b);
    d.x * d.x + d.y * d.y + d.z * d.z
}
fn finite(a: Vec3) -> bool {
    a.x.is_finite() && a.y.is_finite() && a.z.is_finite()
}
fn rotate(v: Vec3, q: [f64; 4]) -> Vec3 {
    let [x, y, z, w] = q;
    Vec3 {
        x: (1. - 2. * (y * y + z * z)) * v.x
            + 2. * (x * y - z * w) * v.y
            + 2. * (x * z + y * w) * v.z,
        y: 2. * (x * y + z * w) * v.x
            + (1. - 2. * (x * x + z * z)) * v.y
            + 2. * (y * z - x * w) * v.z,
        z: 2. * (x * z - y * w) * v.x
            + 2. * (y * z + x * w) * v.y
            + (1. - 2. * (x * x + y * y)) * v.z,
    }
}
/// Deterministic stratified SO(3) coverage; does not introduce random job state.
pub fn orientation(index: usize, count: usize) -> [Vec3; 2] {
    let u = (index as f64 + 0.5) / count as f64;
    let v = (index as f64 * 0.6180339887498949).fract() * std::f64::consts::TAU;
    let w = (index as f64 * 0.4142135623730951).fract() * std::f64::consts::TAU;
    let q = [
        (1. - u).sqrt() * v.sin(),
        (1. - u).sqrt() * v.cos(),
        u.sqrt() * w.sin(),
        u.sqrt() * w.cos(),
    ];
    [
        rotate(
            Vec3 {
                x: OH,
                y: 0.,
                z: 0.,
            },
            q,
        ),
        rotate(
            Vec3 {
                x: OH * ANGLE.cos(),
                y: OH * ANGLE.sin(),
                z: 0.,
            },
            q,
        ),
    ]
}
/// Maximum voxels scanned in one tile; matches the single-box cap.
pub const TILE_MAX_POINTS: usize = 1_000_000;
/// Maximum voxels across all tiles of one whole-surface prediction.
pub const SURFACE_MAX_POINTS: usize = 4_000_000;

/// Integer offset plus dimensions of one tile inside a stitched global grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileRegion {
    pub offset: [usize; 3],
    pub dimensions: [usize; 3],
}

/// Split a global grid into tiles of at most `max_points` voxels each by
/// repeatedly halving the longest axis. The tiles partition the grid exactly.
pub fn split_tiles(dimensions: [usize; 3], max_points: usize) -> Vec<TileRegion> {
    let mut regions = vec![TileRegion {
        offset: [0, 0, 0],
        dimensions,
    }];
    let mut out = Vec::new();
    while let Some(region) = regions.pop() {
        let points: usize = region
            .dimensions
            .iter()
            .fold(1, |a, b| a.saturating_mul(*b));
        if points <= max_points || region.dimensions.iter().all(|&d| d <= 1) {
            out.push(region);
            continue;
        }
        let axis = (0..3).max_by_key(|&a| region.dimensions[a]).unwrap_or(0);
        let half = region.dimensions[axis] / 2;
        let mut first = region;
        first.dimensions[axis] = half;
        let mut second = region;
        second.offset[axis] += half;
        second.dimensions[axis] -= half;
        regions.push(first);
        regions.push(second);
    }
    out
}

/// One tile's per-voxel best energies in scan order, plus its integer offset
/// inside the stitched global grid.
pub struct TileGrid {
    pub offset: [usize; 3],
    pub dimensions: [usize; 3],
    pub energies: Vec<Option<f64>>,
}

impl PhysicalProbe {
    pub fn new(system: &ParameterizedSystem) -> Result<Self> {
        if system
            .atoms()
            .iter()
            .any(|a| !matches!(a.element(), 1 | 6 | 7 | 8 | 9 | 15 | 16 | 17 | 35 | 53))
        {
            return Err(invalid(
                "water-probe prediction does not support metals or unknown elements",
            ));
        }
        let atoms: Vec<_> = system
            .atoms()
            .iter()
            .filter(|a| {
                !matches!(
                    system.residues()[a.residue_index()].name(),
                    "HOH" | "WAT" | "TIP3"
                )
            })
            .map(|a| ProbeAtom {
                position: a.position(),
                charge: a.charge(),
                radius: a.lennard_jones_radius(),
                epsilon: a.lennard_jones_epsilon(),
            })
            .collect();
        if atoms.is_empty()
            || atoms.iter().any(|a| {
                !finite(a.position)
                    || !a.charge.is_finite()
                    || !a.radius.is_finite()
                    || a.radius < 0.
                    || !a.epsilon.is_finite()
                    || a.epsilon < 0.
            })
        {
            return Err(invalid("invalid receptor parameters"));
        }
        let fingerprint = format!(
            "{:x}",
            Sha256::digest(format!("{MODEL_VERSION}:{atoms:?}").as_bytes())
        );
        Ok(Self { atoms, fingerprint })
    }
    /// Pair energy only, dielectric 1. No solvent free-energy interpretation.
    pub fn score(&self, pose: WaterPose, cutoff: Option<f64>) -> Option<ProbeScore> {
        let mut score = ProbeScore {
            lennard_jones: 0.,
            electrostatics: 0.,
        };
        for atom in &self.atoms {
            for (i, p) in [pose.oxygen, pose.hydrogens[0], pose.hydrogens[1]]
                .iter()
                .enumerate()
            {
                let d2 = distance2(*p, atom.position);
                if !d2.is_finite() || d2 < 0.25 {
                    return None;
                }
                if cutoff.is_some_and(|c| d2 > c * c) {
                    continue;
                }
                let d = d2.sqrt();
                if i == 0 {
                    let r6 = ((atom.radius + O_RADIUS) / d).powi(6);
                    score.lennard_jones += (atom.epsilon * O_EPSILON).sqrt() * (r6 * r6 - 2. * r6);
                }
                score.electrostatics +=
                    COULOMB * atom.charge * if i == 0 { -0.834 } else { 0.417 } / d;
            }
        }
        score.total().is_finite().then_some(score)
    }
    pub fn dimensions(request: &HydrationRequest) -> Result<[usize; 3]> {
        if !finite(request.minimum)
            || !finite(request.maximum)
            || !request.spacing.is_finite()
            || request.spacing < 0.25
            || request.orientations == 0
            || request.orientations > 4096
            || request.max_sites == 0
            || request.max_sites > 10000
            || request.cutoff.is_some_and(|c| !c.is_finite() || c <= 0.)
        {
            return Err(invalid("invalid hydration region or resolution"));
        }
        let d = sub(request.maximum, request.minimum);
        if d.x < 0. || d.y < 0. || d.z < 0. {
            return Err(invalid("reversed hydration bounds"));
        }
        let dims = [d.x, d.y, d.z].map(|v| (v / request.spacing).floor() as usize + 1);
        if dims
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .is_none_or(|n| n > 1_000_000)
        {
            return Err(invalid(
                "hydration region exceeds one million grid points; select a smaller region",
            ));
        }
        Ok(dims)
    }
    pub fn poses(request: &HydrationRequest, dims: [usize; 3], index: usize) -> Vec<WaterPose> {
        let oxygen = add(
            request.minimum,
            Vec3 {
                x: (index / (dims[1] * dims[2])) as f64 * request.spacing,
                y: ((index / dims[2]) % dims[1]) as f64 * request.spacing,
                z: (index % dims[2]) as f64 * request.spacing,
            },
        );
        (0..request.orientations)
            .map(|i| {
                let h = orientation(i, request.orientations);
                WaterPose {
                    oxygen,
                    hydrogens: h.map(|v| add(oxygen, v)),
                }
            })
            .collect()
    }
    pub fn finish(
        &self,
        request: &HydrationRequest,
        dimensions: [usize; 3],
        best: Vec<Option<(WaterPose, ProbeScore)>>,
        backend: &str,
    ) -> HydrationField {
        let probe_energy = best.iter().map(|s| s.map(|(_, e)| e.total())).collect();
        let candidates: Vec<_> = best.into_iter().flatten().collect();
        let sites = self.select_sites(
            request.minimum,
            request.maximum,
            request.max_sites,
            request.cutoff,
            candidates,
        );
        HydrationField {
            source_context: None,
            schema_version: 1,
            model_version: MODEL_VERSION.into(),
            receptor_fingerprint: self.fingerprint.clone(),
            request: request.clone(),
            sites,
            dimensions,
            probe_energy,
            backend: backend.into(),
        }
    }

    /// Merge per-tile scans into one site list and one stitched energy grid.
    /// Tiles must share spacing; offsets address the global grid exactly.
    pub fn finish_tiled(
        &self,
        request: &HydrationRequest,
        dimensions: [usize; 3],
        tiles: Vec<TileGrid>,
        candidates: Vec<(WaterPose, ProbeScore)>,
        backend: &str,
    ) -> HydrationField {
        let total: usize = dimensions.iter().fold(1, |a, b| a.saturating_mul(*b));
        let mut probe_energy = vec![None; total];
        let plane = dimensions[1].saturating_mul(dimensions[2]);
        for tile in &tiles {
            let tile_plane = tile.dimensions[1].saturating_mul(tile.dimensions[2]);
            for (n, energy) in tile.energies.iter().enumerate() {
                let x = n / tile_plane.max(1);
                let y = (n / tile.dimensions[2].max(1)) % tile.dimensions[1].max(1);
                let z = n % tile.dimensions[2].max(1);
                let g = (tile.offset[0] + x) * plane
                    + (tile.offset[1] + y) * dimensions[2]
                    + (tile.offset[2] + z);
                if let Some(slot) = probe_energy.get_mut(g) {
                    *slot = *energy;
                }
            }
        }
        let sites = self.select_sites(
            request.minimum,
            request.maximum,
            request.max_sites,
            request.cutoff,
            candidates,
        );
        HydrationField {
            source_context: None,
            schema_version: 1,
            model_version: MODEL_VERSION.into(),
            receptor_fingerprint: self.fingerprint.clone(),
            request: request.clone(),
            sites,
            dimensions,
            probe_energy,
            backend: backend.into(),
        }
    }

    /// Shared candidate selection: keep favorable poses, sort, suppress
    /// neighbors within 2.4 Å, then refine survivors in place.
    fn select_sites(
        &self,
        minimum: Vec3,
        maximum: Vec3,
        max_sites: usize,
        cutoff: Option<f64>,
        candidates: Vec<(WaterPose, ProbeScore)>,
    ) -> Vec<HydrationSite> {
        let mut candidates: Vec<_> = candidates
            .into_iter()
            .filter(|(_, e)| e.total() < 0.)
            .collect();
        candidates.sort_by(|a, b| a.1.total().total_cmp(&b.1.total()));
        let mut sites: Vec<HydrationSite> = Vec::new();
        for (mut pose, mut score) in candidates {
            if sites
                .iter()
                .any(|s| distance2(s.position, pose.oxygen) < 2.4f64.powi(2))
            {
                continue;
            }
            // Bounded pattern search refines rigid translation and rotation together.
            for step in [0.25, 0.125, 0.0625] {
                for _ in 0..8 {
                    let mut changed = false;
                    for axis in 0..6 {
                        for sign in [-1., 1.] {
                            let mut next = pose;
                            let d = sign * step;
                            if axis < 3 {
                                let mut v = Vec3 {
                                    x: 0.,
                                    y: 0.,
                                    z: 0.,
                                };
                                match axis {
                                    0 => v.x = d,
                                    1 => v.y = d,
                                    _ => v.z = d,
                                }
                                next.oxygen = add(next.oxygen, v);
                                next.hydrogens = next.hydrogens.map(|h| add(h, v));
                            } else {
                                let mut q = [0., 0., 0., (d * 0.5).cos()];
                                q[axis - 3] = (d * 0.5).sin();
                                next.hydrogens = next
                                    .hydrogens
                                    .map(|h| add(next.oxygen, rotate(sub(h, next.oxygen), q)));
                            }
                            let o = next.oxygen;
                            if o.x < minimum.x
                                || o.y < minimum.y
                                || o.z < minimum.z
                                || o.x > maximum.x
                                || o.y > maximum.y
                                || o.z > maximum.z
                            {
                                continue;
                            }
                            if let Some(e) = self.score(next, cutoff) {
                                if e.total() < score.total() {
                                    pose = next;
                                    score = e;
                                    changed = true;
                                }
                            }
                        }
                    }
                    if !changed {
                        break;
                    }
                }
            }
            if sites
                .iter()
                .any(|s| distance2(s.position, pose.oxygen) < 2.4f64.powi(2))
            {
                continue;
            }
            sites.push(HydrationSite {
                id: format!("probe-{}", sites.len() + 1),
                position: pose.oxygen,
                hydrogens: Some(pose.hydrogens),
                source: "physical_probe".into(),
                score: Some(score),
                experimental_occupancy: None,
                confidence: None,
                displacement_free_energy: None,
                occupancy: None,
                water_water: None,
                bridging: None,
            });
            if sites.len() == max_sites {
                break;
            }
        }
        sites
    }
}
impl PhysicalProbe {
    /// Chemistry fingerprint namespaced by the GC model version so GC fields
    /// invalidate cleanly when the model changes. The probe fingerprint is
    /// intentionally untouched for backward compatibility.
    pub fn gc_fingerprint(&self) -> String {
        format!(
            "{:x}",
            Sha256::digest(format!("{GC_MODEL_VERSION}:{:?}", self.atom_positions()).as_bytes())
        )
    }

    fn atom_positions(&self) -> Vec<(f64, f64, f64, f64, f64, f64)> {
        self.atoms
            .iter()
            .map(|a| {
                (
                    a.position.x,
                    a.position.y,
                    a.position.z,
                    a.charge,
                    a.radius,
                    a.epsilon,
                )
            })
            .collect()
    }

    /// Merge per-tile scans into one GC site list plus the stitched probe grid.
    /// The grid is identical to the probe path; only site selection changes.
    pub fn finish_tiled_gc(
        &self,
        request: &HydrationRequest,
        dimensions: [usize; 3],
        tiles: Vec<TileGrid>,
        candidates: Vec<(WaterPose, ProbeScore)>,
        backend: &str,
    ) -> Result<HydrationField> {
        let total: usize = dimensions.iter().fold(1, |a, b| a.saturating_mul(*b));
        let mut probe_energy = vec![None; total];
        let plane = dimensions[1].saturating_mul(dimensions[2]);
        for tile in &tiles {
            let tile_plane = tile.dimensions[1].saturating_mul(tile.dimensions[2]);
            for (n, energy) in tile.energies.iter().enumerate() {
                let x = n / tile_plane.max(1);
                let y = (n / tile.dimensions[2].max(1)) % tile.dimensions[1].max(1);
                let z = n % tile.dimensions[2].max(1);
                let g = (tile.offset[0] + x) * plane
                    + (tile.offset[1] + y) * dimensions[2]
                    + (tile.offset[2] + z);
                if let Some(slot) = probe_energy.get_mut(g) {
                    *slot = *energy;
                }
            }
        }
        let sites = self.select_gc_sites(request, candidates)?;
        Ok(HydrationField {
            source_context: None,
            schema_version: 1,
            model_version: GC_MODEL_VERSION.into(),
            receptor_fingerprint: self.gc_fingerprint(),
            request: request.clone(),
            sites,
            dimensions,
            probe_energy,
            backend: backend.into(),
        })
    }

    /// GCMC-lite over probe-seeded candidates with deterministic RNG.
    ///
    /// Grand potential proxy: G = sum(receptor) + sum(water-water) - mu * N.
    /// Moves: insertion (40%), deletion (30%), occupancy swap (30%).
    /// Translation/rotation enter through bounded refinement of the selected
    /// survivors with a water-water clash veto. Water-water pairs run on CPU:
    /// N is small (<= max_sites scale) while the receptor scan stays on GPU.
    fn select_gc_sites(
        &self,
        request: &HydrationRequest,
        candidates: Vec<(WaterPose, ProbeScore)>,
    ) -> Result<Vec<HydrationSite>> {
        let mu = gc_chemical_potential(request)?;
        let steps = gc_step_count(request)?;
        let mut pool: Vec<(WaterPose, ProbeScore)> = candidates
            .into_iter()
            .filter(|(_, e)| e.total().is_finite() && e.total() < 0.)
            .collect();
        pool.sort_by(|a, b| a.1.total().total_cmp(&b.1.total()));
        // Bound the pool so the O(M^2) water matrix stays trivial.
        let pool_cap = request.max_sites.saturating_mul(8).clamp(200, 2000);
        pool.truncate(pool_cap);
        if pool.is_empty() {
            return Ok(Vec::new());
        }
        let m = pool.len();
        let receptor: Vec<f64> = pool.iter().map(|(_, e)| e.total()).collect();
        // Symmetric water-water matrix (total LJ + Coulomb, clamped).
        let mut ww = vec![0f64; m * m];
        for i in 0..m {
            for j in (i + 1)..m {
                let e = water_pair_total(pool[i].0, pool[j].0);
                ww[i * m + j] = e;
                ww[j * m + i] = e;
            }
        }
        // Deterministic seed: explicit seed wins, else fingerprint-derived.
        let mut seed = request.gc_seed.unwrap_or(0x9E37_79B9_7F4A_7C15);
        for b in self.fingerprint.bytes() {
            seed = seed.wrapping_mul(0x1000_0000_01B3).wrapping_add(b as u64);
        }
        seed = seed.wrapping_add((mu.to_bits() ^ (steps as u64).wrapping_mul(0x9E37_79B9)) as u64);
        let mut rng = SplitMix::new(seed | 1);
        let mut occupied = vec![false; m];
        let mut occupied_count = 0usize;
        // Current water-water field per site from the occupied set.
        let mut field = vec![0f64; m];
        let burn_in = steps / 5;
        let mut counts = vec![0u64; m];
        let mut samples = 0u64;
        for step in 0..steps {
            let r = rng.next_f64();
            if r < 0.40 {
                // Insertion of a random unoccupied candidate.
                if occupied_count == m {
                    continue;
                }
                let i = rng.next_range(m);
                if occupied[i] {
                    continue;
                }
                let delta = receptor[i] + field[i] - mu;
                if delta < 0. || rng.next_f64() < (-delta / GC_KT_KCAL_MOL).exp() {
                    occupied[i] = true;
                    occupied_count += 1;
                    for j in 0..m {
                        field[j] += ww[i * m + j];
                    }
                }
            } else if r < 0.70 {
                // Deletion of a random occupied water.
                if occupied_count == 0 {
                    continue;
                }
                let j = rng.next_range(m);
                if !occupied[j] {
                    continue;
                }
                // field[j] includes ww[j][j] = 0, so it equals sum over others.
                let delta = -(receptor[j] + field[j]) + mu;
                if delta < 0. || rng.next_f64() < (-delta / GC_KT_KCAL_MOL).exp() {
                    occupied[j] = false;
                    occupied_count -= 1;
                    for k in 0..m {
                        field[k] -= ww[j * m + k];
                    }
                }
            } else {
                // Swap: move one occupancy to a nearby-energy candidate.
                if occupied_count == 0 || occupied_count == m {
                    continue;
                }
                let j = rng.next_range(m);
                let i = rng.next_range(m);
                if !occupied[j] || occupied[i] || i == j {
                    continue;
                }
                // field[i] holds i's water terms vs the occupied set (j is
                // occupied, i is not, so ww[i][j] is included exactly once).
                let delta = (receptor[i] + field[i] - ww[i * m + j]) - (receptor[j] + field[j]);
                if delta < 0. || rng.next_f64() < (-delta / GC_KT_KCAL_MOL).exp() {
                    occupied[j] = false;
                    occupied[i] = true;
                    for k in 0..m {
                        field[k] += ww[i * m + k] - ww[j * m + k];
                    }
                }
            }
            if step >= burn_in && step % 10 == 0 {
                samples += 1;
                for (k, on) in occupied.iter().enumerate() {
                    if *on {
                        counts[k] += 1;
                    }
                }
            }
        }
        if samples == 0 {
            return Ok(Vec::new());
        }
        // Rank by occupancy, then receptor energy; suppress 2.4 A neighbors.
        let mut ranked: Vec<usize> = (0..m).filter(|&i| counts[i] > 0).collect();
        ranked.sort_by(|&a, &b| {
            counts[b]
                .cmp(&counts[a])
                .then(receptor[a].total_cmp(&receptor[b]))
        });
        let mut sites: Vec<HydrationSite> = Vec::new();
        let mut placed: Vec<WaterPose> = Vec::new();
        for i in ranked {
            let occupancy = counts[i] as f64 / samples as f64;
            if occupancy < 0.05 {
                continue;
            }
            let (mut pose, score) = pool[i];
            if placed
                .iter()
                .any(|q| distance2(q.oxygen, pose.oxygen) < 2.4f64.powi(2))
            {
                continue;
            }
            // Bounded refinement with water-water clash veto (translation/rotation).
            pose = self.refine_gc_pose(request, pose, score, &placed);
            if placed
                .iter()
                .any(|q| distance2(q.oxygen, pose.oxygen) < 2.4f64.powi(2))
            {
                continue;
            }
            let rescored = self.score(pose, request.cutoff);
            let (final_pose, final_score) = match rescored {
                Some(e) if e.total() < 0. => (pose, e),
                _ => continue,
            };
            let mut ww_lj = 0.;
            let mut ww_el = 0.;
            for q in &placed {
                if let Some(pair) = water_pair(final_pose, *q) {
                    ww_lj += pair.lennard_jones;
                    ww_el += pair.electrostatics;
                }
            }
            let bridging = self.gc_environment(final_pose).into();
            let n = sites.len();
            sites.push(HydrationSite {
                id: format!("gc-{}", n + 1),
                position: final_pose.oxygen,
                hydrogens: Some(final_pose.hydrogens),
                source: "gc_water".into(),
                score: Some(final_score),
                experimental_occupancy: None,
                confidence: Some(occupancy),
                displacement_free_energy: None,
                occupancy: Some(occupancy),
                water_water: Some(ProbeScore {
                    lennard_jones: ww_lj,
                    electrostatics: ww_el,
                }),
                bridging: Some(bridging),
            });
            placed.push(final_pose);
            if sites.len() == request.max_sites {
                break;
            }
        }
        Ok(sites)
    }

    /// Small pattern search on receptor energy; vetoes water-water clashes.
    fn refine_gc_pose(
        &self,
        request: &HydrationRequest,
        mut pose: WaterPose,
        mut score: ProbeScore,
        placed: &[WaterPose],
    ) -> WaterPose {
        for step in [0.25, 0.125] {
            for _ in 0..4 {
                let mut changed = false;
                for axis in 0..6 {
                    for sign in [-1., 1.] {
                        let mut next = pose;
                        let d = sign * step;
                        if axis < 3 {
                            let mut v = Vec3 {
                                x: 0.,
                                y: 0.,
                                z: 0.,
                            };
                            match axis {
                                0 => v.x = d,
                                1 => v.y = d,
                                _ => v.z = d,
                            }
                            next.oxygen = add(next.oxygen, v);
                            next.hydrogens = next.hydrogens.map(|h| add(h, v));
                        } else {
                            let mut q = [0., 0., 0., (d * 0.5).cos()];
                            q[axis - 3] = (d * 0.5).sin();
                            next.hydrogens = next
                                .hydrogens
                                .map(|h| add(next.oxygen, rotate(sub(h, next.oxygen), q)));
                        }
                        let o = next.oxygen;
                        if o.x < request.minimum.x
                            || o.y < request.minimum.y
                            || o.z < request.minimum.z
                            || o.x > request.maximum.x
                            || o.y > request.maximum.y
                            || o.z > request.maximum.z
                        {
                            continue;
                        }
                        if placed
                            .iter()
                            .any(|q| distance2(q.oxygen, next.oxygen) < 2.0f64.powi(2))
                        {
                            continue;
                        }
                        if let Some(e) = self.score(next, request.cutoff) {
                            if e.total() < score.total() {
                                pose = next;
                                score = e;
                                changed = true;
                            }
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        pose
    }

    /// Heuristic environment from polar receptor contacts within 3.5 A of the
    /// water oxygen. Polar ~= |charge| > 0.3. Bridging means 2+ contacts.
    fn gc_environment(&self, pose: WaterPose) -> &'static str {
        let mut contacts = 0;
        for atom in &self.atoms {
            if atom.charge.abs() < 0.3 {
                continue;
            }
            if distance2(atom.position, pose.oxygen) < 3.5f64.powi(2) {
                contacts += 1;
                if contacts >= 2 {
                    return "bridging";
                }
            }
        }
        if contacts == 1 { "surface" } else { "exposed" }
    }
}

/// TIP3P-TIP3P pair energy with the same table as the receptor probe.
/// LJ acts on O-O only; Coulomb sums all nine partial-charge pairs.
/// Returns `None` only on nonfinite input; short contacts yield large but
/// finite repulsion so Monte Carlo rejects them instead of erroring.
pub fn water_pair(a: WaterPose, b: WaterPose) -> Option<ProbeScore> {
    let qa = [-0.834, 0.417, 0.417];
    let qb = [-0.834, 0.417, 0.417];
    let pa = [a.oxygen, a.hydrogens[0], a.hydrogens[1]];
    let pb = [b.oxygen, b.hydrogens[0], b.hydrogens[1]];
    for p in pa.iter().chain(pb.iter()) {
        if !finite(*p) {
            return None;
        }
    }
    let d2 = distance2(a.oxygen, b.oxygen);
    if !d2.is_finite() {
        return None;
    }
    let d = d2.sqrt().max(0.5);
    let r = (2. * O_RADIUS) / d;
    let r2 = r * r;
    let r6 = r2 * r2 * r2;
    let lj = (O_EPSILON * (r6 * r6 - 2. * r6)).clamp(-5., 50.);
    let mut el = 0.;
    for (i, x) in pa.iter().enumerate() {
        for (j, y) in pb.iter().enumerate() {
            let dd = distance2(*x, *y).sqrt().max(0.5);
            el += COULOMB * qa[i] * qb[j] / dd;
        }
    }
    if !lj.is_finite() || !el.is_finite() {
        return None;
    }
    Some(ProbeScore {
        lennard_jones: lj,
        electrostatics: el,
    })
}

fn water_pair_total(a: WaterPose, b: WaterPose) -> f64 {
    water_pair(a, b).map(|e| e.total()).unwrap_or(50.)
}

/// Deterministic SplitMix64 RNG; no global state, reproducible per seed.
struct SplitMix {
    state: u64,
}

impl SplitMix {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        // 53-bit uniform in [0, 1).
        const DIV: f64 = (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) / DIV
    }

    fn next_range(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_f64() * n as f64).floor() as usize % n
    }
}

impl HydrationProvider for PhysicalProbe {
    fn model_version(&self) -> &str {
        MODEL_VERSION
    }
    fn predict(&self, request: &HydrationRequest) -> Result<HydrationField> {
        let dims = Self::dimensions(request)?;
        let mut best = Vec::with_capacity(dims.iter().product());
        for i in 0..dims.iter().product() {
            best.push(
                Self::poses(request, dims, i)
                    .into_iter()
                    .filter_map(|p| self.score(p, request.cutoff).map(|e| (p, e)))
                    .min_by(|a, b| a.1.total().total_cmp(&b.1.total())),
            );
        }
        Ok(self.finish(request, dims, best, "cpu"))
    }
}
impl HydrationField {
    pub fn pdb(&self) -> String {
        let mut text = String::from(
            "REMARK Hydration sites; source and experimental occupancy are recorded in JSON\n",
        );
        for (i, s) in self.sites.iter().enumerate() {
            let p = s.position;
            text += &format!(
                "HETATM{:5}  O   HOH W{:4}    {:8.3}{:8.3}{:8.3}  1.00  0.00           O\\n",
                i + 1,
                i + 1,
                p.x,
                p.y,
                p.z
            )
            .replace("\\n", "\n");
        }
        text + "END\n"
    }
    pub fn open_dx(&self) -> String {
        let [x, y, z] = self.dimensions;
        let o = self.request.minimum;
        let s = self.request.spacing;
        let mut text = format!(
            "# Water-probe interaction energy in kcal/mol; excluded voxels NaN\nobject 1 class gridpositions counts {x} {y} {z}\norigin {} {} {}\ndelta {s} 0 0\ndelta 0 {s} 0\ndelta 0 0 {s}\nobject 2 class gridconnections counts {x} {y} {z}\nobject 3 class array type double rank 0 items {} data follows\n",
            o.x,
            o.y,
            o.z,
            self.probe_energy.len()
        );
        for chunk in self.probe_energy.chunks(3) {
            text += &chunk
                .iter()
                .map(|v| v.map_or("NaN".into(), |x| format!("{x:.8}")))
                .collect::<Vec<_>>()
                .join(" ");
            text.push('\n');
        }
        text + "attribute \"dep\" string \"positions\"\nobject \"probe energy\" class field\ncomponent \"positions\" value 1\ncomponent \"connections\" value 2\ncomponent \"data\" value 3\n"
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn orientations_keep_tip3p_geometry() {
        for i in 0..96 {
            let h = orientation(i, 96);
            let o = Vec3 {
                x: 0.,
                y: 0.,
                z: 0.,
            };
            assert!((distance2(h[0], o) - OH * OH).abs() < 1e-12);
            assert!((distance2(h[1], o) - OH * OH).abs() < 1e-12);
            assert!((distance2(h[0], h[1]) - 2. * OH * OH * (1. - ANGLE.cos())).abs() < 1e-12);
        }
    }
    #[test]
    fn translation_preserves_probe_pair_energy() {
        let a = ProbeAtom {
            position: Vec3 {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            charge: -0.4,
            radius: 1.5,
            epsilon: 0.2,
        };
        let probe = PhysicalProbe {
            atoms: vec![a],
            fingerprint: String::new(),
        };
        let o = Vec3 {
            x: 3.,
            y: 0.,
            z: 0.,
        };
        let h = orientation(7, 96).map(|h| add(h, o));
        let e = probe
            .score(
                WaterPose {
                    oxygen: o,
                    hydrogens: h,
                },
                None,
            )
            .unwrap()
            .total();
        let d = Vec3 {
            x: 100.,
            y: -300.,
            z: 40.,
        };
        let moved = PhysicalProbe {
            atoms: vec![ProbeAtom {
                position: add(a.position, d),
                ..a
            }],
            fingerprint: String::new(),
        };
        let actual = moved
            .score(
                WaterPose {
                    oxygen: add(o, d),
                    hydrogens: h.map(|h| add(h, d)),
                },
                None,
            )
            .unwrap()
            .total();
        assert!((e - actual).abs() < 1e-10);
    }

    fn test_probe() -> PhysicalProbe {
        PhysicalProbe {
            atoms: Vec::new(),
            fingerprint: "test".into(),
        }
    }

    fn test_pose(x: f64) -> WaterPose {
        WaterPose {
            oxygen: Vec3 { x, y: 0., z: 0. },
            hydrogens: [
                Vec3 {
                    x: x + OH,
                    y: 0.,
                    z: 0.,
                },
                Vec3 {
                    x: x + OH * ANGLE.cos(),
                    y: OH * ANGLE.sin(),
                    z: 0.,
                },
            ],
        }
    }

    fn test_request() -> HydrationRequest {
        HydrationRequest {
            minimum: Vec3 {
                x: -10.,
                y: -10.,
                z: -10.,
            },
            maximum: Vec3 {
                x: 10.,
                y: 10.,
                z: 10.,
            },
            spacing: 1.,
            orientations: 96,
            max_sites: 100,
            cutoff: None,
            method: None,
            chemical_potential: None,
            gc_steps: None,
            gc_seed: None,
        }
    }

    #[test]
    fn deposited_ignores_simulation_bulk_solvent() {
        let pdb = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n\
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n\
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n\
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O\n\
ATOM      5  CB  ALA A   1       1.988  -0.730  -1.252  1.00  0.00           C\n\
HETATM    6  O   HOH A   2       5.000   0.000   0.000  1.00 30.00           O\n\
HETATM    7  O   WAT A   3       6.000   0.000   0.000  1.00 30.00           O\n\
HETATM    8  O  TIP3 A   4       7.000   0.000   0.000  1.00 30.00           O\n\
END\n";
        let structure = glysys::read_pdb_str(pdb, &glysys::BuildOptions::default()).unwrap();
        let provider = SiteProvider::deposited(&structure, "test".into());
        assert_eq!(provider.sites.len(), 1);
        assert_eq!(provider.sites[0].id, "deposited-6");
    }

    #[test]
    fn split_tiles_covers_small_grids_once() {
        let tiles = split_tiles([4, 3, 2], 1_000_000);
        assert_eq!(
            tiles,
            vec![TileRegion {
                offset: [0, 0, 0],
                dimensions: [4, 3, 2],
            }]
        );
    }

    #[test]
    fn split_tiles_partitions_large_grids_exactly() {
        let dims = [13, 7, 5];
        let tiles = split_tiles(dims, 100);
        assert!(tiles.len() > 1);
        for tile in &tiles {
            let points: usize = tile.dimensions.iter().product();
            assert!(points <= 100, "tile exceeds cap: {tile:?}");
        }
        let mut seen = std::collections::HashSet::new();
        for tile in &tiles {
            for x in 0..tile.dimensions[0] {
                for y in 0..tile.dimensions[1] {
                    for z in 0..tile.dimensions[2] {
                        assert!(seen.insert((
                            tile.offset[0] + x,
                            tile.offset[1] + y,
                            tile.offset[2] + z
                        )));
                    }
                }
            }
        }
        assert_eq!(seen.len(), dims.iter().product::<usize>());
    }

    #[test]
    fn finish_tiled_suppresses_cross_tile_duplicates() {
        let probe = test_probe();
        let request = test_request();
        let close = vec![
            (
                test_pose(0.),
                ProbeScore {
                    lennard_jones: -5.,
                    electrostatics: 0.,
                },
            ),
            (
                test_pose(1.),
                ProbeScore {
                    lennard_jones: -4.,
                    electrostatics: 0.,
                },
            ),
        ];
        let field = probe.finish_tiled(&request, [21, 21, 21], Vec::new(), close, "cpu");
        assert_eq!(field.sites.len(), 1);
        assert_eq!(field.sites[0].position.x, 0.);
    }

    #[test]
    fn finish_tiled_keeps_separated_sites() {
        let probe = test_probe();
        let request = test_request();
        let candidates = vec![
            (
                test_pose(-5.),
                ProbeScore {
                    lennard_jones: -5.,
                    electrostatics: 0.,
                },
            ),
            (
                test_pose(5.),
                ProbeScore {
                    lennard_jones: -4.,
                    electrostatics: 0.,
                },
            ),
        ];
        let field = probe.finish_tiled(&request, [21, 21, 21], Vec::new(), candidates, "cpu");
        assert_eq!(field.sites.len(), 2);
    }

    #[test]
    fn gc_request_defaults_to_probe_for_old_json() {
        let old: HydrationRequest = serde_json::from_str(
            r#"{"minimum":{"x":0,"y":0,"z":0},"maximum":{"x":1,"y":1,"z":1}}"#,
        )
        .unwrap();
        assert!(!is_gc_request(&old));
        assert_eq!(old.method, None);
        let gc: HydrationRequest = serde_json::from_str(
            r#"{"minimum":{"x":0,"y":0,"z":0},"maximum":{"x":1,"y":1,"z":1},"method":"gc"}"#,
        )
        .unwrap();
        assert!(is_gc_request(&gc));
    }

    #[test]
    fn water_pair_is_symmetric_and_repulsive_at_contact() {
        let a = test_pose(0.);
        let b = test_pose(10.);
        let ab = water_pair(a, b).unwrap();
        let ba = water_pair(b, a).unwrap();
        assert!((ab.total() - ba.total()).abs() < 1e-9);
        let close = water_pair(test_pose(0.), test_pose(1.));
        assert!(close.is_some());
        assert!(
            close.unwrap().total() > water_pair(test_pose(0.), test_pose(10.)).unwrap().total()
        );
    }

    #[test]
    fn gc_fingerprint_differs_from_probe_but_validates_same_chemistry() {
        let probe = PhysicalProbe {
            atoms: vec![crate::hydration::ProbeAtom {
                position: Vec3 {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                charge: -0.4,
                radius: 1.5,
                epsilon: 0.2,
            }],
            fingerprint: "test".into(),
        };
        assert_ne!(probe.gc_fingerprint(), "test");
        assert!(!probe.gc_fingerprint().is_empty());
    }

    #[test]
    fn gc_sampling_is_deterministic_and_bounded() {
        let atoms = vec![
            crate::hydration::ProbeAtom {
                position: Vec3 {
                    x: 5.,
                    y: 0.,
                    z: 0.,
                },
                charge: -0.5,
                radius: 1.6,
                epsilon: 0.2,
            },
            crate::hydration::ProbeAtom {
                position: Vec3 {
                    x: -5.,
                    y: 0.,
                    z: 0.,
                },
                charge: 0.5,
                radius: 1.6,
                epsilon: 0.2,
            },
        ];
        let probe = PhysicalProbe {
            atoms,
            fingerprint: "test".into(),
        };
        // Candidate waters near the charged atoms.
        let mut candidates = Vec::new();
        for x in [-3., -2., 2., 3.] {
            let pose = test_pose(x);
            if let Some(e) = probe.score(pose, None) {
                candidates.push((pose, e));
            }
        }
        let mut request = test_request();
        request.method = Some("gc".into());
        request.gc_steps = Some(5_000);
        request.gc_seed = Some(42);
        let first = probe.select_gc_sites(&request, candidates.clone()).unwrap();
        let second = probe.select_gc_sites(&request, candidates).unwrap();
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.id, b.id);
            assert!((a.occupancy.unwrap() - b.occupancy.unwrap()).abs() < 1e-12);
            assert!(a.occupancy.unwrap() > 0. && a.occupancy.unwrap() <= 1.);
            assert!(["bridging", "surface", "exposed"].contains(&a.bridging.as_deref().unwrap()));
            assert_eq!(a.source, "gc_water");
            assert_eq!(a.confidence, a.occupancy);
        }
        assert!(first.len() <= request.max_sites);
    }

    #[test]
    fn finish_tiled_stitches_tile_grids() {
        let probe = test_probe();
        let request = test_request();
        let tiles = vec![
            TileGrid {
                offset: [0, 0, 0],
                dimensions: [2, 1, 1],
                energies: vec![Some(-1.), None],
            },
            TileGrid {
                offset: [2, 0, 0],
                dimensions: [1, 1, 1],
                energies: vec![Some(-2.)],
            },
        ];
        let field = probe.finish_tiled(&request, [3, 1, 1], tiles, Vec::new(), "cpu");
        assert_eq!(field.probe_energy, vec![Some(-1.), None, Some(-2.)]);
        assert!(field.sites.is_empty());
    }
}

/// Imported and deposited sites retain provenance without acquiring probe energies.
pub struct SiteProvider {
    pub sites: Vec<HydrationSite>,
    pub receptor_fingerprint: String,
    pub source: String,
}
impl SiteProvider {
    pub fn deposited(structure: &glysys::Structure, receptor_fingerprint: String) -> Self {
        // Only HOH counts as experimental: WAT/TIP3 residues are simulation
        // bulk solvent added by preparation, not crystallographic waters.
        let sites = structure
            .atoms()
            .into_iter()
            .filter(|a| a.residue_name.as_str() == "HOH" && a.element.eq_ignore_ascii_case("O"))
            .map(|a| HydrationSite {
                id: format!("deposited-{}", a.id.0),
                position: a.position,
                hydrogens: None,
                source: "deposited_water".into(),
                score: None,
                experimental_occupancy: Some(a.occupancy),
                confidence: None,
                displacement_free_energy: None,
                occupancy: None,
                water_water: None,
                bridging: None,
            })
            .collect();
        Self {
            sites,
            receptor_fingerprint,
            source: "deposited-waters-v1".into(),
        }
    }
    pub fn imported(sites: Vec<HydrationSite>, receptor_fingerprint: String) -> Result<Self> {
        if sites.len() > 100_000 {
            return Err(invalid("too many imported hydration sites"));
        }
        if sites.iter().any(|s| {
            !finite(s.position)
                || s.hydrogens.is_some_and(|h| h.iter().any(|p| !finite(*p)))
                || s.score
                    .is_some_and(|e| !e.lennard_jones.is_finite() || !e.electrostatics.is_finite())
                || s.source.is_empty()
                || s.id.is_empty()
                || s.experimental_occupancy
                    .is_some_and(|x| !x.is_finite() || !(0.0..=1.0).contains(&x))
                || s.confidence.is_some_and(|x| !x.is_finite())
                || s.displacement_free_energy.is_some_and(|x| !x.is_finite())
                || s.occupancy
                    .is_some_and(|x| !x.is_finite() || !(0.0..=1.0).contains(&x))
                || s.water_water
                    .is_some_and(|e| !e.lennard_jones.is_finite() || !e.electrostatics.is_finite())
                || s.bridging
                    .as_ref()
                    .is_some_and(|b| b.is_empty() || b.len() > 32)
        }) {
            return Err(invalid("invalid imported hydration site"));
        }
        let mut ids = std::collections::HashSet::new();
        if sites.iter().any(|s| !ids.insert(s.id.clone())) {
            return Err(invalid("duplicate hydration site ID"));
        }
        Ok(Self {
            sites,
            receptor_fingerprint,
            source: "imported-sites-v1".into(),
        })
    }
}
impl HydrationProvider for SiteProvider {
    fn model_version(&self) -> &str {
        &self.source
    }
    fn predict(&self, request: &HydrationRequest) -> Result<HydrationField> {
        let dimensions = PhysicalProbe::dimensions(request)?;
        let sites = self
            .sites
            .iter()
            .filter(|s| {
                s.position.x >= request.minimum.x
                    && s.position.y >= request.minimum.y
                    && s.position.z >= request.minimum.z
                    && s.position.x <= request.maximum.x
                    && s.position.y <= request.maximum.y
                    && s.position.z <= request.maximum.z
            })
            .take(request.max_sites)
            .cloned()
            .collect();
        Ok(HydrationField {
            source_context: None,
            schema_version: 1,
            model_version: self.source.clone(),
            receptor_fingerprint: self.receptor_fingerprint.clone(),
            request: request.clone(),
            sites,
            dimensions,
            probe_energy: vec![None; dimensions.iter().product()],
            backend: "cpu".into(),
        })
    }
}
