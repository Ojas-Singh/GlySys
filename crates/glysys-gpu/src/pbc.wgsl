//! Canonical periodic-boundary nonbonded kernels (CutoffPeriodic-style).
//!
//! This is the single implementation used by the browser laboratory engine
//! and by the native validation harness (`tests/pbc.rs`): no shader forks.
//! The execution model mirrors the CPU reference (`glysys-energy::pbc`):
//! exact-fit cells over the periodic box, Verlet pair enumeration within
//! cutoff+skin, and physics evaluated strictly inside the cutoff with the
//! minimum-image convention.
//!
//! Binding budget: one uniform plus five storage bindings, within the
//! WebGPU-guaranteed eight storage buffers per shader stage, so this runs
//! on strict browsers as well as native Vulkan. Logical arrays share two
//! backing regions: interleaved per-atom data (`sys`: params, coords) and
//! packed u32 metadata (`meta`: cell heads, linked-list next, specials
//! ranges, pair counter).
//!
//! Determinism: pair discovery order varies (linked-list insertion races),
//! so every thread gathers its neighbor indices, sorts them by atom index,
//! and only then accumulates. Per-thread partials reduce through a fixed
//! indexing tree. Identical inputs therefore yield identical bits on a
//! given device. Halving pair energy per side is exact in binary floating
//! point, so the i/j split introduces no rounding asymmetry.
//!
//! Electrostatics dispatch: `electro.x` selects the method (0 = reaction
//! field). RF constants are precomputed on the host in f64 and passed as
//! f32; PME later adds a backend variant reusing these lists, traversal,
//! and integration — the pair-list and integrator code paths do not change.
//! Unsupported methods are rejected on the host before dispatch.
struct PbcConfig {
  dims: vec4<u32>,    // n_atoms, nx, ny, nz
  box_: vec4<f32>,    // Lx, Ly, Lz, limit (cutoff + skin)
  electro: vec4<f32>, // method (0 = RF), cutoff, krf, crf
  misc: vec4<f32>,    // compute_gradients (0/1), max_pairs, cell_count, spare
  counts: vec4<u32>,  // bonds, angles, torsion-pairs, restraints
  dynamic_: vec4<f32>, // waters, solute constraints, dt/ps, friction/ps
  thermo: vec4<f32>,  // temperature/K, inverse drift interval multiplier, adjacency offset, spare
}
// sys[2i] = params (charge, sigma, epsilon, unused); sys[2i+1] = coords.
@group(0) @binding(0) var<uniform> config: PbcConfig;
@group(0) @binding(1) var<storage, read_write> sys: array<vec4<f32>>;
// meta layout: [0..ncells) cell heads, [ncells..ncells+n) next links,
// [ncells+n..ncells+3n) specials ranges (2/atom), [ncells+3n] pair counter,
// [ncells+3n+1] numerical status flag.
// One u32 region; atomic-typed so head insertion and the pair counter can
// use atomics. Plain reads/writes go through atomicLoad/atomicStore, which
// keeps a single binding instead of splitting scalar metadata out.
@group(0) @binding(2) var<storage, read_write> aux: array<atomic<u32>>;
// Same 16-byte layout as device::Special: (other, scee, scnb, spare).
// Excluded pairs carry (0, 0) scales; 1-4 pairs carry their Amber scales.
struct Special { other: u32, scee: f32, scnb: f32, spare: u32 }
@group(0) @binding(3) var<storage, read> specials: array<Special>;
@group(0) @binding(4) var<storage, read_write> pairs: array<u32>;
// out layout: [0..n) nonbonded partials, [n..2n) total gradients,
// [2n..3n) intermolecular pair-virial partials, [3n..4n) bonded partials,
// [4n] nonbonded totals, [4n+1] pair virial total, [5n..6n) step positions.
@group(0) @binding(5) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> bonded: array<vec4<f32>>;
// Molecule-centered coordinates. Bonded terms are translation invariant;
// using these avoids f32 cancellation from large absolute laboratory coords.
@group(0) @binding(7) var<storage, read_write> bcoords: array<vec4<f32>>;
// state: [0..n) velocities xyz+w, [n..2n) positions at the start of a step.
@group(0) @binding(8) var<storage, read_write> state: array<vec4<f32>>;
const COULOMB: f32 = 332.063713299;
const ACCEL: f32 = 418.4;
const U32MAX: u32 = 4294967295u;
// A dense protein interior can legitimately have more than 512 atoms within
// the Verlet radius at a 9 A cutoff.  Keep a bounded local gather, but leave
// enough headroom for those valid systems; the resident host still reports a
// hard capacity error instead of truncating physics.

fn n_atoms() -> u32 { return config.dims.x; }
fn n_cells() -> u32 { return u32(config.misc.z); }
fn head_idx(c: u32) -> u32 { return c; }
fn next_idx(i: u32) -> u32 { return n_cells() + i; }
fn range_idx(i: u32) -> u32 { return n_cells() + n_atoms() + 2u * i; }
fn count_idx() -> u32 { return n_cells() + 3u * n_atoms(); }
fn status_idx() -> u32 { return count_idx() + 1u; }

fn cell_of(p: vec3<f32>) -> vec3<u32> {
  let nx = f32(config.dims.y);
  let ny = f32(config.dims.z);
  let nz = f32(config.dims.w);
  let cx = config.box_.x / nx;
  let cy = config.box_.y / ny;
  let cz = config.box_.z / nz;
  // Bonded terms require unwrapped same-molecule coordinates, so wrap only
  // when assigning an atom to a cell. Minimum-image displacement itself also
  // handles either unwrapped or wrapped endpoints.
  let q = vec3<f32>(
    config.box_.x * floor(p.x / config.box_.x),
    config.box_.y * floor(p.y / config.box_.y),
    config.box_.z * floor(p.z / config.box_.z)
  );
  let w = p - q;
  return vec3<u32>(
    u32(clamp(floor(w.x / cx), 0.0, nx - 1.0)),
    u32(clamp(floor(w.y / cy), 0.0, ny - 1.0)),
    u32(clamp(floor(w.z / cz), 0.0, nz - 1.0)),
  );
}

fn cell_linear(c: vec3<u32>) -> u32 {
  return (c.z * config.dims.z + c.y) * config.dims.y + c.x;
}

// Minimum image of a-b for an orthorhombic box.
fn min_image(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
  var d = a - b;
  d = d - config.box_.xyz * round(d / config.box_.xyz);
  return d;
}

@compute @workgroup_size(64)
fn insert_atoms(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= n_atoms() || atomicLoad(&aux[rebuild_idx()]) == 0u) { return; }
  let c = cell_linear(cell_of(sys[2u * i + 1u].xyz));
  let prev = atomicExchange(&aux[head_idx(c)], i);
  atomicStore(&aux[next_idx(i)], prev);
}

// Looks up (scee, scnb, is_exception) for pair (i, j): (0, 0, _) means
// excluded; is_exception is 1 for 1-4 pairs (plain-Coulomb exceptions) and 0
// otherwise. Regular pairs (absent from the table) return (1, 1, 0).
fn special_scale(i: u32, j: u32) -> vec3<f32> {
  let r0 = atomicLoad(&aux[range_idx(i)]);
  let r1 = atomicLoad(&aux[range_idx(i) + 1u]);
  for (var k = r0; k < r1; k++) {
    if (specials[k].other == j) {
      return vec3<f32>(specials[k].scee, specials[k].scnb, f32(specials[k].spare));
    }
  }
  return vec3<f32>(1.0, 1.0, 0.0);
}

fn rebuild_idx() -> u32 { return status_idx() + 1u; }
fn rebuild_count_idx() -> u32 { return status_idx() + 2u; }
fn reference_idx(i: u32) -> u32 { return status_idx() + 3u + 3u * i; }
fn neighbor_count_idx(i: u32) -> u32 { return reference_idx(n_atoms()) + i; }
fn neighbor_offset_idx(i: u32) -> u32 { return neighbor_count_idx(n_atoms()) + i; }
fn block_idx(i: u32) -> u32 { return neighbor_offset_idx(n_atoms() + 1u) + i; }

@compute @workgroup_size(64)
fn check_neighbors(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= n_atoms()) { return; }
  let p = sys[2u * i + 1u].xyz;
  if (!all(abs(p) < vec3<f32>(1e20))) { atomicStore(&aux[status_idx()], 4u); return; }
  let k = reference_idx(i);
  let old = bitcast<vec3<f32>>(vec3<u32>(atomicLoad(&aux[k]), atomicLoad(&aux[k + 1u]), atomicLoad(&aux[k + 2u])));
  let d = min_image(p, old);
  let half_skin = 0.5 * (config.box_.w - config.electro.y);
  if (dot(d, d) >= half_skin * half_skin) { atomicStore(&aux[rebuild_idx()], 1u); }
}

// Count and fill traverse the same exact cell stencil. CSR capacity is
// checked after prefix scan, before any index write is permitted.
fn visit_neighbors(i: u32, write: bool) -> u32 {
  var count = 0u;
  let start = atomicLoad(&aux[neighbor_offset_idx(i)]);
  let pi = sys[2u * i + 1u].xyz;
  let c0 = cell_of(pi);
  let nx = config.dims.y;
  let ny = config.dims.z;
  let nz = config.dims.w;
  let limit2 = config.box_.w * config.box_.w;
  // Distinct neighbor cells only: with fewer than 3 cells per axis the raw
  // +-1 stencil revisits cells (the CPU reference dedups afterwards).
  let ex = min(nx, 3u);
  let ey = min(ny, 3u);
  let ez = min(nz, 3u);
  for (var dx = 0u; dx < ex; dx++) {
    for (var dy = 0u; dy < ey; dy++) {
      for (var dz = 0u; dz < ez; dz++) {
        let c = vec3<u32>((c0.x + dx + nx - 1u) % nx, (c0.y + dy + ny - 1u) % ny, (c0.z + dz + nz - 1u) % nz);
        var j = atomicLoad(&aux[head_idx(cell_linear(c))]);
        while (j != U32MAX) {
          if (j != i) {
            let d = min_image(pi, sys[2u * j + 1u].xyz);
            if (dot(d, d) <= limit2) {
              if (write) { pairs[start + count] = j; }
              count += 1u;
            }
          }
          j = atomicLoad(&aux[next_idx(j)]);
        }
      }
    }
  }
  return count;
}

@compute @workgroup_size(64)
fn count_neighbors(@builtin(global_invocation_id) id: vec3<u32>) {
  if (atomicLoad(&aux[rebuild_idx()]) == 0u || id.x >= n_atoms()) { return; }
  atomicStore(&aux[neighbor_count_idx(id.x)], visit_neighbors(id.x, false));
}

var<workgroup> scan_counts: array<u32, 64>;
@compute @workgroup_size(64)
fn scan_neighbors(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  // No data-dependent exit before barriers; inactive jobs scan zeros.
  var count = 0u;
  if (id.x < n_atoms() && atomicLoad(&aux[rebuild_idx()]) != 0u) { count = atomicLoad(&aux[neighbor_count_idx(id.x)]); }
  scan_counts[lane] = count;
  workgroupBarrier();
  for (var offset = 1u; offset < 64u; offset *= 2u) {
    var addend = 0u;
    if (lane >= offset) { addend = scan_counts[lane - offset]; }
    workgroupBarrier();
    scan_counts[lane] += addend;
    workgroupBarrier();
  }
  if (atomicLoad(&aux[rebuild_idx()]) != 0u) {
    if (id.x < n_atoms()) { atomicStore(&aux[neighbor_offset_idx(id.x)], scan_counts[lane] - count); }
    if (lane == 63u) { atomicStore(&aux[block_idx(group.x)], scan_counts[63]); }
  }
}

@compute @workgroup_size(1)
fn scan_neighbor_blocks() {
  if (atomicLoad(&aux[rebuild_idx()]) == 0u) { return; }
  var sum = 0u;
  let capacity = 2u * u32(config.misc.y);
  for (var block = 0u; block < (n_atoms() + 63u) / 64u; block++) {
    let count = atomicLoad(&aux[block_idx(block)]);
    atomicStore(&aux[block_idx(block)], sum);
    // Saturate before addition: a dense rejected system must not wrap the
    // global count and appear to fit the bounded neighbor buffer.
    if (sum > capacity || count > capacity - min(sum, capacity)) {
      sum = capacity + 1u;
    } else { sum += count; }
  }
  atomicStore(&aux[neighbor_offset_idx(n_atoms())], sum);
  atomicStore(&aux[count_idx()], sum);
  if (sum > 2u * u32(config.misc.y)) { atomicStore(&aux[status_idx()], 2u); }
}

@compute @workgroup_size(64)
fn apply_neighbor_offsets(@builtin(global_invocation_id) id: vec3<u32>) {
  if (atomicLoad(&aux[rebuild_idx()]) == 0u || id.x >= n_atoms()) { return; }
  let local = atomicLoad(&aux[neighbor_offset_idx(id.x)]);
  atomicStore(&aux[neighbor_offset_idx(id.x)], local + atomicLoad(&aux[block_idx(id.x / 64u)]));
}

@compute @workgroup_size(64)
fn fill_neighbors(@builtin(global_invocation_id) id: vec3<u32>) {
  if (atomicLoad(&aux[rebuild_idx()]) == 0u || atomicLoad(&aux[status_idx()]) != 0u || id.x >= n_atoms()) { return; }
  let count = visit_neighbors(id.x, true);
}

fn sift_neighbors(start: u32, root_in: u32, length: u32) {
  var root = root_in;
  loop {
    var child = 2u * root + 1u;
    if (child >= length) { break; }
    if (child + 1u < length && pairs[start + child] < pairs[start + child + 1u]) { child += 1u; }
    if (pairs[start + root] >= pairs[start + child]) { break; }
    let tmp = pairs[start + root]; pairs[start + root] = pairs[start + child]; pairs[start + child] = tmp;
    root = child;
  }
}

@compute @workgroup_size(64)
fn sort_neighbors(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (atomicLoad(&aux[rebuild_idx()]) == 0u || atomicLoad(&aux[status_idx()]) != 0u || i >= n_atoms()) { return; }
  let start = atomicLoad(&aux[neighbor_offset_idx(i)]);
  let count = atomicLoad(&aux[neighbor_offset_idx(i + 1u)]) - start;
  for (var root = count / 2u; root > 0u; root--) { sift_neighbors(start, root - 1u, count); }
  for (var end = count; end > 1u; end--) {
    let tmp = pairs[start]; pairs[start] = pairs[start + end - 1u]; pairs[start + end - 1u] = tmp;
    sift_neighbors(start, 0u, end - 1u);
  }
  let p = bitcast<vec3<u32>>(sys[2u * i + 1u].xyz);
  let k = reference_idx(i);
  atomicStore(&aux[k], p.x); atomicStore(&aux[k + 1u], p.y); atomicStore(&aux[k + 2u], p.z);
}

@compute @workgroup_size(1)
fn finish_neighbors() {
  if (atomicLoad(&aux[rebuild_idx()]) != 0u && atomicLoad(&aux[status_idx()]) == 0u) {
    atomicAdd(&aux[rebuild_count_idx()], 1u);
    atomicStore(&aux[rebuild_idx()], 0u);
  }
}

// Pair physics at minimum-image separation d (|d| = r <= cutoff).
// Returns (lj_energy, rf_energy, force_magnitude) with force_magnitude such
// that grad[i] += fmag * d and grad[j] -= fmag * d. 1-4 exceptions
// (is_exception != 0) bypass reaction-field screening with plain Coulomb,
// matching the CPU engine and OpenMM exception convention.
fn pair_terms(ai: vec4<f32>, aj: vec4<f32>, d: vec3<f32>, r: f32, scee: f32, scnb: f32, is_exception: f32) -> vec3<f32> {
  let sig = ai.y + aj.y;
  let eps = sqrt(ai.z * aj.z) / scnb;
  let ratio6 = pow(sig / r, 6.0);
  let lj = eps * (ratio6 * ratio6 - 2.0 * ratio6);
  let qq = COULOMB * ai.x * aj.x / scee;
  var ecoul = 0.0;
  var dcoul = 0.0;
  if (is_exception != 0.0) {
    ecoul = qq / r;
    dcoul = -qq / (r * r);
  } else {
    ecoul = qq * (1.0 / r + config.electro.z * r * r - config.electro.w);
    dcoul = qq * (-1.0 / (r * r) + 2.0 * config.electro.z * r);
  }
  let flj = 12.0 * eps * (ratio6 - ratio6 * ratio6) / r;
  return vec3<f32>(lj, ecoul, (flj + dcoul) / r);
}

@compute @workgroup_size(64)
fn eval(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= n_atoms()) { return; }
  let pi = sys[2u * i + 1u].xyz;
  let ai = sys[2u * i];
  let cutoff = config.electro.y;
  let cutoff2 = cutoff * cutoff;
  let start = atomicLoad(&aux[neighbor_offset_idx(i)]);
  let end = atomicLoad(&aux[neighbor_offset_idx(i + 1u)]);
  let overflow = atomicLoad(&aux[status_idx()]) != 0u;
  let count = select(end - start, 0u, overflow);
  var elj = 0.0;
  var erf = 0.0;
  var g = vec3<f32>(0.0);
  var evaluated = 0.0;
  var virial = 0.0;
  var pair_virial = 0.0;
  for (var k = 0u; k < count; k++) {
    let j = pairs[start + k];
    let aj = sys[2u * j];
    let sc = special_scale(i, j);
    if (sc.x == 0.0) { continue; }
    let d = min_image(pi, sys[2u * j + 1u].xyz);
    let r2 = dot(d, d);
    if (r2 > cutoff2) { continue; }
    let r = max(sqrt(r2), 1e-8);
    let t = pair_terms(ai, aj, d, r, sc.x, sc.y, sc.z);
    // Halving is exact in binary FP: symmetric split, no rounding asymmetry.
    elj += 0.5 * t.x;
    erf += 0.5 * t.y;
    // Keep the historical total virial in the gradient lane for CPU/GPU
    // force-parity consumers. A molecular volume move translates each
    // molecule as a rigid body, so its configurational derivative uses only
    // the separate intermolecular lane below.
    virial -= 0.5 * t.z * r2;
    if (bitcast<u32>(bcoords[i].w) != bitcast<u32>(bcoords[j].w)) {
      pair_virial -= 0.5 * t.z * r2;
    }
    if (config.misc.x != 0.0) {
      g += t.z * d;
    }
    evaluated += 1.0;
  }
  out[i] = vec4<f32>(elj, erf, evaluated, select(0.0, 1.0, overflow));
  if (config.misc.x != 0.0) {
    out[n_atoms() + i] = vec4<f32>(g, virial);
    out[2u * n_atoms() + i] = vec4<f32>(0.0, 0.0, 0.0, pair_virial);
  }
}

fn angle_gradient(a: vec3<f32>, c: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
  let u = a - c;
  let v = b - c;
  let ru = max(length(u), 1e-8);
  let rv = max(length(v), 1e-8);
  let cos_t = clamp(dot(u, v) / (ru * rv), -1.0, 1.0);
  let theta = acos(cos_t);
  let sin_t = max(sin(theta), 1e-8);
  let f = -1.0 / sin_t;
  let gu = f * (v / (ru * rv) - cos_t * u / (ru * ru));
  let gv = f * (u / (ru * rv) - cos_t * v / (rv * rv));
  let gc = -gu - gv;
  return mat3x3<f32>(gu, gc, gv);
}

fn dihedral_phi(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>) -> f32 {
  let b0 = p1 - p0;
  let b1 = p2 - p1;
  let b2 = p3 - p2;
  let n1 = cross(b0, b1);
  let n2 = cross(b1, b2);
  return atan2(dot(cross(n1, n2), normalize(b1)), dot(n1, n2));
}

fn dihedral_gradient(
  p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>,
  de: f32,
  g0: ptr<function, vec3<f32>>, g1: ptr<function, vec3<f32>>,
  g2: ptr<function, vec3<f32>>, g3: ptr<function, vec3<f32>>
) {
  let b0 = p1 - p0;
  let b1 = p2 - p1;
  let b2 = p3 - p2;
  let n1 = cross(b0, b1);
  let n2 = cross(b1, b2);
  let norm_b1 = max(length(b1), 1e-12);
  let n1_2 = max(dot(n1, n1), 1e-16);
  let n2_2 = max(dot(n2, n2), 1e-16);
  // OpenMM Reference Cartesian force decomposition, returned here as
  // gradients (negative forces). The four b vectors follow p1-p0, p2-p1,
  // and p3-p2.
  let force0 = de * norm_b1 / n1_2 * n1;
  let force3 = -de * norm_b1 / n2_2 * n2;
  let b1_2 = max(dot(b1, b1), 1e-16);
  let f1 = -dot(b0, b1) / b1_2;
  let f2 = -dot(b2, b1) / b1_2;
  let s = f1 * force0 - f2 * force3;
  *g0 = -force0;
  *g1 = force0 - s;
  *g2 = force3 + s;
  *g3 = -force3;
}

fn bonded_position(atom: u32) -> vec3<f32> { return bcoords[atom].xyz; }

fn unit(v: vec3<f32>) -> vec3<f32> {
  let n = length(v);
  if (n < 1e-14) { return vec3<f32>(0.0); }
  return v / n;
}

fn project(v: vec3<f32>, x_axis: vec3<f32>, y_axis: vec3<f32>, z_axis: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(x_axis, v), dot(y_axis, v), dot(z_axis, v));
}

fn unproject(v: vec3<f32>, x_axis: vec3<f32>, y_axis: vec3<f32>, z_axis: vec3<f32>) -> vec3<f32> {
  return x_axis * v.x + y_axis * v.y + z_axis * v.z;
}

// Reset the resident cell heads and pair counter before a new force step.
// Coordinate uploads perform the same reset on the host; the dynamic loop
// uses this bounded dispatch so positions and velocities stay resident.
@compute @workgroup_size(64)
fn clear_meta(@builtin(global_invocation_id) id: vec3<u32>) {
  if (atomicLoad(&aux[rebuild_idx()]) == 0u) { return; }
  if (id.x < n_cells()) {
    atomicStore(&aux[head_idx(id.x)], U32MAX);
  }
  if (id.x == 0u) {
    atomicStore(&aux[count_idx()], 0u);
  }
}

@compute @workgroup_size(64)
fn integrate_first(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  let n = n_atoms();
  if (i >= n) { return; }
  let dt = config.dynamic_.z;
  let mass = max(sys[2u * i].w, 1e-12);
  var velocity = state[i].xyz;
  velocity += -0.5 * dt * ACCEL / mass * out[n + i].xyz;
  let old_position = sys[2u * i + 1u].xyz;
  let position = sys[2u * i + 1u].xyz + velocity * dt;
  state[i] = vec4<f32>(velocity, state[i].w);
  state[n + i] = vec4<f32>(old_position, 0.0);
  out[5u * n + i] = vec4<f32>(old_position, 0.0);
  sys[2u * i + 1u] = vec4<f32>(position, 0.0);
}

// First A/2 drift for the resident BAOAB Langevin path. The constrained
// SETTLE pass that follows uses config.thermo.y as the inverse drift multiplier
// (2 for a half-step) when applying its position impulse to velocities.
@compute @workgroup_size(64)
fn integrate_first_half(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  let n = n_atoms();
  if (i >= n) { return; }
  let dt = config.dynamic_.z;
  let mass = max(sys[2u * i].w, 1e-12);
  var velocity = state[i].xyz;
  velocity += -0.5 * dt * ACCEL / mass * out[n + i].xyz;
  let old_position = sys[2u * i + 1u].xyz;
  let position = old_position + velocity * (0.5 * dt);
  state[i] = vec4<f32>(velocity, state[i].w);
  state[n + i] = vec4<f32>(old_position, 0.0);
  sys[2u * i + 1u] = vec4<f32>(position, 0.0);
}

// Stateful per-atom OU thermostat. The evolving state word is the only
// mutable random state, so independent runs and checkpoint/resume do not need
// a host-side random upload each step.
@compute @workgroup_size(64)
fn langevin_ou(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  let n = n_atoms();
  if (i >= n) { return; }
  let friction = max(config.dynamic_.w, 0.0);
  let dt = config.dynamic_.z;
  let decay = exp(-friction * dt);
  let mass = max(sys[2u * i].w, 1e-12);
  let sigma = sqrt(max((1.0 - decay * decay) * 0.00198720425864083
      * config.thermo.x * ACCEL / mass, 0.0));
  let random = rng_normal3(bitcast<u32>(state[i].w));
  state[i] = vec4<f32>(decay * state[i].xyz + sigma * random.xyz, random.w);
}

// Second A/2 drift after the OU thermostat. It replaces the old position in
// the state buffer so SETTLE can apply the same displacement-based velocity
// impulse as on the first half-step.
@compute @workgroup_size(64)
fn integrate_second_half(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  let n = n_atoms();
  if (i >= n) { return; }
  let dt = config.dynamic_.z;
  let old_position = sys[2u * i + 1u].xyz;
  let position = old_position + state[i].xyz * (0.5 * dt);
  state[n + i] = vec4<f32>(old_position, 0.0);
  sys[2u * i + 1u] = vec4<f32>(position, 0.0);
}

@compute @workgroup_size(64)
fn refresh_bonded_coords(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= n_atoms()) { return; }
  let anchor = u32(bitcast<u32>(bcoords[i].w));
  let relative = sys[2u * i + 1u].xyz - sys[2u * anchor + 1u].xyz;
  bcoords[i] = vec4<f32>(relative, bcoords[i].w);
}

@compute @workgroup_size(64)
fn settle(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = n_atoms();
  let water_base = config.counts.x + 2u * config.counts.y + 2u * config.counts.z;
  let solute_base = water_base + 2u * u32(config.dynamic_.x);
  if (id.x == 0u) {
    let inv_dt = config.thermo.y / max(config.dynamic_.z, 1e-12);
    // Solute X-H bonds use bounded sequential SHAKE, matching the CPU path.
    for (var iteration = 0u; iteration < 12u; iteration++) {
      for (var k = 0u; k < u32(config.dynamic_.y); k++) {
        let t = bonded[solute_base + k];
        let a = u32(t.x); let b = u32(t.y);
        let d = sys[2u * a + 1u].xyz - sys[2u * b + 1u].xyz;
        let r = max(length(d), 1e-12);
      if (abs(r - t.z) < 1e-8) { continue; }
        let ia = 1.0 / max(sys[2u * a].w, 1e-12);
        let ib = 1.0 / max(sys[2u * b].w, 1e-12);
        let lambda = (r - t.z) / (r * (ia + ib));
        let da = -lambda * ia * d;
        let db = lambda * ib * d;
        sys[2u * a + 1u] = vec4<f32>(sys[2u * a + 1u].xyz + da, 0.0);
        sys[2u * b + 1u] = vec4<f32>(sys[2u * b + 1u].xyz + db, 0.0);
        state[a] = vec4<f32>(state[a].xyz + da * inv_dt, state[a].w);
        state[b] = vec4<f32>(state[b].xyz + db * inv_dt, state[b].w);
      }
    }
  }
  if (id.x >= u32(config.dynamic_.x)) { return; }

  let head = bonded[water_base + 2u * id.x];
  let geom = bonded[water_base + 2u * id.x + 1u];
  let o = u32(head.x); let h1 = u32(head.y); let h2 = u32(head.z);
  let doh = 0.5 * (head.w + geom.x);
  let dhh = geom.y;
  let mo = geom.z; let mh1 = geom.w; let mh2 = geom.w;
  let total = mo + mh1 + mh2;
  let old_o = state[n + o].xyz; let old_1 = state[n + h1].xyz; let old_2 = state[n + h2].xyz;
  let qo = sys[2u * o + 1u].xyz; let q1 = sys[2u * h1 + 1u].xyz; let q2 = sys[2u * h2 + 1u].xyz;
  let xp0 = qo - old_o;
  let xp1 = q1 - old_1;
  let xp2 = q2 - old_2;
  let xb0 = old_1 - old_o;
  let xc0 = old_2 - old_o;
  let xcom = (xp0 * mo + (xb0 + xp1) * mh1 + (xc0 + xp2) * mh2) / total;
  let xa1 = xp0 - xcom;
  let xb1 = xb0 + xp1 - xcom;
  let xc1 = xc0 + xp2 - xcom;
  let zaks = cross(xb0, xc0);
  let xaks = cross(xa1, zaks);
  let yaks = cross(zaks, xaks);
  let z_axis = unit(zaks);
  let x_axis = unit(xaks);
  let y_axis = unit(yaks);
  let xb0d3 = project(xb0, x_axis, y_axis, z_axis);
  let xc0d3 = project(xc0, x_axis, y_axis, z_axis);
  let xa1d3 = project(xa1, x_axis, y_axis, z_axis);
  let xb1d3 = project(xb1, x_axis, y_axis, z_axis);
  let xc1d3 = project(xc1, x_axis, y_axis, z_axis);
  let xb0d = xb0d3.x; let yb0d = xb0d3.y;
  let xc0d = xc0d3.x; let yc0d = xc0d3.y;
  let za1d = xa1d3.z;
  let xb1d = xb1d3.x; let yb1d = xb1d3.y; let zb1d = xb1d3.z;
  let xc1d = xc1d3.x; let yc1d = xc1d3.y; let zc1d = xc1d3.z;
  let rc = 0.5 * dhh;
  let rb0_sq = doh * doh - rc * rc;
  let rb0 = sqrt(max(rb0_sq, 0.0));
  let ra = rb0 * (mh1 + mh2) / total;
  let rb = rb0 - ra;
  let sinphi = za1d / max(ra, 1e-12);
  let cosphi = sqrt(max(1.0 - sinphi * sinphi, 0.0));
  let sinpsi = (zb1d - zc1d) / (2.0 * rc * max(cosphi, 1e-12));
  let cospsi = sqrt(max(1.0 - sinpsi * sinpsi, 0.0));
  let ya2d = ra * cosphi;
  var xb2d = -rc * cospsi;
  let yb2d = -rb * cosphi - rc * sinpsi * sinphi;
  let yc2d = -rb * cosphi + rc * sinpsi * sinphi;
  let hh2 = 4.0 * xb2d * xb2d + (yb2d - yc2d) * (yb2d - yc2d) + (zb1d - zc1d) * (zb1d - zc1d);
  let hh_root = 4.0 * xb2d * xb2d - hh2 + dhh * dhh;
  xb2d -= 0.5 * (2.0 * xb2d + sqrt(max(hh_root, 0.0)));
  let alpha = xb2d * (xb0d - xc0d) + yb0d * yb2d + yc0d * yc2d;
  let beta = xb2d * (yc0d - yb0d) + xb0d * yb2d + xc0d * yc2d;
  let gamma = xb0d * yb1d - xb1d * yb0d + xc0d * yc1d - xc1d * yc0d;
  let al2be2 = alpha * alpha + beta * beta;
  let theta_root = al2be2 - gamma * gamma;
  if (abs(sinphi) >= 1.0 || abs(sinpsi) >= 1.0 || al2be2 <= 0.0 || theta_root < 0.0) {
    atomicStore(&aux[status_idx()], 1u);
    return;
  }
  let sintheta = (alpha * gamma - beta * sqrt(theta_root)) / al2be2;
  if (abs(sintheta) >= 1.0) {
    atomicStore(&aux[status_idx()], 1u);
    return;
  }
  let costheta = sqrt(max(1.0 - sintheta * sintheta, 0.0));
  let xa3d = vec3<f32>(-ya2d * sintheta, ya2d * costheta, za1d);
  let xb3d = vec3<f32>(xb2d * costheta - yb2d * sintheta, xb2d * sintheta + yb2d * costheta, zb1d);
  let xc3d = vec3<f32>(-xb2d * costheta - yc2d * sintheta, -xb2d * sintheta + yc2d * costheta, zc1d);
  let xa3 = unproject(xa3d, x_axis, y_axis, z_axis);
  let xb3 = unproject(xb3d, x_axis, y_axis, z_axis);
  let xc3 = unproject(xc3d, x_axis, y_axis, z_axis);
  let d_o = xa3 - xa1; let d_1 = xb3 - xb1; let d_2 = xc3 - xc1;
  let inv_dt = config.thermo.y / max(config.dynamic_.z, 1e-12);
  sys[2u * o + 1u] = vec4<f32>(qo + d_o, 0.0);
  sys[2u * h1 + 1u] = vec4<f32>(q1 + d_1, 0.0);
  sys[2u * h2 + 1u] = vec4<f32>(q2 + d_2, 0.0);
  state[o] = vec4<f32>(state[o].xyz + d_o * inv_dt, state[o].w);
  state[h1] = vec4<f32>(state[h1].xyz + d_1 * inv_dt, state[h1].w);
  state[h2] = vec4<f32>(state[h2].xyz + d_2 * inv_dt, state[h2].w);
}

@compute @workgroup_size(64)
fn kick_second(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= n_atoms()) { return; }
  let dt = config.dynamic_.z;
  let mass = max(sys[2u * i].w, 1e-12);
  state[i] = vec4<f32>(state[i].xyz - 0.5 * dt * ACCEL / mass * out[n_atoms() + i].xyz, state[i].w);
}

@compute @workgroup_size(64)
fn rattle(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = n_atoms();
  let water_base = config.counts.x + 2u * config.counts.y + 2u * config.counts.z;
  let solute_base = water_base + 2u * u32(config.dynamic_.x);
  if (id.x == 0u) {
    for (var iteration = 0u; iteration < 12u; iteration++) {
      for (var k = 0u; k < u32(config.dynamic_.y); k++) {
        let t = bonded[solute_base + k];
        let a = u32(t.x); let b = u32(t.y);
        let d = sys[2u * a + 1u].xyz - sys[2u * b + 1u].xyz;
        let r = max(length(d), 1e-12);
        let rel = state[a].xyz - state[b].xyz;
        let rv = dot(rel, d) / r;
        if (abs(rv) < 1e-7) { continue; }
        let ia = 1.0 / max(sys[2u * a].w, 1e-12);
        let ib = 1.0 / max(sys[2u * b].w, 1e-12);
        let lambda = rv / (r * (ia + ib));
        state[a] = vec4<f32>(state[a].xyz - lambda * ia * d, state[a].w);
        state[b] = vec4<f32>(state[b].xyz + lambda * ib * d, state[b].w);
      }
    }
  }
  if (id.x >= u32(config.dynamic_.x)) { return; }
  let head = bonded[water_base + 2u * id.x];
  let a = u32(head.x); let b = u32(head.y); let c = u32(head.z);
  // Closed-form three-mass RATTLE velocity solve. This is the same
  // mathematical system as settle_velocity_triangle in the CPU reference;
  // keeping it analytic avoids iterative, order-dependent water projection.
  let pa = sys[2u * a + 1u].xyz;
  let pb = sys[2u * b + 1u].xyz;
  let pc = sys[2u * c + 1u].xyz;
  let va = state[a].xyz;
  let vb = state[b].xyz;
  let vc = state[c].xyz;
  let eab = unit(pb - pa);
  let ebc = unit(pc - pb);
  let eca = unit(pa - pc);
  let vab = dot(vb - va, eab);
  let vbc = dot(vc - vb, ebc);
  let vca = dot(va - vc, eca);
  let ca = -dot(eab, eca);
  let cb = -dot(eab, ebc);
  let cc = -dot(ebc, eca);
  let s2a = max(1.0 - ca * ca, 0.0);
  let s2b = max(1.0 - cb * cb, 0.0);
  let s2c = max(1.0 - cc * cc, 0.0);
  let ma = max(sys[2u * a].w, 1e-12);
  let mb = max(sys[2u * b].w, 1e-12);
  let mc = max(sys[2u * c].w, 1e-12);
  let mabc_inv = 1.0 / (ma * mb * mc);
  let denom = (((s2a * mb + s2b * ma) * mc
      + (s2a * mb * mb + 2.0 * (ca * cb * cc + 1.0) * ma * mb + s2b * ma * ma))
      * mc + s2c * ma * mb * (ma + mb)) * mabc_inv;
  let tab = ((cb * cc * ma - ca * mb - ca * mc) * vca
      + (ca * cc * mb - cb * mc - cb * ma) * vbc
      + (s2c * ma * ma * mb * mb * mabc_inv + (ma + mb + mc)) * vab) / denom;
  let tbc = ((ca * cb * mc - cc * mb - cc * ma) * vca
      + (s2a * mb * mb * mc * mc * mabc_inv + (ma + mb + mc)) * vbc
      + (ca * cc * mb - cb * ma - cb * mc) * vab) / denom;
  let tca = ((s2b * ma * ma * mc * mc * mabc_inv + (ma + mb + mc)) * vca
      + (ca * cb * mc - cc * mb - cc * ma) * vbc
      + (cb * cc * ma - ca * mb - ca * mc) * vab) / denom;
  let va_new = va + (eab * tab - eca * tca) / ma;
  let vb_new = vb + (ebc * tbc - eab * tab) / mb;
  let vc_new = vc + (eca * tca - ebc * tbc) / mc;
  state[a] = vec4<f32>(va_new, state[a].w);
  state[b] = vec4<f32>(vb_new, state[b].w);
  state[c] = vec4<f32>(vc_new, state[c].w);
}

@compute @workgroup_size(64)
fn bonded_energy(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  let n = n_atoms();
  if (i >= n) { return; }
  var partial = vec4<f32>(0.0);
  var gradient = select(vec3<f32>(0.0), out[n + i].xyz, config.misc.x != 0.0);
  var virial = select(0.0, out[n + i].w, config.misc.x != 0.0);

  let adjacent = bitcast<vec4<u32>>(bonded[u32(config.thermo.z) + i]);
  let index_base = u32(config.thermo.z) + n;
  for (var entry = adjacent.x; entry < adjacent.y; entry++) {
    let k = bitcast<vec4<u32>>(bonded[index_base + entry / 4u])[entry % 4u];
    let t = bonded[k];
    let a = u32(t.x); let b = u32(t.y);
    if (i != a && i != b) { continue; }
    let d = bonded_position(a) - bonded_position(b);
    let r = max(length(d), 1e-8);
    let e = t.z * (r - t.w) * (r - t.w);
    let f = 2.0 * t.z * (r - t.w) / r;
    partial.x += 0.5 * e;
    gradient += select(-f * d, f * d, i == a);
    virial -= 0.5 * f * dot(d, d);
  }

  for (var entry = adjacent.y; entry < adjacent.z; entry++) {
    let k = bitcast<vec4<u32>>(bonded[index_base + entry / 4u])[entry % 4u];
    let head = bonded[config.counts.x + 2u * k];
    let theta0 = bonded[config.counts.x + 2u * k + 1u].x;
    let a = u32(head.x); let c = u32(head.y); let b = u32(head.z);
    if (i != a && i != c && i != b) { continue; }
    let g = angle_gradient(bonded_position(a), bonded_position(c), bonded_position(b));
    let delta = acos(clamp(dot(normalize(bonded_position(a) - bonded_position(c)), normalize(bonded_position(b) - bonded_position(c))), -1.0, 1.0)) - theta0;
    let e = head.w * delta * delta;
    let f = 2.0 * head.w * delta;
    partial.y += e / 3.0;
    if (i == a) { gradient += f * g[0]; }
    if (i == c) { gradient += f * g[1]; }
    if (i == b) { gradient += f * g[2]; }
  }

  let torsion_base = config.counts.x + 2u * config.counts.y;
  for (var entry = adjacent.z; entry < adjacent.w; entry++) {
    let k = bitcast<vec4<u32>>(bonded[index_base + entry / 4u])[entry % 4u];
    let atoms = bonded[torsion_base + 2u * k];
    let p = bonded[torsion_base + 2u * k + 1u];
    let a0 = u32(atoms.x); let a1 = u32(atoms.y);
    let a2 = u32(atoms.z); let a3 = u32(atoms.w);
    if (i != a0 && i != a1 && i != a2 && i != a3) { continue; }
    let p0 = bonded_position(a0); let p1 = bonded_position(a1);
    let p2 = bonded_position(a2); let p3 = bonded_position(a3);
    var g0 = vec3<f32>(0.0); var g1 = vec3<f32>(0.0);
    var g2 = vec3<f32>(0.0); var g3 = vec3<f32>(0.0);
    let phi = dihedral_phi(p0, p1, p2, p3);
    let arg = p.y * phi - p.z;
    let e = p.x * (1.0 + cos(arg));
    let f = -p.y * p.x * sin(arg);
    dihedral_gradient(p0, p1, p2, p3, f, &g0, &g1, &g2, &g3);
    if (p.w != 0.0) { partial.w += 0.25 * e; } else { partial.z += 0.25 * e; }
    if (i == a0) { gradient += g0; }
    if (i == a1) { gradient += g1; }
    if (i == a2) { gradient += g2; }
    if (i == a3) { gradient += g3; }
  }

  out[3u * n + i] = partial;
  // Angles and torsions are invariant under uniform scaling, so their
  // scalar virial is analytically zero. Pair virials use minimum images;
  // bonds use the same whole-molecule displacement as their force.
  if (config.misc.x != 0.0) { out[n + i] = vec4<f32>(gradient, virial); }
}

var<workgroup> red: array<vec4<f32>, 64>;
var<workgroup> pair_red: array<f32, 64>;
@compute @workgroup_size(64)
fn reduce(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let n = n_atoms();
  // Kahan summation over a fixed strided partition (matches energy.wgsl).
  var acc = vec4<f32>(0.0);
  var comp = vec4<f32>(0.0);
  var pair_acc = 0.0;
  var pair_comp = 0.0;
  var k = lane;
  while (k < n) {
    let v = out[k] - comp;
    let t = acc + v;
    comp = (t - acc) - v;
    acc = t;
    let pv = out[2u * n + k].w - pair_comp;
    let pt = pair_acc + pv;
    pair_comp = (pt - pair_acc) - pv;
    pair_acc = pt;
    k += 64u;
  }
  red[lane] = acc;
  pair_red[lane] = pair_acc;
  workgroupBarrier();
  // Fixed tree reduction over lanes: deterministic for a given n.
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      red[lane] += red[lane + stride];
      pair_red[lane] += pair_red[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    out[4u * n + group.x] = red[0];
    out[4u * n + 1u] = vec4<f32>(pair_red[0], 0.0, 0.0, 0.0);
  }
}
