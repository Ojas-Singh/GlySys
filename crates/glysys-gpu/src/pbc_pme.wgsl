//! Reciprocal-space smooth particle-mesh Ewald (Essmann et al., J. Chem.
//! Phys. 103, 8577 (1995)) for the periodic explicit-solvent engine: the same
//! sum as `glysys_energy::pme::PmeEngine`, in f32 on the device.
//!
//! One evaluation is seven dispatches:
//!
//!   spread          charges onto the mesh with cardinal B-splines of order 4
//!   fft_x_forward   real-to-complex along x, two mesh lines per transform
//!   fft_y           forward along y
//!   fft_z           forward along z, influence function, inverse along z
//!   fft_y           inverse along y
//!   fft_x_inverse   complex-to-real along x, two mesh lines per transform
//!   gather          gradient of every atom from the mesh potential
//!
//! Charges are spread with `atomicAdd` on a mesh of 32-bit two's-complement
//! fixed point (scale 2^26). Integer addition is associative, so the mesh, and
//! with it every later number, does not depend on how the device schedules
//! the atoms. The first transform clears each word as it reads it, which
//! leaves the mesh zero for the next evaluation without a clear dispatch.
//!
//! The mesh is real, so only the half spectrum kx <= Kx/2 is kept, and the x
//! transforms carry two real lines (z and z + 1) as the real and imaginary
//! part of one complex line. A workgroup owns one line: it copies the line to
//! workgroup memory, runs the Stockham autosort stages there (radix 2, grid
//! sizes are powers of two up to 256) and writes the line back, so no two
//! workgroups touch the same element and the transforms run in place.
//!
//! Everything that depends on the box is computed from the three box lengths
//! in the uniform at every dispatch, so a barostat kernel can rewrite them on
//! the device. The tables are constant for a mesh size.
//!
//! Only core WGSL is used. Every store writes a whole element (see
//! `rank_idx` in `pbc_tiles.wgsl`), and each entry point binds at most four
//! storage buffers.

struct PmeConfig {
  grid: vec4<u32>, // Kx, Ky, Kz, atoms
  box_: vec4<f32>, // Lx, Ly, Lz in A (byte offset 16), Ewald alpha in 1/A
  bits: vec4<u32>, // log2 of Kx, Ky, Kz; spare
}

@group(0) @binding(0) var<uniform> pc: PmeConfig;
// sys[2i] = (charge, sigma, epsilon, mass); sys[2i+1] = position, centred on
// the box and not wrapped.
@group(0) @binding(1) var<storage, read> sys: array<vec4<f32>>;
// Charge mesh, index (x * Ky + y) * Kz + z, fixed point with scale 2^26.
@group(0) @binding(2) var<storage, read_write> charge_mesh: array<atomic<u32>>;
// The same buffer without atomics, for the transform that reads and clears
// words that it alone owns.
@group(0) @binding(3) var<storage, read_write> charge_words: array<u32>;
// Half spectrum, index (kx * Ky + y) * Kz + z with kx <= Kx/2. The y and z
// indices are positions or frequencies depending on the stage.
@group(0) @binding(4) var<storage, read_write> spectrum: array<vec2<f32>>;
// Mesh potential in kcal/mol/e, indexed like the charge mesh.
@group(0) @binding(5) var<storage, read_write> potential: array<f32>;
// twiddle[j] = exp(-2 pi i j / 256).
@group(0) @binding(6) var<storage, read> twiddle: array<vec2<f32>>;
// 1/|b(m)|^2 of the order-4 spline for x, then y, then z.
@group(0) @binding(7) var<storage, read> moduli: array<f32>;
// Pair gradient per atom; the reciprocal gradient is added to it.
@group(0) @binding(8) var<storage, read_write> pair_grad: array<vec4<f32>>;
// Fixed-point accumulators of `pbc_tiles.wgsl` (energy variant only).
@group(0) @binding(9) var<storage, read_write> acc: array<atomic<u32>>;

override COMPUTE_ENERGY: bool = false;
override INVERSE: bool = false;

const COULOMB: f32 = 332.063713299;
const PI: f32 = 3.14159265358979;
const WIDE: u32 = 32768u;
const MESH_SCALE: f32 = 67108864.0;      // 2^26
const MESH_UNIT: f32 = 1.4901161193847656e-8; // 2^-26
// Half of `line`: the longest mesh line.
const LINE: u32 = 256u;

fn n_atoms() -> u32 { return pc.grid.w; }
fn wide_index(group: vec3<u32>) -> u32 { return group.x + group.y * WIDE; }

// ---------------------------------------------------------------------------
// 64-bit fixed point in two u32 words, exactly as in `pbc_tiles.wgsl`:
// value = (hi:lo) * 2^-32, truncated toward zero.

fn fixed_from_f32(v: f32) -> vec2<u32> {
  let a = min(abs(v), 2147483520.0);
  let whole = floor(a);
  let lo = u32((a - whole) * 4294967296.0);
  let hi = u32(whole);
  if (v >= 0.0) { return vec2<u32>(lo, hi); }
  return vec2<u32>(~lo + 1u, ~hi + select(0u, 1u, lo == 0u));
}

fn acc_add(word: u32, v: f32) {
  if (v == 0.0) { return; }
  let x = fixed_from_f32(v);
  let old = atomicAdd(&acc[word], x.x);
  let carry = select(0u, 1u, old + x.x < old);
  let hi = x.y + carry;
  if (hi != 0u) { atomicAdd(&acc[word + 1u], hi); }
}

// Energy slots of the tile engine: 1 electrostatics, 3 virial.
fn energy_word(slot: u32) -> u32 { return 6u * n_atoms() + 2u * slot; }

// ---------------------------------------------------------------------------
// Cardinal B-splines of order 4. `theta[k]` holds, for the three axes, the
// weight of mesh point `cell + k`; `dtheta[k]` its derivative with respect to
// the scaled coordinate. The recursion is the one of the CPU engine.

struct Spline {
  cell: vec3<u32>,
  theta: array<vec3<f32>, 4>,
  dtheta: array<vec3<f32>, 4>,
}

fn spline(p: vec3<f32>) -> Spline {
  let points = vec3<f32>(pc.grid.xyz);
  let scaled = p / pc.box_.xyz;
  var u = (scaled - floor(scaled)) * points;
  // A fraction that rounds to one is mesh point zero.
  u = select(vec3<f32>(0.0), u, u < points);
  let cell = floor(u);
  let w = u - cell;
  // Order 3, whose differences are the derivative of order 4.
  let a0 = 0.5 * (1.0 - w) * (1.0 - w);
  let a1 = 0.5 * ((w + 1.0) * (1.0 - w) + (2.0 - w) * w);
  let a2 = 0.5 * w * w;
  let third = 1.0 / 3.0;
  var s: Spline;
  s.cell = min(vec3<u32>(cell), pc.grid.xyz - vec3<u32>(1u));
  s.dtheta[0] = -a0;
  s.dtheta[1] = a0 - a1;
  s.dtheta[2] = a1 - a2;
  s.dtheta[3] = a2;
  s.theta[0] = third * (1.0 - w) * a0;
  s.theta[1] = third * ((w + 2.0) * a0 + (2.0 - w) * a1);
  s.theta[2] = third * ((w + 1.0) * a1 + (3.0 - w) * a2);
  s.theta[3] = third * w * a2;
  return s;
}

fn usable(q: f32, p: vec3<f32>) -> bool {
  return q != 0.0 && all(abs(p) < vec3<f32>(1e20));
}

@compute @workgroup_size(64)
fn spread(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let i = wide_index(group) * 64u + lid;
  if (i >= n_atoms()) { return; }
  let q = sys[2u * i].x;
  let p = sys[2u * i + 1u].xyz;
  if (!usable(q, p)) { return; }
  let s = spline(p);
  let grid = pc.grid.xyz;
  let mask = grid - vec3<u32>(1u);
  for (var jx = 0u; jx < 4u; jx++) {
    let x = (s.cell.x + jx) & mask.x;
    let wx = q * MESH_SCALE * s.theta[jx].x;
    for (var jy = 0u; jy < 4u; jy++) {
      let y = (s.cell.y + jy) & mask.y;
      let wxy = wx * s.theta[jy].y;
      let row = (x * grid.y + y) * grid.z;
      for (var jz = 0u; jz < 4u; jz++) {
        let z = (s.cell.z + jz) & mask.z;
        let value = i32(round(wxy * s.theta[jz].z));
        atomicAdd(&charge_mesh[row + z], bitcast<u32>(value));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// One-dimensional transforms in workgroup memory. `line` is two buffers of
// 256 elements; a stage reads one and writes the other. Lane `lid` computes
// the output elements lid, lid + 64, ..., each from two inputs, so a stage
// has no ordering among the lanes and the barrier after it is the only
// synchronisation.

var<workgroup> line: array<vec2<f32>, 512>;

// Stage `ls` of the decimation-in-frequency Stockham transform of
// `1 << bits` points: the sequences have length n = N >> ls and stride
// s = 1 << ls, and output q + s (2 p + r) is
// (x[q + s p] + (-1)^r x[q + s (p + n/2)]) exp(-+2 pi i r p / n).
fn fft_stage(lid: u32, bits: u32, ls: u32, src: u32, dst: u32, direction: f32) {
  let n = 1u << bits;
  let half = n >> 1u;
  let low = (1u << ls) - 1u;
  for (var j = lid; j < n; j += 64u) {
    let t = j >> ls;
    let turn = (t >> 1u) << ls;
    let ia = src + (j & low) + turn;
    let a = line[ia];
    let b = line[ia + half];
    var value = a + b;
    if ((t & 1u) != 0u) {
      let w = twiddle[turn << (8u - bits)];
      let d = a - b;
      value = vec2<f32>(d.x * w.x - direction * d.y * w.y, direction * d.x * w.y + d.y * w.x);
    }
    line[dst + j] = value;
  }
}

// Stages `first..last` on the buffer at `start`; returns where the result is.
// `direction` is 1 for the forward transform and -1 for the (unnormalised)
// inverse. Must be called by every lane: the barriers need uniform control
// flow, which the loop bounds (uniform values) keep.
fn fft_stages(lid: u32, bits: u32, first: u32, last: u32, start: u32, direction: f32) -> u32 {
  var src = start;
  for (var ls = first; ls < last; ls++) {
    fft_stage(lid, bits, ls, src, LINE - src, direction);
    workgroupBarrier();
    src = LINE - src;
  }
  return src;
}

// x forward. The workgroup (z / 2, y) transforms mesh lines z and z + 1 as
// one complex line c = a + i b; with C its transform, the transforms of the
// two real lines are A[k] = (C[k] + conj(C[-k])) / 2 and
// B[k] = (C[k] - conj(C[-k])) / 2i. Only k <= Kx/2 is stored.
@compute @workgroup_size(64)
fn fft_x_forward(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let points = pc.grid.x;
  let plane = pc.grid.y * pc.grid.z;
  let column = group.y * pc.grid.z + 2u * group.x;
  for (var x = lid; x < points; x += 64u) {
    let at = x * plane + column;
    let re = bitcast<i32>(charge_words[at]);
    let im = bitcast<i32>(charge_words[at + 1u]);
    charge_words[at] = 0u;
    charge_words[at + 1u] = 0u;
    line[x] = MESH_UNIT * vec2<f32>(f32(re), f32(im));
  }
  workgroupBarrier();
  let at = fft_stages(lid, pc.bits.x, 0u, pc.bits.x, 0u, 1.0);
  for (var k = lid; k <= points / 2u; k += 64u) {
    let direct = line[at + k];
    let other = line[at + ((points - k) & (points - 1u))];
    let mirror = vec2<f32>(other.x, -other.y);
    let odd = direct - mirror;
    let out = k * plane + column;
    spectrum[out] = 0.5 * (direct + mirror);
    spectrum[out + 1u] = 0.5 * vec2<f32>(odd.y, -odd.x);
  }
}

// y, forward or inverse: the workgroup (z, kx) owns one line.
@compute @workgroup_size(64)
fn fft_y(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let points = pc.grid.y;
  let stride = pc.grid.z;
  let base = group.y * points * stride + group.x;
  for (var y = lid; y < points; y += 64u) { line[y] = spectrum[base + y * stride]; }
  workgroupBarrier();
  let at = fft_stages(lid, pc.bits.y, 0u, pc.bits.y, 0u, select(1.0, -1.0, INVERSE));
  for (var y = lid; y < points; y += 64u) { spectrum[base + y * stride] = line[at + y]; }
}

// Frequency of mesh index `k`: indices in the upper half are negative.
fn frequency(k: u32, points: u32) -> f32 {
  return select(f32(k), f32(k) - f32(points), 2u * k > points);
}

var<workgroup> sums: array<vec2<f32>, 64>;

// z forward, influence function, z inverse: the workgroup (ky, kx) owns one
// line of the half spectrum. The influence function
//   G(m) = C exp(-pi^2 m^2 / alpha^2) / (pi V m^2 |b(m)|^2),  m = k / L,
// multiplies the outputs of the last forward stage, which needs no twiddle
// (out[j] = x[j] + x[j + N/2], out[j + N/2] = x[j] - x[j + N/2]). With
// COMPUTE_ENERGY the line's energy 1/2 sum G |S|^2 and virial
// sum e(m) (1 - 2 pi^2 m^2 / alpha^2) are returned by lane 0; a line with
// 0 < kx < Kx/2 stands for its mirror image too.
fn convolve_z(group: vec3<u32>, lid: u32) -> vec2<f32> {
  let points = pc.grid.z;
  let bits = pc.bits.z;
  let half = points >> 1u;
  let base = (group.y * pc.grid.y + group.x) * points;
  for (var z = lid; z < points; z += 64u) { line[z] = spectrum[base + z]; }
  workgroupBarrier();
  let at = fft_stages(lid, bits, 0u, bits - 1u, 0u, 1.0);
  let scaled = LINE - at;

  let lengths = pc.box_.xyz;
  let factor = PI * PI / (pc.box_.w * pc.box_.w);
  let mx = f32(group.y) / lengths.x;
  let my = frequency(group.x, pc.grid.y) / lengths.y;
  let m2_xy = mx * mx + my * my;
  let prefactor = COULOMB / (PI * lengths.x * lengths.y * lengths.z)
    * moduli[group.y] * moduli[pc.grid.x + group.x];
  let weight = select(1.0, 0.5, group.y == 0u || 2u * group.y == pc.grid.x);
  var total = vec2<f32>(0.0);
  for (var k = lid; k < points; k += 64u) {
    let low = k & (half - 1u);
    let a = line[at + low];
    let b = line[at + low + half];
    let value = select(a + b, a - b, k >= half);
    let mz = frequency(k, points) / lengths.z;
    let m2 = m2_xy + mz * mz;
    var g = 0.0;
    // The m = 0 term is the uniform background, not part of this sum.
    if (m2 > 0.0) {
      g = prefactor * moduli[pc.grid.x + pc.grid.y + k] * exp(-factor * m2) / m2;
    }
    if (COMPUTE_ENERGY) {
      let e = weight * g * dot(value, value);
      total += vec2<f32>(e, e * (1.0 - 2.0 * factor * m2));
    }
    line[scaled + k] = g * value;
  }
  workgroupBarrier();
  let back = fft_stages(lid, bits, 0u, bits, scaled, -1.0);
  for (var z = lid; z < points; z += 64u) { spectrum[base + z] = line[back + z]; }

  if (COMPUTE_ENERGY) {
    sums[lid] = total;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
      if (lid < stride) { sums[lid] += sums[lid + stride]; }
      workgroupBarrier();
    }
    return sums[0];
  }
  return vec2<f32>(0.0);
}

// The force-only pipeline: `acc` is not among its bindings.
@compute @workgroup_size(64)
fn fft_z(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  convolve_z(group, lid);
}

// The same with COMPUTE_ENERGY set: one fixed-point add per line into the
// electrostatic energy and the virial of the tile engine. The words are
// cleared by the kernel that publishes them, not here.
@compute @workgroup_size(64)
fn fft_z_energy(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let total = convolve_z(group, lid);
  if (lid == 0u) {
    acc_add(energy_word(1u), total.x);
    acc_add(energy_word(3u), total.y);
  }
}

// x inverse. The workgroup (z / 2, y) rebuilds the complex line
// A + i B of the two real lines z and z + 1 from their half spectra (the
// mirror half is conj(A) + i conj(B)), transforms it and stores the real
// part as line z and the imaginary part as line z + 1 of the potential.
@compute @workgroup_size(64)
fn fft_x_inverse(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let points = pc.grid.x;
  let half = points >> 1u;
  let plane = pc.grid.y * pc.grid.z;
  let column = group.y * pc.grid.z + 2u * group.x;
  for (var k = lid; k <= half; k += 64u) {
    let a = spectrum[k * plane + column];
    let b = spectrum[k * plane + column + 1u];
    line[k] = vec2<f32>(a.x - b.y, a.y + b.x);
    if (k != 0u && k != half) { line[points - k] = vec2<f32>(a.x + b.y, b.x - a.y); }
  }
  workgroupBarrier();
  let at = fft_stages(lid, pc.bits.x, 0u, pc.bits.x, 0u, -1.0);
  for (var x = lid; x < points; x += 64u) {
    let value = line[at + x];
    potential[x * plane + column] = value.x;
    potential[x * plane + column + 1u] = value.y;
  }
}

// ---------------------------------------------------------------------------
// Gradient dE/dr of every atom from the mesh potential and the analytic
// spline derivatives, added to the pair gradient the pair kernel stored
// earlier in the pass. One invocation owns the atom and stores the element
// whole.

@compute @workgroup_size(64)
fn gather(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let i = wide_index(group) * 64u + lid;
  if (i >= n_atoms()) { return; }
  let q = sys[2u * i].x;
  let p = sys[2u * i + 1u].xyz;
  if (!usable(q, p)) { return; }
  let s = spline(p);
  let grid = pc.grid.xyz;
  let mask = grid - vec3<u32>(1u);
  var sum = vec3<f32>(0.0);
  for (var jx = 0u; jx < 4u; jx++) {
    let x = (s.cell.x + jx) & mask.x;
    var plane = vec3<f32>(0.0);
    for (var jy = 0u; jy < 4u; jy++) {
      let y = (s.cell.y + jy) & mask.y;
      let row = (x * grid.y + y) * grid.z;
      var value = 0.0;
      var slope = 0.0;
      for (var jz = 0u; jz < 4u; jz++) {
        let phi = potential[row + ((s.cell.z + jz) & mask.z)];
        value += s.theta[jz].z * phi;
        slope += s.dtheta[jz].z * phi;
      }
      plane += vec3<f32>(s.theta[jy].y * value, s.dtheta[jy].y * value, s.theta[jy].y * slope);
    }
    sum += vec3<f32>(s.dtheta[jx].x * plane.x, s.theta[jx].x * plane.y, s.theta[jx].x * plane.z);
  }
  let g = q * vec3<f32>(grid) / pc.box_.xyz * sum;
  pair_grad[i] = vec4<f32>(pair_grad[i].xyz + g, 0.0);
}
