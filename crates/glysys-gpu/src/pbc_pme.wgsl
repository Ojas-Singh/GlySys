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
//! part of one complex line. A workgroup owns a few neighbouring lines: it
//! copies them to workgroup memory, runs the Stockham autosort stages there
//! (radix 2, grid sizes are powers of two up to 256) and writes them back, so
//! no two workgroups touch the same element and the transforms run in place.
//!
//! What a transform costs on a device is mostly how its loads and stores of
//! the meshes are laid out, not its arithmetic: the lanes of a workgroup
//! execute one load together, and it is served fastest when their addresses
//! are neighbours. Hence the layouts (x contiguous in the real meshes, z in
//! the half spectrum) and, in every copy loop below, lanes that walk along
//! the contiguous index.
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
// Charge mesh, index (y * Kz + z) * Kx + x, fixed point with scale 2^26.
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
const MESH_SCALE: f32 = 67108864.0;           // 2^26
const MESH_UNIT: f32 = 1.4901161193847656e-8; // 2^-26
// Elements of one of the two buffers of `line`: the mesh lines a workgroup
// holds have this many points together. At least the longest line (256).
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
// Cardinal B-splines of order 4. For each axis a vector holds the weights of
// the mesh points `cell`, `cell + 1`, ... and a second one their derivatives
// with respect to the scaled coordinate. The recursion is the one of the CPU
// engine. Vectors, not arrays: an array indexed in a loop lives in memory on
// a device, and each read of it then costs as much as a read of the mesh.

struct Spline {
  cell: vec3<u32>,
  tx: vec4<f32>,
  ty: vec4<f32>,
  tz: vec4<f32>,
  dx: vec4<f32>,
  dy: vec4<f32>,
  dz: vec4<f32>,
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
  let t0 = third * (1.0 - w) * a0;
  let t1 = third * ((w + 2.0) * a0 + (2.0 - w) * a1);
  let t2 = third * ((w + 1.0) * a1 + (3.0 - w) * a2);
  let t3 = third * w * a2;
  let d1 = a0 - a1;
  let d2 = a1 - a2;
  return Spline(
    min(vec3<u32>(cell), pc.grid.xyz - vec3<u32>(1u)),
    vec4<f32>(t0.x, t1.x, t2.x, t3.x),
    vec4<f32>(t0.y, t1.y, t2.y, t3.y),
    vec4<f32>(t0.z, t1.z, t2.z, t3.z),
    vec4<f32>(-a0.x, d1.x, d2.x, a2.x),
    vec4<f32>(-a0.y, d1.y, d2.y, a2.y),
    vec4<f32>(-a0.z, d1.z, d2.z, a2.z));
}

fn usable(q: f32, p: vec3<f32>) -> bool {
  return q != 0.0 && all(abs(p) < vec3<f32>(1e20));
}

// Spread and gather give an atom four lanes, one per mesh point along x. A
// lane then has 16 mesh words to touch instead of 64, one after the other,
// and the four lanes of an atom touch four neighbouring words together.

@compute @workgroup_size(64)
fn spread(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let i = wide_index(group) * 16u + (lid >> 2u);
  if (i >= n_atoms()) { return; }
  let q = sys[2u * i].x;
  let p = sys[2u * i + 1u].xyz;
  if (!usable(q, p)) { return; }
  let s = spline(p);
  let grid = pc.grid.xyz;
  let mask = grid - vec3<u32>(1u);
  let jx = lid & 3u;
  let x = (s.cell.x + jx) & mask.x;
  let wx = q * MESH_SCALE * s.tx[jx];
  for (var jy = 0u; jy < 4u; jy++) {
    let y = (s.cell.y + jy) & mask.y;
    let wxy = wx * s.ty[jy];
    for (var jz = 0u; jz < 4u; jz++) {
      let z = (s.cell.z + jz) & mask.z;
      let value = i32(round(wxy * s.tz[jz]));
      atomicAdd(&charge_mesh[(y * grid.z + z) * grid.x + x], bitcast<u32>(value));
    }
  }
}

// ---------------------------------------------------------------------------
// One-dimensional transforms in workgroup memory. `line` is two buffers of
// LINE elements; a stage reads one and writes the other. A workgroup holds
// as many mesh lines as fit, LINE / K of them side by side (line l at
// elements l K ... l K + K - 1). Lane `lid` computes the output elements
// lid, lid + 64, ..., each from two inputs, so a stage has no ordering among
// the lanes and the barrier after it is the only synchronisation.

var<workgroup> line: array<vec2<f32>, 2u * LINE>;
// The twiddles of the workgroup's transform, exp(-2 pi i j / N) for j < N/2.
// A stage reads a twiddle for every other output; taken from the storage
// table each of those reads would wait on the device's memory, here it is as
// fast as the line itself.
var<workgroup> turns: array<vec2<f32>, 128>;

// Copy the twiddles of a `1 << bits`-point transform. Every lane calls it
// before the barrier that follows the copy of the lines.
fn load_turns(lid: u32, bits: u32) {
  for (var j = lid; j < (1u << bits) / 2u; j += 64u) { turns[j] = twiddle[j << (8u - bits)]; }
}

// Lines of `points` elements held by one workgroup, of `available` in the
// direction the workgroups are counted along. A power of two.
fn lines_per_group(points: u32, available: u32) -> u32 {
  return min(LINE / points, available);
}

// Stage `ls` of the decimation-in-frequency Stockham transform of
// `1 << bits` points, on `count` elements (whole lines): the sequences have
// length n = N >> ls and stride s = 1 << ls, and output q + s (2 p + r) is
// (x[q + s p] + (-1)^r x[q + s (p + n/2)]) exp(-+2 pi i r p / n).
fn fft_stage(lid: u32, bits: u32, count: u32, ls: u32, src: u32, dst: u32, direction: f32) {
  let n = 1u << bits;
  let half = n >> 1u;
  let low = (1u << ls) - 1u;
  for (var e = lid; e < count; e += 64u) {
    let j = e & (n - 1u);
    let t = j >> ls;
    let turn = (t >> 1u) << ls;
    let ia = src + (e - j) + (j & low) + turn;
    let a = line[ia];
    let b = line[ia + half];
    var value = a + b;
    if ((t & 1u) != 0u) {
      let w = turns[turn];
      let d = a - b;
      value = vec2<f32>(d.x * w.x - direction * d.y * w.y, direction * d.x * w.y + d.y * w.x);
    }
    line[dst + e] = value;
  }
}

// Stages `first..last` on the buffer at `start`; returns where the result is.
// `direction` is 1 for the forward transform and -1 for the (unnormalised)
// inverse. Must be called by every lane: the barriers need uniform control
// flow, which the loop bounds (uniform values) keep.
fn fft_stages(lid: u32, bits: u32, count: u32, first: u32, last: u32, start: u32,
              direction: f32) -> u32 {
  var src = start;
  for (var ls = first; ls < last; ls++) {
    fft_stage(lid, bits, count, ls, src, LINE - src, direction);
    workgroupBarrier();
    src = LINE - src;
  }
  return src;
}

// x forward. Mesh lines z and z + 1 are transformed as one complex line
// c = a + i b; with C its transform, the transforms of the two real lines
// are A[k] = (C[k] + conj(C[-k])) / 2 and B[k] = (C[k] - conj(C[-k])) / 2i.
// Only k <= Kx/2 is stored. The workgroup (z group, y) holds the complex
// lines of consecutive z pairs: a contiguous block of the charge mesh.
@compute @workgroup_size(64)
fn fft_x_forward(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let points = pc.grid.x;
  let bits = pc.bits.x;
  let lines = lines_per_group(points, pc.grid.z / 2u);
  let count = lines * points;
  let z0 = 2u * lines * group.x;
  let row0 = group.y * pc.grid.z + z0;
  for (var e = lid; e < count; e += 64u) {
    let at = (row0 + 2u * (e >> bits)) * points + (e & (points - 1u));
    let re = bitcast<i32>(charge_words[at]);
    let im = bitcast<i32>(charge_words[at + points]);
    charge_words[at] = 0u;
    charge_words[at + points] = 0u;
    line[e] = MESH_UNIT * vec2<f32>(f32(re), f32(im));
  }
  load_turns(lid, bits);
  workgroupBarrier();
  let done = fft_stages(lid, bits, count, 0u, bits, 0u, 1.0);
  // One real line per lane and step: lanes take neighbouring z, which are
  // neighbours in the half spectrum.
  let reals = 2u * lines;
  let real_bits = countTrailingZeros(reals);
  let outputs = (points / 2u + 1u) * reals;
  for (var i = lid; i < outputs; i += 64u) {
    let zr = i & (reals - 1u);
    let k = i >> real_bits;
    let first = done + (zr >> 1u) * points;
    let direct = line[first + k];
    let other = line[first + ((points - k) & (points - 1u))];
    let mirror = vec2<f32>(other.x, -other.y);
    let odd = direct - mirror;
    let value = select(direct + mirror, vec2<f32>(odd.y, -odd.x), (zr & 1u) != 0u);
    spectrum[(k * pc.grid.y + group.y) * pc.grid.z + z0 + zr] = 0.5 * value;
  }
}

// y, forward or inverse: the workgroup (z group, kx) holds the lines of
// consecutive z, and its lanes take neighbouring z.
@compute @workgroup_size(64)
fn fft_y(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let points = pc.grid.y;
  let bits = pc.bits.y;
  let stride = pc.grid.z;
  let lines = lines_per_group(points, stride);
  let line_bits = countTrailingZeros(lines);
  let count = lines * points;
  let base = group.y * points * stride + lines * group.x;
  for (var i = lid; i < count; i += 64u) {
    let l = i & (lines - 1u);
    let y = i >> line_bits;
    line[l * points + y] = spectrum[base + y * stride + l];
  }
  load_turns(lid, bits);
  workgroupBarrier();
  let done = fft_stages(lid, bits, count, 0u, bits, 0u, select(1.0, -1.0, INVERSE));
  for (var i = lid; i < count; i += 64u) {
    let l = i & (lines - 1u);
    let y = i >> line_bits;
    spectrum[base + y * stride + l] = line[done + l * points + y];
  }
}

// Frequency of mesh index `k`: indices in the upper half are negative.
fn frequency(k: u32, points: u32) -> f32 {
  return select(f32(k), f32(k) - f32(points), 2u * k > points);
}

var<workgroup> sums: array<vec2<f32>, 64>;

// z forward, influence function, z inverse: the workgroup (ky group, kx)
// holds the lines of consecutive ky, a contiguous block of the half
// spectrum. The influence function
//   G(m) = C exp(-pi^2 m^2 / alpha^2) / (pi V m^2 |b(m)|^2),  m = k / L,
// multiplies the outputs of the last forward stage, which needs no twiddle
// (out[j] = x[j] + x[j + N/2], out[j + N/2] = x[j] - x[j + N/2]). With
// COMPUTE_ENERGY the energy 1/2 sum G |S|^2 and the virial
// sum e(m) (1 - 2 pi^2 m^2 / alpha^2) of these lines are returned by lane 0;
// a line with 0 < kx < Kx/2 stands for its mirror image too.
fn convolve_z(group: vec3<u32>, lid: u32) -> vec2<f32> {
  let points = pc.grid.z;
  let bits = pc.bits.z;
  let half = points >> 1u;
  let lines = lines_per_group(points, pc.grid.y);
  let count = lines * points;
  let first_ky = lines * group.x;
  let base = (group.y * pc.grid.y + first_ky) * points;
  for (var e = lid; e < count; e += 64u) { line[e] = spectrum[base + e]; }
  load_turns(lid, bits);
  workgroupBarrier();
  let done = fft_stages(lid, bits, count, 0u, bits - 1u, 0u, 1.0);
  let scaled = LINE - done;

  let lengths = pc.box_.xyz;
  let factor = PI * PI / (pc.box_.w * pc.box_.w);
  let mx = f32(group.y) / lengths.x;
  let prefactor = COULOMB / (PI * lengths.x * lengths.y * lengths.z) * moduli[group.y];
  let weight = select(1.0, 0.5, group.y == 0u || 2u * group.y == pc.grid.x);
  var total = vec2<f32>(0.0);
  for (var e = lid; e < count; e += 64u) {
    let k = e & (points - 1u);
    let ky = first_ky + (e >> bits);
    let low = done + (e - k) + (k & (half - 1u));
    let a = line[low];
    let b = line[low + half];
    let value = select(a + b, a - b, k >= half);
    let my = frequency(ky, pc.grid.y) / lengths.y;
    let mz = frequency(k, points) / lengths.z;
    let m2 = mx * mx + my * my + mz * mz;
    var g = 0.0;
    // The m = 0 term is the uniform background, not part of this sum.
    if (m2 > 0.0) {
      g = prefactor * moduli[pc.grid.x + ky] * moduli[pc.grid.x + pc.grid.y + k]
        * exp(-factor * m2) / m2;
    }
    if (COMPUTE_ENERGY) {
      let energy = weight * g * dot(value, value);
      total += vec2<f32>(energy, energy * (1.0 - 2.0 * factor * m2));
    }
    line[scaled + e] = g * value;
  }
  workgroupBarrier();
  let back = fft_stages(lid, bits, count, 0u, bits, scaled, -1.0);
  for (var e = lid; e < count; e += 64u) { spectrum[base + e] = line[back + e]; }

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

// The same with COMPUTE_ENERGY set: one fixed-point add per workgroup into
// the electrostatic energy and the virial of the tile engine. The words are
// cleared by the kernel that publishes them, not here.
@compute @workgroup_size(64)
fn fft_z_energy(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let total = convolve_z(group, lid);
  if (lid == 0u) {
    acc_add(energy_word(1u), total.x);
    acc_add(energy_word(3u), total.y);
  }
}

// x inverse. The workgroup (z group, y) rebuilds the complex lines A + i B
// of the real lines z and z + 1 from their half spectra (the mirror half is
// conj(A) + i conj(B)), transforms them and stores the real part as line z
// and the imaginary part as line z + 1 of the potential.
@compute @workgroup_size(64)
fn fft_x_inverse(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let points = pc.grid.x;
  let bits = pc.bits.x;
  let half = points >> 1u;
  let lines = lines_per_group(points, pc.grid.z / 2u);
  let line_bits = countTrailingZeros(lines);
  let count = lines * points;
  let z0 = 2u * lines * group.x;
  let row0 = group.y * pc.grid.z + z0;
  // One complex line per lane and step: lanes take neighbouring z pairs.
  let inputs = (half + 1u) * lines;
  for (var i = lid; i < inputs; i += 64u) {
    let l = i & (lines - 1u);
    let k = i >> line_bits;
    let at = (k * pc.grid.y + group.y) * pc.grid.z + z0 + 2u * l;
    let a = spectrum[at];
    let b = spectrum[at + 1u];
    line[l * points + k] = vec2<f32>(a.x - b.y, a.y + b.x);
    if (k != 0u && k != half) {
      line[l * points + points - k] = vec2<f32>(a.x + b.y, b.x - a.y);
    }
  }
  load_turns(lid, bits);
  workgroupBarrier();
  let done = fft_stages(lid, bits, count, 0u, bits, 0u, -1.0);
  for (var e = lid; e < count; e += 64u) {
    let value = line[done + e];
    let at = (row0 + 2u * (e >> bits)) * points + (e & (points - 1u));
    potential[at] = value.x;
    potential[at + points] = value.y;
  }
}

// ---------------------------------------------------------------------------
// Gradient dE/dr of every atom from the mesh potential and the analytic
// spline derivatives, added to the pair gradient the pair kernel stored
// earlier in the pass. Each of the atom's four lanes sums its plane of the
// stencil; the atom's first lane adds the four parts in lane order and
// stores the element whole.

var<workgroup> parts: array<vec4<f32>, 64>;

@compute @workgroup_size(64)
fn gather(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let i = wide_index(group) * 16u + (lid >> 2u);
  var part = vec3<f32>(0.0);
  var valid = false;
  if (i < n_atoms()) {
    let q = sys[2u * i].x;
    let p = sys[2u * i + 1u].xyz;
    if (usable(q, p)) {
      valid = true;
      let s = spline(p);
      let grid = pc.grid.xyz;
      let mask = grid - vec3<u32>(1u);
      let jx = lid & 3u;
      let x = (s.cell.x + jx) & mask.x;
      // Sums over the lane's 16 points with the weights of y and z: plain,
      // differentiated along y, differentiated along z.
      var sum = vec3<f32>(0.0);
      for (var jy = 0u; jy < 4u; jy++) {
        let y = (s.cell.y + jy) & mask.y;
        var value = 0.0;
        var slope = 0.0;
        for (var jz = 0u; jz < 4u; jz++) {
          let z = (s.cell.z + jz) & mask.z;
          let phi = potential[(y * grid.z + z) * grid.x + x];
          value += s.tz[jz] * phi;
          slope += s.dz[jz] * phi;
        }
        sum += vec3<f32>(s.ty[jy] * value, s.dy[jy] * value, s.ty[jy] * slope);
      }
      part = q * vec3<f32>(grid) / pc.box_.xyz * vec3<f32>(s.dx[jx], s.tx[jx], s.tx[jx]) * sum;
    }
  }
  parts[lid] = vec4<f32>(part, 0.0);
  workgroupBarrier();
  if (valid && jx_first(lid)) {
    let g = ((parts[lid].xyz + parts[lid + 1u].xyz) + parts[lid + 2u].xyz) + parts[lid + 3u].xyz;
    pair_grad[i] = vec4<f32>(pair_grad[i].xyz + g, 0.0);
  }
}

fn jx_first(lid: u32) -> bool { return (lid & 3u) == 0u; }
