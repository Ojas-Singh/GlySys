//! Tiled periodic nonbonded engine (OpenMM-style atom blocks).
//!
//! Atoms are ordered along a Hilbert curve of a fine periodic grid and grouped
//! into blocks of 32. Every Verlet rebuild recomputes the block bounding boxes
//! and, for each block I, the atoms of every other block that lie within
//! cutoff + skin of block I's box. Those atoms are packed into 32-wide tiles,
//! preceded by block I's own diagonal tile, with one exclusion mask per block
//! atom. Each pair is evaluated from both of its blocks, so a block writes its
//! own pair gradients and the pair kernel needs no atomics or j-scatter.
//! Excluded and 1-4 pairs are masked here; 1-4 pairs are evaluated by the
//! per-term bonded kernel with their Amber scales (OpenMM exceptions).
//!
//! Electrostatics are reaction field or, with `PME`, the direct-space part of
//! particle-mesh Ewald: regular pairs use `qq erfc(alpha r)/r` and every
//! excluded pair (1-4 pairs included) gets the correction `-qq erf(alpha r)/r`
//! from the bonded kernel, which removes what the mesh adds for that pair.
//! The mesh itself is `pbc_pme.wgsl`.
//!
//! Bonded forces and all energies are accumulated as 64-bit two's-complement
//! fixed point (scale 2^32) held in pairs of u32 words. Integer addition is
//! associative, so totals do not depend on workgroup scheduling; pair
//! gradients follow a fixed tile and lane order, and the atom order and tile
//! contents are deterministic functions of the positions at the rebuild.
//! Identical inputs therefore give identical bits on a given device.
//!
//! Only core WGSL is used (no subgroups, no 64-bit or float atomics), and each
//! entry point binds at most eight storage buffers, within default WebGPU
//! limits. Each pipeline uses an automatic layout of the bindings it touches.

struct TileConfig {
  sizes: vec4<u32>,     // atoms, blocks, tiles per block, far-exclusion offset in `atoms`
  box_: vec4<f32>,      // Lx, Ly, Lz, cutoff
  rf: vec4<f32>,        // krf, crf, list radius (cutoff + skin), half skin
  grid: vec4<u32>,      // sort cells per axis (2^bits), bits, spare, bucket count
  work: vec4<u32>,      // `work` offsets: order, bucket of atom, counts, starts
  work2: vec4<u32>,     // `work` offsets: scan partials, block tiles, tile atoms, tile masks
  work3: vec4<u32>,     // `work` offsets: reference positions, counters; aux status, aux rebuild
  terms: vec4<u32>,     // exceptions, bonds, angles, torsions (all terms)
  dyn_terms: vec4<u32>, // bonds, angles without constrained terms; scan groups; far exclusions
  offsets: vec4<u32>,   // `terms` offsets: bonds, angles, torsions; exceptions offset in `atoms`
  ewald: vec4<f32>,     // 2 / cutoff^2, Ewald coefficient, spare, spare
  ewald_terms: vec4<u32>, // excluded-pair corrections: all, without rigid water; `terms` offset; spare
  // Degree-15 polynomials in u = 2 r^2 / cutoff^2 - 1 (monomial coefficients,
  // fitted on the host): alpha^3 P_F and alpha P_V of the Ewald pair term.
  ewald_force: array<vec4<f32>, 4>,
  ewald_energy: array<vec4<f32>, 4>,
}

struct Special { other: u32, scee: f32, scnb: f32, spare: u32 }

@group(0) @binding(0) var<uniform> tc: TileConfig;
// sys[2i] = (charge, sigma, epsilon, mass); sys[2i+1] = position.
@group(0) @binding(1) var<storage, read> sys: array<vec4<f32>>;
// Legacy PBC metadata: numerical status, rebuild flag and rebuild counter.
@group(0) @binding(2) var<storage, read_write> aux: array<atomic<u32>>;
// atoms[i] = (sigma, sqrt(epsilon), charge, molecule bits) for i < n;
// atoms[n + i] = (specials start, end, 64-bit exclusion window) bits;
// 1-4 exceptions and far exclusion pairs follow.
@group(0) @binding(3) var<storage, read> atoms: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> specials: array<Special>;
@group(0) @binding(5) var<storage, read_write> work: array<atomic<u32>>;
// blocks[2b] = (center, needs per-pair minimum image); blocks[2b+1] = half extent.
@group(0) @binding(6) var<storage, read_write> blocks: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> bcoords: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> terms: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> args: array<u32>;
// Pair (nonbonded) gradient per atom, written once per evaluation by the
// atom's block; bonded and 1-4 terms accumulate separately in `acc`.
@group(0) @binding(12) var<storage, read_write> pair_grad: array<vec4<f32>>;
// The same `work` buffer viewed without atomics for kernels that only read
// it. Atomic loads of shared words would serialize across workgroups.
@group(0) @binding(13) var<storage, read> work_ro: array<u32>;

override COMPUTE_ENERGY: bool = false;
override INCLUDE_CONSTRAINED: bool = true;
override PME: bool = false;

const U32MAX: u32 = 4294967295u;
const COULOMB: f32 = 332.063713299;
const WIDE: u32 = 32768u;

fn n_atoms() -> u32 { return tc.sizes.x; }
fn n_blocks() -> u32 { return tc.sizes.y; }
fn wide_index(group: vec3<u32>) -> u32 { return group.x + group.y * WIDE; }

fn min_image(d: vec3<f32>) -> vec3<f32> {
  return d - tc.box_.xyz * round(d / tc.box_.xyz);
}

fn position(atom: u32) -> vec3<f32> { return sys[2u * atom + 1u].xyz; }

// Degree-15 polynomial by Estrin's scheme, operation for operation the CPU
// cluster kernel's `ewald_polynomial`.
fn estrin16(c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, c3: vec4<f32>, u: f32) -> f32 {
  let u2 = u * u;
  let u4 = u2 * u2;
  let u8 = u4 * u4;
  let q0 = (c0.x + c0.y * u) + (c0.z + c0.w * u) * u2;
  let q1 = (c1.x + c1.y * u) + (c1.z + c1.w * u) * u2;
  let q2 = (c2.x + c2.y * u) + (c2.z + c2.w * u) * u2;
  let q3 = (c3.x + c3.y * u) + (c3.z + c3.w * u) * u2;
  return (q0 + q1 * u4) + (q2 + q3 * u4) * u8;
}

// `blocks` holds two vec4 per block, one cached sorted position per slot,
// then the sorted slot of every atom, one element each (x: the slot's bits).
// The slots must not share an element: a store to one component of a vector
// may rewrite the whole vector, so two invocations writing different
// components of one vec4 race, and on Metal the loser's slot is lost. Each
// atom's element is written whole by the one invocation that owns the atom.
fn sorted_idx(slot: u32) -> u32 { return 2u * n_blocks() + slot; }
fn rank_idx(atom: u32) -> u32 { return 34u * n_blocks() + atom; }
fn rank_of(atom: u32) -> u32 { return bitcast<u32>(blocks[rank_idx(atom)].x); }

fn order_at(slot: u32) -> u32 { return atomicLoad(&work[tc.work.x + slot]); }

// ---------------------------------------------------------------------------
// 64-bit fixed point in two u32 words: value = (hi:lo) * 2^-32, truncated
// toward zero. Negation is exact, so equal and opposite pair contributions
// cancel exactly.

fn fixed_from_f32(v: f32) -> vec2<u32> {
  let a = min(abs(v), 2147483520.0);
  let whole = floor(a);
  let lo = u32((a - whole) * 4294967296.0);
  let hi = u32(whole);
  if (v >= 0.0) { return vec2<u32>(lo, hi); }
  return vec2<u32>(~lo + 1u, ~hi + select(0u, 1u, lo == 0u));
}

fn f32_from_fixed(lo: u32, hi: u32) -> f32 {
  return f32(bitcast<i32>(hi)) + f32(lo) * 2.3283064365386963e-10;
}

fn acc_add(word: u32, v: f32) {
  if (v == 0.0) { return; }
  let x = fixed_from_f32(v);
  let old = atomicAdd(&acc[word], x.x);
  let carry = select(0u, 1u, old + x.x < old);
  let hi = x.y + carry;
  if (hi != 0u) { atomicAdd(&acc[word + 1u], hi); }
}

fn acc_read(word: u32) -> f32 {
  return f32_from_fixed(atomicLoad(&acc[word]), atomicLoad(&acc[word + 1u]));
}

fn acc_clear(word: u32) {
  atomicStore(&acc[word], 0u);
  atomicStore(&acc[word + 1u], 0u);
}

const STATUS_NONFINITE: u32 = 4u;

fn add_gradient(atom: u32, g: vec3<f32>) {
  if (!all(abs(g) < vec3<f32>(1e18))) {
    atomicStore(&aux[tc.work3.z], STATUS_NONFINITE);
    return;
  }
  acc_add(6u * atom, g.x);
  acc_add(6u * atom + 2u, g.y);
  acc_add(6u * atom + 4u, g.z);
}

// Energy slots: 0 LJ, 1 electrostatics, 2 evaluated directed pairs, 3 virial,
// 4 intermolecular pair virial, 5 bonds, 6 angles, 7 proper, 8 improper.
fn energy_word(slot: u32) -> u32 { return 6u * n_atoms() + 2u * slot; }

// ---------------------------------------------------------------------------
// Rebuild control. A rebuild is requested by a coordinate upload (aux flag)
// or by any atom moving at least half the skin from its rebuild position.

@compute @workgroup_size(128)
fn check_rebuild(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= n_atoms()) { return; }
  let p = position(i);
  if (!all(abs(p) < vec3<f32>(1e20))) {
    atomicStore(&aux[tc.work3.z], STATUS_NONFINITE);
    return;
  }
  if (atomicLoad(&aux[tc.work3.w]) != 0u) { return; }
  let k = tc.work3.x + 3u * i;
  let r = bitcast<vec3<f32>>(vec3<u32>(
    atomicLoad(&work[k]), atomicLoad(&work[k + 1u]), atomicLoad(&work[k + 2u])));
  let d = min_image(p - r);
  if (dot(d, d) >= tc.rf.w * tc.rf.w) { atomicStore(&aux[tc.work3.w], 1u); }
}

fn counter_idx(slot: u32) -> u32 { return tc.work3.y + slot; }
// counters: 1 tile overflow.

fn write_args(stage: u32, groups: u32) {
  let x = min(groups, WIDE);
  args[3u * stage] = x;
  args[3u * stage + 1u] = select(1u, (groups + WIDE - 1u) / WIDE, groups > WIDE);
  args[3u * stage + 2u] = 1u;
  if (groups == 0u) { args[3u * stage + 1u] = 1u; }
}

// Indirect stages: 0 block bounds, 1 tile build, 2 far exclusions, 3 finish.
// The spatial resort is scheduled by the host on a fixed step cadence and
// always forces a rebuild, so only the rebuild depends on the device flag.
@compute @workgroup_size(1)
fn prepare_args() {
  let rebuild = atomicLoad(&aux[tc.work3.w]) != 0u;
  write_args(0u, select(0u, n_blocks(), rebuild));
  write_args(1u, select(0u, n_blocks(), rebuild));
  write_args(2u, select(0u, tc.dyn_terms.w, rebuild));
  write_args(3u, select(0u, 1u, rebuild));
}

// ---------------------------------------------------------------------------
// Deterministic spatial sort: counting sort into Hilbert-ordered cells, then
// ascending atom index within each cell. The grid has 2^b cells on every
// axis, so consecutive cells along the curve are always face neighbours.

// Skilling's transposed Hilbert index of a cell in a 2^bits cube.
fn hilbert_index(cell: vec3<u32>, bits: u32) -> u32 {
  var x = array<u32, 3>(cell.x, cell.y, cell.z);
  let top = 1u << (bits - 1u);
  var q = top;
  while (q > 1u) {
    let p = q - 1u;
    for (var i = 0u; i < 3u; i++) {
      if ((x[i] & q) != 0u) {
        x[0] ^= p;
      } else {
        let t = (x[0] ^ x[i]) & p;
        x[0] ^= t;
        x[i] ^= t;
      }
    }
    q >>= 1u;
  }
  x[1] ^= x[0];
  x[2] ^= x[1];
  var t = 0u;
  q = top;
  while (q > 1u) {
    if ((x[2] & q) != 0u) { t ^= q - 1u; }
    q >>= 1u;
  }
  x[0] ^= t;
  x[1] ^= t;
  x[2] ^= t;
  var code = 0u;
  for (var k = bits; k > 0u; k--) {
    let bit = k - 1u;
    code = (code << 3u) | (((x[0] >> bit) & 1u) << 2u) | (((x[1] >> bit) & 1u) << 1u)
      | ((x[2] >> bit) & 1u);
  }
  return code;
}

fn bucket_of(p: vec3<f32>) -> u32 {
  let l = tc.box_.xyz;
  let w = p - l * floor(p / l);
  let g = f32(tc.grid.x);
  let c = vec3<u32>(clamp(floor(w * g / l), vec3<f32>(0.0), vec3<f32>(g - 1.0)));
  return hilbert_index(c, tc.grid.y);
}

// A resort changes block membership, so it always forces a list rebuild.
@compute @workgroup_size(128)
fn bucket_clear(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x == 0u) { atomicStore(&aux[tc.work3.w], 1u); }
  if (id.x < tc.grid.w) { atomicStore(&work[tc.work.z + id.x], 0u); }
}

@compute @workgroup_size(128)
fn bucket_count(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= n_atoms()) { return; }
  let b = bucket_of(position(i));
  atomicStore(&work[tc.work.y + i], b);
  atomicAdd(&work[tc.work.z + b], 1u);
}

var<workgroup> scan_tmp: array<u32, 256>;

fn scan256(lid: u32, value: u32) -> u32 {
  scan_tmp[lid] = value;
  workgroupBarrier();
  for (var offset = 1u; offset < 256u; offset <<= 1u) {
    var add = 0u;
    if (lid >= offset) { add = scan_tmp[lid - offset]; }
    workgroupBarrier();
    scan_tmp[lid] += add;
    workgroupBarrier();
  }
  return scan_tmp[lid];
}

// Exclusive prefix of the bucket counts, 1024 buckets per workgroup.
@compute @workgroup_size(256)
fn scan_local(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let buckets = tc.grid.w;
  let base = (group.x * 256u + lid) * 4u;
  var local: array<u32, 4>;
  var sum = 0u;
  for (var k = 0u; k < 4u; k++) {
    var count = 0u;
    if (base + k < buckets) { count = atomicLoad(&work[tc.work.z + base + k]); }
    local[k] = sum;
    sum += count;
  }
  let inclusive = scan256(lid, sum);
  let exclusive = inclusive - sum;
  for (var k = 0u; k < 4u; k++) {
    if (base + k < buckets) { atomicStore(&work[tc.work.w + base + k], exclusive + local[k]); }
  }
  if (lid == 255u) { atomicStore(&work[tc.work2.x + group.x], inclusive); }
}

@compute @workgroup_size(256)
fn scan_partials(@builtin(local_invocation_index) lid: u32) {
  let groups = tc.dyn_terms.z;
  var local: array<u32, 8>;
  var sum = 0u;
  for (var k = 0u; k < 8u; k++) {
    let index = lid * 8u + k;
    var value = 0u;
    if (index < groups) { value = atomicLoad(&work[tc.work2.x + index]); }
    local[k] = sum;
    sum += value;
  }
  let inclusive = scan256(lid, sum);
  let exclusive = inclusive - sum;
  for (var k = 0u; k < 8u; k++) {
    let index = lid * 8u + k;
    if (index < groups) { atomicStore(&work[tc.work2.x + index], exclusive + local[k]); }
  }
  if (lid == 255u) { atomicStore(&work[tc.work.w + tc.grid.w], inclusive); }
}

@compute @workgroup_size(256)
fn scan_add(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let offset = atomicLoad(&work[tc.work2.x + group.x]);
  let base = (group.x * 256u + lid) * 4u;
  for (var k = 0u; k < 4u; k++) {
    if (base + k < tc.grid.w) { atomicAdd(&work[tc.work.w + base + k], offset); }
  }
}

@compute @workgroup_size(128)
fn bucket_scatter(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= n_atoms()) { return; }
  let b = atomicLoad(&work[tc.work.y + i]);
  let slot = atomicLoad(&work[tc.work.w + b]) + atomicSub(&work[tc.work.z + b], 1u) - 1u;
  atomicStore(&work[tc.work.x + slot], i);
}

// Scatter order within a cell depends on atomic timing; restoring ascending
// atom index makes the final order a pure function of the positions.
@compute @workgroup_size(128)
fn bucket_order(@builtin(global_invocation_id) id: vec3<u32>) {
  let b = id.x;
  if (b >= tc.grid.w) { return; }
  let start = atomicLoad(&work[tc.work.w + b]);
  let end = atomicLoad(&work[tc.work.w + b + 1u]);
  // Physical cells hold a few atoms. Only pathological inputs (many
  // coincident atoms) exceed this bound; they keep scatter order rather than
  // risk a quadratic sort tripping a device watchdog.
  if (end - start > 256u) { return; }
  for (var k = start + 1u; k < end; k++) {
    let value = atomicLoad(&work[tc.work.x + k]);
    var m = k;
    loop {
      if (m <= start) { break; }
      let previous = atomicLoad(&work[tc.work.x + m - 1u]);
      if (previous <= value) { break; }
      atomicStore(&work[tc.work.x + m], previous);
      m -= 1u;
    }
    atomicStore(&work[tc.work.x + m], value);
  }
}

// ---------------------------------------------------------------------------
// Block bounding boxes, recorded in the minimum-image frame of the block's
// first atom so a block straddling a periodic face stays compact.

var<workgroup> bb_low: array<vec3<f32>, 32>;
var<workgroup> bb_high: array<vec3<f32>, 32>;

@compute @workgroup_size(32)
fn block_bounds(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let block = wide_index(group);
  if (block >= n_blocks()) { return; }
  let anchor = position(order_at(32u * block));
  let atom = order_at(32u * block + lid);
  var low = vec3<f32>(1e30);
  var high = vec3<f32>(-1e30);
  var cached = vec4<f32>(0.0, 0.0, 0.0, bitcast<f32>(atom));
  if (atom != U32MAX) {
    let p = position(atom);
    let relative = min_image(p - anchor);
    low = relative;
    high = relative;
    let bits = bitcast<vec3<u32>>(p);
    let k = tc.work3.x + 3u * atom;
    atomicStore(&work[k], bits.x);
    atomicStore(&work[k + 1u], bits.y);
    atomicStore(&work[k + 2u], bits.z);
    cached = vec4<f32>(p, bitcast<f32>(atom));
    blocks[rank_idx(atom)] = vec4<f32>(bitcast<f32>(32u * block + lid), 0.0, 0.0, 0.0);
  }
  // Sorted copy of the rebuild positions (w = atom index bits, all-ones for
  // padding) so tile building reads candidate blocks contiguously and
  // without atomic loads of words that many workgroups share.
  blocks[sorted_idx(32u * block + lid)] = cached;
  bb_low[lid] = low;
  bb_high[lid] = high;
  workgroupBarrier();
  for (var stride = 16u; stride > 0u; stride >>= 1u) {
    if (lid < stride) {
      bb_low[lid] = min(bb_low[lid], bb_low[lid + stride]);
      bb_high[lid] = max(bb_high[lid], bb_high[lid + stride]);
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    let center = anchor + 0.5 * (bb_low[0] + bb_high[0]);
    let half_extent = 0.5 * (bb_high[0] - bb_low[0]);
    let reach = half_extent + vec3<f32>(tc.box_.w + tc.rf.w);
    let per_pair = any(reach >= 0.5 * tc.box_.xyz);
    blocks[2u * block] = vec4<f32>(center, select(0.0, 1.0, per_pair));
    blocks[2u * block + 1u] = vec4<f32>(half_extent, 0.0);
  }
}

// ---------------------------------------------------------------------------
// Tile lists. Block I owns a diagonal tile plus tiles of the atoms of every
// other block that lie within the list radius of I's box, so each pair is
// listed from both sides and pair forces never scatter to another block. A
// workgroup counts the atoms, reserves a contiguous tile range, then repeats
// the traversal to fill it in block and lane order. Ranks come from popcounts
// of OR-combined ballots, which are independent of scheduling.

var<workgroup> tb_ballot: array<atomic<u32>, 2>;
var<workgroup> tb_sum: u32;
var<workgroup> tb_candidates: array<u32, 64>;
var<workgroup> tb_count: u32;
var<workgroup> tb_start: u32;
var<workgroup> tb_ok: u32;

// Workgroup-wide (rank, total) of `flag` over 64 lanes in lane order.
fn ballot64(lid: u32, flag: bool) -> vec2<u32> {
  if (lid == 0u) {
    atomicStore(&tb_ballot[0], 0u);
    atomicStore(&tb_ballot[1], 0u);
  }
  workgroupBarrier();
  if (flag) { atomicOr(&tb_ballot[lid >> 5u], 1u << (lid & 31u)); }
  workgroupBarrier();
  let low = atomicLoad(&tb_ballot[0]);
  let high = atomicLoad(&tb_ballot[1]);
  if (lid == 0u) { tb_sum = countOneBits(low) + countOneBits(high); }
  let total = workgroupUniformLoad(&tb_sum);
  let below = (1u << (lid & 31u)) - 1u;
  var rank = countOneBits(low & below);
  if (lid >= 32u) { rank = countOneBits(low) + countOneBits(high & below); }
  return vec2<u32>(rank, total);
}

fn box_gap2(a: u32, b: u32) -> f32 {
  let gap = max(
    abs(min_image(blocks[2u * a].xyz - blocks[2u * b].xyz))
      - blocks[2u * a + 1u].xyz - blocks[2u * b + 1u].xyz,
    vec3<f32>(0.0));
  return dot(gap, gap);
}

fn atom_gap2(block: u32, p: vec3<f32>) -> f32 {
  let gap = max(abs(min_image(p - blocks[2u * block].xyz)) - blocks[2u * block + 1u].xyz,
    vec3<f32>(0.0));
  return dot(gap, gap);
}

fn tile_atom_idx(tile: u32, slot: u32) -> u32 { return tc.work2.z + 32u * tile + slot; }
fn tile_mask_idx(tile: u32, slot: u32) -> u32 { return tc.work2.w + 32u * tile + slot; }

var<workgroup> tb_block_atom: array<u32, 32>;
var<workgroup> tb_block_window: array<vec2<u32>, 32>;
var<workgroup> tb_sorted_atom: array<u32, 32>;
var<workgroup> tb_sorted_row: array<u32, 32>;
var<workgroup> tb_masks: array<atomic<u32>, 64>;
var<workgroup> tb_offsets: array<u32, 64>;

// Exclusions within 32 atom indices are recorded per atom as a 64-bit window
// (bit k: atom index - 32 + k); `far_exclusions` handles the rest.
fn window_excludes(window: vec2<u32>, atom: u32, other: u32) -> bool {
  let offset = i32(other) - i32(atom) + 32;
  if (offset < 0 || offset >= 64) { return false; }
  let word = select(window.x, window.y, offset >= 32);
  return ((word >> (u32(offset) & 31u)) & 1u) != 0u;
}

// Clear the mask bits of block rows whose window excludes `partner` (in
// `slot` of `tile`). Only rows with atom indices in [partner - 31,
// partner + 32] can qualify; they are found by binary search over the block's
// atoms sorted by index, entirely in workgroup memory.
fn clear_window_exclusions(partner: u32, tile: u32, slot: u32) {
  let lowest = select(0u, partner - 31u, partner >= 31u);
  var low = 0u;
  var high = 32u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (tb_sorted_atom[middle] < lowest) { low = middle + 1u; } else { high = middle; }
  }
  for (var j = low; j < 32u; j++) {
    let atom = tb_sorted_atom[j];
    if (atom == U32MAX || atom > partner + 32u) { break; }
    let row = tb_sorted_row[j];
    if (atom != partner && window_excludes(tb_block_window[row], atom, partner)) {
      atomicAnd(&work[tile_mask_idx(tile, row)], ~(1u << slot));
    }
  }
}

// Block I owns tiles [I * stride, (I + 1) * stride): its diagonal tile then
// the packed atoms of nearby blocks in block and lane order. Candidate blocks
// are tested 32 atoms at a time against cached sorted positions; inclusion
// masks combine with workgroup atomicOr and a serial prefix places them, and
// exclusions clear mask bits with atomicAnd. Every combining operation is
// commutative, so the lists are a pure function of the rebuild positions.
@compute @workgroup_size(64)
fn tile_build(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let block = wide_index(group);
  if (block >= n_blocks()) { return; }
  let blocks_total = n_blocks();
  let radius2 = tc.rf.z * tc.rf.z;
  let stride = tc.sizes.z;
  let start = block * stride;
  let slots = 32u * (stride - 1u);
  let lane = lid & 31u;
  let half = lid >> 5u;
  if (lid < 32u) {
    let atom = bitcast<u32>(blocks[sorted_idx(32u * block + lid)].w);
    tb_block_atom[lid] = atom;
    var window = vec2<u32>(0u);
    if (atom != U32MAX) { window = bitcast<vec2<u32>>(atoms[n_atoms() + atom].zw); }
    tb_block_window[lid] = window;
  }
  // Every reserved mask starts full; invalid slots and exclusions are
  // removed below.
  for (var word = lid; word < 32u * stride; word += 64u) {
    atomicStore(&work[tile_mask_idx(start, word)], 0xffffffffu);
  }
  workgroupBarrier();
  if (lid < 32u) {
    // Rank by atom index (padding last); ties cannot occur among real atoms.
    let atom = tb_block_atom[lid];
    var rank = 0u;
    for (var j = 0u; j < 32u; j++) {
      let other = tb_block_atom[j];
      if (other < atom || (other == atom && j < lid)) { rank += 1u; }
    }
    tb_sorted_atom[rank] = atom;
    tb_sorted_row[rank] = lid;
  }
  storageBarrier();
  workgroupBarrier();
  var total = 0u;
  for (var first = 0u; first < blocks_total; first += 64u) {
    let other = first + lid;
    let near = other < blocks_total && other != block && box_gap2(block, other) < radius2;
    let candidate = ballot64(lid, near);
    if (near) { tb_candidates[candidate.x] = other; }
    atomicStore(&tb_masks[lid], 0u);
    workgroupBarrier();
    for (var q = half; q < candidate.y; q += 2u) {
      let cached = blocks[sorted_idx(32u * tb_candidates[q] + lane)];
      if (bitcast<u32>(cached.w) != U32MAX && atom_gap2(block, cached.xyz) < radius2) {
        atomicOr(&tb_masks[q], 1u << lane);
      }
    }
    workgroupBarrier();
    if (lid == 0u) {
      var running = 0u;
      for (var q = 0u; q < candidate.y; q++) {
        tb_offsets[q] = running;
        running += countOneBits(atomicLoad(&tb_masks[q]));
      }
      tb_sum = running;
    }
    let added = workgroupUniformLoad(&tb_sum);
    for (var q = half; q < candidate.y; q += 2u) {
      let mask = atomicLoad(&tb_masks[q]);
      if (((mask >> lane) & 1u) == 0u) { continue; }
      let index = total + tb_offsets[q] + countOneBits(mask & ((1u << lane) - 1u));
      if (index >= slots) { continue; }
      let partner = bitcast<u32>(blocks[sorted_idx(32u * tb_candidates[q] + lane)].w);
      let tile = start + 1u + index / 32u;
      let slot = index % 32u;
      atomicStore(&work[tile_atom_idx(tile, slot)], partner);
      clear_window_exclusions(partner, tile, slot);
    }
    total += added;
    workgroupBarrier();
  }
  var tiles = 1u + (total + 31u) / 32u;
  if (total > slots) {
    tiles = 0u;
    if (lid == 0u) { atomicStore(&work[counter_idx(1u)], 1u); }
  }
  if (lid == 0u) {
    atomicStore(&work[tc.work2.y + 2u * block], start);
    atomicStore(&work[tc.work2.y + 2u * block + 1u], tiles);
  }
  if (tiles == 0u) { return; }
  for (var slot = total + lid; slot < 32u * (tiles - 1u); slot += 64u) {
    atomicStore(&work[tile_atom_idx(start + 1u + slot / 32u, slot % 32u)], U32MAX);
  }
  // Diagonal tile: the block's own atoms, without i == j or exclusions.
  if (lid < 32u) { atomicStore(&work[tile_atom_idx(start, lid)], tb_block_atom[lid]); }
  let row = lane;
  let row_atom = tb_block_atom[row];
  for (var t = half; t < tiles; t += 2u) {
    var keep = 0u;
    if (row_atom != U32MAX) {
      if (t == 0u) {
        let window = tb_block_window[row];
        for (var s = 0u; s < 32u; s++) {
          let other = tb_block_atom[s];
          if (s != row && other != U32MAX && !window_excludes(window, row_atom, other)) {
            keep |= 1u << s;
          }
        }
      } else {
        let filled = min(total - 32u * (t - 1u), 32u);
        keep = select((1u << filled) - 1u, 0xffffffffu, filled == 32u);
      }
    }
    atomicAnd(&work[tile_mask_idx(start + t, row)], keep);
  }
}

// Exclusions between atoms more than 32 indices apart (disulfides, glycan
// and other inter-residue links). One workgroup per directed pair (a, c)
// scans a's block tiles for c and clears that slot in a's row.
@compute @workgroup_size(64)
fn far_exclusions(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let index = wide_index(group);
  if (index >= tc.dyn_terms.w) { return; }
  let pair = bitcast<vec2<u32>>(atoms[tc.sizes.w + index].xy);
  let rank = rank_of(pair.x);
  let block = rank / 32u;
  let row = rank % 32u;
  let start = atomicLoad(&work[tc.work2.y + 2u * block]);
  let tiles = atomicLoad(&work[tc.work2.y + 2u * block + 1u]);
  for (var slot = lid; slot < 32u * tiles; slot += 64u) {
    if (atomicLoad(&work[tile_atom_idx(start, slot)]) == pair.y) {
      atomicAnd(&work[tile_mask_idx(start + slot / 32u, row)], ~(1u << (slot % 32u)));
    }
  }
}

// An overflowed list stays dirty, so a retry cannot reuse partial tiles; the
// sticky capacity status reaches the host at its next poll.
@compute @workgroup_size(1)
fn finish_rebuild() {
  let overflow = atomicLoad(&work[counter_idx(1u)]) != 0u;
  atomicStore(&work[counter_idx(1u)], 0u);
  if (overflow) {
    if (atomicLoad(&aux[tc.work3.z]) == 0u) { atomicStore(&aux[tc.work3.z], 3u); }
    return;
  }
  atomicStore(&aux[tc.work3.w], 0u);
  atomicAdd(&aux[tc.work3.w + 1u], 1u);
}

// ---------------------------------------------------------------------------
// Pair forces. A 128-lane workgroup owns one block: each 32-lane quarter
// holds the block atoms in registers and walks every fourth tile, reading the
// tile atoms as workgroup broadcasts. Each pair is computed from both sides,
// so the block writes its own pair gradients without atomics; the quarters
// combine in a fixed order. Energies and virials are halved accordingly.

var<workgroup> tile_pos: array<vec4<f32>, 128>;
var<workgroup> tile_par: array<vec4<f32>, 128>;
var<workgroup> tile_range: vec2<u32>;
var<workgroup> pair_red: array<vec4<f32>, 128>;
var<workgroup> pair_red2: array<vec4<f32>, 128>;

@compute @workgroup_size(128)
fn nb_tiles(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let block = wide_index(group);
  if (block >= n_blocks()) { return; }
  if (lid == 0u) {
    tile_range = vec2<u32>(work_ro[tc.work2.y + 2u * block], work_ro[tc.work2.y + 2u * block + 1u]);
  }
  let range = workgroupUniformLoad(&tile_range);
  let quarter = lid >> 5u;
  let lane = lid & 31u;
  let base = 32u * quarter;
  let center = blocks[2u * block].xyz;
  let per_pair = blocks[2u * block].w != 0.0;
  let atom = work_ro[tc.work.x + 32u * block + lane];
  let valid = atom != U32MAX;
  var pi = vec3<f32>(0.0);
  var ai = vec4<f32>(0.0);
  if (valid) {
    pi = center + min_image(position(atom) - center);
    ai = atoms[atom];
  }
  let cutoff2 = tc.box_.w * tc.box_.w;
  let krf2 = 2.0 * tc.rf.x;
  let qi = COULOMB * ai.z;
  let ew_scale = tc.ewald.x;
  let ef0 = tc.ewald_force[0];
  let ef1 = tc.ewald_force[1];
  let ef2 = tc.ewald_force[2];
  let ef3 = tc.ewald_force[3];
  var gradient = vec3<f32>(0.0);
  var e_lj = 0.0;
  var e_rf = 0.0;
  var virial = 0.0;
  var pair_virial = 0.0;
  var evaluated = 0.0;
  let rounds = (range.y + 3u) / 4u;
  for (var round = 0u; round < rounds; round++) {
    let t = 4u * round + quarter;
    var mask = 0u;
    var jp = vec4<f32>(0.0);
    var jq = vec4<f32>(0.0);
    if (t < range.y) {
      let tile = range.x + t;
      let partner = work_ro[tile_atom_idx(tile, lane)];
      mask = work_ro[tile_mask_idx(tile, lane)];
      if (partner != U32MAX) {
        jq = atoms[partner];
        jp = vec4<f32>(center + min_image(position(partner) - center), jq.z);
      }
    }
    tile_pos[lid] = jp;
    tile_par[lid] = jq;
    workgroupBarrier();
    if (mask != 0u) {
      for (var k = 0u; k < 32u; k++) {
        let pj = tile_pos[base + k];
        var d = pi - pj.xyz;
        if (per_pair) { d = min_image(d); }
        let r2 = max(dot(d, d), 1e-12);
        if (((mask >> k) & 1u) != 0u && r2 < cutoff2) {
          let qj = tile_par[base + k];
          let inv_r = inverseSqrt(r2);
          let inv_r2 = inv_r * inv_r;
          let sigma = ai.x + qj.x;
          let eps = ai.y * qj.y;
          let s2 = sigma * sigma * inv_r2;
          let s6 = s2 * s2 * s2;
          let qq = qi * pj.w;
          var coulomb = krf2;
          let u = min(r2 * ew_scale - 1.0, 1.0);
          if (PME) { coulomb = estrin16(ef0, ef1, ef2, ef3, u); }
          let fmag = 12.0 * eps * (s6 - s6 * s6) * inv_r2 + qq * (coulomb - inv_r * inv_r2);
          gradient += fmag * d;
          if (COMPUTE_ENERGY) {
            e_lj += eps * (s6 * s6 - 2.0 * s6);
            if (PME) {
              e_rf += qq * (inv_r - estrin16(tc.ewald_energy[0], tc.ewald_energy[1],
                                             tc.ewald_energy[2], tc.ewald_energy[3], u));
            } else {
              e_rf += qq * (inv_r + tc.rf.x * r2 - tc.rf.y);
            }
            let w = -fmag * r2;
            virial += w;
            if (bitcast<u32>(ai.w) != bitcast<u32>(qj.w)) { pair_virial += w; }
            evaluated += 1.0;
          }
        }
      }
    }
    workgroupBarrier();
  }
  // One whole-element store per lane (see `rank_idx` on component stores).
  var reduced = vec4<f32>(gradient, 0.0);
  if (COMPUTE_ENERGY) { reduced.w = evaluated; }
  pair_red[lid] = reduced;
  pair_red2[lid] = vec4<f32>(e_lj, e_rf, virial, pair_virial);
  workgroupBarrier();
  if (quarter == 0u) {
    let total = ((pair_red[lane] + pair_red[lane + 32u]) + pair_red[lane + 64u]) + pair_red[lane + 96u];
    if (valid) {
      if (!all(abs(total.xyz) < vec3<f32>(1e18))) { atomicStore(&aux[tc.work3.z], STATUS_NONFINITE); }
      pair_grad[atom] = vec4<f32>(total.xyz, 0.0);
    }
    if (COMPUTE_ENERGY) {
      pair_red[lane] = vec4<f32>(0.0, 0.0, 0.0, total.w);
      pair_red2[lane] = ((pair_red2[lane] + pair_red2[lane + 32u]) + pair_red2[lane + 64u]) + pair_red2[lane + 96u];
    }
  }
  if (COMPUTE_ENERGY) {
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride >>= 1u) {
      if (lid < stride) {
        pair_red[lid] += pair_red[lid + stride];
        pair_red2[lid] += pair_red2[lid + stride];
      }
      workgroupBarrier();
    }
    if (lid == 0u) {
      // Every pair was evaluated from both sides.
      acc_add(energy_word(0u), 0.5 * pair_red2[0].x);
      acc_add(energy_word(1u), 0.5 * pair_red2[0].y);
      acc_add(energy_word(2u), pair_red[0].w);
      acc_add(energy_word(3u), 0.5 * pair_red2[0].z);
      acc_add(energy_word(4u), 0.5 * pair_red2[0].w);
    }
  }
}

// ---------------------------------------------------------------------------
// Bonded terms and 1-4 exceptions, one invocation per term. Bonds and angles
// fully determined by rigid constraints are ordered last so dynamics can skip
// them (their energy and constrained force components are identically zero).

fn angle_gradient(a: vec3<f32>, c: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
  let u = a - c;
  let v = b - c;
  let ru = max(length(u), 1e-8);
  let rv = max(length(v), 1e-8);
  let cos_t = clamp(dot(u, v) / (ru * rv), -1.0, 1.0);
  let sin_t = max(length(cross(u, v)) / (ru * rv), 1e-8);
  let f = -1.0 / sin_t;
  let gu = f * (v / (ru * rv) - cos_t * u / (ru * ru));
  let gv = f * (u / (ru * rv) - cos_t * v / (rv * rv));
  return mat3x3<f32>(gu, -gu - gv, gv);
}

var<workgroup> bonded_red: array<vec4<f32>, 64>;
var<workgroup> bonded_red2: array<vec4<f32>, 64>;

@compute @workgroup_size(64)
fn bonded_terms(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let index = wide_index(group) * 64u + lid;
  let exceptions = tc.terms.x;
  let bonds = select(tc.dyn_terms.x, tc.terms.y, INCLUDE_CONSTRAINED);
  let angles = select(tc.dyn_terms.y, tc.terms.z, INCLUDE_CONSTRAINED);
  let torsions = tc.terms.w;
  var corrections = 0u;
  if (PME) { corrections = select(tc.ewald_terms.y, tc.ewald_terms.x, INCLUDE_CONSTRAINED); }
  var energy = vec4<f32>(0.0);  // bonds, angles, proper, improper
  var energy2 = vec4<f32>(0.0); // LJ, electrostatics, evaluated, virial
  if (index < exceptions) {
    let e0 = atoms[tc.offsets.w + 2u * index];
    let e1 = atoms[tc.offsets.w + 2u * index + 1u];
    let a = bitcast<u32>(e0.x);
    let b = bitcast<u32>(e0.y);
    let d = min_image(position(a) - position(b));
    let r2 = max(dot(d, d), 1e-12);
    if (r2 <= tc.box_.w * tc.box_.w) {
      let inv_r2 = 1.0 / r2;
      let inv_r = inverseSqrt(r2);
      let s2 = e1.x * e1.x * inv_r2;
      let s6 = s2 * s2 * s2;
      let fmag = 12.0 * e0.w * (s6 - s6 * s6) * inv_r2 - e0.z * inv_r * inv_r2;
      add_gradient(a, fmag * d);
      add_gradient(b, -fmag * d);
      energy2 = vec4<f32>(e0.w * (s6 * s6 - 2.0 * s6), e0.z * inv_r, 2.0, -fmag * r2);
    }
  } else if (index < exceptions + bonds) {
    let t = terms[tc.offsets.x + index - exceptions];
    let a = u32(t.x);
    let b = u32(t.y);
    let d = bcoords[a].xyz - bcoords[b].xyz;
    let r = max(length(d), 1e-8);
    let f = 2.0 * t.z * (r - t.w) / r;
    add_gradient(a, f * d);
    add_gradient(b, -f * d);
    energy.x = t.z * (r - t.w) * (r - t.w);
    energy2.w = -f * dot(d, d);
  } else if (index < exceptions + bonds + angles) {
    let k = index - exceptions - bonds;
    let head = terms[tc.offsets.y + 2u * k];
    let theta0 = terms[tc.offsets.y + 2u * k + 1u].x;
    let a = u32(head.x);
    let c = u32(head.y);
    let b = u32(head.z);
    let pa = bcoords[a].xyz;
    let pc = bcoords[c].xyz;
    let pb = bcoords[b].xyz;
    let u = pa - pc;
    let v = pb - pc;
    let theta = atan2(length(cross(u, v)), dot(u, v));
    let delta = theta - theta0;
    let f = 2.0 * head.w * delta;
    let g = angle_gradient(pa, pc, pb);
    add_gradient(a, f * g[0]);
    add_gradient(c, f * g[1]);
    add_gradient(b, f * g[2]);
    energy.y = head.w * delta * delta;
  } else if (index < exceptions + bonds + angles + torsions) {
    let k = index - exceptions - bonds - angles;
    let ids = terms[tc.offsets.z + 2u * k];
    let p = terms[tc.offsets.z + 2u * k + 1u];
    let a0 = u32(ids.x);
    let a1 = u32(ids.y);
    let a2 = u32(ids.z);
    let a3 = u32(ids.w);
    let p0 = bcoords[a0].xyz;
    let p1 = bcoords[a1].xyz;
    let p2 = bcoords[a2].xyz;
    let p3 = bcoords[a3].xyz;
    let b0 = p1 - p0;
    let b1 = p2 - p1;
    let b2 = p3 - p2;
    let n1 = cross(b0, b1);
    let n2 = cross(b1, b2);
    let norm_b1 = max(length(b1), 1e-12);
    let phi = atan2(dot(cross(n1, n2), b1 / norm_b1), dot(n1, n2));
    let arg = p.y * phi - p.z;
    let de = -p.y * p.x * sin(arg);
    let n1_2 = max(dot(n1, n1), 1e-16);
    let n2_2 = max(dot(n2, n2), 1e-16);
    let force0 = de * norm_b1 / n1_2 * n1;
    let force3 = -de * norm_b1 / n2_2 * n2;
    let b1_2 = max(dot(b1, b1), 1e-16);
    let s = (-dot(b0, b1) / b1_2) * force0 - (-dot(b2, b1) / b1_2) * force3;
    add_gradient(a0, -force0);
    add_gradient(a1, force0 - s);
    add_gradient(a2, force3 + s);
    add_gradient(a3, -force3);
    let e = p.x * (1.0 + cos(arg));
    if (p.w != 0.0) { energy.w = e; } else { energy.z = e; }
  } else if (index < exceptions + bonds + angles + torsions + corrections) {
    // Ewald correction of an excluded pair: -qq erf(alpha r)/r. Both atoms
    // are in one molecule, so the molecule frame gives their separation.
    let t = terms[tc.ewald_terms.z + index - exceptions - bonds - angles - torsions];
    let a = u32(t.x);
    let b = u32(t.y);
    let d = bcoords[a].xyz - bcoords[b].xyz;
    let r2 = dot(d, d);
    let u = min(r2 * tc.ewald.x - 1.0, 1.0);
    let g = t.z * estrin16(tc.ewald_force[0], tc.ewald_force[1], tc.ewald_force[2], tc.ewald_force[3], u);
    add_gradient(a, g * d);
    add_gradient(b, -g * d);
    energy2 = vec4<f32>(0.0, -t.z * estrin16(tc.ewald_energy[0], tc.ewald_energy[1],
                                              tc.ewald_energy[2], tc.ewald_energy[3], u), 0.0, -g * r2);
  }
  if (COMPUTE_ENERGY) {
    bonded_red[lid] = energy;
    bonded_red2[lid] = energy2;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
      if (lid < stride) {
        bonded_red[lid] += bonded_red[lid + stride];
        bonded_red2[lid] += bonded_red2[lid + stride];
      }
      workgroupBarrier();
    }
    if (lid == 0u) {
      acc_add(energy_word(5u), bonded_red[0].x);
      acc_add(energy_word(6u), bonded_red[0].y);
      acc_add(energy_word(7u), bonded_red[0].z);
      acc_add(energy_word(8u), bonded_red[0].w);
      acc_add(energy_word(0u), bonded_red2[0].x);
      acc_add(energy_word(1u), bonded_red2[0].y);
      acc_add(energy_word(2u), bonded_red2[0].z);
      acc_add(energy_word(3u), bonded_red2[0].w);
    }
  }
}

// ---------------------------------------------------------------------------
// Convert accumulated fixed-point gradients to the float layout consumed by
// the integrator and observers, and clear the accumulators for the next
// evaluation. Energy totals are published by invocation 0.

@compute @workgroup_size(128)
fn finalize_forces(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  let n = n_atoms();
  if (i >= n) { return; }
  let g = pair_grad[i].xyz
    + vec3<f32>(acc_read(6u * i), acc_read(6u * i + 2u), acc_read(6u * i + 4u));
  acc_clear(6u * i);
  acc_clear(6u * i + 2u);
  acc_clear(6u * i + 4u);
  var w = 0.0;
  if (i == 0u) {
    var e: array<f32, 9>;
    for (var slot = 0u; slot < 9u; slot++) {
      e[slot] = acc_read(energy_word(slot));
      acc_clear(energy_word(slot));
    }
    let overflow = atomicLoad(&aux[tc.work3.z]) != 0u;
    out[4u * n] = vec4<f32>(e[0], e[1], e[2], select(0.0, 1.0, overflow));
    out[4u * n + 1u] = vec4<f32>(e[4], 0.0, 0.0, 0.0);
    out[3u * n] = vec4<f32>(e[5], e[6], e[7], e[8]);
    w = e[3];
  }
  out[n + i] = vec4<f32>(g, w);
}
