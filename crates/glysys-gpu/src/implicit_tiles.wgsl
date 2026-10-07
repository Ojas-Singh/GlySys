//! Tiled OBC2 implicit-solvent forces for resident dynamics.
//!
//! Replaces the per-target direct-force pass of `energy.wgsl` in the
//! LF-middle loop (Born radii and their adjoints still come from the
//! `born_radii_md` / `born_adjoint_md` passes, which fill `born`). One 32-lane
//! workgroup evaluates a 32 x 32 tile of directed pairs (i block, j block)
//! with the j atoms staged in workgroup memory; bonded terms and restraints
//! run one invocation per term. Every contribution is added to 64-bit
//! fixed-point accumulators (two u32 words, scale 2^32), so totals are
//! independent of scheduling and identical inputs give identical bits.
//!
//! The physics and summation conventions match `evaluate_md`: GB pairs and
//! their Born-radius chain rule over all atoms, Lennard-Jones and Coulomb with
//! the Amber exclusion/1-4 scales, no cutoff, and the per-atom ACE surface
//! term. Energies are produced only by the `COMPUTE_ENERGY` variants and are
//! published into the per-atom partial layout read by the existing reduction.

const COMPUTE_GRADIENTS: bool = true;
struct Config { size: vec4<u32>, energy: vec4<f32>, solvent: vec4<f32>, spare: vec4<f32> }
struct Atom { ff: vec4<f32>, more: vec4<f32>, ranges: vec4<u32> }
struct Term { ids: vec4<u32>, parameters: vec4<f32>, reference: vec4<f32> }
struct Special { other: u32, scee: f32, scnb: f32, spare: u32 }

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> atoms: array<Atom>;
@group(0) @binding(2) var<storage, read> coordinates: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> terms: array<Term>;
@group(0) @binding(5) var<storage, read> specials: array<Special>;
@group(0) @binding(6) var<storage, read_write> born: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> acc: array<atomic<u32>>;
// x = bonded and restraint term count.
@group(0) @binding(9) var<uniform> tiles: vec4<u32>;

override COMPUTE_ENERGY: bool = false;

const C: f32 = 332.063713299;
const PI: f32 = 3.141592653589793;
const U32MAX: u32 = 4294967295u;

fn n_atoms() -> u32 { return config.size.x; }

// Energy slots after the 6 n gradient words: 0 bonds, 1 angles, 2 proper,
// 3 improper, 4 LJ, 5 electrostatics, 6 GB, 7 restraints, 8 pair count.
fn energy_word(slot: u32) -> u32 { return 6u * n_atoms() + 2u * slot; }

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

fn acc_take(word: u32) -> f32 {
  let value = f32_from_fixed(atomicLoad(&acc[word]), atomicLoad(&acc[word + 1u]));
  atomicStore(&acc[word], 0u);
  atomicStore(&acc[word + 1u], 0u);
  return value;
}

fn add_gradient(atom: u32, g: vec3<f32>) {
  acc_add(6u * atom, g.x);
  acc_add(6u * atom + 2u, g.y);
  acc_add(6u * atom + 4u, g.z);
}

// Per-atom scalar sums (Born integrals, then Born adjoints) follow the
// energy slots.
fn scalar_word(atom: u32) -> u32 { return 6u * n_atoms() + 18u + 2u * atom; }

// The OBC pair integral (the `.x` lane of energy.wgsl's `radial`).
fn radial_value(r: f32, s: f32, d: f32) -> f32 {
  if (d + s <= r) { return 0.0; }
  let candidate = abs(d - s);
  let l = max(r, candidate);
  let u = d + s;
  if (l >= u) { return 0.0; }
  let a = 1.0 / l;
  let b = 1.0 / u;
  let c = d - s * s / d;
  let q = b * b - a * a;
  return 0.5 * (a - b + 0.25 * c * q + 0.5 * log(l / u) / d);
}

// d/dd of the OBC pair integral (the `.y` lane of energy.wgsl's `radial`).
fn radial_derivative(r: f32, s: f32, d: f32) -> f32 {
  if (d + s <= r) { return 0.0; }
  let candidate = abs(d - s);
  let l = max(r, candidate);
  let u = d + s;
  if (l >= u) { return 0.0; }
  var dl = select(1.0, -1.0, d < s);
  if (candidate < r) { dl = 0.0; }
  let a = 1.0 / l;
  let b = 1.0 / u;
  let c = d - s * s / d;
  let q = b * b - a * a;
  let lg = log(l / u);
  return 0.5 * (-dl * a * a + b * b + 0.25 * ((1.0 + s * s / (d * d)) * q
    + c * (-2.0 * b * b * b + 2.0 * dl * a * a * a)) + 0.5 * ((dl * a - b) / d - lg / (d * d)));
}

fn offset_radius(atom: Atom) -> f32 { return max(atom.ff.w - 0.09, 0.1); }

// (scee, scnb) for an ordered pair; (0, _) means excluded.
fn pair_scales(i: u32, j: u32, ranges: vec4<u32>) -> vec2<f32> {
  if (config.spare.y != 0.0) {
    let pair = specials[bitcast<u32>(config.spare.x) + i * n_atoms() + j];
    return vec2<f32>(pair.scee, pair.scnb);
  }
  var low = ranges.z;
  var high = ranges.w;
  for (var iteration = 0u; iteration < 32u && low < high; iteration++) {
    let middle = low + (high - low) / 2u;
    if (specials[middle].other < j) { low = middle + 1u; } else { high = middle; }
  }
  if (low < ranges.w && specials[low].other == j) {
    return vec2<f32>(specials[low].scee, specials[low].scnb);
  }
  return vec2<f32>(1.0, 1.0);
}

var<workgroup> tile_pos: array<vec4<f32>, 32>;   // position, charge
var<workgroup> tile_lj: array<vec4<f32>, 32>;    // LJ radius, epsilon, offset radius, scaled radius
var<workgroup> tile_born: array<vec4<f32>, 32>;  // Born radius, chain factor A
var<workgroup> tile_red: array<vec4<f32>, 32>;
var<workgroup> tile_red2: array<f32, 32>;

@compute @workgroup_size(32)
fn gb_tiles(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let n = n_atoms();
  let i = 32u * group.x + lane;
  let j_load = 32u * group.y + lane;
  if (j_load < n) {
    let aj = atoms[j_load];
    let bj = born[j_load];
    tile_pos[lane] = vec4<f32>(coordinates[j_load].xyz, aj.ff.x);
    let rho = offset_radius(aj);
    tile_lj[lane] = vec4<f32>(aj.ff.y, aj.ff.z, rho, rho * aj.more.x);
    tile_born[lane] = vec4<f32>(bj.x, bj.z * bj.y, 0.0, 0.0);
  } else {
    tile_pos[lane] = vec4<f32>(0.0);
    tile_lj[lane] = vec4<f32>(0.0);
    tile_born[lane] = vec4<f32>(1.0, 0.0, 0.0, 0.0);
  }
  workgroupBarrier();
  let valid = i < n;
  var ai = Atom(vec4<f32>(0.0), vec4<f32>(0.0), vec4<u32>(0u));
  var ci = vec3<f32>(0.0);
  var bi = vec4<f32>(1.0, 0.0, 0.0, 0.0);
  if (valid) {
    ai = atoms[i];
    ci = coordinates[i].xyz;
    bi = born[i];
  }
  let rho_i = offset_radius(ai);
  let scaled_i = rho_i * ai.more.x;
  let a_i = bi.z * bi.y;
  let dielectric = 1.0 / config.solvent.x - 1.0 / config.solvent.y;
  let gb_charge = -C * dielectric * ai.ff.x;
  let coulomb_charge = C * ai.ff.x / config.energy.y;
  var g = vec3<f32>(0.0);
  var e_gb = 0.0;
  var e_lj = 0.0;
  var e_coul = 0.0;
  var pairs = 0.0;
  let count = min(32u, n - min(n, 32u * group.y));
  if (valid) {
    for (var k = 0u; k < count; k++) {
      let j = 32u * group.y + k;
      let pj = tile_pos[k];
      let delta = ci - pj.xyz;
      let d2 = dot(delta, delta);
      let b = tile_born[k];
      let p = bi.x * b.x;
      let e = exp(-d2 / (4.0 * p));
      let f = max(sqrt(d2 + p * e), 1e-8);
      let coefficient = gb_charge * pj.w;
      if (COMPUTE_ENERGY) { e_gb += 0.5 * coefficient / f; }
      if (j == i) { continue; }
      g += delta * (-coefficient / (f * f * f)) * (1.0 - 0.25 * e);
      let d = sqrt(d2);
      let lj = tile_lj[k];
      if (d >= 1e-8) {
        let dri = radial_derivative(rho_i, lj.w, d);
        let drj = radial_derivative(lj.z, scaled_i, d);
        g += delta * (a_i * dri + b.y * drj) / d;
      }
      let scales = pair_scales(i, j, ai.ranges);
      if (scales.x == 0.0) { continue; }
      let dd = max(d, 1e-8);
      let radius = ai.ff.y + lj.x;
      let epsilon = sqrt(ai.ff.z * lj.y);
      let ratio = radius / dd;
      let ratio2 = ratio * ratio;
      let ratio6 = ratio2 * ratio2 * ratio2;
      let coulomb = coulomb_charge * pj.w / (scales.x * dd);
      g += delta * (12.0 * epsilon * (ratio6 - ratio6 * ratio6) / (scales.y * dd) - coulomb / dd) / dd;
      if (COMPUTE_ENERGY) {
        e_lj += 0.5 * epsilon * (ratio6 * ratio6 - 2.0 * ratio6) / scales.y;
        e_coul += 0.5 * coulomb;
        pairs += 0.5;
      }
    }
    add_gradient(i, g);
  }
  if (COMPUTE_ENERGY) {
    tile_red[lane] = vec4<f32>(e_gb, e_lj, e_coul, pairs);
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride >>= 1u) {
      if (lane < stride) { tile_red[lane] += tile_red[lane + stride]; }
      workgroupBarrier();
    }
    if (lane == 0u) {
      acc_add(energy_word(6u), tile_red[0].x);
      acc_add(energy_word(4u), tile_red[0].y);
      acc_add(energy_word(5u), tile_red[0].z);
      acc_add(energy_word(8u), tile_red[0].w);
    }
  }
}

// ---------------------------------------------------------------------------
// Bonded terms (kind 0 bond, 1 angle, 2 proper, 3 improper torsion,
// 4 positional restraint), one invocation per term with analytic gradients.

fn angle_gradient(a: vec3<f32>, c: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
  let u = a - c;
  let v = b - c;
  let ru = max(length(u), 1e-8);
  let rv = max(length(v), 1e-8);
  let cos_t = clamp(dot(u, v) / (ru * rv), -1.0, 1.0);
  let sin_t = max(length(cross(u, v)) / (ru * rv), 1e-8);
  let factor = -1.0 / sin_t;
  let gu = factor * (v / (ru * rv) - cos_t * u / (ru * ru));
  let gv = factor * (u / (ru * rv) - cos_t * v / (rv * rv));
  return mat3x3<f32>(gu, -gu - gv, gv);
}

var<workgroup> bonded_red: array<vec4<f32>, 64>;
var<workgroup> bonded_red2: array<f32, 64>;

@compute @workgroup_size(64)
fn bonded_terms(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let index = (group.x + group.y * 32768u) * 64u + lane;
  var energy = vec4<f32>(0.0);
  var restraint = 0.0;
  if (index < tiles.x) {
    let t = terms[index];
    let kind = u32(t.parameters.w);
    let p0 = coordinates[t.ids.x].xyz;
    if (kind == 0u) {
      let d = p0 - coordinates[t.ids.y].xyz;
      let r = max(length(d), 1e-8);
      let stretch = r - t.parameters.y;
      let f = 2.0 * t.parameters.x * stretch / r;
      add_gradient(t.ids.x, f * d);
      add_gradient(t.ids.y, -f * d);
      energy.x = t.parameters.x * stretch * stretch;
    } else if (kind == 1u) {
      let pc = coordinates[t.ids.y].xyz;
      let pb = coordinates[t.ids.z].xyz;
      let u = p0 - pc;
      let v = pb - pc;
      let theta = atan2(length(cross(u, v)), dot(u, v));
      let delta = theta - t.parameters.y;
      let f = 2.0 * t.parameters.x * delta;
      let gradient = angle_gradient(p0, pc, pb);
      add_gradient(t.ids.x, f * gradient[0]);
      add_gradient(t.ids.y, f * gradient[1]);
      add_gradient(t.ids.z, f * gradient[2]);
      energy.y = t.parameters.x * delta * delta;
    } else if (kind == 2u || kind == 3u) {
      let p1 = coordinates[t.ids.y].xyz;
      let p2 = coordinates[t.ids.z].xyz;
      let p3 = coordinates[t.ids.w].xyz;
      let b0 = p1 - p0;
      let b1 = p2 - p1;
      let b2 = p3 - p2;
      let n1 = cross(b0, b1);
      let n2 = cross(b1, b2);
      let norm_b1 = max(length(b1), 1e-12);
      let phi = atan2(dot(cross(n1, n2), b1 / norm_b1), dot(n1, n2));
      let arg = t.parameters.y * phi - t.parameters.z;
      let de = -t.parameters.y * t.parameters.x * sin(arg);
      let n1_2 = max(dot(n1, n1), 1e-16);
      let n2_2 = max(dot(n2, n2), 1e-16);
      let force0 = de * norm_b1 / n1_2 * n1;
      let force3 = -de * norm_b1 / n2_2 * n2;
      let b1_2 = max(dot(b1, b1), 1e-16);
      let s = (-dot(b0, b1) / b1_2) * force0 - (-dot(b2, b1) / b1_2) * force3;
      add_gradient(t.ids.x, -force0);
      add_gradient(t.ids.y, force0 - s);
      add_gradient(t.ids.z, force3 + s);
      add_gradient(t.ids.w, -force3);
      let value = t.parameters.x * (1.0 + cos(arg));
      if (kind == 2u) { energy.z = value; } else { energy.w = value; }
    } else if (kind == 4u) {
      let d = p0 - t.reference.xyz;
      add_gradient(t.ids.x, 2.0 * t.parameters.x * d);
      restraint = t.parameters.x * dot(d, d);
    }
  }
  if (COMPUTE_ENERGY) {
    bonded_red[lane] = energy;
    bonded_red2[lane] = restraint;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
      if (lane < stride) {
        bonded_red[lane] += bonded_red[lane + stride];
        bonded_red2[lane] += bonded_red2[lane + stride];
      }
      workgroupBarrier();
    }
    if (lane == 0u) {
      acc_add(energy_word(0u), bonded_red[0].x);
      acc_add(energy_word(1u), bonded_red[0].y);
      acc_add(energy_word(2u), bonded_red[0].z);
      acc_add(energy_word(3u), bonded_red[0].w);
      acc_add(energy_word(7u), bonded_red2[0]);
    }
  }
}

// Publish gradients to the integrator layout (`output[3n + i]`) and clear the
// accumulators. The energy variant also writes the per-atom partials read by
// the batch reduction: totals on atom 0 plus each atom's ACE surface term.
@compute @workgroup_size(64)
fn finalize_forces(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  let n = n_atoms();
  if (i >= n) { return; }
  let g = vec3<f32>(acc_take(6u * i), acc_take(6u * i + 2u), acc_take(6u * i + 4u));
  output[3u * n + i] = vec4<f32>(g, 0.0);
  if (COMPUTE_ENERGY) {
    let atom = atoms[i];
    let ratio = atom.ff.w / born[i].x;
    let ratio2 = ratio * ratio;
    let surface = 4.0 * PI * config.solvent.w * (atom.ff.w + config.solvent.z)
      * (atom.ff.w + config.solvent.z) * ratio2 * ratio2 * ratio2;
    var bonded = vec4<f32>(0.0);
    var nonbonded = vec4<f32>(0.0, 0.0, 0.0, surface);
    var extra = vec4<f32>(0.0);
    if (i == 0u) {
      bonded = vec4<f32>(acc_take(energy_word(0u)), acc_take(energy_word(1u)),
        acc_take(energy_word(2u)), acc_take(energy_word(3u)));
      nonbonded = vec4<f32>(acc_take(energy_word(4u)), acc_take(energy_word(5u)),
        acc_take(energy_word(6u)), surface);
      extra = vec4<f32>(acc_take(energy_word(7u)), acc_take(energy_word(8u)), 0.0, 0.0);
    }
    output[i] = bonded;
    output[n + i] = nonbonded;
    output[2u * n + i] = extra;
  }
}

// ---------------------------------------------------------------------------
// Born radii and adjoints on the same 32 x 32 tiles. Each tile adds its
// partial per-atom sums in fixed point; the finishing kernels apply the OBC
// rescaling (`born_radii_md`) and the ACE term (`born_adjoint_md`).

var<workgroup> born_pos: array<vec4<f32>, 32>;   // position, scaled radius or charge
var<workgroup> born_aux: array<f32, 32>;          // Born radius (adjoint pass)

@compute @workgroup_size(32)
fn born_tiles(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let n = n_atoms();
  let i = 32u * group.x + lane;
  let j_load = 32u * group.y + lane;
  if (j_load < n) {
    let aj = atoms[j_load];
    born_pos[lane] = vec4<f32>(coordinates[j_load].xyz, offset_radius(aj) * aj.more.x);
  } else {
    born_pos[lane] = vec4<f32>(0.0);
  }
  workgroupBarrier();
  if (i >= n) { return; }
  let ci = coordinates[i].xyz;
  let r = offset_radius(atoms[i]);
  let count = min(32u, n - min(n, 32u * group.y));
  var sum = 0.0;
  for (var k = 0u; k < count; k++) {
    if (32u * group.y + k == i) { continue; }
    let pj = born_pos[k];
    let d = max(distance(ci, pj.xyz), 1e-8);
    sum += radial_value(r, pj.w, d);
  }
  acc_add(scalar_word(i), sum);
}

@compute @workgroup_size(64)
fn born_finish(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= n_atoms()) { return; }
  let atom = atoms[i];
  let r = offset_radius(atom);
  let psi = r * acc_take(scalar_word(i));
  let t = tanh(psi - 0.8 * psi * psi + 4.85 * psi * psi * psi);
  let denominator = 1.0 / r - t / atom.ff.w;
  let b = 1.0 / max(denominator, 1e-6);
  var derivative = 0.0;
  if (denominator >= 1e-6) {
    derivative = b * b * (1.0 - t * t) * (1.0 - 1.6 * psi + 14.55 * psi * psi) * r / atom.ff.w;
  }
  born[i] = vec4<f32>(b, derivative, 0.0, 0.0);
}

@compute @workgroup_size(32)
fn adjoint_tiles(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let n = n_atoms();
  let i = 32u * group.x + lane;
  let j_load = 32u * group.y + lane;
  if (j_load < n) {
    born_pos[lane] = vec4<f32>(coordinates[j_load].xyz, atoms[j_load].ff.x);
    born_aux[lane] = born[j_load].x;
  } else {
    born_pos[lane] = vec4<f32>(0.0);
    born_aux[lane] = 1.0;
  }
  workgroupBarrier();
  if (i >= n) { return; }
  let ci = coordinates[i].xyz;
  let bi = born[i].x;
  let dielectric = 1.0 / config.solvent.x - 1.0 / config.solvent.y;
  let charge = -C * dielectric * atoms[i].ff.x;
  let count = min(32u, n - min(n, 32u * group.y));
  var db = 0.0;
  for (var k = 0u; k < count; k++) {
    let pj = born_pos[k];
    let delta = ci - pj.xyz;
    let r2 = dot(delta, delta);
    let bj = born_aux[k];
    let p = bi * bj;
    let e = exp(-r2 / (4.0 * p));
    let f = sqrt(r2 + p * e);
    if (f >= 1e-8) {
      let coefficient = charge * pj.w;
      db += (-0.5 * coefficient / (f * f * f)) * e * (1.0 + r2 / (4.0 * p)) * bj;
    }
  }
  acc_add(scalar_word(i), db);
}

@compute @workgroup_size(64)
fn adjoint_finish(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= n_atoms()) { return; }
  let radius = atoms[i].ff.w;
  let b = born[i];
  let ratio = radius / b.x;
  let ratio2 = ratio * ratio;
  let surface = 4.0 * PI * config.solvent.w * (radius + config.solvent.z)
    * (radius + config.solvent.z) * ratio2 * ratio2 * ratio2;
  born[i] = vec4<f32>(b.x, b.y, acc_take(scalar_word(i)) - 6.0 * surface / b.x, 0.0);
}
