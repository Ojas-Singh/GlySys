//! Leap-frog with Nose-Hoover temperature coupling and isotropic
//! Parrinello-Rahman pressure coupling for the resident explicit engine.
//!
//! The equations, their discretization and the order of operations are those
//! of `glysys-dynamics::coupling` and `ExplicitSimulation::leapfrog_step`
//! (the GROMACS `md` integrator); the host schedules the kernels of each step
//! from the step number, so no coupling decision is taken on the device and
//! nothing is read back between steps. One step, with the stored velocities
//! half a step behind the stored positions:
//!
//!   thermostat step     reduce (old), couple
//!   every step          kick_drift, SETTLE/SHAKE (pbc.wgsl)
//!   pressure step       reduce (new), pressure
//!   barostat step       scale, apply_box
//!   COM removal step    reduce (new), remove_com
//!   every step          forces at the new positions
//!
//! Thermostat and barostat variables live in `cs`, in f32. The box lengths
//! are also written into the uniforms of the pair, bonded and mesh kernels,
//! which this module binds as storage for that one purpose.

struct Coupling {
  sizes: vec4<u32>,       // atoms, partial groups, aux status word, aux rebuild word
  box_: vec4<f32>,        // Lx, Ly, Lz (A); cutoff
  box_velocity: vec4<f32>, // A/ps
  xi: vec4<f32>,          // Nose-Hoover friction of groups 0, 1 (1/ps); their time integrals
  factor: vec4<f32>,      // this step: 1/2 dt_T xi of groups 0, 1
  drag: vec4<f32>,        // this step: dt_p (db/dt)/b per axis; w: a pressure is stored
  kinetic_old: vec4<f32>, // kinetic energy of groups 0, 1 before the step (kcal/mol)
  kinetic_new: vec4<f32>, // the same after the step; z: constraint virial (amu A^2)
  momentum0: vec4<f32>,   // sum m v of COM group 0 (amu A/ps)
  momentum1: vec4<f32>,
  pressure: vec4<f32>,    // bar; mean kinetic energy of the step (kcal/mol); virial used (kcal/mol)
  // Constants of the run, written by the host.
  inverse_q: vec4<f32>,   // 1/Q of groups 0, 1 (1/(K ps^2)); reference temperatures (K)
  per_dof: vec4<f32>,     // 2/(dof kB) of groups 0, 1 (K mol/kcal); dt_T; dt_p (ps)
  barostat: vec4<f32>,    // 4 pi^2 beta/(3 tau_p^2) (1/(bar ps^2)); P0 (bar); dispersion term (kcal A^3/mol); dt (ps)
  com: vec4<f32>,         // 1/mass of COM groups 0, 1 (1/amu)
  list: vec4<f32>,        // box change since the pair list was built (A); allowance (A); rebuild count seen (bits)
}

@group(0) @binding(0) var<storage, read_write> cs: Coupling;
// sys[2i] = (charge, sigma, epsilon, mass); sys[2i+1] = position.
@group(0) @binding(1) var<storage, read_write> sys: array<vec4<f32>>;
// state[i] = velocity; state[n + i] = position at the start of the step.
@group(0) @binding(2) var<storage, read_write> state: array<vec4<f32>>;
// out[n + i] = gradient (w of atom 0: virial of the last energy pass);
// out[2n + k].x = constraint virial of SETTLE/SHAKE invocation k.
@group(0) @binding(3) var<storage, read> out: array<vec4<f32>>;
// Bit 0: temperature group; bit 1: center-of-mass group.
@group(0) @binding(4) var<storage, read> groups: array<u32>;
// Three vec4 per partial group: (ke0, ke1, constraint virial, 0), p0, p1.
@group(0) @binding(5) var<storage, read_write> partials: array<vec4<f32>>;
// Uniforms of the other kernels, bound as storage to rewrite the box.
@group(0) @binding(6) var<storage, read_write> pbc_config: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> tile_config: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> mesh_config: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> aux: array<atomic<u32>>;

override COUPLED: bool = false;
override THERMOSTAT: bool = false;
override BAROSTAT: bool = false;
override REDUCE_NEW: bool = false;
// Element of `mesh_config` that holds the box lengths.
override MESH_BOX_ELEMENT: u32 = 1u;

const ACCEL: f32 = 418.4;
const BAR_PER_KCAL_MOL_A3: f32 = 69476.95457055373;
const PARTIAL_ATOMS: u32 = 1024u;
const STATUS_BOX_TOO_SMALL: u32 = 9u;
const STATUS_BOX_RUNAWAY: u32 = 10u;

fn n_atoms() -> u32 { return cs.sizes.x; }

// ---------------------------------------------------------------------------
// Kinetic energy of the temperature groups, momentum of the center-of-mass
// groups and the constraint virial, in two fixed-order levels: 1024 atoms per
// workgroup, then every workgroup's partial.

var<workgroup> red_a: array<vec4<f32>, 64>;
var<workgroup> red_b: array<vec4<f32>, 64>;
var<workgroup> red_c: array<vec4<f32>, 64>;

@compute @workgroup_size(64)
fn reduce_partial(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let n = n_atoms();
  var a = vec4<f32>(0.0);
  var b = vec4<f32>(0.0);
  var c = vec4<f32>(0.0);
  for (var t = 0u; t < PARTIAL_ATOMS / 64u; t++) {
    let i = group.x * PARTIAL_ATOMS + 64u * t + lid;
    if (i < n) {
      let v = state[i].xyz;
      let mass = sys[2u * i].w;
      let kinetic = mass * dot(v, v) / (2.0 * ACCEL);
      let bits = groups[i];
      if ((bits & 1u) == 0u) { a.x += kinetic; } else { a.y += kinetic; }
      if (REDUCE_NEW) {
        a.z += out[2u * n + i].x;
        if ((bits & 2u) == 0u) { b += vec4<f32>(mass * v, 0.0); } else { c += vec4<f32>(mass * v, 0.0); }
      }
    }
  }
  red_a[lid] = a;
  red_b[lid] = b;
  red_c[lid] = c;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (lid < stride) {
      red_a[lid] += red_a[lid + stride];
      red_b[lid] += red_b[lid + stride];
      red_c[lid] += red_c[lid + stride];
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    partials[3u * group.x] = red_a[0];
    partials[3u * group.x + 1u] = red_b[0];
    partials[3u * group.x + 2u] = red_c[0];
  }
}

@compute @workgroup_size(64)
fn reduce_final(@builtin(local_invocation_index) lid: u32) {
  let count = cs.sizes.y;
  var a = vec4<f32>(0.0);
  var b = vec4<f32>(0.0);
  var c = vec4<f32>(0.0);
  for (var k = lid; k < count; k += 64u) {
    a += partials[3u * k];
    b += partials[3u * k + 1u];
    c += partials[3u * k + 2u];
  }
  red_a[lid] = a;
  red_b[lid] = b;
  red_c[lid] = c;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (lid < stride) {
      red_a[lid] += red_a[lid + stride];
      red_b[lid] += red_b[lid + stride];
      red_c[lid] += red_c[lid + stride];
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    if (REDUCE_NEW) {
      cs.kinetic_new = red_a[0];
      cs.momentum0 = red_b[0];
      cs.momentum1 = red_c[0];
    } else {
      cs.kinetic_old = red_a[0];
    }
  }
}

// ---------------------------------------------------------------------------
// Thermostat frictions from the group temperatures of the stored velocities,
// box velocities from the pressure of the previous step, and the factors the
// velocity update of this step uses.

@compute @workgroup_size(1)
fn couple() {
  var factor = vec4<f32>(0.0);
  if (THERMOSTAT) {
    let dt_t = cs.per_dof.z;
    let temperature = cs.kinetic_old.xy * cs.per_dof.xy;
    let old = cs.xi.xy;
    let next = old + dt_t * cs.inverse_q.xy * (temperature - cs.inverse_q.zw);
    cs.xi = vec4<f32>(next, cs.xi.zw + 0.5 * dt_t * (old + next));
    factor = vec4<f32>(0.5 * dt_t * next, 0.0, 0.0);
  }
  cs.factor = factor;
  var drag = vec3<f32>(0.0);
  if (BAROSTAT) {
    let dt_p = cs.per_dof.w;
    let b = cs.box_.xyz;
    var velocity = cs.box_velocity.xyz;
    // The first step of a run has no pressure yet.
    if (cs.drag.w != 0.0) {
      let inverse_mass = cs.barostat.x / max(b.x, max(b.y, b.z));
      let volume = b.x * b.y * b.z;
      let relative = (cs.pressure.x - cs.barostat.y) * dot(1.0 / (b * b), vec3<f32>(1.0)) / 3.0;
      velocity += dt_p * inverse_mass * volume * relative * b;
      cs.box_velocity = vec4<f32>(velocity, 0.0);
    }
    // The box and the coordinates are scaled by 1 + drag in f32; using that
    // same rounded number in the velocity update keeps the three consistent.
    drag = (vec3<f32>(1.0) + dt_p * velocity / b) - vec3<f32>(1.0);
    if (!all(abs(drag) <= vec3<f32>(0.1))) {
      atomicStore(&aux[cs.sizes.z], STATUS_BOX_RUNAWAY);
      drag = vec3<f32>(0.0);
    }
  }
  cs.drag = vec4<f32>(drag, 0.0);
}

// v' = [v (1 - f - drag) + dt a] / (1 + f), x' = x + dt v'.
@compute @workgroup_size(64)
fn kick_drift(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  let n = n_atoms();
  if (i >= n) { return; }
  let dt = cs.barostat.w;
  let mass = max(sys[2u * i].w, 1e-12);
  let gradient = out[n + i].xyz;
  let v = state[i];
  var keep = vec3<f32>(1.0);
  var kick = -dt * ACCEL / mass;
  if (COUPLED) {
    var f = cs.factor.x;
    if ((groups[i] & 1u) != 0u) { f = cs.factor.y; }
    let inverse = 1.0 / (1.0 + f);
    keep = (vec3<f32>(1.0 - f) - cs.drag.xyz) * inverse;
    kick *= inverse;
  }
  let velocity = v.xyz * keep + kick * gradient;
  let position = sys[2u * i + 1u].xyz;
  state[i] = vec4<f32>(velocity, v.w);
  state[n + i] = vec4<f32>(position, 0.0);
  sys[2u * i + 1u] = vec4<f32>(position + dt * velocity, 0.0);
}

// Pressure of this step from the virial of the stored forces, the constraint
// forces of this step and the mean of the two half-step kinetic energies.
@compute @workgroup_size(1)
fn pressure() {
  let n = n_atoms();
  let dt = cs.barostat.w;
  let kinetic = 0.5 * ((cs.kinetic_old.x + cs.kinetic_old.y) + (cs.kinetic_new.x + cs.kinetic_new.y));
  let virial = out[n].w + cs.kinetic_new.z / (dt * dt * ACCEL);
  let b = cs.box_.xyz;
  let volume = b.x * b.y * b.z;
  let value = (2.0 * kinetic + virial) / (3.0 * volume) + cs.barostat.z / (volume * volume);
  cs.pressure = vec4<f32>(value * BAR_PER_KCAL_MOL_A3, kinetic, virial, 0.0);
  cs.drag = vec4<f32>(cs.drag.xyz, 1.0);
}

// Scale the coordinates with the box. They are stored relative to the box
// center, which is the fixed point of the scaling.
@compute @workgroup_size(64)
fn scale(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= n_atoms()) { return; }
  let factor = vec3<f32>(1.0) + cs.drag.xyz;
  sys[2u * i + 1u] = vec4<f32>(sys[2u * i + 1u].xyz * factor, 0.0);
}

// The new box for every kernel. A pair list built for an older box stays
// valid while the box has changed by less than the allowance taken from the
// skin; beyond it a rebuild is requested.
@compute @workgroup_size(1)
fn apply_box() {
  let old = cs.box_.xyz;
  let next = old * (vec3<f32>(1.0) + cs.drag.xyz);
  let cutoff = cs.box_.w;
  if (!(min(next.x, min(next.y, next.z)) > 2.0 * cutoff)) {
    atomicStore(&aux[cs.sizes.z], STATUS_BOX_TOO_SMALL);
    return;
  }
  cs.box_ = vec4<f32>(next, cutoff);
  pbc_config[1] = vec4<f32>(next, pbc_config[1].w);
  tile_config[1] = vec4<f32>(next, tile_config[1].w);
  mesh_config[MESH_BOX_ELEMENT] = vec4<f32>(next, mesh_config[MESH_BOX_ELEMENT].w);
  let change = abs(next - old);
  let rebuilds = atomicLoad(&aux[cs.sizes.w + 1u]);
  var since = cs.list.x;
  if (rebuilds != bitcast<u32>(cs.list.z)) { since = 0.0; }
  // A pair across a periodic face moves by the box change along each axis.
  since += length(change);
  if (since > cs.list.y) {
    atomicStore(&aux[cs.sizes.w], 1u);
    since = 0.0;
  }
  cs.list = vec4<f32>(since, cs.list.y, bitcast<f32>(rebuilds), 0.0);
}

@compute @workgroup_size(64)
fn remove_com(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= n_atoms()) { return; }
  var drift = cs.momentum0.xyz * cs.com.x;
  if ((groups[i] & 2u) != 0u) { drift = cs.momentum1.xyz * cs.com.y; }
  state[i] = vec4<f32>(state[i].xyz - drift, state[i].w);
}
