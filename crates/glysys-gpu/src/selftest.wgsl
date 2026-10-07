// Device self-test: do integers survive in float lanes?
//
// The kernels of this crate carry integers (atom indices, sentinels, the
// thermostat's random-number words) as bit patterns in the w lane of vec4<f32>
// values. Small integers are denormal floats and 0xffffffff is a NaN, and
// WGSL does not promise that such patterns survive being copied as floats.
// Each path below copies a set of patterns the way some kernel does and
// returns the bits it ends up with.

@group(0) @binding(0) var<storage, read> patterns: array<u32>;
// xyz: ordinary floats; w: the pattern, uploaded as raw bits
@group(0) @binding(1) var<storage, read> floats: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> scratch: array<vec4<f32>>;
// two entries per pattern
@group(0) @binding(3) var<storage, read_write> results: array<vec4<u32>>;

var<workgroup> shared_values: array<vec4<f32>, 64>;

fn rebuilt(v: vec4<f32>, scale: f32) -> vec4<f32> {
  return vec4<f32>(v.xyz * scale, v.w);
}

// As a kernel that seeds a float buffer with integer bits.
@compute @workgroup_size(64)
fn write_scratch(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&patterns)) { return; }
  scratch[i] = vec4<f32>(1.0, 2.0, 3.0, bitcast<f32>(patterns[i]));
}

// As the integrator kernels, which rewrite a velocity and keep its w lane.
@compute @workgroup_size(64)
fn rewrite_scratch(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&patterns)) { return; }
  let v = scratch[i];
  scratch[i] = vec4<f32>(v.xyz + vec3<f32>(0.5), v.w);
}

@compute @workgroup_size(64)
fn collect(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32) {
  let i = id.x;
  let live = i < arrayLength(&patterns);
  var value = vec4<f32>(0.0);
  var pattern = 0u;
  if (live) {
    value = floats[i];
    pattern = patterns[i];
  }

  // straight from a float buffer
  let buffer_load = bitcast<u32>(value.w);

  // through a function-local vector that is rebuilt in a loop, as the tile
  // kernels do with the atom they cache
  var cached = vec4<f32>(0.0, 0.0, 0.0, bitcast<f32>(0xffffffffu));
  for (var k = 0u; k < 4u; k++) {
    if (k == (i & 3u)) { cached = rebuilt(value, f32(k + 1u)); }
  }
  let via_local = bitcast<u32>(cached.w);

  // through workgroup memory
  shared_values[lid] = value;
  workgroupBarrier();
  let via_workgroup = bitcast<u32>(shared_values[lid].w);

  // through select
  let take = (i & 0x80000000u) == 0u;
  let chosen = select(vec4<f32>(-1.0), value, take);
  let selected = bitcast<u32>(chosen.w);

  // integer to float and back
  let round_trip = bitcast<u32>(bitcast<f32>(pattern));

  // a vector built around the cast integer, rebuilt once
  var around = vec4<f32>(1.0, 2.0, 3.0, bitcast<f32>(pattern));
  around = rebuilt(around, 2.0);
  let local_from_cast = bitcast<u32>(around.w);

  // written to a float buffer by one kernel, rewritten by another, read here
  var stored = 0u;
  if (live) { stored = bitcast<u32>(scratch[i].w); }

  // the comparison the tile kernels make against their sentinel
  let is_sentinel = u32(bitcast<u32>(cached.w) == 0xffffffffu);

  if (live) {
    results[2u * i] = vec4<u32>(buffer_load, via_local, via_workgroup, selected);
    results[2u * i + 1u] = vec4<u32>(round_trip, local_from_cast, stored, is_sentinel);
  }
}
