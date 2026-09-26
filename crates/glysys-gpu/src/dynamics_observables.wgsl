@group(0) @binding(0) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> masses: array<f32>;
@group(0) @binding(2) var<storage, read> energies: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<vec4<f32>>;

// The energy evaluator stores one 12-float summary after its per-atom output.
// One invocation is sufficient for the small implicit qualification systems,
// and keeps every coordinate/velocity/force buffer resident on the GPU.
@compute @workgroup_size(1)
fn reduce_dynamics_observation() {
  let n = arrayLength(&velocities);
  let summary = 16u * n;
  var potential = 0.0;
  var kinetic = 0.0;
  var potential_compensation = 0.0;
  var kinetic_compensation = 0.0;
  for (var i = 0u; i < 9u; i++) {
    let y = energies[summary + i] - potential_compensation;
    let next = potential + y;
    potential_compensation = (next - potential) - y;
    potential = next;
  }
  for (var i = 0u; i < n; i++) {
    let velocity = velocities[i].xyz;
    let value = masses[i] * dot(velocity, velocity) / (2.0 * 418.4);
    let y = value - kinetic_compensation;
    let next = kinetic + y;
    kinetic_compensation = (next - kinetic) - y;
    kinetic = next;
  }
  result[0] = vec4<f32>(potential, kinetic, 0.0, 0.0);
}
