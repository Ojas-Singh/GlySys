fn rng_next(value: u32) -> u32 {
  var x = value;
  x ^= x << 13u;
  x ^= x >> 17u;
  x ^= x << 5u;
  return x;
}

fn rng_uniform(value: u32) -> f32 {
  return (f32(value & 0x00ffffffu) + 0.5) / 16777216.0;
}

// Version 1: six xorshift draws and three Box–Muller cosine variates.
// Shared by explicit and implicit integrators; CPU reference: resident_rng.rs.
fn rng_normal3(initial: u32) -> vec4<f32> {
  var seed = initial;
  seed = rng_next(seed); let u1 = max(rng_uniform(seed), 1e-7);
  seed = rng_next(seed); let u2 = rng_uniform(seed);
  let radius = sqrt(-2.0 * log(u1));
  let normal0 = radius * cos(6.283185307179586 * u2);
  seed = rng_next(seed); let u3 = max(rng_uniform(seed), 1e-7);
  seed = rng_next(seed); let u4 = rng_uniform(seed);
  let radius2 = sqrt(-2.0 * log(u3));
  let normal1 = radius2 * cos(6.283185307179586 * u4);
  seed = rng_next(seed); let u5 = max(rng_uniform(seed), 1e-7);
  seed = rng_next(seed); let u6 = rng_uniform(seed);
  let normal2 = sqrt(-2.0 * log(u5)) * cos(6.283185307179586 * u6);
  return vec4<f32>(normal0, normal1, normal2, bitcast<f32>(seed));
}
