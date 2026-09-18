//! Version-one resident thermostat stream, matching canonical pbc.wgsl.
//! The integer stream is exact across backends; transcendental f32 results
//! are subject to the same numerical tolerances as the force kernels.
use glysys::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResidentThermostatRng {
    pub version: u32,
    pub words: Vec<u32>,
}
impl ResidentThermostatRng {
    pub fn seeded(seed: u64, atoms: usize) -> Self {
        let mut stream = seed ^ 0xD1B5_4A32_D192_ED03;
        let words = (0..atoms)
            .map(|_| {
                stream = stream.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = stream;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((z ^ (z >> 31)) as u32).max(1)
            })
            .collect();
        Self { version: 1, words }
    }
    pub fn validate(&self, atoms: usize) -> crate::Result<()> {
        if self.version != 1 || self.words.len() != atoms || self.words.contains(&0) {
            return Err(crate::invalid("incompatible resident thermostat RNG state"));
        }
        Ok(())
    }
    pub fn normal3(&mut self, atom: usize) -> Vec3 {
        let word = &mut self.words[atom];
        let mut uniform = || {
            *word ^= *word << 13;
            *word ^= *word >> 17;
            *word ^= *word << 5;
            ((*word & 0x00ff_ffff) as f32 + 0.5) / 16_777_216.
        };
        let mut normal = || {
            let u = uniform().max(1e-7);
            let v = uniform();
            ((-2. * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()) as f64
        };
        Vec3 {
            x: normal(),
            y: normal(),
            z: normal(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn thermostat_variates_have_gaussian_moments_and_no_lag_one_bias() {
        let atoms = 256;
        let mut rng = ResidentThermostatRng::seeded(11, atoms);
        let mut previous = vec![0.; atoms];
        let mut sum = 0.;
        let mut square = 0.;
        let mut fourth = 0.;
        let mut cross = 0.;
        let mut count = 0.;
        for step in 0..8192 {
            for atom in 0..atoms {
                let p = rng.normal3(atom);
                sum += p.x;
                square += p.x * p.x;
                fourth += p.x.powi(4);
                if step > 0 {
                    cross += p.x * previous[atom];
                }
                previous[atom] = p.x;
                count += 1.;
            }
        }
        // About two million samples. Bounds exceed five independent-sample
        // standard errors, including the larger uncertainty of the 4th moment.
        assert!((sum / count).abs() < 0.004);
        assert!((square / count - 1.).abs() < 0.006);
        assert!((fourth / count - 3.).abs() < 0.04);
        assert!((cross / (count - atoms as f64)).abs() < 0.004);
    }
}
