//! Full-energy, nonperiodic dynamics. Distances Å, time ps, energy kcal/mol.
pub mod analysis;
use glysys::{ParameterizedSystem, Vec3};
use glysys_energy::{EnergyEvaluator, EnergyOptions, Obc2Options};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
const KB: f64 = 0.00198720425864083;
const ACCEL: f64 = 418.4; // (kcal/mol/Å)/amu -> Å/ps²
pub const MODEL_VERSION: &str = "obc2-baoab-v1";
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid simulation: {0}")]
    Invalid(String),
    #[error(transparent)]
    Energy(#[from] glysys_energy::EnergyError),
    #[error(transparent)]
    Optimization(#[from] glysys_opt::OptimizationError),
}
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "camelCase")]
pub struct SimulationProtocol {
    pub temperature_k: f64,
    pub timestep_fs: f64,
    pub friction_per_ps: f64,
    pub equilibration_steps: usize,
    pub production_steps: usize,
    pub save_every: usize,
    pub minimization_iterations: usize,
    #[serde(with="u64_string")]
    pub seed: u64,
}
impl Default for SimulationProtocol {
    fn default() -> Self {
        Self {
            temperature_k: 300.,
            timestep_fs: 0.5,
            friction_per_ps: 1.,
            equilibration_steps: 2000,
            production_steps: 18000,
            save_every: 100,
            minimization_iterations: 500,
            seed: 0,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulationState {
    pub schema_version: u32,
    pub model_version: String,
    pub system_fingerprint: String,
    pub protocol: SimulationProtocol,
    pub step: usize,
    pub coordinates: Vec<Vec3>,
    pub reference_coordinates: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub gradient: Vec<Vec3>,
    pub potential_energy: f64,
    #[serde(with = "u64_string")]
    pub rng_state: u64,
    pub integrator_phase: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrajectoryFrame {
    pub step: usize,
    pub time_ps: f64,
    pub segment: String,
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub temperature_k: f64,
    pub coordinates: Vec<Vec3>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrajectoryChunk {
    pub first_step: usize,
    pub last_step: usize,
    pub frames: Vec<TrajectoryFrame>,
}
mod u64_string {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(
        value: &u64,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<u64, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Value {
            Text(String),
            Number(u64),
        }
        match Value::deserialize(deserializer)? {
            Value::Text(s) => s.parse().map_err(serde::de::Error::custom),
            Value::Number(n) => Ok(n),
        }
    }
}
fn invalid(s: &str) -> Error {
    Error::Invalid(s.into())
}
fn add(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
    }
}
fn scale(a: Vec3, s: f64) -> Vec3 {
    Vec3 {
        x: a.x * s,
        y: a.y * s,
        z: a.z * s,
    }
}
fn norm2(a: Vec3) -> f64 {
    a.x * a.x + a.y * a.y + a.z * a.z
}
fn finite(a: &Vec3) -> bool {
    a.x.is_finite() && a.y.is_finite() && a.z.is_finite()
}
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    ((x >> 11) as f64 + 0.5) / 9007199254740992.
}
fn normal(state: &mut u64) -> f64 {
    (-2. * uniform(state).ln()).sqrt() * (std::f64::consts::TAU * uniform(state)).cos()
}
/// Advance the same RNG stream as CPU BAOAB, packing independent normal triples.
pub fn normal_noise(state: &mut u64, count: usize) -> Vec<[f32; 4]> {
    (0..count)
        .map(|_| {
            [
                normal(state) as f32,
                normal(state) as f32,
                normal(state) as f32,
                0.,
            ]
        })
        .collect()
}
fn random_velocity(state: &mut u64, s: f64) -> Vec3 {
    Vec3 {
        x: normal(state) * s,
        y: normal(state) * s,
        z: normal(state) * s,
    }
}
fn fingerprint(system: &ParameterizedSystem) -> String {
    format!(
        "{:x}",
        Sha256::digest(format!("{MODEL_VERSION}:{system:?}").as_bytes())
    )
}
impl SimulationProtocol {
    pub fn validate(&self) -> Result<()> {
        if !self.temperature_k.is_finite()
            || self.temperature_k <= 0.
            || !self.timestep_fs.is_finite()
            || self.timestep_fs <= 0.
            || self.timestep_fs > 0.5
            || !self.friction_per_ps.is_finite()
            || self.friction_per_ps < 0.
            || self.save_every == 0
            || self.production_steps == 0
            || self
                .equilibration_steps
                .checked_add(self.production_steps)
                .is_none_or(|s| s > 10_000_000)
        {
            return Err(invalid(
                "positive temperature, timestep ≤0.5 fs, finite friction and bounded step counts required",
            ));
        }
        Ok(())
    }
}
/// Transactional BAOAB: an error leaves the committed state and RNG untouched.
pub fn step_with<F>(state: &mut SimulationState, masses: &[f64], mut evaluate: F) -> Result<()>
where
    F: FnMut(&[Vec3]) -> Result<(f64, Vec<Vec3>)>,
{
    state.protocol.validate()?;
    if masses.len() != state.coordinates.len()
        || state.velocities.len() != masses.len()
        || state.gradient.len() != masses.len()
        || masses.iter().any(|m| !m.is_finite() || *m <= 0.)
    {
        return Err(invalid("invalid masses or state dimensions"));
    }
    let mut next = state.clone();
    let dt = next.protocol.timestep_fs * 0.001;
    let decay = (-next.protocol.friction_per_ps * dt).exp();
    for (i, mass) in masses.iter().enumerate() {
        next.velocities[i] = add(
            next.velocities[i],
            scale(next.gradient[i], -0.5 * dt * ACCEL / mass),
        );
        next.coordinates[i] = add(next.coordinates[i], scale(next.velocities[i], 0.5 * dt));
        let sigma = ((1. - decay * decay) * KB * next.protocol.temperature_k * ACCEL / mass).sqrt();
        next.velocities[i] = add(
            scale(next.velocities[i], decay),
            random_velocity(&mut next.rng_state, sigma),
        );
        next.coordinates[i] = add(next.coordinates[i], scale(next.velocities[i], 0.5 * dt));
    }
    let (energy, gradient) = evaluate(&next.coordinates)?;
    if !energy.is_finite() || gradient.len() != masses.len() || gradient.iter().any(|g| !finite(g))
    {
        return Err(invalid(
            "nonfinite energy or gradient; previous checkpoint retained",
        ));
    }
    for (i, mass) in masses.iter().enumerate() {
        next.velocities[i] = add(
            next.velocities[i],
            scale(gradient[i], -0.5 * dt * ACCEL / mass),
        );
        let displacement = add(next.coordinates[i], scale(state.coordinates[i], -1.));
        if !finite(&next.coordinates[i]) || !finite(&next.velocities[i]) || norm2(displacement) > 1.
        {
            return Err(invalid(
                "unstable integration (>1 Å per step); previous checkpoint retained",
            ));
        }
    }
    next.gradient = gradient;
    next.potential_energy = energy;
    next.step += 1;
    *state = next;
    Ok(())
}
pub struct CpuSimulation<'a> {
    evaluator: EnergyEvaluator<'a>,
    masses: Vec<f64>,
    pub state: SimulationState,
}
impl<'a> CpuSimulation<'a> {
    pub fn new(system: &'a ParameterizedSystem, protocol: SimulationProtocol) -> Result<Self> {
        protocol.validate()?;
        if system.atoms().is_empty()
            || system
                .atoms()
                .iter()
                .any(|a| !a.mass().is_finite() || a.mass() <= 0.)
        {
            return Err(invalid("finite positive masses required"));
        }
        if system.atoms().iter().any(|a| {
            matches!(
                system.residues()[a.residue_index()].name(),
                "HOH" | "WAT" | "TIP3"
            )
        }) {
            return Err(invalid(
                "initial dynamics supports implicit solvent, without explicit waters",
            ));
        }
        let evaluator = EnergyEvaluator::new(
            system,
            EnergyOptions {
                obc2: Some(Obc2Options::default()),
                ..Default::default()
            },
        )?;
        let mut coordinates = system.coordinates();
        if protocol.minimization_iterations > 0 {
            let config = glysys_opt::LbfgsConfig {
                max_iterations: protocol.minimization_iterations,
                ..Default::default()
            };
            let flat: Vec<_> = coordinates.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
            let mut optimizer = glysys_opt::resumable::LbfgsState::new(&flat, &config)?;
            while let Some(p) = optimizer.request() {
                let coords: Vec<_> = p
                    .chunks_exact(3)
                    .map(|p| Vec3 {
                        x: p[0],
                        y: p[1],
                        z: p[2],
                    })
                    .collect();
                let energy = evaluator.energy_and_gradient(&coords)?;
                let total = energy.total();
                let gradient = energy
                    .gradients
                    .unwrap()
                    .iter()
                    .flat_map(|p| [p.x, p.y, p.z])
                    .collect();
                optimizer.submit(total, gradient)?;
            }
            coordinates = optimizer
                .outcome()
                .unwrap()
                .point
                .chunks_exact(3)
                .map(|p| Vec3 {
                    x: p[0],
                    y: p[1],
                    z: p[2],
                })
                .collect();
        }
        let e = evaluator.energy_and_gradient(&coordinates)?;
        let potential_energy = e.total();
        let masses: Vec<_> = system.atoms().iter().map(|a| a.mass()).collect();
        let mut rng = protocol.seed;
        let velocities = masses
            .iter()
            .map(|m| random_velocity(&mut rng, (KB * protocol.temperature_k * ACCEL / m).sqrt()))
            .collect();
        let state = SimulationState {
            schema_version: 1,
            model_version: MODEL_VERSION.into(),
            system_fingerprint: fingerprint(system),
            protocol,
            step: 0,
            reference_coordinates: coordinates.clone(),
            coordinates,
            velocities,
            gradient: e.gradients.unwrap(),
            potential_energy,
            rng_state: rng,
            integrator_phase: "ready".into(),
        };
        Ok(Self {
            evaluator,
            masses,
            state,
        })
    }
    pub fn restore(system: &'a ParameterizedSystem, state: SimulationState) -> Result<Self> {
        state.protocol.validate()?;
        if state.schema_version != 1
            || state.model_version != MODEL_VERSION
            || state.system_fingerprint != fingerprint(system)
            || state.integrator_phase != "ready"
            || state
                .coordinates
                .iter()
                .chain(&state.velocities)
                .chain(&state.gradient)
                .any(|v| !finite(v))
            || !state.potential_energy.is_finite()
            || state.reference_coordinates.len() != system.atom_count()
            || state.reference_coordinates.iter().any(|v| !finite(v))
            || state.coordinates.len() != system.atom_count()
            || state.velocities.len() != system.atom_count()
            || state.gradient.len() != system.atom_count()
            || state.step > state.protocol.equilibration_steps + state.protocol.production_steps
        {
            return Err(invalid("incompatible or invalid simulation checkpoint"));
        }
        let evaluator = EnergyEvaluator::new(
            system,
            EnergyOptions {
                obc2: Some(Obc2Options::default()),
                ..Default::default()
            },
        )?;
        let reference = evaluator.energy_and_gradient(&state.coordinates)?;
        if (reference.total() - state.potential_energy).abs()
            > 1e-3 + 1e-4 * reference.total().abs()
            || reference
                .gradients
                .as_ref()
                .unwrap()
                .iter()
                .zip(&state.gradient)
                .any(|(a, b)| {
                    norm2(add(*a, scale(*b, -1.))).sqrt() > 1e-3 + 1e-3 * norm2(*a).sqrt()
                })
        {
            return Err(invalid("checkpoint forces do not match prepared chemistry"));
        }
        Ok(Self {
            evaluator,
            masses: system.atoms().iter().map(|a| a.mass()).collect(),
            state,
        })
    }
    pub fn advance(&mut self, steps: usize) -> Result<TrajectoryChunk> {
        let first_step = self.state.step;
        let end = (self.state.step + steps.min(100))
            .min(self.state.protocol.equilibration_steps + self.state.protocol.production_steps);
        let mut frames = Vec::new();
        while self.state.step < end {
            step_with(&mut self.state, &self.masses, |p| {
                let e = self.evaluator.energy_and_gradient(p)?;
                Ok((e.total(), e.gradients.unwrap()))
            })?;
            if self
                .state
                .step
                .is_multiple_of(self.state.protocol.save_every)
                || self.state.step == self.state.protocol.equilibration_steps
                || self.state.step
                    == self.state.protocol.equilibration_steps
                        + self.state.protocol.production_steps
            {
                frames.push(self.frame());
            }
        }
        Ok(TrajectoryChunk {
            first_step,
            last_step: self.state.step,
            frames,
        })
    }
    pub fn frame(&self) -> TrajectoryFrame {
        frame_from_state(&self.state, &self.masses)
    }
}
pub fn frame_from_state(state: &SimulationState, masses: &[f64]) -> TrajectoryFrame {
    let kinetic_energy = masses
        .iter()
        .zip(&state.velocities)
        .map(|(m, v)| m * norm2(*v) / (2. * ACCEL))
        .sum::<f64>();
    TrajectoryFrame {
        step: state.step,
        time_ps: state.step as f64 * state.protocol.timestep_fs * 0.001,
        segment: if state.step <= state.protocol.equilibration_steps {
            "equilibration"
        } else {
            "production"
        }
        .into(),
        potential_energy: state.potential_energy,
        kinetic_energy,
        temperature_k: 2. * kinetic_energy / (3. * masses.len() as f64 * KB),
        coordinates: state.coordinates.clone(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn toy() -> SimulationState {
        SimulationState {
            schema_version: 1,
            model_version: MODEL_VERSION.into(),
            system_fingerprint: "toy".into(),
            protocol: SimulationProtocol {
                friction_per_ps: 0.,
                minimization_iterations: 0,
                ..Default::default()
            },
            step: 0,
            coordinates: vec![Vec3 {
                x: 0.1,
                y: 0.,
                z: 0.,
            }],
            reference_coordinates: vec![Vec3 { x: 0.1, y: 0., z: 0. }],
            velocities: vec![Vec3 {
                x: 0.,
                y: 0.,
                z: 0.,
            }],
            gradient: vec![Vec3 {
                x: 0.1,
                y: 0.,
                z: 0.,
            }],
            potential_energy: 0.005,
            rng_state: 0,
            integrator_phase: "ready".into(),
        }
    }
    #[test]
    fn harmonic_nve_energy_and_checkpoint_replay() {
        let mut s = toy();
        let force = |p: &[Vec3]| Ok((0.5 * norm2(p[0]), vec![p[0]]));
        for _ in 0..10000 {
            step_with(&mut s, &[12.], force).unwrap();
            let total = s.potential_energy + 12. * norm2(s.velocities[0]) / (2. * ACCEL);
            assert!((total - 0.005).abs() < 1e-7);
        }
        let mut restored: SimulationState =
            serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        for _ in 0..50 {
            step_with(&mut s, &[12.], force).unwrap();
            step_with(&mut restored, &[12.], force).unwrap();
        }
        assert_eq!(s.coordinates, restored.coordinates);
        assert_eq!(s.velocities, restored.velocities);
    }
    #[test]
    fn failed_step_preserves_rng_and_coordinates() {
        let mut s = toy();
        let before = serde_json::to_string(&s).unwrap();
        assert!(step_with(&mut s, &[12.], |_| Err(invalid("device failure"))).is_err());
        assert_eq!(before, serde_json::to_string(&s).unwrap());
    }
    #[test]
    fn thermostat_recovers_kinetic_temperature() {
        let mut s = toy();
        s.protocol.friction_per_ps = 10.;
        s.gradient[0] = Vec3 {
            x: 0.,
            y: 0.,
            z: 0.,
        };
        let mut sum = 0.;
        for i in 0..200000 {
            step_with(&mut s, &[12.], |_| {
                Ok((
                    0.,
                    vec![Vec3 {
                        x: 0.,
                        y: 0.,
                        z: 0.,
                    }],
                ))
            })
            .unwrap();
            if i >= 10000 {
                sum += 12. * norm2(s.velocities[0]) / (3. * ACCEL * KB);
            }
        }
        assert!((sum / 190000. - 300.).abs() < 20.);
    }
}
