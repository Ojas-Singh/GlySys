//! Resumable preparation minimization. Both synchronous and asynchronous
//! evaluators drive these same pre-relaxation and Armijo decisions.
use crate::{Result, invalid};
use glysys_opt::{LbfgsConfig, resumable::LbfgsState};

#[derive(Clone)]
enum Phase {
    PrePoint,
    PreTrial,
    Lbfgs,
}
#[derive(Clone)]
pub struct PreparationMinimizer {
    phase: Phase,
    point: Vec<f64>,
    request: Vec<f64>,
    gradient: Vec<f64>,
    value: f64,
    step: f64,
    iteration: usize,
    trials: usize,
    evaluations: usize,
    budget: usize,
    optimizer: Option<LbfgsState>,
}
pub struct MinimizationProgress {
    pub stage: &'static str,
    pub completed: usize,
    pub total: usize,
    pub energy: f64,
    pub max_gradient: f64,
}
impl PreparationMinimizer {
    pub fn new(point: &[f64], budget: usize, pre_relax: bool) -> Result<Self> {
        if point.is_empty()
            || point.len() % 3 != 0
            || budget == 0
            || point.iter().any(|x| !x.is_finite())
        {
            return Err(invalid("invalid preparation minimization input"));
        }
        let mut result = Self {
            phase: Phase::PrePoint,
            point: point.to_vec(),
            request: point.to_vec(),
            gradient: Vec::new(),
            value: 0.,
            step: 0.01,
            iteration: 0,
            trials: 0,
            evaluations: 0,
            budget,
            optimizer: None,
        };
        if !pre_relax {
            result.begin_lbfgs()?;
        }
        Ok(result)
    }
    fn begin_lbfgs(&mut self) -> Result<()> {
        self.optimizer = Some(LbfgsState::new(
            &self.point,
            &LbfgsConfig {
                max_iterations: self.budget,
                ..Default::default()
            },
        )?);
        self.phase = Phase::Lbfgs;
        Ok(())
    }
    pub fn request(&mut self) -> Option<&[f64]> {
        match self.phase {
            Phase::Lbfgs => self.optimizer.as_mut().and_then(|state| state.request()),
            _ => Some(&self.request),
        }
    }
    fn trial(&mut self) {
        for ((trial, point), gradient) in
            self.request.iter_mut().zip(&self.point).zip(&self.gradient)
        {
            *trial = point - self.step * gradient;
        }
    }
    pub fn submit(&mut self, value: f64, gradient: Vec<f64>) -> Result<Vec<MinimizationProgress>> {
        if gradient.len() != self.point.len() {
            return Err(invalid("minimization gradient dimensions"));
        }
        self.evaluations += 1;
        let mut events = Vec::new();
        match self.phase {
            Phase::PrePoint => {
                let max_norm = gradient
                    .chunks_exact(3)
                    .map(|g| g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
                    .fold(0., f64::max)
                    .sqrt();
                if !value.is_finite() || gradient.iter().any(|g| !g.is_finite()) {
                    return Err(invalid("nonfinite gradient during minimization"));
                }
                events.push(MinimizationProgress {
                    stage: "minimize-pre",
                    completed: self.iteration + 1,
                    total: self.budget.min(500),
                    energy: value,
                    max_gradient: max_norm,
                });
                if max_norm < 50. {
                    self.begin_lbfgs()?;
                } else {
                    self.value = value;
                    self.gradient = gradient;
                    self.trials = 0;
                    self.phase = Phase::PreTrial;
                    self.trial();
                }
            }
            Phase::PreTrial => {
                let maximum = gradient.iter().map(|x| x.abs()).fold(0., f64::max);
                self.trials += 1;
                events.push(MinimizationProgress {
                    stage: "minimize-pre-trial",
                    completed: self.trials,
                    total: 20,
                    energy: value,
                    max_gradient: maximum,
                });
                if value.is_finite() && value < self.value {
                    self.point.clone_from(&self.request);
                    self.step = (self.step * 1.5).min(0.05);
                    self.iteration += 1;
                    if self.iteration >= self.budget.min(500) {
                        self.begin_lbfgs()?;
                    } else {
                        self.phase = Phase::PrePoint;
                    }
                } else {
                    self.step *= 0.5;
                    if self.step < 1e-8 || self.trials >= 20 {
                        self.begin_lbfgs()?;
                    } else {
                        self.trial();
                    }
                }
            }
            Phase::Lbfgs => {
                let maximum = gradient.iter().map(|x| x.abs()).fold(0., f64::max);
                events.push(MinimizationProgress {
                    stage: "minimize-trial",
                    completed: self.evaluations,
                    total: 0,
                    energy: value,
                    max_gradient: maximum,
                });
                if let Some(event) = self
                    .optimizer
                    .as_mut()
                    .ok_or_else(|| invalid("missing minimizer"))?
                    .submit(value, gradient)?
                {
                    events.push(MinimizationProgress {
                        stage: "minimize-lbfgs",
                        completed: event.iteration.min(self.budget),
                        total: self.budget,
                        energy: event.value,
                        max_gradient: event.max_gradient,
                    });
                }
            }
        }
        Ok(events)
    }
    pub fn outcome(&self) -> Option<Vec<f64>> {
        self.optimizer
            .as_ref()?
            .outcome()
            .map(|outcome| outcome.point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn suspended_trials_replay_without_changing_minimum() {
        let mut optimizer = PreparationMinimizer::new(&[8., -4., 2.], 40, true).unwrap();
        let mut replay = optimizer.clone();
        for _ in 0..1000 {
            let Some(point) = optimizer.request().map(<[f64]>::to_vec) else {
                break;
            };
            assert_eq!(replay.request().unwrap(), point);
            let value = point.iter().map(|x| 100. * x * x).sum();
            let gradient: Vec<_> = point.iter().map(|x| 200. * x).collect();
            optimizer.submit(value, gradient.clone()).unwrap();
            replay.submit(value, gradient).unwrap();
        }
        let result = optimizer.outcome().unwrap();
        assert_eq!(Some(result.clone()), replay.outcome());
        assert!(result.iter().all(|x| x.abs() < 1e-6));
    }
}

/// Reusable reference/fallback evaluator for preparation. It owns immutable
/// chemistry once; line-search trials only change coordinates and neighbors.
pub enum CpuMinimizationEvaluator {
    Implicit(glysys_energy::EnergyEvaluator<'static>),
    Explicit {
        field: glysys_energy::pbc::PbcForceField<'static>,
        box_vectors: glysys_energy::pbc::BoxVectors,
        backend: glysys_energy::pbc::ReactionField,
        cutoff: f64,
        neighbors: Option<glysys_energy::pbc::PbcNeighborList>,
    },
}
impl CpuMinimizationEvaluator {
    pub fn new(
        system: &glysys::ParameterizedSystem,
        protocol: &crate::SimulationProtocol,
    ) -> Result<Self> {
        protocol.validate()?;
        if protocol.solvent == crate::SolventModel::Explicit {
            let cutoff = protocol.cutoff_angstrom.unwrap_or(9.);
            Ok(Self::Explicit {
                field: glysys_energy::pbc::PbcForceField::new(
                    system,
                    crate::explicit::solute_restraints(system, protocol.restraint_force),
                )?
                .into_owned(),
                box_vectors: glysys_energy::pbc::BoxVectors::from_system(system)?,
                backend: glysys_energy::pbc::ReactionField::new(
                    cutoff,
                    protocol.rf_dielectric.unwrap_or(78.5),
                )?,
                cutoff,
                neighbors: None,
            })
        } else {
            Ok(Self::Implicit(
                glysys_energy::EnergyEvaluator::new(
                    system,
                    glysys_energy::EnergyOptions {
                        obc2: Some(Default::default()),
                        ..Default::default()
                    },
                )?
                .into_owned(),
            ))
        }
    }
    pub fn evaluate(&mut self, coordinates: &[glysys::Vec3]) -> Result<(f64, Vec<glysys::Vec3>)> {
        match self {
            Self::Implicit(evaluator) => {
                let energy = evaluator.energy_and_gradient(coordinates)?;
                Ok((
                    energy.total(),
                    energy
                        .gradients
                        .ok_or_else(|| invalid("missing minimization gradient"))?,
                ))
            }
            Self::Explicit {
                field,
                box_vectors,
                backend,
                cutoff,
                neighbors,
            } => {
                let wrapped: Vec<_> = coordinates.iter().map(|p| box_vectors.wrap(*p)).collect();
                if neighbors
                    .as_ref()
                    .is_none_or(|list| list.needs_rebuild(&wrapped))
                {
                    *neighbors = Some(glysys_energy::pbc::PbcNeighborList::build(
                        &wrapped,
                        box_vectors,
                        *cutoff,
                        1.5,
                    )?);
                }
                let energy = field.evaluate(
                    coordinates,
                    box_vectors,
                    &neighbors
                        .as_ref()
                        .ok_or_else(|| invalid("missing minimization neighbors"))?
                        .pairs,
                    backend,
                    *cutoff,
                )?;
                Ok((energy.components.total(), energy.gradients))
            }
        }
    }
}
