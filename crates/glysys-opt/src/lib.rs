//! Deterministic, application-independent optimization algorithms.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

pub type Result<T> = std::result::Result<T, OptimizationError>;

#[derive(Debug, thiserror::Error)]
pub enum OptimizationError {
    #[error("invalid optimizer configuration: {0}")]
    InvalidConfiguration(String),
    #[error("objective returned a non-finite value")]
    NonFiniteObjective,
    #[error("objective dimension mismatch: expected {expected}, received {received}")]
    DimensionMismatch { expected: usize, received: usize },
}

/// Application-defined operations required by the generic genetic algorithm.
pub trait GeneticProblem: Sync {
    type State: Clone + Send + Sync;

    fn generate(&self, rng: &mut ChaCha8Rng) -> Self::State;
    fn crossover(
        &self,
        first: &Self::State,
        second: &Self::State,
        rng: &mut ChaCha8Rng,
    ) -> Self::State;
    fn mutate(&self, state: &mut Self::State, rng: &mut ChaCha8Rng, rate: f64);
    fn repair(&self, _state: &mut Self::State, _rng: &mut ChaCha8Rng) {}
    fn evaluate(&self, state: &Self::State) -> f64;
    fn is_solution(&self, _state: &Self::State, _score: f64) -> bool {
        false
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GeneticAlgorithmConfig {
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub elite_fraction: f64,
    pub tournament_size: usize,
    pub seed: u64,
}

impl Default for GeneticAlgorithmConfig {
    fn default() -> Self {
        Self {
            population_size: 128,
            generations: 100,
            mutation_rate: 0.15,
            elite_fraction: 0.1,
            tournament_size: 3,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenerationRecord {
    pub generation: usize,
    pub best_score: f64,
    pub mean_score: f64,
}

#[derive(Debug, Clone)]
pub struct GeneticAlgorithmOutcome<S> {
    pub best_state: S,
    pub best_score: f64,
    pub generations: usize,
    pub history: Vec<GenerationRecord>,
}

pub fn genetic_optimize<P>(
    problem: &P,
    config: &GeneticAlgorithmConfig,
) -> Result<GeneticAlgorithmOutcome<P::State>>
where
    P: GeneticProblem,
{
    validate_genetic_config(config)?;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut population = (0..config.population_size)
        .map(|_| problem.generate(&mut rng))
        .collect::<Vec<_>>();
    let mut history = Vec::with_capacity(config.generations + 1);
    let elite_count = ((config.population_size as f64 * config.elite_fraction).round() as usize)
        .clamp(1, config.population_size);

    for generation in 0..=config.generations {
        let mut scored = population
            .par_iter()
            .map(|state| (problem.evaluate(state), state.clone()))
            .collect::<Vec<_>>();
        if scored.iter().any(|(score, _)| !score.is_finite()) {
            return Err(OptimizationError::NonFiniteObjective);
        }
        scored.sort_by(|left, right| left.0.total_cmp(&right.0));
        history.push(GenerationRecord {
            generation,
            best_score: scored[0].0,
            mean_score: scored.iter().map(|entry| entry.0).sum::<f64>() / scored.len() as f64,
        });
        if problem.is_solution(&scored[0].1, scored[0].0) {
            return Ok(GeneticAlgorithmOutcome {
                best_state: scored[0].1.clone(),
                best_score: scored[0].0,
                generations: generation,
                history,
            });
        }
        if generation == config.generations {
            return Ok(GeneticAlgorithmOutcome {
                best_state: scored[0].1.clone(),
                best_score: scored[0].0,
                generations: generation,
                history,
            });
        }

        let generation_seed = splitmix64(config.seed ^ generation as u64);
        let mut next = scored
            .iter()
            .take(elite_count)
            .map(|entry| entry.1.clone())
            .collect::<Vec<_>>();
        let needed = config.population_size - next.len();
        let children = (0..needed)
            .into_par_iter()
            .map(|child_index| {
                let mut child_rng =
                    ChaCha8Rng::seed_from_u64(splitmix64(generation_seed ^ child_index as u64));
                let first = tournament(&scored, config.tournament_size, &mut child_rng);
                let second = tournament(&scored, config.tournament_size, &mut child_rng);
                let mut child = problem.crossover(first, second, &mut child_rng);
                problem.mutate(&mut child, &mut child_rng, config.mutation_rate);
                problem.repair(&mut child, &mut child_rng);
                child
            })
            .collect::<Vec<_>>();
        next.extend(children);
        population = next;
    }
    unreachable!()
}

fn validate_genetic_config(config: &GeneticAlgorithmConfig) -> Result<()> {
    if config.population_size < 2
        || config.generations == 0
        || !(0.0..=1.0).contains(&config.mutation_rate)
        || !(0.0..=1.0).contains(&config.elite_fraction)
        || config.elite_fraction == 0.0
        || config.tournament_size == 0
    {
        return Err(OptimizationError::InvalidConfiguration(
            "population>=2, generations>0, rates in [0,1], and tournament>0 are required".into(),
        ));
    }
    Ok(())
}

fn tournament<'a, S>(scored: &'a [(f64, S)], size: usize, rng: &mut ChaCha8Rng) -> &'a S {
    let mut best = rng.random_range(0..scored.len());
    for _ in 1..size {
        let candidate = rng.random_range(0..scored.len());
        if scored[candidate].0 < scored[best].0 {
            best = candidate;
        }
    }
    &scored[best].1
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

/// A differentiable scalar objective.
pub trait DifferentiableObjective {
    fn dimension(&self) -> usize;
    fn value_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> Result<f64>;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LbfgsConfig {
    pub max_iterations: usize,
    pub history_size: usize,
    pub gradient_tolerance: f64,
    pub initial_step: f64,
    pub armijo: f64,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            history_size: 10,
            gradient_tolerance: 1.0e-4,
            initial_step: 0.1,
            armijo: 1.0e-4,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LbfgsOutcome {
    pub point: Vec<f64>,
    pub value: f64,
    pub iterations: usize,
    pub converged: bool,
    pub history: Vec<f64>,
}

pub fn lbfgs_minimize<O: DifferentiableObjective>(
    objective: &mut O,
    initial: &[f64],
    config: &LbfgsConfig,
) -> Result<LbfgsOutcome> {
    if initial.len() != objective.dimension() {
        return Err(OptimizationError::DimensionMismatch {
            expected: objective.dimension(),
            received: initial.len(),
        });
    }
    if config.max_iterations == 0
        || config.history_size == 0
        || config.gradient_tolerance <= 0.0
        || config.initial_step <= 0.0
    {
        return Err(OptimizationError::InvalidConfiguration(
            "positive iteration, history, tolerance, and step values are required".into(),
        ));
    }
    let mut point = initial.to_vec();
    let mut gradient = vec![0.0; point.len()];
    let mut value = objective.value_gradient(&point, &mut gradient)?;
    let mut values = vec![value];
    let mut s_history: Vec<Vec<f64>> = Vec::new();
    let mut y_history: Vec<Vec<f64>> = Vec::new();
    let mut rho_history: Vec<f64> = Vec::new();

    for iteration in 0..config.max_iterations {
        if infinity_norm(&gradient) <= config.gradient_tolerance {
            return Ok(LbfgsOutcome {
                point,
                value,
                iterations: iteration,
                converged: true,
                history: values,
            });
        }
        let direction = lbfgs_direction(&gradient, &s_history, &y_history, &rho_history);
        let slope = dot(&gradient, &direction);
        let direction = if slope < 0.0 {
            direction
        } else {
            gradient.iter().map(|value| -value).collect()
        };
        let slope = dot(&gradient, &direction);
        let mut step = config.initial_step;
        let mut candidate = vec![0.0; point.len()];
        let mut candidate_gradient = vec![0.0; point.len()];
        let candidate_value = loop {
            for index in 0..point.len() {
                candidate[index] = point[index] + step * direction[index];
            }
            let trial = objective.value_gradient(&candidate, &mut candidate_gradient)?;
            if trial.is_finite() && trial <= value + config.armijo * step * slope {
                break trial;
            }
            step *= 0.5;
            if step < 1.0e-12 {
                break trial;
            }
        };
        let s = candidate
            .iter()
            .zip(&point)
            .map(|(new, old)| new - old)
            .collect::<Vec<_>>();
        let y = candidate_gradient
            .iter()
            .zip(&gradient)
            .map(|(new, old)| new - old)
            .collect::<Vec<_>>();
        let curvature = dot(&s, &y);
        if curvature > 1.0e-12 {
            if s_history.len() == config.history_size {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
            }
            s_history.push(s);
            y_history.push(y);
            rho_history.push(1.0 / curvature);
        }
        point = candidate;
        gradient = candidate_gradient;
        value = candidate_value;
        values.push(value);
    }
    Ok(LbfgsOutcome {
        point,
        value,
        iterations: config.max_iterations,
        converged: false,
        history: values,
    })
}

fn lbfgs_direction(
    gradient: &[f64],
    s_history: &[Vec<f64>],
    y_history: &[Vec<f64>],
    rho_history: &[f64],
) -> Vec<f64> {
    let mut q = gradient.to_vec();
    let mut alpha = vec![0.0; s_history.len()];
    for index in (0..s_history.len()).rev() {
        alpha[index] = rho_history[index] * dot(&s_history[index], &q);
        axpy(&mut q, -alpha[index], &y_history[index]);
    }
    if let (Some(s), Some(y)) = (s_history.last(), y_history.last()) {
        let scale = dot(s, y) / dot(y, y).max(1.0e-30);
        for value in &mut q {
            *value *= scale;
        }
    }
    for index in 0..s_history.len() {
        let beta = rho_history[index] * dot(&y_history[index], &q);
        axpy(&mut q, alpha[index] - beta, &s_history[index]);
    }
    q.into_iter().map(|value| -value).collect()
}

fn axpy(target: &mut [f64], scale: f64, source: &[f64]) {
    for (target, source) in target.iter_mut().zip(source) {
        *target += scale * source;
    }
}

fn dot(first: &[f64], second: &[f64]) -> f64 {
    first.iter().zip(second).map(|(a, b)| a * b).sum()
}

fn infinity_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::ThreadPoolBuilder;

    struct Quadratic;

    impl DifferentiableObjective for Quadratic {
        fn dimension(&self) -> usize {
            2
        }

        fn value_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> Result<f64> {
            gradient[0] = 2.0 * (point[0] - 2.0);
            gradient[1] = 4.0 * (point[1] + 1.0);
            Ok((point[0] - 2.0).powi(2) + 2.0 * (point[1] + 1.0).powi(2))
        }
    }

    #[test]
    fn lbfgs_minimizes_a_quadratic() {
        let outcome = lbfgs_minimize(&mut Quadratic, &[8.0, 3.0], &LbfgsConfig::default()).unwrap();
        assert!(outcome.converged);
        assert!((outcome.point[0] - 2.0).abs() < 1.0e-4);
        assert!((outcome.point[1] + 1.0).abs() < 1.0e-4);
        assert!(outcome.history.last().unwrap() < outcome.history.first().unwrap());
    }

    struct IntegerTarget;

    impl GeneticProblem for IntegerTarget {
        type State = i32;

        fn generate(&self, rng: &mut ChaCha8Rng) -> Self::State {
            rng.random_range(-100..=100)
        }

        fn crossover(
            &self,
            first: &Self::State,
            second: &Self::State,
            _rng: &mut ChaCha8Rng,
        ) -> Self::State {
            (first + second) / 2
        }

        fn mutate(&self, state: &mut Self::State, rng: &mut ChaCha8Rng, rate: f64) {
            if rng.random_bool(rate) {
                *state += rng.random_range(-5..=5);
            }
        }

        fn evaluate(&self, state: &Self::State) -> f64 {
            f64::from((*state - 17).pow(2))
        }
    }

    #[test]
    fn genetic_algorithm_is_deterministic_across_thread_counts() {
        let config = GeneticAlgorithmConfig {
            population_size: 64,
            generations: 30,
            seed: 42,
            ..GeneticAlgorithmConfig::default()
        };
        let run = |threads| {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| genetic_optimize(&IntegerTarget, &config).unwrap())
        };
        let serial = run(1);
        let parallel = run(4);
        assert_eq!(serial.best_state, 17);
        assert_eq!(serial.best_state, parallel.best_state);
        assert_eq!(serial.best_score, parallel.best_score);
        assert_eq!(serial.history.len(), parallel.history.len());
        for (left, right) in serial.history.iter().zip(&parallel.history) {
            assert_eq!(left.best_score, right.best_score);
            assert_eq!(left.mean_score, right.mean_score);
        }
    }
}
