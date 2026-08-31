//! Deterministic, application-independent optimization algorithms.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

pub type Result<T> = std::result::Result<T, OptimizationError>;

#[derive(Debug, thiserror::Error)]
pub enum OptimizationError {
    #[error("optimization was cancelled")]
    Cancelled,
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
    genetic_optimize_with_progress(problem, config, |_| {})
}

/// Run the deterministic genetic algorithm and report each completed
/// generation from the coordinating thread.
pub fn genetic_optimize_with_progress<P, F>(
    problem: &P,
    config: &GeneticAlgorithmConfig,
    progress: F,
) -> Result<GeneticAlgorithmOutcome<P::State>>
where
    P: GeneticProblem,
    F: FnMut(&GenerationRecord),
{
    genetic_optimize_with_progress_cancelled(problem, config, progress, || false)
}

/// Run the deterministic genetic algorithm with a cooperative cancellation
/// callback.  The callback is checked between generations and before child
/// generation; objective evaluation remains parallel, so a cancellation
/// request that arrives during one batch is observed as soon as that batch
/// completes.  The original `genetic_optimize_with_progress` API delegates to
/// this function with a callback that never cancels.
pub fn genetic_optimize_with_progress_cancelled<P, F, C>(
    problem: &P,
    config: &GeneticAlgorithmConfig,
    mut progress: F,
    mut cancelled: C,
) -> Result<GeneticAlgorithmOutcome<P::State>>
where
    P: GeneticProblem,
    F: FnMut(&GenerationRecord),
    C: FnMut() -> bool,
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
        if cancelled() {
            return Err(OptimizationError::Cancelled);
        }
        let mut scored = population
            .par_iter()
            .map(|state| (problem.evaluate(state), state.clone()))
            .collect::<Vec<_>>();
        if cancelled() {
            return Err(OptimizationError::Cancelled);
        }
        if scored.iter().any(|(score, _)| !score.is_finite()) {
            return Err(OptimizationError::NonFiniteObjective);
        }
        scored.sort_by(|left, right| left.0.total_cmp(&right.0));
        let record = GenerationRecord {
            generation,
            best_score: scored[0].0,
            mean_score: scored.iter().map(|entry| entry.0).sum::<f64>() / scored.len() as f64,
        };
        progress(&record);
        history.push(record);
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

        if cancelled() {
            return Err(OptimizationError::Cancelled);
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
    /// Optional wall-clock ceiling for one L-BFGS invocation.  The check is
    /// made between objective evaluations/iterations so callers can reserve
    /// time for later phases without making the objective implementation
    /// aware of a deadline.
    #[serde(default)]
    pub time_limit_seconds: Option<f64>,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            history_size: 10,
            gradient_tolerance: 1.0e-4,
            initial_step: 0.1,
            armijo: 1.0e-4,
            time_limit_seconds: None,
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

/// A completed L-BFGS evaluation made available to callers that want to show
/// progress without coupling the optimizer to a user interface.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LbfgsProgress {
    /// Zero denotes the initial objective evaluation.
    pub iteration: usize,
    pub value: f64,
    pub rms_gradient: f64,
    pub max_gradient: f64,
    pub accepted_steps: usize,
}

pub fn lbfgs_minimize<O: DifferentiableObjective>(
    objective: &mut O,
    initial: &[f64],
    config: &LbfgsConfig,
) -> Result<LbfgsOutcome> {
    lbfgs_minimize_with_progress(objective, initial, config, |_| {})
}

/// Minimize a differentiable objective while reporting the initial point and
/// every accepted step. The callback is deliberately synchronous so command
/// line clients can write timely progress messages and applications can
/// capture deterministic diagnostics.
pub fn lbfgs_minimize_with_progress<O, F>(
    objective: &mut O,
    initial: &[f64],
    config: &LbfgsConfig,
    mut progress: F,
) -> Result<LbfgsOutcome>
where
    O: DifferentiableObjective,
    F: FnMut(LbfgsProgress),
{
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
        || config
            .time_limit_seconds
            .is_some_and(|limit| !limit.is_finite() || limit <= 0.0)
    {
        return Err(OptimizationError::InvalidConfiguration(
            "positive iteration, history, tolerance, and step values are required".into(),
        ));
    }
    let started = Instant::now();
    let mut point = initial.to_vec();
    let mut gradient = vec![0.0; point.len()];
    let mut value = objective.value_gradient(&point, &mut gradient)?;
    let mut values = vec![value];
    progress(LbfgsProgress {
        iteration: 0,
        value,
        rms_gradient: rms_norm(&gradient),
        max_gradient: infinity_norm(&gradient),
        accepted_steps: 0,
    });
    let mut s_history: Vec<Vec<f64>> = Vec::new();
    let mut y_history: Vec<Vec<f64>> = Vec::new();
    let mut rho_history: Vec<f64> = Vec::new();

    for iteration in 0..config.max_iterations {
        if config
            .time_limit_seconds
            .is_some_and(|limit| started.elapsed().as_secs_f64() >= limit)
        {
            return Ok(LbfgsOutcome {
                point,
                value,
                iterations: iteration,
                converged: false,
                history: values,
            });
        }
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
        let mut line_search_stalled = false;
        let candidate_value = loop {
            if config
                .time_limit_seconds
                .is_some_and(|limit| started.elapsed().as_secs_f64() >= limit)
            {
                line_search_stalled = true;
                break value;
            }
            for index in 0..point.len() {
                candidate[index] = point[index] + step * direction[index];
            }
            let trial = objective.value_gradient(&candidate, &mut candidate_gradient)?;
            if trial.is_finite() && trial <= value + config.armijo * step * slope {
                break trial;
            }
            step *= 0.5;
            if step < 1.0e-12 {
                line_search_stalled = true;
                break value;
            }
        };
        // A failed Armijo search cannot produce a new point.  Continuing to
        // iterate the same coordinates only repeats expensive force-field
        // evaluations (particularly visible in large glycoproteins), so stop
        // and return the best point found so far.
        if line_search_stalled {
            return Ok(LbfgsOutcome {
                point,
                value,
                iterations: iteration,
                converged: false,
                history: values,
            });
        }
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
        progress(LbfgsProgress {
            iteration: iteration + 1,
            value,
            rms_gradient: rms_norm(&gradient),
            max_gradient: infinity_norm(&gradient),
            accepted_steps: values.len().saturating_sub(1),
        });
    }
    Ok(LbfgsOutcome {
        point,
        value,
        iterations: config.max_iterations,
        converged: false,
        history: values,
    })
}

/// A particle seed with immutable application context.
///
/// Density fitting uses the context to retain the GlycoShape conformer
/// template while the particle position contains only periodic torsion
/// coordinates. Keeping this context outside the numeric vector avoids
/// treating a categorical conformer identifier as a continuous variable.
#[derive(Debug, Clone)]
pub struct ParticleSeed<C> {
    pub context: C,
    pub position: Vec<f64>,
}

/// Operations required by the deterministic particle-swarm optimizer.
pub trait ParticleSwarmProblem: Sync {
    type Context: Clone + Send + Sync;

    /// Evaluate a particle. Larger finite values are better.
    fn evaluate(&self, context: &Self::Context, position: &[f64]) -> f64;

    /// Repair a position and velocity after the PSO update. The default
    /// implementation leaves both vectors unchanged; callers use this hook
    /// for angular wrapping and application-specific bounds.
    fn repair(&self, _context: &Self::Context, _position: &mut [f64], _velocity: &mut [f64]) {}
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ParticleSwarmConfig {
    pub swarms: usize,
    pub particles_per_swarm: usize,
    pub generations: usize,
    pub inertia_start: f64,
    pub inertia_end: f64,
    pub cognitive: f64,
    pub social: f64,
    pub migration_interval: usize,
    pub migration_count: usize,
    pub stall_generations: usize,
    pub tolerance: f64,
    pub seed: u64,
    pub max_evaluations: Option<usize>,
    /// Inclusive numeric bounds for each dimension. Periodic dimensions are
    /// wrapped into this interval after every update.
    pub bounds: Vec<(f64, f64)>,
    pub periodic_dimensions: Vec<bool>,
    /// Optional wall-clock ceiling for this phase.  The optimizer checks at
    /// generation boundaries so a complete synchronous generation remains
    /// deterministic while a caller can reserve time for later phases.
    #[serde(default)]
    pub time_limit_seconds: Option<f64>,
}

impl Default for ParticleSwarmConfig {
    fn default() -> Self {
        Self {
            swarms: 1,
            particles_per_swarm: 32,
            generations: 24,
            inertia_start: 0.8,
            inertia_end: 0.4,
            cognitive: 1.6,
            social: 1.6,
            migration_interval: 4,
            migration_count: 2,
            stall_generations: 6,
            tolerance: 0.002,
            seed: 0,
            max_evaluations: None,
            bounds: Vec::new(),
            periodic_dimensions: Vec::new(),
            time_limit_seconds: None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParticleSwarmGeneration {
    pub generation: usize,
    pub best_score: f64,
    pub mean_score: f64,
    pub evaluations: usize,
    pub migrations: usize,
}

#[derive(Debug, Clone)]
pub struct ParticleSwarmOutcome<C> {
    pub best_context: C,
    pub best_position: Vec<f64>,
    pub best_score: f64,
    pub generations: usize,
    pub evaluations: usize,
    pub converged: bool,
    pub history: Vec<ParticleSwarmGeneration>,
    pub particles: Vec<ParticleSeed<C>>,
    pub particle_scores: Vec<f64>,
}

#[derive(Debug, Clone)]
struct SwarmParticle<C> {
    context: C,
    position: Vec<f64>,
    velocity: Vec<f64>,
    best_context: C,
    best_position: Vec<f64>,
    best_score: f64,
    score: f64,
}

/// Run deterministic, synchronous particle-swarm optimization.
///
/// Particles are evaluated in parallel, but all updates happen in stable
/// index order after the complete generation has been scored. Random values
/// are generated from per-particle ChaCha streams, so Rayon thread count does
/// not change the result. The objective is maximized.
pub fn particle_swarm_optimize<P>(
    problem: &P,
    seeds: Vec<ParticleSeed<P::Context>>,
    config: &ParticleSwarmConfig,
) -> Result<ParticleSwarmOutcome<P::Context>>
where
    P: ParticleSwarmProblem,
{
    particle_swarm_optimize_with_progress(problem, seeds, config, |_| {})
}

/// Progress-reporting variant of [`particle_swarm_optimize`].
pub fn particle_swarm_optimize_with_progress<P, F>(
    problem: &P,
    seeds: Vec<ParticleSeed<P::Context>>,
    config: &ParticleSwarmConfig,
    mut progress: F,
) -> Result<ParticleSwarmOutcome<P::Context>>
where
    P: ParticleSwarmProblem,
    F: FnMut(&ParticleSwarmGeneration),
{
    validate_particle_swarm_config(config, &seeds)?;
    let dimension = config.bounds.len();
    let total_particles = config.swarms * config.particles_per_swarm;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let started = Instant::now();
    let pending = seeds
        .into_iter()
        .enumerate()
        .map(|(_index, seed)| {
            let mut position = seed.position;
            let mut velocity = vec![0.0; dimension];
            for (axis, ((lower, upper), velocity_axis)) in
                config.bounds.iter().zip(velocity.iter_mut()).enumerate()
            {
                let width = (upper - lower).abs().max(1.0e-12);
                *velocity_axis = (rng.random::<f64>() * 2.0 - 1.0) * width * 0.1;
                if config.periodic_dimensions[axis] {
                    position[axis] = wrap_swarm_value(position[axis], *lower, *upper);
                } else {
                    position[axis] = position[axis].clamp(*lower, *upper);
                }
            }
            let context = seed.context;
            problem.repair(&context, &mut position, &mut velocity);
            (context, position, velocity)
        })
        .collect::<Vec<_>>();
    // Initial conformer coverage is part of the deterministic contract, but
    // it must not serialize all templates behind one objective call.  Keep
    // the pending vector in stable index order and evaluate it synchronously
    // through Rayon; the resulting particle order is therefore identical for
    // one or many worker threads.
    let initial_scores = pending
        .par_iter()
        .map(|(context, position, _)| problem.evaluate(context, position))
        .collect::<Vec<_>>();
    if initial_scores.iter().any(|score| !score.is_finite()) {
        return Err(OptimizationError::NonFiniteObjective);
    }
    let mut particles = pending
        .into_iter()
        .zip(initial_scores)
        .map(|((context, position, velocity), score)| SwarmParticle {
            context: context.clone(),
            position: position.clone(),
            velocity,
            best_context: context,
            best_position: position,
            best_score: score,
            score,
        })
        .collect::<Vec<_>>();
    while particles.len() < total_particles {
        let source = particles[particles.len() % particles.len()].clone();
        let mut position = source.best_position;
        let mut velocity = source.velocity;
        for (axis, ((lower, upper), velocity_axis)) in
            config.bounds.iter().zip(velocity.iter_mut()).enumerate()
        {
            let width = (upper - lower).abs().max(1.0e-12);
            position[axis] += (rng.random::<f64>() * 2.0 - 1.0) * width * 0.05;
            *velocity_axis = (rng.random::<f64>() * 2.0 - 1.0) * width * 0.05;
        }
        problem.repair(&source.context, &mut position, &mut velocity);
        let score = problem.evaluate(&source.context, &position);
        if !score.is_finite() {
            return Err(OptimizationError::NonFiniteObjective);
        }
        particles.push(SwarmParticle {
            context: source.context.clone(),
            position: position.clone(),
            velocity,
            best_context: source.context,
            best_position: position,
            best_score: score,
            score,
        });
    }

    let mut evaluations = particles.len();
    let mut history = Vec::with_capacity(config.generations + 1);
    let mut best_index = best_particle_index(&particles);
    let mut global_best_score = particles[best_index].best_score;
    let mut stagnant = 0usize;
    let mut migrations = 0usize;
    let mut completed_generations = 0usize;
    let mut converged = false;

    for generation in 0..=config.generations {
        let mean_score =
            particles.iter().map(|particle| particle.score).sum::<f64>() / particles.len() as f64;
        best_index = best_particle_index(&particles);
        let generation_best = particles[best_index].best_score;
        if generation_best > global_best_score + config.tolerance {
            global_best_score = generation_best;
            stagnant = 0;
        } else if generation > 0 {
            stagnant += 1;
        }
        let record = ParticleSwarmGeneration {
            generation,
            best_score: global_best_score,
            mean_score,
            evaluations,
            migrations,
        };
        progress(&record);
        history.push(record);
        completed_generations = generation;
        if stagnant >= config.stall_generations || generation == config.generations {
            converged = stagnant >= config.stall_generations;
            break;
        }
        if config.time_limit_seconds.is_some_and(|limit| {
            limit.is_finite() && limit > 0.0 && started.elapsed().as_secs_f64() >= limit
        }) {
            break;
        }
        if config
            .max_evaluations
            .is_some_and(|limit| evaluations + particles.len() > limit)
        {
            break;
        }

        let inertia = if config.generations == 0 {
            config.inertia_end
        } else {
            let fraction = generation as f64 / config.generations as f64;
            config.inertia_start + (config.inertia_end - config.inertia_start) * fraction
        };
        let swarm_bests = (0..config.swarms)
            .map(|swarm| {
                let start = swarm * config.particles_per_swarm;
                let end = start + config.particles_per_swarm;
                (start..end)
                    .max_by(|left, right| {
                        particles[*left]
                            .best_score
                            .total_cmp(&particles[*right].best_score)
                            .then_with(|| right.cmp(left))
                    })
                    .unwrap_or(start)
            })
            .collect::<Vec<_>>();
        let generation_seed = splitmix64(config.seed ^ (generation as u64 + 1));
        for index in 0..particles.len() {
            let swarm = (index / config.particles_per_swarm).min(config.swarms - 1);
            let local_best = swarm_bests[swarm];
            let global_best = best_particle_index(&particles);
            let mut local_rng =
                ChaCha8Rng::seed_from_u64(splitmix64(generation_seed ^ index as u64));
            for axis in 0..dimension {
                let r1 = local_rng.random::<f64>();
                let r2 = local_rng.random::<f64>();
                particles[index].velocity[axis] = inertia * particles[index].velocity[axis]
                    + config.cognitive
                        * r1
                        * (particles[index].best_position[axis] - particles[index].position[axis])
                    + config.social
                        * r2
                        * (particles[local_best].best_position[axis]
                            - particles[index].position[axis]);
                // A small global attraction is useful after swarm migration,
                // while retaining local exploration between migrations.
                if generation > 0 && generation % config.migration_interval.max(1) == 0 {
                    particles[index].velocity[axis] += 0.25
                        * r2
                        * (particles[global_best].best_position[axis]
                            - particles[index].position[axis]);
                }
                let width = (config.bounds[axis].1 - config.bounds[axis].0).abs();
                particles[index].velocity[axis] =
                    particles[index].velocity[axis].clamp(-width * 0.25, width * 0.25);
                particles[index].position[axis] += particles[index].velocity[axis];
                if config.periodic_dimensions[axis] {
                    particles[index].position[axis] = wrap_swarm_value(
                        particles[index].position[axis],
                        config.bounds[axis].0,
                        config.bounds[axis].1,
                    );
                } else {
                    particles[index].position[axis] = particles[index].position[axis]
                        .clamp(config.bounds[axis].0, config.bounds[axis].1);
                }
            }
            let context = particles[index].context.clone();
            let particle = &mut particles[index];
            problem.repair(&context, &mut particle.position, &mut particle.velocity);
        }
        let scored = particles
            .par_iter()
            .map(|particle| {
                let score = problem.evaluate(&particle.context, &particle.position);
                (score, score.is_finite())
            })
            .collect::<Vec<_>>();
        for (particle, (score, finite)) in particles.iter_mut().zip(scored) {
            if !finite {
                return Err(OptimizationError::NonFiniteObjective);
            }
            particle.score = score;
            if score > particle.best_score {
                particle.best_score = score;
                particle.best_position = particle.position.clone();
                particle.best_context = particle.context.clone();
            }
        }
        evaluations += particles.len();

        if config.time_limit_seconds.is_some_and(|limit| {
            limit.is_finite() && limit > 0.0 && started.elapsed().as_secs_f64() >= limit
        }) {
            break;
        }

        if config.migration_interval > 0
            && (generation + 1) % config.migration_interval == 0
            && config.swarms > 1
        {
            let bests = (0..config.swarms)
                .map(|swarm| {
                    let start = swarm * config.particles_per_swarm;
                    let end = start + config.particles_per_swarm;
                    (start..end)
                        .max_by(|left, right| {
                            particles[*left]
                                .best_score
                                .total_cmp(&particles[*right].best_score)
                                .then_with(|| right.cmp(left))
                        })
                        .unwrap_or(start)
                })
                .collect::<Vec<_>>();
            for swarm in 0..config.swarms {
                let destination = (swarm + 1) % config.swarms;
                let start = destination * config.particles_per_swarm;
                let source = bests[swarm];
                for offset in 0..config.migration_count.min(config.particles_per_swarm) {
                    let index = start + config.particles_per_swarm - 1 - offset;
                    particles[index].context = particles[source].best_context.clone();
                    particles[index].position = particles[source].best_position.clone();
                    particles[index].best_context = particles[source].best_context.clone();
                    particles[index].best_position = particles[source].best_position.clone();
                    particles[index].best_score = particles[source].best_score;
                    particles[index].score = particles[source].best_score;
                    particles[index].velocity.fill(0.0);
                }
            }
            migrations += config.swarms * config.migration_count.min(config.particles_per_swarm);
        }
    }

    best_index = best_particle_index(&particles);
    let best_context = particles[best_index].best_context.clone();
    let best_position = particles[best_index].best_position.clone();
    let best_score = particles[best_index].best_score;
    let particle_scores = particles.iter().map(|particle| particle.score).collect();
    let particle_seeds = particles
        .into_iter()
        .map(|particle| ParticleSeed {
            context: particle.best_context,
            position: particle.best_position,
        })
        .collect();
    Ok(ParticleSwarmOutcome {
        best_context,
        best_position,
        best_score,
        generations: completed_generations,
        evaluations,
        converged,
        history,
        particles: particle_seeds,
        particle_scores,
    })
}

fn validate_particle_swarm_config<C>(
    config: &ParticleSwarmConfig,
    seeds: &[ParticleSeed<C>],
) -> Result<()> {
    if config.swarms == 0
        || config.particles_per_swarm == 0
        || config.generations == 0
        || config.inertia_start.is_nan()
        || config.inertia_end.is_nan()
        || config.cognitive < 0.0
        || config.social < 0.0
        || config.stall_generations == 0
        || config.tolerance < 0.0
        || config
            .time_limit_seconds
            .is_some_and(|limit| !limit.is_finite() || limit <= 0.0)
        || config.bounds.is_empty()
        || config.periodic_dimensions.len() != config.bounds.len()
        || seeds.is_empty()
        || seeds.len() > config.swarms * config.particles_per_swarm
    {
        return Err(OptimizationError::InvalidConfiguration(
            "valid swarm count, population, generations, coefficients, bounds, and seeds are required"
                .into(),
        ));
    }
    if config
        .bounds
        .iter()
        .any(|(lower, upper)| !lower.is_finite() || !upper.is_finite() || upper <= lower)
    {
        return Err(OptimizationError::InvalidConfiguration(
            "swarm bounds must be finite and increasing".into(),
        ));
    }
    if seeds
        .iter()
        .any(|seed| seed.position.len() != config.bounds.len())
    {
        return Err(OptimizationError::DimensionMismatch {
            expected: config.bounds.len(),
            received: seeds
                .iter()
                .find(|seed| seed.position.len() != config.bounds.len())
                .map_or(0, |seed| seed.position.len()),
        });
    }
    Ok(())
}

fn best_particle_index<C>(particles: &[SwarmParticle<C>]) -> usize {
    particles
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.best_score
                .total_cmp(&right.best_score)
                .then_with(|| right_index.cmp(left_index))
        })
        .map_or(0, |(index, _)| index)
}

fn wrap_swarm_value(value: f64, lower: f64, upper: f64) -> f64 {
    let width = upper - lower;
    lower + (value - lower).rem_euclid(width)
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

fn rms_norm(values: &[f64]) -> f64 {
    (values.iter().map(|value| value * value).sum::<f64>() / values.len().max(1) as f64).sqrt()
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

    #[test]
    fn lbfgs_reports_initial_and_accepted_steps() {
        let mut progress = Vec::new();
        let outcome = lbfgs_minimize_with_progress(
            &mut Quadratic,
            &[8.0, 3.0],
            &LbfgsConfig::default(),
            |record| progress.push(record),
        )
        .unwrap();
        assert_eq!(progress.first().unwrap().iteration, 0);
        assert_eq!(progress.len(), outcome.history.len());
        assert!(progress.iter().all(|record| record.value.is_finite()));
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

    #[derive(Debug)]
    struct SwarmTarget;

    impl ParticleSwarmProblem for SwarmTarget {
        type Context = usize;

        fn evaluate(&self, context: &Self::Context, position: &[f64]) -> f64 {
            // The categorical context participates in the objective but is
            // immutable while particles move, which catches accidental
            // context interpolation or migration aliasing.
            -((position[0] - (*context as f64) * 0.1).powi(2) + position[1].sin().powi(2))
        }
    }

    #[test]
    fn particle_swarm_is_periodic_and_thread_deterministic() {
        let seeds = (0..8)
            .map(|context| ParticleSeed {
                context,
                position: vec![170.0 - context as f64, 2.0],
            })
            .collect::<Vec<_>>();
        let config = ParticleSwarmConfig {
            swarms: 2,
            particles_per_swarm: 4,
            generations: 12,
            seed: 91,
            bounds: vec![(-180.0, 180.0), (-180.0, 180.0)],
            periodic_dimensions: vec![true, true],
            ..ParticleSwarmConfig::default()
        };
        let run = |threads| {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| particle_swarm_optimize(&SwarmTarget, seeds.clone(), &config).unwrap())
        };
        let serial = run(1);
        let parallel = run(4);
        assert_eq!(serial.best_context, parallel.best_context);
        assert_eq!(serial.best_position, parallel.best_position);
        assert_eq!(serial.history.len(), parallel.history.len());
        assert!(
            serial
                .best_position
                .iter()
                .all(|value| (-180.0..180.0).contains(value))
        );
    }
}
