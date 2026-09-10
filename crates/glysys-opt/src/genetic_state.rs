//! Generation checkpoints shared by synchronous CPU and asynchronous batch drivers.
use super::*;
#[derive(Clone)]
pub struct GeneticState<S> {
    config: GeneticAlgorithmConfig,
    population: Vec<S>,
    generation: usize,
    history: Vec<GenerationRecord>,
    outcome: Option<GeneticAlgorithmOutcome<S>>,
}
impl<S: Clone + Send + Sync> GeneticState<S> {
    pub fn new<P: GeneticProblem<State = S>>(
        problem: &P,
        config: &GeneticAlgorithmConfig,
    ) -> Result<Self> {
        validate_genetic_config(config)?;
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        let population = (0..config.population_size)
            .map(|_| problem.generate(&mut rng))
            .collect();
        Ok(Self {
            config: config.clone(),
            population,
            generation: 0,
            history: Vec::with_capacity(config.generations + 1),
            outcome: None,
        })
    }
    /// No RNG is advanced by requesting or retrying this batch.
    pub fn population(&self) -> &[S] {
        &self.population
    }
    pub fn outcome(&self) -> Option<&GeneticAlgorithmOutcome<S>> {
        self.outcome.as_ref()
    }
    pub fn submit<P, F, C>(
        &mut self,
        problem: &P,
        scores: Vec<f64>,
        mut progress: F,
        mut cancelled: C,
    ) -> Result<()>
    where
        P: GeneticProblem<State = S>,
        F: FnMut(&GenerationRecord),
        C: FnMut() -> bool,
    {
        if self.outcome.is_some() {
            return Err(OptimizationError::InvalidConfiguration(
                "genetic search is complete".into(),
            ));
        }
        if scores.len() != self.population.len() {
            return Err(OptimizationError::DimensionMismatch {
                expected: self.population.len(),
                received: scores.len(),
            });
        }
        if scores.iter().any(|s| !s.is_finite()) {
            return Err(OptimizationError::NonFiniteObjective);
        }
        if cancelled() {
            return Err(OptimizationError::Cancelled);
        }
        let mut scored = scores
            .into_iter()
            .zip(self.population.iter().cloned())
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        let record = GenerationRecord {
            generation: self.generation,
            best_score: scored[0].0,
            mean_score: scored.iter().map(|s| s.0).sum::<f64>() / scored.len() as f64,
        };
        progress(&record);
        if problem.is_solution(&scored[0].1, scored[0].0)
            || self.generation == self.config.generations
        {
            self.history.push(record);
            self.outcome = Some(GeneticAlgorithmOutcome {
                best_state: scored[0].1.clone(),
                best_score: scored[0].0,
                generations: self.generation,
                history: self.history.clone(),
            });
            return Ok(());
        }
        if cancelled() {
            return Err(OptimizationError::Cancelled);
        }
        let elite = ((self.config.population_size as f64 * self.config.elite_fraction).round()
            as usize)
            .clamp(1, self.config.population_size);
        let seed = splitmix64(self.config.seed ^ self.generation as u64);
        let mut next = scored
            .iter()
            .take(elite)
            .map(|s| s.1.clone())
            .collect::<Vec<_>>();
        let children = (0..self.config.population_size - next.len())
            .into_par_iter()
            .map(|index| {
                let mut rng = ChaCha8Rng::seed_from_u64(splitmix64(seed ^ index as u64));
                let first = tournament(&scored, self.config.tournament_size, &mut rng);
                let second = tournament(&scored, self.config.tournament_size, &mut rng);
                let mut child = problem.crossover(first, second, &mut rng);
                problem.mutate(&mut child, &mut rng, self.config.mutation_rate);
                problem.repair(&mut child, &mut rng);
                child
            })
            .collect::<Vec<_>>();
        next.extend(children);
        self.population = next;
        self.history.push(record);
        self.generation += 1;
        Ok(())
    }
}
/// Async evaluations are owned by the caller's coordinating task. Parallel CPU
/// generation prepares states but never sees a GPU device or queue.
pub async fn optimize_batched<P, E, Fut, F, C>(
    problem: &P,
    state: &mut GeneticState<P::State>,
    mut evaluate: E,
    mut progress: F,
    mut cancelled: C,
) -> Result<GeneticAlgorithmOutcome<P::State>>
where
    P: GeneticProblem,
    E: FnMut(Vec<P::State>) -> Fut,
    Fut: std::future::Future<Output = Result<Vec<f64>>>,
    F: FnMut(&GenerationRecord),
    C: FnMut() -> bool,
{
    loop {
        if cancelled() {
            return Err(OptimizationError::Cancelled);
        }
        if let Some(outcome) = state.outcome() {
            return Ok(outcome.clone());
        }
        let scores = evaluate(state.population().to_vec()).await?;
        state.submit(problem, scores, &mut progress, &mut cancelled)?;
    }
}
