//! Resumable L-BFGS. A driver can batch independent requests without sharing
//! candidate histories or accepting dependent line-search trials together.
use super::*;

#[derive(Clone)]
enum Phase {
    Initial,
    Trial {
        direction: Vec<f64>,
        slope: f64,
        step: f64,
        candidate: Vec<f64>,
    },
    Done {
        converged: bool,
    },
}
/// Clone this state at a committed evaluation boundary to retain a CPU fallback
/// checkpoint. It contains no GPU objects, callbacks, or borrowed coordinates.
#[derive(Clone)]
pub struct LbfgsState {
    config: LbfgsConfig,
    started: Instant,
    phase: Phase,
    point: Vec<f64>,
    gradient: Vec<f64>,
    value: f64,
    iteration: usize,
    values: Vec<f64>,
    s: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    rho: Vec<f64>,
}
impl LbfgsState {
    pub fn new(initial: &[f64], config: &LbfgsConfig) -> Result<Self> {
        if config.max_iterations == 0
            || config.history_size == 0
            || !config.gradient_tolerance.is_finite()
            || config.gradient_tolerance <= 0.
            || !config.initial_step.is_finite()
            || config.initial_step <= 0.
            || !config.armijo.is_finite()
            || !(0.0..1.0).contains(&config.armijo)
            || config
                .time_limit_seconds
                .is_some_and(|x| !x.is_finite() || x <= 0.)
        {
            return Err(OptimizationError::InvalidConfiguration(
                "positive iteration, history, tolerance, step and valid Armijo values are required"
                    .into(),
            ));
        }
        if initial.iter().any(|x| !x.is_finite()) {
            return Err(OptimizationError::NonFiniteObjective);
        }
        Ok(Self {
            config: config.clone(),
            started: Instant::now(),
            phase: Phase::Initial,
            point: initial.to_vec(),
            gradient: vec![0.; initial.len()],
            value: f64::NAN,
            iteration: 0,
            values: Vec::new(),
            s: Vec::new(),
            y: Vec::new(),
            rho: Vec::new(),
        })
    }
    fn expired(&self) -> bool {
        self.config
            .time_limit_seconds
            .is_some_and(|t| self.started.elapsed().as_secs_f64() >= t)
    }
    /// At most one outstanding evaluation per candidate. Repeated calls return
    /// the same request, allowing a failed GPU dispatch to be retried on CPU.
    pub fn request(&mut self) -> Option<&[f64]> {
        if !matches!(self.phase, Phase::Initial | Phase::Done { .. }) && self.expired() {
            self.phase = Phase::Done { converged: false };
        }
        match &self.phase {
            Phase::Initial => Some(&self.point),
            Phase::Trial { candidate, .. } => Some(candidate),
            Phase::Done { .. } => None,
        }
    }
    /// Commit a completed evaluation. Returns progress only for an initial or
    /// accepted point. Rejected Armijo trials keep the committed point intact.
    pub fn submit(&mut self, value: f64, gradient: Vec<f64>) -> Result<Option<LbfgsProgress>> {
        if gradient.len() != self.point.len() {
            return Err(OptimizationError::DimensionMismatch {
                expected: self.point.len(),
                received: gradient.len(),
            });
        }
        if matches!(self.phase, Phase::Done { .. }) {
            return Err(OptimizationError::InvalidConfiguration(
                "completed optimizer has no pending evaluation".into(),
            ));
        }
        match &mut self.phase {
            Phase::Initial => {
                if !value.is_finite() || gradient.iter().any(|x| !x.is_finite()) {
                    return Err(OptimizationError::NonFiniteObjective);
                }
                self.value = value;
                self.gradient = gradient;
                self.values.push(value);
            }
            Phase::Trial {
                direction,
                slope,
                step,
                candidate,
            } => {
                if !value.is_finite()
                    || gradient.iter().any(|x| !x.is_finite())
                    || value > self.value + self.config.armijo * (*step) * (*slope)
                {
                    *step *= 0.5;
                    if *step < 1e-12 {
                        self.phase = Phase::Done { converged: false };
                    } else {
                        for ((c, p), d) in candidate.iter_mut().zip(&self.point).zip(direction) {
                            *c = p + *step * *d;
                        }
                    }
                    return Ok(None);
                }
                let s = candidate
                    .iter()
                    .zip(&self.point)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>();
                let y = gradient
                    .iter()
                    .zip(&self.gradient)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>();
                let curvature = dot(&s, &y);
                if curvature > 1e-12 {
                    if self.s.len() == self.config.history_size {
                        self.s.remove(0);
                        self.y.remove(0);
                        self.rho.remove(0);
                    }
                    self.s.push(s);
                    self.y.push(y);
                    self.rho.push(1. / curvature);
                }
                self.point = std::mem::take(candidate);
                self.gradient = gradient;
                self.value = value;
                self.values.push(value);
                self.iteration += 1;
            }
            Phase::Done { .. } => {
                return Err(OptimizationError::InvalidConfiguration(
                    "completed optimizer has no pending evaluation".into(),
                ));
            }
        }
        let progress = LbfgsProgress {
            iteration: self.iteration,
            value: self.value,
            rms_gradient: rms_norm(&self.gradient),
            max_gradient: infinity_norm(&self.gradient),
            accepted_steps: self.values.len() - 1,
        };
        self.advance();
        Ok(Some(progress))
    }
    fn advance(&mut self) {
        if self.iteration == self.config.max_iterations || self.expired() {
            self.phase = Phase::Done { converged: false };
            return;
        }
        if infinity_norm(&self.gradient) <= self.config.gradient_tolerance {
            self.phase = Phase::Done { converged: true };
            return;
        }
        let mut direction = lbfgs_direction(&self.gradient, &self.s, &self.y, &self.rho);
        if dot(&self.gradient, &direction) >= 0. {
            direction = self.gradient.iter().map(|x| -x).collect();
        }
        let slope = dot(&self.gradient, &direction);
        let step = self.config.initial_step;
        let candidate = self
            .point
            .iter()
            .zip(&direction)
            .map(|(p, d)| p + step * d)
            .collect();
        self.phase = Phase::Trial {
            direction,
            slope,
            step,
            candidate,
        };
    }
    pub fn outcome(&self) -> Option<LbfgsOutcome> {
        if let Phase::Done { converged } = self.phase {
            Some(LbfgsOutcome {
                point: self.point.clone(),
                value: self.value,
                iterations: self.iteration,
                converged,
                history: self.values.clone(),
            })
        } else {
            None
        }
    }
}

/// Driver supplies ordered (value, gradient) results. CPU and GPU drivers can
/// reuse exactly the same state transitions. The future need not be Send,
/// keeping browser GPU handles on the coordinating worker.
pub async fn minimize_batch<F, Fut, C>(
    states: &mut [LbfgsState],
    mut evaluate: F,
    mut cancelled: C,
) -> Result<Vec<LbfgsOutcome>>
where
    F: FnMut(Vec<Vec<f64>>) -> Fut,
    Fut: std::future::Future<Output = Result<Vec<(f64, Vec<f64>)>>>,
    C: FnMut() -> bool,
{
    loop {
        if cancelled() {
            return Err(OptimizationError::Cancelled);
        }
        let mut indices = Vec::new();
        let mut requests = Vec::new();
        for (index, state) in states.iter_mut().enumerate() {
            if let Some(point) = state.request() {
                indices.push(index);
                requests.push(point.to_vec());
            }
        }
        if requests.is_empty() {
            return states
                .iter()
                .map(|s| {
                    s.outcome().ok_or_else(|| {
                        OptimizationError::InvalidConfiguration(
                            "optimizer stopped without a completed state".into(),
                        )
                    })
                })
                .collect();
        }
        let values = evaluate(requests).await?;
        if cancelled() {
            return Err(OptimizationError::Cancelled);
        }
        if values.len() != indices.len() {
            return Err(OptimizationError::DimensionMismatch {
                expected: indices.len(),
                received: values.len(),
            });
        }
        // Validate the entire result shape before committing any candidate.
        for (&index, (value, gradient)) in indices.iter().zip(&values) {
            if matches!(states[index].phase, Phase::Initial)
                && (!value.is_finite() || gradient.iter().any(|x| !x.is_finite()))
            {
                return Err(OptimizationError::NonFiniteObjective);
            }
            if gradient.len() != states[index].point.len() {
                return Err(OptimizationError::DimensionMismatch {
                    expected: states[index].point.len(),
                    received: gradient.len(),
                });
            }
        }
        for (index, (value, gradient)) in indices.into_iter().zip(values) {
            states[index].submit(value, gradient)?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_line_searches_and_checkpoint_replay() {
        let config = LbfgsConfig {
            initial_step: 1.,
            max_iterations: 100,
            ..Default::default()
        };
        let mut states = [
            LbfgsState::new(&[2.], &config).unwrap(),
            LbfgsState::new(&[3.], &config).unwrap(),
        ];
        let curvature = [1., 100.];
        let mut requests = [0; 2];
        while states.iter().any(|s| s.outcome().is_none()) {
            for (i, s) in states.iter_mut().enumerate() {
                if let Some(point) = s.request() {
                    let x = point[0];
                    let checkpoint = s.clone();
                    requests[i] += 1;
                    let value = 0.5 * curvature[i] * x * x;
                    let gradient = vec![curvature[i] * x];
                    s.submit(value, gradient.clone()).unwrap();
                    // A failed dispatch can repeat the same pending request on CPU.
                    let mut replay = checkpoint;
                    replay.submit(value, gradient).unwrap();
                    assert_eq!(s.request(), replay.request());
                    assert_eq!(s.values, replay.values);
                }
            }
        }
        assert!(requests[1] > requests[0]);
        for s in states {
            let outcome = s.outcome().unwrap();
            assert!(outcome.converged);
            assert!(outcome.point[0].abs() < 1e-4);
            assert!(outcome.history.windows(2).all(|p| p[1] <= p[0]));
        }
    }
    #[test]
    fn rejected_trial_keeps_committed_coordinates() {
        let mut state = LbfgsState::new(
            &[1.],
            &LbfgsConfig {
                initial_step: 1.,
                ..Default::default()
            },
        )
        .unwrap();
        state.submit(1., vec![2.]).unwrap();
        let checkpoint = state.point.clone();
        assert!(
            state
                .submit(f64::INFINITY, vec![f64::NAN])
                .unwrap()
                .is_none()
        );
        assert_eq!(state.point, checkpoint);
        assert_eq!(state.iteration, 0);
        assert_eq!(state.request(), Some([0.].as_slice()));
    }
}
