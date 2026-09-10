//! Normalized circular distributions. Angles and densities use radians.
use crate::{EnergyError, Result};
use std::f64::consts::{PI, TAU};

#[derive(Debug, Clone, Copy)]
pub struct CircularComponent {
    pub mean: f64,
    pub concentration: f64,
    pub weight: f64,
}
#[derive(Debug, Clone)]
pub struct CircularMixture {
    components: Vec<CircularComponent>,
    log_constants: Vec<f64>,
}
fn invalid() -> EnergyError {
    EnergyError::InvalidConfiguration("invalid circular mixture".into())
}
/// Scaled integral avoids overflow even for sharply concentrated distributions.
pub fn log_i0(k: f64) -> f64 {
    if k == 0.0 {
        return 0.0;
    }
    if k > 50.0 {
        let t = 1.0 / k;
        return k - 0.5 * (TAU * k).ln()
            + (1.0
                + t / 8.0
                + 9.0 * t * t / 128.0
                + 225.0 * t * t * t / 3072.0
                + 11025.0 * t.powi(4) / 98304.0)
                .ln();
    }
    let mut term = 1.0;
    let mut sum = term;
    for i in 1..1000 {
        term *= (k * 0.5 / i as f64).powi(2);
        sum += term;
        if term < sum * 1e-16 {
            break;
        }
    }
    sum.ln()
}
impl CircularMixture {
    pub fn new(mut components: Vec<CircularComponent>) -> Result<Self> {
        if components.is_empty() {
            components.push(CircularComponent {
                mean: 0.,
                concentration: 0.,
                weight: 1.,
            });
        }
        if components.iter().any(|c| {
            !c.mean.is_finite()
                || !c.concentration.is_finite()
                || c.concentration < 0.
                || !c.weight.is_finite()
                || c.weight < 0.
        }) {
            return Err(invalid());
        }
        let sum: f64 = components.iter().map(|c| c.weight).sum();
        if !sum.is_finite() || sum <= 0. {
            return Err(invalid());
        }
        for c in &mut components {
            c.weight /= sum;
        }
        let log_constants = components
            .iter()
            .map(|c| c.weight.ln() - TAU.ln() - log_i0(c.concentration))
            .collect();
        Ok(Self {
            components,
            log_constants,
        })
    }
    pub fn components(&self) -> &[CircularComponent] {
        &self.components
    }
    pub fn log_probability(&self, angle: f64) -> f64 {
        self.log_probability_and_derivative(angle).0
    }
    pub fn log_probability_and_derivative(&self, angle: f64) -> (f64, f64) {
        let values: Vec<_> = self
            .components
            .iter()
            .zip(&self.log_constants)
            .map(|(c, l)| l + c.concentration * (angle - c.mean).cos())
            .collect();
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = values.iter().map(|v| (v - maximum).exp()).sum();
        let derivative = values
            .iter()
            .zip(&self.components)
            .map(|(v, c)| (v - maximum).exp() * (-c.concentration * (angle - c.mean).sin()))
            .sum::<f64>()
            / sum;
        (maximum + sum.ln(), derivative)
    }
}
/// Numerically integrated central interval for one von Mises component.
/// This is a circular probability interval, not a Gaussian sigma approximation.
pub fn credible_half_width(concentration: f64, probability: f64) -> Result<f64> {
    if !concentration.is_finite() || concentration < 0. || !(0.0..1.0).contains(&probability) {
        return Err(invalid());
    }
    if concentration < 1e-12 {
        return Ok(PI * probability);
    }
    let norm = TAU.ln() + log_i0(concentration);
    let integral = |end: f64| {
        let n = 1024usize;
        let h = end / n as f64;
        let mut sum = (concentration - norm).exp() + (concentration * end.cos() - norm).exp();
        for i in 1..n {
            sum += if i % 2 == 0 { 2. } else { 4. }
                * (concentration * (i as f64 * h).cos() - norm).exp();
        }
        2.0 * sum * h / 3.0
    };
    let mut lo = 0.;
    let mut hi = PI.min(12.0 / concentration.sqrt());
    for _ in 0..48 {
        let mid = (lo + hi) * 0.5;
        if integral(mid) < probability {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((lo + hi) * 0.5)
}
/// Log MH ratio includes proposal asymmetry. Rejections remain chain states.
pub fn log_acceptance(
    current_target: f64,
    proposed_target: f64,
    forward: f64,
    reverse: f64,
) -> f64 {
    (proposed_target - current_target + reverse - forward).min(0.0)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mixture_normalizes_and_derivative_matches_difference() {
        let m = CircularMixture::new(vec![
            CircularComponent {
                mean: 0.3,
                concentration: 2.,
                weight: 3.,
            },
            CircularComponent {
                mean: -2.,
                concentration: 20.,
                weight: 1.,
            },
        ])
        .unwrap();
        let n = 20000;
        let step = TAU / n as f64;
        let mass = (0..n)
            .map(|i| m.log_probability(-PI + (i as f64 + 0.5) * step).exp() * step)
            .sum::<f64>();
        assert!((mass - 1.).abs() < 1e-10);
        let (p, g) = m.log_probability_and_derivative(0.7);
        assert!((p - m.log_probability(0.7 + TAU)).abs() < 1e-12);
        assert!(
            (g - (m.log_probability(0.700001) - m.log_probability(0.699999)) / 2e-6).abs() < 1e-7
        );
    }
    #[test]
    fn credible_region_has_requested_mass() {
        for k in [0., 0.01, 1., 20., 1000.] {
            let width = credible_half_width(k, 0.95).unwrap();
            let n = 10000;
            let h = 2. * width / n as f64;
            let m = CircularMixture::new(vec![CircularComponent {
                mean: 0.,
                concentration: k,
                weight: 1.,
            }])
            .unwrap();
            let mass = (0..n)
                .map(|i| m.log_probability(-width + (i as f64 + 0.5) * h).exp() * h)
                .sum::<f64>();
            assert!((mass - 0.95).abs() < 1e-7, "{k} {mass}");
        }
    }
    #[test]
    fn proposal_ratio_cancels_prior_for_independence_moves() {
        assert!(
            log_acceptance(0.2f64.ln() - 3., 0.8f64.ln() - 4., 0.8f64.ln(), 0.2f64.ln()) + 1.
                < 1e-12
        );
        assert_eq!(
            log_acceptance(0.2f64.ln(), 0.8f64.ln(), 0.8f64.ln(), 0.2f64.ln()),
            0.
        );
    }
}
