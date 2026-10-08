//! Smooth particle-mesh Ewald electrostatics for orthorhombic boxes
//! (Essmann et al., J. Chem. Phys. 103, 8577 (1995)).
//!
//! The Ewald split with coefficient `alpha` writes the periodic Coulomb
//! energy as a short-range pair sum of `qq erfc(alpha r)/r`, evaluated by the
//! pair code ([`crate::pbc::PmeBackend`] on the f64 path, the cluster engine
//! in PME mode for dynamics), plus the long-range remainder computed here by
//! [`PmeEngine`]: the reciprocal-space sum on a mesh, the self energy, the
//! correction for excluded pairs, and the uniform background energy of a box
//! with a net charge.
//!
//! Charges are spread on the mesh with cardinal B-splines of order 4 to 6,
//! transformed with three passes of 1D FFTs (the first one real-to-complex,
//! so only half the spectrum is kept), multiplied by the influence function
//! and transformed back; forces come from the analytic derivative of the
//! splines, so they are the exact gradient of the mesh energy. All arithmetic
//! is f64. Every parallel loop reduces in a fixed order that does not depend
//! on how rayon splits the work, so results are reproducible bit for bit for
//! any thread count.
use crate::pbc::{BoxVectors, COULOMB, PmeBackend};
use crate::{EnergyError, Result};
use glysys::{ParameterizedSystem, Vec3};
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::f64::consts::PI;
use std::sync::Arc;

/// GROMACS `ewald-rtol` default.
pub const DEFAULT_EWALD_RTOL: f64 = 1.0e-5;
/// GROMACS `fourierspacing` default (0.12 nm).
pub const DEFAULT_FOURIER_SPACING_ANGSTROM: f64 = 1.2;
/// GROMACS `pme-order` default; OpenMM uses 5.
pub const DEFAULT_INTERPOLATION_ORDER: usize = 4;

const MIN_ORDER: usize = 4;
const MAX_ORDER: usize = 6;
const TASKS_PER_THREAD: usize = 4;
const ATOM_CHUNK: usize = 256;
const EXCLUDED_CHUNK: usize = 512;
const TWO_OVER_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

// W. J. Cody's rational Chebyshev approximations for erf and erfc (Math.
// Comp. 23, 631 (1969)); coefficients as printed in the netlib SPECFUN
// `CALERF`, hence the digits beyond f64.
#[allow(clippy::excessive_precision)]
const ERF_A: [f64; 5] = [
    3.161_123_743_870_565_60e00,
    1.138_641_541_510_501_56e02,
    3.774_852_376_853_020_21e02,
    3.209_377_589_138_469_47e03,
    1.857_777_061_846_031_53e-1,
];
#[allow(clippy::excessive_precision)]
const ERF_B: [f64; 4] = [
    2.360_129_095_234_412_09e01,
    2.440_246_379_344_441_73e02,
    1.282_616_526_077_372_28e03,
    2.844_236_833_439_170_62e03,
];
#[allow(clippy::excessive_precision)]
const ERFC_C: [f64; 9] = [
    5.641_884_969_886_700_89e-1,
    8.883_149_794_388_375_94e00,
    6.611_919_063_714_162_95e01,
    2.986_351_381_974_001_31e02,
    8.819_522_212_417_690_90e02,
    1.712_047_612_634_070_58e03,
    2.051_078_377_826_071_47e03,
    1.230_339_354_797_997_25e03,
    2.153_115_354_744_038_46e-8,
];
#[allow(clippy::excessive_precision)]
const ERFC_D: [f64; 8] = [
    1.574_492_611_070_983_47e01,
    1.176_939_508_913_124_99e02,
    5.371_811_018_620_098_58e02,
    1.621_389_574_566_690_19e03,
    3.290_799_235_733_459_63e03,
    4.362_619_090_143_247_16e03,
    3.439_367_674_143_721_64e03,
    1.230_339_354_803_749_42e03,
];
#[allow(clippy::excessive_precision)]
const ERFC_P: [f64; 6] = [
    3.053_266_349_612_323_44e-1,
    3.603_448_999_498_044_39e-1,
    1.257_817_261_112_292_46e-1,
    1.608_378_514_874_227_66e-2,
    6.587_491_615_298_378_03e-4,
    1.631_538_713_730_209_78e-2,
];
#[allow(clippy::excessive_precision)]
const ERFC_Q: [f64; 5] = [
    2.568_520_192_289_822_42e00,
    1.872_952_849_923_460_47e00,
    5.279_051_029_514_284_12e-1,
    6.051_834_131_244_131_91e-2,
    2.335_204_976_268_691_85e-3,
];
const ERF_SPLIT: f64 = 0.468_75;
const ERFC_UNDERFLOW: f64 = 26.543;

/// `erf(x)` for `|x| <= 0.46875`.
fn erf_core(x: f64) -> f64 {
    let y = x.abs();
    let ysq = if y > 1.11e-16 { y * y } else { 0. };
    let mut numerator = ERF_A[4] * ysq;
    let mut denominator = ysq;
    for i in 0..3 {
        numerator = (numerator + ERF_A[i]) * ysq;
        denominator = (denominator + ERF_B[i]) * ysq;
    }
    x * (numerator + ERF_A[3]) / (denominator + ERF_B[3])
}

/// `erfc(y)` for `y > 0.46875`.
fn erfc_tail(y: f64) -> f64 {
    if y >= ERFC_UNDERFLOW {
        return 0.;
    }
    let rational = if y <= 4. {
        let mut numerator = ERFC_C[8] * y;
        let mut denominator = y;
        for i in 0..7 {
            numerator = (numerator + ERFC_C[i]) * y;
            denominator = (denominator + ERFC_D[i]) * y;
        }
        (numerator + ERFC_C[7]) / (denominator + ERFC_D[7])
    } else {
        let ysq = 1. / (y * y);
        let mut numerator = ERFC_P[5] * ysq;
        let mut denominator = ysq;
        for i in 0..4 {
            numerator = (numerator + ERFC_P[i]) * ysq;
            denominator = (denominator + ERFC_Q[i]) * ysq;
        }
        let tail = ysq * (numerator + ERFC_P[4]) / (denominator + ERFC_Q[4]);
        (0.5 * TWO_OVER_SQRT_PI - tail) / y
    };
    // exp(-y^2) in two factors so the rounding of y^2 does not cost digits.
    let coarse = (y * 16.).trunc() / 16.;
    let delta = (y - coarse) * (y + coarse);
    (-coarse * coarse).exp() * (-delta).exp() * rational
}

/// Error function, accurate to about 1e-16 relative.
pub fn erf(x: f64) -> f64 {
    if x.abs() <= ERF_SPLIT {
        return erf_core(x);
    }
    let tail = erfc_tail(x.abs());
    if x < 0. {
        (tail - 0.5) - 0.5
    } else {
        (0.5 - tail) + 0.5
    }
}

/// Complementary error function, accurate to about 1e-16 relative for
/// positive arguments up to the underflow limit (x = 26.5).
pub fn erfc(x: f64) -> f64 {
    if x.abs() <= ERF_SPLIT {
        return 1. - erf_core(x);
    }
    let tail = erfc_tail(x.abs());
    if x < 0. { 2. - tail } else { tail }
}

/// `(erf(z)/z, erf(z)/z^3 - (2/sqrt(pi)) exp(-z^2)/z^2)` as functions of
/// `t = z^2`. With `z = alpha r` the long-range Ewald pair energy is
/// `qq alpha P_V(t)` and its `(dE/dr)/r` is `-qq alpha^3 P_F(t)`. Both are
/// smooth at `t = 0`, where the closed forms cancel, so a Taylor series
/// takes over there.
pub(crate) fn long_range_pair_functions(t: f64) -> (f64, f64) {
    if t < 0.25 {
        let (mut energy, mut force) = (0., 0.);
        let mut term = 1.;
        for n in 0..18 {
            energy += term / (2 * n + 1) as f64;
            force += 2. * term / (2 * n + 3) as f64;
            term *= -t / (n + 1) as f64;
        }
        return (TWO_OVER_SQRT_PI * energy, TWO_OVER_SQRT_PI * force);
    }
    let z = t.sqrt();
    let energy = erf(z) / z;
    (energy, (energy - TWO_OVER_SQRT_PI * (-t).exp()) / t)
}

/// Ewald splitting coefficient (1/angstrom) for a direct-space cutoff, the
/// way GROMACS `calc_ewaldcoeff_q` computes it: the root of
/// `erfc(alpha rc) = rtol` by bisection to 2^-60. Note that this is the
/// potential at the cutoff relative to the bare Coulomb potential there;
/// GROMACS does not divide by `rc` (the `erfc(alpha rc)/rc = rtol` form
/// would make the result depend on the length unit).
pub fn ewald_coefficient(cutoff_angstrom: f64, ewald_rtol: f64) -> Result<f64> {
    if !cutoff_angstrom.is_finite() || cutoff_angstrom <= 0. {
        return Err(EnergyError::InvalidConfiguration(
            "Ewald cutoff must be positive".into(),
        ));
    }
    if !ewald_rtol.is_finite() || ewald_rtol <= 0. || ewald_rtol >= 1. {
        return Err(EnergyError::InvalidConfiguration(
            "Ewald relative tolerance must be between 0 and 1".into(),
        ));
    }
    // GROMACS starts from 5 nm^-1 and doubles until the tolerance is met.
    let mut beta = 0.5;
    let mut doublings = 0;
    loop {
        doublings += 1;
        beta *= 2.;
        if erfc(beta * cutoff_angstrom) <= ewald_rtol {
            break;
        }
    }
    let (mut low, mut high) = (0., beta);
    for _ in 0..doublings + 60 {
        beta = 0.5 * (low + high);
        if erfc(beta * cutoff_angstrom) > ewald_rtol {
            low = beta;
        } else {
            high = beta;
        }
    }
    Ok(beta)
}

/// Smallest FFT-friendly size (prime factors 2, 3, 5 and 7 only) that is at
/// least `minimum`.
pub fn fft_grid_size(minimum: usize) -> usize {
    let mut size = minimum.max(1);
    loop {
        let mut rest = size;
        for factor in [2, 3, 5, 7] {
            while rest.is_multiple_of(factor) {
                rest /= factor;
            }
        }
        if rest == 1 {
            return size;
        }
        size += 1;
    }
}

/// Mesh Ewald settings. The grid dimensions and `alpha` stay fixed when the
/// box changes under a barostat, as in GROMACS.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PmeParameters {
    pub alpha_per_angstrom: f64,
    pub grid: [usize; 3],
    pub interpolation_order: usize,
}

impl PmeParameters {
    pub fn new(
        alpha_per_angstrom: f64,
        grid: [usize; 3],
        interpolation_order: usize,
    ) -> Result<Self> {
        let parameters = Self {
            alpha_per_angstrom,
            grid,
            interpolation_order,
        };
        parameters.validate()?;
        Ok(parameters)
    }

    /// Settings for a box the way GROMACS derives them from `rcoulomb`,
    /// `ewald-rtol`, `fourierspacing` and `pme-order` (defaults 1e-5, 1.2 A
    /// and 4): `alpha` from [`ewald_coefficient`], and per axis the smallest
    /// FFT-friendly grid ([`fft_grid_size`]) whose spacing does not exceed
    /// `fourier_spacing_angstrom`, with the GROMACS minimum of
    /// `2 (order - 1)` points. GROMACS picks from a table tuned for FFTW, so
    /// its grid can be a few points larger than this one.
    pub fn for_box(
        box_vec: &BoxVectors,
        cutoff_angstrom: f64,
        ewald_rtol: f64,
        fourier_spacing_angstrom: f64,
        interpolation_order: usize,
    ) -> Result<Self> {
        if !fourier_spacing_angstrom.is_finite() || fourier_spacing_angstrom <= 0. {
            return Err(EnergyError::InvalidConfiguration(
                "Fourier spacing must be positive".into(),
            ));
        }
        let alpha = ewald_coefficient(cutoff_angstrom, ewald_rtol)?;
        let mut grid = [0; 3];
        for (points, length) in grid.iter_mut().zip(box_vec.as_array()) {
            // The small tolerance keeps an exact fit (36 A at 1.2 A) from
            // rounding up to the next size.
            let needed = length / fourier_spacing_angstrom;
            let needed = (needed - 1e-9 * needed.max(1.)).ceil();
            if !needed.is_finite() || needed > 4096. {
                return Err(EnergyError::InvalidConfiguration(
                    "Fourier spacing gives an unreasonable PME grid".into(),
                ));
            }
            let minimum = 2 * interpolation_order.saturating_sub(1);
            *points = fft_grid_size((needed as usize).max(minimum));
        }
        Self::new(alpha, grid, interpolation_order)
    }

    /// The direct-space pair function of these settings for
    /// [`crate::pbc::PbcForceField::evaluate`].
    pub fn backend(&self) -> PmeBackend {
        PmeBackend {
            alpha_per_angstrom: self.alpha_per_angstrom,
            grid: self.grid,
            interpolation_order: self.interpolation_order,
        }
    }

    fn validate(&self) -> Result<()> {
        if !self.alpha_per_angstrom.is_finite() || self.alpha_per_angstrom <= 0. {
            return Err(EnergyError::InvalidConfiguration(
                "Ewald coefficient must be positive".into(),
            ));
        }
        if !(MIN_ORDER..=MAX_ORDER).contains(&self.interpolation_order) {
            return Err(EnergyError::InvalidConfiguration(format!(
                "PME interpolation order must be {MIN_ORDER} to {MAX_ORDER}"
            )));
        }
        if self
            .grid
            .iter()
            .any(|&points| points < self.interpolation_order || points > 4096)
        {
            return Err(EnergyError::InvalidConfiguration(
                "each PME grid dimension must be between the interpolation order and 4096".into(),
            ));
        }
        Ok(())
    }
}

/// Long-range electrostatics of one evaluation: energy in kcal/mol and
/// virial `sum r.F` in kcal/mol.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PmeLongRange {
    pub energy: f64,
    pub virial: f64,
}

/// The parts of [`PmeLongRange::energy`] (kcal/mol), for diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PmeEnergyTerms {
    pub reciprocal: f64,
    pub self_energy: f64,
    pub excluded_pairs: f64,
    pub background: f64,
}

#[derive(Clone, Copy, Debug)]
struct ExcludedPair {
    a: u32,
    b: u32,
    qq: f64,
}

/// Base grid point and B-spline weights (with derivatives) of one atom per
/// axis: `theta[axis][k]` multiplies grid point `cell[axis] + k`.
#[derive(Clone, Copy, Debug, Default)]
struct AtomSpline {
    cell: [u32; 3],
    theta: [[f64; MAX_ORDER]; 3],
    dtheta: [[f64; MAX_ORDER]; 3],
}

/// Line buffer and FFT scratch of one parallel task.
#[derive(Clone, Default)]
struct Worker {
    line: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}

/// Reusable long-range PME evaluator for one prepared system.
#[derive(Clone)]
pub struct PmeEngine {
    parameters: PmeParameters,
    charge: Vec<f64>,
    self_energy: f64,
    net_charge: f64,
    excluded: Vec<ExcludedPair>,
    moduli: [Vec<f64>; 3],
    forward: [Arc<dyn Fft<f64>>; 3],
    inverse: [Arc<dyn Fft<f64>>; 3],
    scratch_len: usize,
    // Influence function, half spectrum in [ky][kz][kx] order, for one box.
    influence: Vec<f64>,
    influence_box: Option<BoxVectors>,
    frequency2: [Vec<f64>; 3],
    // Per-evaluation work, reused across calls.
    splines: Vec<AtomSpline>,
    bucket_start: Vec<u32>,
    bucket_atoms: Vec<u32>,
    mesh: Vec<f64>,
    half: Vec<Complex<f64>>,
    spectrum: Vec<Complex<f64>>,
    row_sums: Vec<[f64; 2]>,
    excluded_gradients: Vec<[f64; 3]>,
    workers: Vec<Worker>,
    last_terms: PmeEnergyTerms,
}

impl PmeEngine {
    /// Engine for a prepared system: its charges and every excluded pair,
    /// including the 1-4 pairs whose scaled Coulomb term the pair code adds.
    pub fn new(system: &ParameterizedSystem, parameters: PmeParameters) -> Result<Self> {
        let charges: Vec<f64> = system.atoms().iter().map(|atom| atom.charge()).collect();
        let mut pairs = BTreeSet::new();
        for (atom, set) in system.exclusions().iter().enumerate() {
            for &other in set {
                pairs.insert((atom.min(other), atom.max(other)));
            }
        }
        for (pair, _, _) in system.one_four_pairs() {
            pairs.insert((pair[0].min(pair[1]), pair[0].max(pair[1])));
        }
        let pairs: Vec<(usize, usize)> = pairs.into_iter().collect();
        Self::from_charges(&charges, &pairs, parameters)
    }

    /// Engine for bare charges (in units of e) and a list of excluded pairs.
    pub fn from_charges(
        charges: &[f64],
        excluded_pairs: &[(usize, usize)],
        parameters: PmeParameters,
    ) -> Result<Self> {
        parameters.validate()?;
        let n = charges.len();
        if n == 0 || n >= u32::MAX as usize {
            return Err(EnergyError::InvalidConfiguration(
                "PME needs between one and 2^32 atoms".into(),
            ));
        }
        if charges.iter().any(|q| !q.is_finite()) {
            return Err(EnergyError::InvalidConfiguration(
                "PME charges must be finite".into(),
            ));
        }
        let mut pairs = BTreeSet::new();
        for &(a, b) in excluded_pairs {
            if a >= n || b >= n {
                return Err(EnergyError::InvalidConfiguration(
                    "excluded pair names an atom outside the system".into(),
                ));
            }
            if a != b {
                pairs.insert((a.min(b), a.max(b)));
            }
        }
        let excluded = pairs
            .into_iter()
            .map(|(a, b)| ExcludedPair {
                a: a as u32,
                b: b as u32,
                qq: COULOMB * charges[a] * charges[b],
            })
            .filter(|pair| pair.qq != 0.)
            .collect();
        let alpha = parameters.alpha_per_angstrom;
        let order = parameters.interpolation_order;
        let [k1, k2, k3] = parameters.grid;
        let k3c = k3 / 2 + 1;
        let mut planner = FftPlanner::<f64>::new();
        let forward = parameters
            .grid
            .map(|points| planner.plan_fft_forward(points));
        let inverse = parameters
            .grid
            .map(|points| planner.plan_fft_inverse(points));
        let scratch_len = forward
            .iter()
            .chain(&inverse)
            .map(|plan| plan.get_inplace_scratch_len())
            .max()
            .unwrap_or(0);
        Ok(Self {
            parameters,
            charge: charges.to_vec(),
            self_energy: -COULOMB
                * alpha
                * 0.5
                * TWO_OVER_SQRT_PI
                * charges.iter().map(|q| q * q).sum::<f64>(),
            net_charge: charges.iter().sum(),
            excluded,
            moduli: parameters.grid.map(|points| spline_moduli(order, points)),
            forward,
            inverse,
            scratch_len,
            influence: vec![0.; k1 * k2 * k3c],
            influence_box: None,
            frequency2: Default::default(),
            splines: vec![AtomSpline::default(); n],
            bucket_start: Vec::new(),
            bucket_atoms: Vec::new(),
            mesh: vec![0.; k1 * k2 * k3],
            half: vec![Complex::default(); k1 * k3c * k2],
            spectrum: vec![Complex::default(); k2 * k3c * k1],
            row_sums: vec![[0.; 2]; k2],
            excluded_gradients: Vec::new(),
            workers: Vec::new(),
            last_terms: PmeEnergyTerms::default(),
        })
    }

    pub fn parameters(&self) -> &PmeParameters {
        &self.parameters
    }

    /// Number of excluded pairs that carry a correction.
    pub fn excluded_pair_count(&self) -> usize {
        self.excluded.len()
    }

    /// Energy terms of the last evaluation made with observables.
    pub fn last_terms(&self) -> PmeEnergyTerms {
        self.last_terms
    }

    /// Reciprocal-space sum + self energy + excluded-pair corrections (+ the
    /// uniform background term for a non-neutral box). Adds gradients into
    /// `gradients`. Excluded pairs use their unwrapped separation: they are
    /// bonded neighbours, never images. Without `observables` only the
    /// gradients are computed and the returned energy and virial are zero.
    pub fn evaluate_into(
        &mut self,
        unwrapped: &[Vec3],
        box_vec: &BoxVectors,
        gradients: &mut [Vec3],
        observables: bool,
    ) -> Result<PmeLongRange> {
        let n = self.charge.len();
        if unwrapped.len() != n || gradients.len() != n {
            return Err(EnergyError::CoordinateCount {
                expected: n,
                received: unwrapped.len().min(gradients.len()),
            });
        }
        if unwrapped
            .iter()
            .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
        {
            return Err(EnergyError::NonFiniteCoordinate);
        }
        let lengths = box_vec.as_array();
        if !lengths
            .iter()
            .all(|length| length.is_finite() && *length > 0.)
        {
            return Err(EnergyError::InvalidConfiguration(
                "periodic box vectors must be finite and positive".into(),
            ));
        }
        let tasks = TASKS_PER_THREAD * rayon::current_num_threads().max(1);
        if self.workers.len() != tasks {
            let line = self.parameters.grid.into_iter().max().unwrap_or(0);
            self.workers = vec![
                Worker {
                    line: vec![Complex::default(); line],
                    scratch: vec![Complex::default(); self.scratch_len],
                };
                tasks
            ];
        }
        if self.influence_box != Some(*box_vec) {
            self.update_influence(box_vec);
        }
        self.assign_splines(unwrapped, lengths);
        self.spread();
        let (reciprocal, reciprocal_virial) = self.convolve(observables);
        self.interpolate(lengths, gradients);
        let (excluded, excluded_virial) = self.excluded_into(unwrapped, gradients, observables);
        if !observables {
            return Ok(PmeLongRange::default());
        }
        let alpha = self.parameters.alpha_per_angstrom;
        // Uniform neutralizing background of a charged box; E ~ 1/V.
        let background = -COULOMB * PI * self.net_charge * self.net_charge
            / (2. * alpha * alpha * box_vec.volume());
        self.last_terms = PmeEnergyTerms {
            reciprocal,
            self_energy: self.self_energy,
            excluded_pairs: excluded,
            background,
        };
        let energy = reciprocal + self.self_energy + excluded + background;
        if !energy.is_finite() {
            return Err(EnergyError::InvalidConfiguration(
                "nonfinite PME energy".into(),
            ));
        }
        Ok(PmeLongRange {
            energy,
            virial: reciprocal_virial + excluded_virial + 3. * background,
        })
    }

    /// Influence function `C exp(-pi^2 m^2/alpha^2) / (pi V m^2 |b(m)|^2)`
    /// for one box, so the reciprocal energy is `1/2 sum_m G(m) |S(m)|^2`.
    fn update_influence(&mut self, box_vec: &BoxVectors) {
        let [k1, _, k3] = self.parameters.grid;
        let k3c = k3 / 2 + 1;
        let alpha = self.parameters.alpha_per_angstrom;
        let factor = PI * PI / (alpha * alpha);
        let lengths = box_vec.as_array();
        let mut gauss: [Vec<f64>; 3] = Default::default();
        for axis in 0..3 {
            let points = self.parameters.grid[axis];
            // Indices in the upper half are negative frequencies.
            self.frequency2[axis] = (0..points)
                .map(|k| {
                    let m = if 2 * k <= points {
                        k as f64
                    } else {
                        k as f64 - points as f64
                    };
                    (m / lengths[axis]).powi(2)
                })
                .collect();
            gauss[axis] = self.frequency2[axis]
                .iter()
                .zip(&self.moduli[axis])
                .map(|(m2, modulus)| (-factor * m2).exp() / modulus)
                .collect();
        }
        let prefactor = COULOMB / (PI * box_vec.volume());
        let frequency2 = &self.frequency2;
        self.influence
            .par_chunks_mut(k3c * k1)
            .enumerate()
            .for_each(|(ky, plane)| {
                for (kz, row) in plane.chunks_mut(k1).enumerate() {
                    let yz = prefactor * gauss[1][ky] * gauss[2][kz];
                    let m2_yz = frequency2[1][ky] + frequency2[2][kz];
                    for (kx, value) in row.iter_mut().enumerate() {
                        *value = yz * gauss[0][kx] / (frequency2[0][kx] + m2_yz);
                    }
                }
            });
        // The m = 0 term is the uniform background, handled analytically.
        self.influence[0] = 0.;
        self.influence_box = Some(*box_vec);
    }

    fn assign_splines(&mut self, unwrapped: &[Vec3], lengths: [f64; 3]) {
        let grid = self.parameters.grid;
        let order = self.parameters.interpolation_order;
        self.splines
            .par_iter_mut()
            .zip(unwrapped.par_iter())
            .with_min_len(ATOM_CHUNK)
            .for_each(|(spline, p)| {
                for (axis, value) in [p.x, p.y, p.z].into_iter().enumerate() {
                    let points = grid[axis] as f64;
                    let scaled = value / lengths[axis];
                    let mut u = (scaled - scaled.floor()) * points;
                    // A fraction that rounds to one is grid point zero.
                    if u >= points {
                        u = 0.;
                    }
                    let cell = u.floor();
                    spline.cell[axis] = cell as u32;
                    bspline(
                        u - cell,
                        order,
                        &mut spline.theta[axis],
                        &mut spline.dtheta[axis],
                    );
                }
            });
        // Charged atoms bucketed by their first x plane, in atom order.
        let k1 = grid[0];
        self.bucket_start.clear();
        self.bucket_start.resize(k1 + 1, 0);
        for (spline, &q) in self.splines.iter().zip(&self.charge) {
            if q != 0. {
                self.bucket_start[spline.cell[0] as usize + 1] += 1;
            }
        }
        for x in 0..k1 {
            self.bucket_start[x + 1] += self.bucket_start[x];
        }
        self.bucket_atoms.resize(self.bucket_start[k1] as usize, 0);
        let mut cursor = self.bucket_start.clone();
        for (atom, (spline, &q)) in self.splines.iter().zip(&self.charge).enumerate() {
            if q != 0. {
                let slot = &mut cursor[spline.cell[0] as usize];
                self.bucket_atoms[*slot as usize] = atom as u32;
                *slot += 1;
            }
        }
    }

    /// Charge spreading, one x plane per task: a plane collects, in a fixed
    /// order, every atom whose stencil reaches it, so no two tasks write the
    /// same memory and the sum does not depend on the thread count.
    fn spread(&mut self) {
        let [k1, k2, k3] = self.parameters.grid;
        let order = self.parameters.interpolation_order;
        let (splines, charge) = (&self.splines, &self.charge);
        let (bucket_start, bucket_atoms) = (&self.bucket_start, &self.bucket_atoms);
        self.mesh
            .par_chunks_mut(k2 * k3)
            .enumerate()
            .for_each(|(x, plane)| {
                plane.fill(0.);
                for jx in 0..order {
                    let first = (x + k1 - jx) % k1;
                    let range = bucket_start[first] as usize..bucket_start[first + 1] as usize;
                    for &atom in &bucket_atoms[range] {
                        let spline = &splines[atom as usize];
                        let wx = charge[atom as usize] * spline.theta[0][jx];
                        let (iy, iz) = (spline.cell[1] as usize, spline.cell[2] as usize);
                        for jy in 0..order {
                            let y = if iy + jy >= k2 { iy + jy - k2 } else { iy + jy };
                            let wxy = wx * spline.theta[1][jy];
                            let row = &mut plane[y * k3..(y + 1) * k3];
                            for jz in 0..order {
                                let z = if iz + jz >= k3 { iz + jz - k3 } else { iz + jz };
                                row[z] += wxy * spline.theta[2][jz];
                            }
                        }
                    }
                }
            });
    }

    /// Mesh charges to mesh potential: forward transform, multiplication by
    /// the influence function, inverse transform. Returns the reciprocal
    /// energy and virial when `observables` is set.
    fn convolve(&mut self, observables: bool) -> (f64, f64) {
        let [k1, k2, k3] = self.parameters.grid;
        let k3c = k3 / 2 + 1;
        let alpha = self.parameters.alpha_per_angstrom;
        let tasks = self.workers.len();
        let planes = k1.div_ceil(tasks);
        let rows = k2.div_ceil(tasks);
        let [forward_x, forward_y, forward_z] = &self.forward;
        let [inverse_x, inverse_y, inverse_z] = &self.inverse;

        // z (two real lines per complex transform) then y, per x plane; the
        // half spectrum lands in [x][kz][ky] order.
        self.mesh
            .par_chunks(planes * k2 * k3)
            .zip(self.half.par_chunks_mut(planes * k3c * k2))
            .zip(self.workers.par_iter_mut())
            .for_each(|((mesh, half), worker)| {
                let line = &mut worker.line[..k3];
                for (real, plane) in mesh.chunks(k2 * k3).zip(half.chunks_mut(k3c * k2)) {
                    for y in (0..k2).step_by(2) {
                        let first = &real[y * k3..(y + 1) * k3];
                        if y + 1 < k2 {
                            let second = &real[(y + 1) * k3..(y + 2) * k3];
                            for z in 0..k3 {
                                line[z] = Complex::new(first[z], second[z]);
                            }
                            forward_z.process_with_scratch(line, &mut worker.scratch);
                            for kz in 0..k3c {
                                let direct = line[kz];
                                let mirror = line[if kz == 0 { 0 } else { k3 - kz }].conj();
                                let odd = direct - mirror;
                                plane[kz * k2 + y] = (direct + mirror) * 0.5;
                                plane[kz * k2 + y + 1] = Complex::new(odd.im, -odd.re) * 0.5;
                            }
                        } else {
                            for z in 0..k3 {
                                line[z] = Complex::new(first[z], 0.);
                            }
                            forward_z.process_with_scratch(line, &mut worker.scratch);
                            for kz in 0..k3c {
                                plane[kz * k2 + y] = line[kz];
                            }
                        }
                    }
                    forward_y.process_with_scratch(plane, &mut worker.scratch);
                }
            });

        // x transform, influence function and inverse x, per ky plane of the
        // [ky][kz][kx] spectrum; energy and virial are kept per plane.
        let half = &self.half;
        let frequency2 = &self.frequency2;
        let virial_factor = 2. * PI * PI / (alpha * alpha);
        self.spectrum
            .par_chunks_mut(rows * k3c * k1)
            .zip(self.influence.par_chunks(rows * k3c * k1))
            .zip(self.row_sums.par_chunks_mut(rows))
            .zip(self.workers.par_iter_mut())
            .enumerate()
            .for_each(|(task, (((spectrum, influence), sums), worker))| {
                let slabs = spectrum
                    .chunks_mut(k3c * k1)
                    .zip(influence.chunks(k3c * k1))
                    .zip(sums.iter_mut());
                for (offset, ((plane, g), sum)) in slabs.enumerate() {
                    let ky = task * rows + offset;
                    for x in 0..k1 {
                        let source = &half[x * k3c * k2..(x + 1) * k3c * k2];
                        for kz in 0..k3c {
                            plane[kz * k1 + x] = source[kz * k2 + ky];
                        }
                    }
                    forward_x.process_with_scratch(plane, &mut worker.scratch);
                    if observables {
                        let (mut energy, mut virial) = (0., 0.);
                        for kz in 0..k3c {
                            // Interior kz stand for their mirror images too.
                            let weight = if kz == 0 || 2 * kz == k3 { 1. } else { 2. };
                            let m2_yz = frequency2[1][ky] + frequency2[2][kz];
                            let (mut e_row, mut w_row) = (0., 0.);
                            let row = kz * k1..(kz + 1) * k1;
                            let values = plane[row.clone()].iter_mut().zip(&g[row]);
                            for ((value, g), m2_x) in values.zip(&frequency2[0]) {
                                let term = g * value.norm_sqr();
                                e_row += term;
                                w_row += term * (1. - virial_factor * (m2_x + m2_yz));
                                *value *= *g;
                            }
                            energy += weight * e_row;
                            virial += weight * w_row;
                        }
                        *sum = [0.5 * energy, 0.5 * virial];
                    } else {
                        for (value, g) in plane.iter_mut().zip(g) {
                            *value *= *g;
                        }
                    }
                    inverse_x.process_with_scratch(plane, &mut worker.scratch);
                }
            });

        // Inverse y then z, per x plane, back onto the real mesh.
        let spectrum = &self.spectrum;
        self.mesh
            .par_chunks_mut(planes * k2 * k3)
            .zip(self.half.par_chunks_mut(planes * k3c * k2))
            .zip(self.workers.par_iter_mut())
            .enumerate()
            .for_each(|(task, ((mesh, half), worker))| {
                let line = &mut worker.line[..k3];
                let slabs = mesh.chunks_mut(k2 * k3).zip(half.chunks_mut(k3c * k2));
                for (offset, (real, plane)) in slabs.enumerate() {
                    let x = task * planes + offset;
                    for ky in 0..k2 {
                        let source = &spectrum[ky * k3c * k1..(ky + 1) * k3c * k1];
                        for kz in 0..k3c {
                            plane[kz * k2 + ky] = source[kz * k1 + x];
                        }
                    }
                    inverse_y.process_with_scratch(plane, &mut worker.scratch);
                    for y in (0..k2).step_by(2) {
                        let paired = y + 1 < k2;
                        for kz in 0..k3c {
                            let a = plane[kz * k2 + y];
                            let b = if paired {
                                plane[kz * k2 + y + 1]
                            } else {
                                Complex::default()
                            };
                            // a + i b, and its Hermitian mirror conj(a) + i conj(b).
                            line[kz] = Complex::new(a.re - b.im, a.im + b.re);
                            if kz != 0 && 2 * kz != k3 {
                                line[k3 - kz] = Complex::new(a.re + b.im, b.re - a.im);
                            }
                        }
                        inverse_z.process_with_scratch(line, &mut worker.scratch);
                        for z in 0..k3 {
                            real[y * k3 + z] = line[z].re;
                        }
                        if paired {
                            for z in 0..k3 {
                                real[(y + 1) * k3 + z] = line[z].im;
                            }
                        }
                    }
                }
            });

        if !observables {
            return (0., 0.);
        }
        self.row_sums
            .iter()
            .fold((0., 0.), |total, sum| (total.0 + sum[0], total.1 + sum[1]))
    }

    /// Reciprocal gradients from the mesh potential and the analytic spline
    /// derivatives; atoms are independent.
    fn interpolate(&self, lengths: [f64; 3], gradients: &mut [Vec3]) {
        let [k1, k2, k3] = self.parameters.grid;
        let order = self.parameters.interpolation_order;
        let scale = [
            k1 as f64 / lengths[0],
            k2 as f64 / lengths[1],
            k3 as f64 / lengths[2],
        ];
        let mesh = &self.mesh;
        gradients
            .par_iter_mut()
            .zip(self.splines.par_iter())
            .zip(self.charge.par_iter())
            .with_min_len(ATOM_CHUNK)
            .for_each(|((gradient, spline), &q)| {
                if q == 0. {
                    return;
                }
                let [ix, iy, iz] = spline.cell.map(|cell| cell as usize);
                let mut sum = [0.; 3];
                for jx in 0..order {
                    let x = if ix + jx >= k1 { ix + jx - k1 } else { ix + jx };
                    let mut plane = [0.; 3];
                    for jy in 0..order {
                        let y = if iy + jy >= k2 { iy + jy - k2 } else { iy + jy };
                        let row = &mesh[(x * k2 + y) * k3..(x * k2 + y + 1) * k3];
                        let (mut value, mut slope) = (0., 0.);
                        for jz in 0..order {
                            let z = if iz + jz >= k3 { iz + jz - k3 } else { iz + jz };
                            value += spline.theta[2][jz] * row[z];
                            slope += spline.dtheta[2][jz] * row[z];
                        }
                        plane[0] += spline.theta[1][jy] * value;
                        plane[1] += spline.dtheta[1][jy] * value;
                        plane[2] += spline.theta[1][jy] * slope;
                    }
                    sum[0] += spline.dtheta[0][jx] * plane[0];
                    sum[1] += spline.theta[0][jx] * plane[1];
                    sum[2] += spline.theta[0][jx] * plane[2];
                }
                gradient.x += q * scale[0] * sum[0];
                gradient.y += q * scale[1] * sum[1];
                gradient.z += q * scale[2] * sum[2];
            });
    }

    /// Remove `qq erf(alpha r)/r` for every excluded pair: the reciprocal
    /// sum contains it, the pair code never adds the matching `erfc` part.
    fn excluded_into(
        &mut self,
        unwrapped: &[Vec3],
        gradients: &mut [Vec3],
        observables: bool,
    ) -> (f64, f64) {
        let alpha = self.parameters.alpha_per_angstrom;
        let alpha3 = alpha * alpha * alpha;
        self.excluded_gradients.resize(self.excluded.len(), [0.; 3]);
        // Fixed-size chunks summed in order: independent of the thread count.
        let sums: Vec<[f64; 2]> = self
            .excluded
            .par_chunks(EXCLUDED_CHUNK)
            .zip(self.excluded_gradients.par_chunks_mut(EXCLUDED_CHUNK))
            .map(|(pairs, outputs)| {
                let (mut energy, mut virial) = (0., 0.);
                for (pair, output) in pairs.iter().zip(outputs) {
                    let (a, b) = (unwrapped[pair.a as usize], unwrapped[pair.b as usize]);
                    let d = [a.x - b.x, a.y - b.y, a.z - b.z];
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    let (potential, force) = long_range_pair_functions(alpha * alpha * r2);
                    // E = -qq alpha P_V and (dE/dr)/r = qq alpha^3 P_F.
                    let fmag = pair.qq * alpha3 * force;
                    *output = [fmag * d[0], fmag * d[1], fmag * d[2]];
                    energy -= pair.qq * alpha * potential;
                    virial -= fmag * r2;
                }
                [energy, virial]
            })
            .collect();
        for (pair, gradient) in self.excluded.iter().zip(&self.excluded_gradients) {
            let (a, b) = (pair.a as usize, pair.b as usize);
            gradients[a].x += gradient[0];
            gradients[a].y += gradient[1];
            gradients[a].z += gradient[2];
            gradients[b].x -= gradient[0];
            gradients[b].y -= gradient[1];
            gradients[b].z -= gradient[2];
        }
        if !observables {
            return (0., 0.);
        }
        sums.iter()
            .fold((0., 0.), |total, sum| (total.0 + sum[0], total.1 + sum[1]))
    }
}

/// Cardinal B-spline weights of `order` points and their derivatives for a
/// particle at fraction `w` in [0, 1) past its base grid point (the
/// recursion of Essmann et al., as in GROMACS and OpenMM).
fn bspline(w: f64, order: usize, theta: &mut [f64; MAX_ORDER], dtheta: &mut [f64; MAX_ORDER]) {
    theta[order - 1] = 0.;
    theta[1] = w;
    theta[0] = 1. - w;
    for k in 3..order {
        let div = 1. / (k - 1) as f64;
        theta[k - 1] = div * w * theta[k - 2];
        for l in 1..k - 1 {
            theta[k - l - 1] =
                div * ((w + l as f64) * theta[k - l - 2] + ((k - l) as f64 - w) * theta[k - l - 1]);
        }
        theta[0] *= div * (1. - w);
    }
    dtheta[0] = -theta[0];
    for k in 1..order {
        dtheta[k] = theta[k - 1] - theta[k];
    }
    let div = 1. / (order - 1) as f64;
    theta[order - 1] = div * w * theta[order - 2];
    for l in 1..order - 1 {
        theta[order - l - 1] = div
            * ((w + l as f64) * theta[order - l - 2]
                + ((order - l) as f64 - w) * theta[order - l - 1]);
    }
    theta[0] *= div * (1. - w);
}

/// `|b(m)|^-2` of Essmann et al.: the squared modulus of the discrete
/// Fourier transform of the spline at the grid points. For an odd order on
/// an even grid it vanishes at the Nyquist index; like GROMACS and OpenMM
/// (and Darden's original code) the mean of the neighbours stands in there.
fn spline_moduli(order: usize, points: usize) -> Vec<f64> {
    let (mut theta, mut dtheta) = ([0.; MAX_ORDER], [0.; MAX_ORDER]);
    bspline(0., order, &mut theta, &mut dtheta);
    let raw: Vec<f64> = (0..points)
        .map(|m| {
            let (mut re, mut im) = (0., 0.);
            for (k, weight) in theta[..order].iter().enumerate() {
                let arg = 2. * PI * (m * (k + 1)) as f64 / points as f64;
                re += weight * arg.cos();
                im += weight * arg.sin();
            }
            re * re + im * im
        })
        .collect();
    (0..points)
        .map(|m| {
            if raw[m] < 1e-7 {
                0.5 * (raw[(m + points - 1) % points] + raw[(m + 1) % points])
            } else {
                raw[m]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::{ElectrostaticsBackend, PbcForceField, PbcNeighborList};

    fn v(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    fn zeros(n: usize) -> Vec<Vec3> {
        vec![v(0., 0., 0.); n]
    }

    /// Deterministic uniform numbers in [0, 1).
    struct Uniform(u64);

    impl Uniform {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    const BOX: [f64; 3] = [11.0, 12.5, 10.2];
    /// Bonded-like close pairs of [`random_system`], excluded in some tests.
    const BONDED: [(usize, usize); 3] = [(0, 1), (2, 3), (2, 4)];

    /// Random charges (net charge `net`) at random positions, with a few
    /// atoms moved next to each other like bonded neighbours.
    fn random_system(seed: u64, n: usize, net: f64) -> (Vec<f64>, Vec<Vec3>) {
        let mut random = Uniform(seed);
        let mut charges: Vec<f64> = (0..n).map(|_| 2. * random.next() - 1.).collect();
        let shift = (net - charges.iter().sum::<f64>()) / n as f64;
        charges.iter_mut().for_each(|q| *q += shift);
        let mut coords: Vec<Vec3> = (0..n)
            .map(|_| {
                v(
                    BOX[0] * random.next(),
                    BOX[1] * random.next(),
                    BOX[2] * random.next(),
                )
            })
            .collect();
        coords[1] = v(coords[0].x + 0.6, coords[0].y + 0.5, coords[0].z - 0.4);
        coords[3] = v(coords[2].x - 0.9, coords[2].y + 0.3, coords[2].z + 0.2);
        coords[4] = v(coords[2].x + 0.2, coords[2].y - 0.8, coords[2].z + 0.7);
        (charges, coords)
    }

    struct Reference {
        real: f64,
        reciprocal: f64,
        self_energy: f64,
        background: f64,
        gradients: Vec<[f64; 3]>,
    }

    impl Reference {
        fn energy(&self) -> f64 {
            self.real + self.reciprocal + self.self_energy + self.background
        }
    }

    /// Brute-force Ewald sum over every pair: explicit image shells and an
    /// explicit sum over reciprocal vectors, both converged to roundoff.
    fn ewald_reference(
        charges: &[f64],
        coords: &[Vec3],
        lengths: [f64; 3],
        alpha: f64,
    ) -> Reference {
        let n = charges.len();
        let volume = lengths[0] * lengths[1] * lengths[2];
        let shortest = lengths.iter().copied().fold(f64::INFINITY, f64::min);
        let shells = ((6.5 / (alpha * shortest) - 0.5).ceil() as i32).max(1);
        let mut gradients = vec![[0.; 3]; n];
        let mut real = 0.;
        for i in 0..n {
            for j in i..n {
                let mut d = [
                    coords[i].x - coords[j].x,
                    coords[i].y - coords[j].y,
                    coords[i].z - coords[j].z,
                ];
                for axis in 0..3 {
                    d[axis] -= lengths[axis] * (d[axis] / lengths[axis]).round();
                }
                let qq = COULOMB * charges[i] * charges[j] * if i == j { 0.5 } else { 1. };
                for nx in -shells..=shells {
                    for ny in -shells..=shells {
                        for nz in -shells..=shells {
                            if i == j && nx == 0 && ny == 0 && nz == 0 {
                                continue;
                            }
                            let image = [
                                d[0] + f64::from(nx) * lengths[0],
                                d[1] + f64::from(ny) * lengths[1],
                                d[2] + f64::from(nz) * lengths[2],
                            ];
                            let r =
                                (image[0] * image[0] + image[1] * image[1] + image[2] * image[2])
                                    .sqrt();
                            let screened = erfc(alpha * r);
                            real += qq * screened / r;
                            let slope = -qq
                                * (screened / (r * r)
                                    + alpha * TWO_OVER_SQRT_PI * (-alpha * alpha * r * r).exp()
                                        / r);
                            for axis in 0..3 {
                                gradients[i][axis] += slope * image[axis] / r;
                                gradients[j][axis] -= slope * image[axis] / r;
                            }
                        }
                    }
                }
            }
        }
        // exp(-pi^2 m^2/alpha^2) < exp(-36) beyond this radius.
        let m_max = 6. * alpha / PI;
        let k_max: [i32; 3] = std::array::from_fn(|axis| (m_max * lengths[axis]).ceil() as i32);
        let mut reciprocal = 0.;
        let mut phase = vec![(0., 0.); n];
        for kx in 0..=k_max[0] {
            for ky in -k_max[1]..=k_max[1] {
                for kz in -k_max[2]..=k_max[2] {
                    // Half space; the mirror image doubles each term.
                    if kx == 0 && (ky < 0 || (ky == 0 && kz <= 0)) {
                        continue;
                    }
                    let m = [
                        f64::from(kx) / lengths[0],
                        f64::from(ky) / lengths[1],
                        f64::from(kz) / lengths[2],
                    ];
                    let m2 = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
                    let f = (-PI * PI * m2 / (alpha * alpha)).exp() / m2;
                    let (mut s_re, mut s_im) = (0., 0.);
                    for (j, p) in coords.iter().enumerate() {
                        let angle = 2. * PI * (m[0] * p.x + m[1] * p.y + m[2] * p.z);
                        phase[j] = (angle.cos(), angle.sin());
                        s_re += charges[j] * phase[j].0;
                        s_im += charges[j] * phase[j].1;
                    }
                    reciprocal += COULOMB / (PI * volume) * f * (s_re * s_re + s_im * s_im);
                    for i in 0..n {
                        let g = 2. * COULOMB / (PI * volume)
                            * f
                            * charges[i]
                            * 2.
                            * PI
                            * (phase[i].0 * s_im - phase[i].1 * s_re);
                        for axis in 0..3 {
                            gradients[i][axis] += g * m[axis];
                        }
                    }
                }
            }
        }
        let net: f64 = charges.iter().sum();
        Reference {
            real,
            reciprocal,
            self_energy: -COULOMB * alpha / PI.sqrt() * charges.iter().map(|q| q * q).sum::<f64>(),
            background: -COULOMB * PI * net * net / (2. * alpha * alpha * volume),
            gradients,
        }
    }

    /// Minimum-image `erfc` pair sum inside the cutoff, skipping `excluded`.
    fn direct_sum(
        charges: &[f64],
        coords: &[Vec3],
        box_vec: &BoxVectors,
        backend: &PmeBackend,
        cutoff: f64,
        excluded: &[(usize, usize)],
        gradients: &mut [Vec3],
    ) -> (f64, f64) {
        let (mut energy, mut virial) = (0., 0.);
        for i in 0..charges.len() {
            for j in i + 1..charges.len() {
                if excluded.contains(&(i, j)) {
                    continue;
                }
                let d = box_vec.displacement(coords[i], coords[j]);
                let r = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
                if r > cutoff {
                    continue;
                }
                let (e, slope) = backend.pair(r, COULOMB * charges[i] * charges[j]);
                energy += e;
                virial -= slope * r;
                for (a, sign) in [(i, 1.), (j, -1.)] {
                    gradients[a].x += sign * slope * d.x / r;
                    gradients[a].y += sign * slope * d.y / r;
                    gradients[a].z += sign * slope * d.z / r;
                }
            }
        }
        (energy, virial)
    }

    fn rms(values: impl Iterator<Item = f64>) -> f64 {
        let (mut sum, mut count) = (0., 0usize);
        for value in values {
            sum += value * value;
            count += 1;
        }
        (sum / count.max(1) as f64).sqrt()
    }

    #[test]
    fn erfc_matches_reference_values() {
        // glibc erfc, correct to under one unit in the last place.
        let cases = [
            (-3.0, 1.9999779095030015),
            (-0.75, 1.7111556336535152),
            (-0.2, 1.2227025892104786),
            (0.0, 1.0),
            (1e-10, 0.999999999887162),
            (0.05, 0.9436280222029834),
            (0.3, 0.6713732405408726),
            (0.46875, 0.507386526782062),
            (0.47, 0.5062549491139179),
            (0.5, 0.4795001221869535),
            (0.9, 0.20309178757716786),
            (1.0, 0.15729920705028513),
            (1.5, 0.033894853524689274),
            (2.0, 0.004677734981047265),
            (3.0, 2.2090496998585438e-05),
            (3.123413, 1.0000017942986116e-05),
            (4.0, 1.541725790028002e-08),
            (4.0000001, 1.541724520205039e-08),
            (5.0, 1.5374597944280351e-12),
            (8.0, 1.1224297172982928e-29),
            (12.0, 1.3562611692059042e-64),
            (20.0, 5.3958656116079005e-176),
            (26.0, 5.663192408856143e-296),
        ];
        for (x, want) in cases {
            let got = erfc(x);
            assert!(
                (got - want).abs() <= 4e-16 * want,
                "erfc({x}) = {got:e}, expected {want:e}"
            );
        }
        for (x, want) in [
            (0.1, 0.1124629160182849),
            (0.5, 0.5204998778130465),
            (1.0, 0.8427007929497149),
            (2.5, 0.999593047982555),
        ] {
            assert!((erf(x) - want).abs() <= 4e-16 * want, "erf({x})");
            assert!((erf(-x) + want).abs() <= 4e-16 * want, "erf(-{x})");
        }
        assert_eq!(erfc(27.), 0.);
        assert_eq!(erfc(f64::INFINITY), 0.);
        assert_eq!(erfc(f64::NEG_INFINITY), 2.);
        assert!(erfc(f64::NAN).is_nan());
        // Continuity across the three approximation intervals and the
        // derivative -2/sqrt(pi) exp(-x^2) everywhere.
        for x in [0.2, 0.46874, 0.46876, 1.3, 3.9999, 4.0001, 6.5] {
            let h = 1e-5;
            let numeric = (erfc(x + h) - erfc(x - h)) / (2. * h);
            let exact = -TWO_OVER_SQRT_PI * f64::exp(-x * x);
            assert!(
                (numeric - exact).abs() <= 2e-9 * exact.abs().max(1e-12),
                "erfc slope at {x}: {numeric:e} vs {exact:e}"
            );
        }
    }

    #[test]
    fn long_range_pair_functions_match_their_closed_forms() {
        // The limits at t = 0 are 2/sqrt(pi) and 4/(3 sqrt(pi)); the series
        // and the closed forms meet at t = 0.25.
        let (potential, force) = long_range_pair_functions(0.);
        assert!((potential - TWO_OVER_SQRT_PI).abs() < 1e-15);
        assert!((force - 2. / 3. * TWO_OVER_SQRT_PI).abs() < 1e-15);
        for t in [1e-8, 0.01, 0.2, 0.249_999, 0.25, 0.3, 1.7, 9.7, 40.] {
            let z = f64::sqrt(t);
            let (potential, force) = long_range_pair_functions(t);
            let want = erf(z) / z;
            assert!((potential - want).abs() <= 4e-16 * want, "P_V({t})");
            // P_F = -2 dP_V/dt.
            let h = 1e-5 * t.max(1e-3);
            let low = (t - h).max(0.);
            let (up, _) = long_range_pair_functions(t + h);
            let (down, _) = long_range_pair_functions(low);
            let numeric = -2. * (up - down) / (t + h - low);
            assert!(
                (force - numeric).abs() <= 1e-7 * force.abs().max(1e-3),
                "P_F({t}) = {force} vs {numeric}"
            );
        }
        let below = long_range_pair_functions(f64::from_bits(0.25f64.to_bits() - 1));
        let above = long_range_pair_functions(0.25);
        assert!((below.0 - above.0).abs() < 1e-15 && (below.1 - above.1).abs() < 2e-15);
    }

    #[test]
    fn parameters_follow_the_gromacs_rules() {
        // GROMACS reports an Ewald coefficient of 3.12341 nm^-1 for
        // rc = 1 nm and 3.47046 nm^-1 for rc = 0.9 nm at ewald-rtol 1e-5.
        let alpha = ewald_coefficient(10., 1e-5).unwrap();
        assert!((alpha - 0.312_341).abs() < 5e-7, "{alpha}");
        let alpha = ewald_coefficient(9., 1e-5).unwrap();
        assert!((alpha - 0.347_046).abs() < 5e-7, "{alpha}");
        assert!((erfc(alpha * 9.) - 1e-5).abs() < 1e-18);
        assert!(ewald_coefficient(9., 0.).is_err());
        assert!(ewald_coefficient(-1., 1e-5).is_err());

        assert_eq!(
            [1, 7, 11, 13, 34, 37, 43, 97, 121].map(fft_grid_size),
            [1, 7, 12, 14, 35, 40, 45, 98, 125]
        );
        let box_vec = BoxVectors::new(41.918, 50.479, 39.631).unwrap();
        let parameters = PmeParameters::for_box(&box_vec, 9., 1e-5, 1.2, 4).unwrap();
        assert_eq!(parameters.grid, [35, 45, 35]);
        assert_eq!(parameters.interpolation_order, 4);
        assert_eq!(parameters.alpha_per_angstrom, alpha);
        for (points, length) in parameters.grid.iter().zip(box_vec.as_array()) {
            assert!(length / *points as f64 <= 1.2);
        }
        // An exact fit is not rounded up; tiny boxes get the minimum grid.
        let exact = BoxVectors::new(36., 12., 2.).unwrap();
        let parameters = PmeParameters::for_box(&exact, 0.9, 1e-5, 1.2, 6).unwrap();
        assert_eq!(parameters.grid, [30, 10, 10]);
        for order in [3, 7] {
            assert!(PmeParameters::for_box(&box_vec, 9., 1e-5, 1.2, order).is_err());
        }
        assert!(PmeParameters::new(0.3, [32, 32, 5], 6).is_err());
        assert!(PmeParameters::new(0., [32, 32, 32], 4).is_err());
    }

    #[test]
    fn splines_partition_unity_and_moduli_are_positive() {
        for order in MIN_ORDER..=MAX_ORDER {
            let (mut theta, mut dtheta) = ([0.; MAX_ORDER], [0.; MAX_ORDER]);
            for w in [0., 0.13, 0.5, 0.999_999] {
                bspline(w, order, &mut theta, &mut dtheta);
                let sum: f64 = theta[..order].iter().sum();
                let slope: f64 = dtheta[..order].iter().sum();
                assert!((sum - 1.).abs() < 1e-14 && slope.abs() < 1e-14);
                // Derivative weights against a finite difference in w.
                let h = 1e-6;
                let low = (w - h).max(0.);
                let (mut up, mut down) = ([0.; MAX_ORDER], [0.; MAX_ORDER]);
                let mut unused = [0.; MAX_ORDER];
                bspline(w + h, order, &mut up, &mut unused);
                bspline(low, order, &mut down, &mut unused);
                for k in 0..order {
                    let numeric = (up[k] - down[k]) / (w + h - low);
                    assert!(
                        (dtheta[k] - numeric).abs() < 2e-6,
                        "order {order} w {w} k {k}"
                    );
                }
            }
            for points in [order, 16, 35, 36] {
                let moduli = spline_moduli(order, points);
                assert!((moduli[0] - 1.).abs() < 1e-14);
                assert!(
                    moduli.iter().all(|m| *m > 1e-7),
                    "order {order} grid {points}"
                );
                // Symmetric in the frequency sign.
                for m in 1..points {
                    assert!((moduli[m] - moduli[points - m]).abs() < 1e-14);
                }
            }
        }
        // Order 5 on an even grid: the zero at the Nyquist index is replaced
        // by the mean of its neighbours.
        let moduli = spline_moduli(5, 36);
        assert_eq!(moduli[18], 0.5 * (moduli[17] + moduli[19]));
    }

    #[test]
    fn matches_brute_force_ewald_on_random_systems() {
        let box_vec = BoxVectors::new(BOX[0], BOX[1], BOX[2]).unwrap();
        let cutoff = 5.0;
        for (net, seed) in [(0., 0x9e37_79b9_7f4a_7c15u64), (1.7, 0x2545_f491_4f6c_dd1d)] {
            let (charges, coords) = random_system(seed, 30, net);
            // A different splitting coefficient makes the oracle independent
            // of how the engine divides the sum.
            let reference = ewald_reference(&charges, &coords, BOX, 0.55);
            let same = ewald_reference(&charges, &coords, BOX, 0.92);
            let force_rms = rms(reference.gradients.iter().flatten().copied());
            // Mesh errors fall with the spline order and the grid spacing;
            // each tolerance is about three times the measured error.
            let coarse = [64, 72, 60];
            for (order, grid, energy_tolerance, force_tolerance) in [
                (4, coarse, 7e-2, 8e-4),
                (5, coarse, 2e-3, 2e-5),
                (6, coarse, 4e-4, 4e-6),
                (6, [108, 120, 100], 2e-5, 3e-7),
            ] {
                let parameters = PmeParameters::new(0.92, grid, order).unwrap();
                let mut engine = PmeEngine::from_charges(&charges, &[], parameters).unwrap();
                let mut gradients = zeros(charges.len());
                let long_range = engine
                    .evaluate_into(&coords, &box_vec, &mut gradients, true)
                    .unwrap();
                let (direct, direct_virial) = direct_sum(
                    &charges,
                    &coords,
                    &box_vec,
                    &parameters.backend(),
                    cutoff,
                    &[],
                    &mut gradients,
                );
                let energy = direct + long_range.energy;
                let error = rms(gradients
                    .iter()
                    .zip(&reference.gradients)
                    .flat_map(|(g, w)| [g.x - w[0], g.y - w[1], g.z - w[2]]));
                let terms = engine.last_terms();
                assert!(
                    (energy - reference.energy()).abs() < energy_tolerance,
                    "order {order} net {net}: energy {energy} vs {}",
                    reference.energy()
                );
                assert!(
                    error < force_tolerance * force_rms,
                    "order {order} net {net}: force error {error} of rms {force_rms}"
                );
                // The mesh sum alone against the exact reciprocal sum at the
                // same alpha, and the analytic terms.
                assert!(
                    (terms.reciprocal - same.reciprocal).abs() < energy_tolerance,
                    "order {order}: reciprocal {} vs {}",
                    terms.reciprocal,
                    same.reciprocal
                );
                assert!((terms.self_energy - same.self_energy).abs() < 1e-9);
                assert!((terms.background - same.background).abs() < 1e-9);
                assert_eq!(terms.excluded_pairs, 0.);
                // Coulomb energy is homogeneous of degree -1 in a uniform
                // scaling, so the complete virial equals the energy.
                let virial = direct_virial + long_range.virial;
                assert!(
                    (virial - energy).abs() < 20. * energy_tolerance,
                    "order {order} net {net}: virial {virial} vs energy {energy}"
                );
            }
        }
    }

    #[test]
    fn gradients_match_finite_differences_of_the_energy() {
        let box_vec = BoxVectors::new(BOX[0], BOX[1], BOX[2]).unwrap();
        let (charges, mut coords) = random_system(0x1234_5678_9abc_def1, 12, 0.4);
        // One excluded pair almost on top of each other: the r -> 0 series.
        coords[6] = v(coords[5].x + 2e-3, coords[5].y - 1e-3, coords[5].z + 1e-3);
        let mut excluded = BONDED.to_vec();
        excluded.push((5, 6));
        for order in MIN_ORDER..=MAX_ORDER {
            let parameters = PmeParameters::new(0.45, [16, 18, 15], order).unwrap();
            let mut engine = PmeEngine::from_charges(&charges, &excluded, parameters).unwrap();
            let mut gradients = zeros(charges.len());
            engine
                .evaluate_into(&coords, &box_vec, &mut gradients, true)
                .unwrap();
            let mut energy = |coords: &[Vec3]| {
                let mut scratch = zeros(coords.len());
                engine
                    .evaluate_into(coords, &box_vec, &mut scratch, true)
                    .unwrap()
                    .energy
            };
            let h = 1e-4;
            for atom in 0..charges.len() {
                for axis in 0..3 {
                    let mut plus = coords.clone();
                    let mut minus = coords.clone();
                    for (moved, sign) in [(&mut plus, 1.), (&mut minus, -1.)] {
                        match axis {
                            0 => moved[atom].x += sign * h,
                            1 => moved[atom].y += sign * h,
                            _ => moved[atom].z += sign * h,
                        }
                    }
                    let numeric = (energy(&plus) - energy(&minus)) / (2. * h);
                    let analytic = [gradients[atom].x, gradients[atom].y, gradients[atom].z][axis];
                    assert!(
                        (analytic - numeric).abs() < 2e-6 * analytic.abs().max(1.),
                        "order {order} atom {atom} axis {axis}: {analytic} vs {numeric}"
                    );
                }
            }
            // The gradients do not depend on the observables flag.
            let mut forces_only = zeros(charges.len());
            let silent = engine
                .evaluate_into(&coords, &box_vec, &mut forces_only, false)
                .unwrap();
            assert_eq!(silent, PmeLongRange::default());
            assert_eq!(forces_only, gradients);
        }
    }

    #[test]
    fn virial_matches_the_volume_derivative() {
        // W = -dE/d(ln s) with the box and every atom scaled by s; alpha and
        // the grid dimensions stay fixed, as under a barostat.
        for net in [0., -1.3] {
            let (charges, coords) = random_system(0x0dd0_f00d_cafe_beef, 14, net);
            for order in MIN_ORDER..=MAX_ORDER {
                let parameters = PmeParameters::new(0.5, [20, 24, 18], order).unwrap();
                let mut engine = PmeEngine::from_charges(&charges, &BONDED, parameters).unwrap();
                let mut at = |s: f64| {
                    let box_vec = BoxVectors::new(BOX[0] * s, BOX[1] * s, BOX[2] * s).unwrap();
                    let scaled: Vec<Vec3> = coords
                        .iter()
                        .map(|p| v(p.x * s, p.y * s, p.z * s))
                        .collect();
                    let mut scratch = zeros(coords.len());
                    engine
                        .evaluate_into(&scaled, &box_vec, &mut scratch, true)
                        .unwrap()
                };
                let h = 1e-5;
                let (up, down) = (at(1. + h), at(1. - h));
                let numeric = -(up.energy - down.energy) / ((1. + h).ln() - (1. - h).ln());
                let analytic = at(1.).virial;
                assert!(
                    (analytic - numeric).abs() < 1e-7 * analytic.abs().max(1.),
                    "order {order} net {net}: virial {analytic} vs {numeric}"
                );
            }
        }
    }

    #[test]
    fn energy_is_invariant_under_translation_and_reimaging() {
        let box_vec = BoxVectors::new(BOX[0], BOX[1], BOX[2]).unwrap();
        let (charges, coords) = random_system(0x7f4a_7c15_9e37_79b9, 20, 0.);
        let grid = [48, 54, 45];
        let parameters = PmeParameters::new(0.6, grid, 6).unwrap();
        let mut engine = PmeEngine::from_charges(&charges, &BONDED, parameters).unwrap();
        let mut evaluate = |coords: &[Vec3]| {
            let mut gradients = zeros(coords.len());
            let energy = engine
                .evaluate_into(coords, &box_vec, &mut gradients, true)
                .unwrap()
                .energy;
            (energy, gradients)
        };
        let close = |a: &[Vec3], b: &[Vec3]| {
            a.iter().zip(b).all(|(g, w)| {
                (g.x - w.x).abs() < 1e-9 && (g.y - w.y).abs() < 1e-9 && (g.z - w.z).abs() < 1e-9
            })
        };
        let (base, base_gradients) = evaluate(&coords);
        let shifted = |shift: [f64; 3]| -> Vec<Vec3> {
            coords
                .iter()
                .map(|p| v(p.x + shift[0], p.y + shift[1], p.z + shift[2]))
                .collect()
        };
        // Whole grid spacings and whole box vectors map the mesh onto
        // itself: exact up to roundoff.
        for shift in [
            [3. * BOX[0] / grid[0] as f64, 0., 0.],
            [
                0.,
                -7. * BOX[1] / grid[1] as f64,
                11. * BOX[2] / grid[2] as f64,
            ],
            [-2. * BOX[0], 5. * BOX[1], BOX[2]],
        ] {
            let (energy, gradients) = evaluate(&shifted(shift));
            assert!(
                (energy - base).abs() < 1e-9,
                "{shift:?}: {energy} vs {base}"
            );
            assert!(close(&gradients, &base_gradients), "{shift:?}");
        }
        // An arbitrary translation moves the atoms relative to the mesh:
        // invariant to the interpolation accuracy only.
        let (energy, _) = evaluate(&shifted([0.123, -4.56, 7.891]));
        assert!((energy - base).abs() < 1e-4, "{energy} vs {base}");
        assert!(
            (energy - base).abs() > 1e-12,
            "mesh error should be visible"
        );
        // Re-imaging one atom without excluded partners, or a bonded group
        // as a whole, by box vectors changes nothing.
        let mut reimaged = coords.clone();
        reimaged[9].x += BOX[0];
        reimaged[10].y -= 3. * BOX[1];
        reimaged[11].z += 2. * BOX[2];
        for atom in [2, 3, 4] {
            reimaged[atom].x -= BOX[0];
            reimaged[atom].z += BOX[2];
        }
        let (energy, gradients) = evaluate(&reimaged);
        assert!((energy - base).abs() < 1e-9, "{energy} vs {base}");
        assert!(close(&gradients, &base_gradients));
    }

    #[test]
    fn results_do_not_depend_on_the_thread_count() {
        let box_vec = BoxVectors::new(BOX[0], BOX[1], BOX[2]).unwrap();
        let (charges, coords) = random_system(0x5851_f42d_4c95_7f2d, 700, 0.3);
        // More excluded pairs than one reduction chunk holds.
        let excluded: Vec<(usize, usize)> = (0..699).map(|atom| (atom, atom + 1)).collect();
        let parameters = PmeParameters::new(0.5, [25, 27, 21], 5).unwrap();
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut engine = PmeEngine::from_charges(&charges, &excluded, parameters).unwrap();
                let mut gradients = zeros(charges.len());
                let first = engine
                    .evaluate_into(&coords, &box_vec, &mut gradients, true)
                    .unwrap();
                // Work buffers carry nothing over between calls.
                let mut again = zeros(charges.len());
                let second = engine
                    .evaluate_into(&coords, &box_vec, &mut again, true)
                    .unwrap();
                assert_eq!(first, second);
                assert_eq!(gradients, again);
                (first, gradients)
            })
        };
        let (single, single_gradients) = run(1);
        for threads in [2, 5] {
            let (many, many_gradients) = run(threads);
            assert_eq!(single.energy.to_bits(), many.energy.to_bits());
            assert_eq!(single.virial.to_bits(), many.virial.to_bits());
            for (a, b) in single_gradients.iter().zip(&many_gradients) {
                assert_eq!(
                    [a.x.to_bits(), a.y.to_bits(), a.z.to_bits()],
                    [b.x.to_bits(), b.y.to_bits(), b.z.to_bits()]
                );
            }
        }
    }

    #[test]
    fn rejects_mismatched_input() {
        let parameters = PmeParameters::new(0.4, [16, 16, 16], 4).unwrap();
        assert!(PmeEngine::from_charges(&[], &[], parameters).is_err());
        assert!(PmeEngine::from_charges(&[1., -1.], &[(0, 2)], parameters).is_err());
        assert!(PmeEngine::from_charges(&[1., f64::NAN], &[], parameters).is_err());
        let mut engine =
            PmeEngine::from_charges(&[1., -1.], &[(1, 0), (0, 1), (1, 1)], parameters).unwrap();
        assert_eq!(engine.excluded_pair_count(), 1);
        let box_vec = BoxVectors::new(10., 10., 10.).unwrap();
        let mut gradients = zeros(2);
        assert!(
            engine
                .evaluate_into(&[v(1., 1., 1.)], &box_vec, &mut gradients, true)
                .is_err()
        );
        let broken = [v(1., 1., 1.), v(f64::NAN, 0., 0.)];
        assert!(
            engine
                .evaluate_into(&broken, &box_vec, &mut gradients, true)
                .is_err()
        );
    }

    struct NoElectrostatics;

    impl ElectrostaticsBackend for NoElectrostatics {
        fn name(&self) -> &'static str {
            "none"
        }

        fn pair(&self, _r: f64, _qq: f64) -> (f64, f64) {
            (0., 0.)
        }
    }

    fn solvated_dipeptide() -> ParameterizedSystem {
        let pdb = include_str!("../../../tests/fixtures/dipeptide.pdb");
        let options = glysys::BuildOptions {
            add_water: true,
            add_ions: false,
            padding_angstrom: 6.0,
            ..Default::default()
        };
        glysys::SystemBuilder::new(options)
            .unwrap()
            .prepare_pdb_str(pdb)
            .unwrap()
    }

    #[test]
    fn water_box_with_exclusions_matches_brute_force_ewald() {
        // 135 rigid waters around a dipeptide: every water contributes three
        // excluded intramolecular pairs, the solute 1-2, 1-3 and 1-4 pairs.
        let system = solvated_dipeptide();
        let box_vec = BoxVectors::from_system(&system).unwrap();
        let coords = system.coordinates();
        let charges: Vec<f64> = system.atoms().iter().map(|atom| atom.charge()).collect();
        let cutoff = 7.0;
        let parameters = PmeParameters::new(0.66, [96, 90, 80], 6).unwrap();
        let field = PbcForceField::new(&system, vec![]).unwrap();
        let mut engine = PmeEngine::new(&system, parameters).unwrap();
        let excluded: usize = system.exclusions().iter().map(|set| set.len()).sum();
        assert_eq!(engine.excluded_pair_count(), excluded / 2);
        let wrapped: Vec<Vec3> = coords.iter().map(|p| box_vec.wrap(*p)).collect();
        let list = PbcNeighborList::build(&wrapped, &box_vec, cutoff, 0.).unwrap();
        let full = field
            .evaluate_pme(&mut engine, &coords, &box_vec, &list.pairs, cutoff)
            .unwrap();
        // Everything except the regular-pair electrostatics, which leaves
        // the plain 1-4 Coulomb terms in both evaluations.
        let rest = field
            .evaluate(&coords, &box_vec, &list.pairs, &NoElectrostatics, cutoff)
            .unwrap();
        let energy = full.components.electrostatics - rest.components.electrostatics;

        // Oracle: Ewald over all pairs, minus the bare Coulomb interaction
        // of each excluded pair at its (unwrapped) bonded separation.
        let reference = ewald_reference(&charges, &coords, box_vec.as_array(), 0.42);
        let mut want_energy = reference.energy();
        let mut want = reference.gradients.clone();
        for (a, set) in system.exclusions().iter().enumerate() {
            for &b in set.iter().filter(|&&b| b > a) {
                let d = [
                    coords[a].x - coords[b].x,
                    coords[a].y - coords[b].y,
                    coords[a].z - coords[b].z,
                ];
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let qq = COULOMB * charges[a] * charges[b];
                want_energy -= qq / r;
                for axis in 0..3 {
                    want[a][axis] += qq * d[axis] / (r * r * r);
                    want[b][axis] -= qq * d[axis] / (r * r * r);
                }
            }
        }
        let force_rms = rms(want.iter().flatten().copied());
        let error = rms(full
            .gradients
            .iter()
            .zip(&rest.gradients)
            .zip(&want)
            .flat_map(|((g, r), w)| [g.x - r.x - w[0], g.y - r.y - w[1], g.z - r.z - w[2]]));
        assert!(
            (energy - want_energy).abs() < 1e-3,
            "electrostatic energy {energy} vs {want_energy}"
        );
        // The exclusion correction is a large part of the answer here.
        assert!(engine.last_terms().excluded_pairs.abs() > 1e3);
        assert!(
            error < 2e-6 * force_rms,
            "force error {error} of rms {force_rms}"
        );
    }

    #[test]
    fn complete_hamiltonian_virial_matches_volume_derivative() {
        // The atomic virial of bonded terms, Lennard-Jones, direct space and
        // long-range PME against -dU/d(ln s) with every atom scaled affinely.
        // alpha is large enough that the erfc tail at the cutoff (2e-13)
        // cannot show up in the finite difference.
        let system = solvated_dipeptide();
        let field = PbcForceField::new(&system, vec![]).unwrap();
        let box_vec = BoxVectors::from_system(&system).unwrap();
        let coords = system.coordinates();
        let cutoff = 4.0;
        let parameters = PmeParameters::new(1.3, [36, 32, 30], 5).unwrap();
        let mut engine = PmeEngine::new(&system, parameters).unwrap();
        let mut at = |s: f64| {
            let scaled_box = BoxVectors::new(box_vec.x * s, box_vec.y * s, box_vec.z * s).unwrap();
            let scaled: Vec<Vec3> = coords
                .iter()
                .map(|p| v(p.x * s, p.y * s, p.z * s))
                .collect();
            let wrapped: Vec<Vec3> = scaled.iter().map(|p| scaled_box.wrap(*p)).collect();
            let list = PbcNeighborList::build(&wrapped, &scaled_box, cutoff, 1.5).unwrap();
            field
                .evaluate_pme(&mut engine, &scaled, &scaled_box, &list.pairs, cutoff)
                .unwrap()
        };
        // Small enough that no hard-cutoff Lennard-Jones pair enters or
        // leaves, as in the reaction-field virial test.
        let h = 1.0e-6;
        let (up, down, base) = (at(1. + h), at(1. - h), at(1.));
        let numeric =
            -(up.components.total() - down.components.total()) / ((1. + h).ln() - (1. - h).ln());
        assert!(
            (base.virial - numeric).abs() < 1e-3 * numeric.abs().max(1.),
            "virial {} vs volume derivative {numeric}",
            base.virial
        );
        assert!((base.virial - base.virial_terms.iter().sum::<f64>()).abs() < 1e-9);
        let pair_split = base.virial_pair_split[0] + base.virial_pair_split[1];
        assert!((pair_split - base.virial_terms[3]).abs() < 1e-6 * pair_split.abs().max(1.));
    }
}
