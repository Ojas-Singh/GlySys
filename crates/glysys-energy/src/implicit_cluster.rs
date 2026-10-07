//! Single-precision implicit-solvent forces for dynamics integration steps.
//!
//! Evaluates the same no-cutoff Lennard-Jones/Coulomb and OBC2 (GB plus ACE
//! surface) gradients as [`crate::EnergyEvaluator`] in f32 with the OpenMM
//! CPU-platform precision model; bonded terms, energies and the reference
//! evaluator remain f64. Every pass walks full rows for blocks of eight atoms
//! (one atom per SIMD lane, each partner broadcast), so each atom owns its
//! sums: there is no scatter, and results do not depend on the thread count.
//! Transcendentals use fixed polynomials, so the AVX2 and portable builds of
//! the one kernel source produce identical bits.
use crate::{COULOMB_KCAL_ANGSTROM, EnergyError, Obc2Options, Result};
use glysys::{ParameterizedSystem, Vec3};
use rayon::prelude::*;
use std::collections::BTreeMap;

const LANES: usize = 8;

#[derive(Clone, Copy)]
struct Exception {
    a: usize,
    b: usize,
    qq: f64,
    epsilon: f64,
    sigma: f64,
}

/// Reusable implicit-solvent force engine for one prepared system.
#[derive(Clone)]
pub struct ImplicitPairEngine {
    n: usize,
    padded: usize,
    options: Obc2Options,
    coulomb_dielectric: f64,
    charge: Vec<f32>,
    sigma: Vec<f32>,
    sqrt_epsilon: Vec<f32>,
    gb_radius: Vec<f32>,
    descreen: Vec<f32>,
    scaled: Vec<f32>,
    window: Vec<u64>,
    /// Per lane block, partners excluded beyond the index window, ascending,
    /// with the mask of lanes they are excluded from.
    far_rows: Vec<Vec<(u32, u8)>>,
    exceptions: Vec<Exception>,
}

/// Per-atom f32 inputs of one evaluation, padded to whole lane blocks.
struct Frame {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
}

impl ImplicitPairEngine {
    /// `dielectric` is the Coulomb dielectric of [`crate::EnergyOptions`].
    pub fn new(
        system: &ParameterizedSystem,
        options: Obc2Options,
        dielectric: f64,
    ) -> Result<Self> {
        let n = system.atom_count();
        if n == 0 || n >= (u32::MAX as usize) / 2 {
            return Err(EnergyError::InvalidConfiguration(
                "implicit engine needs between one and 2^31 atoms".into(),
            ));
        }
        let padded = n.div_ceil(LANES) * LANES;
        let mut window = vec![0u64; n];
        let mut far = Vec::new();
        let mut mark = |atom: usize, other: usize, window: &mut [u64]| {
            let offset = other as i64 - atom as i64 + 32;
            if (0..64).contains(&offset) {
                window[atom] |= 1 << offset;
            } else {
                far.push((atom as u32, other as u32));
            }
        };
        for (atom, set) in system.exclusions().iter().enumerate() {
            for &other in set {
                mark(atom, other, &mut window);
                mark(other, atom, &mut window);
            }
        }
        let mut one_four = BTreeMap::new();
        for (pair, scee, scnb) in system.one_four_pairs() {
            one_four.insert((pair[0].min(pair[1]), pair[0].max(pair[1])), (scee, scnb));
        }
        let atoms = system.atoms();
        let mut exceptions = Vec::with_capacity(one_four.len());
        for (&(a, b), &(scee, scnb)) in &one_four {
            mark(a, b, &mut window);
            mark(b, a, &mut window);
            exceptions.push(Exception {
                a,
                b,
                qq: COULOMB_KCAL_ANGSTROM * atoms[a].charge() * atoms[b].charge()
                    / (dielectric * scee),
                epsilon: (atoms[a].lennard_jones_epsilon() * atoms[b].lennard_jones_epsilon())
                    .sqrt()
                    / scnb,
                sigma: atoms[a].lennard_jones_radius() + atoms[b].lennard_jones_radius(),
            });
        }
        let mut far_masks = BTreeMap::new();
        for &(atom, other) in &far {
            *far_masks
                .entry((atom as usize / LANES, other))
                .or_insert(0u8) |= 1 << (atom as usize % LANES);
        }
        let mut far_rows = vec![Vec::new(); padded / LANES];
        for ((block, other), mask) in far_masks {
            far_rows[block].push((other, mask));
        }
        let pad = |values: Vec<f32>, fill: f32| {
            let mut values = values;
            values.resize(padded, fill);
            values
        };
        let gb: Vec<f64> = atoms.iter().map(|a| a.gb_radius()).collect();
        Ok(Self {
            n,
            padded,
            options,
            coulomb_dielectric: dielectric,
            charge: pad(atoms.iter().map(|a| a.charge() as f32).collect(), 0.0),
            sigma: pad(
                atoms
                    .iter()
                    .map(|a| a.lennard_jones_radius() as f32)
                    .collect(),
                0.0,
            ),
            sqrt_epsilon: pad(
                atoms
                    .iter()
                    .map(|a| a.lennard_jones_epsilon().max(0.).sqrt() as f32)
                    .collect(),
                0.0,
            ),
            gb_radius: pad(gb.iter().map(|&r| r as f32).collect(), 1.0),
            descreen: pad(
                gb.iter().map(|&r| (r - 0.09).max(0.1) as f32).collect(),
                0.1,
            ),
            scaled: pad(
                gb.iter()
                    .zip(atoms)
                    .map(|(&r, a)| ((r - 0.09).max(0.1) * a.gb_screen()) as f32)
                    .collect(),
                0.0,
            ),
            window,
            far_rows,
            exceptions,
        })
    }

    /// Add the nonbonded and OBC2 gradients at `coordinates` to `gradients`.
    pub fn gradient_into(&self, coordinates: &[Vec3], gradients: &mut [Vec3]) -> Result<()> {
        if coordinates.len() != self.n || gradients.len() != self.n {
            return Err(EnergyError::CoordinateCount {
                expected: self.n,
                received: coordinates.len(),
            });
        }
        // f32 positions relative to the centroid keep differences precise.
        let center = coordinates.iter().fold([0.0f64; 3], |acc, p| {
            [acc[0] + p.x, acc[1] + p.y, acc[2] + p.z]
        });
        let inv = 1.0 / self.n as f64;
        let center = center.map(|v| v * inv);
        let mut frame = Frame {
            x: vec![1e6; self.padded],
            y: vec![1e6; self.padded],
            z: vec![1e6; self.padded],
        };
        for (atom, p) in coordinates.iter().enumerate() {
            frame.x[atom] = (p.x - center[0]) as f32;
            frame.y[atom] = (p.y - center[1]) as f32;
            frame.z[atom] = (p.z - center[2]) as f32;
        }
        let blocks = self.padded / LANES;
        let simd = simd_available();
        // Pass 1: Born integrals, radii and their derivative.
        let born_rows: Vec<[[f32; LANES]; 2]> = (0..blocks)
            .into_par_iter()
            .map(|block| self.born_block(&frame, block, simd))
            .collect();
        let mut born = vec![1.0f32; self.padded];
        let mut dbdi = vec![0.0f32; self.padded];
        for (block, rows) in born_rows.iter().enumerate() {
            born[block * LANES..(block + 1) * LANES].copy_from_slice(&rows[0]);
            dbdi[block * LANES..(block + 1) * LANES].copy_from_slice(&rows[1]);
        }
        // Pass 2: GB direct forces and dE/dB, plus LJ/Coulomb.
        let polar_rows: Vec<([[f32; LANES]; 3], [f32; LANES])> = (0..blocks)
            .into_par_iter()
            .map(|block| self.polar_block(&frame, &born, block, simd))
            .collect();
        let mut chain = vec![0.0f32; self.padded];
        for (block, (_, de_db)) in polar_rows.iter().enumerate() {
            for (lane, &value) in de_db.iter().enumerate() {
                let atom = block * LANES + lane;
                chain[atom] = value * dbdi[atom];
            }
        }
        // Pass 3: Born-radius chain rule.
        let chain_rows: Vec<[[f32; LANES]; 3]> = (0..blocks)
            .into_par_iter()
            .map(|block| self.chain_row(&frame, &chain, block, simd))
            .collect();
        for block in 0..blocks {
            for lane in 0..LANES {
                let atom = block * LANES + lane;
                if atom >= self.n {
                    break;
                }
                let (polar, _) = &polar_rows[block];
                let g = &mut gradients[atom];
                g.x += f64::from(polar[0][lane]) + f64::from(chain_rows[block][0][lane]);
                g.y += f64::from(polar[1][lane]) + f64::from(chain_rows[block][1][lane]);
                g.z += f64::from(polar[2][lane]) + f64::from(chain_rows[block][2][lane]);
            }
        }
        // Scaled 1-4 pairs (plain Coulomb with the solute dielectric) in f64.
        for e in &self.exceptions {
            let d = Vec3 {
                x: coordinates[e.a].x - coordinates[e.b].x,
                y: coordinates[e.a].y - coordinates[e.b].y,
                z: coordinates[e.a].z - coordinates[e.b].z,
            };
            let r = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt().max(1e-8);
            let ratio6 = (e.sigma / r).powi(6);
            // dE/dr divided by r, as in the evaluator's pair loop.
            let derivative =
                (12.0 * e.epsilon * (ratio6 - ratio6 * ratio6) / r - e.qq / (r * r)) / r;
            gradients[e.a].x += derivative * d.x;
            gradients[e.a].y += derivative * d.y;
            gradients[e.a].z += derivative * d.z;
            gradients[e.b].x -= derivative * d.x;
            gradients[e.b].y -= derivative * d.y;
            gradients[e.b].z -= derivative * d.z;
        }
        if gradients
            .iter()
            .any(|g| !g.x.is_finite() || !g.y.is_finite() || !g.z.is_finite())
        {
            return Err(EnergyError::InvalidConfiguration(
                "nonfinite implicit-solvent gradient".into(),
            ));
        }
        Ok(())
    }

    #[inline(always)]
    fn lane_atoms(block: usize) -> [usize; LANES] {
        std::array::from_fn(|lane| block * LANES + lane)
    }

    fn born_block(&self, frame: &Frame, block: usize, simd: bool) -> [[f32; LANES]; 2] {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `simd` is only set when AVX2 support was detected at runtime.
        let sum = if simd {
            unsafe { avx2::born_sums(self, frame, block) }
        } else {
            self.born_sums(frame, block)
        };
        #[cfg(not(target_arch = "x86_64"))]
        let sum = {
            let _ = simd;
            self.born_sums(frame, block)
        };
        let mut born = [1.0f32; LANES];
        let mut derivative = [0.0f32; LANES];
        for lane in 0..LANES {
            let atom = block * LANES + lane;
            if atom >= self.n {
                continue;
            }
            let r = f64::from(self.descreen[atom]);
            let rho = f64::from(self.gb_radius[atom]);
            let psi = r * f64::from(sum[lane]);
            let t = (psi - 0.8 * psi * psi + 4.85 * psi.powi(3)).tanh();
            let denominator = 1.0 / r - t / rho;
            let b = 1.0 / denominator.max(1.0e-6);
            born[lane] = b as f32;
            derivative[lane] = if denominator >= 1.0e-6 {
                (b * b * (1.0 - t * t) * (1.0 - 1.6 * psi + 14.55 * psi * psi) * r / rho) as f32
            } else {
                0.0
            };
        }
        [born, derivative]
    }

    /// Born integral sums of one block (portable form of [`avx2::born_sums`]).
    #[inline(always)]
    fn born_sums(&self, frame: &Frame, block: usize) -> [f32; LANES] {
        let atoms = Self::lane_atoms(block);
        let xi = atoms.map(|a| frame.x[a]);
        let yi = atoms.map(|a| frame.y[a]);
        let zi = atoms.map(|a| frame.z[a]);
        let ri = atoms.map(|a| self.descreen[a]);
        let mut sum = [0.0f32; LANES];
        for j in 0..self.n {
            let (xj, yj, zj, sj) = (frame.x[j], frame.y[j], frame.z[j], self.scaled[j]);
            for lane in 0..LANES {
                let dx = xi[lane] - xj;
                let dy = yi[lane] - yj;
                let dz = zi[lane] - zj;
                let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-8);
                let own = if atoms[lane] == j { 0.0 } else { 1.0 };
                sum[lane] += own * radial_value(ri[lane], sj, d, 1.0 / d);
            }
        }
        sum
    }

    /// Lennard-Jones/Coulomb lane mask for partner `j`, or `None` when no
    /// lane of the block is excluded from it. Exclusions and 1-4 pairs sit
    /// within each atom's index window or in the block's far row, which
    /// `cursor` walks as `j` ascends.
    #[inline(always)]
    fn pair_keep(&self, block: usize, j: usize, cursor: &mut usize) -> Option<[f32; LANES]> {
        let first = block * LANES;
        let mut excluded = 0u8;
        if j + 32 >= first && j < first + LANES + 32 {
            for lane in 0..LANES {
                let atom = first + lane;
                let offset = (j + 32).wrapping_sub(atom);
                if atom >= self.n
                    || atom == j
                    || (offset < 64 && (self.window[atom] >> offset) & 1 != 0)
                {
                    excluded |= 1 << lane;
                }
            }
        }
        let far = &self.far_rows[block];
        if let Some(&(partner, mask)) = far.get(*cursor)
            && partner as usize == j
        {
            excluded |= mask;
            *cursor += 1;
        }
        if excluded == 0 {
            return None;
        }
        Some(std::array::from_fn(|lane| {
            if excluded & (1 << lane) != 0 || first + lane >= self.n {
                0.0
            } else {
                1.0
            }
        }))
    }

    fn polar_block(
        &self,
        frame: &Frame,
        born: &[f32],
        block: usize,
        simd: bool,
    ) -> ([[f32; LANES]; 3], [f32; LANES]) {
        let atoms = Self::lane_atoms(block);
        let dielectric =
            (1.0 / self.options.solute_dielectric - 1.0 / self.options.solvent_dielectric) as f32;
        let gb_charge =
            atoms.map(|a| -(COULOMB_KCAL_ANGSTROM as f32) * dielectric * self.charge[a]);
        let coulomb_charge = atoms
            .map(|a| (COULOMB_KCAL_ANGSTROM / self.coulomb_dielectric) as f32 * self.charge[a]);
        #[cfg(target_arch = "x86_64")]
        let (gradient, mut de_db) = if simd {
            // SAFETY: `simd` is only set when AVX2 support was detected at runtime.
            unsafe { avx2::polar_sums(self, frame, born, block, &gb_charge, &coulomb_charge) }
        } else {
            self.polar_sums(frame, born, block, &gb_charge, &coulomb_charge)
        };
        #[cfg(not(target_arch = "x86_64"))]
        let (gradient, mut de_db) = {
            let _ = simd;
            self.polar_sums(frame, born, block, &gb_charge, &coulomb_charge)
        };
        for lane in 0..LANES {
            let atom = atoms[lane];
            if atom >= self.n {
                continue;
            }
            let rho = f64::from(self.gb_radius[atom]);
            let b = f64::from(born[atom]);
            let surface = 4.0
                * std::f64::consts::PI
                * self.options.surface_tension
                * (rho + self.options.probe_radius).powi(2)
                * (rho / b).powi(6);
            de_db[lane] -= (6.0 * surface / b) as f32;
        }
        (gradient, de_db)
    }

    /// GB direct and LJ/Coulomb gradient rows plus dE/dB of one block
    /// (portable form of [`avx2::polar_sums`]).
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    fn polar_sums(
        &self,
        frame: &Frame,
        born: &[f32],
        block: usize,
        gb_charge: &[f32; LANES],
        coulomb_charge: &[f32; LANES],
    ) -> ([[f32; LANES]; 3], [f32; LANES]) {
        let atoms = Self::lane_atoms(block);
        let xi = atoms.map(|a| frame.x[a]);
        let yi = atoms.map(|a| frame.y[a]);
        let zi = atoms.map(|a| frame.z[a]);
        let bi = atoms.map(|a| born[a]);
        let si = atoms.map(|a| self.sigma[a]);
        let ei = atoms.map(|a| self.sqrt_epsilon[a]);
        let valid: [f32; LANES] = atoms.map(|a| if a < self.n { 1.0 } else { 0.0 });
        let mut gx = [0.0f32; LANES];
        let mut gy = [0.0f32; LANES];
        let mut gz = [0.0f32; LANES];
        let mut de_db = [0.0f32; LANES];
        let mut cursor = 0;
        for j in 0..self.n {
            let (xj, yj, zj) = (frame.x[j], frame.y[j], frame.z[j]);
            let (bj, qj, sj, ej) = (born[j], self.charge[j], self.sigma[j], self.sqrt_epsilon[j]);
            let nonbonded = self.pair_keep(block, j, &mut cursor).unwrap_or(valid);
            for lane in 0..LANES {
                let dx = xi[lane] - xj;
                let dy = yi[lane] - yj;
                let dz = zi[lane] - zj;
                let r2 = dx * dx + dy * dy + dz * dz;
                let own = atoms[lane] == j;
                let p = bi[lane] * bj;
                let t = r2 / (4.0 * p);
                let e = exp_f32(-t);
                let f = (r2 + p * e).sqrt().max(1e-8);
                let coefficient = gb_charge[lane] * qj * if own { 0.5 } else { 1.0 };
                let de_dz = -0.5 * coefficient / (f * f * f);
                let de_dp = de_dz * e * (1.0 + t);
                de_db[lane] += de_dp * bj * if own { 2.0 } else { 1.0 };
                let direct = if own {
                    0.0
                } else {
                    2.0 * de_dz * (1.0 - 0.25 * e)
                };
                // Lennard-Jones and Coulomb (excluded and 1-4 lanes masked).
                let keep = nonbonded[lane];
                let radius = (r2.sqrt()).max(1e-8);
                let inv = keep / radius;
                let ratio = (si[lane] + sj) * inv;
                let ratio2 = ratio * ratio;
                let ratio6 = ratio2 * ratio2 * ratio2;
                let coulomb = coulomb_charge[lane] * qj * inv;
                let pair =
                    (12.0 * ei[lane] * ej * (ratio6 - ratio6 * ratio6) - coulomb) * inv * inv;
                let factor = direct + pair;
                gx[lane] += factor * dx;
                gy[lane] += factor * dy;
                gz[lane] += factor * dz;
            }
        }
        ([gx, gy, gz], de_db)
    }

    fn chain_row(
        &self,
        frame: &Frame,
        chain: &[f32],
        block: usize,
        simd: bool,
    ) -> [[f32; LANES]; 3] {
        #[cfg(target_arch = "x86_64")]
        if simd {
            // SAFETY: `simd` is only set when AVX2 support was detected at runtime.
            return unsafe { avx2::chain_sums(self, frame, chain, block) };
        }
        let _ = simd;
        self.chain_sums(frame, chain, block)
    }

    /// Born-radius chain-rule gradient rows of one block (portable form of
    /// [`avx2::chain_sums`]).
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    fn chain_sums(&self, frame: &Frame, chain: &[f32], block: usize) -> [[f32; LANES]; 3] {
        let atoms = Self::lane_atoms(block);
        let xi = atoms.map(|a| frame.x[a]);
        let yi = atoms.map(|a| frame.y[a]);
        let zi = atoms.map(|a| frame.z[a]);
        let ri = atoms.map(|a| self.descreen[a]);
        let si = atoms.map(|a| self.scaled[a]);
        let ci = atoms.map(|a| chain[a]);
        let mut gx = [0.0f32; LANES];
        let mut gy = [0.0f32; LANES];
        let mut gz = [0.0f32; LANES];
        for j in 0..self.n {
            let (xj, yj, zj) = (frame.x[j], frame.y[j], frame.z[j]);
            let (rj, sj, cj) = (self.descreen[j], self.scaled[j], chain[j]);
            for lane in 0..LANES {
                let dx = xi[lane] - xj;
                let dy = yi[lane] - yj;
                let dz = zi[lane] - zj;
                let raw = (dx * dx + dy * dy + dz * dz).sqrt();
                let d = raw.max(1e-8);
                let inv_d = 1.0 / d;
                let keep = if atoms[lane] == j || raw < 1e-8 {
                    0.0
                } else {
                    1.0
                };
                let factor = (ci[lane] * radial_derivative(ri[lane], sj, d, inv_d)
                    + cj * radial_derivative(rj, si[lane], d, inv_d))
                    * (keep * inv_d);
                gx[lane] += factor * dx;
                gy[lane] += factor * dy;
                gz[lane] += factor * dz;
            }
        }
        [gx, gy, gz]
    }
}

/// Whether the explicit AVX2 kernels can run on this processor.
fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Cephes `logf` (about 1 ulp for positive finite inputs), branch-free.
#[inline(always)]
#[allow(clippy::excessive_precision)]
fn ln_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let mut exponent = ((bits >> 23) & 0xff) as i32 - 126;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000);
    let small = m < std::f32::consts::FRAC_1_SQRT_2;
    exponent -= i32::from(small);
    m = if small { m + m - 1.0 } else { m - 1.0 };
    let z = m * m;
    let mut y = 7.037_683_6e-2f32;
    y = y * m - 1.151_461e-1;
    y = y * m + 1.167_699_9e-1;
    y = y * m - 1.242_014_1e-1;
    y = y * m + 1.424_932_3e-1;
    y = y * m - 1.666_805_8e-1;
    y = y * m + 2.000_071_5e-1;
    y = y * m - 2.499_999_4e-1;
    y = y * m + 3.333_333_1e-1;
    y = y * m * z;
    let e = exponent as f32;
    y += -2.121_944_4e-4 * e;
    y += -0.5 * z;
    let mut result = m + y;
    result += 0.693_359_4 * e;
    result
}

/// Cephes `expf` (about 1 ulp), branch-free, for arguments in [-87, 88].
#[inline(always)]
#[allow(clippy::excessive_precision)]
fn exp_f32(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let fx = (x * std::f32::consts::LOG2_E + 0.5).floor();
    let mut r = x - fx * 0.693_359_4;
    r -= fx * -2.121_944_4e-4;
    let z = r * r;
    let mut y = 1.987_569_1e-4f32;
    y = y * r + 1.398_199_9e-3;
    y = y * r + 8.333_452e-3;
    y = y * r + 4.166_579_6e-2;
    y = y * r + 1.666_666_5e-1;
    y = y * r + 5.000_000_1e-1;
    y = y * z + r + 1.0;
    let scale = f32::from_bits((((fx as i32) + 127) as u32) << 23);
    y * scale
}

#[inline(always)]
/// OBC pair integral of a sphere of radius `s` at distance `d` (with
/// `inv_d = 1 / d`) over the region outside radius `r`.
fn radial_value(r: f32, s: f32, d: f32, inv_d: f32) -> f32 {
    let candidate = (d - s).abs();
    let l = r.max(candidate);
    let u = d + s;
    let active = u > r && l < u;
    let lu = if active { l } else { 1.0 };
    let uu = if active { u } else { 2.0 };
    let a = 1.0 / lu;
    let b = 1.0 / uu;
    let c = d - s * (s * inv_d);
    let q = b * b - a * a;
    let value = 0.5 * (a - b + 0.25 * c * q + 0.5 * ln_f32(lu * b) * inv_d);
    if active { value } else { 0.0 }
}

#[inline(always)]
/// d/dd of [`radial_value`].
fn radial_derivative(r: f32, s: f32, d: f32, inv_d: f32) -> f32 {
    let candidate = (d - s).abs();
    let l = r.max(candidate);
    let u = d + s;
    let active = u > r && l < u;
    let lu = if active { l } else { 1.0 };
    let uu = if active { u } else { 2.0 };
    let dl = if candidate < r {
        0.0
    } else if d < s {
        -1.0
    } else {
        1.0
    };
    let a = 1.0 / lu;
    let b = 1.0 / uu;
    let sd = s * inv_d;
    let c = d - s * sd;
    let q = b * b - a * a;
    let log = ln_f32(lu * b);
    let derivative = 0.5
        * (-dl * a * a
            + b * b
            + 0.25 * ((1.0 + sd * sd) * q + c * (-2.0 * b * b * b + 2.0 * dl * a * a * a))
            + 0.5 * ((dl * a - b) * inv_d - log * inv_d * inv_d));
    if active { derivative } else { 0.0 }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    //! Explicit AVX2 forms of the three implicit passes. Every lane performs
    //! the portable kernels' IEEE operations in the same order (no fused
    //! multiply-add, the same polynomial transcendentals), so both builds
    //! produce identical bits.
    use super::{Frame, ImplicitPairEngine, LANES};
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    fn splat(v: f32) -> __m256 {
        _mm256_set1_ps(v)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn select(mask: __m256, yes: __m256, no: __m256) -> __m256 {
        _mm256_blendv_ps(no, yes, mask)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn load(values: &[f32; LANES]) -> __m256 {
        // SAFETY: `values` addresses eight contiguous floats.
        unsafe { _mm256_loadu_ps(values.as_ptr()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn store(v: __m256) -> [f32; LANES] {
        let mut out = [0f32; LANES];
        // SAFETY: `out` addresses eight contiguous floats.
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), v) };
        out
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::excessive_precision)]
    fn ln(x: __m256) -> __m256 {
        let bits = _mm256_castps_si256(x);
        let mut exponent = _mm256_sub_epi32(
            _mm256_and_si256(_mm256_srli_epi32::<23>(bits), _mm256_set1_epi32(0xff)),
            _mm256_set1_epi32(126),
        );
        let mut m = _mm256_castsi256_ps(_mm256_or_si256(
            _mm256_and_si256(bits, _mm256_set1_epi32(0x007f_ffff)),
            _mm256_set1_epi32(0x3f00_0000),
        ));
        let small = _mm256_cmp_ps::<_CMP_LT_OQ>(m, splat(std::f32::consts::FRAC_1_SQRT_2));
        exponent = _mm256_add_epi32(exponent, _mm256_castps_si256(small));
        m = select(
            small,
            _mm256_sub_ps(_mm256_add_ps(m, m), splat(1.0)),
            _mm256_sub_ps(m, splat(1.0)),
        );
        let z = _mm256_mul_ps(m, m);
        let mut y = splat(7.037_683_6e-2);
        y = _mm256_sub_ps(_mm256_mul_ps(y, m), splat(1.151_461e-1));
        y = _mm256_add_ps(_mm256_mul_ps(y, m), splat(1.167_699_9e-1));
        y = _mm256_sub_ps(_mm256_mul_ps(y, m), splat(1.242_014_1e-1));
        y = _mm256_add_ps(_mm256_mul_ps(y, m), splat(1.424_932_3e-1));
        y = _mm256_sub_ps(_mm256_mul_ps(y, m), splat(1.666_805_8e-1));
        y = _mm256_add_ps(_mm256_mul_ps(y, m), splat(2.000_071_5e-1));
        y = _mm256_sub_ps(_mm256_mul_ps(y, m), splat(2.499_999_4e-1));
        y = _mm256_add_ps(_mm256_mul_ps(y, m), splat(3.333_333_1e-1));
        y = _mm256_mul_ps(_mm256_mul_ps(y, m), z);
        let e = _mm256_cvtepi32_ps(exponent);
        y = _mm256_add_ps(y, _mm256_mul_ps(splat(-2.121_944_4e-4), e));
        y = _mm256_add_ps(y, _mm256_mul_ps(splat(-0.5), z));
        let result = _mm256_add_ps(m, y);
        _mm256_add_ps(result, _mm256_mul_ps(splat(0.693_359_4), e))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::excessive_precision)]
    fn exp(x: __m256) -> __m256 {
        let x = _mm256_max_ps(_mm256_min_ps(x, splat(88.0)), splat(-87.0));
        let fx = _mm256_floor_ps(_mm256_add_ps(
            _mm256_mul_ps(x, splat(std::f32::consts::LOG2_E)),
            splat(0.5),
        ));
        let mut r = _mm256_sub_ps(x, _mm256_mul_ps(fx, splat(0.693_359_4)));
        r = _mm256_sub_ps(r, _mm256_mul_ps(fx, splat(-2.121_944_4e-4)));
        let z = _mm256_mul_ps(r, r);
        let mut y = splat(1.987_569_1e-4);
        y = _mm256_add_ps(_mm256_mul_ps(y, r), splat(1.398_199_9e-3));
        y = _mm256_add_ps(_mm256_mul_ps(y, r), splat(8.333_452e-3));
        y = _mm256_add_ps(_mm256_mul_ps(y, r), splat(4.166_579_6e-2));
        y = _mm256_add_ps(_mm256_mul_ps(y, r), splat(1.666_666_5e-1));
        y = _mm256_add_ps(_mm256_mul_ps(y, r), splat(5.000_000_1e-1));
        y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(y, z), r), splat(1.0));
        let scale = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
            _mm256_cvttps_epi32(fx),
            _mm256_set1_epi32(127),
        )));
        _mm256_mul_ps(y, scale)
    }

    /// Shared prologue of the OBC pair integral and its derivative.
    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::type_complexity)]
    fn integral_terms(
        r: __m256,
        s: __m256,
        d: __m256,
        inv_d: __m256,
    ) -> (
        __m256,
        __m256,
        __m256,
        __m256,
        __m256,
        __m256,
        __m256,
        __m256,
    ) {
        let candidate = _mm256_andnot_ps(splat(-0.0), _mm256_sub_ps(d, s));
        let l = _mm256_max_ps(r, candidate);
        let u = _mm256_add_ps(d, s);
        let active = _mm256_and_ps(
            _mm256_cmp_ps::<_CMP_GT_OQ>(u, r),
            _mm256_cmp_ps::<_CMP_LT_OQ>(l, u),
        );
        let lu = select(active, l, splat(1.0));
        let uu = select(active, u, splat(2.0));
        let a = _mm256_div_ps(splat(1.0), lu);
        let b = _mm256_div_ps(splat(1.0), uu);
        let sd = _mm256_mul_ps(s, inv_d);
        let c = _mm256_sub_ps(d, _mm256_mul_ps(s, sd));
        let q = _mm256_sub_ps(_mm256_mul_ps(b, b), _mm256_mul_ps(a, a));
        let log = ln(_mm256_mul_ps(lu, b));
        (active, candidate, a, b, c, q, log, sd)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn radial_value(r: __m256, s: __m256, d: __m256, inv_d: __m256) -> __m256 {
        let (active, _, a, b, c, q, log, _) = integral_terms(r, s, d, inv_d);
        let value = _mm256_mul_ps(
            splat(0.5),
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_sub_ps(a, b),
                    _mm256_mul_ps(_mm256_mul_ps(splat(0.25), c), q),
                ),
                _mm256_mul_ps(_mm256_mul_ps(splat(0.5), log), inv_d),
            ),
        );
        _mm256_and_ps(active, value)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn radial_derivative(r: __m256, s: __m256, d: __m256, inv_d: __m256) -> __m256 {
        let (active, candidate, a, b, c, q, log, sd) = integral_terms(r, s, d, inv_d);
        let dl = select(
            _mm256_cmp_ps::<_CMP_LT_OQ>(candidate, r),
            splat(0.0),
            select(_mm256_cmp_ps::<_CMP_LT_OQ>(d, s), splat(-1.0), splat(1.0)),
        );
        let neg_dl = _mm256_xor_ps(dl, splat(-0.0));
        let t1 = _mm256_add_ps(
            _mm256_mul_ps(_mm256_mul_ps(neg_dl, a), a),
            _mm256_mul_ps(b, b),
        );
        let x = _mm256_mul_ps(_mm256_add_ps(splat(1.0), _mm256_mul_ps(sd, sd)), q);
        let y = _mm256_mul_ps(
            c,
            _mm256_add_ps(
                _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(splat(-2.0), b), b), b),
                _mm256_mul_ps(
                    _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(splat(2.0), dl), a), a),
                    a,
                ),
            ),
        );
        let t2 = _mm256_mul_ps(splat(0.25), _mm256_add_ps(x, y));
        let t3 = _mm256_mul_ps(
            splat(0.5),
            _mm256_sub_ps(
                _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(dl, a), b), inv_d),
                _mm256_mul_ps(_mm256_mul_ps(log, inv_d), inv_d),
            ),
        );
        let derivative = _mm256_mul_ps(splat(0.5), _mm256_add_ps(_mm256_add_ps(t1, t2), t3));
        _mm256_and_ps(active, derivative)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn distance2(
        xi: __m256,
        yi: __m256,
        zi: __m256,
        frame: &Frame,
        j: usize,
    ) -> (__m256, __m256, __m256, __m256) {
        let dx = _mm256_sub_ps(xi, splat(frame.x[j]));
        let dy = _mm256_sub_ps(yi, splat(frame.y[j]));
        let dz = _mm256_sub_ps(zi, splat(frame.z[j]));
        let r2 = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)),
            _mm256_mul_ps(dz, dz),
        );
        (dx, dy, dz, r2)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn own_mask(block: usize, j: usize) -> __m256 {
        let lanes = _mm256_add_epi32(
            _mm256_set1_epi32((block * LANES) as i32),
            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
        );
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(lanes, _mm256_set1_epi32(j as i32)))
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn born_sums(
        engine: &ImplicitPairEngine,
        frame: &Frame,
        block: usize,
    ) -> [f32; LANES] {
        let base = block * LANES;
        let xi = load(frame.x[base..base + LANES].try_into().unwrap());
        let yi = load(frame.y[base..base + LANES].try_into().unwrap());
        let zi = load(frame.z[base..base + LANES].try_into().unwrap());
        let ri = load(engine.descreen[base..base + LANES].try_into().unwrap());
        let mut sum = splat(0.0);
        for j in 0..engine.n {
            let (_, _, _, r2) = distance2(xi, yi, zi, frame, j);
            let d = _mm256_max_ps(_mm256_sqrt_ps(r2), splat(1e-8));
            let inv_d = _mm256_div_ps(splat(1.0), d);
            let own = select(own_mask(block, j), splat(0.0), splat(1.0));
            sum = _mm256_add_ps(
                sum,
                _mm256_mul_ps(own, radial_value(ri, splat(engine.scaled[j]), d, inv_d)),
            );
        }
        store(sum)
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::needless_range_loop)]
    pub(super) fn polar_sums(
        engine: &ImplicitPairEngine,
        frame: &Frame,
        born: &[f32],
        block: usize,
        gb_charge: &[f32; LANES],
        coulomb_charge: &[f32; LANES],
    ) -> ([[f32; LANES]; 3], [f32; LANES]) {
        let base = block * LANES;
        let xi = load(frame.x[base..base + LANES].try_into().unwrap());
        let yi = load(frame.y[base..base + LANES].try_into().unwrap());
        let zi = load(frame.z[base..base + LANES].try_into().unwrap());
        let bi = load(born[base..base + LANES].try_into().unwrap());
        let si = load(engine.sigma[base..base + LANES].try_into().unwrap());
        let ei = load(engine.sqrt_epsilon[base..base + LANES].try_into().unwrap());
        let gq = load(gb_charge);
        let cq = load(coulomb_charge);
        let valid: [f32; LANES] =
            std::array::from_fn(|lane| if base + lane < engine.n { 1.0 } else { 0.0 });
        let valid = load(&valid);
        let (mut gx, mut gy, mut gz, mut de_db) = (splat(0.0), splat(0.0), splat(0.0), splat(0.0));
        let mut cursor = 0;
        for j in 0..engine.n {
            let (dx, dy, dz, r2) = distance2(xi, yi, zi, frame, j);
            let bj = splat(born[j]);
            let qj = splat(engine.charge[j]);
            let own = own_mask(block, j);
            let p = _mm256_mul_ps(bi, bj);
            let four_p = _mm256_mul_ps(splat(4.0), p);
            let t = _mm256_div_ps(r2, four_p);
            let e = exp(_mm256_xor_ps(t, splat(-0.0)));
            let f = _mm256_max_ps(
                _mm256_sqrt_ps(_mm256_add_ps(r2, _mm256_mul_ps(p, e))),
                splat(1e-8),
            );
            let coefficient =
                _mm256_mul_ps(_mm256_mul_ps(gq, qj), select(own, splat(0.5), splat(1.0)));
            let de_dz = _mm256_div_ps(
                _mm256_mul_ps(splat(-0.5), coefficient),
                _mm256_mul_ps(_mm256_mul_ps(f, f), f),
            );
            let de_dp = _mm256_mul_ps(_mm256_mul_ps(de_dz, e), _mm256_add_ps(splat(1.0), t));
            de_db = _mm256_add_ps(
                de_db,
                _mm256_mul_ps(
                    _mm256_mul_ps(de_dp, bj),
                    select(own, splat(2.0), splat(1.0)),
                ),
            );
            let direct = select(
                own,
                splat(0.0),
                _mm256_mul_ps(
                    _mm256_mul_ps(splat(2.0), de_dz),
                    _mm256_sub_ps(splat(1.0), _mm256_mul_ps(splat(0.25), e)),
                ),
            );
            let keep = match engine.pair_keep(block, j, &mut cursor) {
                Some(mask) => load(&mask),
                None => valid,
            };
            let radius = _mm256_max_ps(_mm256_sqrt_ps(r2), splat(1e-8));
            let inv = _mm256_div_ps(keep, radius);
            let ratio = _mm256_mul_ps(_mm256_add_ps(si, splat(engine.sigma[j])), inv);
            let ratio2 = _mm256_mul_ps(ratio, ratio);
            let ratio6 = _mm256_mul_ps(_mm256_mul_ps(ratio2, ratio2), ratio2);
            let coulomb = _mm256_mul_ps(_mm256_mul_ps(cq, qj), inv);
            let pair = _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_sub_ps(
                        _mm256_mul_ps(
                            _mm256_mul_ps(
                                _mm256_mul_ps(splat(12.0), ei),
                                splat(engine.sqrt_epsilon[j]),
                            ),
                            _mm256_sub_ps(ratio6, _mm256_mul_ps(ratio6, ratio6)),
                        ),
                        coulomb,
                    ),
                    inv,
                ),
                inv,
            );
            let factor = _mm256_add_ps(direct, pair);
            gx = _mm256_add_ps(gx, _mm256_mul_ps(factor, dx));
            gy = _mm256_add_ps(gy, _mm256_mul_ps(factor, dy));
            gz = _mm256_add_ps(gz, _mm256_mul_ps(factor, dz));
        }
        ([store(gx), store(gy), store(gz)], store(de_db))
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::needless_range_loop)]
    pub(super) fn chain_sums(
        engine: &ImplicitPairEngine,
        frame: &Frame,
        chain: &[f32],
        block: usize,
    ) -> [[f32; LANES]; 3] {
        let base = block * LANES;
        let xi = load(frame.x[base..base + LANES].try_into().unwrap());
        let yi = load(frame.y[base..base + LANES].try_into().unwrap());
        let zi = load(frame.z[base..base + LANES].try_into().unwrap());
        let ri = load(engine.descreen[base..base + LANES].try_into().unwrap());
        let si = load(engine.scaled[base..base + LANES].try_into().unwrap());
        let ci = load(chain[base..base + LANES].try_into().unwrap());
        let (mut gx, mut gy, mut gz) = (splat(0.0), splat(0.0), splat(0.0));
        for j in 0..engine.n {
            let (dx, dy, dz, r2) = distance2(xi, yi, zi, frame, j);
            let raw = _mm256_sqrt_ps(r2);
            let d = _mm256_max_ps(raw, splat(1e-8));
            let inv_d = _mm256_div_ps(splat(1.0), d);
            let skip = _mm256_or_ps(
                own_mask(block, j),
                _mm256_cmp_ps::<_CMP_LT_OQ>(raw, splat(1e-8)),
            );
            let keep = select(skip, splat(0.0), splat(1.0));
            let factor = _mm256_mul_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(ci, radial_derivative(ri, splat(engine.scaled[j]), d, inv_d)),
                    _mm256_mul_ps(
                        splat(chain[j]),
                        radial_derivative(splat(engine.descreen[j]), si, d, inv_d),
                    ),
                ),
                _mm256_mul_ps(keep, inv_d),
            );
            gx = _mm256_add_ps(gx, _mm256_mul_ps(factor, dx));
            gy = _mm256_add_ps(gy, _mm256_mul_ps(factor, dy));
            gz = _mm256_add_ps(gz, _mm256_mul_ps(factor, dz));
        }
        [store(gx), store(gy), store(gz)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EnergyEvaluator, EnergyOptions};

    #[test]
    fn polynomial_transcendentals_track_libm() {
        for i in 1..4000 {
            let x = i as f32 * 0.013;
            let ln = ln_f32(x);
            assert!(
                (ln - x.ln()).abs() <= 2e-7 * x.ln().abs().max(1.0),
                "ln {x}"
            );
            let e = exp_f32(-x);
            assert!((e - (-x).exp()).abs() <= 3e-7 * (-x).exp(), "exp {x}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_and_portable_kernels_agree_bitwise() {
        if !simd_available() {
            return;
        }
        let system = glysys::SystemBuilder::new(glysys::BuildOptions {
            add_water: false,
            add_ions: false,
            ..Default::default()
        })
        .unwrap()
        .prepare_pdb_str(include_str!("../../../tests/fixtures/glycan.pdb"))
        .unwrap();
        let options = EnergyOptions::default();
        let engine =
            ImplicitPairEngine::new(&system, options.obc2.clone().unwrap(), options.dielectric)
                .unwrap();
        let mut frame = Frame {
            x: vec![1e6; engine.padded],
            y: vec![1e6; engine.padded],
            z: vec![1e6; engine.padded],
        };
        for (atom, p) in system.coordinates().iter().enumerate() {
            frame.x[atom] = p.x as f32;
            frame.y[atom] = p.y as f32;
            frame.z[atom] = p.z as f32;
        }
        let bits = |rows: &[[f32; LANES]]| -> Vec<u32> {
            rows.iter().flatten().map(|v| v.to_bits()).collect()
        };
        let blocks = engine.padded / LANES;
        let born: Vec<f32> = (0..blocks)
            .flat_map(|block| engine.born_block(&frame, block, false)[0])
            .collect();
        let chain: Vec<f32> = (0..engine.padded)
            .map(|atom| (atom as f32 * 0.37).sin())
            .collect();
        let charges: [f32; LANES] = std::array::from_fn(|lane| lane as f32 * 0.1 - 0.3);
        for block in 0..blocks {
            // SAFETY: AVX2 support was checked above.
            let simd = unsafe { avx2::born_sums(&engine, &frame, block) };
            assert_eq!(
                bits(&[engine.born_sums(&frame, block)]),
                bits(&[simd]),
                "born {block}"
            );
            let portable = engine.polar_sums(&frame, &born, block, &charges, &charges);
            // SAFETY: as above.
            let simd =
                unsafe { avx2::polar_sums(&engine, &frame, &born, block, &charges, &charges) };
            assert_eq!(bits(&portable.0), bits(&simd.0), "polar {block}");
            assert_eq!(bits(&[portable.1]), bits(&[simd.1]), "dE/dB {block}");
            // SAFETY: as above.
            let simd = unsafe { avx2::chain_sums(&engine, &frame, &chain, block) };
            assert_eq!(
                bits(&engine.chain_sums(&frame, &chain, block)),
                bits(&simd),
                "chain {block}"
            );
        }
    }

    #[test]
    fn matches_the_f64_evaluator() {
        for fixture in [
            include_str!("../../../tests/fixtures/dipeptide.pdb"),
            include_str!("../../../tests/fixtures/glycan.pdb"),
        ] {
            let system = glysys::SystemBuilder::new(glysys::BuildOptions {
                add_water: false,
                add_ions: false,
                ..Default::default()
            })
            .unwrap()
            .prepare_pdb_str(fixture)
            .unwrap();
            let options = EnergyOptions::default();
            let evaluator = EnergyEvaluator::new(&system, options.clone()).unwrap();
            let coordinates = system.coordinates();
            let reference = evaluator.gradient_only(&coordinates).unwrap();
            let mut fast = evaluator
                .weighted_gradient(&coordinates, [1., 1., 1., 1., 0., 0., 0., 0., 1.])
                .unwrap();
            let engine =
                ImplicitPairEngine::new(&system, options.obc2.clone().unwrap(), options.dielectric)
                    .unwrap();
            engine.gradient_into(&coordinates, &mut fast).unwrap();
            let (mut error2, mut reference2) = (0.0, 0.0);
            for (atom, (a, b)) in reference.iter().zip(&fast).enumerate() {
                for (want, got) in [(a.x, b.x), (a.y, b.y), (a.z, b.z)] {
                    assert!(
                        (want - got).abs() <= 2e-3 + 2e-4 * want.abs(),
                        "atom {atom}: {got} vs {want}"
                    );
                    error2 += (want - got).powi(2);
                    reference2 += want * want;
                }
            }
            assert!(
                (error2 / reference2).sqrt() < 2e-5,
                "RMS {}",
                (error2 / reference2).sqrt()
            );
        }
    }
}
