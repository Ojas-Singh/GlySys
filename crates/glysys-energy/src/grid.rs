//! A tabulated receptor for docking a flexible ligand into a rigid binding site.
//!
//! The ligand-receptor interaction of a prepared system is a sum over atom
//! pairs of the same terms [`crate::EnergyEvaluator`] evaluates: a 12-6
//! Lennard-Jones term from the Amber/GLYCAM radii and well depths and a
//! Coulomb term from the partial charges. With the receptor fixed, the sum
//! over receptor atoms is a function of the ligand atom's position alone, so
//! it can be tabulated once on a regular grid: one potential for the
//! electrostatics (per unit charge) and one Lennard-Jones potential per
//! distinct ligand atom type. A pose then costs one trilinear interpolation
//! per ligand atom instead of a sum over the whole receptor.
//!
//! The grid is a search tool: interpolating a steep repulsive wall is
//! approximate, so a pose found on the grid should be rescored and relaxed
//! with the exact evaluator.

use glysys::{ParameterizedSystem, Vec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{EnergyError, Result};

/// Coulomb constant in kcal Å / (mol e²), as the evaluator uses.
const COULOMB: f64 = 332.063_713_3;
/// Pair distances are floored here before evaluating the potentials.
const MIN_DISTANCE: f64 = 0.4;
/// Grid values are capped, so that interpolation near a clash stays bounded.
const CAP: f64 = 500.0;
/// Receptor atoms beyond this distance do not contribute to the van der Waals grids.
const VDW_CUTOFF: f64 = 20.0;
/// The van der Waals grids hold `asinh(E / VDW_SCALE)`: interpolating that
/// instead of `E` follows the r^-12 repulsive wall far more closely.
const VDW_SCALE: f64 = 1.0;

/// How the grid screens the Coulomb term.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Screening {
    /// `q_i q_j / (eps r)`: the evaluator's own model.
    Constant { dielectric: f64 },
    /// `q_i q_j / (eps(r) r)` with the sigmoidal distance-dependent dielectric
    /// of Mehler and Solmajer (Protein Eng. 4, 903, 1991), which grows from
    /// about 1.4 at contact to 78 in the bulk: a cheap stand-in for solvent screening.
    MehlerSolmajer,
}

impl Screening {
    /// The relative permittivity at distance `r` (angstrom).
    pub fn permittivity(self, r: f64) -> f64 {
        match self {
            Screening::Constant { dielectric } => dielectric,
            Screening::MehlerSolmajer => {
                const A: f64 = -8.5525;
                const EPSILON0: f64 = 78.4;
                const K: f64 = 7.7839;
                const LAMBDA: f64 = 0.003_627;
                let b = EPSILON0 - A;
                A + b / (1.0 + K * (-LAMBDA * b * r).exp())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct DockingGrid {
    origin: [f64; 3],
    spacing: f64,
    dims: [usize; 3],
    /// Lennard-Jones (radius, well depth) of each ligand atom class.
    classes: Vec<(f64, f64)>,
    /// One grid per class: `asinh(E / VDW_SCALE)` of the van der Waals energy
    /// of a probe of that class.
    vdw: Vec<Vec<f32>>,
    /// The electrostatic potential per unit charge, kcal/mol/e.
    potential: Vec<f32>,
    ligand_class: Vec<usize>,
    ligand_charge: Vec<f64>,
    screening: Screening,
}

/// A trilinear interpolant's value and gradient (per angstrom).
fn trilinear(
    grid: &[f32],
    dims: [usize; 3],
    spacing: f64,
    f: [f64; 3],
    cell: [usize; 3],
) -> (f64, [f64; 3]) {
    let index = |i: usize, j: usize, k: usize| {
        ((cell[0] + i) * dims[1] + (cell[1] + j)) * dims[2] + (cell[2] + k)
    };
    let c = |i, j, k| f64::from(grid[index(i, j, k)]);
    let (c000, c100, c010, c110) = (c(0, 0, 0), c(1, 0, 0), c(0, 1, 0), c(1, 1, 0));
    let (c001, c101, c011, c111) = (c(0, 0, 1), c(1, 0, 1), c(0, 1, 1), c(1, 1, 1));
    let [fx, fy, fz] = f;
    let (gx, gy, gz) = (1.0 - fx, 1.0 - fy, 1.0 - fz);
    let value = gx * (gy * (gz * c000 + fz * c001) + fy * (gz * c010 + fz * c011))
        + fx * (gy * (gz * c100 + fz * c101) + fy * (gz * c110 + fz * c111));
    let dx = gy * (gz * (c100 - c000) + fz * (c101 - c001))
        + fy * (gz * (c110 - c010) + fz * (c111 - c011));
    let dy = gx * (gz * (c010 - c000) + fz * (c011 - c001))
        + fx * (gz * (c110 - c100) + fz * (c111 - c101));
    let dz = gx * (gy * (c001 - c000) + fy * (c011 - c010))
        + fx * (gy * (c101 - c100) + fy * (c111 - c110));
    (value, [dx / spacing, dy / spacing, dz / spacing])
}

impl DockingGrid {
    /// Tabulate the receptor of `system` (atoms `receptor`, at `coordinates`)
    /// for the ligand atoms `ligand`, over a cube of half-width `half_width`
    /// angstrom around `center`, on a grid of `spacing` angstrom.
    pub fn build(
        system: &ParameterizedSystem,
        coordinates: &[Vec3],
        receptor: &[usize],
        ligand: &[usize],
        center: Vec3,
        half_width: f64,
        spacing: f64,
        screening: Screening,
    ) -> Result<Self> {
        if spacing <= 0.0 || half_width <= spacing || coordinates.len() != system.atom_count() {
            return Err(EnergyError::InvalidConfiguration(
                "grid needs a positive spacing, a larger half-width and one coordinate per atom"
                    .into(),
            ));
        }
        let atoms = system.atoms();
        // Ligand atom classes: atoms with the same Lennard-Jones parameters share a grid.
        let mut classes: Vec<(f64, f64)> = Vec::new();
        let mut ligand_class = Vec::with_capacity(ligand.len());
        for &i in ligand {
            let key = (
                atoms[i].lennard_jones_radius(),
                atoms[i].lennard_jones_epsilon(),
            );
            let class = classes
                .iter()
                .position(|c| (c.0 - key.0).abs() < 1.0e-9 && (c.1 - key.1).abs() < 1.0e-9)
                .unwrap_or_else(|| {
                    classes.push(key);
                    classes.len() - 1
                });
            ligand_class.push(class);
        }
        let n = (2.0 * half_width / spacing).ceil() as usize + 1;
        let dims = [n, n, n];
        let origin = [
            center.x - half_width,
            center.y - half_width,
            center.z - half_width,
        ];

        let sources: Vec<([f64; 3], f64, f64, f64)> = receptor
            .iter()
            .map(|&j| {
                let c = coordinates[j];
                (
                    [c.x, c.y, c.z],
                    atoms[j].charge(),
                    atoms[j].lennard_jones_radius(),
                    atoms[j].lennard_jones_epsilon(),
                )
            })
            .collect();
        let source_count = sources.len();
        let positions: Vec<[f64; 3]> = sources.iter().map(|s| s.0).collect();
        // Charge per source with the Coulomb constant folded in.
        let charges: Vec<f64> = sources.iter().map(|s| COULOMB * s.1).collect();
        // Per (class, source): the well depth and sigma^6 of the pair, so the
        // inner loop is a few multiply-adds.
        let mut well_depth = vec![0.0f64; classes.len() * source_count];
        let mut sigma6 = vec![0.0f64; classes.len() * source_count];
        for (k, class) in classes.iter().enumerate() {
            for (j, source) in sources.iter().enumerate() {
                well_depth[k * source_count + j] = (class.1 * source.3).sqrt();
                sigma6[k * source_count + j] = (class.0 + source.2).powi(6);
            }
        }
        // 1 / (eps(r) r) on a fine table; the sigmoid is too costly per pair.
        const TABLE_STEP: f64 = 0.01;
        let table_len = (400.0 / TABLE_STEP) as usize + 2;
        let screened: Vec<f64> = (0..table_len)
            .map(|i| {
                let r = (i as f64 * TABLE_STEP).max(MIN_DISTANCE);
                1.0 / (screening.permittivity(r) * r)
            })
            .collect();

        let total = n * n * n;
        let mut vdw = vec![vec![0.0f32; total]; classes.len()];
        let mut potential = vec![0.0f32; total];
        // One x-slab per task; each fills its slice of every grid.
        let slab = n * n;
        let mut vdw_slabs: Vec<Vec<&mut [f32]>> = (0..n).map(|_| Vec::new()).collect();
        for grid in &mut vdw {
            for (x, chunk) in grid.chunks_mut(slab).enumerate() {
                vdw_slabs[x].push(chunk);
            }
        }
        let cutoff2 = VDW_CUTOFF * VDW_CUTOFF;
        let class_count = classes.len();
        potential
            .par_chunks_mut(slab)
            .zip(vdw_slabs.into_par_iter())
            .enumerate()
            .for_each(|(x, (potential_slab, mut vdw_slab))| {
                let mut well = vec![0.0f64; class_count];
                for y in 0..n {
                    for z in 0..n {
                        let p = [
                            origin[0] + x as f64 * spacing,
                            origin[1] + y as f64 * spacing,
                            origin[2] + z as f64 * spacing,
                        ];
                        well.iter_mut().for_each(|w| *w = 0.0);
                        let mut phi = 0.0;
                        for (j, q) in positions.iter().enumerate() {
                            let d2 = (p[0] - q[0]).powi(2)
                                + (p[1] - q[1]).powi(2)
                                + (p[2] - q[2]).powi(2);
                            let r = d2.sqrt().max(MIN_DISTANCE);
                            let at = r / TABLE_STEP;
                            let (i, f) = (at as usize, at - (at as usize) as f64);
                            phi += charges[j] * (screened[i] * (1.0 - f) + screened[i + 1] * f);
                            if d2 < cutoff2 {
                                let inv6 = 1.0 / (r * r).powi(3);
                                for k in 0..class_count {
                                    let sig6 = sigma6[k * source_count + j];
                                    well[k] += well_depth[k * source_count + j]
                                        * sig6
                                        * inv6
                                        * (sig6 * inv6 - 2.0);
                                }
                            }
                        }
                        let at = y * n + z;
                        potential_slab[at] = phi.clamp(-CAP, CAP) as f32;
                        for (k, slab_k) in vdw_slab.iter_mut().enumerate() {
                            slab_k[at] = (well[k].min(CAP) / VDW_SCALE).asinh() as f32;
                        }
                    }
                }
            });
        Ok(Self {
            origin,
            spacing,
            dims,
            classes,
            vdw,
            potential,
            ligand_class,
            ligand_charge: ligand.iter().map(|&i| atoms[i].charge()).collect(),
            screening,
        })
    }

    pub fn screening(&self) -> Screening {
        self.screening
    }

    pub fn ligand_atoms(&self) -> usize {
        self.ligand_class.len()
    }

    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    /// Edge length of the tabulated cube, angstrom.
    pub fn extent(&self) -> f64 {
        (self.dims[0] - 1) as f64 * self.spacing
    }

    /// Interaction energy of the ligand atoms at `positions` (in the order
    /// given to `build`) with the receptor, kcal/mol; the gradient with
    /// respect to each position is added to `gradient` when it is given.
    /// A position outside the cube is evaluated at the nearest point of it
    /// plus a quadratic penalty, so a search is pushed back inside.
    pub fn energy_gradient(&self, positions: &[Vec3], mut gradient: Option<&mut [Vec3]>) -> f64 {
        const OUTSIDE: f64 = 10.0;
        let mut total = 0.0;
        for (a, position) in positions.iter().enumerate() {
            let p = [position.x, position.y, position.z];
            let mut inside = [0.0; 3];
            let mut cell = [0usize; 3];
            let mut f = [0.0; 3];
            let mut penalty = 0.0;
            let mut penalty_gradient = [0.0; 3];
            for k in 0..3 {
                let top = (self.dims[k] - 1) as f64 * self.spacing;
                let local = p[k] - self.origin[k];
                let clamped = local.clamp(0.0, top);
                let outside = local - clamped;
                penalty += OUTSIDE * outside * outside;
                penalty_gradient[k] = 2.0 * OUTSIDE * outside;
                inside[k] = clamped;
                let scaled = clamped / self.spacing;
                cell[k] = (scaled.floor() as usize).min(self.dims[k] - 2);
                f[k] = scaled - cell[k] as f64;
            }
            let class = self.ligand_class[a];
            let (compressed, compressed_gradient) =
                trilinear(&self.vdw[class], self.dims, self.spacing, f, cell);
            let well = VDW_SCALE * compressed.sinh();
            let slope = VDW_SCALE * compressed.cosh();
            let well_gradient = [
                slope * compressed_gradient[0],
                slope * compressed_gradient[1],
                slope * compressed_gradient[2],
            ];
            let (phi, phi_gradient) = trilinear(&self.potential, self.dims, self.spacing, f, cell);
            let charge = self.ligand_charge[a];
            total += well + charge * phi + penalty;
            if let Some(gradient) = gradient.as_deref_mut() {
                gradient[a].x += well_gradient[0] + charge * phi_gradient[0] + penalty_gradient[0];
                gradient[a].y += well_gradient[1] + charge * phi_gradient[1] + penalty_gradient[1];
                gradient[a].z += well_gradient[2] + charge * phi_gradient[2] + penalty_gradient[2];
            }
        }
        total
    }

    pub fn energy(&self, positions: &[Vec3]) -> f64 {
        self.energy_gradient(positions, None)
    }

    /// The van der Waals and electrostatic parts of [`Self::energy`] (inside the cube only).
    pub fn components(&self, positions: &[Vec3]) -> (f64, f64) {
        let (mut well_total, mut coulomb_total) = (0.0, 0.0);
        for (a, position) in positions.iter().enumerate() {
            let p = [position.x, position.y, position.z];
            let mut cell = [0usize; 3];
            let mut f = [0.0; 3];
            for k in 0..3 {
                let top = (self.dims[k] - 1) as f64 * self.spacing;
                let scaled = (p[k] - self.origin[k]).clamp(0.0, top) / self.spacing;
                cell[k] = (scaled.floor() as usize).min(self.dims[k] - 2);
                f[k] = scaled - cell[k] as f64;
            }
            well_total += VDW_SCALE
                * trilinear(
                    &self.vdw[self.ligand_class[a]],
                    self.dims,
                    self.spacing,
                    f,
                    cell,
                )
                .0
                .sinh();
            coulomb_total += self.ligand_charge[a]
                * trilinear(&self.potential, self.dims, self.spacing, f, cell).0;
        }
        (well_total, coulomb_total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mehler_solmajer_screening_grows_from_contact_to_bulk() {
        let s = Screening::MehlerSolmajer;
        assert!((s.permittivity(0.0) - 1.35).abs() < 0.05);
        assert!(s.permittivity(3.0) > 10.0 && s.permittivity(3.0) < 16.0);
        assert!(s.permittivity(30.0) > 70.0 && s.permittivity(30.0) < 80.0);
    }
}
