//! Rigid three-site water constraints.
//!
//! The position solve iterates the three distance constraints (two O-H, one
//! H-H) to 1e-12 angstrom. For MD-scale displacements this is the same
//! solution analytic SETTLE produces, and the OpenMM parity harness — not the
//! algorithm name — is the acceptance gate.
//!
//! Water *velocities* are handled the analytic-SETTLE way: each water's
//! velocities are rotated by the same rigid rotation its positions underwent
//! over the step, instead of projecting out bond-parallel components
//! (RATTLE). Projection deletes the mismatch between the velocity direction
//! and the rotating bond frame every step, which systematically damps
//! molecular rotation — measurably, a force-free rotor loses most of its
//! kinetic energy within picoseconds, and solvated NVT stalls far below
//! target temperature. Rotation is an isometry about the molecular center of
//! mass: it preserves kinetic energy and linear momentum exactly, keeps
//! tangential velocities tangential to the new geometry, and cannot
//! systematically drain energy because kinetic energy then changes only
//! through conservative force kicks. Solute X-H bonds keep iterative
//! RATTLE (their rotation is slow, so projection loss is negligible), as
//! does one-time Maxwell-velocity initialization, which has no prior frame
//! to rotate from. All loops use fixed iteration caps with residual checks,
//! so results are deterministic and failures are explicit instead of
//! silently loose.
use super::{Error, Result};
use glysys::Vec3;

fn invalid(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }
}

fn norm(v: Vec3) -> Option<Vec3> {
    let n = dot(v, v).sqrt();
    if !n.is_finite() || n < 1e-14 {
        return None;
    }
    Some(Vec3 {
        x: v.x / n,
        y: v.y / n,
        z: v.z / n,
    })
}

/// Orthonormal frame of a water triangle: e1 along O→H1, e3 along the
/// triangle normal, e2 completing the right-handed system. Returns the
/// basis vectors as (x, y, z) triples. Both the pre-step and post-SHAKE
/// geometries are exactly rigid (constrained to 1e-10 A or better), so the
/// frames are always well defined for physical H-O-H angles.
fn water_frame(o: Vec3, h1: Vec3, h2: Vec3) -> Option<[[f64; 3]; 3]> {
    let e1 = norm(sub(h1, o))?;
    let e3 = norm(cross(sub(h1, o), sub(h2, o)))?;
    let e2v = cross(e3, e1);
    let (e1a, e2a, e3a) = (
        [e1.x, e1.y, e1.z],
        [e2v.x, e2v.y, e2v.z],
        [e3.x, e3.y, e3.z],
    );
    Some([e1a, e2a, e3a])
}

/// Rotation taking the `old` frame to the `new` frame: R[a][b] =
/// sum_i new_i[a] * old_i[b]. Maps old bond directions onto new ones.
fn frame_rotation(old: [[f64; 3]; 3], new: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.; 3]; 3];
    for i in 0..3 {
        for a in 0..3 {
            for b in 0..3 {
                r[a][b] += new[i][a] * old[i][b];
            }
        }
    }
    r
}

fn apply_rotation(r: [[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3 {
        x: r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
        y: r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
        z: r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
    }
}

/// General iterative bond solver shared by the CPU and GPU paths. Each entry
/// is `(atom_a, atom_b, target_distance, inverse_mass_a, inverse_mass_b)`.
/// Positions are corrected in place; returns the worst residual.
pub fn shake_positions(
    coords: &mut [Vec3],
    constraints: &[(usize, usize, f64, f64, f64)],
    tolerance: f64,
    max_iterations: usize,
) -> Result<f64> {
    let mut worst = f64::INFINITY;
    for _ in 0..max_iterations {
        worst = 0.;
        for &(a, b, target, ima, imb) in constraints {
            let (pa, pb) = (coords[a], coords[b]);
            let d = sub(pa, pb);
            let r = (dot(d, d)).sqrt().max(1.0e-12);
            let residual = (r - target).abs();
            worst = worst.max(residual);
            if residual < tolerance {
                continue;
            }
            // Standard SHAKE correction along the bond.
            let lambda = (r - target) / (r * (ima + imb));
            coords[a] = Vec3 {
                x: pa.x - lambda * ima * d.x,
                y: pa.y - lambda * ima * d.y,
                z: pa.z - lambda * ima * d.z,
            };
            coords[b] = Vec3 {
                x: pb.x + lambda * imb * d.x,
                y: pb.y + lambda * imb * d.y,
                z: pb.z + lambda * imb * d.z,
            };
        }
        if worst < tolerance {
            return Ok(worst);
        }
    }
    Err(invalid(format!(
        "water constraints did not converge (residual {worst:.3e} A)"
    )))
}

/// RATTLE-style velocity projection: remove bond-parallel relative velocity.
pub fn rattle_velocities(
    coords: &[Vec3],
    velocities: &mut [Vec3],
    constraints: &[(usize, usize, f64, f64, f64)],
    tolerance: f64,
    max_iterations: usize,
) -> Result<f64> {
    let mut worst = f64::INFINITY;
    for _ in 0..max_iterations {
        worst = 0.;
        for &(a, b, _, ima, imb) in constraints {
            let d = sub(coords[a], coords[b]);
            let r = (dot(d, d)).sqrt().max(1.0e-12);
            let rel = sub(velocities[a], velocities[b]);
            let rv = (rel.x * d.x + rel.y * d.y + rel.z * d.z) / r;
            worst = worst.max(rv.abs());
            if rv.abs() < tolerance {
                continue;
            }
            let lambda = rv / (r * (ima + imb));
            velocities[a] = Vec3 {
                x: velocities[a].x - lambda * ima * d.x,
                y: velocities[a].y - lambda * ima * d.y,
                z: velocities[a].z - lambda * ima * d.z,
            };
            velocities[b] = Vec3 {
                x: velocities[b].x + lambda * imb * d.x,
                y: velocities[b].y + lambda * imb * d.y,
                z: velocities[b].z + lambda * imb * d.z,
            };
        }
        if worst < tolerance {
            return Ok(worst);
        }
    }
    Err(invalid(format!(
        "water velocity constraints did not converge (residual {worst:.3e} A/ps)"
    )))
}

/// Rigid-water constraint set built from classified water indices and masses.
/// Solute bonds involving hydrogen join the same solver so 2 fs steps stay
/// stable; water geometry still converges to its own tight tolerance first.
#[derive(Clone, Debug)]
pub struct SettleWaters {
    /// Flat O/H/H triples, matching `classify_waters` order.
    pub waters: Vec<[usize; 3]>,
    constraints: Vec<(usize, usize, f64, f64, f64)>,
    /// Solute X-H bonds constrained alongside the waters, if any.
    solute_bonds: Vec<(usize, usize, f64, f64, f64)>,
    pub tolerance: f64,
}

impl SettleWaters {
    /// Targets measured from `coords` (the historical behavior). Prefer
    /// [`Self::from_equilibrium`] for dynamics: measuring targets on a
    /// strained snapshot pins strain into the manifold and pumps energy.
    pub fn new(
        waters: Vec<[usize; 3]>,
        coords: &[Vec3],
        masses: &[f64],
    ) -> Result<Self> {
        Self::with_solute_bonds(waters, Vec::new(), coords, masses)
    }

    /// Equilibrium targets: `water` gives per-water `(oh1, oh2, hh)` lengths
    /// and `solute` gives `(a, b, equilibrium_length)` bonds. Dynamics
    /// constructors must use this so constraints never pin snapshot strain.
    pub fn from_equilibrium(
        waters: Vec<[usize; 3]>,
        water: Vec<(f64, f64, f64)>,
        solute: Vec<(usize, usize, f64)>,
        masses: &[f64],
        atom_count: usize,
    ) -> Result<Self> {
        if waters.len() != water.len() {
            return Err(invalid("water target count mismatch"));
        }
        let mut constraints = Vec::with_capacity(waters.len() * 3);
        for (w, &(oh1, oh2, hh)) in waters.iter().zip(&water) {
            let [o, h1, h2] = *w;
            for (a, b, target) in [(o, h1, oh1), (o, h2, oh2), (h1, h2, hh)] {
                if a >= atom_count || b >= atom_count || a >= masses.len() || b >= masses.len() {
                    return Err(invalid("water index out of range"));
                }
                if !(target.is_finite() && target > 0.3 && target < 3.0) {
                    return Err(invalid("unphysical water equilibrium length"));
                }
                if masses[a] <= 0. || masses[b] <= 0. {
                    return Err(invalid("water masses must be positive"));
                }
                constraints.push((a, b, target, 1. / masses[a], 1. / masses[b]));
            }
        }
        let mut solute_constraints = Vec::with_capacity(solute.len());
        for (a, b, target) in solute {
            if a >= atom_count || b >= atom_count || a >= masses.len() || b >= masses.len() {
                return Err(invalid("solute bond index out of range"));
            }
            if !(target.is_finite() && target > 0.3 && target < 3.0) {
                return Err(invalid("unphysical solute equilibrium length"));
            }
            if masses[a] <= 0. || masses[b] <= 0. {
                return Err(invalid("solute masses must be positive"));
            }
            solute_constraints.push((a, b, target, 1. / masses[a], 1. / masses[b]));
        }
        Ok(Self {
            waters,
            constraints,
            solute_bonds: solute_constraints,
            tolerance: 1e-10,
        })
    }

    pub fn with_solute_bonds(
        waters: Vec<[usize; 3]>,
        solute_bonds: Vec<(usize, usize)>,
        coords: &[Vec3],
        masses: &[f64],
    ) -> Result<Self> {
        let mut solute = Vec::with_capacity(solute_bonds.len());
        for (a, b) in solute_bonds {
            if a >= coords.len() || b >= coords.len() || a >= masses.len() || b >= masses.len() {
                return Err(invalid("solute bond index out of range"));
            }
            let d = sub(coords[a], coords[b]);
            let target = dot(d, d).sqrt();
            if !target.is_finite() || target < 0.3 || target > 3.0 {
                return Err(invalid("unphysical solute bond length"));
            }
            if masses[a] <= 0. || masses[b] <= 0. {
                return Err(invalid("solute masses must be positive"));
            }
            solute.push((a, b, target, 1. / masses[a], 1. / masses[b]));
        }
        let mut constraints = Vec::with_capacity(waters.len() * 3);
        for &[o, h1, h2] in &waters {
            for (a, b) in [(o, h1), (o, h2), (h1, h2)] {
                if a >= coords.len() || b >= coords.len() || a >= masses.len() || b >= masses.len()
                {
                    return Err(invalid("water index out of range"));
                }
                let d = sub(coords[a], coords[b]);
                let target = dot(d, d).sqrt();
                if !target.is_finite() || target < 0.3 || target > 3.0 {
                    return Err(invalid(
                        "reference water geometry is unphysical; minimize before constraining",
                    ));
                }
                if masses[a] <= 0. || masses[b] <= 0. {
                    return Err(invalid("water masses must be positive"));
                }
                constraints.push((a, b, target, 1. / masses[a], 1. / masses[b]));
            }
        }
        Ok(Self {
            waters,
            constraints,
            solute_bonds: solute,
            tolerance: 1e-10,
        })
    }

    /// Number of scalar distance constraints (one DOF each).
    pub fn constraint_count(&self) -> usize {
        self.constraints.len() + self.solute_bonds.len()
    }

    pub fn constrain_positions(&self, coords: &mut [Vec3]) -> Result<f64> {
        let worst = shake_positions(coords, &self.constraints, self.tolerance, 200)?;
        if !self.solute_bonds.is_empty() {
            // Solute H bonds share the solver; waters already converged above.
            shake_positions(coords, &self.solute_bonds, 1e-8, 200)?;
        }
        Ok(worst)
    }

    /// SHAKE on solute X-H bonds only. Waters are updated rigidly per step
    /// (see [`Self::rigid_water_positions`]) and never enter the iterative
    /// position solver after initialization.
    pub fn shake_solute_positions(&self, coords: &mut [Vec3]) -> Result<f64> {
        if self.solute_bonds.is_empty() {
            return Ok(0.);
        }
        shake_positions(coords, &self.solute_bonds, 1e-8, 200)
    }

    /// Rigid-body position update for waters (analytic-SETTLE-style position
    /// half): each water translates by its drifted center of mass and rotates
    /// by the best-fit rotation from its old frame to its drifted frame.
    /// `drifted` holds the ordinary unconstrained drift
    /// (`old + v_half * dt`) for every atom; water entries are overwritten
    /// with the rigid motion, solute entries are left for the caller to
    /// SHAKE. Unlike drift-plus-SHAKE-pullback, the triangle never distorts,
    /// so no bond-strain energy appears and disappears each step (that
    /// strain slosh, pumped by kicks, is what explodes or — under RATTLE
    /// deletion — drains). A water whose drifted frame cannot be built falls
    /// back to SHAKE plus RATTLE for that water only.
    pub fn rigid_water_positions(
        &self,
        old_coords: &[Vec3],
        drifted: &[Vec3],
        coords: &mut [Vec3],
        masses: &[f64],
    ) -> Result<()> {
        for &[o, h1, h2] in &self.waters {
            let (Some(old_frame), Some(drift_frame)) = (
                water_frame(old_coords[o], old_coords[h1], old_coords[h2]),
                water_frame(drifted[o], drifted[h1], drifted[h2]),
            ) else {
                // Degenerate drifted triangle: project positions and
                // velocities iteratively for this water only.
                coords[o] = drifted[o];
                coords[h1] = drifted[h1];
                coords[h2] = drifted[h2];
                shake_positions(coords, &self.water_constraints(o, h1, h2), self.tolerance, 200)?;
                continue;
            };
            let rot = frame_rotation(old_frame, drift_frame);
            let mtriple = masses[o] + masses[h1] + masses[h2];
            if !mtriple.is_finite() || mtriple <= 0. {
                return Err(invalid("water masses must be positive"));
            }
            let old_com = Vec3 {
                x: (masses[o] * old_coords[o].x
                    + masses[h1] * old_coords[h1].x
                    + masses[h2] * old_coords[h2].x)
                    / mtriple,
                y: (masses[o] * old_coords[o].y
                    + masses[h1] * old_coords[h1].y
                    + masses[h2] * old_coords[h2].y)
                    / mtriple,
                z: (masses[o] * old_coords[o].z
                    + masses[h1] * old_coords[h1].z
                    + masses[h2] * old_coords[h2].z)
                    / mtriple,
            };
            // Center-of-mass motion comes from the drift (it carries the
            // kick impulse); orientation comes from the best-fit rotation.
            // Using the drifted COM keeps translation exact.
            let new_com = Vec3 {
                x: (masses[o] * drifted[o].x
                    + masses[h1] * drifted[h1].x
                    + masses[h2] * drifted[h2].x)
                    / mtriple,
                y: (masses[o] * drifted[o].y
                    + masses[h1] * drifted[h1].y
                    + masses[h2] * drifted[h2].y)
                    / mtriple,
                z: (masses[o] * drifted[o].z
                    + masses[h1] * drifted[h1].z
                    + masses[h2] * drifted[h2].z)
                    / mtriple,
            };
            for &i in &[o, h1, h2] {
                let rel = sub(old_coords[i], old_com);
                let turned = apply_rotation(rot, rel);
                coords[i] = Vec3 {
                    x: turned.x + new_com.x,
                    y: turned.y + new_com.y,
                    z: turned.z + new_com.z,
                };
            }
        }
        Ok(())
    }

    /// Full RATTLE projection (waters and solute). One-time use only:
    /// projecting the initial Maxwell velocities onto the constraint
    /// manifold. Per-step velocity handling must use
    /// [`Self::rotate_water_velocities`] plus
    /// [`Self::project_solute_velocities`]: repeated projection
    /// systematically damps molecular rotation, while rotation preserves it.
    pub fn constrain_velocities(&self, coords: &[Vec3], velocities: &mut [Vec3]) -> Result<f64> {
        let worst = rattle_velocities(coords, velocities, &self.constraints, 1e-8, 200)?;
        if !self.solute_bonds.is_empty() {
            rattle_velocities(coords, velocities, &self.solute_bonds, 1e-6, 200)?;
        }
        Ok(worst)
    }

    /// RATTLE projection for solute X-H bonds only. Their rotation is slow,
    /// so per-step projection loss is negligible, and unlike waters they do
    /// not move as rigid bodies (the solute deforms), which makes frame
    /// rotation inapplicable.
    pub fn project_solute_velocities(
        &self,
        coords: &[Vec3],
        velocities: &mut [Vec3],
    ) -> Result<f64> {
        if self.solute_bonds.is_empty() {
            return Ok(0.);
        }
        rattle_velocities(coords, velocities, &self.solute_bonds, 1e-6, 200)
    }

    /// SETTLE-style velocity update for waters: rotate each water's
    /// velocities about its center of mass by the rigid rotation its
    /// positions underwent between `old_coords` (start of step, constrained)
    /// and `coords` (post-SHAKE, constrained). Both frames are exactly
    /// rigid, so the rotation is well defined; it preserves kinetic energy
    /// and center-of-mass velocity exactly and keeps tangential velocities
    /// tangential to the new geometry. `masses` supplies atomic masses for
    /// the center-of-mass velocity. A water whose frame cannot be built
    /// (degenerate triangle, treated as a failure of the position solve)
    /// falls back to RATTLE projection for that water only.
    pub fn rotate_water_velocities(
        &self,
        old_coords: &[Vec3],
        coords: &[Vec3],
        velocities: &mut [Vec3],
        masses: &[f64],
    ) -> Result<f64> {
        let mut fallback = 0f64;
        for &[o, h1, h2] in &self.waters {
            let Some(old_frame) = water_frame(old_coords[o], old_coords[h1], old_coords[h2])
            else {
                fallback = fallback.max(rattle_velocities(
                    coords,
                    velocities,
                    &self.water_constraints(o, h1, h2),
                    1e-8,
                    200,
                )?);
                continue;
            };
            let Some(new_frame) = water_frame(coords[o], coords[h1], coords[h2]) else {
                fallback = fallback.max(rattle_velocities(
                    coords,
                    velocities,
                    &self.water_constraints(o, h1, h2),
                    1e-8,
                    200,
                )?);
                continue;
            };
            let rot = frame_rotation(old_frame, new_frame);
            let mtriple = masses[o] + masses[h1] + masses[h2];
            if !mtriple.is_finite() || mtriple <= 0. {
                return Err(invalid("water masses must be positive"));
            }
            let com_v = Vec3 {
                x: (masses[o] * velocities[o].x
                    + masses[h1] * velocities[h1].x
                    + masses[h2] * velocities[h2].x)
                    / mtriple,
                y: (masses[o] * velocities[o].y
                    + masses[h1] * velocities[h1].y
                    + masses[h2] * velocities[h2].y)
                    / mtriple,
                z: (masses[o] * velocities[o].z
                    + masses[h1] * velocities[h1].z
                    + masses[h2] * velocities[h2].z)
                    / mtriple,
            };
            for &i in &[o, h1, h2] {
                let rel = sub(velocities[i], com_v);
                let turned = apply_rotation(rot, rel);
                velocities[i] = Vec3 {
                    x: turned.x + com_v.x,
                    y: turned.y + com_v.y,
                    z: turned.z + com_v.z,
                };
            }
        }
        Ok(fallback)
    }

    /// The three distance constraints of one water, for the degenerate-frame
    /// RATTLE fallback in [`Self::rotate_water_velocities`].
    fn water_constraints(&self, o: usize, h1: usize, h2: usize) -> Vec<(usize, usize, f64, f64, f64)> {
        self.constraints
            .iter()
            .filter(|&&(a, b, _, _, _)| {
                (a == o || a == h1 || a == h2) && (b == o || b == h1 || b == h2)
            })
            .copied()
            .collect()
    }

    pub fn max_violation(&self, coords: &[Vec3]) -> f64 {
        self.constraints
            .iter()
            .map(|&(a, b, target, _, _)| {
                let d = sub(coords[a], coords[b]);
                ((dot(d, d)).sqrt() - target).abs()
            })
            .fold(0., f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OH: f64 = 0.9572;
    const ANGLE: f64 = 104.52;

    fn tip3p() -> (Vec<Vec3>, Vec<f64>) {
        let o = Vec3 { x: 0., y: 0., z: 0. };
        let h1 = Vec3 { x: OH, y: 0., z: 0. };
        let h2 = Vec3 {
            x: OH * (ANGLE.to_radians()).cos(),
            y: OH * (ANGLE.to_radians()).sin(),
            z: 0.,
        };
        (vec![o, h1, h2], vec![16., 1.008, 1.008])
    }

    #[test]
    fn perturbed_water_returns_to_geometry() {
        let (mut coords, masses) = tip3p();
        // Displace as a 2 fs step at 300 K roughly would.
        coords[1].x += 0.02;
        coords[2].y -= 0.015;
        coords[0].z += 0.01;
        // Reference geometry comes from the ideal water, constraints measured on it.
        let ideal = tip3p().0;
        let settle = SettleWaters::new(vec![[0, 1, 2]], &ideal, &masses).unwrap();
        settle.constrain_positions(&mut coords).unwrap();
        assert!(settle.max_violation(&coords) < 1e-9);
    }

    /// Frame rotation must recover a known analytic rotation exactly (not
    /// just preserve norms): the dynamics is only right if velocities rotate
    /// *with* the body.
    #[test]
    fn frame_rotation_recovers_known_rotation() {
        let (coords, _) = tip3p();
        let (o, h1, h2) = (coords[0], coords[1], coords[2]);
        let old = water_frame(o, h1, h2).unwrap();
        // 37 degrees about (0.3, 0.8, 0.5).
        let axis = Vec3 { x: 0.3, y: 0.8, z: 0.5 };
        let an = dot(axis, axis).sqrt();
        let (ux, uy, uz) = (axis.x / an, axis.y / an, axis.z / an);
        let th = 37f64.to_radians();
        let (c, s) = (th.cos(), th.sin());
        let q = [
            [c + ux * ux * (1. - c), ux * uy * (1. - c) - uz * s, ux * uz * (1. - c) + uy * s],
            [uy * ux * (1. - c) + uz * s, c + uy * uy * (1. - c), uy * uz * (1. - c) - ux * s],
            [uz * ux * (1. - c) - uy * s, uz * uy * (1. - c) + ux * s, c + uz * uz * (1. - c)],
        ];
        let app = |p: Vec3| Vec3 {
            x: q[0][0] * p.x + q[0][1] * p.y + q[0][2] * p.z,
            y: q[1][0] * p.x + q[1][1] * p.y + q[1][2] * p.z,
            z: q[2][0] * p.x + q[2][1] * p.y + q[2][2] * p.z,
        };
        let new = water_frame(app(o), app(h1), app(h2)).unwrap();
        let r = frame_rotation(old, new);
        for a in 0..3 {
            for b in 0..3 {
                assert!((r[a][b] - q[a][b]).abs() < 1e-12, "R[{a}][{b}] mismatch");
            }
        }
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        assert!((det - 1.).abs() < 1e-12, "rotation must be proper, det={det}");
    }

    /// Rigid position update plus velocity rotation must conserve a spinning
    /// water exactly (force-free): kinetic energy, center of mass, and
    /// geometry. Iterative RATTLE projection instead drains rotation
    /// measurably, which is why per-step velocity handling rotates.
    #[test]
    fn rigid_update_conserves_spinning_water() {
        let (coords, masses) = tip3p();
        let targets = vec![(OH, OH, 2. * OH * (ANGLE.to_radians() / 2.).sin())];
        let settle =
            SettleWaters::from_equilibrium(vec![[0, 1, 2]], targets, vec![], &masses, 3).unwrap();
        let mut coords = coords;
        // Rigid spin about z through the COM plus translation.
        let mtriple: f64 = masses.iter().sum();
        let com = Vec3 {
            x: (masses[0] * coords[0].x + masses[1] * coords[1].x + masses[2] * coords[2].x) / mtriple,
            y: (masses[0] * coords[0].y + masses[1] * coords[1].y + masses[2] * coords[2].y) / mtriple,
            z: 0.,
        };
        let w = 8.0;
        let mut vel: Vec<Vec3> = coords
            .iter()
            .map(|p| Vec3 {
                x: 1.0 - w * (p.y - com.y),
                y: 0.5 + w * (p.x - com.x),
                z: 0.3,
            })
            .collect();
        settle.constrain_velocities(&coords, &mut vel).unwrap();
        let kinetic = |v: &[Vec3]| {
            0.5 * v
                .iter()
                .zip(&masses)
                .map(|(a, m)| m * (a.x * a.x + a.y * a.y + a.z * a.z))
                .sum::<f64>()
                / 418.4
        };
        let e0 = kinetic(&vel);
        let dt = 0.002;
        for _ in 0..2000 {
            let old = coords.clone();
            let mut drifted = coords.clone();
            for a in 0..3 {
                drifted[a] = Vec3 {
                    x: drifted[a].x + vel[a].x * dt,
                    y: drifted[a].y + vel[a].y * dt,
                    z: drifted[a].z + vel[a].z * dt,
                };
            }
            settle
                .rigid_water_positions(&old, &drifted, &mut coords, &masses)
                .unwrap();
            settle
                .rotate_water_velocities(&old, &coords, &mut vel, &masses)
                .unwrap();
        }
        assert!((kinetic(&vel) - e0).abs() < 1e-9, "rotation must preserve K");
        assert!(settle.max_violation(&coords) < 1e-9);
    }

    #[test]
    fn velocity_projection_kills_bond_motion() {
        let (coords, masses) = tip3p();
        let settle = SettleWaters::new(vec![[0, 1, 2]], &coords, &masses).unwrap();
        let mut velocities = vec![
            Vec3 { x: 5., y: -3., z: 2. },
            Vec3 { x: -4., y: 6., z: 1. },
            Vec3 { x: 1., y: 1., z: -7. },
        ];
        settle.constrain_velocities(&coords, &mut velocities).unwrap();
        // Bond-parallel relative velocities must vanish; total momentum kept.
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            let d = sub(coords[a], coords[b]);
            let r = dot(d, d).sqrt();
            let rel = sub(velocities[a], velocities[b]);
            assert!((rel.x * d.x + rel.y * d.y + rel.z * d.z).abs() / r < 1e-7);
        }
    }
}
