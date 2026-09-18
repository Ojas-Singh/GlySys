//! Rigid three-site water constraints.
//!
//! Analytic SETTLE positions are coupled to RATTLE velocity corrections.
//! The position solve returns its coordinate correction, and the matching
//! half-step velocity is reconstructed from the constrained displacement.
//! Waters never use a frame-rotation heuristic. Solute X-H bonds use the
//! general iterative SHAKE/RATTLE solver, as does one-time projection of
//! initialized Maxwell velocities. All loops have bounded iterations and
//! explicit residual failures.
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

fn add(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
    }
}

fn scale(a: Vec3, s: f64) -> Vec3 {
    Vec3 {
        x: a.x * s,
        y: a.y * s,
        z: a.z * s,
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

/// Analytic SETTLE solve for one isosceles water (Miyamoto & Kollman 1992,
/// J. Comput. Chem. 13:952). The solve is relative to the old constrained
/// triangle and returns both constrained coordinates and the coordinate
/// correction. The latter is required by RATTLE: the matching velocity
/// correction is `v += dx / dt`.
///
/// This is the canonical formulation intended for both CPU and WGSL. It has a
/// fixed flop count, no convergence branch, and no history-dependent iteration
/// path. It requires equal hydrogen masses, a non-degenerate old triangle, and
/// an MD-scale unconstrained displacement.
pub fn settle_triangle(
    old_o: Vec3,
    old_h1: Vec3,
    old_h2: Vec3,
    qo: Vec3,
    q1: Vec3,
    q2: Vec3,
    mo: f64,
    m1: f64,
    m2: f64,
    doh: f64,
    dhh: f64,
) -> Result<([Vec3; 3], [Vec3; 3])> {
    // This is a direct translation of OpenMM's ReferenceSETTLEAlgorithm.
    // Keeping the displacement variables explicit is important: the old
    // triangle supplies the reference frame while the predicted coordinates
    // supply the center-of-mass displacement and orientation.
    let total = mo + m1 + m2;
    if !total.is_finite() || total <= 0. || !mo.is_finite() || !m1.is_finite() || !m2.is_finite() {
        return Err(invalid("water masses must be finite and positive"));
    }
    if !(doh.is_finite() && dhh.is_finite() && doh > 0.3 && doh < 3.0 && dhh > 0.3 && dhh < 3.0) {
        return Err(invalid("unphysical water equilibrium length"));
    }
    let rc = 0.5 * dhh;
    let rb0_sq = doh * doh - rc * rc;
    if !(rb0_sq > 1e-20) {
        return Err(invalid("H-H target incompatible with O-H target"));
    }

    let xb0 = old_h1.x - old_o.x;
    let yb0 = old_h1.y - old_o.y;
    let zb0 = old_h1.z - old_o.z;
    let xc0 = old_h2.x - old_o.x;
    let yc0 = old_h2.y - old_o.y;
    let zc0 = old_h2.z - old_o.z;
    let xp0 = sub(qo, old_o);
    let xp1 = sub(q1, old_h1);
    let xp2 = sub(q2, old_h2);
    let inv_total = 1.0 / total;
    let xcom = (xp0.x * mo + (xb0 + xp1.x) * m1 + (xc0 + xp2.x) * m2) * inv_total;
    let ycom = (xp0.y * mo + (yb0 + xp1.y) * m1 + (yc0 + xp2.y) * m2) * inv_total;
    let zcom = (xp0.z * mo + (zb0 + xp1.z) * m1 + (zc0 + xp2.z) * m2) * inv_total;

    let xa1 = xp0.x - xcom;
    let ya1 = xp0.y - ycom;
    let za1 = xp0.z - zcom;
    let xb1 = xb0 + xp1.x - xcom;
    let yb1 = yb0 + xp1.y - ycom;
    let zb1 = zb0 + xp1.z - zcom;
    let xc1 = xc0 + xp2.x - xcom;
    let yc1 = yc0 + xp2.y - ycom;
    let zc1 = zc0 + xp2.z - zcom;

    // Old-water normal, followed by the mass-weighted COM axis. These three
    // cross products are the SETTLE reference frame; no best-fit rotation is
    // involved.
    let zaks = Vec3 {
        x: yb0 * zc0 - zb0 * yc0,
        y: zb0 * xc0 - xb0 * zc0,
        z: xb0 * yc0 - yb0 * xc0,
    };
    let xaks = cross(
        Vec3 {
            x: xa1,
            y: ya1,
            z: za1,
        },
        zaks,
    );
    let yaks = cross(zaks, xaks);
    let ax = norm(xaks).ok_or_else(|| invalid("degenerate old water triangle"))?;
    let ay = norm(yaks).ok_or_else(|| invalid("degenerate old water triangle"))?;
    let az = norm(zaks).ok_or_else(|| invalid("degenerate old water triangle"))?;
    let project = |v: Vec3| (dot(ax, v), dot(ay, v), dot(az, v));
    let (xb0d, yb0d, _) = project(Vec3 {
        x: xb0,
        y: yb0,
        z: zb0,
    });
    let (xc0d, yc0d, _) = project(Vec3 {
        x: xc0,
        y: yc0,
        z: zc0,
    });
    let (_, _, za1d) = project(Vec3 {
        x: xa1,
        y: ya1,
        z: za1,
    });
    let (xb1d, yb1d, zb1d) = project(Vec3 {
        x: xb1,
        y: yb1,
        z: zb1,
    });
    let (xc1d, yc1d, zc1d) = project(Vec3 {
        x: xc1,
        y: yc1,
        z: zc1,
    });

    let rb = rb0_sq.sqrt();
    let ra = rb * (m1 + m2) * inv_total;
    let rb = rb - ra;
    let sinphi = za1d / ra;
    let cosphi2 = 1.0 - sinphi * sinphi;
    if !(cosphi2 > 1e-14) {
        return Err(invalid("water displacement leaves the SETTLE branch (phi)"));
    }
    let cosphi = cosphi2.sqrt();
    let sinpsi = (zb1d - zc1d) / (2.0 * rc * cosphi);
    let cospsi2 = 1.0 - sinpsi * sinpsi;
    if !(cospsi2 > 1e-14) {
        return Err(invalid("water displacement leaves the SETTLE branch (psi)"));
    }
    let cospsi = cospsi2.sqrt();
    let ya2d = ra * cosphi;
    let mut xb2d = -rc * cospsi;
    let yb2d = -rb * cosphi - rc * sinpsi * sinphi;
    let yc2d = -rb * cosphi + rc * sinpsi * sinphi;
    // The H-H quadratic correction is essential. Omitting it gives a
    // plausible-looking triangle but introduces a systematic O(dt^2)
    // irreversibility and was the source of the Phase-0 drift.
    let hh2 = 4.0 * xb2d * xb2d + (yb2d - yc2d) * (yb2d - yc2d) + (zb1d - zc1d) * (zb1d - zc1d);
    let root = 4.0 * xb2d * xb2d - hh2 + dhh * dhh;
    if !(root >= -1e-12) {
        return Err(invalid("water displacement leaves the SETTLE branch (H-H)"));
    }
    let deltx = 2.0 * xb2d + root.max(0.0).sqrt();
    xb2d -= 0.5 * deltx;

    let alpha = xb2d * (xb0d - xc0d) + yb0d * yb2d + yc0d * yc2d;
    let beta = xb2d * (yc0d - yb0d) + xb0d * yb2d + xc0d * yc2d;
    let gamma = xb0d * yb1d - xb1d * yb0d + xc0d * yc1d - xc1d * yc0d;
    let ab2 = alpha * alpha + beta * beta;
    let theta_root = ab2 - gamma * gamma;
    if !(ab2 > 1e-24 && theta_root >= -1e-12) {
        return Err(invalid(
            "water displacement leaves the SETTLE branch (theta)",
        ));
    }
    let sintheta = (alpha * gamma - beta * theta_root.max(0.0).sqrt()) / ab2;
    let costheta2 = 1.0 - sintheta * sintheta;
    if !(costheta2 > 1e-14) {
        return Err(invalid(
            "water displacement leaves the SETTLE branch (theta)",
        ));
    }
    let costheta = costheta2.sqrt();
    let xa3d = Vec3 {
        x: -ya2d * sintheta,
        y: ya2d * costheta,
        z: za1d,
    };
    let xb3d = Vec3 {
        x: xb2d * costheta - yb2d * sintheta,
        y: xb2d * sintheta + yb2d * costheta,
        z: zb1d,
    };
    let xc3d = Vec3 {
        x: -xb2d * costheta - yc2d * sintheta,
        y: -xb2d * sintheta + yc2d * costheta,
        z: zc1d,
    };
    let inverse = |v: Vec3| add(add(scale(ax, v.x), scale(ay, v.y)), scale(az, v.z));
    let xa3 = inverse(xa3d);
    let xb3 = inverse(xb3d);
    let xc3 = inverse(xc3d);
    let correction_o = sub(
        xa3,
        Vec3 {
            x: xa1,
            y: ya1,
            z: za1,
        },
    );
    let correction_h1 = sub(
        xb3,
        Vec3 {
            x: xb1,
            y: yb1,
            z: zb1,
        },
    );
    let correction_h2 = sub(
        xc3,
        Vec3 {
            x: xc1,
            y: yc1,
            z: zc1,
        },
    );
    Ok((
        [
            add(qo, correction_o),
            add(q1, correction_h1),
            add(q2, correction_h2),
        ],
        [correction_o, correction_h1, correction_h2],
    ))
}

/// Analytic SETTLE/RATTLE velocity projection for one water triangle. This is
/// OpenMM's general three-mass velocity solve, expressed in the same vector
/// convention as [`settle_triangle`]. It removes all bond-parallel relative
/// velocities while preserving the mass-weighted center-of-mass velocity.
pub fn settle_velocity_triangle(
    coords: [Vec3; 3],
    velocities: &mut [Vec3; 3],
    masses: [f64; 3],
) -> Result<f64> {
    let [a, b, c] = coords;
    let [mut va, mut vb, mut vc] = *velocities;
    let [ma, mb, mc] = masses;
    if !ma.is_finite() || !mb.is_finite() || !mc.is_finite() || ma <= 0. || mb <= 0. || mc <= 0. {
        return Err(invalid("water masses must be finite and positive"));
    }
    let unit = |v: Vec3| norm(v).ok_or_else(|| invalid("degenerate water triangle"));
    let eab = unit(sub(b, a))?;
    let ebc = unit(sub(c, b))?;
    let eca = unit(sub(a, c))?;
    let vab = dot(sub(vb, va), eab);
    let vbc = dot(sub(vc, vb), ebc);
    let vca = dot(sub(va, vc), eca);
    let ca = -dot(eab, eca);
    let cb = -dot(eab, ebc);
    let cc = -dot(ebc, eca);
    let s2a = (1.0 - ca * ca).max(0.0);
    let s2b = (1.0 - cb * cb).max(0.0);
    let s2c = (1.0 - cc * cc).max(0.0);
    let mabc_inv = 1.0 / (ma * mb * mc);
    let denom = (((s2a * mb + s2b * ma) * mc
        + (s2a * mb * mb + 2.0 * (ca * cb * cc + 1.0) * ma * mb + s2b * ma * ma))
        * mc
        + s2c * ma * mb * (ma + mb))
        * mabc_inv;
    if !(denom.is_finite() && denom > 1e-18) {
        return Err(invalid("singular SETTLE velocity system"));
    }
    let tab = ((cb * cc * ma - ca * mb - ca * mc) * vca
        + (ca * cc * mb - cb * mc - cb * ma) * vbc
        + (s2c * ma * ma * mb * mb * mabc_inv + (ma + mb + mc)) * vab)
        / denom;
    let tbc = ((ca * cb * mc - cc * mb - cc * ma) * vca
        + (s2a * mb * mb * mc * mc * mabc_inv + (ma + mb + mc)) * vbc
        + (ca * cc * mb - cb * ma - cb * mc) * vab)
        / denom;
    let tca = ((s2b * ma * ma * mc * mc * mabc_inv + (ma + mb + mc)) * vca
        + (ca * cb * mc - cc * mb - cc * ma) * vbc
        + (cb * cc * ma - ca * mb - ca * mc) * vab)
        / denom;
    va = add(va, scale(sub(scale(eab, tab), scale(eca, tca)), 1.0 / ma));
    vb = add(vb, scale(sub(scale(ebc, tbc), scale(eab, tab)), 1.0 / mb));
    vc = add(vc, scale(sub(scale(eca, tca), scale(ebc, tbc)), 1.0 / mc));
    *velocities = [va, vb, vc];
    let residual = [
        dot(sub(vb, va), sub(b, a)).abs(),
        dot(sub(vc, vb), sub(c, b)).abs(),
        dot(sub(va, vc), sub(a, c)).abs(),
    ]
    .into_iter()
    .fold(0.0, f64::max);
    Ok(residual)
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
    /// Per-water analytic-SETTLE geometry `(doh, dhh, isosceles)`, parallel
    /// to `waters`. Non-isosceles waters (never from TIP3P equilibrium
    /// targets) use the explicit iterative fallback in
    /// [`Self::rigid_water_positions`].
    water_geom: Vec<(f64, f64, bool)>,
    /// Per-water masses in O/H1/H2 order for the analytic velocity solve.
    water_masses: Vec<[f64; 3]>,
    constraints: Vec<(usize, usize, f64, f64, f64)>,
    /// Solute X-H bonds constrained alongside the waters, if any.
    solute_bonds: Vec<(usize, usize, f64, f64, f64)>,
    pub tolerance: f64,
}

impl SettleWaters {
    /// Targets measured from `coords` (the historical behavior). Prefer
    /// [`Self::from_equilibrium`] for dynamics: measuring targets on a
    /// strained snapshot pins strain into the manifold and pumps energy.
    pub fn new(waters: Vec<[usize; 3]>, coords: &[Vec3], masses: &[f64]) -> Result<Self> {
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
        let mut water_geom = Vec::with_capacity(waters.len());
        let mut water_masses = Vec::with_capacity(waters.len());
        for (w, &(oh1, oh2, hh)) in waters.iter().zip(&water) {
            let [o, h1, h2] = *w;
            // Analytic SETTLE requires an isosceles triangle; TIP3P
            // equilibrium targets satisfy this exactly. Reject anything else
            // explicitly rather than solving the wrong triangle.
            if (oh1 - oh2).abs() > 1e-9 {
                return Err(invalid(
                    "non-isosceles water target; analytic SETTLE needs oh1 == oh2",
                ));
            }
            water_geom.push((0.5 * (oh1 + oh2), hh, true));
            water_masses.push([masses[o], masses[h1], masses[h2]]);
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
            water_geom,
            water_masses,
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
        let mut water_geom = Vec::with_capacity(waters.len());
        let mut water_masses = Vec::with_capacity(waters.len());
        for &[o, h1, h2] in &waters {
            let measured = |a: usize, b: usize| {
                let d = sub(coords[a], coords[b]);
                dot(d, d).sqrt()
            };
            let (moh1, moh2, mhh) = (measured(o, h1), measured(o, h2), measured(h1, h2));
            water_geom.push((0.5 * (moh1 + moh2), mhh, (moh1 - moh2).abs() <= 1e-6));
            if o >= masses.len() || h1 >= masses.len() || h2 >= masses.len() {
                return Err(invalid("water index out of range"));
            }
            water_masses.push([masses[o], masses[h1], masses[h2]]);
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
            water_geom,
            water_masses,
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

    /// Rigid-body position update for waters: analytic SETTLE per water
    /// ([`settle_triangle`]). `drifted` holds the ordinary unconstrained drift
    /// (`old + v_half * dt`) for every atom; water entries are overwritten
    /// with the exact rigid solution, solute entries are left for the caller
    /// to SHAKE. The center of mass comes from the drift (it carries the kick
    /// impulse exactly); orientation comes from the analytic construction in
    /// the drifted triangle's plane. Unlike fitting a rotation between the
    /// old and drifted frames, this map depends only on the unconstrained
    /// triple: it is time-symmetric up to rounding and introduces no
    /// O(dt^2) deformation-contaminated rotation error (Phase-0 diagnosis:
    /// the frame fit accumulated ~3e-4 A/step/water of systematic
    /// irreversibility even with zero forces; the analytic SETTLE
    /// reversibility regression test covers this case).
    /// Non-isosceles waters take an explicit iterative SHAKE projection from
    /// the drifted positions (nearest-manifold, likewise symmetric); this
    /// path never triggers for TIP3P equilibrium targets.
    pub fn rigid_water_positions(
        &self,
        old_coords: &[Vec3],
        drifted: &[Vec3],
        coords: &mut [Vec3],
        masses: &[f64],
    ) -> Result<()> {
        self.settle_positions(old_coords, drifted, coords, masses)
    }

    /// Apply the analytic SETTLE position solve to every isosceles water.
    /// `old_coords` must satisfy the target geometry and `drifted` contains
    /// the unconstrained post-drift coordinates. The solve preserves the
    /// mass-weighted center-of-mass displacement and applies the exact
    /// three-distance projection used by OpenMM.
    pub fn settle_positions(
        &self,
        old_coords: &[Vec3],
        drifted: &[Vec3],
        coords: &mut [Vec3],
        masses: &[f64],
    ) -> Result<()> {
        for (w, &(doh, dhh, iso)) in self.waters.iter().zip(&self.water_geom) {
            let &[o, h1, h2] = w;
            if iso {
                let (corrected, _) = settle_triangle(
                    old_coords[o],
                    old_coords[h1],
                    old_coords[h2],
                    drifted[o],
                    drifted[h1],
                    drifted[h2],
                    masses[o],
                    masses[h1],
                    masses[h2],
                    doh,
                    dhh,
                )?;
                [coords[o], coords[h1], coords[h2]] = corrected;
            } else {
                // This path is retained only for legacy/non-TIP3P callers.
                // Supported TIP3P dynamics always takes the fixed-flop
                // analytic branch above.
                coords[o] = drifted[o];
                coords[h1] = drifted[h1];
                coords[h2] = drifted[h2];
                shake_positions(
                    coords,
                    &self.water_constraints(o, h1, h2),
                    self.tolerance,
                    200,
                )?;
            }
        }
        let worst = self.max_violation(coords);
        if !worst.is_finite() || worst > 1e-8 {
            return Err(invalid(format!(
                "rigid-water position residual {worst:.3e} A exceeds 1e-8"
            )));
        }
        Ok(())
    }

    /// Compatibility helper that applies SETTLE positions and, when a
    /// velocity slice is supplied, reconstructs the matching RATTLE
    /// half-step velocity from the constrained displacement. This is the
    /// operation used by a constrained velocity-Verlet drift: the position
    /// constraint impulse is `dx/dt`, so retaining the unconstrained half-step
    /// velocity would omit that impulse and produce secular energy drift.
    /// New integrator code should call [`Self::settle_positions`] after the
    /// drift and [`Self::constrain_velocities`] after the final kick. The
    /// `inverse_dt` is the inverse drift interval in ps.
    pub fn settle_positions_and_velocities(
        &self,
        old_coords: &[Vec3],
        drifted: &[Vec3],
        coords: &mut [Vec3],
        mut velocities: Option<&mut [Vec3]>,
        masses: &[f64],
        inverse_dt: f64,
    ) -> Result<()> {
        self.settle_positions(old_coords, drifted, coords, masses)?;
        if let Some(velocities) = velocities.as_deref_mut() {
            if !inverse_dt.is_finite() || inverse_dt <= 0. {
                return Err(invalid("SETTLE velocity projection requires dt > 0"));
            }
            self.velocity_from_displacement(old_coords, coords, velocities, inverse_dt)?;
        }
        Ok(())
    }

    /// Reconstruct velocities after a constrained position drift. The
    /// constrained displacement is the RATTLE position impulse divided by the
    /// drift interval. Replacing the trial velocities with this quotient is
    /// the canonical velocity-Verlet update used by the OpenMM reference
    /// integrator; it also leaves unconstrained atoms unchanged up to roundoff.
    pub fn velocity_from_displacement(
        &self,
        old_coords: &[Vec3],
        constrained_coords: &[Vec3],
        velocities: &mut [Vec3],
        inverse_dt: f64,
    ) -> Result<f64> {
        if old_coords.len() != constrained_coords.len() || velocities.len() != old_coords.len() {
            return Err(invalid("RATTLE displacement length mismatch"));
        }
        if !inverse_dt.is_finite() || inverse_dt <= 0. {
            return Err(invalid("RATTLE displacement requires dt > 0"));
        }
        // Only constrained atoms receive the position-constraint impulse.
        // Unconstrained solute atoms must retain the velocity produced by the
        // force kick and drift.  Reconstructing every velocity here silently
        // removed those atoms' dynamics and was a major source of distorted
        // NVE/NVT behavior in mixed solute/water systems.
        let mut constrained_atoms = vec![false; old_coords.len()];
        for &[o, h1, h2] in &self.waters {
            constrained_atoms[o] = true;
            constrained_atoms[h1] = true;
            constrained_atoms[h2] = true;
        }
        for &(a, b, _, _, _) in &self.solute_bonds {
            constrained_atoms[a] = true;
            constrained_atoms[b] = true;
        }
        let mut worst: f64 = 0.0;
        for (index, ((old, constrained), velocity)) in old_coords
            .iter()
            .zip(constrained_coords)
            .zip(velocities.iter_mut())
            .enumerate()
        {
            if !constrained_atoms[index] {
                continue;
            }
            let updated = scale(sub(*constrained, *old), inverse_dt);
            let delta = sub(updated, *velocity);
            worst = worst.max(dot(delta, delta).sqrt());
            *velocity = updated;
        }
        Ok(worst)
    }

    /// Full RATTLE projection (waters and solute), used to initialize
    /// Maxwell velocities and after force kicks. Waters use the closed-form
    /// SETTLE velocity solve; solute X-H bonds use the bounded general solver.
    pub fn constrain_velocities(&self, coords: &[Vec3], velocities: &mut [Vec3]) -> Result<f64> {
        let mut worst: f64 = 0.0;
        for (index, (w, &(_, _, iso))) in self.waters.iter().zip(&self.water_geom).enumerate() {
            let &[o, h1, h2] = w;
            if iso {
                let mut vv = [velocities[o], velocities[h1], velocities[h2]];
                worst = worst.max(settle_velocity_triangle(
                    [coords[o], coords[h1], coords[h2]],
                    &mut vv,
                    self.water_masses[index],
                )?);
                velocities[o] = vv[0];
                velocities[h1] = vv[1];
                velocities[h2] = vv[2];
            } else {
                worst = worst.max(rattle_velocities(
                    coords,
                    velocities,
                    &self.water_constraints(o, h1, h2),
                    1e-8,
                    200,
                )?);
            }
        }
        if !self.solute_bonds.is_empty() {
            worst = worst.max(rattle_velocities(
                coords,
                velocities,
                &self.solute_bonds,
                1e-6,
                200,
            )?);
        }
        Ok(worst)
    }

    /// RATTLE projection for solute X-H bonds only.
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

    /// The three distance constraints of one water, used by the explicit
    /// non-isosceles fallback.
    fn water_constraints(
        &self,
        o: usize,
        h1: usize,
        h2: usize,
    ) -> Vec<(usize, usize, f64, f64, f64)> {
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

    /// Maximum velocity-constraint residual `|(v_a-v_b)·r_hat|` in A/ps.
    /// This is a diagnostic only; the integrator uses the analytic water
    /// solve and bounded RATTLE projection directly.
    pub fn max_velocity_violation(&self, coords: &[Vec3], velocities: &[Vec3]) -> f64 {
        self.constraints
            .iter()
            .filter_map(|&(a, b, _, _, _)| {
                let d = sub(coords[a], coords[b]);
                let r = dot(d, d).sqrt();
                (r > 1e-14).then(|| dot(sub(velocities[a], velocities[b]), d).abs() / r)
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
        let o = Vec3 {
            x: 0.,
            y: 0.,
            z: 0.,
        };
        let h1 = Vec3 {
            x: OH,
            y: 0.,
            z: 0.,
        };
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

    /// The canonical SETTLE map is reversible for a force-free rigid water.
    #[test]
    fn analytic_settle_operator_reverses() {
        let (coords, masses) = tip3p();
        let targets = vec![(OH, OH, 2. * OH * (ANGLE.to_radians() / 2.).sin())];
        let settle =
            SettleWaters::from_equilibrium(vec![[0, 1, 2]], targets, vec![], &masses, 3).unwrap();
        let mut velocities = vec![
            Vec3 {
                x: 12.,
                y: -7.,
                z: 4.,
            },
            Vec3 {
                x: -9.,
                y: 14.,
                z: -6.,
            },
            Vec3 {
                x: 5.,
                y: 3.,
                z: -11.,
            },
        ];
        settle
            .constrain_velocities(&coords, &mut velocities)
            .unwrap();
        let mut forward = coords.clone();
        let dt = 0.002;
        let step = |coords: &mut Vec<Vec3>, velocities: &mut Vec<Vec3>| {
            let old = coords.clone();
            let mut drifted = old.clone();
            for (atom, velocity) in drifted.iter_mut().zip(velocities.iter()) {
                *atom = add(
                    *atom,
                    Vec3 {
                        x: velocity.x * dt,
                        y: velocity.y * dt,
                        z: velocity.z * dt,
                    },
                );
            }
            settle
                .settle_positions_and_velocities(
                    &old,
                    &drifted,
                    coords,
                    Some(velocities),
                    &masses,
                    1. / dt,
                )
                .unwrap();
        };
        for _ in 0..50 {
            step(&mut forward, &mut velocities);
        }
        velocities.iter_mut().for_each(|v| {
            *v = Vec3 {
                x: -v.x,
                y: -v.y,
                z: -v.z,
            }
        });
        for _ in 0..50 {
            step(&mut forward, &mut velocities);
        }
        for (ended, started) in forward.iter().zip(coords.iter()) {
            let delta = sub(*ended, *started);
            assert!(dot(delta, delta).sqrt() < 1e-10);
        }
        assert!(settle.max_violation(&forward) < 1e-10);
    }

    /// Rigid position and velocity updates conserve a force-free spinning
    /// water in kinetic energy, center-of-mass motion, and geometry.
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
            x: (masses[0] * coords[0].x + masses[1] * coords[1].x + masses[2] * coords[2].x)
                / mtriple,
            y: (masses[0] * coords[0].y + masses[1] * coords[1].y + masses[2] * coords[2].y)
                / mtriple,
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
                .settle_positions_and_velocities(
                    &old,
                    &drifted,
                    &mut coords,
                    Some(&mut vel),
                    &masses,
                    1. / dt,
                )
                .unwrap();
            settle.constrain_velocities(&coords, &mut vel).unwrap();
        }
        assert!((kinetic(&vel) - e0).abs() < 1e-9, "SETTLE must preserve K");
        assert!(settle.max_violation(&coords) < 1e-9);
    }

    #[test]
    fn velocity_projection_kills_bond_motion() {
        let (coords, masses) = tip3p();
        let settle = SettleWaters::new(vec![[0, 1, 2]], &coords, &masses).unwrap();
        let mut velocities = vec![
            Vec3 {
                x: 5.,
                y: -3.,
                z: 2.,
            },
            Vec3 {
                x: -4.,
                y: 6.,
                z: 1.,
            },
            Vec3 {
                x: 1.,
                y: 1.,
                z: -7.,
            },
        ];
        settle
            .constrain_velocities(&coords, &mut velocities)
            .unwrap();
        // Bond-parallel relative velocities must vanish; total momentum kept.
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            let d = sub(coords[a], coords[b]);
            let r = dot(d, d).sqrt();
            let rel = sub(velocities[a], velocities[b]);
            assert!((rel.x * d.x + rel.y * d.y + rel.z * d.z).abs() / r < 1e-7);
        }
    }

    #[test]
    fn displacement_velocity_reconstruction_leaves_free_atoms_untouched() {
        let (water, water_masses) = tip3p();
        let mut old = water.clone();
        old.push(Vec3 {
            x: 4.,
            y: -2.,
            z: 1.,
        });
        let mut drifted = old.clone();
        drifted[0].x += 0.01;
        drifted[1].x += 0.01;
        drifted[2].x += 0.01;
        drifted[3].x += 0.25;
        let mut corrected = drifted.clone();
        let mut masses = water_masses;
        masses.push(12.);
        let settle = SettleWaters::new(vec![[0, 1, 2]], &old, &masses).unwrap();
        settle
            .settle_positions(&old, &drifted, &mut corrected, &masses)
            .unwrap();
        let free_before = Vec3 {
            x: 7.,
            y: 8.,
            z: 9.,
        };
        let mut velocities = vec![free_before; 4];
        settle
            .velocity_from_displacement(&old, &corrected, &mut velocities, 500.)
            .unwrap();
        assert_eq!(velocities[3], free_before);
        assert!(settle.max_velocity_violation(&corrected, &velocities) < 1e-8);
    }
}
