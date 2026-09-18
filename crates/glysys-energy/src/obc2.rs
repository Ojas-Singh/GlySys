//! Reverse chain rule for OBC2: O(N²) time and O(N) scratch, independent of
//! the number of movable coordinates. Keep the force-field/clamping convention
//! identical to the scalar energy and test-only full forward-AD reference.
use super::*;

fn integral(radius: f64, scaled: f64, d: f64) -> (f64, f64) {
    if d + scaled <= radius {
        return (0.0, 0.0);
    }
    let candidate = (d - scaled).abs();
    let l = radius.max(candidate);
    let u = d + scaled;
    if l >= u {
        return (0.0, 0.0);
    }
    let dl = if candidate < radius {
        0.0
    } else if d < scaled {
        -1.0
    } else {
        1.0
    };
    let a = 1.0 / l;
    let b = 1.0 / u;
    let c = d - scaled * scaled / d;
    let q = b * b - a * a;
    let log = (l / u).ln();
    let value = 0.5 * (a - b + 0.25 * c * q + 0.5 * log / d);
    let derivative = 0.5
        * (-dl * a * a
            + b * b
            + 0.25
                * ((1.0 + scaled * scaled / (d * d)) * q
                    + c * (-2.0 * b * b * b + 2.0 * dl * a * a * a))
            + 0.5 * ((dl * a - b) / d - log / (d * d)));
    (value, derivative)
}

pub(super) fn gradient(atoms: &[Atom], coordinates: &[Vec3], options: &Obc2Options) -> Vec<Vec3> {
    let n = atoms.len();
    let mut born = vec![0.0; n];
    let mut db_di = vec![0.0; n];
    for i in 0..n {
        let radius = (atoms[i].gb_radius() - 0.09).max(0.1);
        let mut sum = 0.0;
        for j in 0..n {
            if i != j {
                sum += integral(
                    radius,
                    (atoms[j].gb_radius() - 0.09).max(0.1) * atoms[j].gb_screen(),
                    distance(coordinates[i], coordinates[j]).max(1.0e-8),
                )
                .0;
            }
        }
        let psi = radius * sum;
        let t = (psi - 0.8 * psi * psi + 4.85 * psi.powi(3)).tanh();
        let denominator = 1.0 / radius - t / atoms[i].gb_radius();
        born[i] = 1.0 / denominator.max(1.0e-6);
        if denominator >= 1.0e-6 {
            db_di[i] =
                born[i] * born[i] * (1.0 - t * t) * (1.0 - 1.6 * psi + 14.55 * psi * psi) * radius
                    / atoms[i].gb_radius();
        }
    }
    let mut gradient = vec![
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0
        };
        n
    ];
    let mut de_db = vec![0.0; n];
    let dielectric = 1.0 / options.solute_dielectric - 1.0 / options.solvent_dielectric;
    for i in 0..n {
        for j in i..n {
            let delta = subtract(coordinates[i], coordinates[j]);
            let r2 = dot(delta, delta);
            let p = born[i] * born[j];
            let exp = (-r2 / (4.0 * p)).exp();
            let f = (r2 + p * exp).sqrt();
            if f < 1.0e-8 {
                continue;
            }
            let coefficient = -(if i == j { 0.5 } else { 1.0 })
                * COULOMB_KCAL_ANGSTROM
                * dielectric
                * atoms[i].charge()
                * atoms[j].charge();
            let de_dz = -0.5 * coefficient / (f * f * f);
            let de_dp = de_dz * exp * (1.0 + r2 / (4.0 * p));
            de_db[i] += de_dp * born[j];
            de_db[j] += de_dp * born[i];
            if i != j {
                let factor = 2.0 * de_dz * (1.0 - 0.25 * exp);
                add_scaled(&mut gradient[i], delta, factor);
                add_scaled(&mut gradient[j], delta, -factor);
            }
        }
        let radius = atoms[i].gb_radius();
        let surface = 4.0
            * std::f64::consts::PI
            * options.surface_tension
            * (radius + options.probe_radius).powi(2)
            * (radius / born[i]).powi(6);
        de_db[i] -= 6.0 * surface / born[i];
    }
    for i in 0..n {
        let radius = (atoms[i].gb_radius() - 0.09).max(0.1);
        for j in 0..n {
            if i == j {
                continue;
            }
            let delta = subtract(coordinates[i], coordinates[j]);
            let d = norm(delta);
            if d < 1.0e-8 {
                continue;
            }
            let derivative = integral(
                radius,
                (atoms[j].gb_radius() - 0.09).max(0.1) * atoms[j].gb_screen(),
                d,
            )
            .1;
            let factor = de_db[i] * db_di[i] * derivative / d;
            add_scaled(&mut gradient[i], delta, factor);
            add_scaled(&mut gradient[j], delta, -factor);
        }
    }
    gradient
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn radial_derivative_matches_finite_difference_in_all_branches() {
        for r in [0.1_f64, 1.4, 2.0] {
            for s in [0.1, 1.2, 2.1] {
                for d in [0.03, 0.4, 1.0, 1.7, 2.8, 8.0] {
                    if (d + s - r).abs() < 1e-4 || ((d - s).abs() - r).abs() < 1e-4 {
                        continue;
                    }
                    let (_, actual) = integral(r, s, d);
                    let expected = (integral(r, s, d + 1e-6).0 - integral(r, s, d - 1e-6).0) / 2e-6;
                    assert!(
                        (actual - expected).abs() < 1e-6,
                        "r={r} s={s} d={d} actual={actual} expected={expected}"
                    );
                }
            }
        }
    }
}
