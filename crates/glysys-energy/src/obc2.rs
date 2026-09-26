//! Reverse chain rule for OBC2: O(N²) time and O(N) scratch, independent of
//! the number of movable coordinates. Keep the force-field/clamping convention
//! identical to the scalar energy and test-only full forward-AD reference.
use super::*;
use rayon::prelude::*;

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
    if n >= 128 {
        return gradient_parallel(atoms, coordinates, options);
    }
    gradient_serial(atoms, coordinates, options)
}

fn gradient_serial(atoms: &[Atom], coordinates: &[Vec3], options: &Obc2Options) -> Vec<Vec3> {
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

/// Parallel O(N²) OBC2 reverse pass. Each Born-radius row and each polar
/// derivative row is independent; the final descreening scatter uses Rayon
/// worker-local O(N) accumulators before a deterministic ordered reduction.
/// Small systems keep the serial path to avoid thread-pool overhead.
fn gradient_parallel(atoms: &[Atom], coordinates: &[Vec3], options: &Obc2Options) -> Vec<Vec3> {
    let n = atoms.len();
    let gb_radii: Vec<_> = atoms.iter().map(Atom::gb_radius).collect();
    let descreen_radii: Vec<_> = gb_radii.iter().map(|r| (r - 0.09).max(0.1)).collect();
    let scaled_radii: Vec<_> = atoms
        .iter()
        .zip(&descreen_radii)
        .map(|(atom, radius)| radius * atom.gb_screen())
        .collect();

    // Cache each directional descreening derivative while its OBC integral
    // is already being evaluated for the Born-radius row. The reverse force
    // pass needs the same derivative, and recomputing the logarithm and
    // associated geometry for every pair was a substantial CPU cost.
    let mut descreening_derivative_over_distance = vec![0.0; n * n];
    let born_and_derivative: Vec<(f64, f64)> = descreening_derivative_over_distance
        .par_chunks_mut(n)
        .enumerate()
        .map(|(i, derivative_row)| {
            let mut sum = 0.0;
            for j in 0..n {
                if i != j {
                    let raw_distance = distance(coordinates[i], coordinates[j]);
                    let d = raw_distance.max(1.0e-8);
                    let (value, derivative) = integral(descreen_radii[i], scaled_radii[j], d);
                    sum += value;
                    if raw_distance >= 1.0e-8 {
                        derivative_row[j] = derivative / raw_distance;
                    }
                }
            }
            let psi = descreen_radii[i] * sum;
            let t = (psi - 0.8 * psi * psi + 4.85 * psi.powi(3)).tanh();
            let denominator = 1.0 / descreen_radii[i] - t / gb_radii[i];
            let born = 1.0 / denominator.max(1.0e-6);
            let derivative = if denominator >= 1.0e-6 {
                born * born
                    * (1.0 - t * t)
                    * (1.0 - 1.6 * psi + 14.55 * psi * psi)
                    * descreen_radii[i]
                    / gb_radii[i]
            } else {
                0.0
            };
            (born, derivative)
        })
        .collect();
    let born: Vec<_> = born_and_derivative.iter().map(|v| v.0).collect();
    let db_di: Vec<_> = born_and_derivative.iter().map(|v| v.1).collect();

    let dielectric = 1.0 / options.solute_dielectric - 1.0 / options.solvent_dielectric;
    let polar_rows: Vec<(Vec3, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut direct = Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
            let mut de_db = 0.0;
            for j in 0..n {
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
                de_db += de_dp * born[j] * if i == j { 2.0 } else { 1.0 };
                if i != j {
                    let factor = 2.0 * de_dz * (1.0 - 0.25 * exp);
                    add_scaled(&mut direct, delta, factor);
                }
            }
            let surface = 4.0
                * std::f64::consts::PI
                * options.surface_tension
                * (gb_radii[i] + options.probe_radius).powi(2)
                * (gb_radii[i] / born[i]).powi(6);
            de_db -= 6.0 * surface / born[i];
            (direct, de_db)
        })
        .collect();
    let mut gradient: Vec<_> = polar_rows.iter().map(|row| row.0).collect();
    let de_db: Vec<_> = polar_rows.iter().map(|row| row.1).collect();
    let descreening_factors: Vec<_> = de_db
        .iter()
        .zip(&db_di)
        .map(|(energy_derivative, born_derivative)| energy_derivative * born_derivative)
        .collect();
    // Give each output atom exclusive ownership rather than scattering to
    // both endpoints from Rayon workers and allocating one N-vector per
    // worker. For atom i, combine its outgoing and incoming pair terms in a
    // single deterministic row reduction.
    gradient.par_iter_mut().enumerate().for_each(|(i, output)| {
        for j in 0..n {
            if i == j {
                continue;
            }
            let delta = subtract(coordinates[i], coordinates[j]);
            let factor = descreening_factors[i] * descreening_derivative_over_distance[i * n + j]
                + descreening_factors[j] * descreening_derivative_over_distance[j * n + i];
            add_scaled(output, delta, factor);
        }
    });
    gradient
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rayon_gradient_matches_serial_obc2_for_protein_sized_system() {
        let system = glysys::SystemBuilder::new(glysys::BuildOptions {
            add_water: false,
            add_ions: false,
            ..Default::default()
        })
        .unwrap()
        .prepare_pdb_str(include_str!("../../../benchmarks/hydration/1UBQ.pdb"))
        .unwrap();
        let atoms = system.atoms();
        assert!(atoms.len() >= 128);
        let coordinates: Vec<_> = atoms.iter().map(Atom::position).collect();
        let options = Obc2Options::default();
        let serial = gradient_serial(atoms, &coordinates, &options);
        let parallel = gradient_parallel(atoms, &coordinates, &options);
        let max_error = serial
            .iter()
            .zip(&parallel)
            .map(|(a, b)| {
                (a.x - b.x)
                    .abs()
                    .max((a.y - b.y).abs())
                    .max((a.z - b.z).abs())
            })
            .fold(0.0_f64, f64::max);
        assert!(max_error < 1e-6, "max gradient error {max_error}");

        let born = super::super::obc2_born_radii(atoms, &coordinates);
        let dielectric = 1.0 / options.solute_dielectric - 1.0 / options.solvent_dielectric;
        let mut polar = 0.0;
        for first in 0..atoms.len() {
            for second in first..atoms.len() {
                let distance2 = if first == second {
                    0.0
                } else {
                    squared_distance(coordinates[first], coordinates[second])
                };
                let denominator = (distance2
                    + born[first]
                        * born[second]
                        * (-distance2 / (4.0 * born[first] * born[second])).exp())
                .sqrt()
                .max(1.0e-8);
                let factor = if first == second { 0.5 } else { 1.0 };
                polar -= factor
                    * COULOMB_KCAL_ANGSTROM
                    * dielectric
                    * atoms[first].charge()
                    * atoms[second].charge()
                    / denominator;
            }
        }
        let surface = atoms
            .iter()
            .zip(&born)
            .map(|(atom, born)| {
                let radius = atom.gb_radius();
                4.0 * std::f64::consts::PI
                    * options.surface_tension
                    * (radius + options.probe_radius).powi(2)
                    * (radius / born).powi(6)
            })
            .sum::<f64>();
        let actual = options.components(atoms, &coordinates);
        assert!((actual.0 - polar).abs() < 1e-7);
        assert!((actual.1 - surface).abs() < 1e-9);
    }

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
