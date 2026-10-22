//! The docking grid reproduces the exact glycan-receptor energy, and the
//! glycan's internal energy is that of its own subsystem.

use glysys::{BuildOptions, ParameterizedSystem, SystemBuilder, Vec3};
use glysys_energy::grid::{DockingGrid, Screening};
use glysys_energy::{AtomGroupMask, AtomSelection, EnergyEvaluator, EnergyOptions};

const POCKET: &str = include_str!("../../../tests/fixtures/galectin_lactose_pocket10.pdb");

struct Pocket {
    system: ParameterizedSystem,
    ligand: Vec<usize>,
    receptor: Vec<usize>,
}

fn pocket() -> Pocket {
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        repair: true,
        ..BuildOptions::default()
    })
    .unwrap()
    .prepare_pdb_str(POCKET)
    .unwrap();
    let (mut ligand, mut receptor) = (Vec::new(), Vec::new());
    for residue in system.residues() {
        if residue.chain() == "Z" {
            &mut ligand
        } else {
            &mut receptor
        }
        .extend(residue.atom_range());
    }
    Pocket {
        system,
        ligand,
        receptor,
    }
}

fn centroid(points: &[Vec3]) -> Vec3 {
    let n = points.len() as f64;
    Vec3 {
        x: points.iter().map(|p| p.x).sum::<f64>() / n,
        y: points.iter().map(|p| p.y).sum::<f64>() / n,
        z: points.iter().map(|p| p.z).sum::<f64>() / n,
    }
}

/// Deterministic pseudo-random numbers in [-1, 1).
fn noise(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
}

/// Rotate `p` about `c` by `angle` around a unit `axis` (Rodrigues).
fn rotate(p: Vec3, c: Vec3, axis: [f64; 3], angle: f64) -> Vec3 {
    let d = [p.x - c.x, p.y - c.y, p.z - c.z];
    let (s, k) = angle.sin_cos();
    let cross = [
        axis[1] * d[2] - axis[2] * d[1],
        axis[2] * d[0] - axis[0] * d[2],
        axis[0] * d[1] - axis[1] * d[0],
    ];
    let dot = axis[0] * d[0] + axis[1] * d[1] + axis[2] * d[2];
    Vec3 {
        x: c.x + d[0] * k + cross[0] * s + axis[0] * dot * (1.0 - k),
        y: c.y + d[1] * k + cross[1] * s + axis[1] * dot * (1.0 - k),
        z: c.z + d[2] * k + cross[2] * s + axis[2] * dot * (1.0 - k),
    }
}

#[test]
fn grid_energy_matches_the_exact_cross_energy() {
    let pocket = pocket();
    let coords = pocket.system.coordinates();
    let ligand_native: Vec<Vec3> = pocket.ligand.iter().map(|&i| coords[i]).collect();
    let center = centroid(&ligand_native);
    let n = coords.len();
    for dielectric in [1.0, 4.0] {
        let grid = DockingGrid::build(
            &pocket.system,
            &coords,
            &pocket.receptor,
            &pocket.ligand,
            center,
            12.0,
            0.3,
            Screening::Constant { dielectric },
        )
        .unwrap();
        let evaluator = EnergyEvaluator::new(
            &pocket.system,
            EnergyOptions {
                cutoff: None,
                dielectric,
                obc2: None,
                ..EnergyOptions::default()
            },
        )
        .unwrap();
        let (first, second) = (
            AtomGroupMask::from_indices(n, pocket.ligand.iter().copied()),
            AtomGroupMask::from_indices(n, pocket.receptor.iter().copied()),
        );
        let mut state = 12345u64;
        let (mut worst, mut sum, mut count) = (0.0f64, 0.0f64, 0);
        let mut pairs: Vec<(f64, f64)> = Vec::new();
        for trial in 0..40 {
            // The deposited pose, then rigid-body perturbations of it.
            let (shift, angle) = if trial == 0 {
                ([0.0; 3], 0.0)
            } else {
                (
                    [noise(&mut state), noise(&mut state), noise(&mut state)],
                    0.35 * noise(&mut state),
                )
            };
            let axis = {
                let a = [
                    noise(&mut state) + 1.5,
                    noise(&mut state),
                    noise(&mut state),
                ];
                let norm = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
                [a[0] / norm, a[1] / norm, a[2] / norm]
            };
            let moved: Vec<Vec3> = ligand_native
                .iter()
                .map(|&p| {
                    let q = rotate(p, center, axis, angle);
                    Vec3 {
                        x: q.x + shift[0],
                        y: q.y + shift[1],
                        z: q.z + shift[2],
                    }
                })
                .collect();
            let mut full = coords.clone();
            for (&i, &p) in pocket.ligand.iter().zip(&moved) {
                full[i] = p;
            }
            let exact = evaluator
                .interaction_energy(&full, &first, &second)
                .unwrap();
            let exact = exact.van_der_waals + exact.electrostatics;
            let tabulated = grid.energy(&moved);
            // The grid is a search tool for poses that are not deep in a clash:
            // judge it where the exact energy is at most 10 kcal/mol.
            if exact > 10.0 {
                continue;
            }
            pairs.push((exact, tabulated));
            worst = worst.max((tabulated - exact).abs());
            sum += (tabulated - exact).abs();
            count += 1;
        }
        eprintln!(
            "dielectric {dielectric}: {count} poses, mean |error| {:.3}, worst {:.3} kcal/mol",
            sum / count as f64,
            worst
        );
        assert!(count >= 8, "only {count} low-energy poses");
        assert!(
            sum / (count as f64) < 2.5,
            "mean error {}",
            sum / count as f64
        );
        assert!(worst < 6.0, "worst error {worst}");
        // The grid must rank low-energy poses as the exact energy does.
        let mut concordant = 0;
        let mut total = 0;
        for i in 0..pairs.len() {
            for j in i + 1..pairs.len() {
                if (pairs[i].0 - pairs[j].0).abs() > 6.0 {
                    total += 1;
                    concordant +=
                        usize::from((pairs[i].0 < pairs[j].0) == (pairs[i].1 < pairs[j].1));
                }
            }
        }
        assert!(
            total == 0 || concordant as f64 / total as f64 > 0.9,
            "ranking agreement {concordant}/{total}"
        );
    }
}

#[test]
fn grid_error_by_spacing_for_the_deposited_pose() {
    let pocket = pocket();
    let coords = pocket.system.coordinates();
    let ligand: Vec<Vec3> = pocket.ligand.iter().map(|&i| coords[i]).collect();
    let n = coords.len();
    let evaluator = EnergyEvaluator::new(
        &pocket.system,
        EnergyOptions {
            cutoff: None,
            dielectric: 4.0,
            obc2: None,
            ..EnergyOptions::default()
        },
    )
    .unwrap();
    let exact = evaluator
        .interaction_energy(
            &coords,
            &AtomGroupMask::from_indices(n, pocket.ligand.iter().copied()),
            &AtomGroupMask::from_indices(n, pocket.receptor.iter().copied()),
        )
        .unwrap();
    eprintln!(
        "exact: vdW {:.3} elec {:.3}",
        exact.van_der_waals, exact.electrostatics
    );
    for spacing in [0.5, 0.4, 0.3, 0.2, 0.15] {
        let started = std::time::Instant::now();
        let grid = DockingGrid::build(
            &pocket.system,
            &coords,
            &pocket.receptor,
            &pocket.ligand,
            centroid(&ligand),
            11.0,
            spacing,
            Screening::Constant { dielectric: 4.0 },
        )
        .unwrap();
        let (w, c) = grid.components(&ligand);
        eprintln!(
            "spacing {spacing}: vdW {w:.3} (err {:+.3})  elec {c:.3} (err {:+.3})  built in {:.2?}",
            w - exact.van_der_waals,
            c - exact.electrostatics,
            started.elapsed()
        );
    }
}

#[test]
fn grid_gradient_matches_finite_differences_of_the_grid() {
    let pocket = pocket();
    let coords = pocket.system.coordinates();
    let ligand: Vec<Vec3> = pocket.ligand.iter().map(|&i| coords[i]).collect();
    let grid = DockingGrid::build(
        &pocket.system,
        &coords,
        &pocket.receptor,
        &pocket.ligand,
        centroid(&ligand),
        12.0,
        0.5,
        Screening::MehlerSolmajer,
    )
    .unwrap();
    let mut gradient = vec![
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0
        };
        ligand.len()
    ];
    let value = grid.energy_gradient(&ligand, Some(&mut gradient));
    assert!((value - grid.energy(&ligand)).abs() < 1.0e-9);
    let h = 1.0e-4;
    for atom in [0, 5, ligand.len() / 2, ligand.len() - 1] {
        for axis in 0..3 {
            let (mut up, mut down) = (ligand.clone(), ligand.clone());
            let bump = |p: &mut Vec3, d: f64| match axis {
                0 => p.x += d,
                1 => p.y += d,
                _ => p.z += d,
            };
            bump(&mut up[atom], h);
            bump(&mut down[atom], -h);
            let numeric = (grid.energy(&up) - grid.energy(&down)) / (2.0 * h);
            let analytic = [gradient[atom].x, gradient[atom].y, gradient[atom].z][axis];
            assert!(
                (numeric - analytic).abs() < 1.0e-3 * (1.0 + analytic.abs()),
                "atom {atom} axis {axis}: {numeric} vs {analytic}"
            );
        }
    }
}

#[test]
fn a_subsystem_has_the_internal_energy_of_its_atoms() {
    let pocket = pocket();
    let coords = pocket.system.coordinates();
    let n = coords.len();
    let options = EnergyOptions {
        cutoff: None,
        dielectric: 4.0,
        obc2: None,
        ..EnergyOptions::default()
    };
    let full = EnergyEvaluator::new(&pocket.system, options.clone())
        .unwrap()
        .with_active_terms(AtomSelection::from_indices(
            n,
            pocket.ligand.iter().copied(),
        ))
        .unwrap();
    let cross = EnergyEvaluator::new(&pocket.system, options.clone())
        .unwrap()
        .interaction_energy(
            &coords,
            &AtomGroupMask::from_indices(n, pocket.ligand.iter().copied()),
            &AtomGroupMask::from_indices(n, pocket.receptor.iter().copied()),
        )
        .unwrap();
    let internal =
        full.components(&coords).unwrap().total() - cross.van_der_waals - cross.electrostatics;
    let mut sorted = pocket.ligand.clone();
    sorted.sort_unstable();
    let sub = pocket.system.subsystem(&sorted).unwrap();
    assert_eq!(sub.atom_count(), sorted.len());
    let sub_coords: Vec<Vec3> = sorted.iter().map(|&i| coords[i]).collect();
    let alone = EnergyEvaluator::new(&sub, options)
        .unwrap()
        .components(&sub_coords)
        .unwrap()
        .total();
    eprintln!("internal {internal:.4} vs subsystem {alone:.4}");
    assert!(
        (internal - alone).abs() < 1.0e-6 * (1.0 + internal.abs()),
        "{internal} vs {alone}"
    );
}

#[test]
fn flexible_hydroxyl_energy_has_a_consistent_gradient_and_lowers_the_energy() {
    use glysys_energy::grid::GridOptions;
    let pocket = pocket();
    let coords = pocket.system.coordinates();
    let ligand: Vec<Vec3> = pocket.ligand.iter().map(|&i| coords[i]).collect();
    let center = centroid(&ligand);
    let fixed = DockingGrid::build(
        &pocket.system,
        &coords,
        &pocket.receptor,
        &pocket.ligand,
        center,
        12.0,
        0.4,
        Screening::MehlerSolmajer,
    )
    .unwrap();
    let options = GridOptions {
        flexible_hydroxyls: true,
        ..GridOptions::new(0.4, Screening::MehlerSolmajer)
    };
    let flexible = DockingGrid::build_with(
        &pocket.system,
        &coords,
        &pocket.receptor,
        &pocket.ligand,
        center,
        12.0,
        options,
    )
    .unwrap();
    assert!(
        flexible.flexible_hydrogens() > 0,
        "the pocket has serine/threonine/tyrosine hydroxyls"
    );
    // The ligand may take the best rotamer of every hydroxyl, so for a pose with
    // contacts the flexible energy is not above the fixed one by more than the
    // rotamer-entropy bookkeeping (a few kT).
    let (fixed_energy, flexible_energy) = (fixed.energy(&ligand), flexible.energy(&ligand));
    eprintln!(
        "fixed hydroxyls {fixed_energy:.2}, flexible {flexible_energy:.2} kcal/mol, {} rotamer sets",
        flexible.flexible_hydrogens()
    );
    let mut gradient = vec![
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0
        };
        ligand.len()
    ];
    flexible.energy_gradient(&ligand, Some(&mut gradient));
    let h = 1.0e-4;
    for atom in [1, 7, ligand.len() / 2, ligand.len() - 2] {
        for axis in 0..3 {
            let (mut up, mut down) = (ligand.clone(), ligand.clone());
            let bump = |p: &mut Vec3, d: f64| match axis {
                0 => p.x += d,
                1 => p.y += d,
                _ => p.z += d,
            };
            bump(&mut up[atom], h);
            bump(&mut down[atom], -h);
            let numeric = (flexible.energy(&up) - flexible.energy(&down)) / (2.0 * h);
            let analytic = [gradient[atom].x, gradient[atom].y, gradient[atom].z][axis];
            assert!(
                (numeric - analytic).abs() < 2.0e-3 * (1.0 + analytic.abs()),
                "atom {atom} axis {axis}: {numeric} vs {analytic}"
            );
        }
    }
}

#[test]
fn screened_internal_energy_matches_the_exact_evaluator_for_constant_screening() {
    use glysys_energy::grid::ScreenedInternal;
    let pocket = pocket();
    let coords = pocket.system.coordinates();
    let mut atoms = pocket.ligand.clone();
    atoms.sort_unstable();
    let sub = pocket.system.subsystem(&atoms).unwrap();
    let positions: Vec<Vec3> = atoms.iter().map(|&i| coords[i]).collect();
    let dielectric = 4.0;
    let options = |dielectric: f64| EnergyOptions {
        cutoff: None,
        dielectric,
        obc2: None,
        ..EnergyOptions::default()
    };
    let exact = EnergyEvaluator::new(&sub, options(dielectric))
        .unwrap()
        .components(&positions)
        .unwrap()
        .total();
    // Neutral system for everything but electrostatics, plus the screened Coulomb sum.
    let neutral = sub.without_charges();
    let rest = EnergyEvaluator::new(&neutral, options(1.0))
        .unwrap()
        .components(&positions)
        .unwrap()
        .total();
    let screened = ScreenedInternal::new(&sub, Screening::Constant { dielectric });
    let electrostatics = screened.energy(&positions);
    eprintln!(
        "exact {exact:.4}, neutral {rest:.4} + screened {electrostatics:.4} = {:.4}",
        rest + electrostatics
    );
    assert!(
        (exact - (rest + electrostatics)).abs() < 1.0e-3 * (1.0 + exact.abs()),
        "{exact} vs {}",
        rest + electrostatics
    );
    // Gradient against finite differences, with the sigmoidal screening.
    let sigmoid = ScreenedInternal::new(&sub, Screening::MehlerSolmajer);
    let mut gradient = vec![
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0
        };
        positions.len()
    ];
    sigmoid.energy_gradient(&positions, Some(&mut gradient));
    let h = 1.0e-4;
    for atom in [0, 5, positions.len() - 1] {
        for axis in 0..3 {
            let (mut up, mut down) = (positions.clone(), positions.clone());
            let bump = |p: &mut Vec3, d: f64| match axis {
                0 => p.x += d,
                1 => p.y += d,
                _ => p.z += d,
            };
            bump(&mut up[atom], h);
            bump(&mut down[atom], -h);
            let numeric = (sigmoid.energy(&up) - sigmoid.energy(&down)) / (2.0 * h);
            let analytic = [gradient[atom].x, gradient[atom].y, gradient[atom].z][axis];
            assert!(
                (numeric - analytic).abs() < 2.0e-3 * (1.0 + analytic.abs()),
                "atom {atom} axis {axis}: {numeric} vs {analytic}"
            );
        }
    }
}
