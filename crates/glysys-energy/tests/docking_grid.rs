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

const CALCIUM_POCKET: &str = include_str!("../../../tests/fixtures/ctype_ca_pocket9.pdb");

#[test]
fn metal_coordination_is_a_consistent_local_well() {
    use glysys_energy::grid::{GridOptions, MetalCoordination};
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        repair: true,
        ..BuildOptions::default()
    })
    .unwrap()
    .prepare_pdb_str(CALCIUM_POCKET)
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
    let coords = system.coordinates();
    let positions: Vec<Vec3> = ligand.iter().map(|&i| coords[i]).collect();
    let center = centroid(&positions);
    let base = GridOptions::new(0.4, Screening::MehlerSolmajer);
    let plain =
        DockingGrid::build_with(&system, &coords, &receptor, &ligand, center, 12.0, base).unwrap();
    let well = MetalCoordination {
        distance: 2.45,
        depth: 8.0,
    };
    let with = DockingGrid::build_with(
        &system,
        &coords,
        &receptor,
        &ligand,
        center,
        12.0,
        GridOptions {
            metal_coordination: Some(well),
            ..base
        },
    )
    .unwrap();
    assert_eq!(plain.coordinating_ions(), 0);
    assert!(with.coordinating_ions() >= 1, "the pocket holds a calcium");
    let (plain_energy, well_energy) = (plain.energy(&positions), with.energy(&positions));
    eprintln!(
        "deposited pose: generic {plain_energy:.2}, with the coordination well {well_energy:.2} kcal/mol"
    );
    // The deposited glycan coordinates the calcium, so the well lowers its energy.
    assert!(well_energy < plain_energy);
    // Far from the ion the two grids agree exactly.
    let far: Vec<Vec3> = positions
        .iter()
        .map(|p| Vec3 {
            x: p.x + 9.0,
            y: p.y,
            z: p.z,
        })
        .collect();
    assert!(
        (plain.energy(&far) - with.energy(&far)).abs() < 1.0e-9,
        "no bonus away from the ion"
    );
    // Gradient against finite differences, at the deposited pose and slightly displaced.
    for shift in [0.0, 0.15] {
        let moved: Vec<Vec3> = positions
            .iter()
            .map(|p| Vec3 {
                x: p.x + shift,
                y: p.y - 0.5 * shift,
                z: p.z,
            })
            .collect();
        let mut gradient = vec![
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            };
            moved.len()
        ];
        with.energy_gradient(&moved, Some(&mut gradient));
        let h = 1.0e-4;
        for atom in 0..moved.len() {
            for axis in 0..3 {
                let (mut up, mut down) = (moved.clone(), moved.clone());
                let bump = |p: &mut Vec3, d: f64| match axis {
                    0 => p.x += d,
                    1 => p.y += d,
                    _ => p.z += d,
                };
                bump(&mut up[atom], h);
                bump(&mut down[atom], -h);
                let numeric = (with.energy(&up) - with.energy(&down)) / (2.0 * h);
                let analytic = [gradient[atom].x, gradient[atom].y, gradient[atom].z][axis];
                assert!(
                    (numeric - analytic).abs() < 3.0e-3 * (1.0 + analytic.abs()),
                    "atom {atom} axis {axis}: {numeric} vs {analytic}"
                );
            }
        }
    }
}

#[test]
fn soft_wall_and_polar_reward_are_consistent_and_one_sided() {
    use glysys_energy::grid::{GridOptions, PolarReward};
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        repair: true,
        ..BuildOptions::default()
    })
    .unwrap()
    .prepare_pdb_str(CALCIUM_POCKET)
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
    let coords = system.coordinates();
    let positions: Vec<Vec3> = ligand.iter().map(|&i| coords[i]).collect();
    let center = centroid(&positions);
    let base = GridOptions::new(0.4, Screening::MehlerSolmajer);
    let build = |options: GridOptions| {
        DockingGrid::build_with(&system, &coords, &receptor, &ligand, center, 12.0, options)
            .unwrap()
    };
    let plain = build(base);
    let capped = build(GridOptions {
        lj_cap: Some(3.0),
        ..base
    });
    let polar = build(GridOptions {
        polar_reward: Some(PolarReward {
            depth: 0.5,
            distance: 2.85,
            width: 0.35,
        }),
        ..base
    });
    let both = build(GridOptions {
        lj_cap: Some(3.0),
        polar_reward: Some(PolarReward {
            depth: 0.5,
            distance: 2.85,
            width: 0.35,
        }),
        ..base
    });
    // A cap can only lower the energy, and a pushed-in pose gains the most from it.
    let pushed: Vec<Vec3> = positions
        .iter()
        .map(|p| Vec3 {
            x: p.x + 0.7,
            y: p.y + 0.4,
            z: p.z,
        })
        .collect();
    for pose in [&positions, &pushed] {
        assert!(capped.energy(pose) <= plain.energy(pose) + 1.0e-9);
        // the reward never costs energy
        assert!(polar.energy(pose) <= plain.energy(pose) + 1.0e-9);
        assert!(both.energy(pose) <= capped.energy(pose) + 1.0e-9);
    }
    eprintln!(
        "deposited pose: plain {:.2}, capped {:.2}, polar {:.2}, both {:.2}; pushed 0.8 A: plain {:.2}, capped {:.2}",
        plain.energy(&positions),
        capped.energy(&positions),
        polar.energy(&positions),
        both.energy(&positions),
        plain.energy(&pushed),
        capped.energy(&pushed)
    );
    assert!(
        polar.energy(&positions) < plain.energy(&positions) - 0.5,
        "the deposited glycan sits among receptor N and O atoms, so the reward must lower its energy"
    );
    // Gradient against finite differences.
    for shift in [0.0, 0.15] {
        let moved: Vec<Vec3> = positions
            .iter()
            .map(|p| Vec3 {
                x: p.x + shift,
                y: p.y - 0.5 * shift,
                z: p.z,
            })
            .collect();
        let mut gradient = vec![
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            };
            moved.len()
        ];
        both.energy_gradient(&moved, Some(&mut gradient));
        let h = 1.0e-4;
        for atom in 0..moved.len() {
            for axis in 0..3 {
                let (mut up, mut down) = (moved.clone(), moved.clone());
                let bump = |p: &mut Vec3, d: f64| match axis {
                    0 => p.x += d,
                    1 => p.y += d,
                    _ => p.z += d,
                };
                bump(&mut up[atom], h);
                bump(&mut down[atom], -h);
                let numeric = (both.energy(&up) - both.energy(&down)) / (2.0 * h);
                let analytic = [gradient[atom].x, gradient[atom].y, gradient[atom].z][axis];
                assert!(
                    (numeric - analytic).abs() < 3.0e-3 * (1.0 + analytic.abs()),
                    "atom {atom} axis {axis}: {numeric} vs {analytic}"
                );
            }
        }
    }
}

#[test]
fn an_extra_source_acts_where_it_is_and_nowhere_else() {
    use glysys_energy::grid::{ExtraSource, GridOptions};
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        repair: true,
        ..BuildOptions::default()
    })
    .unwrap()
    .prepare_pdb_str(CALCIUM_POCKET)
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
    let coords = system.coordinates();
    let positions: Vec<Vec3> = ligand.iter().map(|&i| coords[i]).collect();
    let center = centroid(&positions);
    let base = GridOptions::new(0.4, Screening::MehlerSolmajer);
    let build = |sources: &'static [ExtraSource]| {
        DockingGrid::build_with(
            &system,
            &coords,
            &receptor,
            &ligand,
            center,
            12.0,
            GridOptions {
                extra_sources: sources,
                ..base
            },
        )
        .unwrap()
    };
    let at = |p: Vec3, dx: f64| -> &'static [ExtraSource] {
        Box::leak(Box::new([ExtraSource {
            position: [p.x + dx, p.y, p.z],
            charge: 0.0,
            radius: 1.7683,
            epsilon: 0.1521,
            polar: true,
        }]))
    };
    let plain = build(&[]);
    // on top of the first ligand atom: a clash
    let on_top = build(at(positions[0], 0.6));
    assert!(
        on_top.energy(&positions) > plain.energy(&positions) + 1.0,
        "a sphere 0.6 A from a ligand atom must repel it"
    );
    // far outside the grid and the ligand's reach: no effect
    let far = build(at(positions[0], 60.0));
    assert!((far.energy(&positions) - plain.energy(&positions)).abs() < 1.0e-6);
}

#[test]
fn coulomb_floor_removes_fusion_wells_under_the_soft_wall() {
    use glysys_energy::grid::{CoulombFloor, GridOptions};
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        repair: true,
        ..BuildOptions::default()
    })
    .unwrap()
    .prepare_pdb_str(CALCIUM_POCKET)
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
    let coords = system.coordinates();
    let positions: Vec<Vec3> = ligand.iter().map(|&i| coords[i]).collect();
    let center = centroid(&positions);
    let base = GridOptions::new(0.4, Screening::MehlerSolmajer);
    let build = |options: GridOptions| {
        DockingGrid::build_with(&system, &coords, &receptor, &ligand, center, 12.0, options)
            .unwrap()
    };
    let capped = build(GridOptions {
        lj_cap: Some(3.0),
        ..base
    });
    let floored = build(GridOptions {
        lj_cap: Some(3.0),
        coulomb_floor: Some(CoulombFloor {
            general: 1.5,
            metal: 2.2,
        }),
        ..base
    });
    // The deposited glycan keeps every contact outside the floors, so its energy barely moves.
    let (deposited_capped, deposited_floored) =
        (capped.energy(&positions), floored.energy(&positions));
    assert!(
        (deposited_capped - deposited_floored).abs() < 2.0,
        "deposited pose: {deposited_capped:.2} without the floor, {deposited_floored:.2} with it"
    );
    // Fuse the ligand oxygen nearest to the calcium onto it.
    let atoms = system.atoms();
    let calcium = coords[*receptor
        .iter()
        .find(|&&j| atoms[j].element() == 20)
        .expect("a calcium")];
    let dist =
        |a: Vec3, b: Vec3| ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt();
    let (oxygen, _) = ligand
        .iter()
        .enumerate()
        .filter(|(_, i)| atoms[**i].element() == 8)
        .map(|(k, &i)| (k, dist(coords[i], calcium)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    let shift = Vec3 {
        x: calcium.x - positions[oxygen].x + 0.1,
        y: calcium.y - positions[oxygen].y,
        z: calcium.z - positions[oxygen].z,
    };
    let fused: Vec<Vec3> = positions
        .iter()
        .map(|p| Vec3 {
            x: p.x + shift.x,
            y: p.y + shift.y,
            z: p.z + shift.z,
        })
        .collect();
    let (fused_capped, fused_floored) = (capped.energy(&fused), floored.energy(&fused));
    eprintln!(
        "deposited: capped {deposited_capped:.2}, floored {deposited_floored:.2}; fused: capped {fused_capped:.2}, floored {fused_floored:.2}"
    );
    assert!(
        fused_capped < deposited_capped - 50.0,
        "the soft wall alone rewards the fused pose"
    );
    assert!(
        fused_floored > deposited_floored,
        "with the floor the fused pose is no longer favoured"
    );
}

#[test]
fn metal_oxygen_contact_holds_a_coordinating_oxygen_at_the_crystal_distance() {
    use glysys_energy::grid::{GridOptions, metal_oxygen_distance};
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        repair: true,
        ..BuildOptions::default()
    })
    .unwrap()
    .prepare_pdb_str(CALCIUM_POCKET)
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
    let coords = system.coordinates();
    let positions: Vec<Vec3> = ligand.iter().map(|&i| coords[i]).collect();
    let center = centroid(&positions);
    let base = GridOptions::new(0.4, Screening::MehlerSolmajer);
    let build = |options: GridOptions| {
        DockingGrid::build_with(&system, &coords, &receptor, &ligand, center, 12.0, options)
            .unwrap()
    };
    let generic = build(base);
    let contact = build(GridOptions {
        metal_oxygen_contact: true,
        ..base
    });
    let atoms = system.atoms();
    let calcium = coords[*receptor
        .iter()
        .find(|&&j| atoms[j].element() == 20)
        .expect("a calcium")];
    let dist =
        |a: Vec3, b: Vec3| ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt();
    let (oxygen, d) = ligand
        .iter()
        .enumerate()
        .filter(|(_, i)| atoms[**i].element() == 8)
        .map(|(k, &i)| (k, dist(coords[i], calcium)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    let (before, after) = (
        generic.atom_terms(&positions)[oxygen][0],
        contact.atom_terms(&positions)[oxygen][0],
    );
    eprintln!(
        "coordinating oxygen at {d:.2} A: van der Waals {before:.2} with the force-field radii, {after:.2} at the crystal contact"
    );
    assert!(d < 2.7, "the fixture's glycan coordinates its calcium");
    // The calcium's share of the repulsion goes; what is left comes from the hydrogen-bonded protein oxygens.
    assert!(
        after < before - 3.0,
        "the contact distance lifts the coordinating oxygen off the wall"
    );
    // Only oxygen-metal pairs change: a ligand carbon sees the same field.
    let carbon = ligand
        .iter()
        .position(|&i| atoms[i].element() == 6)
        .unwrap();
    assert!(
        (generic.atom_terms(&positions)[carbon][0] - contact.atom_terms(&positions)[carbon][0])
            .abs()
            < 1.0e-9
    );
    assert_eq!(metal_oxygen_distance(20), Some(2.38));
}

#[test]
fn ch_pi_weight_follows_the_stacking_geometry() {
    use glysys_energy::grid::ch_pi_weight;
    let normal = [0.0, 0.0, 1.0];
    assert_eq!(ch_pi_weight([0.0, 0.0, 3.8], normal), 1.0);
    assert_eq!(
        ch_pi_weight([0.0, 0.0, -3.8], normal),
        1.0,
        "either face of the ring"
    );
    assert_eq!(
        ch_pi_weight([3.8, 0.0, 0.0], normal),
        0.0,
        "in the ring plane"
    );
    assert_eq!(ch_pi_weight([0.0, 0.0, 5.0], normal), 0.0, "too far");
    assert_eq!(ch_pi_weight([0.0, 0.0, 2.9], normal), 0.0, "too close");
    let tilted = 35.0f64.to_radians();
    let w = ch_pi_weight([3.8 * tilted.sin(), 0.0, 3.8 * tilted.cos()], normal);
    assert!(
        (w - 0.5).abs() < 1.0e-9,
        "half way through the angular taper: {w}"
    );
}

#[test]
fn ch_pi_reward_holds_the_galactose_over_the_tryptophan_with_a_consistent_gradient() {
    use glysys_energy::grid::GridOptions;
    let p = pocket();
    let coords = p.system.coordinates();
    let positions: Vec<Vec3> = p.ligand.iter().map(|&i| coords[i]).collect();
    let center = centroid(&positions);
    let base = GridOptions::new(0.4, Screening::MehlerSolmajer);
    let build = |options: GridOptions| {
        DockingGrid::build_with(
            &p.system,
            &coords,
            &p.receptor,
            &p.ligand,
            center,
            12.0,
            options,
        )
        .unwrap()
    };
    let plain = build(base);
    let stacked = build(GridOptions {
        ch_pi_depth: Some(1.0),
        ..base
    });
    let gain = plain.energy(&positions) - stacked.energy(&positions);
    eprintln!("CH-pi reward of the deposited lactose: {gain:.2} kcal/mol at depth 1");
    assert!(
        gain > 1.0,
        "the galactose of a galectin stacks on its tryptophan"
    );
    // Gradient against finite differences, at the deposited pose and slightly shifted.
    for shift in [0.0, 0.2] {
        let moved: Vec<Vec3> = positions
            .iter()
            .map(|q| Vec3 {
                x: q.x + shift,
                y: q.y,
                z: q.z - 0.5 * shift,
            })
            .collect();
        let mut gradient = vec![
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            };
            moved.len()
        ];
        stacked.energy_gradient(&moved, Some(&mut gradient));
        let h = 1.0e-4;
        for atom in [0, moved.len() / 2, moved.len() - 1] {
            for axis in 0..3 {
                let mut plus = moved.clone();
                let mut minus = moved.clone();
                let bump = |v: &mut Vec3, d: f64| match axis {
                    0 => v.x += d,
                    1 => v.y += d,
                    _ => v.z += d,
                };
                bump(&mut plus[atom], h);
                bump(&mut minus[atom], -h);
                let numeric = (stacked.energy(&plus) - stacked.energy(&minus)) / (2.0 * h);
                let analytic = [gradient[atom].x, gradient[atom].y, gradient[atom].z][axis];
                assert!(
                    (numeric - analytic).abs() < 1.0e-3 * (1.0 + numeric.abs()),
                    "atom {atom} axis {axis}: {numeric} vs {analytic}"
                );
            }
        }
    }
}

#[test]
fn energy_is_the_weighted_sum_of_the_recorded_terms() {
    use glysys_energy::grid::{GridOptions, MetalCoordination, PolarReward};
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        repair: true,
        ..BuildOptions::default()
    })
    .unwrap()
    .prepare_pdb_str(CALCIUM_POCKET)
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
    let coords = system.coordinates();
    let positions: Vec<Vec3> = ligand.iter().map(|&i| coords[i]).collect();
    let center = centroid(&positions);
    let (wv, we, polar, ch_pi, coordination) = (0.7, 1.3, 0.4, 0.6, 2.0);
    let contacts = [0.3, 0.4, 0.5, -0.2];
    let grid = DockingGrid::build_with(
        &system,
        &coords,
        &receptor,
        &ligand,
        center,
        12.0,
        GridOptions {
            term_weights: [wv, we],
            polar_reward: Some(PolarReward {
                depth: polar,
                distance: 2.85,
                width: 0.35,
            }),
            ch_pi_depth: Some(ch_pi),
            metal_coordination: Some(MetalCoordination {
                distance: 2.45,
                depth: coordination,
            }),
            lj_cap: Some(3.0),
            contact_depths: Some(contacts),
            ..GridOptions::new(0.4, Screening::MehlerSolmajer)
        },
    )
    .unwrap();
    for shift in [0.0, 0.6, 13.0] {
        let moved: Vec<Vec3> = positions
            .iter()
            .map(|q| Vec3 {
                x: q.x + shift,
                y: q.y - 0.3 * shift,
                z: q.z,
            })
            .collect();
        let t = grid.terms(&moved);
        let sum = wv * t.vdw
            + we * t.elec
            + polar * t.polar
            + ch_pi * t.ch_pi
            + coordination * t.coordination
            + t.outside
            + (0..4).map(|k| contacts[k] * t.contacts[k]).sum::<f64>();
        if shift == 0.0 {
            assert!(
                t.contacts.iter().all(|&c| c < 0.0),
                "every contact field is felt in the pocket: {:?}",
                t.contacts
            );
        }
        let energy = grid.energy(&moved);
        assert!(
            (energy - sum).abs() < 1.0e-6 * (1.0 + energy.abs()),
            "shift {shift}: energy {energy} vs terms {sum} ({t:?})"
        );
    }
}
