use glysys::{BuildOptions, SystemBuilder, Vec3};
use glysys_energy::{EnergyOptions, geometry::KinematicTree, scoring::*};
use std::sync::Arc;
fn scene() -> PreparedScene {
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        ..Default::default()
    })
    .unwrap()
    .prepare_pdb_str(include_str!("../../../tests/fixtures/glycan.pdb"))
    .unwrap();
    PreparedScene::new(
        Arc::new(system),
        EnergyOptions::default(),
        Boundary::NonPeriodic,
    )
    .unwrap()
}
#[test]
fn topology_tree_preserves_rings_and_cartesian_chain_rule() {
    let scene = scene();
    let atom = |name: &str| {
        scene
            .system
            .atoms()
            .iter()
            .position(|a| a.name() == name)
            .unwrap()
    };
    assert!(KinematicTree::new(&scene.system, &[[atom("C1"), atom("C2")]]).is_err());
    let tree = KinematicTree::new(&scene.system, &[[atom("C5"), atom("C6")]]).unwrap();
    assert_eq!(tree.rigid_fragments.len(), 2);
    let mut pose = Pose::cartesian(1, scene.system.coordinates());
    pose.torsions = tree.updates(&[0.3]).unwrap();
    let eval = PreparedEvaluator::new(scene, ScoreModel::amber()).unwrap();
    let result = eval
        .evaluate(
            &PoseBatch {
                poses: vec![pose.clone()],
            },
            &EvaluationRequest {
                pose_derivatives: true,
                ..Default::default()
            },
        )
        .unwrap();
    let mut plus = pose.clone();
    let mut minus = pose;
    plus.torsions[0].radians += 1e-5;
    minus.torsions[0].radians -= 1e-5;
    let values = eval
        .evaluate(
            &PoseBatch {
                poses: vec![plus, minus],
            },
            &Default::default(),
        )
        .unwrap();
    let finite = (values[0].total - values[1].total) / 2e-5;
    let analytic = result[0].pose_derivatives.as_ref().unwrap().torsions[0];
    assert!(
        (finite - analytic).abs() < 1e-3,
        "torsion {finite} {analytic}"
    );
}
#[test]
fn rigid_pose_rotation_and_translation_derivatives() {
    let scene = scene();
    let mut pose = Pose::cartesian(2, scene.system.coordinates());
    pose.transformed_atoms = (3..8).collect();
    pose.transform.translation = Vec3 {
        x: 0.2,
        y: 0.1,
        z: 0.,
    };
    let eval = PreparedEvaluator::new(scene, ScoreModel::amber()).unwrap();
    let result = eval
        .evaluate(
            &PoseBatch {
                poses: vec![pose.clone()],
            },
            &EvaluationRequest {
                pose_derivatives: true,
                ..Default::default()
            },
        )
        .unwrap();
    let d = result[0].pose_derivatives.as_ref().unwrap();
    for rotation in [false, true] {
        let mut plus = pose.clone();
        let mut minus = pose.clone();
        if rotation {
            let angle = 1e-6_f64;
            plus.transform.rotation = [
                [angle.cos(), -angle.sin(), 0.],
                [angle.sin(), angle.cos(), 0.],
                [0., 0., 1.],
            ];
            minus.transform.rotation = [
                [angle.cos(), angle.sin(), 0.],
                [-angle.sin(), angle.cos(), 0.],
                [0., 0., 1.],
            ];
        } else {
            plus.transform.translation.x += 1e-6;
            minus.transform.translation.x -= 1e-6;
        }
        let values = eval
            .evaluate(
                &PoseBatch {
                    poses: vec![plus, minus],
                },
                &Default::default(),
            )
            .unwrap();
        let finite = (values[0].total - values[1].total) / 2e-6;
        let analytic = if rotation {
            d.rotation.z
        } else {
            d.translation.x
        };
        assert!(
            (finite - analytic).abs() < 1e-3 + 1e-6 * analytic.abs(),
            "pose {finite} {analytic}"
        );
    }
}
#[test]
fn exported_generated_hydrogens_retain_evaluated_coordinates() {
    let options = BuildOptions {
        add_water: false,
        add_ions: false,
        ..Default::default()
    };
    let source = include_str!("../../../tests/fixtures/dipeptide.pdb");
    let mut structure = glysys::read_pdb_str(source, &options).unwrap();
    let original = structure.atoms().len();
    let mut system = SystemBuilder::new(options)
        .unwrap()
        .prepare_structure(&structure)
        .unwrap();
    let mut coordinates = system.coordinates();
    for (atom, p) in system.atoms().iter().zip(&mut coordinates) {
        if atom.element() == 1 {
            p.x += 0.123;
        }
    }
    system.set_coordinates(&coordinates).unwrap();
    structure
        .update_with_parameterized_hydrogens(&system)
        .unwrap();
    assert!(structure.atoms().len() > original);
    for atom in system.atoms() {
        let r = &system.residues()[atom.residue_index()];
        let id = glysys::ResidueId {
            chain: r.chain().into(),
            number: r.number(),
            insertion_code: r.insertion_code(),
        };
        let source = structure.find_atom(&id, atom.name()).unwrap();
        assert_eq!(structure.atom_position(source).unwrap(), atom.position());
    }
}

/// The fused single-pass energy-and-gradient must agree with the two passes it replaces, which the
/// docking search relies on: it is adopted only because it changes nothing.
#[test]
fn fused_energy_and_gradient_matches_the_two_pass_path() {
    use glysys::SystemBuilder;
    use glysys_energy::EnergyEvaluator;
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        ..Default::default()
    })
    .unwrap()
    .prepare_pdb_str(include_str!("../../../tests/fixtures/glycan.pdb"))
    .unwrap();
    for (dielectric, charged) in [(1.0, false), (4.0, true)] {
        let subject = if charged { system.clone() } else { system.without_charges() };
        let evaluator = EnergyEvaluator::new(
            &subject,
            EnergyOptions {
                cutoff: None,
                dielectric,
                obc2: None,
                ..EnergyOptions::default()
            },
        )
        .unwrap();
        let base = subject.coordinates();
        // the deposited geometry and a displaced one, so no term sits at a stationary point
        for shift in [0.0_f64, 0.21] {
            let coords: Vec<Vec3> = base
                .iter()
                .enumerate()
                .map(|(i, p)| Vec3 {
                    x: p.x + shift * ((i % 3) as f64 - 1.0),
                    y: p.y - shift * ((i % 5) as f64 - 2.0) * 0.5,
                    z: p.z + shift * ((i % 7) as f64 - 3.0) * 0.25,
                })
                .collect();
            let expected_energy = evaluator.components(&coords).unwrap().total();
            let expected_gradient = evaluator.gradient_only(&coords).unwrap();
            let mut fused = vec![Vec3 { x: 0.0, y: 0.0, z: 0.0 }; coords.len()];
            let energy = evaluator.total_and_gradient_into(&coords, &mut fused).unwrap();
            assert_eq!(
                energy.to_bits(),
                expected_energy.to_bits(),
                "dielectric {dielectric} charged {charged} shift {shift}: energy {energy} vs {expected_energy}"
            );
            for (i, (a, b)) in fused.iter().zip(&expected_gradient).enumerate() {
                for (axis, (x, y)) in [(a.x, b.x), (a.y, b.y), (a.z, b.z)].iter().enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "atom {i} axis {axis} (dielectric {dielectric}, shift {shift}): {x} vs {y}"
                    );
                }
            }
        }
    }
}
