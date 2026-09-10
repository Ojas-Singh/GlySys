use glysys::{BuildOptions, SystemBuilder};
use glysys_energy::{AtomSelection, EnergyEvaluator, EnergyOptions, HarmonicRestraint};
use glysys_gpu::{
    device::{Config, ResidentEvaluator},
    topology::PreparedTopology,
};
#[test]
#[ignore = "requires a Vulkan adapter"]
fn full_energy_and_obc2_parity() {
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        ..Default::default()
    })
    .unwrap()
    .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))
    .unwrap();
    for (obc2, active, cutoff, restrained) in [
        (false, false, None, false),
        (true, false, None, false),
        (false, true, Some(4.0), true),
        (true, true, Some(4.0), true),
    ] {
        let options = EnergyOptions {
            obc2: obc2.then(Default::default),
            cutoff,
            restraints: if restrained {
                vec![HarmonicRestraint {
                    atom: 0,
                    reference: glysys::Vec3 {
                        x: 0.,
                        y: 0.,
                        z: 0.,
                    },
                    force: 0.2,
                }]
            } else {
                vec![]
            },
            ..Default::default()
        };
        let evaluator = EnergyEvaluator::new(&system, options.clone()).unwrap();
        let evaluator = if active {
            evaluator
                .with_active_terms(AtomSelection::from_indices(system.atom_count(), [0, 1, 2]))
                .unwrap()
        } else {
            evaluator
        };
        let cpu = evaluator
            .energy_and_gradient(&system.coordinates())
            .unwrap();
        let packed =
            PreparedTopology::new(&system, &options, &vec![0; system.atom_count()]).unwrap();
        let coordinates = packed
            .coordinates(
                &system.coordinates(),
                &(0..system.atom_count())
                    .map(|i| !active || i < 3)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let result = pollster::block_on(async {
            let mut gpu = ResidentEvaluator::new(packed.view(), 1).await.unwrap();
            gpu.evaluate(
                Config {
                    size: [system.atom_count() as u32, 1, 0, u32::from(active)],
                    energy: [
                        cutoff.unwrap_or(0.) as f32,
                        1.,
                        if obc2 { 1. } else { 0. },
                        0.,
                    ],
                    solvent: [1., 78.5, 1.4, 0.00542],
                    spare: [0.; 4],
                },
                &coordinates,
                true,
            )
            .await
            .unwrap()
        });
        let c = cpu.components;
        let expected = [
            c.bonds,
            c.angles,
            c.proper_torsions,
            c.improper_torsions,
            c.van_der_waals,
            c.electrostatics,
            c.generalized_born,
            c.surface_area,
            c.restraints,
        ];
        for (i, (actual, expected)) in result.components[0].iter().zip(expected).enumerate() {
            assert!(
                (*actual as f64 - expected).abs() <= 1e-3 + 1e-4 * expected.abs(),
                "OBC={obc2} component={i} GPU={actual} CPU={expected}"
            );
        }
        for (i, (actual, expected)) in result
            .gradients
            .unwrap()
            .iter()
            .zip(cpu.gradients.unwrap())
            .enumerate()
        {
            for (axis, (a, e)) in actual
                .iter()
                .zip([expected.x, expected.y, expected.z])
                .enumerate()
            {
                assert!(
                    (*a as f64 - e).abs() <= 1e-3 + 1e-3 * e.abs(),
                    "OBC={obc2} atom={i} axis={axis} GPU={a} CPU={e}"
                );
            }
        }
    }
}
