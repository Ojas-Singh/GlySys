use glysys::{BuildOptions, SystemBuilder, Vec3};
use glysys_energy::hydration::{HydrationRequest, PhysicalProbe};
#[test]
fn water_probe_gpu_matches_reference_components() {
    pollster::block_on(async {
        let system = SystemBuilder::new(BuildOptions {
            add_water: false,
            add_ions: false,
            ..Default::default()
        })
        .unwrap()
        .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))
        .unwrap();
        let probe = PhysicalProbe::new(&system).unwrap();
        let mut gpu = glysys_gpu::hydration::ResidentWaterProbe::new(&probe, 512)
            .await
            .unwrap();
        let o = probe.atoms[0].position;
        let r = HydrationRequest {
            minimum: Vec3 {
                x: o.x + 3.,
                y: o.y,
                z: o.z,
            },
            maximum: Vec3 {
                x: o.x + 4.,
                y: o.y,
                z: o.z,
            },
            spacing: 1.,
            orientations: 96,
            max_sites: 4,
            cutoff: None,
            method: None,
            chemical_potential: None,
            gc_steps: None,
            gc_seed: None,
        };
        for cutoff in [None, Some(12.)] {
            let poses = PhysicalProbe::poses(&r, [2, 1, 1], 0);
            let values = gpu.evaluate(&poses, cutoff).await.unwrap();
            for (p, g) in poses.into_iter().zip(values) {
                let c = probe.score(p, cutoff);
                assert_eq!(g.is_some(), c.is_some());
                if let (Some(g), Some(c)) = (g, c) {
                    for (a, b) in [
                        (g.lennard_jones, c.lennard_jones),
                        (g.electrostatics, c.electrostatics),
                    ] {
                        assert!((a - b).abs() <= 1e-3 + 1e-4 * b.abs(), "{a} {b}");
                    }
                }
            }
        }
    });
}
