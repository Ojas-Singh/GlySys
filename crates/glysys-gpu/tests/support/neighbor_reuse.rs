use super::*;
#[test]
fn resident_neighbors_rebuild_at_skin_and_preserve_periodic_forces() {
    pollster::block_on(async {
        let _guard = super::gpu_test_guard();
        let system = solvated_system(6.0);
        let coordinates = minimized_coords(&system);
        let velocities = vec![
            Vec3 {
                x: 100.,
                y: 0.,
                z: 0.
            };
            system.atom_count()
        ];
        let mut ctx = setup(&system, 4., 1.5).await;
        ctx.gpu
            .initialize_dynamics(&coordinates, &velocities, ctx.box_f32, 0.001)
            .unwrap();
        ctx.gpu.energy_and_forces(true).await.unwrap();
        let initial = ctx.gpu.neighbor_rebuild_count().await.unwrap();
        ctx.gpu.energy_and_forces(true).await.unwrap();
        assert_eq!(
            initial,
            ctx.gpu.neighbor_rebuild_count().await.unwrap(),
            "unchanged coordinates rebuilt neighbors"
        );
        ctx.gpu.dynamics_steps(8).await.unwrap();
        assert_eq!(ctx.gpu.dynamics_status().await.unwrap(), None);
        assert!(
            ctx.gpu.neighbor_rebuild_count().await.unwrap() > initial,
            "skin crossing did not rebuild neighbors"
        );
        let state = ctx.gpu.read_dynamics_checkpoint(ctx.box_f32).await.unwrap();
        let gpu = ctx.gpu.read_dynamics_observables(true).await.unwrap();
        let list = PbcNeighborList::build(&state.coordinates, &ctx.box_vec, 4., 1.5).unwrap();
        let cpu = PbcForceField::new(&system, vec![])
            .unwrap()
            .evaluate(
                &state.coordinates,
                &ctx.box_vec,
                &list.pairs,
                &ReactionField {
                    cutoff_angstrom: 4.,
                    solvent_dielectric: 78.5,
                },
                4.,
            )
            .unwrap();
        assert!(
            (gpu.lj - cpu.components.van_der_waals).abs() < e_tol(cpu.components.van_der_waals)
        );
        assert!(
            (gpu.rf - cpu.components.electrostatics).abs() < e_tol(cpu.components.electrostatics)
        );
        for (gpu, cpu) in gpu.gradients.unwrap().iter().zip(cpu.gradients) {
            for (g, c) in gpu[..3].iter().zip([cpu.x, cpu.y, cpu.z]) {
                assert!(
                    (*g as f64 - c).abs() < gpu_g_tol(c),
                    "reused-neighbor force mismatch"
                );
            }
        }
    });
}

#[test]
fn failed_neighbor_build_remains_invalid_on_retry() {
    pollster::block_on(async {
        let _guard = super::gpu_test_guard();
        let system = solvated_system(6.0);
        let packing = PbcPacking::new(&system, 4., 1.5).unwrap();
        let backend = NonbondedElectrostatics::ReactionField {
            cutoff_angstrom: 4.,
            solvent_dielectric: 78.5,
        };
        let context = glysys_gpu::GpuContext::new(glysys_gpu::GpuContextOptions::default())
            .await
            .unwrap();
        let gpu = ResidentPbc::with_context(&context, &packing, &backend, 1)
            .await
            .unwrap();
        let b = system.box_angstrom();
        // Centered coordinates equal the zeroed rebuild reference. Recovery
        // must retain the dirty flag rather than rely on displacement alone.
        let coords = vec![
            Vec3 {
                x: b[0] * 0.5,
                y: b[1] * 0.5,
                z: b[2] * 0.5
            };
            system.atom_count()
        ];
        gpu.set_coordinates(&coords, [b[0] as f32, b[1] as f32, b[2] as f32], true);
        assert!(gpu.neighbor_list().await.is_err());
        for _ in 0..2 {
            assert!(gpu.energy_and_forces(true).await.unwrap().neighbor_overflow);
            assert!(gpu.dynamics_status().await.unwrap().is_some());
        }
        assert_eq!(gpu.neighbor_rebuild_count().await.unwrap(), 0);
    });
}
