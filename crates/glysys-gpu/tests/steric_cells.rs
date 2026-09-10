use glysys_gpu::steric::{AttachmentLibrary,AttachmentPose,ReceptorUpdate,ResidentSteric};
#[test]
fn cell_streams_preserve_first_contact_and_flexible_updates() {
    pollster::block_on(async {
        for flexible in [false,true] {
            let library=AttachmentLibrary{
                protein:vec![[1.5,0.,0.,0.],[-1.,0.,0.,0.]],
                coordinates:vec![[0.,0.,1.,0.],[1.,0.,1.,0.],[0.,0.,0.,0.]],
                poses:vec![AttachmentPose{bounds:[0,3,2,0],indices:[1,0,u32::from(flexible),0],b:[0.,1.,0.,0.],link:[0.;4]}],
                updates:if flexible{vec![ReceptorUpdate{value:[100.,0.,0.,0.],indices:[0,0,0,0]}]}else{vec![]},
                sites:1,candidate_atoms:3,
            };
            let mut gpu=ResidentSteric::new(&library,1).await.unwrap();
            for cutoff in [1.7,2.2] {
                let score=gpu.evaluate(&[[0,(-std::f32::consts::FRAC_PI_2).to_bits(),0f32.to_bits(),0]],cutoff).await.unwrap()[0];
                let d2:f32=if flexible{1.}else{2.25};let expected=1.+200.*(-d2).exp();
                assert!((score-expected).abs()<1e-3,"{score} != {expected}");
            }
        }
    });
}
