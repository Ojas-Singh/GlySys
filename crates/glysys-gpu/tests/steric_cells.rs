use glysys_gpu::steric::{AttachmentLibrary, AttachmentPose, ReceptorUpdate, ResidentSteric};

fn ordered_reference_score(library: &AttachmentLibrary, flexible: bool, cutoff: f32) -> f32 {
    let query = [0.0f32, 0.0, 0.0];
    let cutoff2 = cutoff * cutoff;
    let mut score = 1.0;
    for (index, original) in library.protein.iter().enumerate() {
        let point = if flexible {
            library
                .updates
                .iter()
                .find(|update| update.indices[0] == index as u32)
                .map(|update| &update.value)
                .unwrap_or(original)
        } else {
            original
        };
        let d2 = (0..3)
            .map(|axis| (query[axis] - point[axis]).powi(2))
            .sum::<f32>();
        if (d2 - cutoff2).abs() < 0.002 {
            return -1.0;
        }
        if d2 < cutoff2 {
            score += 200.0 * (-d2).exp();
            if score > 2.0 {
                break;
            }
        }
    }
    score
}

#[test]
fn cell_streams_preserve_first_contact_and_flexible_updates() {
    pollster::block_on(async {
        for flexible in [false, true] {
            let library = AttachmentLibrary {
                protein: vec![[1.5, 0., 0., 0.], [-1., 0., 0., 0.]],
                coordinates: vec![[0., 0., 1., 0.], [1., 0., 1., 0.], [0., 0., 0., 0.]],
                poses: vec![AttachmentPose {
                    bounds: [0, 3, 2, 0],
                    indices: [1, 0, u32::from(flexible), 0],
                    b: [0., 1., 0., 0.],
                    link: [0.; 4],
                }],
                updates: if flexible {
                    vec![ReceptorUpdate {
                        value: [100., 0., 0., 0.],
                        indices: [0, 0, 0, 0],
                    }]
                } else {
                    vec![]
                },
                sites: 1,
                candidate_atoms: 3,
            };
            let context = glysys_gpu::GpuContext::new(glysys_gpu::GpuContextOptions::default())
                .await
                .unwrap();
            let mut gpu = ResidentSteric::with_context(&context, &library, 1)
                .await
                .unwrap();
            for cutoff in [1.7, 2.2] {
                let score = gpu
                    .evaluate(
                        &[[
                            0,
                            (-std::f32::consts::FRAC_PI_2).to_bits(),
                            0f32.to_bits(),
                            0,
                        ]],
                        cutoff,
                    )
                    .await
                    .unwrap()[0];
                let expected = ordered_reference_score(&library, flexible, cutoff);
                assert!((score - expected).abs() < 1e-3, "{score} != {expected}");
            }
        }
    });
}
