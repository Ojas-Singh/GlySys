//! Named scoring adapter. Shader component slots are private to this backend.
//! Unsupported derivative/feature plans explicitly use the reference evaluator.
use crate::{
    context::GpuContext,
    device::{Config, Error, ResidentEvaluator},
    topology::PreparedTopology,
};
use glysys::Vec3;
use glysys_energy::scoring::*;
use std::collections::BTreeMap;

const TERMS: [&str; 9] = [
    "bonds",
    "angles",
    "proper_torsions",
    "improper_torsions",
    "van_der_waals",
    "electrostatics",
    "generalized_born",
    "surface_area",
    "restraints",
];

pub struct PreparedGpuEvaluator {
    reference: PreparedEvaluator,
    topology: PreparedTopology,
    resident: ResidentEvaluator,
}
impl PreparedGpuEvaluator {
    pub async fn with_context(
        context: &GpuContext,
        scene: PreparedScene,
        model: ScoreModel,
        capacity: u32,
    ) -> Result<Self, Error> {
        let reference =
            PreparedEvaluator::new(scene, model).map_err(|_| Error::Input("scoring model"))?;
        let mut groups = vec![0; reference.scene.system.atom_count()];
        if let Some((a, b)) = &reference.model.interaction {
            for &i in &reference.scene.groups[a] {
                groups[i] = 1;
            }
            for &i in &reference.scene.groups[b] {
                groups[i] = 2;
            }
        }
        let topology =
            PreparedTopology::new(&reference.scene.system, &reference.scene.options, &groups)?;
        let resident = ResidentEvaluator::with_context(context, topology.view(), capacity).await?;
        Ok(Self {
            reference,
            topology,
            resident,
        })
    }
    pub async fn evaluate(
        &mut self,
        batch: &PoseBatch,
        request: &EvaluationRequest,
    ) -> Result<Vec<EvaluationResult>, Error> {
        // Weighted derivatives, extensions, sparse features and pose Jacobians
        // remain supported through the CPU implementation until qualified kernels exist.
        let weighted = TERMS
            .iter()
            .any(|n| self.reference.model.weights.get(*n).copied().unwrap_or(0.) != 1.);
        if !self.reference.model.extensions.is_empty()
            || request.pose_derivatives
            || request.per_term_gradients
            || request.feature_distance.is_some()
            || (weighted && (request.gradients || request.forces))
        {
            return self
                .reference
                .evaluate(batch, request)
                .map_err(|e| Error::Execution(e.to_string()));
        }
        if let Some(selected) = &request.components {
            if selected.iter().any(|n| !TERMS.contains(&n.as_str())) {
                return Err(Error::Input("unknown requested component"));
            }
        }
        let n = self.reference.scene.system.atom_count();
        let movable = vec![true; n];
        let options = &self.reference.scene.options;
        let solvent = options.obc2.clone().unwrap_or_default();
        let interaction = self.reference.model.interaction.is_some();
        let gradient = request.gradients || request.forces;
        let mut results = Vec::with_capacity(batch.poses.len());
        for chunk in batch.poses.chunks(self.resident.batch_capacity() as usize) {
            let mut coordinates = Vec::with_capacity(n * chunk.len());
            for pose in chunk {
                let p = pose
                    .materialize()
                    .map_err(|e| Error::Execution(e.to_string()))?;
                coordinates.extend(self.topology.coordinates(&p, &movable)?);
            }
            let config = Config {
                size: [
                    n as u32,
                    chunk.len() as u32,
                    u32::from(interaction),
                    u32::from(gradient),
                ],
                energy: [
                    options.cutoff.unwrap_or(0.) as f32,
                    options.dielectric as f32,
                    if options.obc2.is_some() && !interaction {
                        1.
                    } else {
                        0.
                    },
                    0.,
                ],
                solvent: [
                    solvent.solute_dielectric as f32,
                    solvent.solvent_dielectric as f32,
                    solvent.probe_radius as f32,
                    solvent.surface_tension as f32,
                ],
                spare: [0.; 4],
            };
            let output = self
                .resident
                .evaluate(config, &coordinates, gradient)
                .await?;
            if output.components.len() != chunk.len() {
                return Err(Error::Execution(format!(
                    "GPU scoring returned {} component rows for {} poses",
                    output.components.len(),
                    chunk.len()
                )));
            }
            if gradient {
                let expected = chunk.len().saturating_mul(n);
                let actual = output.gradients.as_ref().map_or(0, Vec::len);
                if actual != expected {
                    return Err(Error::Execution(format!(
                        "GPU scoring returned {} gradient vectors for {} poses with {} atoms",
                        actual,
                        chunk.len(),
                        n
                    )));
                }
            }
            for (i, pose) in chunk.iter().enumerate() {
                let mut terms = BTreeMap::new();
                let mut total = 0.;
                for (j, name) in TERMS.iter().enumerate() {
                    let raw = output.components[i][j] as f64;
                    let weighted = raw
                        * self
                            .reference
                            .model
                            .weights
                            .get(*name)
                            .copied()
                            .unwrap_or(0.);
                    total += weighted;
                    if request
                        .components
                        .as_ref()
                        .is_none_or(|s| s.contains(*name))
                    {
                        terms.insert(
                            name.to_string(),
                            TermValue {
                                raw,
                                weighted,
                                unit: Unit::KcalPerMol,
                            },
                        );
                    }
                }
                let gradients: Option<Vec<Vec3>> = output.gradients.as_ref().map(|g| {
                    g[i * n..(i + 1) * n]
                        .iter()
                        .map(|v| Vec3 {
                            x: v[0] as f64,
                            y: v[1] as f64,
                            z: v[2] as f64,
                        })
                        .collect()
                });
                let forces = if request.forces {
                    gradients.as_ref().map(|g| {
                        g.iter()
                            .map(|v| Vec3 {
                                x: -v.x,
                                y: -v.y,
                                z: -v.z,
                            })
                            .collect()
                    })
                } else {
                    None
                };
                results.push(EvaluationResult {
                    candidate_id: pose.id,
                    total,
                    terms,
                    gradients: if request.gradients { gradients } else { None },
                    forces,
                    pose_derivatives: None,
                    features: Vec::new(),
                    term_gradients: BTreeMap::new(),
                    backend: "webgpu".into(),
                    model_version: self.reference.model.id.clone(),
                });
            }
        }
        Ok(results)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::sync::Arc;
    #[test]
    #[ignore = "requires Vulkan; software adapters establish correctness only"]
    fn named_adapter_matches_reference() {
        pollster::block_on(async {
            let context = GpuContext::new(Default::default()).await.unwrap();
            let system = glysys::SystemBuilder::new(glysys::BuildOptions {
                add_water: false,
                add_ions: false,
                ..Default::default()
            })
            .unwrap()
            .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))
            .unwrap();
            let scene =
                PreparedScene::new(Arc::new(system), Default::default(), Boundary::NonPeriodic)
                    .unwrap();
            let batch = PoseBatch {
                poses: vec![Pose::cartesian(93, scene.system.coordinates())],
            };
            let cpu = PreparedEvaluator::new(scene.clone(), ScoreModel::amber()).unwrap();
            let mut gpu =
                PreparedGpuEvaluator::with_context(&context, scene, ScoreModel::amber(), 2)
                    .await
                    .unwrap();
            for gradients in [false, true] {
                let request = EvaluationRequest {
                    gradients,
                    ..Default::default()
                };
                let expected = cpu.evaluate(&batch, &request).unwrap();
                let actual = gpu.evaluate(&batch, &request).await.unwrap();
                assert_eq!(actual[0].candidate_id, 93);
                for (name, e) in &expected[0].terms {
                    let a = &actual[0].terms[name];
                    assert!(
                        (a.raw - e.raw).abs() <= 1e-3 + 1e-4 * e.raw.abs(),
                        "{name}: {} {}",
                        a.raw,
                        e.raw
                    );
                }
                if gradients {
                    for (a, b) in actual[0]
                        .gradients
                        .as_ref()
                        .unwrap()
                        .iter()
                        .zip(expected[0].gradients.as_ref().unwrap())
                    {
                        for (a, b) in [(a.x, b.x), (a.y, b.y), (a.z, b.z)] {
                            assert!((a - b).abs() <= 1e-3 + 1e-3 * b.abs(), "gradient {a} {b}");
                        }
                    }
                }
            }
        });
    }
}
