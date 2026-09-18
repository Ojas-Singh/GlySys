//! Geometry analysis with explicit atom-index definitions.
use glysys::{ParameterizedSystem, Vec3};
use serde::{Deserialize, Serialize};
fn sub(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
}
fn dot(a: Vec3, b: Vec3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}
fn scale(a: Vec3, s: f64) -> Vec3 {
    Vec3 {
        x: a.x * s,
        y: a.y * s,
        z: a.z * s,
    }
}
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TorsionDefinition {
    pub label: String,
    pub atoms: [usize; 4],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameAnalysis {
    pub aligned_heavy_atom_rmsd: Option<f64>,
    pub torsion_degrees: Vec<Option<f64>>,
}

/// Geometry-only input for an analysis worker; no force field or simulation
/// needs to be reconstructed from the displayed PDB.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisDefinition {
    pub schema_version: u32,
    pub atom_count: usize,
    pub rmsd_atoms: Vec<usize>,
    pub rmsd_reference: Vec<Vec3>,
    pub torsions: Vec<TorsionDefinition>,
}
impl AnalysisDefinition {
    pub fn from_system(
        system: &ParameterizedSystem,
        reference: &[Vec3],
        torsions: &[TorsionDefinition],
    ) -> crate::Result<Self> {
        if reference.len() != system.atom_count() {
            return Err(crate::invalid("analysis reference atom count"));
        }
        let rmsd_atoms: Vec<_> = system
            .atoms()
            .iter()
            .enumerate()
            .filter(|(_, a)| a.element() != 1)
            .map(|(i, _)| i)
            .collect();
        let rmsd_reference = rmsd_atoms.iter().map(|&i| reference[i]).collect();
        Ok(Self {
            schema_version: 1,
            atom_count: system.atom_count(),
            rmsd_atoms,
            rmsd_reference,
            torsions: torsions.to_vec(),
        })
    }
}

pub struct PreparedAnalysis {
    definition: AnalysisDefinition,
    moving: Vec<Vec3>,
}
impl PreparedAnalysis {
    pub fn new(definition: AnalysisDefinition) -> crate::Result<Self> {
        if definition.schema_version != 1
            || definition.rmsd_atoms.len() != definition.rmsd_reference.len()
            || definition
                .rmsd_atoms
                .iter()
                .chain(definition.torsions.iter().flat_map(|t| &t.atoms))
                .any(|&i| i >= definition.atom_count)
            || definition
                .rmsd_reference
                .iter()
                .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
        {
            return Err(crate::invalid("invalid analysis definition"));
        }
        Ok(Self {
            moving: Vec::with_capacity(definition.rmsd_atoms.len()),
            definition,
        })
    }
    pub fn analyze_flat(&mut self, coordinates: &[f64]) -> crate::Result<FrameAnalysis> {
        if coordinates.len() != self.definition.atom_count * 3
            || coordinates.iter().any(|x| !x.is_finite())
        {
            return Err(crate::invalid("invalid analysis coordinates"));
        }
        let point = |i: usize| Vec3 {
            x: coordinates[3 * i],
            y: coordinates[3 * i + 1],
            z: coordinates[3 * i + 2],
        };
        self.moving.clear();
        self.moving
            .extend(self.definition.rmsd_atoms.iter().map(|&i| point(i)));
        Ok(FrameAnalysis {
            aligned_heavy_atom_rmsd: aligned_rmsd(&self.definition.rmsd_reference, &self.moving),
            torsion_degrees: self
                .definition
                .torsions
                .iter()
                .map(|t| torsion(t.atoms.map(point)))
                .collect(),
        })
    }
}
/// Maximum-eigenvalue quaternion superposition; translation and rotation removed.
pub fn aligned_rmsd(a: &[Vec3], b: &[Vec3]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let centroid = |p: &[Vec3]| {
        let n = p.len() as f64;
        Vec3 {
            x: p.iter().map(|p| p.x).sum::<f64>() / n,
            y: p.iter().map(|p| p.y).sum::<f64>() / n,
            z: p.iter().map(|p| p.z).sum::<f64>() / n,
        }
    };
    let ca = centroid(a);
    let cb = centroid(b);
    let mut s = [[0.; 3]; 3];
    let mut squared = 0.;
    for (a, b) in a.iter().zip(b) {
        let a = sub(*a, ca);
        let b = sub(*b, cb);
        squared += dot(a, a) + dot(b, b);
        for (i, x) in [a.x, a.y, a.z].into_iter().enumerate() {
            for (j, y) in [b.x, b.y, b.z].into_iter().enumerate() {
                s[i][j] += x * y;
            }
        }
    }
    let [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = s;
    let k = [
        [xx + yy + zz, yz - zy, zx - xz, xy - yx],
        [yz - zy, xx - yy - zz, xy + yx, zx + xz],
        [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
        [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
    ];
    let shift = k
        .iter()
        .map(|r| r.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0., f64::max)
        + 1.;
    let mut eigen = f64::NEG_INFINITY;
    for basis in 0..4 {
        let mut q = [0.; 4];
        q[basis] = 1.;
        for _ in 0..150 {
            let mut next = [0.; 4];
            for i in 0..4 {
                next[i] = shift * q[i] + (0..4).map(|j| k[i][j] * q[j]).sum::<f64>();
            }
            let n = next.iter().map(|x| x * x).sum::<f64>().sqrt();
            if n == 0. {
                break;
            }
            for i in 0..4 {
                q[i] = next[i] / n;
            }
        }
        let value = (0..4)
            .map(|i| q[i] * (0..4).map(|j| k[i][j] * q[j]).sum::<f64>())
            .sum::<f64>();
        eigen = eigen.max(value);
    }
    let value = ((squared - 2. * eigen).max(0.) / a.len() as f64).sqrt();
    value.is_finite().then_some(value)
}
pub fn torsion(p: [Vec3; 4]) -> Option<f64> {
    let axis = sub(p[2], p[1]);
    let norm = dot(axis, axis).sqrt();
    if norm < 1e-12 {
        return None;
    }
    let axis = scale(axis, 1. / norm);
    let b0 = sub(p[0], p[1]);
    let b2 = sub(p[3], p[2]);
    let v = sub(b0, scale(axis, dot(b0, axis)));
    let w = sub(b2, scale(axis, dot(b2, axis)));
    if dot(v, v) < 1e-20 || dot(w, w) < 1e-20 {
        return None;
    }
    Some(dot(cross(axis, v), w).atan2(dot(v, w)).to_degrees())
}
/// Supported aldopyranose C1–O(n) linkages; unknown geometries stay unannotated.
pub fn glycan_torsions(system: &ParameterizedSystem) -> Vec<TorsionDefinition> {
    let atoms = system.atoms();
    let mut adjacent = vec![Vec::new(); atoms.len()];
    for b in system.bonds() {
        let [a, b] = b.atoms();
        adjacent[a].push(b);
        adjacent[b].push(a);
    }
    let mut result = Vec::new();
    for bond in system.bonds() {
        let [a, b] = bond.atoms();
        for (donor, oxygen) in [(a, b), (b, a)] {
            if atoms[donor].name() != "C1"
                || atoms[oxygen].element() != 8
                || atoms[donor].residue_index() == atoms[oxygen].residue_index()
            {
                continue;
            }
            let Some(ring) = adjacent[donor].iter().copied().find(|&i| {
                atoms[i].name() == "O5" && atoms[i].residue_index() == atoms[donor].residue_index()
            }) else {
                continue;
            };
            let Some(carbon) = adjacent[oxygen].iter().copied().find(|&i| {
                atoms[i].element() == 6 && atoms[i].residue_index() == atoms[oxygen].residue_index()
            }) else {
                continue;
            };
            let label = format!(
                "{}:{}→{}:{}",
                system.residues()[atoms[donor].residue_index()].chain(),
                system.residues()[atoms[donor].residue_index()].number(),
                system.residues()[atoms[oxygen].residue_index()].chain(),
                system.residues()[atoms[oxygen].residue_index()].number()
            );
            result.push(TorsionDefinition {
                label: format!("{label} φ"),
                atoms: [ring, donor, oxygen, carbon],
            });
            let Some(index) = atoms[carbon]
                .name()
                .strip_prefix('C')
                .and_then(|s| s.parse::<usize>().ok())
            else {
                continue;
            };
            let name = format!("C{}", if index > 1 { index - 1 } else { 2 });
            if let Some(next) = adjacent[carbon].iter().copied().find(|&i| {
                atoms[i].name() == name && atoms[i].residue_index() == atoms[carbon].residue_index()
            }) {
                result.push(TorsionDefinition {
                    label: format!("{label} ψ"),
                    atoms: [donor, oxygen, carbon, next],
                });
            }
        }
    }
    result
}
pub fn analyze(
    system: &ParameterizedSystem,
    reference: &[Vec3],
    coordinates: &[Vec3],
    torsions: &[TorsionDefinition],
) -> FrameAnalysis {
    let indices: Vec<_> = system
        .atoms()
        .iter()
        .enumerate()
        .filter(|(_, a)| a.element() != 1)
        .map(|(i, _)| i)
        .collect();
    FrameAnalysis {
        aligned_heavy_atom_rmsd: aligned_rmsd(
            &indices.iter().map(|&i| reference[i]).collect::<Vec<_>>(),
            &indices.iter().map(|&i| coordinates[i]).collect::<Vec<_>>(),
        ),
        torsion_degrees: torsions
            .iter()
            .map(|t| torsion(t.atoms.map(|i| coordinates[i])))
            .collect(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_analysis_matches_geometry_and_rejects_invalid_frames() {
        let points = vec![
            Vec3 {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Vec3 {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            Vec3 {
                x: 1.,
                y: 1.,
                z: 0.,
            },
            Vec3 {
                x: 2.,
                y: 1.,
                z: 1.,
            },
        ];
        let definition = AnalysisDefinition {
            schema_version: 1,
            atom_count: 4,
            rmsd_atoms: vec![0, 1, 2, 3],
            rmsd_reference: points.clone(),
            torsions: vec![TorsionDefinition {
                label: "test".into(),
                atoms: [0, 1, 2, 3],
            }],
        };
        let encoded = serde_json::to_string(&definition).unwrap();
        let mut worker = PreparedAnalysis::new(serde_json::from_str(&encoded).unwrap()).unwrap();
        let moved: Vec<_> = points
            .iter()
            .map(|p| Vec3 {
                x: 8. - p.y,
                y: p.x + 4.,
                z: p.z - 2.,
            })
            .collect();
        let flat: Vec<_> = moved.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let result = worker.analyze_flat(&flat).unwrap();
        assert!(result.aligned_heavy_atom_rmsd.unwrap() < 1e-6);
        assert!(
            (result.torsion_degrees[0].unwrap() - torsion(points.try_into().unwrap()).unwrap())
                .abs()
                < 1e-10
        );
        assert!(worker.analyze_flat(&flat[..9]).is_err());
        let mut invalid = flat;
        invalid[2] = f64::NAN;
        assert!(worker.analyze_flat(&invalid).is_err());
        let mut invalid_definition = definition;
        invalid_definition.torsions[0].atoms[3] = 4;
        assert!(PreparedAnalysis::new(invalid_definition).is_err());
    }
    #[test]
    fn rmsd_removes_rigid_motion() {
        let p = [
            Vec3 {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Vec3 {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            Vec3 {
                x: 0.,
                y: 2.,
                z: 0.,
            },
            Vec3 {
                x: 0.,
                y: 0.,
                z: 3.,
            },
        ];
        let q = p.map(|p| Vec3 {
            x: 10. - p.y,
            y: p.x + 20.,
            z: p.z + 30.,
        });
        assert!(aligned_rmsd(&p, &q).unwrap() < 1e-6);
    }
    #[test]
    fn degenerate_torsion_is_unknown() {
        let p = Vec3 {
            x: 0.,
            y: 0.,
            z: 0.,
        };
        assert_eq!(torsion([p; 4]), None);
    }
}
