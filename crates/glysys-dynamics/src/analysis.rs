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
