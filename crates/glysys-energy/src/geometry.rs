//! Topology-bound coordinate mapping and rigid-fragment geometry.
use crate::{EnergyError, Result};
use glysys::{ParameterizedSystem, Structure, Vec3};
use std::collections::HashMap;

pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
    }
}
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
}
pub fn scale(a: Vec3, s: f64) -> Vec3 {
    Vec3 {
        x: a.x * s,
        y: a.y * s,
        z: a.z * s,
    }
}
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }
}
pub fn unit(a: Vec3) -> Result<Vec3> {
    let n = dot(a, a).sqrt();
    if n < 1e-12 || !n.is_finite() {
        return Err(EnergyError::InvalidConfiguration(
            "degenerate molecular frame".into(),
        ));
    }
    Ok(scale(a, 1. / n))
}
fn basis(origin: Vec3, a: Vec3, b: Vec3) -> Result<[Vec3; 3]> {
    let x = unit(sub(a, origin))?;
    let z = unit(cross(x, sub(b, origin)))?;
    Ok([x, cross(z, x), z])
}
type Key = (String, i32, Option<char>, String);
#[derive(Debug, Clone)]
pub struct CoordinateMap {
    reference: Vec<Vec3>,
    keys: Vec<Key>,
    elements: Vec<u8>,
    adjacency: Vec<Vec<usize>>,
}
impl CoordinateMap {
    pub fn new(system: &ParameterizedSystem) -> Self {
        let keys = system
            .atoms()
            .iter()
            .map(|a| {
                let r = &system.residues()[a.residue_index()];
                (
                    r.chain().to_owned(),
                    r.number(),
                    r.insertion_code(),
                    a.name().to_owned(),
                )
            })
            .collect();
        let mut adjacency = vec![Vec::new(); system.atom_count()];
        for b in system.bonds() {
            let [a, b] = b.atoms();
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
        Self {
            reference: system.coordinates(),
            keys,
            elements: system.atoms().iter().map(|a| a.element()).collect(),
            adjacency,
        }
    }
    pub fn coordinates(&self, source: &Structure) -> Result<Vec<Vec3>> {
        let mut lookup = HashMap::new();
        for a in source.iter_atoms() {
            let key = (
                a.residue.chain,
                a.residue.number,
                a.residue.insertion_code,
                a.name.to_owned(),
            );
            if lookup.insert(key, a.position).is_some() {
                return Err(EnergyError::InvalidConfiguration(
                    "ambiguous source atom identity".into(),
                ));
            }
        }
        let mut points = self.reference.clone();
        let mut present = vec![false; points.len()];
        for (i, k) in self.keys.iter().enumerate() {
            if let Some(p) = lookup.get(k) {
                points[i] = *p;
                present[i] = true;
            }
        }
        for i in 0..points.len() {
            if present[i] {
                continue;
            }
            if self.elements[i] != 1 {
                return Err(EnergyError::InvalidConfiguration(format!(
                    "missing heavy atom {:?}",
                    self.keys[i]
                )));
            }
            let parent = *self.adjacency[i]
                .iter()
                .find(|&&j| present[j])
                .ok_or_else(|| {
                    EnergyError::InvalidConfiguration(
                        "generated hydrogen has no mapped parent".into(),
                    )
                })?;
            // Prefer bonded heavy neighbors, then their bonded neighbors. No
            // geometry-only chemical inference and no remote global fit.
            let mut neighbors: Vec<_> = self.adjacency[parent]
                .iter()
                .copied()
                .filter(|&j| j != i && present[j] && self.elements[j] != 1)
                .collect();
            let direct = neighbors.clone();
            for j in direct {
                for &k in &self.adjacency[j] {
                    if k != parent && present[k] && self.elements[k] != 1 && !neighbors.contains(&k)
                    {
                        neighbors.push(k);
                    }
                }
            }
            let mut mapped = None;
            for a in 0..neighbors.len() {
                for b in a + 1..neighbors.len() {
                    let (j, k) = (neighbors[a], neighbors[b]);
                    if let (Ok(old), Ok(new)) = (
                        basis(self.reference[parent], self.reference[j], self.reference[k]),
                        basis(points[parent], points[j], points[k]),
                    ) {
                        let delta = sub(self.reference[i], self.reference[parent]);
                        mapped = Some(add(
                            points[parent],
                            add(
                                add(
                                    scale(new[0], dot(delta, old[0])),
                                    scale(new[1], dot(delta, old[1])),
                                ),
                                scale(new[2], dot(delta, old[2])),
                            ),
                        ));
                        break;
                    }
                }
                if mapped.is_some() {
                    break;
                }
            }
            if let Some(p) = mapped {
                points[i] = p;
            } else if self.adjacency[parent]
                .iter()
                .filter(|&&j| self.elements[j] != 1)
                .count()
                == 0
            {
                // Isolated water/ion-like fragment: translation preserves the
                // reference orientation; explicit source H coordinates override.
                points[i] = add(
                    points[parent],
                    sub(self.reference[i], self.reference[parent]),
                );
            } else {
                return Err(EnergyError::InvalidConfiguration(
                    "cannot update generated hydrogen frame".into(),
                ));
            }
        }
        if points
            .iter()
            .any(|p| ![p.x, p.y, p.z].iter().all(|v| v.is_finite()))
        {
            return Err(EnergyError::NonFiniteCoordinate);
        }
        Ok(points)
    }
}
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct RigidTransform {
    pub rotation: [[f64; 3]; 3],
    pub translation: Vec3,
}
impl Default for RigidTransform {
    fn default() -> Self {
        Self {
            rotation: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            translation: Vec3 {
                x: 0.,
                y: 0.,
                z: 0.,
            },
        }
    }
}
impl RigidTransform {
    pub fn apply(&self, p: Vec3) -> Vec3 {
        let a = [p.x, p.y, p.z];
        let mut v = [0.; 3];
        for i in 0..3 {
            v[i] = (0..3).map(|j| self.rotation[i][j] * a[j]).sum();
        }
        add(
            Vec3 {
                x: v[0],
                y: v[1],
                z: v[2],
            },
            self.translation,
        )
    }
}
#[derive(Debug, Clone)]
pub struct TorsionUpdate {
    pub axis: [usize; 2],
    pub moving: Vec<usize>,
    pub radians: f64,
}
pub fn rotate_torsion(points: &mut [Vec3], t: &TorsionUpdate) -> Result<()> {
    if t.axis
        .iter()
        .chain(t.moving.iter())
        .any(|&i| i >= points.len())
        || !t.radians.is_finite()
    {
        return Err(EnergyError::InvalidConfiguration(
            "invalid torsion update".into(),
        ));
    }
    let origin = points[t.axis[0]];
    let axis = unit(sub(points[t.axis[1]], origin))?;
    let (s, c) = t.radians.sin_cos();
    for &i in &t.moving {
        let v = sub(points[i], origin);
        points[i] = add(
            origin,
            add(
                add(scale(v, c), scale(cross(axis, v), s)),
                scale(axis, dot(axis, v) * (1. - c)),
            ),
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generated_hydrogens_follow_rigid_rotation() {
        let opts = glysys::BuildOptions {
            add_water: false,
            add_ions: false,
            ..Default::default()
        };
        let source =
            glysys::read_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"), &opts)
                .unwrap();
        let sys = glysys::SystemBuilder::new(opts)
            .unwrap()
            .prepare_structure(&source)
            .unwrap();
        let map = CoordinateMap::new(&sys);
        let before = map.coordinates(&source).unwrap();
        let mut moved = source.clone();
        for atom in source.iter_atoms() {
            moved
                .set_atom_position(
                    atom.id,
                    Vec3 {
                        x: -atom.position.y + 2.,
                        y: atom.position.x + 3.,
                        z: atom.position.z - 1.,
                    },
                )
                .unwrap();
        }
        let after = map.coordinates(&moved).unwrap();
        for (a, b) in before.iter().zip(after) {
            assert!((b.x + a.y - 2.).abs() < 1e-8);
            assert!((b.y - a.x - 3.).abs() < 1e-8);
            assert!((b.z - a.z + 1.).abs() < 1e-8);
        }
    }
}

/// A topology-derived torsion tree. Removing each rotatable bond must separate
/// its moving subtree; ring bonds are rejected instead of distorting a ring.
#[derive(Debug, Clone)]
pub struct KinematicTree {
    pub rigid_fragments: Vec<Vec<usize>>,
    pub torsions: Vec<TorsionUpdate>,
}
impl KinematicTree {
    pub fn new(system: &ParameterizedSystem, axes: &[[usize; 2]]) -> Result<Self> {
        use std::collections::BTreeSet;
        let n = system.atom_count();
        let mut adjacency = vec![Vec::new(); n];
        for bond in system.bonds() {
            let [a, b] = bond.atoms();
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
        let mut removed = BTreeSet::new();
        let mut torsions = Vec::new();
        for &[a, b] in axes {
            if a >= n
                || b >= n
                || !adjacency[a].contains(&b)
                || !removed.insert((a.min(b), a.max(b)))
            {
                return Err(EnergyError::InvalidConfiguration(
                    "invalid or duplicate rotatable bond".into(),
                ));
            }
            let mut moving = BTreeSet::new();
            let mut pending = vec![b];
            while let Some(i) = pending.pop() {
                if !moving.insert(i) {
                    continue;
                }
                for &j in &adjacency[i] {
                    if (i == a && j == b) || (i == b && j == a) {
                        continue;
                    }
                    pending.push(j);
                }
            }
            if moving.contains(&a) {
                return Err(EnergyError::InvalidConfiguration(
                    "ring torsions require a supported ring-pucker model".into(),
                ));
            }
            torsions.push(TorsionUpdate {
                axis: [a, b],
                moving: moving.into_iter().collect(),
                radians: 0.,
            });
        }
        // Apply ancestors before descendants, independently of caller order.
        torsions.sort_by_key(|t| std::cmp::Reverse(t.moving.len()));
        for (i, a) in torsions.iter().enumerate() {
            for b in &torsions[i + 1..] {
                let overlap = b.moving.iter().any(|j| a.moving.contains(j));
                if overlap && !b.moving.iter().all(|j| a.moving.contains(j)) {
                    return Err(EnergyError::InvalidConfiguration(
                        "inconsistent torsion tree orientation".into(),
                    ));
                }
            }
        }
        let mut seen = BTreeSet::new();
        let mut rigid_fragments = Vec::new();
        for root in 0..n {
            if seen.contains(&root) {
                continue;
            }
            let mut fragment = Vec::new();
            let mut pending = vec![root];
            while let Some(i) = pending.pop() {
                if !seen.insert(i) {
                    continue;
                }
                fragment.push(i);
                for &j in &adjacency[i] {
                    if !removed.contains(&(i.min(j), i.max(j))) {
                        pending.push(j);
                    }
                }
            }
            fragment.sort_unstable();
            rigid_fragments.push(fragment);
        }
        Ok(Self {
            rigid_fragments,
            torsions,
        })
    }
    /// Angles follow `torsions` order; each moving set includes its hydrogens.
    pub fn updates(&self, radians: &[f64]) -> Result<Vec<TorsionUpdate>> {
        if radians.len() != self.torsions.len() || radians.iter().any(|v| !v.is_finite()) {
            return Err(EnergyError::InvalidConfiguration(
                "invalid torsion vector".into(),
            ));
        }
        Ok(self
            .torsions
            .iter()
            .zip(radians)
            .map(|(t, &radians)| TorsionUpdate {
                radians,
                ..t.clone()
            })
            .collect())
    }
}
