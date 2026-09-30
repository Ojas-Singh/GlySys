//! Small, dependency-free 3D geometry used by the structure fixer.

use std::collections::HashMap;

use crate::model::Vec3;

pub(crate) type V = [f64; 3];

pub(crate) fn v(point: Vec3) -> V {
    [point.x, point.y, point.z]
}

pub(crate) fn p(value: V) -> Vec3 {
    Vec3 {
        x: value[0],
        y: value[1],
        z: value[2],
    }
}

pub(crate) fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn scale(a: V, factor: f64) -> V {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

pub(crate) fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn norm(a: V) -> f64 {
    dot(a, a).sqrt()
}

pub(crate) fn normalize(a: V) -> Option<V> {
    let length = norm(a);
    (length > 1.0e-9).then(|| scale(a, 1.0 / length))
}

pub(crate) fn distance(a: V, b: V) -> f64 {
    norm(sub(a, b))
}

pub(crate) fn distance2(a: V, b: V) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

/// Bond angle a-b-c in radians.
pub(crate) fn angle(a: V, b: V, c: V) -> f64 {
    let first = sub(a, b);
    let second = sub(c, b);
    let cosine = dot(first, second) / (norm(first) * norm(second)).max(1.0e-12);
    cosine.clamp(-1.0, 1.0).acos()
}

/// Dihedral a-b-c-d in radians, IUPAC sign convention.
pub(crate) fn dihedral(a: V, b: V, c: V, d: V) -> f64 {
    let b0 = sub(a, b);
    let b1 = sub(c, b);
    let b2 = sub(d, c);
    let b1n = normalize(b1).unwrap_or([1.0, 0.0, 0.0]);
    let v0 = sub(b0, scale(b1n, dot(b0, b1n)));
    let w = sub(b2, scale(b1n, dot(b2, b1n)));
    let x = dot(v0, w);
    let y = dot(cross(b1n, v0), w);
    y.atan2(x)
}

/// Natural-extension reference frame placement: the point d with |cd| =
/// `bond`, angle b-c-d = `bond_angle` and dihedral a-b-c-d = `torsion`.
pub(crate) fn place(a: V, b: V, c: V, bond: f64, bond_angle: f64, torsion: f64) -> V {
    let bc = normalize(sub(c, b)).unwrap_or([1.0, 0.0, 0.0]);
    let ab = sub(b, a);
    let n = normalize(cross(ab, bc)).unwrap_or_else(|| any_perpendicular(bc));
    let m = cross(n, bc);
    let local = [
        -bond * bond_angle.cos(),
        bond * bond_angle.sin() * torsion.cos(),
        bond * bond_angle.sin() * torsion.sin(),
    ];
    add(
        c,
        add(
            scale(bc, local[0]),
            add(scale(m, local[1]), scale(n, local[2])),
        ),
    )
}

pub(crate) fn any_perpendicular(axis: V) -> V {
    let reference = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    normalize(cross(axis, reference)).unwrap_or([0.0, 0.0, 1.0])
}

/// Rotate `point` by `theta` about the axis through `origin` along unit `axis`.
pub(crate) fn rotate_about(point: V, origin: V, axis: V, theta: f64) -> V {
    let relative = sub(point, origin);
    let (sin, cos) = theta.sin_cos();
    let rotated = add(
        add(scale(relative, cos), scale(cross(axis, relative), sin)),
        scale(axis, dot(axis, relative) * (1.0 - cos)),
    );
    add(origin, rotated)
}

/// Least-squares rigid superposition (Horn's quaternion method).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Superposition {
    rotation: [[f64; 3]; 3],
    from_center: V,
    to_center: V,
}

impl Superposition {
    /// Transform that maps `from` onto `to`; needs three non-collinear pairs.
    pub(crate) fn fit(from: &[V], to: &[V]) -> Option<Self> {
        if from.len() != to.len() || from.len() < 3 {
            return None;
        }
        let count = from.len() as f64;
        let center = |points: &[V]| {
            let sum = points.iter().fold([0.0; 3], |sum, point| add(sum, *point));
            scale(sum, 1.0 / count)
        };
        let from_center = center(from);
        let to_center = center(to);
        let mut s = [[0.0; 3]; 3];
        for (a, b) in from.iter().zip(to) {
            let a = sub(*a, from_center);
            let b = sub(*b, to_center);
            for i in 0..3 {
                for j in 0..3 {
                    s[i][j] += a[i] * b[j];
                }
            }
        }
        let n = [
            [
                s[0][0] + s[1][1] + s[2][2],
                s[1][2] - s[2][1],
                s[2][0] - s[0][2],
                s[0][1] - s[1][0],
            ],
            [
                s[1][2] - s[2][1],
                s[0][0] - s[1][1] - s[2][2],
                s[0][1] + s[1][0],
                s[2][0] + s[0][2],
            ],
            [
                s[2][0] - s[0][2],
                s[0][1] + s[1][0],
                -s[0][0] + s[1][1] - s[2][2],
                s[1][2] + s[2][1],
            ],
            [
                s[0][1] - s[1][0],
                s[2][0] + s[0][2],
                s[1][2] + s[2][1],
                -s[0][0] - s[1][1] + s[2][2],
            ],
        ];
        let (values, vectors) = jacobi4(n);
        let best = (0..4).max_by(|a, b| values[*a].total_cmp(&values[*b]))?;
        let q = [
            vectors[0][best],
            vectors[1][best],
            vectors[2][best],
            vectors[3][best],
        ];
        let [w, x, y, z] = q;
        let rotation = [
            [
                w * w + x * x - y * y - z * z,
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                w * w - x * x + y * y - z * z,
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                w * w - x * x - y * y + z * z,
            ],
        ];
        rotation
            .iter()
            .flatten()
            .all(|value| value.is_finite())
            .then_some(Self {
                rotation,
                from_center,
                to_center,
            })
    }

    pub(crate) fn apply(&self, point: V) -> V {
        let r = sub(point, self.from_center);
        let m = &self.rotation;
        add(
            self.to_center,
            [
                m[0][0] * r[0] + m[0][1] * r[1] + m[0][2] * r[2],
                m[1][0] * r[0] + m[1][1] * r[1] + m[1][2] * r[2],
                m[2][0] * r[0] + m[2][1] * r[1] + m[2][2] * r[2],
            ],
        )
    }
}

/// Eigen-decomposition of a symmetric 4x4 matrix (cyclic Jacobi).
fn jacobi4(mut a: [[f64; 4]; 4]) -> ([f64; 4], [[f64; 4]; 4]) {
    let mut v = [[0.0; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let off = (0..4)
            .flat_map(|i| (i + 1..4).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum::<f64>();
        if off < 1.0e-22 {
            break;
        }
        for p in 0..4 {
            for q in p + 1..4 {
                if a[p][q].abs() < 1.0e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let sign = if theta < 0.0 { -1.0 } else { 1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                let (row_p, row_q) = (a[p], a[q]);
                for k in 0..4 {
                    a[p][k] = c * row_p[k] - s * row_q[k];
                    a[q][k] = s * row_p[k] + c * row_q[k];
                }
                for row in &mut v {
                    let vkp = row[p];
                    let vkq = row[q];
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2], a[3][3]], v)
}

/// Uniform spatial hash for neighbor queries within one cell edge.
#[derive(Debug, Clone)]
pub(crate) struct Grid {
    cell: f64,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl Grid {
    pub(crate) fn new(cell: f64) -> Self {
        Self {
            cell,
            cells: HashMap::new(),
        }
    }

    fn key(&self, point: V) -> (i32, i32, i32) {
        (
            (point[0] / self.cell).floor() as i32,
            (point[1] / self.cell).floor() as i32,
            (point[2] / self.cell).floor() as i32,
        )
    }

    pub(crate) fn insert(&mut self, index: usize, point: V) {
        let key = self.key(point);
        self.cells.entry(key).or_default().push(index);
    }

    pub(crate) fn remove(&mut self, index: usize, point: V) {
        let key = self.key(point);
        if let Some(items) = self.cells.get_mut(&key) {
            items.retain(|item| *item != index);
        }
    }

    /// Candidate indices within `radius` (<= cell edge) of `point`.
    pub(crate) fn near(&self, point: V, radius: f64) -> impl Iterator<Item = usize> + '_ {
        let reach = (radius / self.cell).ceil().max(1.0) as i32;
        let (x, y, z) = self.key(point);
        (-reach..=reach).flat_map(move |dx| {
            (-reach..=reach).flat_map(move |dy| {
                (-reach..=reach).flat_map(move |dz| {
                    self.cells
                        .get(&(x + dx, y + dy, z + dz))
                        .into_iter()
                        .flatten()
                        .copied()
                })
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nerf_reproduces_bond_angle_and_torsion() {
        let a = [0.0, 1.0, 0.3];
        let b = [0.2, 0.0, 0.0];
        let c = [1.5, 0.1, 0.1];
        let d = place(a, b, c, 1.33, 116.2f64.to_radians(), -57.0f64.to_radians());
        assert!((distance(c, d) - 1.33).abs() < 1e-9);
        assert!((angle(b, c, d).to_degrees() - 116.2).abs() < 1e-9);
        assert!((dihedral(a, b, c, d).to_degrees() + 57.0).abs() < 1e-9);
    }

    #[test]
    fn superposition_recovers_rigid_motion() {
        let from = [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.3, 0.2, 1.1],
        ];
        let axis = normalize([1.0, 2.0, -0.5]).unwrap();
        let to = from.map(|point| add(rotate_about(point, [0.0; 3], axis, 1.1), [3.0, -2.0, 5.0]));
        let fit = Superposition::fit(&from, &to).unwrap();
        for (a, b) in from.iter().zip(&to) {
            assert!(distance(fit.apply(*a), *b) < 1e-8);
        }
    }
}
