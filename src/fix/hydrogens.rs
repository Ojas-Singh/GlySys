//! Hydrogen construction.
//!
//! Positions are derived from each parent atom's *actual* heavy-atom
//! neighbours (VSEPR geometry: tetrahedral, trigonal or bent), so they stay
//! correct when a side chain, sugar pucker or ligand conformation differs
//! from the library template.  The template only supplies which hydrogens
//! exist, their names, bond lengths and prochiral/cis-trans assignment.
//! Rotatable polar groups (OH, SH, NH3+) and waters are oriented by a
//! deterministic scan that rewards hydrogen bonds and penalizes clashes.

use std::collections::{HashMap, HashSet};

use super::chemistry::{hydrogen_bond_length, is_metal};
use super::components::Component;
use super::geometry::{
    Grid, Superposition, V, add, angle, any_perpendicular, cross, dihedral, distance, dot,
    normalize, place, scale, sub, v,
};
use super::work::{Origin, WAtom, WResidue};
use crate::forcefield::Template;

/// Atom graph that defines which hydrogens a residue carries.
pub(crate) struct HTemplate {
    pub names: Vec<String>,
    pub elements: Vec<String>,
    pub positions: Vec<V>,
    pub adjacency: Vec<Vec<usize>>,
    pub planar: Vec<bool>,
}

impl HTemplate {
    pub(crate) fn from_amber(template: &Template) -> Self {
        let mut adjacency = vec![Vec::new(); template.atoms.len()];
        for &[a, b] in &template.bonds {
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
        let planar = template
            .atoms
            .iter()
            .enumerate()
            .map(|(index, atom)| {
                let degree = adjacency[index].len()
                    + usize::from(Some(index) == template.head || Some(index) == template.tail);
                match atom.element {
                    6 => degree <= 3,
                    7 => degree <= 3 && !matches!(atom.atom_type.as_str(), "N3" | "NT"),
                    _ => false,
                }
            })
            .collect();
        Self {
            names: template
                .atoms
                .iter()
                .map(|atom| atom.name.clone())
                .collect(),
            elements: template
                .atoms
                .iter()
                .map(|atom| super::complete::element_symbol(atom.element).to_string())
                .collect(),
            positions: template.atoms.iter().map(|atom| v(atom.position)).collect(),
            adjacency,
            planar,
        }
    }

    pub(crate) fn from_component(component: &Component) -> Option<Self> {
        let index = component
            .atoms
            .iter()
            .enumerate()
            .map(|(i, atom)| (atom.name.as_str(), i))
            .collect::<HashMap<_, _>>();
        let mut adjacency = vec![Vec::new(); component.atoms.len()];
        let mut unsaturated = vec![false; component.atoms.len()];
        for bond in &component.bonds {
            let (Some(&a), Some(&b)) = (
                index.get(bond.first.as_str()),
                index.get(bond.second.as_str()),
            ) else {
                continue;
            };
            adjacency[a].push(b);
            adjacency[b].push(a);
            if bond.order > 1 || bond.aromatic {
                unsaturated[a] = true;
                unsaturated[b] = true;
            }
        }
        let positions = component
            .atoms
            .iter()
            .map(|atom| atom.ideal.map(v))
            .collect::<Option<Vec<_>>>()?;
        let planar = (0..component.atoms.len())
            .map(|i| {
                let element = component.atoms[i].element.as_str();
                unsaturated[i] || component.atoms[i].aromatic
                    // Amide / aniline / guanidine nitrogens are conjugated.
                    || (element == "N"
                        && adjacency[i].iter().any(|&n| unsaturated[n] && component.atoms[n].element != "N"))
            })
            .collect();
        Some(Self {
            names: component
                .atoms
                .iter()
                .map(|atom| atom.name.clone())
                .collect(),
            elements: component
                .atoms
                .iter()
                .map(|atom| atom.element.clone())
                .collect(),
            positions,
            adjacency,
            planar,
        })
    }

    fn is_hydrogen(&self, index: usize) -> bool {
        matches!(self.elements[index].as_str(), "H" | "D")
    }
}

#[derive(Debug, Clone, Copy)]
struct EnvAtom {
    acceptor: bool,
    metal: bool,
    hydrogen: bool,
    polar_hydrogen: bool,
}

/// Atoms already present, for clash and hydrogen-bond scoring.
pub(crate) struct Environment {
    grid: Grid,
    points: Vec<V>,
    atoms: Vec<EnvAtom>,
}

impl Environment {
    pub(crate) fn new() -> Self {
        Self {
            grid: Grid::new(3.0),
            points: Vec::new(),
            atoms: Vec::new(),
        }
    }

    pub(crate) fn add_heavy(&mut self, position: V, element: &str, acceptor: bool) {
        self.push(
            position,
            EnvAtom {
                acceptor,
                metal: is_metal(element),
                hydrogen: false,
                polar_hydrogen: false,
            },
        );
    }

    pub(crate) fn add_hydrogen(&mut self, position: V, polar: bool) {
        self.push(
            position,
            EnvAtom {
                acceptor: false,
                metal: false,
                hydrogen: true,
                polar_hydrogen: polar,
            },
        );
    }

    fn push(&mut self, position: V, atom: EnvAtom) {
        self.grid.insert(self.points.len(), position);
        self.points.push(position);
        self.atoms.push(atom);
    }

    /// Clash penalty minus hydrogen-bond reward for hydrogens on `parent`.
    fn score(&self, parent: V, hydrogens: &[V], polar: bool) -> f64 {
        let mut score = 0.0;
        for &hydrogen in hydrogens {
            for index in self.grid.near(hydrogen, 3.0) {
                let point = self.points[index];
                let atom = self.atoms[index];
                // Atoms bonded to the parent (1-3 to this hydrogen) are exempt;
                // hydrogen-bonded neighbours (>= 1.5 Å away) are not.
                let bonded = if atom.hydrogen { 1.25 } else { 1.95 };
                if distance(point, parent) < bonded {
                    continue;
                }
                let d = distance(hydrogen, point);
                if d > 3.0 {
                    continue;
                }
                let limit = if atom.hydrogen {
                    if polar && atom.polar_hydrogen {
                        1.6
                    } else {
                        1.9
                    }
                } else if polar && atom.acceptor {
                    1.55
                } else {
                    2.3
                };
                if d < limit {
                    score += ((limit - d) / 0.3).powi(2);
                }
                if polar {
                    if atom.acceptor
                        && (1.5..=2.6).contains(&d)
                        && angle(parent, hydrogen, point).to_degrees() > 110.0
                    {
                        score -= (1.0 - (d - 1.95).abs() / 0.65).max(0.0);
                    }
                    if atom.metal && d < 3.0 {
                        score += 3.0;
                    }
                    if atom.polar_hydrogen && d < 2.0 {
                        score += 1.0;
                    }
                }
            }
        }
        score
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Hydrogens whose position follows from the heavy atoms.
    Fixed,
    /// Rotatable polar groups, placed after all fixed hydrogens exist.
    Rotors,
}

/// Add the template's missing hydrogens to one residue.  Returns the count.
pub(crate) fn place_residue(
    residue: &mut WResidue,
    template: &HTemplate,
    external: &HashMap<String, Vec<V>>,
    skip_hydrogens: &HashSet<String>,
    environment: &mut Environment,
    phase: Phase,
) -> usize {
    let present = residue
        .atoms
        .iter()
        .map(|atom| (atom.name.clone(), atom.position))
        .collect::<HashMap<_, _>>();
    let mut added = Vec::new();
    for parent in 0..template.names.len() {
        if template.is_hydrogen(parent) {
            continue;
        }
        let Some(&parent_position) = present.get(&template.names[parent]) else {
            continue;
        };
        let hydrogens = template.adjacency[parent]
            .iter()
            .copied()
            .filter(|&h| {
                template.is_hydrogen(h)
                    && !present.contains_key(&template.names[h])
                    && !skip_hydrogens.contains(&template.names[h])
            })
            .collect::<Vec<_>>();
        if hydrogens.is_empty() {
            continue;
        }
        // Heavy neighbours: template-internal (with template index) and external.
        let mut heavy = template.adjacency[parent]
            .iter()
            .copied()
            .filter(|&n| !template.is_hydrogen(n))
            .filter_map(|n| present.get(&template.names[n]).map(|&p| (Some(n), p)))
            .collect::<Vec<_>>();
        for &position in external.get(&template.names[parent]).into_iter().flatten() {
            heavy.push((None, position));
        }
        // Hydrogens already present on this parent count toward coordination.
        let existing_h = template.adjacency[parent]
            .iter()
            .filter(|&&h| template.is_hydrogen(h) && present.contains_key(&template.names[h]))
            .count();
        let polar = matches!(template.elements[parent].as_str(), "N" | "O" | "S");
        let rotor = heavy.len() == 1 && existing_h == 0 && !template.planar[parent];
        let wants = match phase {
            Phase::Fixed => !(rotor && polar),
            Phase::Rotors => rotor && polar,
        };
        if !wants {
            continue;
        }
        let placed = place_group(
            template,
            parent,
            parent_position,
            &heavy,
            &hydrogens,
            existing_h,
            &present,
            environment,
            polar && rotor,
        );
        for (h, position) in placed {
            environment.add_hydrogen(position, polar);
            added.push(WAtom {
                name: template.names[h].clone(),
                element: "H".into(),
                position,
                occupancy: 1.0,
                b_factor: 0.0,
                serial: None,
                origin: Origin::Hydrogen,
            });
        }
    }
    let count = added.len();
    residue.atoms.extend(added);
    count
}

#[allow(clippy::too_many_arguments)]
fn place_group(
    template: &HTemplate,
    parent: usize,
    p: V,
    heavy: &[(Option<usize>, V)],
    hydrogens: &[usize],
    existing_h: usize,
    present: &HashMap<String, V>,
    environment: &Environment,
    scan: bool,
) -> Vec<(usize, V)> {
    let tp = template.positions[parent];
    let length = |h: usize| {
        let templated = distance(template.positions[h], tp);
        if (0.9..=1.25).contains(&templated) {
            templated
        } else {
            hydrogen_bond_length(&template.elements[parent])
        }
    };
    let coordination = heavy.len() + hydrogens.len() + existing_h;
    let units = heavy
        .iter()
        .filter_map(|(_, position)| normalize(sub(*position, p)))
        .collect::<Vec<_>>();
    if units.len() != heavy.len() {
        return Vec::new();
    }
    let planar = template.planar[parent] && coordination == 3;

    match (heavy.len(), hydrogens.len()) {
        // Tetrahedral centre with one missing substituent.
        (3, 1) => {
            let direction = normalize(scale(add(add(units[0], units[1]), units[2]), -1.0))
                .or_else(|| normalize(cross(sub(units[1], units[0]), sub(units[2], units[0]))));
            direction
                .map(|d| vec![(hydrogens[0], add(p, scale(d, length(hydrogens[0]))))])
                .unwrap_or_default()
        }
        // Trigonal centre (aromatic C-H, amide N-H).
        (2, 1) if planar || coordination == 3 && template.elements[parent] != "N" => {
            normalize(scale(add(units[0], units[1]), -1.0))
                .map(|d| vec![(hydrogens[0], add(p, scale(d, length(hydrogens[0]))))])
                .unwrap_or_default()
        }
        // Tetrahedral CH2 / NH2+ / pyramidal amine N-H.
        (2, _) => {
            let Some(bisector) = normalize(scale(add(units[0], units[1]), -1.0)) else {
                return Vec::new();
            };
            let Some(normal) = normalize(cross(units[0], units[1])) else {
                return Vec::new();
            };
            let half = (109.47f64 / 2.0).to_radians();
            // Which side each hydrogen occupies, from the template's handedness.
            let sides = hydrogens
                .iter()
                .enumerate()
                .map(|(ordinal, &h)| {
                    template_side(template, parent, heavy, h).unwrap_or(if ordinal == 0 {
                        1.0
                    } else {
                        -1.0
                    })
                })
                .collect::<Vec<_>>();
            let sides = if sides.len() == 2 && sides[0] == sides[1] {
                vec![1.0, -1.0]
            } else {
                sides
            };
            hydrogens
                .iter()
                .zip(sides)
                .map(|(&h, side)| {
                    let d = add(
                        scale(bisector, half.cos()),
                        scale(normal, side * half.sin()),
                    );
                    (h, add(p, scale(d, length(h))))
                })
                .collect()
        }
        // One heavy neighbour: in-plane NH2/CH2, or a rotor (CH3, NH3+, OH, SH).
        (1, _) => {
            let n1 = heavy[0].1;
            let axis = units[0];
            // Torsion reference: another heavy neighbour of the neighbour.
            let reference = heavy[0].0.and_then(|n1_index| {
                template.adjacency[n1_index]
                    .iter()
                    .copied()
                    .filter(|&x| x != parent && !template.is_hydrogen(x))
                    .find_map(|x| {
                        present
                            .get(&template.names[x])
                            .map(|&position| (Some(x), position))
                    })
            });
            let (x_index, x_position) =
                reference.unwrap_or_else(|| (None, add(n1, any_perpendicular(axis))));
            if planar {
                // sp2: both hydrogens in the plane of X-N1-P at 120 degrees.
                return hydrogens
                    .iter()
                    .enumerate()
                    .map(|(ordinal, &h)| {
                        let cis = match (heavy[0].0, x_index) {
                            (Some(n1_index), Some(x)) => {
                                dihedral(
                                    template.positions[h],
                                    tp,
                                    template.positions[n1_index],
                                    template.positions[x],
                                )
                                .abs()
                                    < std::f64::consts::FRAC_PI_2
                            }
                            _ => ordinal == 0,
                        };
                        let torsion = if cis { 0.0 } else { std::f64::consts::PI };
                        (
                            h,
                            place(x_position, n1, p, length(h), 120f64.to_radians(), torsion),
                        )
                    })
                    .collect();
            }
            let bond_angle = |h: usize| match heavy[0].0 {
                Some(n1_index) => {
                    let value = angle(template.positions[h], tp, template.positions[n1_index]);
                    if (1.55..=2.1).contains(&value) {
                        value
                    } else {
                        109.47f64.to_radians()
                    }
                }
                None => match template.elements[parent].as_str() {
                    "O" => 108.5f64.to_radians(),
                    "S" => 96.0f64.to_radians(),
                    _ => 109.47f64.to_radians(),
                },
            };
            // Template torsions relative to X; staggered defaults otherwise.
            let defaults = [180.0f64, 60.0, -60.0];
            let torsions = hydrogens
                .iter()
                .enumerate()
                .map(|(ordinal, &h)| match (heavy[0].0, x_index) {
                    (Some(n1_index), Some(x)) => dihedral(
                        template.positions[h],
                        tp,
                        template.positions[n1_index],
                        template.positions[x],
                    ),
                    _ => defaults[ordinal % 3].to_radians(),
                })
                .collect::<Vec<_>>();
            let build = |delta: f64| {
                hydrogens
                    .iter()
                    .zip(&torsions)
                    .map(|(&h, &torsion)| {
                        (
                            h,
                            place(x_position, n1, p, length(h), bond_angle(h), torsion + delta),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            if !scan {
                return build(0.0);
            }
            let mut best = (f64::INFINITY, 0.0);
            for step in 0..36 {
                let delta = (step as f64 * 10.0).to_radians();
                let candidate = build(delta);
                let points = candidate
                    .iter()
                    .map(|(_, position)| *position)
                    .collect::<Vec<_>>();
                // Tiny preference for the template (staggered) orientation.
                let score =
                    environment.score(p, &points, true) + 1.0e-3 * (step.min(36 - step) as f64);
                if score < best.0 {
                    best = (score, delta);
                }
            }
            build(best.1)
        }
        // No heavy neighbour (or unusual valence): superpose the template frame.
        _ => fallback_by_superposition(template, parent, p, hydrogens, present),
    }
}

/// +1/-1: side of the heavy-neighbour plane a hydrogen occupies in the template.
fn template_side(
    template: &HTemplate,
    parent: usize,
    heavy: &[(Option<usize>, V)],
    h: usize,
) -> Option<f64> {
    let (Some(a), Some(b)) = (heavy[0].0, heavy[1].0) else {
        return None;
    };
    let tp = template.positions[parent];
    let normal = normalize(cross(
        sub(template.positions[a], tp),
        sub(template.positions[b], tp),
    ))?;
    let value = dot(sub(template.positions[h], tp), normal);
    (value.abs() > 1.0e-3).then(|| value.signum())
}

fn fallback_by_superposition(
    template: &HTemplate,
    parent: usize,
    p: V,
    hydrogens: &[usize],
    present: &HashMap<String, V>,
) -> Vec<(usize, V)> {
    let mut from = Vec::new();
    let mut to = Vec::new();
    for (index, name) in template.names.iter().enumerate() {
        if template.is_hydrogen(index) {
            continue;
        }
        if let Some(&position) = present.get(name) {
            from.push(template.positions[index]);
            to.push(position);
        }
    }
    let tp = template.positions[parent];
    if let Some(fit) = Superposition::fit(&from, &to) {
        return hydrogens
            .iter()
            .map(|&h| {
                let target = fit.apply(template.positions[h]);
                let direction = normalize(sub(target, p)).unwrap_or([1.0, 0.0, 0.0]);
                (
                    h,
                    add(p, scale(direction, distance(template.positions[h], tp))),
                )
            })
            .collect();
    }
    // Isolated atom: keep the template's local offsets.
    hydrogens
        .iter()
        .map(|&h| (h, add(p, sub(template.positions[h], tp))))
        .collect()
}

/// Orient the two hydrogens of a water oxygen.
pub(crate) fn place_water(oxygen: V, environment: &mut Environment) -> [V; 2] {
    const BOND: f64 = 0.9572;
    let half_angle = (104.52f64 / 2.0).to_radians();
    let directions = sphere_points(32);
    let mut best: Option<(f64, [V; 2])> = None;
    for &bisector in &directions {
        let perpendicular = any_perpendicular(bisector);
        let other = cross(bisector, perpendicular);
        for step in 0..8 {
            let phi = step as f64 * std::f64::consts::PI / 8.0;
            let in_plane = add(scale(perpendicular, phi.cos()), scale(other, phi.sin()));
            let h1 = add(
                oxygen,
                scale(
                    add(
                        scale(bisector, half_angle.cos()),
                        scale(in_plane, half_angle.sin()),
                    ),
                    BOND,
                ),
            );
            let h2 = add(
                oxygen,
                scale(
                    sub(
                        scale(bisector, half_angle.cos()),
                        scale(in_plane, half_angle.sin()),
                    ),
                    BOND,
                ),
            );
            let score = environment.score(oxygen, &[h1, h2], true);
            if best
                .as_ref()
                .is_none_or(|(value, _)| score < *value - 1.0e-9)
            {
                best = Some((score, [h1, h2]));
            }
        }
    }
    let (_, hydrogens) = best.expect("sphere sampling yields candidates");
    for hydrogen in hydrogens {
        environment.add_hydrogen(hydrogen, true);
    }
    hydrogens
}

/// Deterministic, near-uniform unit vectors (Fibonacci sphere).
fn sphere_points(count: usize) -> Vec<V> {
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    (0..count)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
            let radius = (1.0 - y * y).sqrt();
            let theta = golden * i as f64;
            [radius * theta.cos(), y, radius * theta.sin()]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_hydrogens_have_tip3p_geometry() {
        let mut environment = Environment::new();
        environment.add_heavy([2.8, 0.0, 0.0], "O", true);
        let [h1, h2] = place_water([0.0; 3], &mut environment);
        assert!((distance(h1, [0.0; 3]) - 0.9572).abs() < 1e-9);
        assert!((angle(h1, [0.0; 3], h2).to_degrees() - 104.52).abs() < 1e-6);
        // One hydrogen should point at the acceptor.
        let toward = h1[0].max(h2[0]);
        assert!(toward > 0.8, "{toward}");
    }
}
