//! Clash relief for rebuilt and modelled heavy atoms.
//!
//! Only atoms absent from the input move.  Their covalent geometry is held by
//! harmonic restraints taken from the residue templates (bonds, 1-3
//! distances, ring distances and signed chiral/planar volumes) while a soft
//! repulsion removes overlaps, first by a torsion scan of rebuilt side chains
//! and then by L-BFGS minimization.

use std::collections::{HashMap, HashSet, VecDeque};

use super::chemistry::vdw_radius;
use super::geometry::{Grid, V, add, cross, distance, dot, normalize, rotate_about, scale, sub, v};
use super::work::Work;
use crate::forcefield::Template;

/// One heavy atom of the whole structure.
pub(crate) struct Site {
    pub residue: usize,
    pub atom: usize,
}

pub(crate) struct Model {
    pub sites: Vec<Site>,
    pub positions: Vec<V>,
    pub radii: Vec<f64>,
    pub polar: Vec<bool>,
    pub hydrogen: Vec<bool>,
    /// Hydrogen bonded to N or O (a hydrogen-bond donor).
    pub donor: Vec<bool>,
    pub movable: Vec<bool>,
    pub neighbors: Vec<Vec<usize>>,
    pub restraints: Vec<Restraint>,
}

#[derive(Debug, Clone)]
pub(crate) enum Restraint {
    Distance {
        i: usize,
        j: usize,
        target: f64,
        k: f64,
    },
    Volume {
        center: usize,
        a: usize,
        b: usize,
        c: usize,
        target: f64,
        k: f64,
    },
}

/// Collect heavy atoms, covalent graph and template restraints.
pub(crate) fn build_model(
    work: &Work,
    templates: &HashMap<usize, &Template>,
    extra_bonds: &[(super::work::AtomRef, super::work::AtomRef)],
) -> Model {
    let mut sites = Vec::new();
    let mut positions = Vec::new();
    let mut radii = Vec::new();
    let mut polar = Vec::new();
    let mut movable = Vec::new();
    let mut lookup = HashMap::<(usize, &str), usize>::new();
    for (residue_index, residue) in work.residues.iter().enumerate() {
        for (atom_index, atom) in residue.atoms.iter().enumerate() {
            if atom.is_hydrogen() {
                continue;
            }
            lookup.insert((residue_index, atom.name.as_str()), sites.len());
            sites.push(Site {
                residue: residue_index,
                atom: atom_index,
            });
            positions.push(atom.position);
            radii.push(vdw_radius(&atom.element));
            polar.push(matches!(atom.element.as_str(), "N" | "O"));
            movable.push(atom.movable());
        }
    }
    let mut neighbors = vec![Vec::new(); sites.len()];
    let mut restraints = Vec::new();
    let bond = |a: usize, b: usize, neighbors: &mut Vec<Vec<usize>>| {
        if a != b && !neighbors[a].contains(&b) {
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
    };
    // Intra-residue bonds: template graph, else covalent distance.
    for (residue_index, residue) in work.residues.iter().enumerate() {
        if let Some(template) = templates.get(&residue_index) {
            for &[first, second] in &template.bonds {
                let (a, b) = (&template.atoms[first], &template.atoms[second]);
                if a.element == 1 || b.element == 1 {
                    continue;
                }
                if let (Some(&i), Some(&j)) = (
                    lookup.get(&(residue_index, a.name.as_str())),
                    lookup.get(&(residue_index, b.name.as_str())),
                ) {
                    bond(i, j, &mut neighbors);
                }
            }
        } else {
            let heavy = residue
                .atoms
                .iter()
                .filter(|atom| !atom.is_hydrogen())
                .filter_map(|atom| lookup.get(&(residue_index, atom.name.as_str())).copied())
                .collect::<Vec<_>>();
            if heavy.len() <= 200 {
                for (position, &i) in heavy.iter().enumerate() {
                    for &j in &heavy[position + 1..] {
                        if distance(positions[i], positions[j]) < 1.95 {
                            bond(i, j, &mut neighbors);
                        }
                    }
                }
            }
        }
    }
    // Polymer links and explicit inter-residue bonds.
    let index_of = work.index_of();
    for (residue_index, residue) in work.residues.iter().enumerate() {
        if let Some(next) = residue.next {
            for (first, second) in [("C", "N"), ("O3'", "P")] {
                if let (Some(&i), Some(&j)) = (
                    lookup.get(&(residue_index, first)),
                    lookup.get(&(next, second)),
                ) {
                    bond(i, j, &mut neighbors);
                }
            }
        }
    }
    for ((first_uid, first_name), (second_uid, second_name)) in work.bonds.iter().chain(extra_bonds)
    {
        if let (Some(&a), Some(&b)) = (index_of.get(first_uid), index_of.get(second_uid))
            && let (Some(&i), Some(&j)) = (
                lookup.get(&(a, first_name.as_str())),
                lookup.get(&(b, second_name.as_str())),
            )
        {
            bond(i, j, &mut neighbors);
        }
    }

    // Template restraints for residues that contain movable atoms.
    for (residue_index, residue) in work.residues.iter().enumerate() {
        if !residue.atoms.iter().any(|atom| atom.movable()) {
            continue;
        }
        let Some(template) = templates.get(&residue_index) else {
            continue;
        };
        let local = template
            .atoms
            .iter()
            .enumerate()
            .filter(|(_, atom)| atom.element != 1)
            .filter_map(|(index, atom)| {
                lookup
                    .get(&(residue_index, atom.name.as_str()))
                    .map(|&site| (index, site))
            })
            .collect::<HashMap<_, _>>();
        let mut adjacency = vec![Vec::new(); template.atoms.len()];
        for &[a, b] in &template.bonds {
            if template.atoms[a].element != 1 && template.atoms[b].element != 1 {
                adjacency[a].push(b);
                adjacency[b].push(a);
            }
        }
        let tpos = |index: usize| v(template.atoms[index].position);
        let any_movable = |sites: &[usize]| sites.iter().any(|&site| movable[site]);
        for (&a, &site_a) in &local {
            // Bonds and 1-3 distances.
            for &b in &adjacency[a] {
                if let Some(&site_b) = local.get(&b)
                    && a < b
                    && any_movable(&[site_a, site_b])
                {
                    restraints.push(Restraint::Distance {
                        i: site_a,
                        j: site_b,
                        target: distance(tpos(a), tpos(b)),
                        k: 300.0,
                    });
                }
                for &c in &adjacency[b] {
                    if c > a
                        && let Some(&site_c) = local.get(&c)
                        && any_movable(&[site_a, site_c])
                    {
                        restraints.push(Restraint::Distance {
                            i: site_a,
                            j: site_c,
                            target: distance(tpos(a), tpos(c)),
                            k: 100.0,
                        });
                    }
                }
            }
            // Chirality (sp3) and planarity (sp2) at centres with three heavy neighbours.
            let heavy = &adjacency[a];
            let degree = template
                .bonds
                .iter()
                .filter(|bond| bond[0] == a || bond[1] == a)
                .count();
            if heavy.len() == 3
                && let (Some(&s1), Some(&s2), Some(&s3)) = (
                    local.get(&heavy[0]),
                    local.get(&heavy[1]),
                    local.get(&heavy[2]),
                )
                && any_movable(&[site_a, s1, s2, s3])
            {
                let target = if degree == 3 {
                    0.0
                } else {
                    signed_volume(tpos(a), tpos(heavy[0]), tpos(heavy[1]), tpos(heavy[2]))
                };
                restraints.push(Restraint::Volume {
                    center: site_a,
                    a: s1,
                    b: s2,
                    c: s3,
                    target,
                    k: 20.0,
                });
            }
        }
        // Ring shapes: all pairs within each ring system.
        for ring in ring_systems(template, &adjacency) {
            let members = ring
                .iter()
                .filter_map(|index| local.get(index).map(|&site| (*index, site)))
                .collect::<Vec<_>>();
            if !members.iter().any(|(_, site)| movable[*site]) {
                continue;
            }
            for (position, &(a, site_a)) in members.iter().enumerate() {
                for &(b, site_b) in &members[position + 1..] {
                    restraints.push(Restraint::Distance {
                        i: site_a,
                        j: site_b,
                        target: distance(tpos(a), tpos(b)),
                        k: 50.0,
                    });
                }
            }
        }
    }
    // Peptide links: bond, 1-3 geometry, trans omega.
    for (residue_index, residue) in work.residues.iter().enumerate() {
        let Some(next) = residue.next else {
            continue;
        };
        let site = |r: usize, name: &str| lookup.get(&(r, name)).copied();
        if let (Some(ca), Some(c), Some(o), Some(n), Some(ca2)) = (
            site(residue_index, "CA"),
            site(residue_index, "C"),
            site(residue_index, "O"),
            site(next, "N"),
            site(next, "CA"),
        ) && [ca, c, o, n, ca2].iter().any(|&site| movable[site])
        {
            // Trans peptide: CA-CA 3.80 Å and O cis to the next CA (2.77 Å).
            for (i, j, target, k) in [
                (c, n, 1.329, 300.0),
                (ca, n, 2.425, 100.0),
                (o, n, 2.247, 100.0),
                (c, ca2, 2.435, 100.0),
                (ca, ca2, 3.804, 100.0),
                (o, ca2, 2.768, 100.0),
            ] {
                restraints.push(Restraint::Distance { i, j, target, k });
            }
            restraints.push(Restraint::Volume {
                center: c,
                a: ca,
                b: o,
                c: n,
                target: 0.0,
                k: 20.0,
            });
        }
    }
    let count = sites.len();
    Model {
        sites,
        positions,
        radii,
        polar,
        hydrogen: vec![false; count],
        donor: vec![false; count],
        movable,
        neighbors,
        restraints,
    }
}

/// All atoms, with hydrogens that touch something (and their siblings) movable.
///
/// Each movable hydrogen keeps its bond length and its 1-3 distances to the
/// parent's other neighbours, so rotors turn and tetrahedral groups tilt only
/// as a whole while contacts are relieved (PDBFixer minimizes its added
/// hydrogens with a force field for the same reason).
pub(crate) fn build_hydrogen_model(work: &Work) -> Model {
    let mut sites = Vec::new();
    let mut positions = Vec::new();
    let mut radii = Vec::new();
    let mut polar = Vec::new();
    let mut hydrogen = Vec::new();
    let mut added = Vec::new();
    let mut residue_of = Vec::new();
    for (residue_index, residue) in work.residues.iter().enumerate() {
        for (atom_index, atom) in residue.atoms.iter().enumerate() {
            sites.push(Site {
                residue: residue_index,
                atom: atom_index,
            });
            positions.push(atom.position);
            radii.push(vdw_radius(&atom.element));
            polar.push(matches!(atom.element.as_str(), "N" | "O"));
            hydrogen.push(atom.is_hydrogen());
            added.push(atom.origin == super::work::Origin::Hydrogen);
            residue_of.push(residue_index);
        }
    }
    let count = sites.len();
    let mut grid = Grid::new(3.0);
    for (index, position) in positions.iter().enumerate() {
        grid.insert(index, *position);
    }
    let mut neighbors = vec![Vec::new(); count];
    let link = |a: usize, b: usize, neighbors: &mut Vec<Vec<usize>>| {
        if a != b && !neighbors[a].contains(&b) {
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
    };
    // Covalent graph from distances (intra-residue) plus polymer/explicit links.
    for i in 0..count {
        for j in grid.near(positions[i], 2.2) {
            if j <= i || residue_of[i] != residue_of[j] || (hydrogen[i] && hydrogen[j]) {
                continue;
            }
            let limit = if hydrogen[i] || hydrogen[j] { 1.3 } else { 2.1 };
            if distance(positions[i], positions[j]) < limit {
                link(i, j, &mut neighbors);
            }
        }
    }
    let mut lookup = HashMap::<(usize, &str), usize>::new();
    for (index, site) in sites.iter().enumerate() {
        lookup.insert(
            (
                site.residue,
                work.residues[site.residue].atoms[site.atom].name.as_str(),
            ),
            index,
        );
    }
    let index_of = work.index_of();
    for (residue_index, residue) in work.residues.iter().enumerate() {
        if let Some(next) = residue.next {
            for (first, second) in [("C", "N"), ("O3'", "P")] {
                if let (Some(&i), Some(&j)) = (
                    lookup.get(&(residue_index, first)),
                    lookup.get(&(next, second)),
                ) {
                    link(i, j, &mut neighbors);
                }
            }
        }
    }
    for ((uid_a, atom_a), (uid_b, atom_b)) in &work.bonds {
        if let (Some(&a), Some(&b)) = (index_of.get(uid_a), index_of.get(uid_b))
            && let (Some(&i), Some(&j)) = (
                lookup.get(&(a, atom_a.as_str())),
                lookup.get(&(b, atom_b.as_str())),
            )
        {
            link(i, j, &mut neighbors);
        }
    }
    let donor = (0..count)
        .map(|i| hydrogen[i] && neighbors[i].iter().any(|&n| polar[n]))
        .collect::<Vec<_>>();
    let mut model = Model {
        sites,
        positions,
        radii,
        polar,
        hydrogen,
        donor,
        movable: vec![false; count],
        neighbors,
        restraints: Vec::new(),
    };
    // Movable: added hydrogens in contact with anything, plus their siblings.
    let mut movable = vec![false; count];
    for i in 0..count {
        if !added[i] {
            continue;
        }
        let (near, third) = topology_sets(&model, i);
        let touching = grid.near(model.positions[i], 3.0).any(|j| {
            !near.contains(&j)
                && distance(model.positions[i], model.positions[j])
                    < contact_distance(&model, i, j, third.contains(&j))
        });
        if touching && let Some(&parent) = model.neighbors[i].first() {
            for &sibling in &model.neighbors[parent] {
                if added[sibling] {
                    movable[sibling] = true;
                }
            }
        }
    }
    let mut restraints = Vec::new();
    for i in 0..count {
        if !movable[i] {
            continue;
        }
        let Some(&parent) = model.neighbors[i].first() else {
            movable[i] = false;
            continue;
        };
        restraints.push(Restraint::Distance {
            i,
            j: parent,
            target: distance(model.positions[i], model.positions[parent]),
            k: 300.0,
        });
        for &other in &model.neighbors[parent] {
            if other != i && (other > i || !movable[other]) {
                restraints.push(Restraint::Distance {
                    i,
                    j: other,
                    target: distance(model.positions[i], model.positions[other]),
                    k: 100.0,
                });
            }
        }
    }
    model.movable = movable;
    model.restraints = restraints;
    model
}

fn signed_volume(center: V, a: V, b: V, c: V) -> f64 {
    dot(sub(a, center), cross(sub(b, center), sub(c, center)))
}

/// Atoms of each fused ring system (heavy-atom template graph).
fn ring_systems(template: &Template, adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut ring_bonds = Vec::new();
    for &[a, b] in &template.bonds {
        if template.atoms[a].element == 1 || template.atoms[b].element == 1 {
            continue;
        }
        // Is b reachable from a without the a-b bond within six steps?
        let mut queue = VecDeque::from([(a, 0)]);
        let mut seen = HashSet::from([a]);
        let mut cyclic = false;
        while let Some((node, depth)) = queue.pop_front() {
            if depth >= 6 {
                continue;
            }
            for &next in &adjacency[node] {
                if node == a && next == b {
                    continue;
                }
                if next == b {
                    cyclic = true;
                    break;
                }
                if seen.insert(next) {
                    queue.push_back((next, depth + 1));
                }
            }
            if cyclic {
                break;
            }
        }
        if cyclic {
            ring_bonds.push((a, b));
        }
    }
    let mut systems: Vec<HashSet<usize>> = Vec::new();
    for (a, b) in ring_bonds {
        let found = systems
            .iter()
            .position(|system| system.contains(&a) || system.contains(&b));
        match found {
            Some(position) => {
                systems[position].insert(a);
                systems[position].insert(b);
            }
            None => systems.push(HashSet::from([a, b])),
        }
    }
    // Merge systems that share atoms.
    let mut merged: Vec<HashSet<usize>> = Vec::new();
    for system in systems {
        if let Some(existing) = merged.iter_mut().find(|m| !m.is_disjoint(&system)) {
            existing.extend(system);
        } else {
            merged.push(system);
        }
    }
    merged
        .into_iter()
        .map(|system| {
            let mut atoms = system.into_iter().collect::<Vec<_>>();
            atoms.sort_unstable();
            atoms
        })
        .collect()
}

/// Atoms within two bonds (excluded from repulsion) and exactly three bonds.
fn topology_sets(model: &Model, atom: usize) -> (HashSet<usize>, HashSet<usize>) {
    let mut near = HashSet::from([atom]);
    let mut frontier = vec![atom];
    for _ in 0..2 {
        let mut next = Vec::new();
        for node in frontier {
            for &neighbor in &model.neighbors[node] {
                if near.insert(neighbor) {
                    next.push(neighbor);
                }
            }
        }
        frontier = next;
    }
    let mut third = HashSet::new();
    for node in frontier {
        for &neighbor in &model.neighbors[node] {
            if !near.contains(&neighbor) {
                third.insert(neighbor);
            }
        }
    }
    (near, third)
}

fn contact_distance(model: &Model, i: usize, j: usize, one_four: bool) -> f64 {
    let (hi, hj) = (model.hydrogen[i], model.hydrogen[j]);
    if hi || hj {
        if one_four {
            return 1.9;
        }
        if hi && hj {
            return if model.donor[i] && model.donor[j] {
                1.5
            } else {
                1.9
            };
        }
        let (h, x) = if hi { (i, j) } else { (j, i) };
        if model.donor[h] && model.polar[x] {
            return 1.65;
        }
        return 0.82 * (model.radii[i] + model.radii[j]);
    }
    if one_four {
        return 2.6;
    }
    if model.polar[i] && model.polar[j] {
        return 2.55;
    }
    0.82 * (model.radii[i] + model.radii[j])
}

/// Torsion scan of rebuilt side chains before minimization.
pub(crate) fn scan_side_chains(
    model: &mut Model,
    work: &Work,
    templates: &HashMap<usize, &Template>,
) {
    let mut grid = Grid::new(4.0);
    for (index, position) in model.positions.iter().enumerate() {
        grid.insert(index, *position);
    }
    let mut by_residue = HashMap::<usize, Vec<usize>>::new();
    for (site, info) in model.sites.iter().enumerate() {
        by_residue.entry(info.residue).or_default().push(site);
    }
    for (residue_index, residue) in work.residues.iter().enumerate() {
        let Some(sites) = by_residue.get(&residue_index) else {
            continue;
        };
        if !sites.iter().any(|&site| model.movable[site]) || templates.get(&residue_index).is_none()
        {
            continue;
        }
        let names = sites
            .iter()
            .map(|&site| (residue.atoms[model.sites[site].atom].name.as_str(), site))
            .collect::<HashMap<_, _>>();
        let Some(&ca) = names.get("CA").or_else(|| names.get("C1'")) else {
            continue;
        };
        // Rotatable bonds ordered outward from CA; downstream must be movable.
        let local = sites.iter().copied().collect::<HashSet<_>>();
        let mut depth = HashMap::from([(ca, 0usize)]);
        let mut queue = VecDeque::from([ca]);
        while let Some(node) = queue.pop_front() {
            for &next in &model.neighbors[node] {
                if local.contains(&next) && !depth.contains_key(&next) {
                    depth.insert(next, depth[&node] + 1);
                    queue.push_back(next);
                }
            }
        }
        let mut axes = Vec::new();
        for &a in sites {
            for &b in &model.neighbors[a] {
                if !local.contains(&b) || depth.get(&b).copied() != depth.get(&a).map(|d| d + 1) {
                    continue;
                }
                let downstream = downstream_atoms(model, a, b, &local);
                if downstream.is_empty()
                    || downstream.contains(&ca)
                    || !downstream.iter().all(|&site| model.movable[site])
                    || model.neighbors[b]
                        .iter()
                        .filter(|n| local.contains(n))
                        .count()
                        < 2
                {
                    continue;
                }
                axes.push((depth[&a], a, b, downstream));
            }
        }
        axes.sort_by_key(|axis| axis.0);
        for _ in 0..2 {
            for (_, a, b, downstream) in &axes {
                let origin = model.positions[*a];
                let Some(axis) = normalize(sub(model.positions[*b], origin)) else {
                    continue;
                };
                let original = downstream
                    .iter()
                    .map(|&s| model.positions[s])
                    .collect::<Vec<_>>();
                let mut best = (f64::INFINITY, 0.0);
                for step in 0..12 {
                    let theta = step as f64 * std::f64::consts::TAU / 12.0;
                    let moved = original
                        .iter()
                        .map(|&point| rotate_about(point, origin, axis, theta))
                        .collect::<Vec<_>>();
                    let score = placement_clash(model, &grid, downstream, &moved);
                    if score < best.0 - 1.0e-9 {
                        best = (score, theta);
                    }
                }
                for (&site, point) in downstream.iter().zip(&original) {
                    let moved = rotate_about(*point, origin, axis, best.1);
                    grid.remove(site, model.positions[site]);
                    model.positions[site] = moved;
                    grid.insert(site, moved);
                }
            }
        }
    }
}

fn downstream_atoms(model: &Model, a: usize, b: usize, local: &HashSet<usize>) -> Vec<usize> {
    let mut seen = HashSet::from([a, b]);
    let mut queue = VecDeque::from([b]);
    let mut result = Vec::new();
    while let Some(node) = queue.pop_front() {
        for &next in &model.neighbors[node] {
            if local.contains(&next) && seen.insert(next) {
                result.push(next);
                queue.push_back(next);
            }
        }
    }
    // A ring back to `a` means the bond is not rotatable.
    if model.neighbors[a]
        .iter()
        .any(|n| *n != b && result.contains(n))
    {
        return Vec::new();
    }
    result
}

fn placement_clash(model: &Model, grid: &Grid, sites: &[usize], positions: &[V]) -> f64 {
    let moving = sites.iter().copied().collect::<HashSet<_>>();
    let mut score = 0.0;
    for (&site, &point) in sites.iter().zip(positions) {
        let (near, third) = topology_sets(model, site);
        for other in grid.near(point, 4.0) {
            if moving.contains(&other) || near.contains(&other) {
                continue;
            }
            let limit = contact_distance(model, site, other, third.contains(&other));
            let d = distance(point, model.positions[other]);
            if d < limit {
                score += (limit - d) * (limit - d);
            }
        }
    }
    score
}

/// Minimize restraint + repulsion energy over the movable atoms.
pub(crate) fn minimize(model: &mut Model, iterations: usize) {
    let variables = (0..model.sites.len())
        .filter(|&site| model.movable[site])
        .collect::<Vec<_>>();
    if variables.is_empty() {
        return;
    }
    let slot = variables
        .iter()
        .enumerate()
        .map(|(slot, &site)| (site, slot))
        .collect::<HashMap<_, _>>();
    let topology = variables
        .iter()
        .map(|&site| topology_sets(model, site))
        .collect::<Vec<_>>();
    let restraints = model
        .restraints
        .iter()
        .filter(|restraint| match restraint {
            Restraint::Distance { i, j, .. } => model.movable[*i] || model.movable[*j],
            Restraint::Volume {
                center, a, b, c, ..
            } => [center, a, b, c].iter().any(|&&site| model.movable[site]),
        })
        .cloned()
        .collect::<Vec<_>>();

    let mut x = variables
        .iter()
        .flat_map(|&site| model.positions[site])
        .collect::<Vec<_>>();
    let mut pairs = Vec::new();
    let rebuild_pairs = |positions: &[V], pairs: &mut Vec<(usize, usize, f64)>| {
        pairs.clear();
        let mut grid = Grid::new(4.5);
        for (index, position) in positions.iter().enumerate() {
            grid.insert(index, *position);
        }
        for (slot_index, &site) in variables.iter().enumerate() {
            let (near, third) = &topology[slot_index];
            for other in grid.near(positions[site], 4.5) {
                if near.contains(&other) {
                    continue;
                }
                // Count movable-movable pairs once.
                if let Some(&other_slot) = slot.get(&other)
                    && other_slot <= slot_index
                {
                    continue;
                }
                if distance(positions[site], positions[other]) < 4.5 {
                    let limit = contact_distance(model, site, other, third.contains(&other));
                    pairs.push((site, other, limit));
                }
            }
        }
    };
    let energy = |x: &[f64],
                  positions: &mut Vec<V>,
                  pairs: &[(usize, usize, f64)],
                  gradient: &mut Vec<V>|
     -> f64 {
        for (slot_index, &site) in variables.iter().enumerate() {
            positions[site] = [
                x[3 * slot_index],
                x[3 * slot_index + 1],
                x[3 * slot_index + 2],
            ];
        }
        gradient.iter_mut().for_each(|g| *g = [0.0; 3]);
        let mut total = 0.0;
        for restraint in &restraints {
            match *restraint {
                Restraint::Distance { i, j, target, k } => {
                    let delta = sub(positions[i], positions[j]);
                    let d = super::geometry::norm(delta).max(1.0e-9);
                    let diff = d - target;
                    total += k * diff * diff;
                    let factor = 2.0 * k * diff / d;
                    gradient[i] = add(gradient[i], scale(delta, factor));
                    gradient[j] = sub(gradient[j], scale(delta, factor));
                }
                Restraint::Volume {
                    center,
                    a,
                    b,
                    c,
                    target,
                    k,
                } => {
                    let (ra, rb, rc) = (
                        sub(positions[a], positions[center]),
                        sub(positions[b], positions[center]),
                        sub(positions[c], positions[center]),
                    );
                    let volume = dot(ra, cross(rb, rc));
                    let diff = volume - target;
                    total += k * diff * diff;
                    let factor = 2.0 * k * diff;
                    let ga = scale(cross(rb, rc), factor);
                    let gb = scale(cross(rc, ra), factor);
                    let gc = scale(cross(ra, rb), factor);
                    gradient[a] = add(gradient[a], ga);
                    gradient[b] = add(gradient[b], gb);
                    gradient[c] = add(gradient[c], gc);
                    gradient[center] = sub(gradient[center], add(ga, add(gb, gc)));
                }
            }
        }
        for &(i, j, limit) in pairs {
            let delta = sub(positions[i], positions[j]);
            let d = super::geometry::norm(delta).max(1.0e-9);
            if d < limit {
                let overlap = limit - d;
                total += 30.0 * overlap * overlap;
                let factor = -60.0 * overlap / d;
                gradient[i] = add(gradient[i], scale(delta, factor));
                gradient[j] = sub(gradient[j], scale(delta, factor));
            }
        }
        total
    };
    let mut positions = model.positions.clone();
    let mut gradient_sites = vec![[0.0; 3]; positions.len()];
    let flatten = |gradient_sites: &[V]| {
        variables
            .iter()
            .flat_map(|&site| gradient_sites[site])
            .collect::<Vec<_>>()
    };
    rebuild_pairs(&positions, &mut pairs);
    let mut value = energy(&x, &mut positions, &pairs, &mut gradient_sites);
    let mut gradient = flatten(&gradient_sites);
    let memory = 8;
    let mut history: VecDeque<(Vec<f64>, Vec<f64>, f64)> = VecDeque::new();
    for iteration in 0..iterations {
        if iteration > 0 && iteration % 25 == 0 {
            rebuild_pairs(&positions, &mut pairs);
            value = energy(&x, &mut positions, &pairs, &mut gradient_sites);
            gradient = flatten(&gradient_sites);
        }
        let gnorm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < 1.0e-3 * (variables.len() as f64).sqrt() {
            break;
        }
        // Two-loop recursion.
        let mut direction = gradient.clone();
        let mut alphas = Vec::with_capacity(history.len());
        for (s, y, rho) in history.iter().rev() {
            let alpha = rho * dot_slices(s, &direction);
            axpy(&mut direction, -alpha, y);
            alphas.push(alpha);
        }
        if let Some((s, y, _)) = history.back() {
            let gamma = dot_slices(s, y) / dot_slices(y, y).max(1.0e-12);
            direction.iter_mut().for_each(|d| *d *= gamma);
        } else {
            let step = 0.1 / gnorm.max(1.0e-12);
            direction.iter_mut().for_each(|d| *d *= step);
        }
        for ((s, y, rho), alpha) in history.iter().zip(alphas.iter().rev()) {
            let beta = rho * dot_slices(y, &direction);
            axpy(&mut direction, alpha - beta, s);
        }
        direction.iter_mut().for_each(|d| *d = -*d);
        let mut slope = dot_slices(&gradient, &direction);
        if slope >= 0.0 {
            history.clear();
            direction = gradient
                .iter()
                .map(|g| -g * 0.1 / gnorm.max(1.0e-12))
                .collect();
            slope = dot_slices(&gradient, &direction);
        }
        // Cap the largest atomic displacement at 0.3 Å.
        let largest = direction
            .chunks(3)
            .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
            .fold(0.0, f64::max);
        let mut step = if largest > 0.3 { 0.3 / largest } else { 1.0 };
        let mut accepted = None;
        for _ in 0..20 {
            let trial = x
                .iter()
                .zip(&direction)
                .map(|(a, d)| a + step * d)
                .collect::<Vec<_>>();
            let trial_value = energy(&trial, &mut positions, &pairs, &mut gradient_sites);
            if trial_value <= value + 1.0e-4 * step * slope {
                accepted = Some((trial, trial_value, flatten(&gradient_sites)));
                break;
            }
            step *= 0.5;
        }
        let Some((trial, trial_value, trial_gradient)) = accepted else {
            // Restore positions for the current point and stop.
            energy(&x, &mut positions, &pairs, &mut gradient_sites);
            break;
        };
        let s = trial.iter().zip(&x).map(|(a, b)| a - b).collect::<Vec<_>>();
        let y = trial_gradient
            .iter()
            .zip(&gradient)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let sy = dot_slices(&s, &y);
        if sy > 1.0e-10 {
            history.push_back((s, y, 1.0 / sy));
            if history.len() > memory {
                history.pop_front();
            }
        }
        let converged = (value - trial_value).abs() < 1.0e-7 * value.abs().max(1.0);
        x = trial;
        value = trial_value;
        gradient = trial_gradient;
        if converged {
            break;
        }
    }
    for (slot_index, &site) in variables.iter().enumerate() {
        model.positions[site] = [
            x[3 * slot_index],
            x[3 * slot_index + 1],
            x[3 * slot_index + 2],
        ];
    }
}

fn dot_slices(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn axpy(target: &mut [f64], factor: f64, source: &[f64]) {
    for (t, s) in target.iter_mut().zip(source) {
        *t += factor * s;
    }
}

/// Write relaxed coordinates back into the working structure.
pub(crate) fn store(model: &Model, work: &mut Work) {
    for (site, info) in model.sites.iter().enumerate() {
        if model.movable[site] {
            work.residues[info.residue].atoms[info.atom].position = model.positions[site];
        }
    }
}

/// Inter-residue contacts under 1.5 Å that involve a hydrogen.
pub(crate) fn hydrogen_clashes(work: &Work) -> usize {
    let mut points = Vec::new();
    for (residue_index, residue) in work.residues.iter().enumerate() {
        for atom in &residue.atoms {
            points.push((residue_index, atom.is_hydrogen(), atom.position));
        }
    }
    let mut grid = Grid::new(2.0);
    for (index, point) in points.iter().enumerate() {
        grid.insert(index, point.2);
    }
    let mut count = 0;
    for (index, point) in points.iter().enumerate() {
        for other in grid.near(point.2, 1.5) {
            if other <= index || points[other].0 == point.0 || !(point.1 || points[other].1) {
                continue;
            }
            if distance(point.2, points[other].2) < 1.5 {
                count += 1;
            }
        }
    }
    count
}

/// Heavy-atom pairs involving new atoms that remain closer than 2.2 Å.
pub(crate) fn remaining_clashes(model: &Model) -> usize {
    let mut grid = Grid::new(4.0);
    for (index, position) in model.positions.iter().enumerate() {
        grid.insert(index, *position);
    }
    let mut count = 0;
    for site in 0..model.sites.len() {
        if !model.movable[site] {
            continue;
        }
        let (near, _) = topology_sets(model, site);
        for other in grid.near(model.positions[site], 2.2) {
            if near.contains(&other) || (model.movable[other] && other < site) {
                continue;
            }
            if distance(model.positions[site], model.positions[other]) < 2.2 {
                count += 1;
            }
        }
    }
    count
}
