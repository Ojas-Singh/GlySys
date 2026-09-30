//! Molecular graph with the chemical perception SMIRKS matching needs:
//! explicit hydrogens, formal charges, bond orders, rings (smallest set of
//! smallest rings) and MDL aromaticity, the model used by SMIRNOFF force
//! fields (`OEAroModel_MDL`).

use std::collections::{BTreeSet, HashSet, VecDeque};

#[derive(Debug, Clone, PartialEq)]
pub struct MolAtom {
    pub name: String,
    /// Atomic number.
    pub element: u8,
    pub formal_charge: i32,
    pub position: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct MolBond {
    pub atoms: [usize; 2],
    /// Kekulé bond order: 1, 2 or 3.
    pub order: u8,
}

/// A molecule with explicit hydrogens and perceived rings/aromaticity.
#[derive(Debug, Clone)]
pub struct Molecule {
    pub atoms: Vec<MolAtom>,
    pub bonds: Vec<MolBond>,
    pub(crate) neighbors: Vec<Vec<(usize, usize)>>,
    pub(crate) aromatic_atom: Vec<bool>,
    pub(crate) aromatic_bond: Vec<bool>,
    pub(crate) ring_bond: Vec<bool>,
    /// Sizes of the smallest-set-of-smallest-rings rings containing each atom.
    pub(crate) ring_sizes: Vec<Vec<usize>>,
    /// Match `-`/`=` by Kekulé order regardless of aromatic flags, and never
    /// match `:` (RDKit behaviour on a kekulized molecule with flags set).
    pub(crate) kekule_bonds: bool,
}

impl Molecule {
    pub fn new(atoms: Vec<MolAtom>, bonds: Vec<MolBond>) -> Self {
        let mut neighbors = vec![Vec::new(); atoms.len()];
        for (index, bond) in bonds.iter().enumerate() {
            let [a, b] = bond.atoms;
            neighbors[a].push((b, index));
            neighbors[b].push((a, index));
        }
        let mut molecule = Self {
            aromatic_atom: vec![false; atoms.len()],
            aromatic_bond: vec![false; bonds.len()],
            ring_bond: vec![false; bonds.len()],
            ring_sizes: vec![Vec::new(); atoms.len()],
            kekule_bonds: false,
            atoms,
            bonds,
            neighbors,
        };
        let rings = molecule.smallest_rings();
        for ring in &rings {
            for (position, &atom) in ring.iter().enumerate() {
                molecule.ring_sizes[atom].push(ring.len());
                let next = ring[(position + 1) % ring.len()];
                if let Some(bond) = molecule.bond_between(atom, next) {
                    molecule.ring_bond[bond] = true;
                }
            }
        }
        // Bonds in larger (non-SSSR) cycles are still ring bonds.
        for index in 0..molecule.bonds.len() {
            if !molecule.ring_bond[index] && molecule.bond_in_cycle(index) {
                molecule.ring_bond[index] = true;
            }
        }
        molecule.perceive_mdl_aromaticity(&rings);
        molecule
    }

    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    pub(crate) fn bond_between(&self, a: usize, b: usize) -> Option<usize> {
        self.neighbors[a]
            .iter()
            .find(|(other, _)| *other == b)
            .map(|(_, bond)| *bond)
    }

    pub(crate) fn degree(&self, atom: usize) -> usize {
        self.neighbors[atom].len()
    }

    pub(crate) fn hydrogen_count(&self, atom: usize) -> usize {
        self.neighbors[atom]
            .iter()
            .filter(|(other, _)| self.atoms[*other].element == 1)
            .count()
    }

    pub(crate) fn ring_connectivity(&self, atom: usize) -> usize {
        self.neighbors[atom]
            .iter()
            .filter(|(_, bond)| self.ring_bond[*bond])
            .count()
    }

    pub(crate) fn in_ring(&self, atom: usize) -> bool {
        self.ring_connectivity(atom) > 0
    }

    /// Sum of bond orders (aromatic bonds count 1.5 for SMARTS `v`).
    pub(crate) fn valence(&self, atom: usize) -> usize {
        let twice: usize = self.neighbors[atom]
            .iter()
            .map(|(_, bond)| {
                if self.aromatic_bond[*bond] {
                    3
                } else {
                    2 * self.bonds[*bond].order as usize
                }
            })
            .sum();
        twice.div_ceil(2)
    }

    fn bond_in_cycle(&self, bond: usize) -> bool {
        let [start, goal] = self.bonds[bond].atoms;
        let mut seen = HashSet::from([start]);
        let mut queue = VecDeque::from([start]);
        while let Some(atom) = queue.pop_front() {
            for &(next, via) in &self.neighbors[atom] {
                if via == bond || !seen.insert(next) {
                    continue;
                }
                if next == goal {
                    return true;
                }
                queue.push_back(next);
            }
        }
        false
    }

    /// Smallest set of smallest rings (Horton candidates + independence).
    fn smallest_rings(&self) -> Vec<Vec<usize>> {
        let atom_count = self.atoms.len();
        let ring_bonds = (0..self.bonds.len())
            .filter(|&bond| self.bond_in_cycle(bond))
            .collect::<Vec<_>>();
        if ring_bonds.is_empty() {
            return Vec::new();
        }
        // Cyclomatic number of the ring-bond subgraph.
        let ring_atoms = ring_bonds
            .iter()
            .flat_map(|&bond| self.bonds[bond].atoms)
            .collect::<BTreeSet<_>>();
        let components = {
            let mut parent = (0..atom_count).collect::<Vec<_>>();
            fn find(parent: &mut [usize], x: usize) -> usize {
                let mut root = x;
                while parent[root] != root {
                    root = parent[root];
                }
                let mut node = x;
                while parent[node] != root {
                    let next = parent[node];
                    parent[node] = root;
                    node = next;
                }
                root
            }
            for &bond in &ring_bonds {
                let [a, b] = self.bonds[bond].atoms;
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                parent[ra] = rb;
            }
            ring_atoms
                .iter()
                .map(|&atom| find(&mut parent, atom))
                .collect::<BTreeSet<_>>()
                .len()
        };
        let needed = ring_bonds.len() + components - ring_atoms.len();
        // Candidate cycles: shortest paths from each vertex through each edge.
        let ring_bond_set = ring_bonds.iter().copied().collect::<HashSet<_>>();
        let mut candidates = Vec::<Vec<usize>>::new();
        for &root in &ring_atoms {
            let (distance, parent) = self.bfs_tree(root, &ring_bond_set);
            for &bond in &ring_bonds {
                let [a, b] = self.bonds[bond].atoms;
                if distance[a].is_none() || distance[b].is_none() {
                    continue;
                }
                if parent[a] == Some(b) || parent[b] == Some(a) {
                    continue;
                }
                let path_a = path_to_root(&parent, a);
                let path_b = path_to_root(&parent, b);
                let set_a = path_a.iter().copied().collect::<HashSet<_>>();
                if path_b.iter().filter(|atom| set_a.contains(atom)).count() != 1 {
                    continue;
                }
                // root .. a, then b .. back towards (excluding) the root.
                let mut ordered = path_a.iter().rev().copied().collect::<Vec<_>>();
                ordered.extend(path_b.iter().copied().take(path_b.len() - 1));
                if ordered.len() >= 3 {
                    candidates.push(ordered);
                }
            }
        }
        candidates.sort_by_key(Vec::len);
        candidates.dedup_by(|a, b| {
            a.len() == b.len()
                && a.iter().collect::<BTreeSet<_>>() == b.iter().collect::<BTreeSet<_>>()
        });
        // Greedy selection of linearly independent cycles over GF(2).
        let bond_index = ring_bonds
            .iter()
            .enumerate()
            .map(|(position, &bond)| (bond, position))
            .collect::<std::collections::HashMap<_, _>>();
        let mut basis: Vec<Vec<u64>> = Vec::new();
        let mut pivots: Vec<usize> = Vec::new();
        let words = ring_bonds.len().div_ceil(64);
        let mut selected = Vec::new();
        for cycle in candidates {
            if selected.len() >= needed {
                break;
            }
            let mut vector = vec![0u64; words];
            let mut valid = true;
            for (position, &atom) in cycle.iter().enumerate() {
                let next = cycle[(position + 1) % cycle.len()];
                match self
                    .bond_between(atom, next)
                    .and_then(|bond| bond_index.get(&bond))
                {
                    Some(&column) => vector[column / 64] ^= 1 << (column % 64),
                    None => valid = false,
                }
            }
            if !valid {
                continue;
            }
            for (row, &pivot) in basis.iter().zip(&pivots) {
                if vector[pivot / 64] >> (pivot % 64) & 1 == 1 {
                    for (word, other) in vector.iter_mut().zip(row) {
                        *word ^= other;
                    }
                }
            }
            if let Some(pivot) =
                (0..ring_bonds.len()).find(|&c| vector[c / 64] >> (c % 64) & 1 == 1)
            {
                basis.push(vector);
                pivots.push(pivot);
                selected.push(cycle);
            }
        }
        selected
    }

    fn bfs_tree(
        &self,
        root: usize,
        allowed: &HashSet<usize>,
    ) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
        let mut distance = vec![None; self.atoms.len()];
        let mut parent = vec![None; self.atoms.len()];
        distance[root] = Some(0);
        let mut queue = VecDeque::from([root]);
        while let Some(atom) = queue.pop_front() {
            for &(next, bond) in &self.neighbors[atom] {
                if allowed.contains(&bond) && distance[next].is_none() {
                    distance[next] = Some(distance[atom].unwrap() + 1);
                    parent[next] = Some(atom);
                    queue.push_back(next);
                }
            }
        }
        (distance, parent)
    }

    /// MDL aromaticity: six-membered rings of C/N whose atoms each carry one
    /// double bond inside the same fused six-ring system and no exocyclic
    /// double bond.  Five-membered heterocycles are not aromatic.
    fn perceive_mdl_aromaticity(&mut self, rings: &[Vec<usize>]) {
        let six = rings
            .iter()
            .filter(|ring| ring.len() == 6)
            .collect::<Vec<_>>();
        if six.is_empty() {
            return;
        }
        // Fused systems of six-membered rings.
        let mut systems: Vec<BTreeSet<usize>> = Vec::new();
        for ring in &six {
            let atoms = ring.iter().copied().collect::<BTreeSet<_>>();
            let mut merged = atoms.clone();
            systems.retain(|system| {
                if system.intersection(&atoms).count() >= 2 {
                    merged.extend(system.iter().copied());
                    false
                } else {
                    true
                }
            });
            systems.push(merged);
        }
        let mut changed = true;
        while changed {
            changed = false;
            for ring in &six {
                if self.ring_is_mdl_aromatic(ring, &systems) {
                    for &atom in ring.iter() {
                        if !self.aromatic_atom[atom] {
                            self.aromatic_atom[atom] = true;
                            changed = true;
                        }
                    }
                    for (position, &atom) in ring.iter().enumerate() {
                        let next = ring[(position + 1) % ring.len()];
                        if let Some(bond) = self.bond_between(atom, next) {
                            self.aromatic_bond[bond] = true;
                        }
                    }
                }
            }
        }
    }

    fn ring_is_mdl_aromatic(&self, ring: &[usize], systems: &[BTreeSet<usize>]) -> bool {
        let Some(system) = systems
            .iter()
            .find(|system| ring.iter().all(|a| system.contains(a)))
        else {
            return false;
        };
        ring.iter().all(|&atom| {
            let element = self.atoms[atom].element;
            if element != 6 && element != 7 {
                return false;
            }
            let doubles = self.neighbors[atom]
                .iter()
                .filter(|(_, bond)| self.bonds[*bond].order == 2)
                .collect::<Vec<_>>();
            doubles.len() == 1 && system.contains(&doubles[0].0)
        })
    }
}

fn path_to_root(parent: &[Option<usize>], mut atom: usize) -> Vec<usize> {
    let mut path = vec![atom];
    while let Some(next) = parent[atom] {
        path.push(next);
        atom = next;
    }
    path
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a molecule from element numbers and (a, b, order) bonds.
    pub(crate) fn molecule(elements: &[u8], bonds: &[(usize, usize, u8)]) -> Molecule {
        Molecule::new(
            elements
                .iter()
                .enumerate()
                .map(|(index, &element)| MolAtom {
                    name: format!("A{index}"),
                    element,
                    formal_charge: 0,
                    position: [0.0; 3],
                })
                .collect(),
            bonds
                .iter()
                .map(|&(a, b, order)| MolBond {
                    atoms: [a, b],
                    order,
                })
                .collect(),
        )
    }

    /// Benzene C6H6 with a Kekulé structure.
    pub(crate) fn benzene() -> Molecule {
        let mut elements = vec![6; 6];
        elements.extend([1; 6]);
        let mut bonds = Vec::new();
        for i in 0..6 {
            bonds.push((i, (i + 1) % 6, if i % 2 == 0 { 2 } else { 1 }));
            bonds.push((i, i + 6, 1));
        }
        molecule(&elements, &bonds)
    }

    #[test]
    fn benzene_is_aromatic_and_has_one_six_ring() {
        let benzene = benzene();
        assert!((0..6).all(|atom| benzene.aromatic_atom[atom]));
        assert!(!benzene.aromatic_atom[6]);
        assert_eq!(benzene.ring_sizes[0], vec![6]);
        assert_eq!(benzene.valence(0), 4);
    }

    #[test]
    fn naphthalene_rings_are_aromatic_for_every_kekule_form() {
        // Kekulé form with the shared bond single and both ring-B junction
        // atoms double bonded into ring A.
        let elements = vec![6u8; 10];
        let bonds = [
            (0, 1, 1),
            (1, 2, 2),
            (2, 3, 1),
            (3, 4, 2),
            (4, 9, 1),
            (9, 0, 2),
            (4, 5, 1),
            (5, 6, 2),
            (6, 7, 1),
            (7, 8, 2),
            (8, 9, 1),
        ];
        let naphthalene = molecule(&elements, &bonds);
        assert!((0..10).all(|atom| naphthalene.aromatic_atom[atom]));
    }

    #[test]
    fn pyrrole_is_not_mdl_aromatic() {
        // N1 C2=C3 C4=C5 ring
        let pyrrole = molecule(
            &[7, 6, 6, 6, 6],
            &[(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 0, 1)],
        );
        assert!(!pyrrole.aromatic_atom.iter().any(|&a| a));
        assert!(pyrrole.ring_bond.iter().all(|&r| r));
    }
}
