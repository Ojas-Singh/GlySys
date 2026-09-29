//! Small molecules (ligands, cofactors, modified residues kept as-is)
//! parameterized with OpenFF Sage 2.2.1 and AM1-BCC charges.

use std::collections::{BTreeSet, HashMap};

use crate::fix::Component;
use crate::forcefield::element_mass;
use crate::model::{Angle, Atom, Bond, Dihedral};
use crate::pdb::PdbResidue;
use crate::smirnoff::{ForceField, MolAtom, MolBond, Molecule, atomic_number};
use crate::{BuildError, Result};

/// Explicit bonded terms for atoms that are not typed by Amber parameters.
#[derive(Debug, Default)]
pub(crate) struct ExplicitTerms {
    pub atoms: BTreeSet<usize>,
    pub bonds: HashMap<[usize; 2], Bond>,
    pub angles: HashMap<[usize; 3], Angle>,
    pub propers: HashMap<[usize; 4], Vec<Dihedral>>,
    pub impropers: Vec<Dihedral>,
}

impl ExplicitTerms {
    pub(crate) fn contains(&self, atom: usize) -> bool {
        self.atoms.contains(&atom)
    }
}

pub(crate) fn canonical<const N: usize>(atoms: [usize; N]) -> [usize; N] {
    let mut reverse = atoms;
    reverse.reverse();
    atoms.min(reverse)
}

/// A parameterized ligand ready to be appended to the system.
pub(crate) struct Ligand {
    pub atoms: Vec<Atom>,
    /// Residue-local bonds.
    pub bonds: Vec<[usize; 2]>,
    /// Map from residue atom index to ligand atom index.
    pub order: Vec<usize>,
    pub charge: i32,
}

/// Build and parameterize one ligand residue from its CCD definition.
///
/// The residue must carry every non-leaving atom of the definition,
/// hydrogens included (run the structure fixer first for raw PDB entries).
pub(crate) fn parameterize(
    residue: &PdbResidue,
    component: &Component,
    force_field: &ForceField,
    residue_index: usize,
    explicit: &mut ExplicitTerms,
    first_atom: usize,
) -> Result<Ligand> {
    let label = residue.reference.to_string();
    let unsupported = |reason: String| BuildError::UnsupportedResidue {
        residue: label.clone(),
        reason,
    };
    let index = component
        .atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| (atom.name.as_str(), i))
        .collect::<HashMap<_, _>>();
    let mut present = vec![None; component.atoms.len()];
    for (residue_atom, atom) in residue.atoms.iter().enumerate() {
        let &definition = index.get(atom.name.as_str()).ok_or_else(|| {
            unsupported(format!(
                "atom {} is not in the {} chemical component definition",
                atom.name, component.id
            ))
        })?;
        present[definition] = Some(residue_atom);
    }
    let missing = component
        .atoms
        .iter()
        .zip(&present)
        .filter(|(atom, slot)| slot.is_none() && !atom.leaving)
        .map(|(atom, _)| atom.name.clone())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        let hydrogens_only = missing.iter().all(|name| {
            index
                .get(name.as_str())
                .is_some_and(|&i| component.atoms[i].element == "H")
        });
        return Err(unsupported(if hydrogens_only {
            format!(
                "{} hydrogens are missing ({}); repair the structure first (glysysbuilder fix, or prepare --fix)",
                missing.len(),
                missing
                    .iter()
                    .take(6)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        } else {
            format!("incomplete ligand, missing {}", missing.join(" "))
        }));
    }
    // Molecule in residue atom order.
    let mut local = vec![usize::MAX; component.atoms.len()];
    let mut atoms = Vec::with_capacity(residue.atoms.len());
    for (definition, slot) in present.iter().enumerate() {
        if let Some(residue_atom) = slot {
            local[definition] = atoms.len();
            let atom = &component.atoms[definition];
            let position = residue.atoms[*residue_atom].position;
            atoms.push((
                *residue_atom,
                MolAtom {
                    name: atom.name.clone(),
                    element: atomic_number(&atom.element)
                        .ok_or_else(|| unsupported(format!("unknown element {}", atom.element)))?,
                    formal_charge: atom.formal_charge,
                    position: [position.x, position.y, position.z],
                },
            ));
        }
    }
    let bonds = component
        .bonds
        .iter()
        .filter_map(|bond| {
            let a = local[*index.get(bond.first.as_str())?];
            let b = local[*index.get(bond.second.as_str())?];
            (a != usize::MAX && b != usize::MAX).then_some(MolBond {
                atoms: [a, b],
                order: bond.order,
            })
        })
        .collect::<Vec<_>>();
    let order = atoms
        .iter()
        .map(|(residue_atom, _)| *residue_atom)
        .collect::<Vec<_>>();
    let molecule = Molecule::new(atoms.into_iter().map(|(_, atom)| atom).collect(), bonds);
    let charge = molecule
        .atoms
        .iter()
        .map(|atom| atom.formal_charge)
        .sum::<i32>();
    let assignment = force_field
        .assign(&molecule)
        .map_err(|error| unsupported(error.0))?;

    let global = |i: usize| first_atom + i;
    let mut result_atoms = Vec::with_capacity(molecule.len());
    for (i, atom) in molecule.atoms.iter().enumerate() {
        let (epsilon, rmin_half, id) = &assignment.vdw[i];
        explicit.atoms.insert(global(i));
        result_atoms.push(Atom {
            name: atom.name.clone(),
            // Sage vdW ids (n1..n35) as short Amber-compatible type names.
            atom_type: format!("s{}", id.trim_start_matches('n')),
            element: atom.element,
            residue: residue_index,
            charge: assignment.charges[i],
            mass: element_mass(atom.element),
            radius: *rmin_half,
            epsilon: *epsilon,
            position: crate::model::Vec3 {
                x: atom.position[0],
                y: atom.position[1],
                z: atom.position[2],
            },
        });
    }
    for (atoms, force, length, _) in &assignment.bonds {
        let key = canonical([global(atoms[0]), global(atoms[1])]);
        explicit.bonds.insert(
            key,
            Bond {
                atoms: key,
                force: *force,
                length: *length,
            },
        );
    }
    for (atoms, force, radians, _) in &assignment.angles {
        let key = canonical([global(atoms[0]), global(atoms[1]), global(atoms[2])]);
        explicit.angles.insert(
            key,
            Angle {
                atoms: key,
                force: *force,
                radians: *radians,
            },
        );
    }
    // Sage 1-4 scaling: vdW 0.5, electrostatics 0.8333 (Amber scnb 2, scee 1.2).
    let scee = 1.0 / force_field.electrostatics_scale14;
    let scnb = 1.0 / force_field.vdw_scale14;
    for (atoms, force, periodicity, phase, improper, _) in &assignment.torsions {
        let mapped = [
            global(atoms[0]),
            global(atoms[1]),
            global(atoms[2]),
            global(atoms[3]),
        ];
        let dihedral = Dihedral {
            atoms: mapped,
            force: *force,
            periodicity: *periodicity,
            phase: *phase,
            improper: *improper,
            scee,
            scnb,
        };
        if *improper {
            explicit.impropers.push(dihedral);
        } else {
            let key = canonical(mapped);
            explicit.propers.entry(key).or_default().push(Dihedral {
                atoms: key,
                ..dihedral
            });
        }
    }
    Ok(Ligand {
        atoms: result_atoms,
        bonds: molecule.bonds.iter().map(|bond| bond.atoms).collect(),
        order,
        charge,
    })
}
