pub(crate) mod amber;
pub(crate) mod gromacs;

use crate::model::System;

const PDB_MAX_RESIDUE_NUMBER: usize = 9_999;
const PDB_CHAIN_IDS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

fn element_symbol(atomic_number: u8) -> &'static str {
    match atomic_number {
        1 => "H",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        11 => "Na",
        12 => "Mg",
        15 => "P",
        16 => "S",
        17 => "Cl",
        19 => "K",
        20 => "Ca",
        26 => "Fe",
        29 => "Cu",
        30 => "Zn",
        53 => "I",
        _ => "X",
    }
}

/// Serialize a fully parameterized (possibly solvated) system as PDB text.
pub(crate) fn write_pdb_system(system: &System) -> String {
    let mut output = String::new();
    let mut serial = 0;
    let addresses = pdb_residue_addresses(system);
    for (index, residue) in system.residues.iter().enumerate() {
        let is_protein = crate::pdb::PROTEIN_RESIDUES.contains(&residue.name.as_str());
        let record = if is_protein { "ATOM  " } else { "HETATM" };
        let address = &addresses[index];
        let chain_ends = system.residues.get(index + 1).is_none_or(|_| {
            addresses
                .get(index + 1)
                .is_none_or(|next| next.chain != address.chain)
        });
        for atom in system
            .atoms
            .iter()
            .skip(residue.first_atom)
            .take(residue.atom_count)
        {
            serial += 1;
            // PDB atom serials are fixed to five columns. Repeating serials
            // after the format limit is preferable to shifting x/y/z columns;
            // the complete unique atom identity remains in the topology files.
            let pdb_serial = (serial - 1) % 99_999 + 1;
            let insertion = residue.insertion_code.unwrap_or(' ');
            let element = format!("{:>2}", element_symbol(atom.element));
            output.push_str(&format!(
                "{record}{pdb_serial:>5} {:<4} {:>3} {:1}{:>4}{insertion}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {element}\n",
                atom.name,
                residue.name,
                address.chain,
                address.number,
                atom.position.x,
                atom.position.y,
                atom.position.z,
                1.00f64,
                0.00f64,
            ));
        }
        if chain_ends {
            output.push_str("TER\n");
        }
    }
    output.push_str("END\n");
    output
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PdbResidueAddress {
    chain: char,
    number: i32,
}

/// PDB can only hold a four-character residue number.  A solvated protein can
/// easily exceed that limit, so solvent and ions are split into consecutive
/// synthetic chains before serializing.  This keeps every coordinate column
/// fixed-width for PDB readers such as Mol* while topology writers retain the
/// original full system identity.
fn pdb_residue_addresses(system: &System) -> Vec<PdbResidueAddress> {
    let used_solute_chains = system
        .residues
        .iter()
        .filter(|residue| residue.first_atom < system.solute_atom_count)
        .filter_map(|residue| residue.chain.chars().next())
        .collect::<std::collections::BTreeSet<_>>();
    let mut available_chains = PDB_CHAIN_IDS
        .iter()
        .map(|id| *id as char)
        .filter(|id| !used_solute_chains.contains(id));

    let mut solvent_chains = Vec::new();
    let mut ion_chains = Vec::new();
    let mut water_index = 0usize;
    let mut ion_index = 0usize;

    system
        .residues
        .iter()
        .map(|residue| match residue.name.as_str() {
            "WAT" => {
                let segment = water_index / PDB_MAX_RESIDUE_NUMBER;
                while solvent_chains.len() <= segment {
                    solvent_chains.push(available_chains.next().unwrap_or('W'));
                }
                let address = PdbResidueAddress {
                    chain: solvent_chains[segment],
                    number: (water_index % PDB_MAX_RESIDUE_NUMBER + 1) as i32,
                };
                water_index += 1;
                address
            }
            "NA" | "CL" => {
                let segment = ion_index / PDB_MAX_RESIDUE_NUMBER;
                while ion_chains.len() <= segment {
                    ion_chains.push(available_chains.next().unwrap_or('I'));
                }
                let address = PdbResidueAddress {
                    chain: ion_chains[segment],
                    number: (ion_index % PDB_MAX_RESIDUE_NUMBER + 1) as i32,
                };
                ion_index += 1;
                address
            }
            _ => PdbResidueAddress {
                chain: residue.chain.chars().next().unwrap_or('A'),
                // Input structures with an out-of-range PDB number have no
                // faithful fixed-column PDB representation. Keep the output
                // parseable; simulation topology retains the original value.
                number: residue.number.clamp(-999, PDB_MAX_RESIDUE_NUMBER as i32),
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::model::{Atom, Residue, System, Vec3};

    use super::write_pdb_system;

    #[test]
    fn large_solvent_pdb_keeps_fixed_coordinate_columns() {
        let residue_count = 10_001;
        let mut atoms = Vec::with_capacity(residue_count);
        let mut residues = Vec::with_capacity(residue_count);
        for index in 0..residue_count {
            atoms.push(Atom {
                name: "O".into(),
                atom_type: "OW".into(),
                element: 8,
                residue: index,
                charge: -0.834,
                mass: 16.0,
                radius: 1.0,
                epsilon: 0.0,
                position: Vec3 {
                    x: 12.345,
                    y: 67.89,
                    z: -1.234,
                },
            });
            residues.push(Residue {
                name: "WAT".into(),
                number: index as i32 + 1,
                insertion_code: None,
                chain: "W".into(),
                first_atom: index,
                atom_count: 1,
                component: index,
            });
        }
        let system = System {
            atoms,
            residues,
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            exclusions: vec![BTreeSet::new(); residue_count],
            box_angstrom: [80.0; 3],
            component_count: residue_count,
            solute_atom_count: 0,
            water_residue_count: residue_count,
            sodium_count: 0,
            chloride_count: 0,
        };

        let pdb = write_pdb_system(&system);
        let atoms = pdb
            .lines()
            .filter(|line| line.starts_with("HETATM"))
            .collect::<Vec<_>>();
        assert_eq!(atoms.len(), residue_count);
        for line in [atoms[0], atoms[9_998], atoms[9_999], atoms[10_000]] {
            assert_eq!(line.len(), 78);
            assert_eq!(line[30..38].trim(), "12.345");
            assert_eq!(line[38..46].trim(), "67.890");
            assert_eq!(line[46..54].trim(), "-1.234");
        }
        assert_eq!(atoms[9_998][22..26].trim(), "9999");
        assert_eq!(atoms[9_999][22..26].trim(), "1");
        assert_ne!(&atoms[9_998][21..22], &atoms[9_999][21..22]);
    }
}
