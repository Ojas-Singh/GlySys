use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

use crate::model::{Dihedral, System};
use crate::{BuildError, Result};

const AMBER_CHARGE_SCALE: f64 = 18.2223;

pub(crate) fn write_inpcrd(system: &System) -> String {
    let mut output = String::new();
    writeln!(output, "GlySysBuilder Amber ff14SB GLYCAM06j-1 TIP3P").unwrap();
    writeln!(output, "{:>6}", system.atoms.len()).unwrap();
    let mut values = Vec::with_capacity(system.atoms.len() * 3);
    for atom in &system.atoms {
        values.extend([atom.position.x, atom.position.y, atom.position.z]);
    }
    write_fixed_floats(&mut output, &values, 6, 12, 7);
    if has_periodic_box(system) {
        write_fixed_floats(
            &mut output,
            &[
                system.box_angstrom[0],
                system.box_angstrom[1],
                system.box_angstrom[2],
                90.0,
                90.0,
                90.0,
            ],
            6,
            12,
            7,
        );
    }
    output
}

fn write_fixed_floats(
    output: &mut String,
    values: &[f64],
    per_line: usize,
    width: usize,
    precision: usize,
) {
    for chunk in values.chunks(per_line) {
        for value in chunk {
            write!(
                output,
                "{value:>width$.precision$}",
                width = width,
                precision = precision
            )
            .unwrap();
        }
        writeln!(output).unwrap();
    }
}

pub(crate) fn write_prmtop(system: &System) -> Result<String> {
    let mut output = String::new();
    writeln!(
        output,
        "%VERSION  VERSION_STAMP = V0001.000  DATE = 01/01/70  00:00:00"
    )
    .unwrap();
    let type_names = system
        .atoms
        .iter()
        .map(|atom| atom.atom_type.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let type_index = type_names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let bond_types = unique_bonds(system);
    let angle_types = unique_angles(system);
    let dihedral_types = unique_dihedrals(system);
    let bond_type_index = index_bonds(&bond_types);
    let angle_type_index = index_angles(&angle_types);
    let dihedral_type_index = index_dihedrals(&dihedral_types);
    let bonds_h = system
        .bonds
        .iter()
        .filter(|bond| {
            bond.atoms
                .iter()
                .any(|atom| system.atoms[*atom].element == 1)
        })
        .collect::<Vec<_>>();
    let bonds_no_h = system
        .bonds
        .iter()
        .filter(|bond| {
            bond.atoms
                .iter()
                .all(|atom| system.atoms[*atom].element != 1)
        })
        .collect::<Vec<_>>();
    let angles_h = system
        .angles
        .iter()
        .filter(|angle| {
            angle
                .atoms
                .iter()
                .any(|atom| system.atoms[*atom].element == 1)
        })
        .collect::<Vec<_>>();
    let angles_no_h = system
        .angles
        .iter()
        .filter(|angle| {
            angle
                .atoms
                .iter()
                .all(|atom| system.atoms[*atom].element != 1)
        })
        .collect::<Vec<_>>();
    let dihedrals_h = system
        .dihedrals
        .iter()
        .filter(|dihedral| {
            dihedral
                .atoms
                .iter()
                .any(|atom| system.atoms[*atom].element == 1)
        })
        .collect::<Vec<_>>();
    let dihedrals_no_h = system
        .dihedrals
        .iter()
        .filter(|dihedral| {
            dihedral
                .atoms
                .iter()
                .all(|atom| system.atoms[*atom].element != 1)
        })
        .collect::<Vec<_>>();
    let excluded = amber_exclusions(system);
    let max_residue = system
        .residues
        .iter()
        .map(|residue| residue.atom_count)
        .max()
        .unwrap_or(0);
    let pointers = vec![
        system.atoms.len(),
        type_names.len(),
        bonds_h.len(),
        bonds_no_h.len(),
        angles_h.len(),
        angles_no_h.len(),
        dihedrals_h.len(),
        dihedrals_no_h.len(),
        0,
        0,
        excluded.values.len(),
        system.residues.len(),
        bonds_no_h.len(),
        angles_no_h.len(),
        dihedrals_no_h.len(),
        bond_types.len(),
        angle_types.len(),
        dihedral_types.len(),
        type_names.len(),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        usize::from(has_periodic_box(system)),
        max_residue,
        0,
        0,
    ];
    int_section(&mut output, "POINTERS", &pointers);
    string_section(
        &mut output,
        "ATOM_NAME",
        system.atoms.iter().map(|atom| atom.name.as_str()),
    );
    float_section(
        &mut output,
        "CHARGE",
        &system
            .atoms
            .iter()
            .map(|atom| atom.charge * AMBER_CHARGE_SCALE)
            .collect::<Vec<_>>(),
    );
    int_section(
        &mut output,
        "ATOMIC_NUMBER",
        &system
            .atoms
            .iter()
            .map(|atom| atom.element as usize)
            .collect::<Vec<_>>(),
    );
    float_section(
        &mut output,
        "MASS",
        &system
            .atoms
            .iter()
            .map(|atom| atom.mass)
            .collect::<Vec<_>>(),
    );
    int_section(
        &mut output,
        "ATOM_TYPE_INDEX",
        &system
            .atoms
            .iter()
            .map(|atom| type_index[atom.atom_type.as_str()] + 1)
            .collect::<Vec<_>>(),
    );
    int_section(&mut output, "NUMBER_EXCLUDED_ATOMS", &excluded.counts);
    let (nonbond_index, acoef, bcoef) = nonbonded_tables(system, &type_names);
    int_section(&mut output, "NONBONDED_PARM_INDEX", &nonbond_index);
    string_section(
        &mut output,
        "RESIDUE_LABEL",
        system.residues.iter().map(|residue| residue.name.as_str()),
    );
    int_section(
        &mut output,
        "RESIDUE_POINTER",
        &system
            .residues
            .iter()
            .map(|residue| residue.first_atom + 1)
            .collect::<Vec<_>>(),
    );
    float_section(
        &mut output,
        "BOND_FORCE_CONSTANT",
        &bond_types.iter().map(|value| value.0).collect::<Vec<_>>(),
    );
    float_section(
        &mut output,
        "BOND_EQUIL_VALUE",
        &bond_types.iter().map(|value| value.1).collect::<Vec<_>>(),
    );
    float_section(
        &mut output,
        "ANGLE_FORCE_CONSTANT",
        &angle_types.iter().map(|value| value.0).collect::<Vec<_>>(),
    );
    float_section(
        &mut output,
        "ANGLE_EQUIL_VALUE",
        &angle_types.iter().map(|value| value.1).collect::<Vec<_>>(),
    );
    float_section(
        &mut output,
        "DIHEDRAL_FORCE_CONSTANT",
        &dihedral_types
            .iter()
            .map(|value| value.0)
            .collect::<Vec<_>>(),
    );
    float_section(
        &mut output,
        "DIHEDRAL_PERIODICITY",
        &dihedral_types
            .iter()
            .map(|value| value.1 as f64)
            .collect::<Vec<_>>(),
    );
    float_section(
        &mut output,
        "DIHEDRAL_PHASE",
        &dihedral_types
            .iter()
            .map(|value| value.2)
            .collect::<Vec<_>>(),
    );
    float_section(
        &mut output,
        "SCEE_SCALE_FACTOR",
        &dihedral_types
            .iter()
            .map(|value| value.3)
            .collect::<Vec<_>>(),
    );
    float_section(
        &mut output,
        "SCNB_SCALE_FACTOR",
        &dihedral_types
            .iter()
            .map(|value| value.4)
            .collect::<Vec<_>>(),
    );
    float_section(&mut output, "SOLTY", &vec![0.0; type_names.len()]);
    float_section(&mut output, "LENNARD_JONES_ACOEF", &acoef);
    float_section(&mut output, "LENNARD_JONES_BCOEF", &bcoef);
    int_section(
        &mut output,
        "BONDS_INC_HYDROGEN",
        &bond_records(&bonds_h, &bond_type_index),
    );
    int_section(
        &mut output,
        "BONDS_WITHOUT_HYDROGEN",
        &bond_records(&bonds_no_h, &bond_type_index),
    );
    int_section(
        &mut output,
        "ANGLES_INC_HYDROGEN",
        &angle_records(&angles_h, &angle_type_index),
    );
    int_section(
        &mut output,
        "ANGLES_WITHOUT_HYDROGEN",
        &angle_records(&angles_no_h, &angle_type_index),
    );
    int_section(
        &mut output,
        "DIHEDRALS_INC_HYDROGEN",
        &dihedral_records(&dihedrals_h, &dihedral_type_index),
    );
    int_section(
        &mut output,
        "DIHEDRALS_WITHOUT_HYDROGEN",
        &dihedral_records(&dihedrals_no_h, &dihedral_type_index),
    );
    int_section(&mut output, "EXCLUDED_ATOMS_LIST", &excluded.values);
    float_section(&mut output, "HBOND_ACOEF", &[]);
    float_section(&mut output, "HBOND_BCOEF", &[]);
    float_section(&mut output, "HBCUT", &[]);
    string_section(
        &mut output,
        "AMBER_ATOM_TYPE",
        system.atoms.iter().map(|atom| atom.atom_type.as_str()),
    );
    string_section(
        &mut output,
        "TREE_CHAIN_CLASSIFICATION",
        system.atoms.iter().map(|_| "M"),
    );
    int_section(&mut output, "JOIN_ARRAY", &vec![0; system.atoms.len()]);
    int_section(&mut output, "IROTAT", &vec![0; system.atoms.len()]);
    string_section(
        &mut output,
        "RADIUS_SET",
        ["modified Bondi radii (mbondi2)"].into_iter(),
    );
    float_section(
        &mut output,
        "RADII",
        &system
            .atoms
            .iter()
            .map(|atom| mbondi2_radius(atom.element, &atom.atom_type))
            .collect::<Vec<_>>(),
    );
    float_section(
        &mut output,
        "SCREEN",
        &system
            .atoms
            .iter()
            .map(|atom| screen(atom.element))
            .collect::<Vec<_>>(),
    );
    if has_periodic_box(system) {
        let component_sizes = component_sizes(system)?;
        let first_solvent_component = system
            .residues
            .iter()
            .find(|residue| residue.name == "WAT")
            .map(|residue| residue.component + 1)
            .unwrap_or(system.component_count + 1);
        let final_solute_residue = system
            .residues
            .iter()
            .position(|residue| residue.name == "WAT")
            .unwrap_or(system.residues.len());
        int_section(
            &mut output,
            "SOLVENT_POINTERS",
            &[
                final_solute_residue,
                system.component_count,
                first_solvent_component,
            ],
        );
        int_section(&mut output, "ATOMS_PER_MOLECULE", &component_sizes);
        float_section(
            &mut output,
            "BOX_DIMENSIONS",
            &[
                90.0,
                system.box_angstrom[0],
                system.box_angstrom[1],
                system.box_angstrom[2],
            ],
        );
    }
    Ok(output)
}

fn has_periodic_box(system: &System) -> bool {
    system.box_angstrom.iter().all(|length| *length > 0.0)
}

struct Exclusions {
    counts: Vec<usize>,
    values: Vec<usize>,
}

fn amber_exclusions(system: &System) -> Exclusions {
    let mut counts = Vec::with_capacity(system.atoms.len());
    let mut values = Vec::new();
    for (atom, exclusions) in system.exclusions.iter().enumerate() {
        let forward = exclusions
            .iter()
            .copied()
            .filter(|excluded| *excluded > atom)
            .collect::<Vec<_>>();
        counts.push(forward.len());
        values.extend(forward.into_iter().map(|value| value + 1));
    }
    Exclusions { counts, values }
}

fn unique_bonds(system: &System) -> Vec<(f64, f64)> {
    let mut result = Vec::new();
    for bond in &system.bonds {
        let value = (bond.force, bond.length);
        if !result.contains(&value) {
            result.push(value);
        }
    }
    result
}

fn unique_angles(system: &System) -> Vec<(f64, f64)> {
    let mut result = Vec::new();
    for angle in &system.angles {
        let value = (angle.force, angle.radians);
        if !result.contains(&value) {
            result.push(value);
        }
    }
    result
}

fn unique_dihedrals(system: &System) -> Vec<(f64, i32, f64, f64, f64)> {
    let mut result = Vec::new();
    for dihedral in &system.dihedrals {
        let value = (
            dihedral.force,
            dihedral.periodicity,
            dihedral.phase,
            dihedral.scee,
            dihedral.scnb,
        );
        if !result.contains(&value) {
            result.push(value);
        }
    }
    result
}

fn index_bonds(values: &[(f64, f64)]) -> HashMap<(u64, u64), usize> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| ((value.0.to_bits(), value.1.to_bits()), index))
        .collect()
}

fn index_angles(values: &[(f64, f64)]) -> HashMap<(u64, u64), usize> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| ((value.0.to_bits(), value.1.to_bits()), index))
        .collect()
}

type DihedralKey = (u64, i32, u64, u64, u64);

fn index_dihedrals(values: &[(f64, i32, f64, f64, f64)]) -> HashMap<DihedralKey, usize> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            (
                (
                    value.0.to_bits(),
                    value.1,
                    value.2.to_bits(),
                    value.3.to_bits(),
                    value.4.to_bits(),
                ),
                index,
            )
        })
        .collect()
}

fn bond_records(bonds: &[&crate::model::Bond], indices: &HashMap<(u64, u64), usize>) -> Vec<usize> {
    bonds
        .iter()
        .flat_map(|bond| {
            [
                bond.atoms[0] * 3,
                bond.atoms[1] * 3,
                indices[&(bond.force.to_bits(), bond.length.to_bits())] + 1,
            ]
        })
        .collect()
}

fn angle_records(
    angles: &[&crate::model::Angle],
    indices: &HashMap<(u64, u64), usize>,
) -> Vec<usize> {
    angles
        .iter()
        .flat_map(|angle| {
            [
                angle.atoms[0] * 3,
                angle.atoms[1] * 3,
                angle.atoms[2] * 3,
                indices[&(angle.force.to_bits(), angle.radians.to_bits())] + 1,
            ]
        })
        .collect()
}

fn dihedral_records(dihedrals: &[&Dihedral], indices: &HashMap<DihedralKey, usize>) -> Vec<isize> {
    let mut seen_pairs = std::collections::BTreeSet::new();
    dihedrals
        .iter()
        .flat_map(|dihedral| {
            let pair = (
                dihedral.atoms[0].min(dihedral.atoms[3]),
                dihedral.atoms[0].max(dihedral.atoms[3]),
            );
            let third = if dihedral.improper {
                -((dihedral.atoms[2] * 3) as isize)
            } else {
                (dihedral.atoms[2] * 3) as isize
            };
            let fourth_value = (dihedral.atoms[3] * 3) as isize;
            let fourth = if dihedral.improper || !seen_pairs.insert(pair) {
                -fourth_value
            } else {
                fourth_value
            };
            [
                (dihedral.atoms[0] * 3) as isize,
                (dihedral.atoms[1] * 3) as isize,
                third,
                fourth,
                (indices[&(
                    dihedral.force.to_bits(),
                    dihedral.periodicity,
                    dihedral.phase.to_bits(),
                    dihedral.scee.to_bits(),
                    dihedral.scnb.to_bits(),
                )] + 1) as isize,
            ]
        })
        .collect()
}

fn nonbonded_tables(system: &System, types: &[String]) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
    let representatives = types
        .iter()
        .map(|name| {
            system
                .atoms
                .iter()
                .find(|atom| &atom.atom_type == name)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut table_index = vec![0usize; types.len() * types.len()];
    let mut acoef = Vec::new();
    let mut bcoef = Vec::new();
    for first in 0..types.len() {
        for second in 0..=first {
            let radius = representatives[first].radius + representatives[second].radius;
            let epsilon = (representatives[first].epsilon * representatives[second].epsilon).sqrt();
            acoef.push(epsilon * radius.powi(12));
            bcoef.push(2.0 * epsilon * radius.powi(6));
            let index = acoef.len();
            table_index[first * types.len() + second] = index;
            table_index[second * types.len() + first] = index;
        }
    }
    (table_index, acoef, bcoef)
}

fn component_sizes(system: &System) -> Result<Vec<usize>> {
    let mut sizes = BTreeMap::new();
    let mut last_component = None;
    for atom in &system.atoms {
        let component = system.residues[atom.residue].component;
        if last_component.is_some_and(|last| component < last) {
            return Err(BuildError::ForceField(
                "molecular component atoms are not contiguous".into(),
            ));
        }
        last_component = Some(component);
        *sizes.entry(component).or_insert(0) += 1;
    }
    Ok(sizes.into_values().collect())
}

fn mbondi2_radius(element: u8, atom_type: &str) -> f64 {
    match element {
        1 if atom_type == "H" => 1.3,
        1 => 1.2,
        6 => 1.7,
        7 => 1.55,
        8 => 1.5,
        15 => 1.85,
        16 => 1.8,
        _ => 1.5,
    }
}

fn screen(element: u8) -> f64 {
    match element {
        1 => 0.85,
        6 => 0.72,
        7 => 0.79,
        8 => 0.85,
        15 => 0.86,
        16 => 0.96,
        _ => 0.8,
    }
}

fn int_section<T: std::fmt::Display>(output: &mut String, name: &str, values: &[T]) {
    writeln!(output, "%FLAG {name}").unwrap();
    writeln!(output, "%FORMAT(10I8)").unwrap();
    for chunk in values.chunks(10) {
        for value in chunk {
            write!(output, "{value:>8}").unwrap();
        }
        writeln!(output).unwrap();
    }
}

fn float_section(output: &mut String, name: &str, values: &[f64]) {
    writeln!(output, "%FLAG {name}").unwrap();
    writeln!(output, "%FORMAT(5E16.8)").unwrap();
    write_floats(output, values, 5, 16, 8);
}

fn string_section<'a>(output: &mut String, name: &str, values: impl Iterator<Item = &'a str>) {
    writeln!(output, "%FLAG {name}").unwrap();
    writeln!(output, "%FORMAT(20a4)").unwrap();
    let mut count = 0;
    for value in values {
        write!(output, "{:<4}", &value[..value.len().min(4)]).unwrap();
        count += 1;
        if count % 20 == 0 {
            writeln!(output).unwrap();
        }
    }
    if count % 20 != 0 {
        writeln!(output).unwrap();
    }
}

fn write_floats(
    output: &mut String,
    values: &[f64],
    per_line: usize,
    width: usize,
    precision: usize,
) {
    for chunk in values.chunks(per_line) {
        for value in chunk {
            write!(
                output,
                "{value:>width$.precision$E}",
                width = width,
                precision = precision
            )
            .unwrap();
        }
        writeln!(output).unwrap();
    }
}
