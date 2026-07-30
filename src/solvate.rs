use std::collections::BTreeSet;

use crate::forcefield::{ParameterSet, TemplateSet};
use crate::model::{Angle, Atom, Bond, Residue, System, Vec3};
use crate::{BuildError, BuildOptions, Result};

const TIP3P_CELL_ANGSTROM: f64 = 18.774_349;
const AVOGADRO_PER_NM3: f64 = 0.602_214_076;

#[derive(Clone)]
struct Water {
    positions: [Vec3; 3],
    potential: f64,
    tie: u64,
}

pub(crate) fn solvate_and_ionize(
    system: &mut System,
    templates: &TemplateSet,
    parameters: &ParameterSet,
    options: &BuildOptions,
) -> Result<()> {
    let mut minimum = Vec3 {
        x: f64::INFINITY,
        y: f64::INFINITY,
        z: f64::INFINITY,
    };
    let mut maximum = Vec3 {
        x: f64::NEG_INFINITY,
        y: f64::NEG_INFINITY,
        z: f64::NEG_INFINITY,
    };
    for atom in &system.atoms {
        minimum.x = minimum.x.min(atom.position.x);
        minimum.y = minimum.y.min(atom.position.y);
        minimum.z = minimum.z.min(atom.position.z);
        maximum.x = maximum.x.max(atom.position.x);
        maximum.y = maximum.y.max(atom.position.y);
        maximum.z = maximum.z.max(atom.position.z);
    }
    let box_lengths = [
        maximum.x - minimum.x + 2.0 * options.padding_angstrom,
        maximum.y - minimum.y + 2.0 * options.padding_angstrom,
        maximum.z - minimum.z + 2.0 * options.padding_angstrom,
    ];
    let solute_center = Vec3 {
        x: (minimum.x + maximum.x) * 0.5,
        y: (minimum.y + maximum.y) * 0.5,
        z: (minimum.z + maximum.z) * 0.5,
    };
    let shift = Vec3 {
        x: box_lengths[0] * 0.5 - solute_center.x,
        y: box_lengths[1] * 0.5 - solute_center.y,
        z: box_lengths[2] * 0.5 - solute_center.z,
    };
    for atom in &mut system.atoms {
        atom.position.x += shift.x;
        atom.position.y += shift.y;
        atom.position.z += shift.z;
    }
    system.box_angstrom = box_lengths;

    let box_template = templates.tip3p_box();
    if !box_template.atoms.len().is_multiple_of(3) {
        return Err(BuildError::ForceField(
            "TIP3P box atom count is not divisible by three".into(),
        ));
    }
    let template_minimum =
        box_template
            .atoms
            .iter()
            .fold([f64::INFINITY; 3], |mut result, atom| {
                result[0] = result[0].min(atom.position.x);
                result[1] = result[1].min(atom.position.y);
                result[2] = result[2].min(atom.position.z);
                result
            });
    let tiles = [
        (box_lengths[0] / TIP3P_CELL_ANGSTROM).ceil() as usize,
        (box_lengths[1] / TIP3P_CELL_ANGSTROM).ceil() as usize,
        (box_lengths[2] / TIP3P_CELL_ANGSTROM).ceil() as usize,
    ];
    let mut waters = Vec::new();
    for ix in 0..tiles[0] {
        for iy in 0..tiles[1] {
            for iz in 0..tiles[2] {
                let offset = [
                    ix as f64 * TIP3P_CELL_ANGSTROM - template_minimum[0],
                    iy as f64 * TIP3P_CELL_ANGSTROM - template_minimum[1],
                    iz as f64 * TIP3P_CELL_ANGSTROM - template_minimum[2],
                ];
                for (water_index, atoms) in box_template.atoms.chunks_exact(3).enumerate() {
                    let positions = std::array::from_fn(|atom_index| Vec3 {
                        x: atoms[atom_index].position.x + offset[0],
                        y: atoms[atom_index].position.y + offset[1],
                        z: atoms[atom_index].position.z + offset[2],
                    });
                    let oxygen = positions[0];
                    if oxygen.x < 0.0
                        || oxygen.y < 0.0
                        || oxygen.z < 0.0
                        || oxygen.x >= box_lengths[0]
                        || oxygen.y >= box_lengths[1]
                        || oxygen.z >= box_lengths[2]
                    {
                        continue;
                    }
                    let overlaps = system.atoms[..system.solute_atom_count]
                        .iter()
                        .any(|atom| oxygen.distance2(atom.position) < 2.4f64.powi(2));
                    if overlaps {
                        continue;
                    }
                    let potential = system.atoms[..system.solute_atom_count]
                        .iter()
                        .map(|atom| {
                            let distance = oxygen.distance2(atom.position).sqrt().max(0.5);
                            atom.charge / distance
                        })
                        .sum();
                    waters.push(Water {
                        positions,
                        potential,
                        tie: splitmix64(
                            options.seed
                                ^ ((ix as u64) << 48)
                                ^ ((iy as u64) << 32)
                                ^ ((iz as u64) << 16)
                                ^ water_index as u64,
                        ),
                    });
                }
            }
        }
    }

    let solute_charge = system.atoms[..system.solute_atom_count]
        .iter()
        .map(|atom| atom.charge)
        .sum::<f64>()
        .round() as i64;
    let volume_nm3 = box_lengths.iter().product::<f64>() / 1000.0;
    let salt_pairs = if options.add_ions {
        (options.salt_molar * AVOGADRO_PER_NM3 * volume_nm3).round() as usize
    } else {
        0
    };
    let sodium = if options.add_ions {
        salt_pairs + usize::try_from((-solute_charge).max(0)).unwrap_or(0)
    } else {
        0
    };
    let chloride = if options.add_ions {
        salt_pairs + usize::try_from(solute_charge.max(0)).unwrap_or(0)
    } else {
        0
    };
    let requested = sodium + chloride;
    if requested > waters.len() {
        return Err(BuildError::InsufficientSolvent {
            requested,
            available: waters.len(),
        });
    }
    let mut available = (0..waters.len()).collect::<Vec<_>>();
    let mut selected = Vec::<(usize, bool)>::new();
    for is_sodium in ion_schedule(sodium, chloride, solute_charge) {
        available.sort_by(|&left, &right| {
            let order = if is_sodium {
                waters[left].potential.total_cmp(&waters[right].potential)
            } else {
                waters[right].potential.total_cmp(&waters[left].potential)
            };
            order.then_with(|| waters[left].tie.cmp(&waters[right].tie))
        });
        let choice_position = available
            .iter()
            .position(|&candidate| {
                selected.iter().all(|(placed, _)| {
                    waters[candidate].positions[0].distance2(waters[*placed].positions[0])
                        >= 5.0f64.powi(2)
                })
            })
            .unwrap_or(0);
        let chosen = available.remove(choice_position);
        selected.push((chosen, is_sodium));
    }
    let replaced = selected
        .iter()
        .map(|(water, _)| *water)
        .collect::<BTreeSet<_>>();

    let (ow_radius, ow_epsilon) = parameters.nonbonded("OW")?;
    let (hw_radius, hw_epsilon) = parameters.nonbonded("HW")?;
    let oh = parameters.bond("OW", "HW")?;
    let hoh = parameters.angle("HW", "OW", "HW")?;
    for (water_index, water) in waters.iter().enumerate() {
        if replaced.contains(&water_index) {
            continue;
        }
        let residue_index = system.residues.len();
        let first_atom = system.atoms.len();
        let component = system.component_count;
        for (name, atom_type, element, charge, mass, radius, epsilon, position) in [
            (
                "O",
                "OW",
                8,
                -0.834,
                16.0,
                ow_radius,
                ow_epsilon,
                water.positions[0],
            ),
            (
                "H1",
                "HW",
                1,
                0.417,
                1.008,
                hw_radius,
                hw_epsilon,
                water.positions[1],
            ),
            (
                "H2",
                "HW",
                1,
                0.417,
                1.008,
                hw_radius,
                hw_epsilon,
                water.positions[2],
            ),
        ] {
            system.atoms.push(Atom {
                name: name.into(),
                atom_type: atom_type.into(),
                element,
                residue: residue_index,
                charge,
                mass,
                radius,
                epsilon,
                position,
            });
        }
        system.bonds.push(Bond {
            atoms: [first_atom, first_atom + 1],
            force: oh.force,
            length: oh.length,
        });
        system.bonds.push(Bond {
            atoms: [first_atom, first_atom + 2],
            force: oh.force,
            length: oh.length,
        });
        system.angles.push(Angle {
            atoms: [first_atom + 1, first_atom, first_atom + 2],
            force: hoh.force,
            radians: hoh.degrees.to_radians(),
        });
        system.exclusions.extend([
            BTreeSet::from([first_atom + 1, first_atom + 2]),
            BTreeSet::from([first_atom, first_atom + 2]),
            BTreeSet::from([first_atom, first_atom + 1]),
        ]);
        system.residues.push(Residue {
            name: "WAT".into(),
            number: residue_index as i32 + 1,
            insertion_code: None,
            chain: "W".into(),
            first_atom,
            atom_count: 3,
            component,
        });
        system.component_count += 1;
        system.water_residue_count += 1;
    }
    for (water_index, is_sodium) in selected {
        add_ion(
            system,
            templates,
            parameters,
            if is_sodium { "Na+" } else { "Cl-" },
            if is_sodium { "NA" } else { "CL" },
            waters[water_index].positions[0],
        )?;
        if is_sodium {
            system.sodium_count += 1;
        } else {
            system.chloride_count += 1;
        }
    }
    Ok(())
}

fn add_ion(
    system: &mut System,
    templates: &TemplateSet,
    parameters: &ParameterSet,
    template_name: &str,
    residue_name: &str,
    position: Vec3,
) -> Result<()> {
    let template = templates
        .ion(template_name)
        .ok_or_else(|| BuildError::ForceField(format!("missing {template_name} ion template")))?;
    let source = template
        .atoms
        .first()
        .ok_or_else(|| BuildError::ForceField(format!("empty {template_name} ion template")))?;
    let (radius, epsilon) = parameters.nonbonded(&source.atom_type)?;
    let residue_index = system.residues.len();
    let first_atom = system.atoms.len();
    system.atoms.push(Atom {
        name: residue_name.into(),
        atom_type: source.atom_type.clone(),
        element: source.element,
        residue: residue_index,
        charge: source.charge,
        mass: parameters.mass(&source.atom_type, source.element),
        radius,
        epsilon,
        position,
    });
    system.exclusions.push(BTreeSet::new());
    system.residues.push(Residue {
        name: residue_name.into(),
        number: residue_index as i32 + 1,
        insertion_code: None,
        chain: "I".into(),
        first_atom,
        atom_count: 1,
        component: system.component_count,
    });
    system.component_count += 1;
    Ok(())
}

fn ion_schedule(sodium: usize, chloride: usize, solute_charge: i64) -> Vec<bool> {
    let mut result = Vec::with_capacity(sodium + chloride);
    let neutral_sodium = usize::try_from((-solute_charge).max(0)).unwrap_or(0);
    let neutral_chloride = usize::try_from(solute_charge.max(0)).unwrap_or(0);
    result.extend(std::iter::repeat_n(true, neutral_sodium));
    result.extend(std::iter::repeat_n(false, neutral_chloride));
    let pairs = sodium
        .saturating_sub(neutral_sodium)
        .min(chloride.saturating_sub(neutral_chloride));
    for _ in 0..pairs {
        result.push(true);
        result.push(false);
    }
    result
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ion_schedule_starts_with_neutralization() {
        assert_eq!(
            ion_schedule(4, 2, -2),
            vec![true, true, true, false, true, false]
        );
        assert_eq!(ion_schedule(1, 3, 2), vec![false, false, true, false]);
    }
}
