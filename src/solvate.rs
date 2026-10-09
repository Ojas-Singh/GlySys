use std::collections::BTreeSet;

use crate::forcefield::{ParameterSet, TemplateSet};
use crate::model::{Angle, Atom, Bond, Residue, System, Vec3};
use crate::{BuildError, BuildOptions, Result};

const TIP3P_CELL_ANGSTROM: f64 = 18.774_349;
/// Molarity of pure water at 25 °C (997.05 g/L, 18.015 g/mol).
const WATER_MOLAR: f64 = 55.34;

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
    let (sodium, chloride) = if options.add_ions {
        ion_counts(waters.len(), solute_charge, options.salt_molar)
    } else {
        (0, 0)
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

/// Sodium and chloride ions for a solute of charge `solute_charge` (in e)
/// among `waters` water molecules, at a salt concentration of `salt_molar`.
///
/// The counts follow SLTCAP (Schmit, Kariyawasam, Needham and Smith, J. Chem.
/// Theory Comput. 2018, 14, 1823). With N0 the ion pairs that the water alone
/// would hold at that concentration, the box is in equilibrium with a bath of
/// the same concentration when N+ N- = N0^2, and it is neutral when
/// N+ - N- = -Q, which gives N+- = sqrt(N0^2 + Q^2/4) -+ Q/2. A charged solute
/// therefore brings fewer co-ions as well as more counter-ions; a neutral one
/// gets N0 pairs, and without salt only the neutralizing counter-ions remain.
/// N0 is counted from the water molecules, not from the volume of the box as
/// built, which also holds the solute and relaxes under pressure coupling.
fn ion_counts(waters: usize, solute_charge: i64, salt_molar: f64) -> (usize, usize) {
    let pairs_in_water = salt_molar * waters as f64 / WATER_MOLAR;
    let excess = usize::try_from(solute_charge.unsigned_abs()).unwrap_or(usize::MAX);
    let half = excess as f64 / 2.0;
    let co_ions = (pairs_in_water.hypot(half) - half).round().max(0.0) as usize;
    let counter_ions = co_ions.saturating_add(excess);
    if solute_charge < 0 {
        (counter_ions, co_ions)
    } else {
        (co_ions, counter_ions)
    }
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

    #[test]
    fn neutral_solute_gets_the_salt_of_its_water() {
        // 0.15 M in 5,534 waters: 15 pairs
        assert_eq!(ion_counts(5_534, 0, 0.15), (15, 15));
        assert_eq!(ion_counts(5_534, 0, 0.0), (0, 0));
    }

    #[test]
    fn charged_solute_trades_co_ions_for_counter_ions() {
        // N0 = 0.2 * 5291 / 55.34 = 19.12; sqrt(N0^2 + 4) = 19.23: 21.23 and 17.23
        assert_eq!(ion_counts(5_291, -4, 0.2), (21, 17));
        assert_eq!(ion_counts(5_291, 4, 0.2), (17, 21));
        // the product stays at N0^2 as the charge grows
        let (sodium, chloride) = ion_counts(5_291, -12, 0.2);
        assert_eq!((sodium, chloride), (26, 14));
    }

    #[test]
    fn ion_counts_are_always_neutral() {
        for charge in [-9_i64, -3, -1, 0, 1, 2, 7] {
            for salt in [0.0, 0.05, 0.15, 0.5] {
                let (sodium, chloride) = ion_counts(3_000, charge, salt);
                assert_eq!(sodium as i64 - chloride as i64, -charge, "{charge} {salt}");
            }
        }
        // without salt only the neutralizing ions are left
        assert_eq!(ion_counts(3_000, -8, 0.0), (8, 0));
        assert_eq!(ion_counts(3_000, 3, 0.0), (0, 3));
    }
}
