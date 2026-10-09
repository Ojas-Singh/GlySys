use std::collections::BTreeSet;

use crate::forcefield::{ParameterSet, TemplateSet};
use crate::model::{Angle, Atom, Bond, Residue, System, Vec3};
use crate::{BuildError, BuildOptions, Result};

const TIP3P_CELL_ANGSTROM: f64 = 18.774_349;
/// Molarity of pure water at 25 °C (997.05 g/L, 18.015 g/mol).
const WATER_MOLAR: f64 = 55.34;
/// An ion replaces a water at least this far from every solute atom and, by
/// the minimum image, from every other ion, while such waters are left.
const ION_SEPARATION_ANGSTROM: f64 = 5.0;

#[derive(Clone)]
struct Water {
    positions: [Vec3; 3],
    /// Squared distance from the oxygen to the nearest solute atom.
    solute_distance2: f64,
    /// Seeded hash of the water's place in the tiling: the order in which
    /// waters are offered to the ions.
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
                    let solute_distance2 = system.atoms[..system.solute_atom_count]
                        .iter()
                        .map(|atom| oxygen.distance2(atom.position))
                        .fold(f64::INFINITY, f64::min);
                    if solute_distance2 < 2.4f64.powi(2) {
                        continue;
                    }
                    waters.push(Water {
                        positions,
                        solute_distance2,
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
    let selected = place_ions(
        &waters,
        box_lengths,
        &ion_schedule(sodium, chloride, solute_charge),
    );
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

/// Choose the water each ion of `schedule` replaces (`true`: sodium).
///
/// The ions are spread through the solvent the way salt is in solution: the
/// waters are offered in a seeded pseudo-random order, and an ion takes the
/// first one that is `ION_SEPARATION_ANGSTROM` from the solute and from the
/// ions already placed, across the periodic faces of the box as well. Where
/// the box has no such water left, the distance to the solute is given up
/// first, then the distance between ions.
///
/// Ions used to go where the solute's own electrostatic potential was lowest
/// (sodium) or highest (chloride), without the potential of the ions already
/// placed. Around a charged solute that put every counter-ion on its surface
/// and every co-ion in one cluster in the far corner of the box.
fn place_ions(waters: &[Water], box_lengths: [f64; 3], schedule: &[bool]) -> Vec<(usize, bool)> {
    let separation2 = ION_SEPARATION_ANGSTROM.powi(2);
    let apart = |left: usize, right: usize| {
        let (a, b) = (waters[left].positions[0], waters[right].positions[0]);
        let image = |delta: f64, length: f64| delta - length * (delta / length).round();
        let (dx, dy, dz) = (
            image(a.x - b.x, box_lengths[0]),
            image(a.y - b.y, box_lengths[1]),
            image(a.z - b.z, box_lengths[2]),
        );
        dx * dx + dy * dy + dz * dz >= separation2
    };
    let mut available = (0..waters.len()).collect::<Vec<_>>();
    available.sort_by_key(|&water| (waters[water].tie, water));
    let mut selected = Vec::<(usize, bool)>::with_capacity(schedule.len());
    for &is_sodium in schedule {
        let clear_of_ions =
            |candidate: usize| selected.iter().all(|&(placed, _)| apart(candidate, placed));
        let position = available
            .iter()
            .position(|&candidate| {
                waters[candidate].solute_distance2 >= separation2 && clear_of_ions(candidate)
            })
            .or_else(|| {
                available
                    .iter()
                    .position(|&candidate| clear_of_ions(candidate))
            })
            .unwrap_or(0);
        selected.push((available.remove(position), is_sodium));
    }
    selected
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

    /// Waters on a 3 Å lattice in a 30 Å box, around a solute atom at its centre.
    fn lattice() -> (Vec<Water>, [f64; 3]) {
        let centre = Vec3 {
            x: 15.0,
            y: 15.0,
            z: 15.0,
        };
        let mut waters = Vec::new();
        for index in 0..1000_u64 {
            let oxygen = Vec3 {
                x: 1.5 + 3.0 * (index % 10) as f64,
                y: 1.5 + 3.0 * (index / 10 % 10) as f64,
                z: 1.5 + 3.0 * (index / 100) as f64,
            };
            waters.push(Water {
                positions: [oxygen; 3],
                solute_distance2: oxygen.distance2(centre),
                tie: splitmix64(index),
            });
        }
        (waters, [30.0; 3])
    }

    fn image_distance(a: Vec3, b: Vec3, box_lengths: [f64; 3]) -> f64 {
        let image = |delta: f64, length: f64| delta - length * (delta / length).round();
        (image(a.x - b.x, box_lengths[0]).powi(2)
            + image(a.y - b.y, box_lengths[1]).powi(2)
            + image(a.z - b.z, box_lengths[2]).powi(2))
        .sqrt()
    }

    #[test]
    fn ions_are_spread_through_the_solvent() {
        let (waters, box_lengths) = lattice();
        let schedule = ion_schedule(14, 10, -4);
        let placed = place_ions(&waters, box_lengths, &schedule);
        assert_eq!(placed.len(), 24);
        assert_eq!(placed.iter().filter(|(_, sodium)| *sodium).count(), 14);
        // the same waters for the same seed
        assert_eq!(placed, place_ions(&waters, box_lengths, &schedule));
        let position = |index: usize| waters[placed[index].0].positions[0];
        let mut octants = BTreeSet::new();
        for i in 0..placed.len() {
            // away from the solute, and from every other ion across the faces of the box too
            assert!(waters[placed[i].0].solute_distance2 >= ION_SEPARATION_ANGSTROM.powi(2));
            for j in 0..i {
                let distance = image_distance(position(i), position(j), box_lengths);
                assert!(distance >= ION_SEPARATION_ANGSTROM, "{i} {j}: {distance}");
            }
            let at = position(i);
            octants.insert((at.x > 15.0, at.y > 15.0, at.z > 15.0));
        }
        // not one corner of the box for the chloride and the solute's surface for the sodium
        assert!(octants.len() >= 6, "ions in {} octants", octants.len());
        for sodium in [true, false] {
            let nearest_solute = (0..placed.len())
                .filter(|&i| placed[i].1 == sodium)
                .map(|i| waters[placed[i].0].solute_distance2.sqrt())
                .fold(f64::INFINITY, f64::min);
            let farthest_solute = (0..placed.len())
                .filter(|&i| placed[i].1 == sodium)
                .map(|i| waters[placed[i].0].solute_distance2.sqrt())
                .fold(0.0, f64::max);
            assert!(farthest_solute - nearest_solute > 5.0, "sodium {sodium}");
        }
    }

    #[test]
    fn a_crowded_box_still_gets_its_ions() {
        let (mut waters, box_lengths) = lattice();
        // eight waters next to each other: no two are 5 Å apart
        waters.retain(|water| {
            let oxygen = water.positions[0];
            oxygen.x < 6.0 && oxygen.y < 6.0 && oxygen.z < 6.0
        });
        assert_eq!(waters.len(), 8);
        let placed = place_ions(&waters, box_lengths, &ion_schedule(3, 3, 0));
        let distinct = placed
            .iter()
            .map(|(water, _)| *water)
            .collect::<BTreeSet<_>>();
        assert_eq!((placed.len(), distinct.len()), (6, 6));
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
