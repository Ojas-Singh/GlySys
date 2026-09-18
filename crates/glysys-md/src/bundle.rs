//! Lossless verification for the files emitted beside a GlySys snapshot.
//!
//! A GROMACS text topology is intentionally not imported into the engine:
//! atom types and parameter provenance can be lost in a `.top` file.  This
//! module therefore performs a strict, read-only cross-check instead.  It
//! catches atom-order, box, constraint, exception, and interaction-function
//! changes before a native run is launched.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use glysys::ParameterizedSystem;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VerificationReport {
    pub topology: String,
    pub coordinates: String,
    pub atoms: usize,
    pub bonds: usize,
    pub angles: usize,
    pub dihedrals: usize,
    pub one_four_pairs: usize,
    pub rigid_waters: usize,
    pub coordinate_tolerance_angstrom: f64,
    pub max_coordinate_error_angstrom: f64,
    pub box_error_angstrom: [f64; 3],
    pub box_angstrom: [f64; 3],
}

#[derive(Debug)]
struct GroAtom {
    residue_name: String,
    atom_name: String,
    serial: usize,
    position_angstrom: [f64; 3],
}

#[derive(Debug)]
struct ParsedGro {
    atoms: Vec<GroAtom>,
    box_angstrom: [f64; 3],
}

#[derive(Debug)]
struct TopAtom {
    atom_type: String,
    residue_name: String,
    atom_name: String,
    charge: f64,
    mass: f64,
}

#[derive(Debug, Default)]
struct ParsedTop {
    atomtypes: BTreeSet<String>,
    atoms: Vec<TopAtom>,
    bonds: BTreeSet<(usize, usize)>,
    angles: BTreeSet<(usize, usize, usize)>,
    dihedrals: Vec<([usize; 4], i32)>,
    pairs: BTreeSet<(usize, usize)>,
    settles: BTreeSet<usize>,
    exclusions: BTreeSet<usize>,
}

fn field<'a>(line: &'a str, start: usize, end: usize, what: &str) -> Result<&'a str> {
    if line.len() < end {
        bail!("GROMACS .gro atom line is too short for {what}");
    }
    Ok(line.get(start..end).unwrap_or_default())
}

fn parse_fixed<T: std::str::FromStr>(line: &str, start: usize, end: usize, what: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    field(line, start, end, what)?
        .trim()
        .parse::<T>()
        .map_err(|error| anyhow::anyhow!("GROMACS .gro {what}: {error}"))
}

fn parse_gro(path: &Path) -> Result<ParsedGro> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading GROMACS coordinates {}", path.display()))?;
    let mut lines = text.lines();
    let _title = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("GROMACS .gro is missing its title"))?;
    let count: usize = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("GROMACS .gro is missing its atom count"))?
        .trim()
        .parse()
        .context("GROMACS .gro atom count is invalid")?;
    if count == 0 {
        bail!("GROMACS .gro contains no atoms");
    }
    let mut atoms = Vec::with_capacity(count);
    for _ in 0..count {
        let line = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("GROMACS .gro ended before all atoms were read"))?;
        let residue_name = field(line, 5, 10, "residue name")?.trim().to_owned();
        let atom_name = field(line, 10, 15, "atom name")?.trim().to_owned();
        let serial = parse_fixed::<usize>(line, 15, 20, "atom serial")?;
        let position_angstrom = [
            parse_fixed::<f64>(line, 20, 28, "x coordinate")? * 10.0,
            parse_fixed::<f64>(line, 28, 36, "y coordinate")? * 10.0,
            parse_fixed::<f64>(line, 36, 44, "z coordinate")? * 10.0,
        ];
        if position_angstrom.iter().any(|value| !value.is_finite()) {
            bail!("GROMACS .gro contains a non-finite coordinate");
        }
        atoms.push(GroAtom {
            residue_name,
            atom_name,
            serial,
            position_angstrom,
        });
    }
    let box_line = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("GROMACS .gro is missing its periodic box"))?;
    let box_values: Vec<f64> = box_line
        .split_whitespace()
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|error| anyhow::anyhow!("GROMACS .gro box value '{value}': {error}"))
        })
        .collect::<Result<_>>()?;
    if box_values.len() != 3 {
        bail!("only orthorhombic three-vector GROMACS boxes are supported");
    }
    let box_angstrom = [
        box_values[0] * 10.0,
        box_values[1] * 10.0,
        box_values[2] * 10.0,
    ];
    if box_angstrom
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.)
    {
        bail!("GROMACS .gro has an invalid periodic box");
    }
    if lines.any(|line| !line.trim().is_empty()) {
        bail!("GROMACS .gro contains data after its box line");
    }
    Ok(ParsedGro {
        atoms,
        box_angstrom,
    })
}

fn uncomment(line: &str) -> &str {
    line.split(';').next().unwrap_or_default().trim()
}

fn section_header(line: &str) -> Option<&str> {
    let line = line.trim();
    (line.starts_with('[') && line.ends_with(']')).then(|| line[1..line.len() - 1].trim())
}

fn parse_usize(value: &str, what: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|error| anyhow::anyhow!("GROMACS topology {what}: {error}"))
}

fn parse_f64(value: &str, what: &str) -> Result<f64> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| anyhow::anyhow!("GROMACS topology {what}: {error}"))?;
    if !parsed.is_finite() {
        bail!("GROMACS topology {what} is not finite");
    }
    Ok(parsed)
}

fn canonical_pair(first: usize, second: usize) -> (usize, usize) {
    (first.min(second), first.max(second))
}

fn parse_topology(path: &Path) -> Result<ParsedTop> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading GROMACS topology {}", path.display()))?;
    let mut parsed = ParsedTop::default();
    let mut section = String::new();
    let supported = BTreeSet::from([
        "defaults",
        "atomtypes",
        "moleculetype",
        "atoms",
        "bonds",
        "angles",
        "dihedrals",
        "pairs",
        "settles",
        "exclusions",
        "system",
        "molecules",
    ]);
    let mut molecule_decl = None::<(String, usize)>;
    let mut system_seen = false;
    let mut defaults_seen = false;
    for (line_number, raw_line) in text.lines().enumerate() {
        let line_number = line_number + 1;
        let line = uncomment(raw_line);
        if line.is_empty() {
            continue;
        }
        if let Some(header) = section_header(line) {
            section = header.to_ascii_lowercase();
            if !supported.contains(section.as_str()) {
                bail!("unsupported GROMACS topology section [{header}] at line {line_number}");
            }
            continue;
        }
        if section.is_empty() {
            bail!("GROMACS topology data precedes a section at line {line_number}");
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        match section.as_str() {
            "defaults" => {
                if defaults_seen || fields.len() < 5 {
                    bail!("GROMACS topology defaults section is malformed at line {line_number}");
                }
                defaults_seen = true;
                if fields[0] != "1"
                    || fields[1] != "2"
                    || fields[2].to_ascii_lowercase() != "yes"
                    || (parse_f64(fields[3], "fudgeLJ")? - 0.5).abs() > 1e-9
                    || (parse_f64(fields[4], "fudgeQQ")? - 0.833333333333).abs() > 1e-9
                {
                    bail!("GROMACS defaults change the GlySys mixing or 1-4 convention");
                }
            }
            "atomtypes" => {
                if fields.len() < 7 {
                    bail!("GROMACS atomtypes row is malformed at line {line_number}");
                }
                let name = fields[0].to_owned();
                if !parsed.atomtypes.insert(name.clone()) {
                    bail!("duplicate GROMACS atomtype '{name}'");
                }
                // Validate numeric columns that are part of the emitted
                // contract. The atomtype charge is deliberately zero; atom
                // charges live in [atoms].
                let _ = parse_f64(fields[2], "atomtype mass")?;
                let _ = parse_f64(fields[3], "atomtype charge")?;
                let _ = parse_f64(fields[5], "atomtype sigma")?;
                let _ = parse_f64(fields[6], "atomtype epsilon")?;
            }
            "moleculetype" => {
                if molecule_decl.is_some() || fields.len() != 2 {
                    bail!("only one simple moleculetype declaration is supported");
                }
                molecule_decl = Some((
                    fields[0].to_owned(),
                    parse_usize(fields[1], "moleculetype nrexcl")?,
                ));
            }
            "atoms" => {
                if fields.len() < 8 {
                    bail!("GROMACS atoms row is malformed at line {line_number}");
                }
                let index = parse_usize(fields[0], "atom index")?;
                if index != parsed.atoms.len() + 1 {
                    bail!("GROMACS atom indices are not contiguous at line {line_number}");
                }
                parsed.atoms.push(TopAtom {
                    atom_type: fields[1].to_owned(),
                    residue_name: fields[3].to_owned(),
                    atom_name: fields[4].to_owned(),
                    charge: parse_f64(fields[6], "atom charge")?,
                    mass: parse_f64(fields[7], "atom mass")?,
                });
            }
            "bonds" => {
                if fields.len() < 3 || fields[2] != "1" {
                    bail!("unsupported GROMACS bond function at line {line_number}");
                }
                let pair = canonical_pair(
                    parse_usize(fields[0], "bond atom")?,
                    parse_usize(fields[1], "bond atom")?,
                );
                if !parsed.bonds.insert(pair) {
                    bail!("duplicate GROMACS bond at line {line_number}");
                }
            }
            "angles" => {
                if fields.len() < 4 || fields[3] != "1" {
                    bail!("unsupported GROMACS angle function at line {line_number}");
                }
                let triple = (
                    parse_usize(fields[0], "angle atom")?,
                    parse_usize(fields[1], "angle atom")?,
                    parse_usize(fields[2], "angle atom")?,
                );
                if !parsed.angles.insert(triple) {
                    bail!("duplicate GROMACS angle at line {line_number}");
                }
            }
            "dihedrals" => {
                if fields.len() < 5 {
                    bail!("GROMACS dihedral row is malformed at line {line_number}");
                }
                let function = fields[4]
                    .parse::<i32>()
                    .map_err(|error| anyhow::anyhow!("GROMACS dihedral function: {error}"))?;
                if function != 4 && function != 9 {
                    bail!("unsupported GROMACS dihedral function {function} at line {line_number}");
                }
                parsed.dihedrals.push((
                    [
                        parse_usize(fields[0], "dihedral atom")?,
                        parse_usize(fields[1], "dihedral atom")?,
                        parse_usize(fields[2], "dihedral atom")?,
                        parse_usize(fields[3], "dihedral atom")?,
                    ],
                    function,
                ));
            }
            "pairs" => {
                if fields.len() < 3 || fields[2] != "1" {
                    bail!("unsupported GROMACS 1-4 pair function at line {line_number}");
                }
                let pair = canonical_pair(
                    parse_usize(fields[0], "pair atom")?,
                    parse_usize(fields[1], "pair atom")?,
                );
                if !parsed.pairs.insert(pair) {
                    bail!("duplicate GROMACS 1-4 pair at line {line_number}");
                }
            }
            "settles" => {
                if fields.len() < 4 || fields[1] != "1" {
                    bail!("unsupported GROMACS settle function at line {line_number}");
                }
                let oxygen = parse_usize(fields[0], "settles oxygen")?;
                let doh = parse_f64(fields[2], "settles OH")?;
                let dhh = parse_f64(fields[3], "settles HH")?;
                if (doh - 0.09572).abs() > 1e-6 || (dhh - 0.151390065).abs() > 1e-6 {
                    bail!("GROMACS SETTLE geometry differs from the prepared TIP3P system");
                }
                if !parsed.settles.insert(oxygen) {
                    bail!("duplicate GROMACS SETTLE declaration at line {line_number}");
                }
            }
            "exclusions" => {
                if fields.len() != 3 {
                    bail!("GlySys water exclusions must contain exactly three atoms");
                }
                let atoms: Vec<usize> = fields
                    .iter()
                    .map(|value| parse_usize(value, "exclusion atom"))
                    .collect::<Result<_>>()?;
                let oxygen = *atoms.iter().min().unwrap_or(&0);
                if atoms.iter().collect::<BTreeSet<_>>().len() != 3
                    || !parsed.exclusions.insert(oxygen)
                {
                    bail!(
                        "duplicate or malformed GROMACS water exclusion row at line {line_number}"
                    );
                }
            }
            "system" => {
                if system_seen {
                    bail!("duplicate GROMACS system declaration");
                }
                system_seen = true;
            }
            "molecules" => {
                if fields.len() != 2 || fields[0] != "PreparedSystem" || fields[1] != "1" {
                    bail!("GROMACS molecules section does not describe one PreparedSystem");
                }
            }
            _ => unreachable!("supported section list and match must agree"),
        }
    }
    if !defaults_seen || molecule_decl != Some(("PreparedSystem".into(), 3)) || !system_seen {
        bail!("GROMACS topology is missing required GlySys declarations");
    }
    if parsed.atoms.is_empty() {
        bail!("GROMACS topology contains no atoms");
    }
    Ok(parsed)
}

fn expected_pair_set(system: &ParameterizedSystem) -> BTreeSet<(usize, usize)> {
    system
        .dihedrals()
        .iter()
        .filter(|dihedral| !dihedral.is_improper())
        .map(|dihedral| {
            let atoms = dihedral.atoms();
            canonical_pair(atoms[0] + 1, atoms[3] + 1)
        })
        .collect()
}

fn is_water_atom(system: &ParameterizedSystem, atom: usize) -> bool {
    system
        .residues()
        .get(system.atoms()[atom].residue_index())
        .is_some_and(|residue| residue.name() == "WAT")
}

fn expected_bond_set(system: &ParameterizedSystem) -> BTreeSet<(usize, usize)> {
    system
        .bonds()
        .iter()
        .filter_map(|bond| {
            let atoms = bond.atoms();
            (!is_water_atom(system, atoms[0]) || !is_water_atom(system, atoms[1]))
                .then(|| canonical_pair(atoms[0] + 1, atoms[1] + 1))
        })
        .collect()
}

fn expected_angle_set(system: &ParameterizedSystem) -> BTreeSet<(usize, usize, usize)> {
    system
        .angles()
        .iter()
        .filter_map(|angle| {
            let atoms = angle.atoms();
            (!is_water_atom(system, atoms[0])
                || !is_water_atom(system, atoms[1])
                || !is_water_atom(system, atoms[2]))
            .then_some((atoms[0] + 1, atoms[1] + 1, atoms[2] + 1))
        })
        .collect()
}

pub fn verify_bundle(
    system: &ParameterizedSystem,
    topology_path: &Path,
    coordinates_path: &Path,
) -> Result<VerificationReport> {
    let gro = parse_gro(coordinates_path)?;
    let top = parse_topology(topology_path)?;
    if gro.atoms.len() != system.atom_count() || top.atoms.len() != system.atom_count() {
        bail!(
            "GROMACS bundle atom count ({}/{}) does not match snapshot ({})",
            gro.atoms.len(),
            top.atoms.len(),
            system.atom_count()
        );
    }
    let tolerance = 0.006; // system.gro is emitted to 0.001 nm (0.005 Å).
    let mut max_coordinate_error = 0.0_f64;
    for (index, ((gro_atom, top_atom), atom)) in gro
        .atoms
        .iter()
        .zip(&top.atoms)
        .zip(system.atoms())
        .enumerate()
    {
        let residue = &system.residues()[atom.residue_index()];
        if gro_atom.serial != (index + 1) % 100_000
            || gro_atom.residue_name != residue.name()
            || gro_atom.atom_name != atom.name()
            || top_atom.residue_name != residue.name()
            || top_atom.atom_name != atom.name()
            || top_atom.atom_type != atom.atom_type()
            || (top_atom.charge - atom.charge()).abs() > 2e-6
            || (top_atom.mass - atom.mass()).abs() > 2e-5
            || !top.atomtypes.contains(atom.atom_type())
        {
            bail!(
                "GROMACS atom {}/snapshot atom disagree at index {}",
                index + 1,
                index
            );
        }
        let expected = atom.position();
        let errors = [
            (gro_atom.position_angstrom[0] - expected.x).abs(),
            (gro_atom.position_angstrom[1] - expected.y).abs(),
            (gro_atom.position_angstrom[2] - expected.z).abs(),
        ];
        max_coordinate_error = max_coordinate_error.max(errors.into_iter().fold(0., f64::max));
    }
    if max_coordinate_error > tolerance {
        bail!(
            "GROMACS coordinates differ from the snapshot by {:.6} Å (tolerance {:.6} Å)",
            max_coordinate_error,
            tolerance
        );
    }
    let expected_box = system.box_angstrom();
    let box_error = [
        (gro.box_angstrom[0] - expected_box[0]).abs(),
        (gro.box_angstrom[1] - expected_box[1]).abs(),
        (gro.box_angstrom[2] - expected_box[2]).abs(),
    ];
    if box_error.iter().any(|error| *error > 0.001) {
        bail!("GROMACS periodic box differs from the snapshot");
    }
    let expected_bonds = expected_bond_set(system);
    let expected_angles = expected_angle_set(system);
    let expected_pairs = expected_pair_set(system);
    if top.bonds != expected_bonds {
        bail!("GROMACS bond set differs from the snapshot");
    }
    if top.angles != expected_angles {
        bail!("GROMACS angle set differs from the snapshot");
    }
    let expected_dihedrals: Vec<([usize; 4], i32)> = system
        .dihedrals()
        .iter()
        .map(|dihedral| {
            let atoms = dihedral.atoms();
            (
                [atoms[0] + 1, atoms[1] + 1, atoms[2] + 1, atoms[3] + 1],
                if dihedral.is_improper() { 4 } else { 9 },
            )
        })
        .collect();
    if top.dihedrals != expected_dihedrals || top.pairs != expected_pairs {
        bail!("GROMACS dihedral or 1-4 pair set differs from the snapshot");
    }
    let expected_water_oxygens: BTreeSet<usize> = system
        .residues()
        .iter()
        .filter(|residue| residue.name() == "WAT" && residue.atom_range().len() == 3)
        .map(|residue| residue.atom_range().start + 1)
        .collect();
    if top.settles != expected_water_oxygens || top.exclusions != expected_water_oxygens {
        bail!("GROMACS SETTLE/exclusion declarations differ from the snapshot");
    }
    Ok(VerificationReport {
        topology: topology_path.display().to_string(),
        coordinates: coordinates_path.display().to_string(),
        atoms: system.atom_count(),
        bonds: top.bonds.len(),
        angles: top.angles.len(),
        dihedrals: top.dihedrals.len(),
        one_four_pairs: top.pairs.len(),
        rigid_waters: top.settles.len(),
        coordinate_tolerance_angstrom: tolerance,
        max_coordinate_error_angstrom: max_coordinate_error,
        box_error_angstrom: box_error,
        box_angstrom: expected_box,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glysys::{BuildOptions, SystemBuilder};

    #[test]
    fn generated_bundle_round_trips() {
        let system = SystemBuilder::new(BuildOptions {
            add_water: true,
            padding_angstrom: 6.0,
            ..BuildOptions::default()
        })
        .unwrap()
        .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let files = system.bundle_strings().unwrap();
        for (name, contents) in &files {
            fs::write(directory.path().join(name), contents).unwrap();
        }
        let report = verify_bundle(
            &system,
            &directory.path().join("system.top"),
            &directory.path().join("system.gro"),
        )
        .unwrap();
        assert_eq!(report.atoms, system.atom_count());
        assert_eq!(report.rigid_waters, system.report().waters);
        assert!(report.max_coordinate_error_angstrom <= report.coordinate_tolerance_angstrom);
    }

    #[test]
    fn unsupported_topology_function_is_rejected() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(
            &mut file,
            b"[ defaults ]\n1 2 yes 0.5 0.833333333333\n[ bonds ]\n1 2 2\n",
        )
        .unwrap();
        let error = parse_topology(file.path()).unwrap_err().to_string();
        assert!(
            error.contains("unsupported GROMACS bond function"),
            "{error}"
        );
    }
}
