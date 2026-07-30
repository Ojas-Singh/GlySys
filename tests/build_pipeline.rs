use glysys::{
    BuildError, BuildOptions, BuildWarning, ResidueAnnotation, ResidueId, SystemBuilder,
    read_pdb_str, write_pdb_string,
};

const DIPEPTIDE: &str = include_str!("fixtures/dipeptide.pdb");
const GLYCAN: &str = include_str!("fixtures/glycan.pdb");
const DIPEPTIDE_TLEAP_PRMTOP: &str = include_str!("fixtures/dipeptide.prmtop");

#[test]
fn prepares_deterministic_solvated_protein_bundle() {
    let options = BuildOptions {
        salt_molar: 0.0,
        ..BuildOptions::default()
    };
    let first = SystemBuilder::new(options.clone())
        .unwrap()
        .prepare_pdb_str(DIPEPTIDE)
        .unwrap();
    let second = SystemBuilder::new(options)
        .unwrap()
        .prepare_pdb_str(DIPEPTIDE)
        .unwrap();
    assert_eq!(first.report().solute_atoms, 20);
    assert!(first.report().waters > 500);
    assert_eq!(first.report().sodium_ions, 0);
    assert_eq!(first.report().chloride_ions, 0);
    assert_eq!(first.report().total_atoms, second.report().total_atoms);
    assert_eq!(first.report().box_angstrom, second.report().box_angstrom);

    let directory = tempfile::tempdir().unwrap();
    first.write_bundle(directory.path()).unwrap();
    for name in [
        "system.prmtop",
        "system.inpcrd",
        "system.top",
        "system.gro",
        "manifest.json",
    ] {
        assert!(directory.path().join(name).is_file(), "{name}");
    }
    let topology = std::fs::read_to_string(directory.path().join("system.top")).unwrap();
    assert!(topology.contains("[ atomtypes ]"));
    assert!(topology.contains("[ settles ]"));
    assert!(!topology.contains("#include"));
    let prmtop = std::fs::read_to_string(directory.path().join("system.prmtop")).unwrap();
    assert!(prmtop.contains("%FLAG POINTERS"));
    assert!(prmtop.contains("%FLAG BOX_DIMENSIONS"));
    let native_charge = numeric_flag(&prmtop, "CHARGE");
    let tleap_charge = numeric_flag(DIPEPTIDE_TLEAP_PRMTOP, "CHARGE");
    assert_eq!(tleap_charge.len(), 20);
    for (native, reference) in native_charge[..20].iter().zip(tleap_charge) {
        assert!((native - reference).abs() < 1.0e-6);
    }
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(directory.path().join("manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["force_field_versions"]["protein"], "Amber ff14SB");
    assert_eq!(manifest["output_sha256"]["system.prmtop"], sha256(&prmtop));
    assert!(matches!(
        first.write_bundle(directory.path()),
        Err(BuildError::OutputExists(_))
    ));
    let coordinates =
        gro_coordinates(&std::fs::read_to_string(directory.path().join("system.gro")).unwrap());
    for axis in 0..3 {
        let minimum = coordinates[..first.report().solute_atoms]
            .iter()
            .map(|point| point[axis])
            .fold(f64::INFINITY, f64::min);
        let maximum = coordinates[..first.report().solute_atoms]
            .iter()
            .map(|point| point[axis])
            .fold(f64::NEG_INFINITY, f64::max);
        let box_length = first.report().box_angstrom[axis] / 10.0;
        assert!((minimum - (box_length - maximum)).abs() < 0.002);
    }
}

#[test]
fn recognizes_and_prepares_standalone_glycam_glycan() {
    let builder = SystemBuilder::new(BuildOptions {
        salt_molar: 0.0,
        ..BuildOptions::default()
    })
    .unwrap();
    let system = builder.prepare_pdb_str(GLYCAN).unwrap();
    assert_eq!(system.report().glycans.len(), 1);
    assert!(system.report().glycans[0].wurcs.starts_with("WURCS=2.0/"));
    assert_eq!(
        system.report().glycans[0].glycam.as_deref(),
        Some("DGlcpNAc")
    );
    assert!(system.report().solute_charge.abs() < 1.0e-3);
}

#[test]
fn preserves_supplied_glycan_hydrogen_coordinates() {
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    })
    .unwrap()
    .prepare_pdb_str(GLYCAN)
    .unwrap();
    assert!(
        system
            .report()
            .warnings
            .iter()
            .any(|warning| matches!(warning, BuildWarning::InputGlycanHydrogensPreserved(_)))
    );
    let directory = tempfile::tempdir().unwrap();
    system.write_bundle(directory.path()).unwrap();
    let gro = std::fs::read_to_string(directory.path().join("system.gro")).unwrap();
    let h1 = gro
        .lines()
        .skip(2)
        .find(|line| line.get(10..15).is_some_and(|name| name.trim() == "H1"))
        .unwrap();
    let coordinate = [
        h1[20..28].trim().parse::<f64>().unwrap(),
        h1[28..36].trim().parse::<f64>().unwrap(),
        h1[36..44].trim().parse::<f64>().unwrap(),
    ];
    let expected = [1.5828, 0.9286, -1.5045];
    for axis in 0..3 {
        assert!((coordinate[axis] - expected[axis]).abs() < 0.001);
    }
}

#[test]
fn builds_missing_glycan_hydrogens_without_heavy_atom_clashes() {
    let hydrogen_free = GLYCAN
        .lines()
        .filter(|line| {
            !(line.starts_with("ATOM  ") || line.starts_with("HETATM"))
                || (!matches!(line.get(76..78).map(str::trim), Some("H" | "D"))
                    && !line
                        .get(12..16)
                        .is_some_and(|name| matches!(name.trim().chars().next(), Some('H' | 'D'))))
        })
        .collect::<Vec<_>>()
        .join("\n");
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    })
    .unwrap()
    .prepare_pdb_str(&hydrogen_free)
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    system.write_bundle(directory.path()).unwrap();
    let records =
        gro_atom_records(&std::fs::read_to_string(directory.path().join("system.gro")).unwrap());
    let prmtop = std::fs::read_to_string(directory.path().join("system.prmtop")).unwrap();
    let h1 = records.iter().find(|(_, name, _)| name == "H1").unwrap().2;
    let c1 = records.iter().find(|(_, name, _)| name == "C1").unwrap().2;
    assert!((distance(h1, c1) - 1.09).abs() < 0.03);

    let atomic_numbers = numeric_flag(&prmtop, "ATOMIC_NUMBER");
    let mut bonded = vec![std::collections::BTreeSet::new(); records.len()];
    for flag in ["BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"] {
        for bond in numeric_flag(&prmtop, flag).chunks_exact(3) {
            let first = bond[0] as usize / 3;
            let second = bond[1] as usize / 3;
            bonded[first].insert(second);
            bonded[second].insert(first);
        }
    }
    for (hydrogen, atomic_number) in atomic_numbers.iter().enumerate() {
        if *atomic_number != 1.0 {
            continue;
        }
        let parent = *bonded[hydrogen].iter().next().unwrap();
        let bond_length = distance(records[hydrogen].2, records[parent].2);
        assert!((0.94..=1.12).contains(&bond_length));
        for (other, other_atomic_number) in atomic_numbers.iter().enumerate() {
            if *other_atomic_number == 1.0 || other == parent || bonded[hydrogen].contains(&other) {
                continue;
            }
            assert!(
                distance(records[hydrogen].2, records[other].2) > 1.35,
                "generated {} clashes with {}",
                records[hydrogen].1,
                records[other].1
            );
        }
    }
}

#[test]
fn ignores_free_form_pdb_metadata_during_glycan_analysis() {
    let input = format!(
        "REMARK| free-form text that is not a numeric PDB field\nREMARK    generated fixture\n{GLYCAN}"
    );
    let system = SystemBuilder::new(BuildOptions {
        salt_molar: 0.0,
        ..BuildOptions::default()
    })
    .unwrap()
    .prepare_pdb_str(&input)
    .unwrap();
    assert_eq!(system.report().glycans.len(), 1);
}

#[test]
fn adds_neutral_salt_pairs() {
    let system = SystemBuilder::new(BuildOptions::default())
        .unwrap()
        .prepare_pdb_str(DIPEPTIDE)
        .unwrap();
    assert!(system.report().sodium_ions > 0);
    assert_eq!(system.report().sodium_ions, system.report().chloride_ions);
    assert!(system.report().total_charge.abs() < 1.0e-3);
}

#[test]
fn supports_water_without_ions_and_fully_dry_outputs() {
    let water_only = SystemBuilder::new(BuildOptions {
        add_ions: false,
        ..BuildOptions::default()
    })
    .unwrap()
    .prepare_pdb_str(DIPEPTIDE)
    .unwrap();
    assert!(water_only.report().waters > 0);
    assert_eq!(water_only.report().sodium_ions, 0);
    assert_eq!(water_only.report().chloride_ions, 0);

    let dry = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    })
    .unwrap()
    .prepare_pdb_str(DIPEPTIDE)
    .unwrap();
    assert_eq!(dry.report().total_atoms, dry.report().solute_atoms);
    assert_eq!(dry.report().waters, 0);
    assert_eq!(dry.report().box_angstrom, [0.0; 3]);
    let directory = tempfile::tempdir().unwrap();
    dry.write_bundle(directory.path()).unwrap();
    let prmtop = std::fs::read_to_string(directory.path().join("system.prmtop")).unwrap();
    assert!(!prmtop.contains("%FLAG BOX_DIMENSIONS"));
    let gro = std::fs::read_to_string(directory.path().join("system.gro")).unwrap();
    let records = gro_atom_records(&gro);
    let previous_carbonyl = records
        .iter()
        .find(|(residue, name, _)| *residue == 1 && name == "C")
        .unwrap()
        .2;
    let nitrogen = records
        .iter()
        .find(|(residue, name, _)| *residue == 2 && name == "N")
        .unwrap()
        .2;
    let amide_hydrogen = records
        .iter()
        .find(|(residue, name, _)| *residue == 2 && name == "H")
        .unwrap()
        .2;
    assert!(distance(previous_carbonyl, amide_hydrogen) > 1.7);
    assert!((distance(nitrogen, amide_hydrogen) - 1.01).abs() < 0.02);
}

#[test]
fn rejects_missing_heavy_atoms() {
    let incomplete = DIPEPTIDE.replace(
        "ATOM     20  OXT GLY     2       9.380   5.481  -0.000  1.00  0.00\n",
        "",
    );
    let error = SystemBuilder::new(BuildOptions::default())
        .unwrap()
        .prepare_pdb_str(&incomplete)
        .unwrap_err();
    assert!(matches!(error, BuildError::MissingHeavyAtoms { .. }));

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("incomplete.pdb");
    std::fs::write(&path, incomplete).unwrap();
    let report = SystemBuilder::new(BuildOptions::default())
        .unwrap()
        .inspect_pdb(path)
        .unwrap();
    assert_eq!(report.missing_heavy_atoms.len(), 1);
}

#[test]
fn validates_user_options() {
    let error = SystemBuilder::new(BuildOptions {
        padding_angstrom: 0.0,
        ..BuildOptions::default()
    })
    .unwrap_err();
    assert!(matches!(error, BuildError::InvalidOption(_)));
}

#[test]
fn loads_json_options() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("options.json");
    std::fs::write(&path, r#"{"padding_angstrom": 9.5, "salt_molar": 0.0}"#).unwrap();
    let options = BuildOptions::from_config_file(path).unwrap();
    assert_eq!(options.padding_angstrom, 9.5);
    assert_eq!(options.salt_molar, 0.0);
}

#[test]
fn parameterizes_an_in_memory_structure_and_preserves_metadata() {
    let options = BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    };
    let mut structure = read_pdb_str(DIPEPTIDE, &options).unwrap();
    let residue = ResidueId {
        chain: String::new(),
        number: 1,
        insertion_code: None,
    };
    structure
        .metadata_mut()
        .residue_annotations
        .push(ResidueAnnotation {
            residue: residue.clone(),
            label: "consumer-owned".into(),
        });

    let system = SystemBuilder::new(options)
        .unwrap()
        .prepare_structure(&structure)
        .unwrap();
    assert_eq!(system.atom_count(), 20);
    assert_eq!(system.atoms().len(), 20);
    assert_eq!(system.metadata().residue_annotations[0].residue, residue);

    let serialized = write_pdb_string(&structure);
    assert!(serialized.contains("ATOM"));
    assert!(serialized.ends_with("END\n"));
}

fn sha256(contents: &str) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(contents.as_bytes()))
}

fn numeric_flag(contents: &str, name: &str) -> Vec<f64> {
    let marker = format!("%FLAG {name}");
    let mut lines = contents
        .lines()
        .skip_while(|line| line.trim_end() != marker)
        .skip(2);
    let mut values = Vec::new();
    for line in &mut lines {
        if line.starts_with("%FLAG") {
            break;
        }
        values.extend(
            line.split_whitespace()
                .map(|value| value.parse::<f64>().unwrap()),
        );
    }
    values
}

fn gro_coordinates(contents: &str) -> Vec<[f64; 3]> {
    let lines = contents.lines().collect::<Vec<_>>();
    let count: usize = lines[1].trim().parse().unwrap();
    lines[2..2 + count]
        .iter()
        .map(|line| {
            [
                line[20..28].trim().parse().unwrap(),
                line[28..36].trim().parse().unwrap(),
                line[36..44].trim().parse().unwrap(),
            ]
        })
        .collect()
}

fn gro_atom_records(contents: &str) -> Vec<(usize, String, [f64; 3])> {
    let lines = contents.lines().collect::<Vec<_>>();
    let count: usize = lines[1].trim().parse().unwrap();
    lines[2..2 + count]
        .iter()
        .map(|line| {
            (
                line[0..5].trim().parse().unwrap(),
                line[10..15].trim().to_string(),
                [
                    line[20..28].trim().parse::<f64>().unwrap() * 10.0,
                    line[28..36].trim().parse::<f64>().unwrap() * 10.0,
                    line[36..44].trim().parse::<f64>().unwrap() * 10.0,
                ],
            )
        })
        .collect()
}

fn distance(first: [f64; 3], second: [f64; 3]) -> f64 {
    first
        .iter()
        .zip(second)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}
