//! Structure fixer: PDBFixer-equivalent repairs on small real structures.

use glysys::{
    BuildOptions, ComponentLibrary, FixOptions, MissingResidues, Naming, StructureFixer,
    SystemBuilder, read_pdb_str,
};

const CRAMBIN: &str = include_str!("fixtures/fix/1crn.pdb");
const SELENOMETHIONINE: &str = include_str!("fixtures/fix/1a62_mse.pdb");
const MICROHETEROGENEITY: &str = include_str!("fixtures/fix/1ejg_microheterogeneity.pdb");

fn fixer(options: FixOptions) -> StructureFixer {
    StructureFixer::new(options).unwrap()
}

fn atoms(pdb: &str) -> Vec<(String, i32, String, String, [f64; 3])> {
    pdb.lines()
        .filter(|line| line.starts_with("ATOM") || line.starts_with("HETATM"))
        .map(|line| {
            (
                line[17..20].trim().to_string(),
                line[22..26].trim().parse().unwrap(),
                line[12..16].trim().to_string(),
                line[76..78].trim().to_string(),
                [
                    line[30..38].trim().parse().unwrap(),
                    line[38..46].trim().parse().unwrap(),
                    line[46..54].trim().parse().unwrap(),
                ],
            )
        })
        .collect()
}

fn position(
    atoms: &[(String, i32, String, String, [f64; 3])],
    number: i32,
    name: &str,
) -> [f64; 3] {
    atoms
        .iter()
        .find(|atom| atom.1 == number && atom.2 == name)
        .unwrap_or_else(|| panic!("{number} {name} missing"))
        .4
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn dihedral(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let sub = |x: [f64; 3], y: [f64; 3]| [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
    let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let cross = |x: [f64; 3], y: [f64; 3]| {
        [
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ]
    };
    let (b0, b1, b2) = (sub(a, b), sub(c, b), sub(d, c));
    let n = dot(b1, b1).sqrt();
    let b1 = [b1[0] / n, b1[1] / n, b1[2] / n];
    let v = sub(
        b0,
        [
            b1[0] * dot(b0, b1),
            b1[1] * dot(b0, b1),
            b1[2] * dot(b0, b1),
        ],
    );
    let w = sub(
        b2,
        [
            b1[0] * dot(b2, b1),
            b1[1] * dot(b2, b1),
            b1[2] * dot(b2, b1),
        ],
    );
    dot(cross(b1, v), w).atan2(dot(v, w)).to_degrees()
}

#[test]
fn fixes_crambin_like_pdbfixer_and_the_result_parameterizes() {
    let fixed = fixer(FixOptions::default()).fix_pdb_str(CRAMBIN).unwrap();
    let report = &fixed.report;
    // PDBFixer adds the same 315 hydrogens to 1CRN at pH 7.
    assert_eq!(report.hydrogens_added, 315);
    assert_eq!(report.disulfides.len(), 3);
    assert_eq!(report.heavy_atoms_added, 0);
    assert_eq!(report.hydrogen_contacts, 0);
    assert!(fixed.pdb.contains("SSBOND   1 CYS A    3    CYS A   40"));
    // The fixed PDB is a valid GlySys preparation input.
    let prepared = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    })
    .unwrap()
    .prepare_pdb_str(&fixed.pdb)
    .unwrap();
    assert_eq!(prepared.atom_count(), 642);
    let charge = prepared.report().solute_charge;
    assert!((charge - charge.round()).abs() < 1e-3);
}

#[test]
fn models_an_internal_gap_listed_in_remark_465() {
    // Delete residues 20-23 of crambin and declare them unobserved.
    let mut pdb = String::from(
        "REMARK 465   M RES C SSSEQI\nREMARK 465     GLY A    20\nREMARK 465     THR A    21\nREMARK 465     PRO A    22\nREMARK 465     GLU A    23\n",
    );
    for line in CRAMBIN.lines() {
        let is_atom = line.starts_with("ATOM") || line.starts_with("HETATM");
        if is_atom && (20..=23).contains(&line[22..26].trim().parse::<i32>().unwrap()) {
            continue;
        }
        pdb.push_str(line);
        pdb.push('\n');
    }
    let fixed = fixer(FixOptions::default()).fix_pdb_str(&pdb).unwrap();
    let report = &fixed.report;
    assert_eq!(report.residues_added, 4);
    assert!(report.missing_residues[0].modelled);
    assert!(report.chain_breaks.is_empty());
    assert_eq!(report.remaining_clashes, 0, "{report:?}");
    let atoms = atoms(&fixed.pdb);
    for number in 19..=23 {
        let c = position(&atoms, number, "C");
        let n = position(&atoms, number + 1, "N");
        assert!(
            (distance(c, n) - 1.33).abs() < 0.06,
            "C{number}-N {}",
            distance(c, n)
        );
        let omega = dihedral(
            position(&atoms, number, "CA"),
            c,
            n,
            position(&atoms, number + 1, "CA"),
        );
        assert!(omega.abs() > 150.0, "omega {number} = {omega}");
    }
    // The modelled residues are complete, including hydrogens.
    assert!(atoms.iter().any(|a| a.1 == 20 && a.2 == "HA3"));
    assert!(atoms.iter().any(|a| a.1 == 21 && a.2 == "HG1"));
    assert!(atoms.iter().any(|a| a.1 == 22 && a.2 == "HD3"));
    assert!(atoms.iter().any(|a| a.1 == 23 && a.2 == "OE2"));
}

#[test]
fn skips_numbering_gaps_without_a_physical_break() {
    // Renumber crambin after residue 10 without removing anything.
    let pdb = CRAMBIN
        .lines()
        .map(|line| {
            if (line.starts_with("ATOM") || line.starts_with("HETATM"))
                && line[22..26].trim().parse::<i32>().unwrap() > 10
            {
                let number = line[22..26].trim().parse::<i32>().unwrap() + 5;
                format!("{}{number:>4}{}", &line[..22], &line[26..])
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let fixed = fixer(FixOptions::default()).fix_pdb_str(&pdb).unwrap();
    assert_eq!(fixed.report.residues_added, 0);
}

#[test]
fn replaces_selenomethionine_using_modres() {
    let options = FixOptions {
        missing_residues: MissingResidues::None,
        ..FixOptions::default()
    };
    let fixed = fixer(options).fix_pdb_str(SELENOMETHIONINE).unwrap();
    assert_eq!(fixed.report.replaced_residues.len(), 3);
    let atoms = atoms(&fixed.pdb);
    assert!(!atoms.iter().any(|a| a.0 == "MSE" || a.3 == "SE"));
    let sd = position(&atoms, 21, "SD");
    assert!((distance(sd, position(&atoms, 21, "CG")) - 1.81).abs() < 1e-3);

    let kept = fixer(FixOptions {
        missing_residues: MissingResidues::None,
        replace_nonstandard: false,
        ..FixOptions::default()
    })
    .fix_pdb_str(SELENOMETHIONINE)
    .unwrap();
    assert!(kept.pdb.contains("MSE A  21"));
    assert!(kept.report.components_missing.contains(&"MSE".to_string()));
}

#[test]
fn resolves_microheterogeneity_to_one_residue_type() {
    let fixed = fixer(FixOptions::default())
        .fix_pdb_str(MICROHETEROGENEITY)
        .unwrap();
    let atoms = atoms(&fixed.pdb);
    let residue_22 = atoms.iter().filter(|a| a.1 == 22).collect::<Vec<_>>();
    assert!(residue_22.iter().all(|a| a.0 == "PRO"));
    assert!(!residue_22.iter().any(|a| a.2 == "OG"));
    // Input hydrogens are rebuilt, not duplicated.
    assert_eq!(residue_22.iter().filter(|a| a.2 == "HA").count(), 1);
}

#[test]
fn adds_ligand_hydrogens_from_a_component_definition() {
    let pdb = "\
HETATM    1  C   ACT A   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  O   ACT A   1       1.230   0.000   0.000  1.00  0.00           O
HETATM    3  OXT ACT A   1      -0.650   1.100   0.000  1.00  0.00           O
HETATM    4  CH3 ACT A   1      -0.750  -1.300   0.000  1.00  0.00           C
END
";
    let cif = "data_ACT
_chem_comp.id ACT
_chem_comp.type NON-POLYMER
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
C C 0 -0.042 0.000 0.001
O O 0 -1.279 0.000 -0.001
OXT O -1 0.656 1.198 0.001
CH3 C 0 0.705 -1.296 0.000
H1 H 0 1.781 -1.113 0.000
H2 H 0 0.425 -1.866 0.887
H3 H 0 0.425 -1.865 -0.887
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
C O DOUB
C OXT SING
C CH3 SING
CH3 H1 SING
CH3 H2 SING
CH3 H3 SING
";
    let fixer = fixer(FixOptions::default());
    assert_eq!(fixer.component_requests(pdb).unwrap(), vec!["ACT"]);
    let without = fixer.fix_pdb_str(pdb).unwrap();
    assert_eq!(without.report.hydrogens_added, 0);
    let mut library = ComponentLibrary::new();
    library.add_cif(cif).unwrap();
    let fixed = fixer.with_components(library).fix_pdb_str(pdb).unwrap();
    assert_eq!(fixed.report.hydrogens_added, 3);
    let atoms = atoms(&fixed.pdb);
    let carbon = position(&atoms, 1, "CH3");
    for name in ["H1", "H2", "H3"] {
        let d = distance(carbon, position(&atoms, 1, name));
        assert!((d - 1.09).abs() < 0.02, "{name} {d}");
    }
}

#[test]
fn amber_naming_writes_protonation_variants() {
    let fixed = fixer(FixOptions {
        naming: Naming::Amber,
        ..FixOptions::default()
    })
    .fix_pdb_str(CRAMBIN)
    .unwrap();
    assert!(fixed.pdb.contains("CYX A   3"));
    let standard = fixer(FixOptions::default()).fix_pdb_str(CRAMBIN).unwrap();
    assert!(standard.pdb.contains("CYS A   3"));
    assert!(read_pdb_str(&standard.pdb, &BuildOptions::default()).is_ok());
}

#[test]
fn rejects_invalid_options() {
    assert!(
        StructureFixer::new(FixOptions {
            ph: 15.0,
            ..FixOptions::default()
        })
        .is_err()
    );
    assert!(
        StructureFixer::new(FixOptions {
            model: 0,
            ..FixOptions::default()
        })
        .is_err()
    );
}
