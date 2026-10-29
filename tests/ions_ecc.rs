//! Divalent salts (CaCl2, MgCl2) in the bath and the electronic continuum
//! correction (ECC, prosECCo75 conventions) on a charged glycan.

use glysys::{BuildOptions, SystemBuilder};

/// Sulfated sialyl Lewis X (Neu5Ac carboxylate and a 6-sulfate), charge -2.
const SULFO_SLEX: &str = include_str!("fixtures/carbohydrates/3znl_sulfo_sialyl_lewis_x.pdb");

fn prepare(options: BuildOptions) -> glysys::ParameterizedSystem {
    SystemBuilder::new(BuildOptions {
        padding_angstrom: 10.0,
        ..options
    })
    .unwrap()
    .prepare_pdb_str(SULFO_SLEX)
    .unwrap()
}

#[test]
fn divalent_salts_add_calcium_and_magnesium_with_their_chlorides() {
    let system = prepare(BuildOptions {
        cacl2_molar: 0.1,
        mgcl2_molar: 0.05,
        ..BuildOptions::default()
    });
    let report = system.report();
    assert!(
        report.calcium_ions >= 1 && report.magnesium_ions >= 1,
        "{report:?}"
    );
    assert!(
        report.total_charge.abs() < 1.0e-6,
        "total charge {}",
        report.total_charge
    );
    let ions = |element: u8| {
        system
            .atoms()
            .iter()
            .filter(|atom| {
                atom.element() == element
                    && system.residues()[atom.residue_index()].name().len() <= 2
            })
            .map(|atom| atom.charge())
            .collect::<Vec<_>>()
    };
    assert_eq!(ions(20).len(), report.calcium_ions);
    assert_eq!(ions(12).len(), report.magnesium_ions);
    assert!(
        ions(20)
            .iter()
            .chain(ions(12).iter())
            .all(|&q| (q - 2.0).abs() < 1.0e-9)
    );
}

#[test]
fn ecc_scales_ions_and_charged_carbohydrate_groups_to_three_quarters() {
    let plain = prepare(BuildOptions {
        cacl2_molar: 0.1,
        mgcl2_molar: 0.05,
        ..BuildOptions::default()
    });
    let ecc = prepare(BuildOptions {
        cacl2_molar: 0.1,
        mgcl2_molar: 0.05,
        ecc: true,
        ..BuildOptions::default()
    });
    let report = ecc.report();
    assert!(
        report.total_charge.abs() < 1.0e-6,
        "total charge {}",
        report.total_charge
    );
    // every bath ion is a prosECCo75 scaled-charge ion
    for atom in ecc.atoms() {
        let residue = ecc.residues()[atom.residue_index()].name();
        let expected = match residue {
            "NA" => Some(("Na_s", 0.75)),
            "CL" => Some(("Cl_s", -0.75)),
            "CA" => Some(("Ca_s", 1.5)),
            "MG" => Some(("Mg_ss", 1.5)),
            _ => None,
        };
        if let Some((kind, charge)) = expected {
            assert_eq!(atom.atom_type(), kind);
            assert!((atom.charge() - charge).abs() < 1.0e-9);
        }
    }
    // the solute carries 0.75 of its charge, and the carboxylate oxygens are
    // 0.75 of the unscaled ones
    let solute = |system: &glysys::ParameterizedSystem| -> f64 {
        system.atoms()[..report.solute_atoms]
            .iter()
            .map(|a| a.charge())
            .sum()
    };
    assert!(
        (solute(&ecc) - 0.75 * solute(&plain)).abs() < 1.0e-6,
        "{} vs {}",
        solute(&ecc),
        solute(&plain)
    );
    let carboxylate = |system: &glysys::ParameterizedSystem| {
        system.atoms()[..report.solute_atoms]
            .iter()
            .filter(|atom| matches!(atom.name(), "O1A" | "O1B"))
            .map(|atom| atom.charge())
            .collect::<Vec<_>>()
    };
    let (scaled, unscaled) = (carboxylate(&ecc), carboxylate(&plain));
    assert_eq!(scaled.len(), 2, "Neu5Ac carboxylate oxygens");
    for (s, u) in scaled.iter().zip(&unscaled) {
        assert!((s - 0.75 * u).abs() < 1.0e-9, "{s} vs {u}");
    }
}

#[test]
fn calcium_counterions_neutralize_the_glycan_in_place_of_sodium() {
    // charge -2: one Ca2+, and without NaCl or CaCl2 nothing else
    let alone = prepare(BuildOptions {
        counterion: glysys::Counterion::Calcium,
        salt_molar: 0.0,
        ..BuildOptions::default()
    })
    .report()
    .clone();
    assert_eq!(
        (
            alone.calcium_ions,
            alone.sodium_ions,
            alone.chloride_ions,
            alone.magnesium_ions
        ),
        (1, 0, 0, 0),
        "{alone:?}"
    );
    assert!(alone.total_charge.abs() < 1.0e-6);
    // with NaCl and CaCl2 on top, the salt is counted as for a neutral solute
    let sodium = prepare(BuildOptions {
        cacl2_molar: 0.1,
        ..BuildOptions::default()
    })
    .report()
    .clone();
    let calcium = prepare(BuildOptions {
        cacl2_molar: 0.1,
        counterion: glysys::Counterion::Calcium,
        ..BuildOptions::default()
    })
    .report()
    .clone();
    assert_eq!(calcium.calcium_ions, sodium.calcium_ions + 1, "{calcium:?}");
    assert_eq!(
        calcium.sodium_ions,
        calcium.chloride_ions - 2 * sodium.calcium_ions
    );
    assert!(
        calcium.sodium_ions < sodium.sodium_ions,
        "{calcium:?} vs {sodium:?}"
    );
    assert!(calcium.total_charge.abs() < 1.0e-6);
    // ECC: Ca_s at +1.5 against the glycan's -1.5
    let ecc = prepare(BuildOptions {
        counterion: glysys::Counterion::Magnesium,
        salt_molar: 0.0,
        ecc: true,
        ..BuildOptions::default()
    });
    assert_eq!(ecc.report().magnesium_ions, 1);
    assert!(ecc.report().total_charge.abs() < 1.0e-6);
}
