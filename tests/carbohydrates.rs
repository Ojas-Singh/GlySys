//! Carbohydrates are read from their atoms, not their names: wwPDB (CCD)
//! ligands from lectin structures and GLYCAM-named input prepare with the
//! GLYCAM residues their coordinates fit.

use glysys::{BuildOptions, BuildWarning, SystemBuilder};

fn inspect(pdb: &str) -> glysys::BuildReport {
    SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    })
    .unwrap()
    .inspect_pdb_str(pdb)
    .unwrap()
}

/// GLYCAM residues chosen, in input order.
fn resolved(report: &glysys::BuildReport) -> Vec<String> {
    report
        .warnings
        .iter()
        .filter_map(|warning| match warning {
            BuildWarning::GlycanNameNormalized(message) => message
                .split(" as GLYCAM ")
                .nth(1)
                .map(|tail| tail.split(';').next().unwrap().to_string()),
            _ => None,
        })
        .collect()
}

fn assert_integral(report: &glysys::BuildReport, charge: f64) {
    assert!(
        report.unsupported_residues.is_empty(),
        "{:?}",
        report.warnings
    );
    assert!(report.solute_atoms > 0);
    assert!(
        (report.solute_charge - charge).abs() < 1.0e-3,
        "charge {} instead of {charge}",
        report.solute_charge
    );
}

#[test]
fn a_methyl_glycoside_splits_into_the_sugar_and_an_ome_aglycone() {
    // 7JUB MMA: methyl alpha-D-mannopyranoside.
    let report = inspect(include_str!(
        "fixtures/carbohydrates/7jub_mma_methyl_mannoside.pdb"
    ));
    assert_integral(&report, 0.0);
    assert_eq!(resolved(&report), ["0MA + OME"]);
}

#[test]
fn a_sulfated_sialyl_lewis_x_carries_exact_glycam_charges() {
    // 3ZNL: Fuc(a1-3)[Neu5Ac(a2-3)Gal(a1-4)]GlcNAc6S, NGS/GLA/SIA/FUC.  The
    // sulfate becomes SO3 on a 3,4,6-linked GlcNAc; the sialic acid's CCD
    // acetyl names become GLYCAM's; -1 (sulfate) -1 (carboxylate) exactly.
    let report = inspect(include_str!(
        "fixtures/carbohydrates/3znl_sulfo_sialyl_lewis_x.pdb"
    ));
    assert_integral(&report, -2.0);
    assert_eq!(resolved(&report), ["QYB + ROH + SO3", "3LA", "0SA", "0fA"]);
}

#[test]
fn sucrose_type_fructose_keeps_its_own_glycosidic_oxygen() {
    // 3ALU raffinose: Gal(a1-6)Glc(a1-2)Fruf(b2-.
    let report = inspect(include_str!("fixtures/carbohydrates/3alu_raffinose.pdb"));
    assert_integral(&report, 0.0);
    assert_eq!(resolved(&report), ["2CU", "6GA", "0LA"]);
}

#[test]
fn a_reducing_end_deposited_without_o1_gets_it_back_from_the_name() {
    // 4LJH GAL 203 has no O1: the anomer cannot be read, GAL means beta.
    let report = inspect(include_str!(
        "fixtures/carbohydrates/4ljh_gal_without_o1.pdb"
    ));
    assert_integral(&report, 0.0);
    assert_eq!(resolved(&report), ["0LB + ROH"]);
    assert!(report.warnings.iter().any(|warning| matches!(
        warning,
        BuildWarning::GlycanNameNormalized(message) if message.contains("rebuilt as ROH")
    )));
}

/// The prepared structure of `pdb`: its PDB text and the chiral volume at
/// C5 of the galactofuranose (C4, C6, O5 around C5).
fn prepared_galf_any(pdb: &str, suffix: &str) -> (String, f64) {
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    })
    .unwrap()
    .prepare_pdb_str(pdb)
    .unwrap();
    let text = system.pdb_string();
    let atom = |name: &str| {
        text.lines()
            .find(|line| {
                { line.starts_with("ATOM") || line.starts_with("HETATM") }
                    .then_some(line)
                    .filter(|line| {
                        line.get(17..20).is_some_and(|r| r.ends_with(suffix))
                            && line.get(12..16).map(str::trim) == Some(name)
                    })
                    .is_some()
            })
            .map(|line| {
                [30..38, 38..46, 46..54].map(|range| line[range].trim().parse::<f64>().unwrap())
            })
            .unwrap_or_else(|| panic!("no {name} in the galactofuranose"))
    };
    let (c5, c4, c6, o5) = (atom("C5"), atom("C4"), atom("C6"), atom("O5"));
    let d = |p: [f64; 3]| [p[0] - c5[0], p[1] - c5[1], p[2] - c5[2]];
    let (u, v, w) = (d(c4), d(c6), d(o5));
    let volume = u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0]);
    (text, volume)
}

/// The furanose GLYCAM names crabWURCS reads from a prepared structure.
fn crabwurcs_galf(text: &str) -> String {
    let glycans = crabwurcs::pdb::extract_glycans_from_str(text, false).unwrap();
    glycans
        .iter()
        .filter_map(|glycan| {
            crabwurcs::write_notation(&glycan.graph, crabwurcs::Format::Glycam).ok()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn glycam_web_galactofuranose_is_rebuilt_as_beta_d_galactofuranose() {
    // DGalfb1-2LArafb1-2LArafb1-OH assembled from GLYCAM-Web's library
    // coordinates (ROH 2aU 2aU 0LU): its "beta-D-Galf" is beta-L-altrofuranose
    // (C1 and C5 inverted; crabWURCS reads LAltf). GlySys rebuilds it from the
    // corrected template with its glycosidic bond kept: the prepared structure
    // is the beta-D-Galf GLYCAM-Web was asked for.
    let pdb = include_str!("fixtures/carbohydrates/glycam_web_galf_araf_araf.pdb");
    assert!(crabwurcs_galf(pdb).starts_with("LAltfb1-2"));
    let report = inspect(pdb);
    assert_integral(&report, 0.0);
    assert!(
        report.warnings.iter().any(|warning| matches!(
            warning,
            BuildWarning::GlycanStereochemistry(message)
                if message.contains("0LU") && message.contains("rebuilt")
        )),
        "{:?}",
        report.warnings
    );
    let (text, _) = prepared_galf_any(pdb, "LU");
    let names = crabwurcs_galf(&text);
    assert!(
        names.starts_with("DGalfb1-2LArafb1-2LArafb1"),
        "crabWURCS reads {names}"
    );
    // the glycosidic bond to the O2 of the arabinofuranose it is on is still a bond
    let positions = |residue: &str, atom: &str| {
        text.lines()
            .filter(|line| {
                line.get(17..20) == Some(residue) && line.get(12..16).map(str::trim) == Some(atom)
            })
            .map(|line| {
                [30..38, 38..46, 46..54].map(|range| line[range].trim().parse::<f64>().unwrap())
            })
            .collect::<Vec<_>>()
    };
    let c1 = positions("0LU", "C1")[0];
    let d = positions("2aU", "O2")
        .iter()
        .map(|o2| {
            ((c1[0] - o2[0]).powi(2) + (c1[1] - o2[1]).powi(2) + (c1[2] - o2[2]).powi(2)).sqrt()
        })
        .fold(f64::INFINITY, f64::min);
    assert!((1.3..1.6).contains(&d), "C1-O2 {d}");
}

#[test]
fn a_galactofuranose_with_c5_inverted_alone_is_alpha_d_galactofuranose() {
    // Inverting only C5 of GLYCAM-Web's "beta" Galf gives alpha-D-Galf: the
    // anomer is defined relative to C5, and GLYCAM-Web built C1 relative to
    // its wrong C5.  GlySys reads the coordinates, so it prepares the alpha
    // template (renaming the input's 0LU), and crabWURCS agrees.
    let (text, _) = prepared_galf_any(&d_galactofuranose_input(), "LD");
    let names = crabwurcs_galf(&text);
    assert!(names.starts_with("DGalfa1-2"), "crabWURCS reads {names}");
}

/// The GLYCAM-Web trisaccharide with C5 of the Galf inverted to D-galacto by
/// swapping its O5 (and HO5) with H5.
fn d_galactofuranose_input() -> String {
    let pdb = include_str!("fixtures/carbohydrates/glycam_web_galf_araf_araf.pdb");
    let atom = |name: &str| {
        pdb.lines()
            .find(|line| {
                line.get(17..20) == Some("0LU") && line.get(12..16).map(str::trim) == Some(name)
            })
            .map(|line| {
                [30..38, 38..46, 46..54].map(|range| line[range].trim().parse::<f64>().unwrap())
            })
            .unwrap()
    };
    let (c5, o5, h5) = (atom("C5"), atom("O5"), atom("H5"));
    let unit = |from: [f64; 3], to: [f64; 3]| {
        let d = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        [d[0] / n, d[1] / n, d[2] / n]
    };
    let (to_o, to_h) = (unit(c5, o5), unit(c5, h5));
    let place = |direction: [f64; 3], length: f64| [0, 1, 2].map(|k| c5[k] + length * direction[k]);
    let new_o5 = place(to_h, 1.43);
    let new_h5 = place(to_o, 1.10);
    let shift = [0, 1, 2].map(|k| new_o5[k] - o5[k]);
    pdb.lines()
        .map(|line| {
            let name = line.get(12..16).map(str::trim).unwrap_or("");
            if line.get(17..20) != Some("0LU") || !matches!(name, "O5" | "H5" | "H5O") {
                return line.to_string();
            }
            let old =
                [30..38, 38..46, 46..54].map(|range| line[range].trim().parse::<f64>().unwrap());
            let new = match name {
                "O5" => new_o5,
                "H5" => new_h5,
                _ => [0, 1, 2].map(|k| old[k] + shift[k]),
            };
            format!(
                "{}{:8.3}{:8.3}{:8.3}{}",
                &line[..30],
                new[0],
                new[1],
                new[2],
                &line[54..]
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn d_galactofuranose_prepares_without_a_stereochemistry_warning() {
    // The GLYCAM-Web trisaccharide with C5 of the Galf inverted (alpha-D-Galf).
    let report = inspect(&d_galactofuranose_input());
    assert_integral(&report, 0.0);
    assert!(
        !report
            .warnings
            .iter()
            .any(|warning| matches!(warning, BuildWarning::GlycanStereochemistry(_))),
        "{:?}",
        report.warnings
    );
    assert!(
        resolved(&report).iter().any(|name| name.ends_with("LD")),
        "{:?}",
        report.warnings
    );
}

#[test]
fn glycam_names_that_contradict_the_coordinates_are_corrected() {
    // The fixture's beta-GlcNAc renamed to the alpha residue 0YA: the
    // coordinates win and the anomer is read as beta (0YB).
    let pdb = include_str!("fixtures/glycan.pdb").replace("0YB", "0YA");
    let report = inspect(&pdb);
    assert_integral(&report, 0.0);
    assert_eq!(resolved(&report), ["0YB"]);
}
