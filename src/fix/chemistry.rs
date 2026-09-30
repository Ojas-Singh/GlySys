//! Residue knowledge used by the structure fixer: names, aliases, classes,
//! modified-residue parents and protonation thresholds.

/// Standard parent of a modified amino acid.
///
/// The table is PDBFixer's substitution table (OpenMM PDBFixer, MIT
/// licence, Copyright (c) 2013-2025 Stanford University and the Authors)
/// with residues that GlySys models natively removed: `HYP` (ff14SB and
/// GLYCAM O-linked forms), `HIP`/`CYM` (Amber protonation states) and `NLN`
/// (GLYCAM N-glycosylated asparagine).
#[rustfmt::skip]
pub(crate) const SUBSTITUTIONS: &[(&str, &str)] = &[
    ("2AS", "ASP"), ("3AH", "HIS"), ("5HP", "GLU"), ("5OW", "LYS"), ("ACL", "ARG"),
    ("AGM", "ARG"), ("AIB", "ALA"), ("ALM", "ALA"), ("ALO", "THR"), ("ALY", "LYS"),
    ("ARM", "ARG"), ("ASA", "ASP"), ("ASB", "ASP"), ("ASK", "ASP"), ("ASL", "ASP"),
    ("ASQ", "ASP"), ("AYA", "ALA"), ("BCS", "CYS"), ("BHD", "ASP"), ("BMT", "THR"),
    ("BNN", "ALA"), ("BUC", "CYS"), ("BUG", "LEU"), ("C5C", "CYS"), ("C6C", "CYS"),
    ("CAS", "CYS"), ("CCS", "CYS"), ("CEA", "CYS"), ("CGU", "GLU"), ("CHG", "ALA"),
    ("CLE", "LEU"), ("CME", "CYS"), ("CSD", "ALA"), ("CSO", "CYS"), ("CSP", "CYS"),
    ("CSS", "CYS"), ("CSW", "CYS"), ("CSX", "CYS"), ("CXM", "MET"), ("CY1", "CYS"),
    ("CY3", "CYS"), ("CYG", "CYS"), ("CYQ", "CYS"), ("DAH", "PHE"), ("DAL", "ALA"),
    ("DAR", "ARG"), ("DAS", "ASP"), ("DCY", "CYS"), ("DGL", "GLU"), ("DGN", "GLN"),
    ("DHA", "ALA"), ("DHI", "HIS"), ("DIL", "ILE"), ("DIV", "VAL"), ("DLE", "LEU"),
    ("DLY", "LYS"), ("DNP", "ALA"), ("DPN", "PHE"), ("DPR", "PRO"), ("DSN", "SER"),
    ("DSP", "ASP"), ("DTH", "THR"), ("DTR", "TRP"), ("DTY", "TYR"), ("DVA", "VAL"),
    ("EFC", "CYS"), ("FLA", "ALA"), ("FME", "MET"), ("GGL", "GLU"), ("GL3", "GLY"),
    ("GLZ", "GLY"), ("GMA", "GLU"), ("GSC", "GLY"), ("HAC", "ALA"), ("HAR", "ARG"),
    ("HIC", "HIS"), ("HMR", "ARG"), ("HPQ", "PHE"), ("HTR", "TRP"), ("IAS", "ASP"),
    ("IIL", "ILE"), ("IYR", "TYR"), ("KCX", "LYS"), ("LLP", "LYS"), ("LLY", "LYS"),
    ("LTR", "TRP"), ("LYM", "LYS"), ("LYZ", "LYS"), ("MAA", "ALA"), ("MEN", "ASN"),
    ("MHS", "HIS"), ("MIS", "SER"), ("MK8", "LEU"), ("MLE", "LEU"), ("MPQ", "GLY"),
    ("MSA", "GLY"), ("MSE", "MET"), ("MVA", "VAL"), ("NEM", "HIS"), ("NEP", "HIS"),
    ("NLE", "LEU"), ("NLP", "LEU"), ("NMC", "GLY"), ("OAS", "SER"), ("OCS", "CYS"),
    ("OMT", "MET"), ("PAQ", "TYR"), ("PCA", "GLU"), ("PEC", "CYS"), ("PHI", "PHE"),
    ("PHL", "PHE"), ("PR3", "CYS"), ("PRR", "ALA"), ("PTR", "TYR"), ("PYX", "CYS"),
    ("SAC", "SER"), ("SAR", "GLY"), ("SCH", "CYS"), ("SCS", "CYS"), ("SCY", "CYS"),
    ("SEL", "SER"), ("SEP", "SER"), ("SET", "SER"), ("SHC", "CYS"), ("SHR", "LYS"),
    ("SMC", "CYS"), ("SOC", "CYS"), ("STY", "TYR"), ("SVA", "SER"), ("TIH", "ALA"),
    ("TPL", "TRP"), ("TPO", "THR"), ("TPQ", "ALA"), ("TRG", "LYS"), ("TRO", "TRP"),
    ("TYB", "TYR"), ("TYI", "TYR"), ("TYQ", "TYR"), ("TYS", "TYR"), ("TYY", "TYR"),
    // Common modified residues missing from PDBFixer's table.
    ("MLY", "LYS"), ("M3L", "LYS"), ("FVA", "VAL"), ("DSG", "ASN"),
];

pub(crate) fn substitution(name: &str) -> Option<&'static str> {
    SUBSTITUTIONS
        .iter()
        .find(|(modified, _)| *modified == name)
        .map(|(_, parent)| *parent)
}

/// The 20 standard amino acids.
pub(crate) const AMINO_ACIDS: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
];

/// Residues with native ff14SB/GLYCAM templates kept under their own name.
pub(crate) const NATIVE_PROTEIN_EXTRAS: &[&str] = &["HYP", "NLN", "OLS", "OLT", "OLP"];

pub(crate) const DNA: &[&str] = &["DA", "DC", "DG", "DT"];
pub(crate) const RNA: &[&str] = &["A", "C", "G", "U"];

/// Map force-field, CHARMM and legacy names to (standard name, forced variant).
pub(crate) fn residue_alias(name: &str) -> (String, Option<&'static str>) {
    let upper = name.trim().to_ascii_uppercase();
    let (standard, variant): (&str, Option<&'static str>) = match upper.as_str() {
        "HID" | "HSD" | "HISD" | "HISA" => ("HIS", Some("HID")),
        "HIE" | "HSE" | "HISE" | "HISB" => ("HIS", Some("HIE")),
        "HIP" | "HSP" | "HISH" | "HIS+" => ("HIS", Some("HIP")),
        "CYX" | "CYS2" => ("CYS", Some("CYX")),
        "CYM" => ("CYS", Some("CYM")),
        "ASH" | "ASPP" | "ASPH" => ("ASP", Some("ASH")),
        "GLH" | "GLUP" | "GLUH" => ("GLU", Some("GLH")),
        "LYN" | "LSN" => ("LYS", Some("LYN")),
        "DA5" | "DA3" | "DAN" | "ADE" => ("DA", None),
        "DC5" | "DC3" | "DCN" | "CYT" => ("DC", None),
        "DG5" | "DG3" | "DGN" | "GUA" => ("DG", None),
        "DT5" | "DT3" | "DTN" | "THY" => ("DT", None),
        "A5" | "A3" | "AN" | "RA" | "RA5" | "RA3" => ("A", None),
        "C5" | "C3" | "CN" | "RC" | "RC5" | "RC3" => ("C", None),
        "G5" | "G3" | "GN" | "RG" | "RG5" | "RG3" => ("G", None),
        "U5" | "U3" | "UN" | "RU" | "RU5" | "RU3" | "URA" => ("U", None),
        "WAT" | "TIP" | "TIP3" | "TP3" | "SOL" | "H2O" | "DOD" | "SPC" | "T3P" => ("HOH", None),
        "NHE" => ("NH2", None),
        "SOD" | "NA+" => ("NA", None),
        "CLA" | "CL-" => ("CL", None),
        "POT" | "K+" => ("K", None),
        "CAL" | "CA2+" => ("CA", None),
        "MG2+" => ("MG", None),
        "ZN2" | "ZN2+" => ("ZN", None),
        _ => return (upper, None),
    };
    (standard.to_string(), variant)
}

/// Normalize legacy and force-field atom names to wwPDB v3 names.
pub(crate) fn atom_alias(residue: &str, name: &str, nucleic: bool) -> String {
    let mut name = name.trim().replace('*', "'");
    if nucleic {
        name = match name.as_str() {
            "O1P" => "OP1".into(),
            "O2P" => "OP2".into(),
            "O3P" => "OP3".into(),
            "C5M" => "C7".into(),
            "C1*" => "C1'".into(),
            _ => name,
        };
    }
    match (residue, name.as_str()) {
        (_, "OT1") | (_, "OC1") | (_, "O1") if !nucleic => "O".into(),
        (_, "OT2") | (_, "OC2") | (_, "O2") | (_, "OT") if !nucleic => "OXT".into(),
        ("ILE", "CD") => "CD1".into(),
        _ => name,
    }
}

/// Monatomic ions and metal residues kept as-is.
pub(crate) fn is_ion_residue(name: &str, heavy_atoms: usize) -> bool {
    heavy_atoms == 1
        && matches!(
            name,
            "NA" | "K"
                | "LI"
                | "RB"
                | "CS"
                | "CL"
                | "BR"
                | "IOD"
                | "I"
                | "F"
                | "MG"
                | "CA"
                | "ZN"
                | "MN"
                | "FE"
                | "FE2"
                | "CO"
                | "NI"
                | "CU"
                | "CU1"
                | "CD"
                | "HG"
                | "SR"
                | "BA"
                | "AL"
                | "PB"
                | "PT"
                | "AU"
                | "AG"
                | "YB"
                | "SM"
                | "GD"
                | "TB"
                | "EU"
                | "LA"
                | "CE"
                | "PR"
                | "ND"
                | "Y1"
                | "3CO"
                | "TL"
                | "CS1"
                | "CO3"
                | "OS"
                | "IR"
                | "RU"
                | "RH"
                | "PD"
                | "HO"
                | "ER"
                | "LU"
                | "CR"
                | "V"
                | "GA"
                | "SE"
        )
}

/// Elements that coordinate protein side chains (metal-binding logic).
pub(crate) fn is_metal(element: &str) -> bool {
    matches!(
        element,
        "ZN" | "FE"
            | "CU"
            | "CO"
            | "NI"
            | "MN"
            | "MG"
            | "CA"
            | "CD"
            | "HG"
            | "PT"
            | "AG"
            | "AU"
            | "PB"
            | "MO"
            | "W"
            | "V"
            | "CR"
            | "GA"
            | "NA"
            | "K"
            | "LI"
    )
}

/// Soft metals that deprotonate a coordinating cysteine thiol.
pub(crate) fn binds_thiolate(element: &str) -> bool {
    matches!(
        element,
        "ZN" | "FE" | "CU" | "CO" | "NI" | "CD" | "HG" | "PT" | "AG" | "AU" | "PB" | "MO" | "W"
    )
}

pub(crate) fn is_water_name(name: &str) -> bool {
    name == "HOH"
}

/// Model pKa values: the side chain is protonated when pH < pKa.
pub(crate) fn protonated_at(residue: &str, ph: f64) -> Option<bool> {
    let pka = match residue {
        "ASP" => 3.9,
        "GLU" => 4.3,
        "HIS" => 6.5,
        "CYS" => 8.3,
        "LYS" => 10.5,
        _ => return None,
    };
    Some(ph < pka)
}

/// wwPDB v3 standard names for Amber-only atom names in output.
pub(crate) fn standard_atom_name(amber_residue: &str, name: &str) -> String {
    match (amber_residue, name) {
        (_, "H5T") => "HO5'".into(),
        (_, "H3T") => "HO3'".into(),
        _ => name.to_string(),
    }
}

/// Typical X-H bond length by heavy-atom element.
pub(crate) fn hydrogen_bond_length(element: &str) -> f64 {
    match element {
        "C" => 1.09,
        "N" => 1.01,
        "O" => 0.96,
        "S" => 1.34,
        "P" => 1.42,
        "B" => 1.19,
        "SE" => 1.47,
        _ => 1.05,
    }
}

/// Heavy-atom van der Waals radius (Bondi) used for clash scoring.
pub(crate) fn vdw_radius(element: &str) -> f64 {
    match element {
        "H" | "D" => 1.10,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.80,
        "P" => 1.80,
        "SE" => 1.90,
        "F" => 1.47,
        "CL" => 1.75,
        "BR" => 1.85,
        "I" => 1.98,
        element if is_metal(element) => 1.2,
        _ => 1.70,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_natively_supported_residues_unsubstituted() {
        for name in ["HYP", "HIP", "CYM", "NLN"] {
            assert!(substitution(name).is_none(), "{name}");
        }
        assert_eq!(substitution("MSE"), Some("MET"));
        assert_eq!(residue_alias("HSD"), ("HIS".to_string(), Some("HID")));
        assert_eq!(atom_alias("ALA", "OT2", false), "OXT");
        assert_eq!(atom_alias("DA", "O1P", true), "OP1");
    }
}
