use std::collections::BTreeMap;

use crate::{BuildOptions, BuildWarning};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ResidueRef {
    pub chain: String,
    pub name: String,
    pub number: i32,
    pub insertion_code: Option<char>,
}

impl std::fmt::Display for ResidueRef {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}/{}/{}{}",
            self.chain,
            self.name,
            self.number,
            self.insertion_code.unwrap_or(' ')
        )
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GlycanReport {
    pub attachment_site: Option<String>,
    pub wurcs: String,
    pub glycam: Option<String>,
    pub residue_count: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct BuildReport {
    pub input_sha256: String,
    pub force_field_versions: BTreeMap<String, String>,
    pub output_sha256: BTreeMap<String, String>,
    pub options: BuildOptions,
    pub selected_model: u32,
    pub chains: Vec<String>,
    pub glycans: Vec<GlycanReport>,
    pub protonation_decisions: Vec<String>,
    pub unsupported_residues: Vec<ResidueRef>,
    pub missing_heavy_atoms: Vec<String>,
    pub warnings: Vec<BuildWarning>,
    pub solute_atoms: usize,
    pub total_atoms: usize,
    pub residues: usize,
    pub waters: usize,
    pub sodium_ions: usize,
    pub chloride_ions: usize,
    pub solute_charge: f64,
    pub total_charge: f64,
    pub box_angstrom: [f64; 3],
}

pub(crate) fn force_field_versions() -> BTreeMap<String, String> {
    [
        ("source", "AmberTools 23.6"),
        ("protein", "Amber ff14SB"),
        ("carbohydrate", "GLYCAM06j-1"),
        ("water", "Amber TIP3P"),
        ("ions", "Joung-Chetham Na+/Cl- for TIP3P"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value.to_string()))
    .collect()
}
