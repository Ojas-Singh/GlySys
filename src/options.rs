use std::collections::BTreeMap;
use std::path::Path;

use crate::{BuildError, Result};

/// A versioned, mutually compatible force-field combination.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ForceFieldProfile {
    /// Amber ff14SB + GLYCAM06j-1 + TIP3P + Joung-Chetham monovalent ions.
    #[default]
    Ff14sbGlycam06j1Tip3p,
}

/// Explicit states keyed as `CHAIN:RESID[ICODE]`, for example `A:42=HID`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ProtonationOverrides {
    #[serde(default)]
    pub residues: BTreeMap<String, String>,
}

/// Reproducible system-building options.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(default)]
pub struct BuildOptions {
    pub profile: ForceFieldProfile,
    /// Minimum solute-to-box-face distance in Å.
    pub padding_angstrom: f64,
    /// NaCl concentration in mol/L of the bath the box is in equilibrium with
    /// (ion counts by the SLTCAP rule; for a neutral solute, the concentration
    /// in its water).
    pub salt_molar: f64,
    /// CaCl2 concentration in mol/L, added to the NaCl: each Ca2+ (Li/Merz 12-6)
    /// comes with two Cl-, counted from the water like the NaCl.
    pub cacl2_molar: f64,
    /// MgCl2 concentration in mol/L, as `cacl2_molar`.
    pub mgcl2_molar: f64,
    /// Electronic continuum correction (prosECCo75 conventions): the bath ions
    /// get the prosECCo75 scaled-charge ions (Na_s, Cl_s, Ca_s, Mg_ss) and every
    /// charged carbohydrate group carries 0.75 of its charge. Proteins are not
    /// scaled.
    pub ecc: bool,
    /// Add a periodic TIP3P water box.
    pub add_water: bool,
    /// Add ions: neutral overall, at the requested salt concentration.
    pub add_ions: bool,
    pub seed: u64,
    pub model: u32,
    pub altloc: Option<char>,
    pub protonation: ProtonationOverrides,
    pub overwrite: bool,
    /// Repair the structure first with the structure fixer (missing atoms,
    /// internal gaps, modified residues, hydrogens), removing crystal water.
    pub repair: bool,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            profile: ForceFieldProfile::default(),
            padding_angstrom: 12.0,
            salt_molar: 0.15,
            cacl2_molar: 0.0,
            mgcl2_molar: 0.0,
            ecc: false,
            add_water: true,
            add_ions: true,
            seed: 0,
            model: 1,
            altloc: None,
            protonation: ProtonationOverrides::default(),
            overwrite: false,
            repair: false,
        }
    }
}

impl BuildOptions {
    /// Load options from TOML or JSON, selected by the `.toml`/`.json` suffix.
    pub fn from_config_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        match path.extension().and_then(|value| value.to_str()) {
            Some(extension) if extension.eq_ignore_ascii_case("json") => Self::from_json_file(path),
            Some(extension) if extension.eq_ignore_ascii_case("toml") => Self::from_toml_file(path),
            _ => Err(BuildError::InvalidOption(format!(
                "{}: config file must use .toml or .json",
                path.display()
            ))),
        }
    }

    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents =
            std::fs::read_to_string(path).map_err(crate::error::read_error(path.to_path_buf()))?;
        toml::from_str(&contents)
            .map_err(|error| BuildError::InvalidOption(format!("{}: {error}", path.display())))
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents =
            std::fs::read_to_string(path).map_err(crate::error::read_error(path.to_path_buf()))?;
        serde_json::from_str(&contents)
            .map_err(|error| BuildError::InvalidOption(format!("{}: {error}", path.display())))
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if !self.padding_angstrom.is_finite() || self.padding_angstrom <= 0.0 {
            return Err(BuildError::InvalidOption(
                "padding must be a positive finite value".into(),
            ));
        }
        for (name, molar) in [("CaCl2", self.cacl2_molar), ("MgCl2", self.mgcl2_molar)] {
            if !molar.is_finite() || molar < 0.0 {
                return Err(BuildError::InvalidOption(format!(
                    "{name} concentration must be non-negative and finite"
                )));
            }
        }
        if !self.salt_molar.is_finite() || self.salt_molar < 0.0 {
            return Err(BuildError::InvalidOption(
                "salt concentration must be non-negative and finite".into(),
            ));
        }
        if self.add_ions && !self.add_water {
            return Err(BuildError::InvalidOption(
                "ions require water; disable both add_water and add_ions for a dry system".into(),
            ));
        }
        if self.model == 0 {
            return Err(BuildError::InvalidOption(
                "PDB model numbers start at 1".into(),
            ));
        }
        Ok(())
    }
}
