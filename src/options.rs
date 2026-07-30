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
    /// Added neutral NaCl concentration in mol/L.
    pub salt_molar: f64,
    /// Add a periodic TIP3P water box.
    pub add_water: bool,
    /// Add neutralizing ions and the requested salt concentration.
    pub add_ions: bool,
    pub seed: u64,
    pub model: u32,
    pub altloc: Option<char>,
    pub protonation: ProtonationOverrides,
    pub overwrite: bool,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            profile: ForceFieldProfile::default(),
            padding_angstrom: 12.0,
            salt_molar: 0.15,
            add_water: true,
            add_ions: true,
            seed: 0,
            model: 1,
            altloc: None,
            protonation: ProtonationOverrides::default(),
            overwrite: false,
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
