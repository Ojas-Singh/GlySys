use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::{BuildError, BuildReport, Result};

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub(crate) fn distance2(self, other: Self) -> f64 {
        (self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Atom {
    pub name: String,
    pub atom_type: String,
    pub element: u8,
    pub residue: usize,
    pub charge: f64,
    pub mass: f64,
    pub radius: f64,
    pub epsilon: f64,
    pub position: Vec3,
}

#[derive(Debug, Clone)]
pub(crate) struct Residue {
    pub name: String,
    pub number: i32,
    pub insertion_code: Option<char>,
    pub chain: String,
    pub first_atom: usize,
    pub atom_count: usize,
    pub component: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Bond {
    pub atoms: [usize; 2],
    pub force: f64,
    pub length: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Angle {
    pub atoms: [usize; 3],
    pub force: f64,
    pub radians: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Dihedral {
    pub atoms: [usize; 4],
    pub force: f64,
    pub periodicity: i32,
    pub phase: f64,
    pub improper: bool,
    pub scee: f64,
    pub scnb: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct System {
    pub atoms: Vec<Atom>,
    pub residues: Vec<Residue>,
    pub bonds: Vec<Bond>,
    pub angles: Vec<Angle>,
    pub dihedrals: Vec<Dihedral>,
    pub exclusions: Vec<BTreeSet<usize>>,
    pub box_angstrom: [f64; 3],
    pub component_count: usize,
    pub solute_atom_count: usize,
    pub water_residue_count: usize,
    pub sodium_count: usize,
    pub chloride_count: usize,
}

impl System {
    pub(crate) fn charge(&self) -> f64 {
        self.atoms.iter().map(|atom| atom.charge).sum()
    }
}

/// A fully parameterized and solvated system ready to be written.
#[derive(Debug, Clone)]
pub struct PreparedSystem {
    pub(crate) system: System,
    pub report: BuildReport,
}

impl PreparedSystem {
    pub fn atom_count(&self) -> usize {
        self.system.atoms.len()
    }

    pub fn report(&self) -> &BuildReport {
        &self.report
    }

    /// Write the Amber, GROMACS, and manifest bundle atomically per file.
    pub fn write_bundle(&self, directory: impl AsRef<Path>) -> Result<()> {
        let directory = directory.as_ref();
        let targets = [
            "system.prmtop",
            "system.inpcrd",
            "system.top",
            "system.gro",
            "manifest.json",
        ];
        if directory.exists()
            && !self.report.options.overwrite
            && targets.iter().any(|name| directory.join(name).exists())
        {
            return Err(BuildError::OutputExists(directory.to_path_buf()));
        }
        std::fs::create_dir_all(directory)
            .map_err(crate::error::write_error(directory.to_path_buf()))?;

        let mut outputs = vec![
            (
                "system.prmtop",
                crate::writers::amber::write_prmtop(&self.system)?,
            ),
            (
                "system.inpcrd",
                crate::writers::amber::write_inpcrd(&self.system),
            ),
            (
                "system.top",
                crate::writers::gromacs::write_topology(&self.system),
            ),
            (
                "system.gro",
                crate::writers::gromacs::write_gro(&self.system),
            ),
        ];
        let mut manifest = self.report.clone();
        manifest.output_sha256 = outputs
            .iter()
            .map(|(name, contents)| {
                (
                    (*name).to_string(),
                    format!("{:x}", Sha256::digest(contents.as_bytes())),
                )
            })
            .collect();
        outputs.push((
            "manifest.json",
            serde_json::to_string_pretty(&manifest)
                .map_err(|error| BuildError::Serialization(error.to_string()))?
                + "\n",
        ));
        for (name, contents) in outputs {
            atomic_write(directory.join(name), contents.as_bytes())?;
        }
        Ok(())
    }
}

fn atomic_write(path: PathBuf, contents: &[u8]) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("")
    ));
    std::fs::write(&temporary, contents).map_err(crate::error::write_error(temporary.clone()))?;
    std::fs::rename(&temporary, &path).map_err(crate::error::write_error(path))
}
