use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::{BuildError, BuildReport, Result, SystemMetadata};

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
pub struct Atom {
    pub(crate) name: String,
    pub(crate) atom_type: String,
    pub(crate) element: u8,
    pub(crate) residue: usize,
    pub(crate) charge: f64,
    pub(crate) mass: f64,
    pub(crate) radius: f64,
    pub(crate) epsilon: f64,
    pub(crate) position: Vec3,
}

#[derive(Debug, Clone)]
pub struct Residue {
    pub(crate) name: String,
    pub(crate) number: i32,
    pub(crate) insertion_code: Option<char>,
    pub(crate) chain: String,
    pub(crate) first_atom: usize,
    pub(crate) atom_count: usize,
    pub(crate) component: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Bond {
    pub(crate) atoms: [usize; 2],
    pub(crate) force: f64,
    pub(crate) length: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Angle {
    pub(crate) atoms: [usize; 3],
    pub(crate) force: f64,
    pub(crate) radians: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Dihedral {
    pub(crate) atoms: [usize; 4],
    pub(crate) force: f64,
    pub(crate) periodicity: i32,
    pub(crate) phase: f64,
    pub(crate) improper: bool,
    pub(crate) scee: f64,
    pub(crate) scnb: f64,
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
pub struct ParameterizedSystem {
    pub(crate) system: System,
    pub report: BuildReport,
    pub(crate) metadata: SystemMetadata,
}

/// Backwards-compatible name for [`ParameterizedSystem`].
pub type PreparedSystem = ParameterizedSystem;

impl ParameterizedSystem {
    pub fn atom_count(&self) -> usize {
        self.system.atoms.len()
    }

    pub fn report(&self) -> &BuildReport {
        &self.report
    }

    pub fn atoms(&self) -> &[Atom] {
        &self.system.atoms
    }

    pub fn residues(&self) -> &[Residue] {
        &self.system.residues
    }

    pub fn bonds(&self) -> &[Bond] {
        &self.system.bonds
    }

    pub fn angles(&self) -> &[Angle] {
        &self.system.angles
    }

    pub fn dihedrals(&self) -> &[Dihedral] {
        &self.system.dihedrals
    }

    pub fn exclusions(&self) -> &[BTreeSet<usize>] {
        &self.system.exclusions
    }

    pub fn box_angstrom(&self) -> [f64; 3] {
        self.system.box_angstrom
    }

    pub fn metadata(&self) -> &SystemMetadata {
        &self.metadata
    }

    /// Return the current Cartesian coordinates in Å.
    pub fn coordinates(&self) -> Vec<Vec3> {
        self.system.atoms.iter().map(|atom| atom.position).collect()
    }

    /// Replace every Cartesian coordinate while preserving topology.
    pub fn set_coordinates(&mut self, coordinates: &[Vec3]) -> Result<()> {
        if coordinates.len() != self.system.atoms.len() {
            return Err(BuildError::InvalidOption(format!(
                "expected {} coordinates, received {}",
                self.system.atoms.len(),
                coordinates.len()
            )));
        }
        if coordinates
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite())
        {
            return Err(BuildError::InvalidOption(
                "coordinates must contain only finite values".into(),
            ));
        }
        for (atom, position) in self.system.atoms.iter_mut().zip(coordinates) {
            atom.position = *position;
        }
        Ok(())
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

impl Atom {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn atom_type(&self) -> &str {
        &self.atom_type
    }

    pub fn element(&self) -> u8 {
        self.element
    }

    pub fn residue_index(&self) -> usize {
        self.residue
    }

    pub fn charge(&self) -> f64 {
        self.charge
    }

    pub fn mass(&self) -> f64 {
        self.mass
    }

    pub fn lennard_jones_radius(&self) -> f64 {
        self.radius
    }

    pub fn lennard_jones_epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn gb_radius(&self) -> f64 {
        match self.element {
            1 if self.atom_type == "H" => 1.3,
            1 => 1.2,
            6 => 1.7,
            7 => 1.55,
            8 => 1.5,
            15 => 1.85,
            16 => 1.8,
            _ => 1.5,
        }
    }

    pub fn gb_screen(&self) -> f64 {
        match self.element {
            1 => 0.85,
            6 => 0.72,
            7 => 0.79,
            8 => 0.85,
            15 => 0.86,
            16 => 0.96,
            _ => 0.8,
        }
    }
}

impl Residue {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn number(&self) -> i32 {
        self.number
    }

    pub fn insertion_code(&self) -> Option<char> {
        self.insertion_code
    }

    pub fn chain(&self) -> &str {
        &self.chain
    }

    pub fn atom_range(&self) -> std::ops::Range<usize> {
        self.first_atom..self.first_atom + self.atom_count
    }

    pub fn component(&self) -> usize {
        self.component
    }
}

impl Bond {
    pub fn atoms(&self) -> [usize; 2] {
        self.atoms
    }

    pub fn force(&self) -> f64 {
        self.force
    }

    pub fn length(&self) -> f64 {
        self.length
    }
}

impl Angle {
    pub fn atoms(&self) -> [usize; 3] {
        self.atoms
    }

    pub fn force(&self) -> f64 {
        self.force
    }

    pub fn radians(&self) -> f64 {
        self.radians
    }
}

impl Dihedral {
    pub fn atoms(&self) -> [usize; 4] {
        self.atoms
    }

    pub fn force(&self) -> f64 {
        self.force
    }

    pub fn periodicity(&self) -> i32 {
        self.periodicity
    }

    pub fn phase(&self) -> f64 {
        self.phase
    }

    pub fn is_improper(&self) -> bool {
        self.improper
    }

    pub fn electrostatic_14_scale(&self) -> f64 {
        self.scee
    }

    pub fn lennard_jones_14_scale(&self) -> f64 {
        self.scnb
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
