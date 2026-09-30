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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Residue {
    pub(crate) name: String,
    pub(crate) number: i32,
    pub(crate) insertion_code: Option<char>,
    pub(crate) chain: String,
    pub(crate) first_atom: usize,
    pub(crate) atom_count: usize,
    pub(crate) component: usize,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Bond {
    pub(crate) atoms: [usize; 2],
    pub(crate) force: f64,
    pub(crate) length: f64,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Angle {
    pub(crate) atoms: [usize; 3],
    pub(crate) force: f64,
    pub(crate) radians: f64,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Dihedral {
    pub(crate) atoms: [usize; 4],
    pub(crate) force: f64,
    pub(crate) periodicity: i32,
    pub(crate) phase: f64,
    pub(crate) improper: bool,
    pub(crate) scee: f64,
    pub(crate) scnb: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

    /// Scaled 1-4 pairs as tleap defines them: end atoms of proper torsions,
    /// once per pair, excluding pairs that are also 1-2 or 1-3 neighbours
    /// (atoms across a five-membered ring).  Returns ((i, j), scee, scnb).
    pub(crate) fn one_four_pairs(&self) -> Vec<([usize; 2], f64, f64)> {
        one_four_pairs(self.atoms.len(), &self.bonds, &self.dihedrals)
    }
}

pub(crate) fn one_four_pairs(
    atom_count: usize,
    bonds: &[Bond],
    dihedrals: &[Dihedral],
) -> Vec<([usize; 2], f64, f64)> {
    let mut neighbors = vec![Vec::new(); atom_count];
    for bond in bonds {
        neighbors[bond.atoms[0]].push(bond.atoms[1]);
        neighbors[bond.atoms[1]].push(bond.atoms[0]);
    }
    let close = |a: usize, b: usize| {
        neighbors[a].contains(&b) || neighbors[a].iter().any(|&m| neighbors[m].contains(&b))
    };
    let mut seen = std::collections::HashSet::new();
    let mut pairs = Vec::new();
    for dihedral in dihedrals.iter().filter(|dihedral| !dihedral.improper) {
        let (a, b) = (dihedral.atoms[0], dihedral.atoms[3]);
        let pair = [a.min(b), a.max(b)];
        if a == b || close(a, b) || !seen.insert(pair) {
            continue;
        }
        pairs.push((pair, dihedral.scee, dihedral.scnb));
    }
    pairs
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

    /// Serialize the current parameterized system as PDB text.
    ///
    /// This is intentionally separate from [`Self::bundle_strings`] so
    /// browser clients that only need a repaired structure do not have to
    /// generate Amber and GROMACS topology files as well.
    pub fn pdb_string(&self) -> String {
        crate::writers::write_pdb_system(&self.system)
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

    /// Scaled 1-4 pairs ((i, j), scee, scnb), consistent with tleap.
    pub fn one_four_pairs(&self) -> Vec<([usize; 2], f64, f64)> {
        self.system.one_four_pairs()
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

    /// The solute alone, without added water, ions or periodic box, as a
    /// preparation with `add_water = false` would produce it.
    pub fn solute(&self) -> Self {
        let n = self.system.solute_atom_count;
        let system = &self.system;
        let within = |atoms: &[usize]| atoms.iter().all(|&atom| atom < n);
        let solute = System {
            atoms: system.atoms[..n].to_vec(),
            residues: system
                .residues
                .iter()
                .filter(|residue| residue.first_atom < n)
                .cloned()
                .collect(),
            bonds: system
                .bonds
                .iter()
                .filter(|bond| within(&bond.atoms))
                .copied()
                .collect(),
            angles: system
                .angles
                .iter()
                .filter(|angle| within(&angle.atoms))
                .copied()
                .collect(),
            dihedrals: system
                .dihedrals
                .iter()
                .filter(|dihedral| within(&dihedral.atoms))
                .cloned()
                .collect(),
            exclusions: system.exclusions[..n].to_vec(),
            box_angstrom: [0.0; 3],
            component_count: system.component_count,
            solute_atom_count: n,
            water_residue_count: 0,
            sodium_count: 0,
            chloride_count: 0,
        };
        let mut report = self.report.clone();
        report.options.add_water = false;
        report.options.add_ions = false;
        report.output_sha256.clear();
        report.total_atoms = n;
        report.residues = solute.residues.len();
        report.waters = 0;
        report.sodium_ions = 0;
        report.chloride_ions = 0;
        report.total_charge = report.solute_charge;
        report.box_angstrom = [0.0; 3];
        Self {
            system: solute,
            report,
            metadata: self.metadata.clone(),
        }
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
            "system.pdb",
            "system.prmtop",
            "system.inpcrd",
            "system.top",
            "system.gro",
            "system.snapshot.json",
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
            ("system.pdb", self.pdb_string()),
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
            ("system.snapshot.json", self.snapshot_json()?),
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

    /// Serialize the full output bundle as in-memory strings (for WASM/embedded
    /// consumers) without touching the filesystem.
    pub fn bundle_strings(&self) -> Result<std::collections::BTreeMap<String, String>> {
        let mut outputs = vec![
            ("system.pdb".to_string(), self.pdb_string()),
            (
                "system.prmtop".to_string(),
                crate::writers::amber::write_prmtop(&self.system)?,
            ),
            (
                "system.inpcrd".to_string(),
                crate::writers::amber::write_inpcrd(&self.system),
            ),
            (
                "system.top".to_string(),
                crate::writers::gromacs::write_topology(&self.system),
            ),
            (
                "system.gro".to_string(),
                crate::writers::gromacs::write_gro(&self.system),
            ),
            ("system.snapshot.json".to_string(), self.snapshot_json()?),
        ];
        let mut manifest = self.report.clone();
        manifest.output_sha256 = outputs
            .iter()
            .map(|(name, contents)| {
                (
                    name.clone(),
                    format!("{:x}", Sha256::digest(contents.as_bytes())),
                )
            })
            .collect();
        outputs.push((
            "manifest.json".to_string(),
            serde_json::to_string_pretty(&manifest)
                .map_err(|error| BuildError::Serialization(error.to_string()))?
                + "\n",
        ));
        Ok(outputs.into_iter().collect())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn bond(a: usize, b: usize) -> Bond {
        Bond {
            atoms: [a, b],
            force: 1.0,
            length: 1.0,
        }
    }

    fn torsion(atoms: [usize; 4]) -> Dihedral {
        Dihedral {
            atoms,
            force: 1.0,
            periodicity: 3,
            phase: 0.0,
            improper: false,
            scee: 1.2,
            scnb: 2.0,
        }
    }

    #[test]
    fn five_ring_torsions_do_not_scale_one_three_pairs() {
        // Cyclopentane ring 0-1-2-3-4 with substituent 5 on atom 0.
        let bonds = [
            bond(0, 1),
            bond(1, 2),
            bond(2, 3),
            bond(3, 4),
            bond(4, 0),
            bond(0, 5),
        ];
        let dihedrals = [
            torsion([0, 1, 2, 3]), // 0 and 3 are 1-3 through 4
            torsion([5, 0, 1, 2]), // a genuine 1-4 pair
            torsion([5, 0, 4, 3]), // 5-3: genuine
            torsion([2, 1, 0, 5]), // duplicate of 5-2
        ];
        let pairs = one_four_pairs(6, &bonds, &dihedrals)
            .into_iter()
            .map(|(pair, _, _)| pair)
            .collect::<Vec<_>>();
        assert_eq!(pairs, vec![[2, 5], [3, 5]]);
    }
}
