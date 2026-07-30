use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::model::Vec3;
use crate::pdb::{DeclaredLink, ParsedPdb, PdbAtom, PdbResidue, ResidueKey};
use crate::report::ResidueRef;
use crate::{BuildError, BuildOptions, Result};

/// Stable identifier for an atom inside a [`Structure`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct AtomId(pub u32);

/// Stable identifier for a residue inside a [`Structure`].
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct ResidueId {
    pub chain: String,
    pub number: i32,
    pub insertion_code: Option<char>,
}

impl std::fmt::Display for ResidueId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}:{}", self.chain, self.number)?;
        if let Some(insertion_code) = self.insertion_code {
            write!(formatter, "{insertion_code}")?;
        }
        Ok(())
    }
}

/// Public, format-independent atom record.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StructureAtom {
    pub id: AtomId,
    pub name: String,
    pub residue: ResidueId,
    pub residue_name: String,
    pub element: String,
    pub position: Vec3,
}

/// Public, format-independent residue record.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StructureResidue {
    pub id: ResidueId,
    pub name: String,
    pub atoms: Vec<AtomId>,
}

/// A glycan tree annotation retained across construction and parameterization.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GlycanTree {
    pub chain: String,
    pub residue_ids: Vec<ResidueId>,
    pub attachment_site: Option<ResidueId>,
}

/// A protein-to-glycan covalent attachment.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GlycosylationSite {
    pub protein_residue: ResidueId,
    pub protein_atom: String,
    pub glycan_residue: ResidueId,
    pub glycan_atom: String,
}

/// Free-form residue annotation for downstream applications.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResidueAnnotation {
    pub residue: ResidueId,
    pub label: String,
}

/// Application metadata that is not representable in generic topology formats.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SystemMetadata {
    pub protein_chains: Vec<String>,
    pub glycan_trees: Vec<GlycanTree>,
    pub glycosylation_sites: Vec<GlycosylationSite>,
    pub residue_annotations: Vec<ResidueAnnotation>,
}

/// An unparameterized molecular structure with stable atom identifiers.
///
/// Fields that could invalidate indices or connectivity remain private. Use the
/// accessors and mutation methods to construct or modify a structure.
#[derive(Debug, Clone)]
pub struct Structure {
    pub(crate) parsed: ParsedPdb,
    metadata: SystemMetadata,
}

/// Identifiers assigned while appending one structure to another.
#[derive(Debug, Clone, Default)]
pub struct AppendMap {
    atom_ids: BTreeMap<AtomId, AtomId>,
    residue_ids: BTreeMap<ResidueId, ResidueId>,
}

impl AppendMap {
    pub fn atom(&self, original: AtomId) -> Option<AtomId> {
        self.atom_ids.get(&original).copied()
    }

    pub fn residue(&self, original: &ResidueId) -> Option<&ResidueId> {
        self.residue_ids.get(original)
    }
}

impl Structure {
    pub(crate) fn from_parsed(parsed: ParsedPdb) -> Self {
        let protein_chains = parsed.chains.clone();
        Self {
            parsed,
            metadata: SystemMetadata {
                protein_chains,
                ..SystemMetadata::default()
            },
        }
    }

    pub fn atoms(&self) -> Vec<StructureAtom> {
        self.parsed
            .residues
            .iter()
            .flat_map(|residue| {
                let residue_id = residue_id(&residue.reference);
                residue.atoms.iter().map(move |atom| StructureAtom {
                    id: AtomId(atom.serial),
                    name: atom.name.clone(),
                    residue: residue_id.clone(),
                    residue_name: residue.reference.name.clone(),
                    element: atom.element.clone(),
                    position: atom.position,
                })
            })
            .collect()
    }

    pub fn residues(&self) -> Vec<StructureResidue> {
        self.parsed
            .residues
            .iter()
            .map(|residue| StructureResidue {
                id: residue_id(&residue.reference),
                name: residue.reference.name.clone(),
                atoms: residue
                    .atoms
                    .iter()
                    .map(|atom| AtomId(atom.serial))
                    .collect(),
            })
            .collect()
    }

    /// Explicit bonds declared by the source or added by a structure builder.
    pub fn bonds(&self) -> Vec<(AtomId, AtomId)> {
        self.parsed
            .conect
            .iter()
            .map(|(first, second)| (AtomId(*first), AtomId(*second)))
            .collect()
    }

    pub fn metadata(&self) -> &SystemMetadata {
        &self.metadata
    }

    pub fn metadata_mut(&mut self) -> &mut SystemMetadata {
        &mut self.metadata
    }

    pub fn atom(&self, id: AtomId) -> Option<StructureAtom> {
        self.atoms().into_iter().find(|atom| atom.id == id)
    }

    pub fn find_atom(&self, residue_id: &ResidueId, atom_name: &str) -> Option<AtomId> {
        let residue = self.find_residue(residue_id)?;
        residue
            .atoms
            .iter()
            .find(|atom| atom.name == atom_name)
            .map(|atom| AtomId(atom.serial))
    }

    pub fn set_atom_position(&mut self, id: AtomId, position: Vec3) -> Result<()> {
        let atom = self
            .parsed
            .residues
            .iter_mut()
            .flat_map(|residue| &mut residue.atoms)
            .find(|atom| atom.serial == id.0)
            .ok_or_else(|| BuildError::InvalidPdb(format!("atom {} does not exist", id.0)))?;
        atom.position = position;
        Ok(())
    }

    pub fn rename_residue(&mut self, id: &ResidueId, name: impl Into<String>) -> Result<()> {
        let residue = self
            .find_residue_mut(id)
            .ok_or_else(|| BuildError::InvalidPdb(format!("residue {id} does not exist")))?;
        let name = name.into();
        residue.reference.name.clone_from(&name);
        for atom in &mut residue.atoms {
            atom.residue_name.clone_from(&name);
        }
        Ok(())
    }

    /// Remove named cap or solvent residues while preserving valid connectivity.
    pub fn remove_residues_named(&mut self, names: &[&str]) {
        let removed_residues = self
            .parsed
            .residues
            .iter()
            .filter(|residue| names.contains(&residue.reference.name.as_str()))
            .map(|residue| residue_id(&residue.reference))
            .collect::<BTreeSet<_>>();
        let removed_serials = self
            .parsed
            .residues
            .iter()
            .filter(|residue| names.contains(&residue.reference.name.as_str()))
            .flat_map(|residue| residue.atoms.iter().map(|atom| atom.serial))
            .collect::<BTreeSet<_>>();
        self.parsed
            .residues
            .retain(|residue| !names.contains(&residue.reference.name.as_str()));
        self.parsed.conect.retain(|(first, second)| {
            !removed_serials.contains(first) && !removed_serials.contains(second)
        });
        self.parsed.links.retain(|link| {
            !removed_residues.contains(&ResidueId {
                chain: link.first.chain.clone(),
                number: link.first.number,
                insertion_code: link.first.insertion_code,
            }) && !removed_residues.contains(&ResidueId {
                chain: link.second.chain.clone(),
                number: link.second.number,
                insertion_code: link.second.insertion_code,
            })
        });
        self.parsed.ssbonds.retain(|(first, second)| {
            !removed_residues.contains(&ResidueId {
                chain: first.chain.clone(),
                number: first.number,
                insertion_code: first.insertion_code,
            }) && !removed_residues.contains(&ResidueId {
                chain: second.chain.clone(),
                number: second.number,
                insertion_code: second.insertion_code,
            })
        });
        self.refresh_chains();
    }

    /// Append all residues and explicit bonds from `other`, assigning a new chain.
    ///
    /// Residue numbers are allocated after existing residues in the destination
    /// chain and atom serials are always regenerated, so identifiers remain
    /// unique and stable.
    pub fn append(&mut self, other: &Structure, chain: impl Into<String>) -> Result<AppendMap> {
        let chain = chain.into();
        if chain.chars().count() != 1 {
            return Err(BuildError::InvalidPdb(
                "PDB-compatible chain identifiers must contain one character".into(),
            ));
        }
        let mut next_serial = self
            .parsed
            .residues
            .iter()
            .flat_map(|residue| &residue.atoms)
            .map(|atom| atom.serial)
            .max()
            .unwrap_or(0)
            + 1;
        let next_residue = self
            .parsed
            .residues
            .iter()
            .filter(|residue| residue.reference.chain == chain)
            .map(|residue| residue.reference.number)
            .max()
            .unwrap_or(0)
            + 1;
        let mut result = AppendMap::default();
        let mut serial_map = BTreeMap::new();
        for (destination_number, source) in (next_residue..).zip(other.parsed.residues.iter()) {
            let source_id = residue_id(&source.reference);
            let destination_id = ResidueId {
                chain: chain.clone(),
                number: destination_number,
                insertion_code: None,
            };
            result.residue_ids.insert(source_id, destination_id.clone());
            let mut atoms = Vec::with_capacity(source.atoms.len());
            for source_atom in &source.atoms {
                let destination_serial = next_serial;
                next_serial += 1;
                serial_map.insert(source_atom.serial, destination_serial);
                result
                    .atom_ids
                    .insert(AtomId(source_atom.serial), AtomId(destination_serial));
                atoms.push(PdbAtom {
                    serial: destination_serial,
                    name: source_atom.name.clone(),
                    residue_name: source.reference.name.clone(),
                    chain: chain.clone(),
                    residue_number: destination_id.number,
                    insertion_code: None,
                    element: source_atom.element.clone(),
                    position: source_atom.position,
                });
            }
            self.parsed.residues.push(PdbResidue {
                reference: ResidueRef {
                    chain: chain.clone(),
                    name: source.reference.name.clone(),
                    number: destination_id.number,
                    insertion_code: None,
                },
                atoms,
            });
        }
        for &(first, second) in &other.parsed.conect {
            if let (Some(&first), Some(&second)) = (serial_map.get(&first), serial_map.get(&second))
            {
                self.parsed.conect.insert(ordered(first, second));
            }
        }
        for link in &other.parsed.links {
            let first = result
                .residue(&ResidueId {
                    chain: link.first.chain.clone(),
                    number: link.first.number,
                    insertion_code: link.first.insertion_code,
                })
                .ok_or_else(|| {
                    BuildError::InvalidPdb("LINK references a missing residue".into())
                })?;
            let second = result
                .residue(&ResidueId {
                    chain: link.second.chain.clone(),
                    number: link.second.number,
                    insertion_code: link.second.insertion_code,
                })
                .ok_or_else(|| {
                    BuildError::InvalidPdb("LINK references a missing residue".into())
                })?;
            self.parsed.links.push(DeclaredLink {
                first: first.into(),
                first_atom: link.first_atom.clone(),
                second: second.into(),
                second_atom: link.second_atom.clone(),
            });
        }
        for (first, second) in &other.parsed.ssbonds {
            let first = result
                .residue(&ResidueId {
                    chain: first.chain.clone(),
                    number: first.number,
                    insertion_code: first.insertion_code,
                })
                .ok_or_else(|| {
                    BuildError::InvalidPdb("SSBOND references a missing residue".into())
                })?;
            let second = result
                .residue(&ResidueId {
                    chain: second.chain.clone(),
                    number: second.number,
                    insertion_code: second.insertion_code,
                })
                .ok_or_else(|| {
                    BuildError::InvalidPdb("SSBOND references a missing residue".into())
                })?;
            self.parsed.ssbonds.push((first.into(), second.into()));
        }
        self.refresh_chains();
        Ok(result)
    }

    pub fn add_bond(&mut self, first: AtomId, second: AtomId) -> Result<()> {
        if first == second || self.atom(first).is_none() || self.atom(second).is_none() {
            return Err(BuildError::InvalidPdb(
                "a bond must connect two existing, distinct atoms".into(),
            ));
        }
        self.parsed.conect.insert(ordered(first.0, second.0));
        Ok(())
    }

    pub fn add_glycosylation_site(&mut self, site: GlycosylationSite) {
        self.metadata.glycosylation_sites.push(site);
    }

    pub fn to_pdb_string(&self) -> String {
        write_pdb_string(self)
    }

    fn find_residue(&self, id: &ResidueId) -> Option<&PdbResidue> {
        self.parsed.residues.iter().find(|residue| {
            residue.reference.chain == id.chain
                && residue.reference.number == id.number
                && residue.reference.insertion_code == id.insertion_code
        })
    }

    fn find_residue_mut(&mut self, id: &ResidueId) -> Option<&mut PdbResidue> {
        self.parsed.residues.iter_mut().find(|residue| {
            residue.reference.chain == id.chain
                && residue.reference.number == id.number
                && residue.reference.insertion_code == id.insertion_code
        })
    }

    fn refresh_chains(&mut self) {
        self.parsed.chains = self
            .parsed
            .residues
            .iter()
            .map(|residue| residue.reference.chain.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
    }
}

/// Parse a PDB file into an editable in-memory structure.
pub fn read_pdb(path: impl AsRef<Path>, options: &BuildOptions) -> Result<Structure> {
    let path = path.as_ref();
    let contents =
        std::fs::read_to_string(path).map_err(crate::error::read_error(path.to_path_buf()))?;
    read_pdb_str(&contents, options)
}

/// Parse PDB text into an editable in-memory structure.
pub fn read_pdb_str(contents: &str, options: &BuildOptions) -> Result<Structure> {
    crate::pdb::parse(contents, options).map(Structure::from_parsed)
}

/// Write an unparameterized structure as PDB.
pub fn write_pdb(structure: &Structure, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, write_pdb_string(structure))
        .map_err(crate::error::write_error(path.to_path_buf()))
}

/// Serialize an unparameterized structure as PDB text.
pub fn write_pdb_string(structure: &Structure) -> String {
    let mut output = String::new();
    for residue in &structure.parsed.residues {
        let record = if crate::pdb::PROTEIN_RESIDUES.contains(&residue.reference.name.as_str()) {
            "ATOM  "
        } else {
            "HETATM"
        };
        for atom in &residue.atoms {
            let insertion = atom.insertion_code.unwrap_or(' ');
            let element = format!("{:>2}", atom.element);
            output.push_str(&format!(
                "{record}{:>5} {:<4} {:>3} {:1}{:>4}{insertion}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {element}\n",
                atom.serial,
                atom.name,
                residue.reference.name,
                residue.reference.chain,
                residue.reference.number,
                atom.position.x,
                atom.position.y,
                atom.position.z,
                1.0,
                0.0,
            ));
        }
        output.push_str("TER\n");
    }
    for (first, second) in &structure.parsed.conect {
        output.push_str(&format!("CONECT{first:>5}{second:>5}\n"));
    }
    for link in &structure.parsed.links {
        output.push_str(&format_link(link));
    }
    output.push_str("END\n");
    output
}

fn residue_id(reference: &ResidueRef) -> ResidueId {
    ResidueId {
        chain: reference.chain.clone(),
        number: reference.number,
        insertion_code: reference.insertion_code,
    }
}

fn ordered(first: u32, second: u32) -> (u32, u32) {
    if first < second {
        (first, second)
    } else {
        (second, first)
    }
}

fn format_link(link: &DeclaredLink) -> String {
    format!(
        "LINK        {:<4} {:>3} {:1}{:>4}                {:<4} {:>3} {:1}{:>4}\n",
        link.first_atom,
        "",
        link.first.chain,
        link.first.number,
        link.second_atom,
        "",
        link.second.chain,
        link.second.number,
    )
}

impl From<&ResidueId> for ResidueKey {
    fn from(value: &ResidueId) -> Self {
        Self {
            chain: value.chain.clone(),
            number: value.number,
            insertion_code: value.insertion_code,
        }
    }
}
