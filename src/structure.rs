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
        let glycosylation_sites = parsed
            .links
            .iter()
            .filter_map(|link| {
                let first = parsed.residues.iter().find(|residue| {
                    residue.reference.chain == link.first.chain
                        && residue.reference.number == link.first.number
                        && residue.reference.insertion_code == link.first.insertion_code
                })?;
                let second = parsed.residues.iter().find(|residue| {
                    residue.reference.chain == link.second.chain
                        && residue.reference.number == link.second.number
                        && residue.reference.insertion_code == link.second.insertion_code
                })?;
                let first_is_protein =
                    crate::pdb::PROTEIN_RESIDUES.contains(&first.reference.name.as_str());
                let second_is_protein =
                    crate::pdb::PROTEIN_RESIDUES.contains(&second.reference.name.as_str());
                match (first_is_protein, second_is_protein) {
                    (true, false) => Some(GlycosylationSite {
                        protein_residue: residue_id(&first.reference),
                        protein_atom: link.first_atom.clone(),
                        glycan_residue: residue_id(&second.reference),
                        glycan_atom: link.second_atom.clone(),
                    }),
                    (false, true) => Some(GlycosylationSite {
                        protein_residue: residue_id(&second.reference),
                        protein_atom: link.second_atom.clone(),
                        glycan_residue: residue_id(&first.reference),
                        glycan_atom: link.first_atom.clone(),
                    }),
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        let glycan_trees = glycosylation_sites
            .iter()
            .map(|site| GlycanTree {
                chain: site.glycan_residue.chain.clone(),
                residue_ids: parsed
                    .residues
                    .iter()
                    .filter(|residue| {
                        residue.reference.chain == site.glycan_residue.chain
                            && !crate::pdb::PROTEIN_RESIDUES
                                .contains(&residue.reference.name.as_str())
                    })
                    .map(|residue| residue_id(&residue.reference))
                    .collect(),
                attachment_site: Some(site.protein_residue.clone()),
            })
            .collect();
        Self {
            parsed,
            metadata: SystemMetadata {
                protein_chains,
                glycan_trees,
                glycosylation_sites,
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
        let declared = DeclaredLink {
            first: (&site.protein_residue).into(),
            first_atom: site.protein_atom.clone(),
            second: (&site.glycan_residue).into(),
            second_atom: site.glycan_atom.clone(),
        };
        if !self.parsed.links.iter().any(|link| {
            link.first == declared.first
                && link.first_atom == declared.first_atom
                && link.second == declared.second
                && link.second_atom == declared.second_atom
        }) {
            self.parsed.links.push(declared);
        }
        self.metadata.glycosylation_sites.push(site);
    }

    /// Copy coordinates for atoms present in both this structure and a
    /// parameterized system. Generated force-field atoms are ignored.
    pub fn update_from_parameterized(&mut self, system: &crate::ParameterizedSystem) -> Result<()> {
        for residue in system.residues() {
            let id = ResidueId {
                chain: residue.chain().to_string(),
                number: residue.number(),
                insertion_code: residue.insertion_code(),
            };
            for atom_index in residue.atom_range() {
                let atom = &system.atoms()[atom_index];
                if let Some(id) = self.find_atom(&id, atom.name()) {
                    self.set_atom_position(id, atom.position())?;
                }
            }
        }
        Ok(())
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
    for link in &structure.parsed.links {
        output.push_str(&format_link(structure, link));
    }
    for (index, residue) in structure.parsed.residues.iter().enumerate() {
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
        let chain_ends = structure
            .parsed
            .residues
            .get(index + 1)
            .is_none_or(|next| next.reference.chain != residue.reference.chain);
        if chain_ends {
            output.push_str("TER\n");
        }
    }
    for (first, second) in &structure.parsed.conect {
        output.push_str(&format!("CONECT{first:>5}{second:>5}\n"));
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

fn format_link(structure: &Structure, link: &DeclaredLink) -> String {
    let first = structure.parsed.residues.iter().find(|residue| {
        residue.reference.chain == link.first.chain
            && residue.reference.number == link.first.number
            && residue.reference.insertion_code == link.first.insertion_code
    });
    let second = structure.parsed.residues.iter().find(|residue| {
        residue.reference.chain == link.second.chain
            && residue.reference.number == link.second.number
            && residue.reference.insertion_code == link.second.insertion_code
    });
    let first_name = first.map_or("", |residue| residue.reference.name.as_str());
    let second_name = second.map_or("", |residue| residue.reference.name.as_str());
    let distance = first
        .and_then(|residue| {
            residue
                .atoms
                .iter()
                .find(|atom| atom.name == link.first_atom)
        })
        .zip(second.and_then(|residue| {
            residue
                .atoms
                .iter()
                .find(|atom| atom.name == link.second_atom)
        }))
        .map(|(first, second)| {
            let dx = first.position.x - second.position.x;
            let dy = first.position.y - second.position.y;
            let dz = first.position.z - second.position.z;
            (dx * dx + dy * dy + dz * dz).sqrt()
        });

    let mut record = vec![b' '; 80];
    put_field(&mut record, 0, "LINK");
    put_field(&mut record, 12, &format!("{:>4}", link.first_atom));
    put_field(&mut record, 17, &format!("{first_name:>3}"));
    put_field(&mut record, 21, &link.first.chain);
    put_field(&mut record, 22, &format!("{:>4}", link.first.number));
    if let Some(code) = link.first.insertion_code {
        put_field(&mut record, 26, &code.to_string());
    }
    put_field(&mut record, 42, &format!("{:>4}", link.second_atom));
    put_field(&mut record, 47, &format!("{second_name:>3}"));
    put_field(&mut record, 51, &link.second.chain);
    put_field(&mut record, 52, &format!("{:>4}", link.second.number));
    if let Some(code) = link.second.insertion_code {
        put_field(&mut record, 56, &code.to_string());
    }
    if let Some(distance) = distance {
        put_field(&mut record, 73, &format!("{distance:>5.2}"));
    }
    format!(
        "{}\n",
        String::from_utf8(record).expect("PDB LINK fields are ASCII")
    )
}

fn put_field(record: &mut [u8], offset: usize, value: &str) {
    for (destination, source) in record[offset..].iter_mut().zip(value.bytes()) {
        *destination = source;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    const GLYCOPROTEIN: &str = "\
ATOM      1  N   ASN A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  ND2 ASN A   1       1.000   0.000   0.000  1.00  0.00           N
ATOM      3  N   ALA A   2       2.000   0.000   0.000  1.00  0.00           N
HETATM    4  C1  NAG B   1       2.450   0.000   0.000  1.00  0.00           C
END
";

    #[test]
    fn pdb_writer_terminates_chains_and_writes_complete_links() {
        let mut structure =
            read_pdb_str(GLYCOPROTEIN, &BuildOptions::default()).expect("parse fixture");
        structure
            .add_bond(AtomId(2), AtomId(4))
            .expect("add attachment bond");
        structure.add_glycosylation_site(GlycosylationSite {
            protein_residue: ResidueId {
                chain: "A".into(),
                number: 1,
                insertion_code: None,
            },
            protein_atom: "ND2".into(),
            glycan_residue: ResidueId {
                chain: "B".into(),
                number: 1,
                insertion_code: None,
            },
            glycan_atom: "C1".into(),
        });

        let output = write_pdb_string(&structure);
        assert_eq!(output.lines().filter(|line| *line == "TER").count(), 2);
        assert!(!output.contains("ND2 ASN A   1\nTER\nATOM"));
        let link = output
            .lines()
            .find(|line| line.starts_with("LINK"))
            .expect("LINK record");
        assert_eq!(&link[12..16], " ND2");
        assert_eq!(&link[17..20], "ASN");
        assert_eq!(&link[42..46], "  C1");
        assert_eq!(&link[47..50], "NAG");
        assert!(output.contains("CONECT    2    4"));

        let reparsed =
            read_pdb_str(&output, &BuildOptions::default()).expect("reparse written PDB");
        assert_eq!(reparsed.metadata().glycosylation_sites.len(), 1);
        assert_eq!(reparsed.metadata().glycan_trees.len(), 1);
        assert_eq!(reparsed.metadata().glycan_trees[0].residue_ids.len(), 1);
    }
}
