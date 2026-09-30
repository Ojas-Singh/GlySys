//! Working representation of a structure while it is being repaired.

use std::collections::{BTreeSet, HashMap};

use super::chemistry::{self, AMINO_ACIDS, DNA, NATIVE_PROTEIN_EXTRAS, RNA};
use super::components::ComponentLibrary;
use super::geometry::{V, distance, v};
use crate::forcefield::TemplateSet;
use crate::pdb::{RawModel, ResidueKey};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Origin {
    /// Coordinates from the input file.
    Input,
    /// A missing heavy atom of a deposited residue.
    Rebuilt,
    /// An atom of a residue absent from the input (SEQRES gap).
    Modelled,
    Hydrogen,
}

#[derive(Debug, Clone)]
pub(crate) struct WAtom {
    pub name: String,
    pub element: String,
    pub position: V,
    pub occupancy: f64,
    pub b_factor: f64,
    pub serial: Option<u32>,
    pub origin: Origin,
}

impl WAtom {
    pub(crate) fn is_hydrogen(&self) -> bool {
        matches!(self.element.as_str(), "H" | "D")
    }

    pub(crate) fn movable(&self) -> bool {
        matches!(self.origin, Origin::Rebuilt | Origin::Modelled)
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ResidueKind {
    Protein,
    Nucleic,
    Glycan,
    Water,
    Ion,
    Ligand,
}

#[derive(Debug, Clone)]
pub(crate) struct WResidue {
    /// Stable identity used by bonds while residues are inserted.
    pub uid: usize,
    pub chain: String,
    pub number: i32,
    pub icode: Option<char>,
    /// Residue name in the input file.
    pub input_name: String,
    /// Standard name (`HIS`, `MET` for a replaced `MSE`, `DA`, `NAG`, ...).
    pub name: String,
    /// Protonation/chemical variant, e.g. `HID`, `CYX`, `ASH`, `NLN`.
    pub variant: Option<String>,
    pub kind: ResidueKind,
    pub atoms: Vec<WAtom>,
    pub ter_after: bool,
    pub prev: Option<usize>,
    pub next: Option<usize>,
    /// Modelled residues will be attached before / after this residue.
    pub open_before: bool,
    pub open_after: bool,
    pub modelled: bool,
    /// Modified residue that could have been replaced by this parent.
    pub parent: Option<String>,
    /// Name of the force-field template used to complete the residue.
    pub template: Option<String>,
}

impl WResidue {
    pub(crate) fn atom(&self, name: &str) -> Option<&WAtom> {
        self.atoms.iter().find(|atom| atom.name == name)
    }

    pub(crate) fn position(&self, name: &str) -> Option<V> {
        self.atom(name).map(|atom| atom.position)
    }

    pub(crate) fn label(&self) -> String {
        format!(
            "{}/{}/{}{}",
            self.chain,
            self.input_name,
            self.number,
            self.icode.map(String::from).unwrap_or_default()
        )
    }

    pub(crate) fn key(&self) -> ResidueKey {
        ResidueKey {
            chain: self.chain.clone(),
            number: self.number,
            insertion_code: self.icode,
        }
    }

    pub(crate) fn is_polymer(&self) -> bool {
        matches!(self.kind, ResidueKind::Protein | ResidueKind::Nucleic)
    }

    pub(crate) fn is_cap(&self) -> bool {
        matches!(self.name.as_str(), "ACE" | "NME" | "NH2")
    }
}

/// Covalent bond between two atoms, referenced by residue uid and atom name.
pub(crate) type AtomRef = (usize, String);

#[derive(Debug, Clone, Default)]
pub(crate) struct Work {
    pub residues: Vec<WResidue>,
    /// Bonds from CONECT/LINK records and disulfides (inter-residue only).
    pub bonds: Vec<(AtomRef, AtomRef)>,
    pub seqres: std::collections::BTreeMap<String, Vec<String>>,
    /// REMARK 465 unobserved residues.
    pub unobserved: Vec<(ResidueKey, String)>,
    pub next_uid: usize,
}

impl Work {
    pub(crate) fn index_of(&self) -> HashMap<usize, usize> {
        self.residues
            .iter()
            .enumerate()
            .map(|(index, residue)| (residue.uid, index))
            .collect()
    }

    /// External bond partners of each (residue, atom) from explicit bonds.
    pub(crate) fn explicit_partners(&self) -> HashMap<(usize, String), Vec<AtomRef>> {
        let mut partners = HashMap::<(usize, String), Vec<AtomRef>>::new();
        for (first, second) in &self.bonds {
            partners
                .entry(first.clone())
                .or_default()
                .push(second.clone());
            partners
                .entry(second.clone())
                .or_default()
                .push(first.clone());
        }
        partners
    }

    pub(crate) fn add_bond(&mut self, first: AtomRef, second: AtomRef) {
        let exists = self
            .bonds
            .iter()
            .any(|(a, b)| (a == &first && b == &second) || (a == &second && b == &first));
        if !exists && first != second {
            self.bonds.push((first, second));
        }
    }
}

pub(crate) struct LoadNotes {
    pub dropped_hydrogens: usize,
    pub aliased_atoms: usize,
}

/// Classify residues and normalize names.
pub(crate) fn load(
    raw: RawModel,
    templates: &TemplateSet,
    components: &ComponentLibrary,
) -> (Work, LoadNotes) {
    let mut serial_owner = HashMap::<u32, AtomRef>::new();
    let mut work = Work {
        seqres: raw.seqres.clone(),
        unobserved: raw.unobserved.clone(),
        ..Work::default()
    };
    let mut notes = LoadNotes {
        dropped_hydrogens: 0,
        aliased_atoms: 0,
    };
    let modres = raw
        .modres
        .iter()
        .map(|(key, name, parent)| ((key.clone(), name.clone()), parent.clone()))
        .collect::<HashMap<_, _>>();
    for residue in raw.residues {
        let (name, variant) = chemistry::residue_alias(&residue.reference.name);
        let heavy = residue
            .atoms
            .iter()
            .filter(|atom| !matches!(atom.element.as_str(), "H" | "D"))
            .count();
        let key = ResidueKey {
            chain: residue.reference.chain.clone(),
            number: residue.reference.number,
            insertion_code: residue.reference.insertion_code,
        };
        let component = components.get(&name);
        // The depositors' MODRES annotation outranks the generic table
        // (e.g. PCA is recorded as pyroglutamate derived from GLN).
        let parent = modres
            .get(&(key.clone(), residue.reference.name.clone()))
            .cloned()
            .or_else(|| chemistry::substitution(&name).map(str::to_string))
            .or_else(|| {
                component
                    .filter(|component| {
                        component.is_peptide_linking() || component.is_nucleotide_linking()
                    })
                    .and_then(|component| component.parent.clone())
            })
            .map(|parent| chemistry::residue_alias(&parent).0)
            .filter(|parent| {
                *parent != name
                    && (AMINO_ACIDS.contains(&parent.as_str())
                        || DNA.contains(&parent.as_str())
                        || RNA.contains(&parent.as_str()))
            });
        let kind = if AMINO_ACIDS.contains(&name.as_str())
            || NATIVE_PROTEIN_EXTRAS.contains(&name.as_str())
            || matches!(name.as_str(), "ACE" | "NME" | "NH2")
        {
            ResidueKind::Protein
        } else if DNA.contains(&name.as_str()) || RNA.contains(&name.as_str()) {
            ResidueKind::Nucleic
        } else if chemistry::is_water_name(&name) {
            ResidueKind::Water
        } else if chemistry::is_ion_residue(&name, heavy) {
            ResidueKind::Ion
        } else if parent.is_none()
            && (templates.glycan(&name).is_some()
                || is_pdb_saccharide(&name)
                || component.is_some_and(|component| component.is_saccharide()))
        {
            ResidueKind::Glycan
        } else {
            ResidueKind::Ligand
        };
        let uid = work.next_uid;
        work.next_uid += 1;
        let nucleic = kind == ResidueKind::Nucleic
            || parent
                .as_deref()
                .is_some_and(|parent| DNA.contains(&parent) || RNA.contains(&parent));
        let rebuild_hydrogens = matches!(
            kind,
            ResidueKind::Protein | ResidueKind::Nucleic | ResidueKind::Water
        );
        let mut atoms = Vec::with_capacity(residue.atoms.len());
        let mut seen = BTreeSet::new();
        for atom in residue.atoms {
            let hydrogen = matches!(atom.element.as_str(), "H" | "D");
            if hydrogen && rebuild_hydrogens {
                notes.dropped_hydrogens += 1;
                continue;
            }
            let atom_name = if matches!(kind, ResidueKind::Protein | ResidueKind::Nucleic)
                || parent.is_some()
            {
                chemistry::atom_alias(&name, &atom.name, nucleic)
            } else {
                atom.name.clone()
            };
            if atom_name != atom.name {
                notes.aliased_atoms += 1;
            }
            if !seen.insert(atom_name.clone()) {
                continue;
            }
            serial_owner.insert(atom.serial, (uid, atom_name.clone()));
            atoms.push(WAtom {
                name: atom_name,
                element: if hydrogen {
                    "H".into()
                } else {
                    atom.element.clone()
                },
                position: v(atom.position),
                occupancy: atom.occupancy,
                b_factor: atom.b_factor,
                serial: Some(atom.serial),
                origin: Origin::Input,
            });
        }
        if atoms.is_empty() {
            continue;
        }
        work.residues.push(WResidue {
            uid,
            chain: residue.reference.chain,
            number: residue.reference.number,
            icode: residue.reference.insertion_code,
            input_name: residue.reference.name,
            name,
            variant: variant.map(str::to_string),
            kind,
            atoms,
            ter_after: residue.ter_after,
            prev: None,
            next: None,
            open_before: false,
            open_after: false,
            modelled: false,
            parent,
            template: None,
        });
    }
    for (first, second) in &raw.conect {
        if let (Some(a), Some(b)) = (serial_owner.get(first), serial_owner.get(second))
            && a.0 != b.0
        {
            work.add_bond(a.clone(), b.clone());
        }
    }
    let by_key = work
        .residues
        .iter()
        .map(|residue| (residue.key(), residue.uid))
        .collect::<HashMap<_, _>>();
    for link in &raw.links {
        if let (Some(&a), Some(&b)) = (by_key.get(&link.first), by_key.get(&link.second))
            && a != b
        {
            work.add_bond((a, link.first_atom.clone()), (b, link.second_atom.clone()));
        }
    }
    for (first, second) in &raw.ssbonds {
        if let (Some(&a), Some(&b)) = (by_key.get(first), by_key.get(second))
            && a != b
        {
            work.add_bond((a, "SG".into()), (b, "SG".into()));
        }
    }
    (work, notes)
}

/// wwPDB saccharide component codes commonly found in glycoproteins.
pub(crate) fn is_pdb_saccharide(name: &str) -> bool {
    matches!(
        name,
        "NAG"
            | "NDG"
            | "BMA"
            | "MAN"
            | "GAL"
            | "GLA"
            | "GLC"
            | "BGC"
            | "FUC"
            | "FUL"
            | "XYS"
            | "XYP"
            | "SIA"
            | "SLB"
            | "NGA"
            | "A2G"
            | "GCU"
            | "BDP"
            | "IDR"
            | "NGC"
            | "KDN"
            | "RAM"
            | "RIB"
            | "ARA"
            | "AHR"
            | "FRU"
            | "GLF"
            | "MAG"
            | "NAN"
            | "G6D"
            | "GMH"
            | "KDO"
            | "LAT"
            | "MAL"
            | "SUC"
            | "TRE"
            | "NEU"
            | "AMN"
            | "GNA"
    )
}

/// Glycosidic and glycan-protein bonds present in the coordinates but not
/// declared by LINK/CONECT (older entries): a glycan carbon within 1.65 Å
/// of an oxygen or nitrogen of another residue.
pub(crate) fn infer_glycan_bonds(work: &mut Work) {
    let mut points = Vec::new();
    for (index, residue) in work.residues.iter().enumerate() {
        for atom in &residue.atoms {
            if matches!(atom.element.as_str(), "O" | "N") {
                points.push((index, atom.name.clone(), atom.position));
            }
        }
    }
    let mut grid = super::geometry::Grid::new(2.0);
    for (i, point) in points.iter().enumerate() {
        grid.insert(i, point.2);
    }
    let mut found = Vec::new();
    for (index, residue) in work.residues.iter().enumerate() {
        if residue.kind != ResidueKind::Glycan {
            continue;
        }
        for atom in residue.atoms.iter().filter(|atom| atom.element == "C") {
            for candidate in grid.near(atom.position, 1.65) {
                let (other, name, position) = &points[candidate];
                if *other != index && distance(atom.position, *position) < 1.65 {
                    found.push((
                        (residue.uid, atom.name.clone()),
                        (work.residues[*other].uid, name.clone()),
                    ));
                }
            }
        }
    }
    for (a, b) in found {
        work.add_bond(a, b);
    }
    // Reducing-end sugars (no O1, C1 unbonded) placed near an Asn/Ser/Thr
    // attachment atom but outside bonding distance, as in some low-resolution
    // models, are treated as attached.
    let partners = work.explicit_partners();
    let mut attachments = Vec::new();
    for residue in &work.residues {
        if residue.kind != ResidueKind::Glycan
            || residue.atom("O1").is_some()
            || partners.contains_key(&(residue.uid, "C1".to_string()))
        {
            continue;
        }
        let Some(c1) = residue.position("C1") else {
            continue;
        };
        let nearest = work
            .residues
            .iter()
            .filter(|other| other.kind == ResidueKind::Protein)
            .filter_map(|other| {
                let atom = match other.name.as_str() {
                    "ASN" => "ND2",
                    "SER" => "OG",
                    "THR" => "OG1",
                    "HYP" => "OD1",
                    _ => return None,
                };
                other
                    .position(atom)
                    .map(|p| (distance(p, c1), other.uid, atom))
            })
            .filter(|(d, _, _)| *d < 3.0)
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((_, uid, atom)) = nearest {
            attachments.push(((uid, atom.to_string()), (residue.uid, "C1".to_string())));
        }
    }
    for (a, b) in attachments {
        work.add_bond(a, b);
    }
}

/// Assign polymer neighbours from peptide / phosphodiester geometry.
pub(crate) fn link_polymers(work: &mut Work) {
    for residue in &mut work.residues {
        residue.prev = None;
        residue.next = None;
    }
    for index in 1..work.residues.len() {
        let (before, after) = work.residues.split_at(index);
        let first = &before[index - 1];
        let second = &after[0];
        if first.chain != second.chain || (first.ter_after && !first.modelled && !second.modelled) {
            continue;
        }
        // Modelled residues are bonded by construction even before relaxation
        // has closed the last peptide bond exactly.
        let modelled = (first.modelled || second.modelled)
            && first.kind == ResidueKind::Protein
            && second.kind == ResidueKind::Protein;
        if modelled || bonded_in_chain(first, second) {
            work.residues[index - 1].next = Some(index);
            work.residues[index].prev = Some(index - 1);
        }
    }
}

/// Drop explicit records (e.g. LINK for a modified residue) that duplicate
/// a polymer backbone bond.
pub(crate) fn prune_polymer_bonds(work: &mut Work) {
    let index_of = work.index_of();
    let residues = &work.residues;
    work.bonds.retain(|((a, atom_a), (b, atom_b))| {
        let (Some(&a), Some(&b)) = (index_of.get(a), index_of.get(b)) else {
            return true;
        };
        let backbone = |first: usize, first_atom: &str, second: usize, second_atom: &str| {
            residues[first].next == Some(second)
                && matches!((first_atom, second_atom), ("C", "N") | ("O3'", "P"))
        };
        !(backbone(a, atom_a, b, atom_b) || backbone(b, atom_b, a, atom_a))
    });
}

pub(crate) fn bonded_in_chain(first: &WResidue, second: &WResidue) -> bool {
    let peptide = first
        .position("C")
        .zip(second.position("N"))
        .is_some_and(|(c, n)| distance(c, n) < 2.0);
    let phosphodiester = first
        .position("O3'")
        .zip(second.position("P"))
        .is_some_and(|(o, p)| distance(o, p) < 2.2);
    peptide || phosphodiester
}

/// Consecutive polymer residues of one chain whose connection is broken.
pub(crate) fn chain_gap(first: &WResidue, second: &WResidue) -> bool {
    first.chain == second.chain
        && first.kind == second.kind
        && first.is_polymer()
        && !bonded_in_chain(first, second)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saccharide_codes_are_recognized() {
        assert!(is_pdb_saccharide("NAG"));
        assert!(!is_pdb_saccharide("HEM"));
    }
}
