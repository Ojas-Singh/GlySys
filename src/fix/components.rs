//! wwPDB Chemical Component Dictionary (CCD) definitions.
//!
//! GlySys never downloads anything itself.  Callers that can reach the RCSB
//! (the CLI, or a browser worker) fetch `https://files.rcsb.org/ligands/download/<ID>.cif`
//! and hand the text to [`ComponentLibrary::add_cif`].  Definitions supply
//! heavy-atom names, hydrogens, bond orders, formal charges, leaving atoms,
//! ideal coordinates and the standard parent of modified residues.

use std::collections::BTreeMap;

use crate::model::Vec3;
use crate::{BuildError, Result};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ComponentAtom {
    pub name: String,
    /// Upper-case element symbol.
    pub element: String,
    pub formal_charge: i32,
    pub aromatic: bool,
    /// Removed when the component is covalently linked (for example OXT).
    pub leaving: bool,
    pub ideal: Option<Vec3>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ComponentBond {
    pub first: String,
    pub second: String,
    /// 1, 2 or 3; aromatic bonds keep their Kekulé order and set `aromatic`.
    pub order: u8,
    pub aromatic: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Component {
    pub id: String,
    pub name: String,
    /// `_chem_comp.type`, e.g. `L-PEPTIDE LINKING` or `NON-POLYMER`.
    pub kind: String,
    /// Standard parent residue, e.g. `MET` for `MSE`.
    pub parent: Option<String>,
    pub atoms: Vec<ComponentAtom>,
    pub bonds: Vec<ComponentBond>,
}

impl Component {
    pub fn atom(&self, name: &str) -> Option<&ComponentAtom> {
        self.atoms.iter().find(|atom| atom.name == name)
    }

    pub fn is_peptide_linking(&self) -> bool {
        self.kind.to_ascii_uppercase().contains("PEPTIDE LINKING")
    }

    pub fn is_nucleotide_linking(&self) -> bool {
        let kind = self.kind.to_ascii_uppercase();
        kind.contains("RNA LINKING") || kind.contains("DNA LINKING")
    }

    pub fn is_saccharide(&self) -> bool {
        self.kind.to_ascii_uppercase().contains("SACCHARIDE")
    }
}

/// Chemical component definitions available to the fixer.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ComponentLibrary {
    components: BTreeMap<String, Component>,
}

impl ComponentLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse every `data_` block of CCD mmCIF text; returns the component IDs.
    pub fn add_cif(&mut self, text: &str) -> Result<Vec<String>> {
        let mut added = Vec::new();
        for block in parse_cif(text)? {
            if let Some(component) = component_from_block(&block) {
                added.push(component.id.clone());
                self.components.insert(component.id.clone(), component);
            }
        }
        Ok(added)
    }

    pub fn insert(&mut self, component: Component) {
        self.components.insert(component.id.clone(), component);
    }

    pub fn get(&self, id: &str) -> Option<&Component> {
        self.components.get(&id.to_ascii_uppercase())
    }

    pub fn len(&self) -> usize {
        self.components.len()
    }

    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.components.keys().map(String::as_str)
    }
}

/// One mmCIF data block: single-valued items and loop tables.
#[derive(Debug, Default)]
struct CifBlock {
    items: BTreeMap<String, String>,
    loops: Vec<(Vec<String>, Vec<Vec<String>>)>,
}

impl CifBlock {
    /// Rows of a category, from a loop or from single-valued items.
    fn category(&self, prefix: &str) -> Vec<BTreeMap<String, String>> {
        for (tags, rows) in &self.loops {
            if tags.first().is_some_and(|tag| tag.starts_with(prefix)) {
                return rows
                    .iter()
                    .map(|row| {
                        tags.iter()
                            .zip(row)
                            .map(|(tag, value)| (tag[prefix.len()..].to_string(), value.clone()))
                            .collect()
                    })
                    .collect();
            }
        }
        let single = self
            .items
            .iter()
            .filter_map(|(tag, value)| {
                tag.strip_prefix(prefix)
                    .map(|field| (field.to_string(), value.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        if single.is_empty() {
            Vec::new()
        } else {
            vec![single]
        }
    }
}

fn component_from_block(block: &CifBlock) -> Option<Component> {
    let comp = block.category("_chem_comp.").into_iter().next()?;
    let id = comp.get("id")?.to_ascii_uppercase();
    let value = |row: &BTreeMap<String, String>, key: &str| {
        row.get(key)
            .filter(|value| !matches!(value.as_str(), "?" | "."))
            .cloned()
    };
    let atoms = block
        .category("_chem_comp_atom.")
        .iter()
        .filter_map(|row| {
            let coordinate = |axis: &str| {
                value(row, &format!("pdbx_model_Cartn_{axis}_ideal"))
                    .or_else(|| value(row, &format!("model_Cartn_{axis}")))
                    .and_then(|text| text.parse::<f64>().ok())
            };
            let ideal = match (coordinate("x"), coordinate("y"), coordinate("z")) {
                (Some(x), Some(y), Some(z)) => Some(Vec3 { x, y, z }),
                _ => None,
            };
            Some(ComponentAtom {
                name: value(row, "atom_id")?,
                element: value(row, "type_symbol")?.to_ascii_uppercase(),
                formal_charge: value(row, "charge")
                    .and_then(|text| text.parse().ok())
                    .unwrap_or(0),
                aromatic: value(row, "pdbx_aromatic_flag").as_deref() == Some("Y"),
                leaving: value(row, "pdbx_leaving_atom_flag").as_deref() == Some("Y"),
                ideal,
            })
        })
        .collect::<Vec<_>>();
    if atoms.is_empty() {
        return None;
    }
    let bonds = block
        .category("_chem_comp_bond.")
        .iter()
        .filter_map(|row| {
            let order = match value(row, "value_order")?.to_ascii_uppercase().as_str() {
                "SING" => 1,
                "DOUB" => 2,
                "TRIP" => 3,
                "AROM" => 1,
                _ => 1,
            };
            Some(ComponentBond {
                first: value(row, "atom_id_1")?,
                second: value(row, "atom_id_2")?,
                order,
                aromatic: value(row, "pdbx_aromatic_flag").as_deref() == Some("Y"),
            })
        })
        .collect();
    Some(Component {
        id,
        name: value(&comp, "name").unwrap_or_default(),
        kind: value(&comp, "type").unwrap_or_default(),
        parent: value(&comp, "mon_nstd_parent_comp_id")
            .map(|parent| {
                parent
                    .split(',')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_ascii_uppercase()
            })
            .filter(|parent| !parent.is_empty()),
        atoms,
        bonds,
    })
}

/// Minimal mmCIF tokenizer: data blocks, items, loops, quoted strings and
/// semicolon text fields.  Save frames are not used by the CCD.
fn parse_cif(text: &str) -> Result<Vec<CifBlock>> {
    let tokens = tokenize(text);
    let mut blocks = Vec::new();
    let mut index = 0;
    let mut current: Option<CifBlock> = None;
    while index < tokens.len() {
        let token = &tokens[index];
        if !token.quoted && token.text.len() >= 5 && token.text[..5].eq_ignore_ascii_case("data_") {
            if let Some(block) = current.take() {
                blocks.push(block);
            }
            current = Some(CifBlock::default());
            index += 1;
        } else if !token.quoted && token.text.eq_ignore_ascii_case("loop_") {
            index += 1;
            let mut tags = Vec::new();
            while index < tokens.len()
                && !tokens[index].quoted
                && tokens[index].text.starts_with('_')
            {
                tags.push(tokens[index].text.clone());
                index += 1;
            }
            let mut values = Vec::new();
            while index < tokens.len() {
                let next = &tokens[index];
                if !next.quoted
                    && (next.text.starts_with('_')
                        || next.text.eq_ignore_ascii_case("loop_")
                        || (next.text.len() >= 5 && next.text[..5].eq_ignore_ascii_case("data_")))
                {
                    break;
                }
                values.push(next.text.clone());
                index += 1;
            }
            if tags.is_empty() || values.len() % tags.len() != 0 {
                return Err(BuildError::InvalidPdb(
                    "malformed mmCIF loop in chemical component".into(),
                ));
            }
            let rows = values.chunks(tags.len()).map(<[String]>::to_vec).collect();
            current
                .get_or_insert_with(CifBlock::default)
                .loops
                .push((tags, rows));
        } else if !token.quoted && token.text.starts_with('_') {
            let tag = token.text.clone();
            let value = tokens
                .get(index + 1)
                .map(|value| value.text.clone())
                .unwrap_or_default();
            current
                .get_or_insert_with(CifBlock::default)
                .items
                .insert(tag, value);
            index += 2;
        } else {
            index += 1;
        }
    }
    if let Some(block) = current {
        blocks.push(block);
    }
    Ok(blocks)
}

struct Token {
    text: String,
    quoted: bool,
}

fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if let Some(first) = line.strip_prefix(';') {
            let mut value = first.to_string();
            for next in lines.by_ref() {
                if next.starts_with(';') {
                    break;
                }
                value.push('\n');
                value.push_str(next);
            }
            tokens.push(Token {
                text: value.trim().to_string(),
                quoted: true,
            });
            continue;
        }
        let bytes = line.as_bytes();
        let mut index = 0;
        while index < bytes.len() {
            let byte = bytes[index];
            if byte.is_ascii_whitespace() {
                index += 1;
            } else if byte == b'#' {
                break;
            } else if byte == b'\'' || byte == b'"' {
                // A quote closes only when followed by whitespace or the end.
                let start = index + 1;
                let mut end = start;
                while end < bytes.len()
                    && !(bytes[end] == byte
                        && (end + 1 == bytes.len() || bytes[end + 1].is_ascii_whitespace()))
                {
                    end += 1;
                }
                tokens.push(Token {
                    text: line[start..end.min(bytes.len())].to_string(),
                    quoted: true,
                });
                index = end + 1;
            } else {
                let start = index;
                while index < bytes.len() && !bytes[index].is_ascii_whitespace() {
                    index += 1;
                }
                tokens.push(Token {
                    text: line[start..index].to_string(),
                    quoted: false,
                });
            }
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZN: &str = "data_ZN
#
_chem_comp.id                                    ZN
_chem_comp.name                                  \"ZINC ION\"
_chem_comp.type                                  NON-POLYMER
_chem_comp.mon_nstd_parent_comp_id               ?
#
_chem_comp_atom.comp_id                    ZN
_chem_comp_atom.atom_id                    ZN
_chem_comp_atom.type_symbol                ZN
_chem_comp_atom.charge                     2
_chem_comp_atom.pdbx_aromatic_flag         N
_chem_comp_atom.pdbx_leaving_atom_flag     N
_chem_comp_atom.pdbx_model_Cartn_x_ideal   0.000
_chem_comp_atom.pdbx_model_Cartn_y_ideal   0.000
_chem_comp_atom.pdbx_model_Cartn_z_ideal   0.000
#
";

    const ACT: &str = "data_ACT
_chem_comp.id ACT
_chem_comp.name 'ACETATE ION'
_chem_comp.type NON-POLYMER
_chem_comp.mon_nstd_parent_comp_id ?
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
ACT C   C  0 N N -0.042 0.000  0.001
ACT O   O  0 N N -1.279 0.000 -0.001
ACT OXT O -1 N N 0.656 1.198 0.001
ACT CH3 C  0 N N 0.705 -1.296 0.000
ACT H1  H  0 N N 1.781 -1.113 0.000
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
ACT C O DOUB N
ACT C OXT SING N
ACT C CH3 SING N
ACT CH3 H1 SING N
";

    #[test]
    fn parses_single_valued_and_looped_components() {
        let mut library = ComponentLibrary::new();
        assert_eq!(library.add_cif(ZN).unwrap(), vec!["ZN"]);
        let zinc = library.get("ZN").unwrap();
        assert_eq!(zinc.atoms[0].formal_charge, 2);
        assert!(zinc.bonds.is_empty());

        // A malformed row (an extra column) must be rejected rather than
        // silently shifting every later value.
        let malformed = ACT.replace("ACT OXT O -1", "ACT OXT O OXT -1");
        assert!(ComponentLibrary::new().add_cif(&malformed).is_err());
        let mut library = ComponentLibrary::new();
        library.add_cif(ACT).unwrap();
        let acetate = library.get("act").unwrap();
        assert_eq!(acetate.name, "ACETATE ION");
        assert_eq!(acetate.atom("OXT").unwrap().formal_charge, -1);
        assert_eq!(acetate.bonds[0].order, 2);
    }
}
