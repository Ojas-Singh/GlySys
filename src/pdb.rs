use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::model::Vec3;
use crate::report::{GlycanReport, ResidueRef};
use crate::{BuildError, BuildOptions, BuildWarning, Result};

#[derive(Debug, Clone)]
pub(crate) struct PdbAtom {
    pub serial: u32,
    pub name: String,
    pub residue_name: String,
    pub chain: String,
    pub residue_number: i32,
    pub insertion_code: Option<char>,
    pub element: String,
    pub position: Vec3,
}

#[derive(Debug, Clone)]
pub(crate) struct PdbResidue {
    pub reference: ResidueRef,
    pub atoms: Vec<PdbAtom>,
}

#[derive(Debug, Clone)]
pub(crate) struct ParsedPdb {
    pub residues: Vec<PdbResidue>,
    pub conect: BTreeSet<(u32, u32)>,
    pub links: Vec<DeclaredLink>,
    pub ssbonds: Vec<(ResidueKey, ResidueKey)>,
    pub chains: Vec<String>,
    pub glycans: Vec<GlycanReport>,
    pub warnings: Vec<BuildWarning>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ResidueKey {
    pub chain: String,
    pub number: i32,
    pub insertion_code: Option<char>,
}

#[derive(Debug, Clone)]
pub(crate) struct DeclaredLink {
    pub first: ResidueKey,
    pub first_atom: String,
    pub second: ResidueKey,
    pub second_atom: String,
}

#[derive(Debug, Clone)]
struct CandidateAtom {
    atom: PdbAtom,
    altloc: Option<char>,
    occupancy: f64,
}

pub(crate) fn parse(contents: &str, options: &BuildOptions) -> Result<ParsedPdb> {
    let available_models = available_models(contents);
    if !available_models.contains(&options.model) {
        return Err(BuildError::ModelNotFound(options.model));
    }

    let mut current_model = if contents.lines().any(|line| line.starts_with("MODEL ")) {
        0
    } else {
        1
    };
    let mut candidates: BTreeMap<(String, i32, Option<char>, String), Vec<CandidateAtom>> =
        BTreeMap::new();
    let mut conect = BTreeSet::new();
    let mut links = Vec::new();
    let mut ssbonds = Vec::new();
    let mut discarded_hydrogen_count = 0usize;
    let mut preserved_glycan_hydrogen_count = 0usize;

    for (line_number, line) in contents.lines().enumerate() {
        if line.starts_with("MODEL ") {
            current_model = field(line, 10, 14).trim().parse().map_err(|_| {
                BuildError::InvalidPdb(format!("line {} has invalid MODEL", line_number + 1))
            })?;
            continue;
        }
        if line.starts_with("ENDMDL") {
            current_model = 0;
            continue;
        }
        if current_model != options.model {
            continue;
        }
        if line.starts_with("ATOM  ") || line.starts_with("HETATM") {
            let atom = parse_atom(line, line_number + 1)?;
            if atom.element.eq_ignore_ascii_case("H") || atom.element.eq_ignore_ascii_case("D") {
                if PROTEIN_RESIDUES.contains(&atom.residue_name.as_str()) {
                    discarded_hydrogen_count += 1;
                    continue;
                }
                preserved_glycan_hydrogen_count += 1;
            }
            let altloc = char_field(line, 16);
            let occupancy = field(line, 54, 60).trim().parse().unwrap_or(0.0);
            let key = (
                atom.chain.clone(),
                atom.residue_number,
                atom.insertion_code,
                atom.name.clone(),
            );
            candidates.entry(key).or_default().push(CandidateAtom {
                atom,
                altloc,
                occupancy,
            });
        } else if line.starts_with("CONECT") {
            let serials = line
                .as_bytes()
                .get(6..)
                .into_iter()
                .flat_map(|rest| rest.chunks(5))
                .filter_map(|chunk| std::str::from_utf8(chunk).ok()?.trim().parse::<u32>().ok())
                .collect::<Vec<_>>();
            if let Some(&first) = serials.first() {
                for &second in &serials[1..] {
                    conect.insert(ordered(first, second));
                }
            }
        } else if line.starts_with("LINK  ") {
            if let Some(link) = parse_link(line) {
                links.push(link);
            }
        } else if line.starts_with("SSBOND")
            && let Some(pair) = parse_ssbond(line)
        {
            ssbonds.push(pair);
        }
    }

    let mut warnings = Vec::new();
    if discarded_hydrogen_count != 0 {
        warnings.push(BuildWarning::InputHydrogensRebuilt(format!(
            "{discarded_hydrogen_count} protein hydrogen/deuterium atoms were discarded and rebuilt"
        )));
    }
    if preserved_glycan_hydrogen_count != 0 {
        warnings.push(BuildWarning::InputGlycanHydrogensPreserved(format!(
            "{preserved_glycan_hydrogen_count} supplied glycan hydrogen/deuterium coordinates were retained when compatible with GLYCAM"
        )));
    }
    let mut selected = Vec::with_capacity(candidates.len());
    for ((chain, number, insertion, name), mut choices) in candidates {
        choices.sort_by(|left, right| {
            let left_requested = altloc_rank(left.altloc, options.altloc);
            let right_requested = altloc_rank(right.altloc, options.altloc);
            right_requested
                .cmp(&left_requested)
                .then_with(|| right.occupancy.total_cmp(&left.occupancy))
                .then_with(|| left.altloc.cmp(&right.altloc))
        });
        let chosen = choices.remove(0);
        if !choices.is_empty() {
            warnings.push(BuildWarning::AlternateLocationSelected(format!(
                "{chain}:{number}{} {name}: selected {}",
                insertion.unwrap_or(' '),
                chosen.altloc.unwrap_or(' ')
            )));
        }
        selected.push(chosen.atom);
    }
    selected.sort_by_key(|atom| atom.serial);
    if selected.is_empty() {
        return Err(BuildError::InvalidPdb(
            "selected model contains no heavy atoms".into(),
        ));
    }

    let mut residue_order = Vec::<ResidueKey>::new();
    let mut residue_atoms: HashMap<ResidueKey, Vec<PdbAtom>> = HashMap::new();
    let mut residue_names = HashMap::new();
    for atom in selected {
        let key = ResidueKey {
            chain: atom.chain.clone(),
            number: atom.residue_number,
            insertion_code: atom.insertion_code,
        };
        if !residue_atoms.contains_key(&key) {
            residue_order.push(key.clone());
        }
        residue_names
            .entry(key.clone())
            .or_insert_with(|| atom.residue_name.clone());
        residue_atoms.entry(key).or_default().push(atom);
    }
    let residues = residue_order
        .into_iter()
        .map(|key| PdbResidue {
            reference: ResidueRef {
                chain: key.chain.clone(),
                name: residue_names[&key].clone(),
                number: key.number,
                insertion_code: key.insertion_code,
            },
            atoms: residue_atoms.remove(&key).unwrap_or_default(),
        })
        .collect::<Vec<_>>();
    let chains = residues
        .iter()
        .map(|residue| residue.reference.chain.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    // pdbtbx (used by crabWURCS) applies column-level validation to every
    // record it receives, including free-form REMARK text.  Our PDB reader is
    // intentionally more permissive, so give crabWURCS a canonical,
    // coordinate-only view of the selected model.  Connectivity records are
    // retained because Re-Glyco/GlycoShape PDBs use curated CONECT records for
    // glycosidic and protein-glycan bonds.
    let crab_contents = crab_pdb_view(contents, options.model);
    let glycans = match crabwurcs::pdb::extract_glycans_from_str(&crab_contents, false) {
        Ok(glycans) => glycans
            .into_iter()
            .filter(|glycan| {
                glycan
                    .attachment_site
                    .as_deref()
                    .and_then(|site| site.split('/').nth(1))
                    .is_none_or(|name| {
                        !PROTEIN_RESIDUES.contains(&name)
                            || matches!(
                                name,
                                "ASN" | "NLN" | "SER" | "OLS" | "THR" | "OLT" | "HYP" | "OLP"
                            )
                    })
            })
            .map(|glycan| {
                let wurcs = crabwurcs::write_notation(&glycan.graph, crabwurcs::Format::Wurcs)
                    .unwrap_or_else(|_| "unavailable".into());
                let glycam =
                    crabwurcs::write_notation(&glycan.graph, crabwurcs::Format::Glycam).ok();
                GlycanReport {
                    attachment_site: glycan.attachment_site,
                    residue_count: glycan.graph.node_count(),
                    wurcs,
                    glycam,
                }
            })
            .collect(),
        Err(error) => {
            return Err(BuildError::Glycan(error.to_string()));
        }
    };

    Ok(ParsedPdb {
        residues,
        conect,
        links,
        ssbonds,
        chains,
        glycans,
        warnings,
    })
}

fn crab_pdb_view(contents: &str, selected_model: u32) -> String {
    let has_models = contents.lines().any(|line| line.starts_with("MODEL "));
    let mut current_model = if has_models { 0 } else { 1 };
    let mut coordinate_records = Vec::new();
    let mut connectivity_records = Vec::new();

    for line in contents.lines() {
        if line.starts_with("MODEL ") {
            current_model = field(line, 10, 14).trim().parse().unwrap_or(0);
            continue;
        }
        if line.starts_with("ENDMDL") {
            current_model = 0;
            continue;
        }
        if (line.starts_with("ATOM  ") || line.starts_with("HETATM"))
            && current_model == selected_model
        {
            coordinate_records.push(format!("{line:<80}"));
        } else if line.starts_with("CONECT")
            || line.starts_with("LINK  ")
            || line.starts_with("SSBOND")
        {
            connectivity_records.push(format!("{line:<80}"));
        }
    }

    coordinate_records.extend(connectivity_records);
    coordinate_records.push(format!("{:<80}", "END"));
    coordinate_records.join("\n") + "\n"
}

fn available_models(contents: &str) -> HashSet<u32> {
    let models = contents
        .lines()
        .filter(|line| line.starts_with("MODEL "))
        .filter_map(|line| field(line, 10, 14).trim().parse().ok())
        .collect::<HashSet<_>>();
    if models.is_empty() {
        HashSet::from([1])
    } else {
        models
    }
}

fn parse_atom(line: &str, line_number: usize) -> Result<PdbAtom> {
    if line.len() < 54 {
        return Err(BuildError::InvalidPdb(format!(
            "line {line_number} is shorter than the coordinate columns"
        )));
    }
    let parse_number = |start, end, label| {
        field(line, start, end)
            .trim()
            .parse::<f64>()
            .map_err(|_| BuildError::InvalidPdb(format!("line {line_number} has invalid {label}")))
    };
    let serial = field(line, 6, 11).trim().parse().map_err(|_| {
        BuildError::InvalidPdb(format!("line {line_number} has invalid atom serial"))
    })?;
    let residue_number = field(line, 22, 26).trim().parse().map_err(|_| {
        BuildError::InvalidPdb(format!("line {line_number} has invalid residue number"))
    })?;
    let name = field(line, 12, 16).trim().to_string();
    let guessed_element = name
        .trim_start_matches(|character: char| character.is_ascii_digit())
        .chars()
        .next()
        .unwrap_or('X')
        .to_ascii_uppercase()
        .to_string();
    Ok(PdbAtom {
        serial,
        name,
        residue_name: field(line, 17, 20).trim().to_string(),
        chain: field(line, 21, 22).trim().to_string(),
        residue_number,
        insertion_code: char_field(line, 26),
        element: {
            let declared = field(line, 76, 78).trim();
            if declared.is_empty() {
                guessed_element
            } else {
                declared.to_string()
            }
        },
        position: Vec3 {
            x: parse_number(30, 38, "x coordinate")?,
            y: parse_number(38, 46, "y coordinate")?,
            z: parse_number(46, 54, "z coordinate")?,
        },
    })
}

fn parse_link(line: &str) -> Option<DeclaredLink> {
    Some(DeclaredLink {
        first_atom: field(line, 12, 16).trim().to_string(),
        first: ResidueKey {
            chain: field(line, 21, 22).trim().to_string(),
            number: field(line, 22, 26).trim().parse().ok()?,
            insertion_code: char_field(line, 26),
        },
        second_atom: field(line, 42, 46).trim().to_string(),
        second: ResidueKey {
            chain: field(line, 51, 52).trim().to_string(),
            number: field(line, 52, 56).trim().parse().ok()?,
            insertion_code: char_field(line, 56),
        },
    })
}

fn parse_ssbond(line: &str) -> Option<(ResidueKey, ResidueKey)> {
    Some((
        ResidueKey {
            chain: field(line, 15, 16).trim().to_string(),
            number: field(line, 17, 21).trim().parse().ok()?,
            insertion_code: char_field(line, 21),
        },
        ResidueKey {
            chain: field(line, 29, 30).trim().to_string(),
            number: field(line, 31, 35).trim().parse().ok()?,
            insertion_code: char_field(line, 35),
        },
    ))
}

fn altloc_rank(value: Option<char>, requested: Option<char>) -> u8 {
    if value.is_none() {
        4
    } else if value == requested {
        3
    } else if requested.is_none() && value == Some('A') {
        2
    } else {
        1
    }
}

fn ordered(first: u32, second: u32) -> (u32, u32) {
    if first < second {
        (first, second)
    } else {
        (second, first)
    }
}

fn field(line: &str, start: usize, end: usize) -> &str {
    line.get(start..end).unwrap_or("")
}

fn char_field(line: &str, index: usize) -> Option<char> {
    line.as_bytes()
        .get(index)
        .copied()
        .map(char::from)
        .filter(|character| !character.is_ascii_whitespace())
}

pub(crate) const PROTEIN_RESIDUES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYM", "CYX", "GLN", "GLU", "GLH", "GLY", "HIS",
    "HID", "HIE", "HIP", "HYP", "ILE", "LEU", "LYN", "LYS", "MET", "PHE", "PRO", "SER", "THR",
    "TRP", "TYR", "VAL", "NLN", "OLS", "OLT", "OLP",
];

pub(crate) fn is_water(name: &str) -> bool {
    matches!(name, "HOH" | "WAT" | "TIP" | "TP3")
}

pub(crate) fn is_free_ion(name: &str) -> bool {
    matches!(
        name.to_ascii_uppercase().as_str(),
        "NA" | "NA+" | "CL" | "CL-" | "K" | "K+" | "MG" | "MG2+" | "CA" | "CA2+"
    )
}
