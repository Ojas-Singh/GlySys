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
    pub occupancy: f64,
    pub b_factor: f64,
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

/// One residue of the selected model, before force-field interpretation.
#[derive(Debug, Clone)]
pub(crate) struct RawResidue {
    pub reference: ResidueRef,
    pub atoms: Vec<PdbAtom>,
    /// A TER record follows this residue in the input.
    pub ter_after: bool,
}

/// Coordinates and connectivity of one model with alternate locations resolved.
#[derive(Debug, Clone, Default)]
pub(crate) struct RawModel {
    pub residues: Vec<RawResidue>,
    pub conect: BTreeSet<(u32, u32)>,
    pub links: Vec<DeclaredLink>,
    pub ssbonds: Vec<(ResidueKey, ResidueKey)>,
    /// SEQRES residue names per chain, in sequence order.
    pub seqres: BTreeMap<String, Vec<String>>,
    /// MODRES records: modified residue and its standard parent name.
    pub modres: Vec<(ResidueKey, String, String)>,
    /// REMARK 465 residues (unobserved), in the order listed.
    pub unobserved: Vec<(ResidueKey, String)>,
    pub warnings: Vec<BuildWarning>,
}

/// Read the selected model's atoms and connectivity records.
///
/// Alternate locations are resolved per residue: a microheterogeneous site
/// (two residue types sharing one residue number) keeps only the residue type
/// of the selected alternate location plus the shared, unlabelled atoms.
/// Distinct residues that merely share a number (no alternate locations) are
/// kept apart.  Hydrogens are retained; callers decide whether to keep them.
pub(crate) fn read_model(contents: &str, model: u32, altloc: Option<char>) -> Result<RawModel> {
    let available_models = available_models(contents);
    if !available_models.contains(&model) {
        return Err(BuildError::ModelNotFound(model));
    }
    let mut current_model = if contents.lines().any(|line| line.starts_with("MODEL ")) {
        0
    } else {
        1
    };
    type Group = ((String, i32, Option<char>), Vec<CandidateAtom>);
    let mut groups: Vec<Group> = Vec::new();
    let mut group_index: HashMap<(String, i32, Option<char>), usize> = HashMap::new();
    let mut ter_after_group = HashSet::new();
    let mut result = RawModel::default();
    let mut used_serials = HashSet::new();
    let mut synthetic_serial = 100_000_000u32;
    let mut in_remark_465_table = false;

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
        // LINK, SSBOND, and CONECT records are commonly written once in the
        // global pre-MODEL section of an NMR/assembly PDB.  They describe the
        // selected coordinates and must not be discarded merely because the
        // parser is currently outside the requested MODEL block.
        let global = current_model == 0 || current_model == model;
        if line.starts_with("CONECT") {
            if global {
                let serials = line
                    .as_bytes()
                    .get(6..)
                    .into_iter()
                    .flat_map(|rest| rest.chunks(5))
                    .filter_map(|chunk| decode_serial(std::str::from_utf8(chunk).ok()?))
                    .collect::<Vec<_>>();
                if let Some(&first) = serials.first() {
                    for &second in &serials[1..] {
                        result.conect.insert(ordered(first, second));
                    }
                }
            }
            continue;
        }
        if line.starts_with("LINK  ") {
            if global && let Some(link) = parse_link(line) {
                result.links.push(link);
            }
            continue;
        }
        if line.starts_with("SSBOND") {
            if global && let Some(pair) = parse_ssbond(line) {
                result.ssbonds.push(pair);
            }
            continue;
        }
        if line.starts_with("SEQRES") {
            let chain = field(line, 11, 12).trim().to_string();
            let names = field(line, 19, 80)
                .split_whitespace()
                .map(str::to_string)
                .collect::<Vec<_>>();
            result.seqres.entry(chain).or_default().extend(names);
            continue;
        }
        if line.starts_with("REMARK 465") {
            let body = field(line, 10, 80);
            if body.contains("RES C SSSEQI") {
                in_remark_465_table = true;
            } else if in_remark_465_table && let Some(entry) = parse_remark_465(body, model) {
                result.unobserved.push(entry);
            }
            continue;
        }
        if line.starts_with("MODRES") {
            let name = field(line, 12, 15).trim().to_string();
            let parent = field(line, 24, 27).trim().to_string();
            if let Some(number) = decode_residue_number(field(line, 18, 22)) {
                result.modres.push((
                    ResidueKey {
                        chain: field(line, 16, 17).trim().to_string(),
                        number,
                        insertion_code: char_field(line, 22),
                    },
                    name,
                    parent,
                ));
            }
            continue;
        }
        if current_model != model {
            continue;
        }
        if line.starts_with("TER") {
            if let Some(last) = groups.len().checked_sub(1) {
                ter_after_group.insert(last);
            }
            continue;
        }
        if line.starts_with("ATOM  ") || line.starts_with("HETATM") {
            let mut atom = parse_atom(line, line_number + 1)?;
            if !used_serials.insert(atom.serial) {
                // Duplicate or overflowed serials cannot be referenced by
                // CONECT unambiguously; keep the atom with a private serial.
                synthetic_serial += 1;
                atom.serial = synthetic_serial;
            }
            let key = (atom.chain.clone(), atom.residue_number, atom.insertion_code);
            let candidate = CandidateAtom {
                altloc: char_field(line, 16),
                occupancy: field(line, 54, 60).trim().parse().unwrap_or(0.0),
                atom,
            };
            // A residue interrupted by a TER record or by other residues
            // restarts only when its number is reused non-contiguously by a
            // different molecule (e.g. waters numbered like the protein).
            let index = match group_index.get(&key) {
                Some(&index)
                    if index + 1 == groups.len()
                        || groups[index].1.iter().any(|existing| {
                            existing.atom.residue_name == candidate.atom.residue_name
                        }) =>
                {
                    index
                }
                _ => {
                    groups.push((key.clone(), Vec::new()));
                    group_index.insert(key, groups.len() - 1);
                    groups.len() - 1
                }
            };
            groups[index].1.push(candidate);
        }
    }

    let mut alternate_selections = 0usize;
    for (group_number, (key, candidates)) in groups.into_iter().enumerate() {
        let ter_after = ter_after_group.contains(&group_number);
        for (name, atoms) in resolve_residue(candidates, altloc, &mut alternate_selections) {
            if atoms.is_empty() {
                continue;
            }
            result.residues.push(RawResidue {
                reference: ResidueRef {
                    chain: key.0.clone(),
                    name,
                    number: key.1,
                    insertion_code: key.2,
                },
                atoms,
                ter_after,
            });
        }
    }
    if alternate_selections != 0 {
        result
            .warnings
            .push(BuildWarning::AlternateLocationSelected(format!(
                "{alternate_selections} atoms had alternate locations; selected {}",
                altloc.map_or("A or the highest occupancy".to_string(), |value| value
                    .to_string())
            )));
    }
    if result.residues.is_empty() {
        return Err(BuildError::InvalidPdb(
            "selected model contains no atoms".into(),
        ));
    }
    Ok(result)
}

/// Split one residue-number group into residues and pick alternate locations.
fn resolve_residue(
    candidates: Vec<CandidateAtom>,
    requested: Option<char>,
    selections: &mut usize,
) -> Vec<(String, Vec<PdbAtom>)> {
    let mut names = Vec::<String>::new();
    for candidate in &candidates {
        if !names.contains(&candidate.atom.residue_name) {
            names.push(candidate.atom.residue_name.clone());
        }
    }
    // Two residue types that both carry alternate-location labels are a
    // microheterogeneous site.  Any other name mismatch inside one contiguous
    // residue is a formatting defect (for example a column-shifted line) and
    // is merged into the residue named by its first atom, as before.
    let labelled_names = names
        .iter()
        .filter(|name| {
            candidates.iter().any(|candidate| {
                &candidate.atom.residue_name == *name && candidate.altloc.is_some()
            })
        })
        .count();
    let residues: Vec<(String, Vec<CandidateAtom>)> = if labelled_names > 1 {
        let chosen_altloc = preferred_altloc(&candidates, requested);
        let chosen_name = candidates
            .iter()
            .find(|candidate| candidate.altloc == chosen_altloc)
            .map(|candidate| candidate.atom.residue_name.clone())
            .unwrap_or_else(|| names[0].clone());
        let kept = candidates
            .into_iter()
            .filter(|candidate| {
                candidate.altloc.is_none() || candidate.atom.residue_name == chosen_name
            })
            .map(|mut candidate| {
                candidate.atom.residue_name = chosen_name.clone();
                candidate
            })
            .collect();
        vec![(chosen_name, kept)]
    } else {
        let name = names[0].clone();
        let merged = candidates
            .into_iter()
            .map(|mut candidate| {
                candidate.atom.residue_name = name.clone();
                candidate
            })
            .collect();
        vec![(name, merged)]
    };

    residues
        .into_iter()
        .map(|(name, candidates)| {
            let mut by_name: Vec<(String, Vec<CandidateAtom>)> = Vec::new();
            for candidate in candidates {
                match by_name
                    .iter_mut()
                    .find(|(atom_name, _)| *atom_name == candidate.atom.name)
                {
                    Some((_, choices)) => choices.push(candidate),
                    None => by_name.push((candidate.atom.name.clone(), vec![candidate])),
                }
            }
            let atoms = by_name
                .into_iter()
                .map(|(_, mut choices)| {
                    choices.sort_by(|left, right| {
                        altloc_rank(right.altloc, requested)
                            .cmp(&altloc_rank(left.altloc, requested))
                            .then_with(|| right.occupancy.total_cmp(&left.occupancy))
                            .then_with(|| left.altloc.cmp(&right.altloc))
                    });
                    if choices.len() > 1 {
                        *selections += 1;
                    }
                    choices.swap_remove(0).atom
                })
                .collect();
            (name, atoms)
        })
        .collect()
}

fn preferred_altloc(candidates: &[CandidateAtom], requested: Option<char>) -> Option<char> {
    if requested.is_some() && candidates.iter().any(|c| c.altloc == requested) {
        return requested;
    }
    if candidates.iter().any(|c| c.altloc == Some('A')) {
        return Some('A');
    }
    let mut totals = BTreeMap::<char, f64>::new();
    for candidate in candidates {
        if let Some(altloc) = candidate.altloc {
            *totals.entry(altloc).or_default() += candidate.occupancy;
        }
    }
    totals
        .into_iter()
        .max_by(|left, right| left.1.total_cmp(&right.1).then(right.0.cmp(&left.0)))
        .map(|(altloc, _)| altloc)
}

pub(crate) fn parse(contents: &str, options: &BuildOptions) -> Result<ParsedPdb> {
    let raw = read_model(contents, options.model, options.altloc)?;
    let mut warnings = raw.warnings;
    let mut discarded_hydrogen_count = 0usize;
    let mut preserved_glycan_hydrogen_count = 0usize;
    let mut residues = Vec::with_capacity(raw.residues.len());
    for mut residue in raw.residues {
        let protein = PROTEIN_RESIDUES.contains(&residue.reference.name.as_str());
        if protein && let Some(state) = protonation_from_hydrogens(&residue) {
            residue.reference.name = state.into();
        }
        let mut atoms = Vec::with_capacity(residue.atoms.len());
        for atom in residue.atoms {
            if atom.element.eq_ignore_ascii_case("H") || atom.element.eq_ignore_ascii_case("D") {
                if protein {
                    discarded_hydrogen_count += 1;
                    continue;
                }
                preserved_glycan_hydrogen_count += 1;
            }
            atoms.push(atom);
        }
        if !atoms.is_empty() {
            residues.push(PdbResidue {
                reference: residue.reference,
                atoms,
            });
        }
    }
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
    if residues.is_empty() {
        return Err(BuildError::InvalidPdb(
            "selected model contains no heavy atoms".into(),
        ));
    }
    let conect = raw.conect;
    let links = raw.links;
    let ssbonds = raw.ssbonds;
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
                // GLYCAM GAG files may represent sulfate as a separate SO3
                // residue joined by CONECT, and free glycans can be capped
                // with ROH. Neither is an external attachment site.
                let attachment_site = glycan
                    .attachment_site
                    .as_deref()
                    .filter(|site| !matches!(site.split('/').nth(1), Some("SO3" | "ROH")))
                    .map(str::to_owned);
                GlycanReport {
                    attachment_site,
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

/// Protonation variant implied by the hydrogens of an input residue
/// (e.g. written by `glysysbuilder fix`); `None` without hydrogens.
fn protonation_from_hydrogens(residue: &RawResidue) -> Option<&'static str> {
    let has = |name: &str| residue.atoms.iter().any(|atom| atom.name == name);
    if !residue
        .atoms
        .iter()
        .any(|atom| matches!(atom.element.as_str(), "H" | "D"))
    {
        return None;
    }
    match residue.reference.name.as_str() {
        "HIS" => match (has("HD1"), has("HE2")) {
            (true, true) => Some("HIP"),
            (true, false) => Some("HID"),
            (false, true) => Some("HIE"),
            _ => None,
        },
        "ASP" if has("HD2") || has("HD1") => Some("ASH"),
        "GLU" if has("HE2") || has("HE1") => Some("GLH"),
        "LYS" if has("HZ1") && has("HZ2") && !has("HZ3") => Some("LYN"),
        _ => None,
    }
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
            // crabWURCS 0.3.1 deliberately limits its loose GLYCAM-code
            // heuristic to HETATM records: otherwise protein names such as
            // VAL can be mistaken for a three-character GLYCAM code.  A few
            // deposited standalone GLYCAM files nevertheless encode their
            // sugar records as ATOM (the common 0YB/0GL-style codes).  Mark
            // only the unambiguous numeric-prefix GLYCAM spelling as HETATM
            // in the compatibility view; ordinary protein ATOM records are
            // left untouched and therefore cannot become glycan residues.
            //
            // Sulfated glycosaminoglycan residues are another unambiguous
            // case.  GLYCAM encodes glucosamine sulfate forms as `?YS` or
            // `?YN` (for example 6YS, QYS, and VYS).  They are often emitted
            // as ATOM records by GLYCAM-Web/GMML, but their second/third
            // characters cannot collide with a standard amino-acid name.
            // Preserving them as HETATM lets crabWURCS retain one connected
            // GAG chain rather than splitting it at every sulfated residue.
            let mut record = format!("{line:<80}");
            if record.starts_with("ATOM  ") {
                let residue_name = record.get(17..20).unwrap_or_default().trim();
                let bytes = residue_name.as_bytes();
                let numeric_glycam = bytes.first().is_some_and(|byte| byte.is_ascii_digit());
                let sulfated_glucosamine = bytes.len() == 3
                    && matches!(bytes[1], b'Y' | b'y')
                    && matches!(bytes[2], b'N' | b'n' | b'S' | b's');
                if numeric_glycam || sulfated_glucosamine {
                    record.replace_range(0..6, "HETATM");
                }
            }
            coordinate_records.push(record);
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
    // Serials beyond 99,999 are written in hybrid-36 or as `*****` by some
    // tools.  An unreadable serial is replaced by a unique private value;
    // it can then no longer be referenced by CONECT, which is correct.
    let serial = decode_serial(field(line, 6, 11)).unwrap_or(900_000_000 + line_number as u32);
    let residue_number = decode_residue_number(field(line, 22, 26)).ok_or_else(|| {
        BuildError::InvalidPdb(format!("line {line_number} has invalid residue number"))
    })?;
    let name = field(line, 12, 16).trim().to_string();
    let residue_name = field(line, 17, 20).trim().to_string();
    Ok(PdbAtom {
        serial,
        name,
        chain: field(line, 21, 22).trim().to_string(),
        residue_number,
        insertion_code: char_field(line, 26),
        element: {
            let declared = field(line, 76, 78).trim();
            if !declared.is_empty() && declared.chars().all(|c| c.is_ascii_alphabetic()) {
                normalize_element(declared)
            } else {
                guess_element(field(line, 12, 16), &residue_name)
            }
        },
        residue_name,
        occupancy: parse_number(54, 60, "occupancy")
            .unwrap_or(1.0)
            .clamp(0.0, 1.0),
        b_factor: parse_number(60, 66, "B factor").unwrap_or(0.0).max(0.0),
        position: Vec3 {
            x: parse_number(30, 38, "x coordinate")?,
            y: parse_number(38, 46, "y coordinate")?,
            z: parse_number(46, 54, "z coordinate")?,
        },
    })
}

/// Element symbols are compared upper-case throughout GlySys.
fn normalize_element(value: &str) -> String {
    value.to_ascii_uppercase()
}

const TWO_LETTER_ELEMENTS: &[&str] = &[
    "LI", "BE", "NA", "MG", "AL", "SI", "CL", "AR", "CA", "SC", "TI", "CR", "MN", "FE", "CO", "NI",
    "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "ZR", "MO", "RU", "RH", "PD", "AG",
    "CD", "IN", "SN", "SB", "TE", "XE", "CS", "BA", "LA", "CE", "GD", "YB", "LU", "HF", "TA", "RE",
    "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI", "SM", "EU", "TB", "HO",
];

/// Guess an element from the raw 4-column atom-name field.
///
/// wwPDB right-justifies one-letter elements in columns 13-14 (" CA " is an
/// alpha carbon) and starts two-letter elements in column 13 ("CA  " is
/// calcium).  A left-justified name is therefore two-letter only when it
/// names a real element and the residue is plausibly that ion/metal.
fn guess_element(raw: &str, residue_name: &str) -> String {
    let raw = format!("{raw:<4}");
    let trimmed = raw.trim();
    let letters = trimmed
        .trim_start_matches(|c: char| c.is_ascii_digit())
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect::<String>()
        .to_ascii_uppercase();
    let first_column_letter = raw.chars().next().is_some_and(|c| c.is_ascii_alphabetic());
    if letters.len() >= 2 {
        let pair = &letters[..2];
        if TWO_LETTER_ELEMENTS.contains(&pair)
            && (residue_name.eq_ignore_ascii_case(pair)
                || (first_column_letter
                    && !PROTEIN_RESIDUES.contains(&residue_name)
                    && !matches!(pair, "CA" | "CD" | "NE" | "HG" | "HO" | "CE")))
        {
            return pair.to_string();
        }
    }
    letters
        .chars()
        .next()
        .map_or_else(|| "X".to_string(), |c| c.to_string())
}

/// Decimal or hybrid-36 atom serial (5 columns).
pub(crate) fn decode_serial(field: &str) -> Option<u32> {
    decode_hybrid36(field.trim(), 5)
}

/// Decimal or hybrid-36 residue sequence number (4 columns).
pub(crate) fn decode_residue_number(field: &str) -> Option<i32> {
    let value = field.trim();
    value
        .parse::<i32>()
        .ok()
        .or_else(|| decode_hybrid36(value, 4).map(|n| n as i32))
}

fn decode_hybrid36(value: &str, width: u32) -> Option<u32> {
    if value.is_empty() {
        return None;
    }
    if let Ok(number) = value.parse::<u32>() {
        return Some(number);
    }
    if value.len() != width as usize || !value.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    let first = value.chars().next()?;
    let digits = |upper: bool| -> Option<u32> {
        value.chars().try_fold(0u32, |accumulator, character| {
            let digit = if character.is_ascii_digit() {
                character as u32 - '0' as u32
            } else if upper && character.is_ascii_uppercase() {
                character as u32 - 'A' as u32 + 10
            } else if !upper && character.is_ascii_lowercase() {
                character as u32 - 'a' as u32 + 10
            } else {
                return None;
            };
            accumulator.checked_mul(36)?.checked_add(digit)
        })
    };
    let decimal_limit = 10u32.pow(width);
    let block = 26 * 36u32.pow(width - 1);
    if first.is_ascii_uppercase() {
        Some(digits(true)? - 10 * 36u32.pow(width - 1) + decimal_limit)
    } else if first.is_ascii_lowercase() {
        Some(digits(false)? - 10 * 36u32.pow(width - 1) + decimal_limit + block)
    } else {
        None
    }
}

/// Hybrid-36 encoding used when a serial or residue number overflows.
pub(crate) fn encode_hybrid36(value: u32, width: u32) -> String {
    let decimal_limit = 10u32.pow(width);
    if value < decimal_limit {
        return format!("{value:>width$}", width = width as usize);
    }
    let block = 26 * 36u32.pow(width - 1);
    let (offset, alphabet) = if value < decimal_limit + block {
        (
            value - decimal_limit + 10 * 36u32.pow(width - 1),
            b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        )
    } else {
        (
            (value - decimal_limit - block).min(block - 1) + 10 * 36u32.pow(width - 1),
            b"0123456789abcdefghijklmnopqrstuvwxyz",
        )
    };
    let mut digits = vec![b'0'; width as usize];
    let mut remaining = offset;
    for slot in digits.iter_mut().rev() {
        *slot = alphabet[(remaining % 36) as usize];
        remaining /= 36;
    }
    String::from_utf8(digits).expect("hybrid-36 digits are ASCII")
}

/// `[model] RES C SSSEQI` of a REMARK 465 table row.
fn parse_remark_465(body: &str, model: u32) -> Option<(ResidueKey, String)> {
    let tokens = body.split_whitespace().collect::<Vec<_>>();
    let (entry_model, rest) = match tokens.len() {
        3 => (None, &tokens[..]),
        4 => (tokens[0].parse::<u32>().ok(), &tokens[1..]),
        _ => return None,
    };
    if entry_model.is_some_and(|value| value != model) {
        return None;
    }
    let name = rest[0].to_string();
    let chain = rest[1].to_string();
    if chain.len() != 1 || name.len() > 3 {
        return None;
    }
    let sequence = rest[2];
    let split = sequence
        .char_indices()
        .rfind(|(_, c)| c.is_ascii_digit())
        .map(|(index, c)| index + c.len_utf8())?;
    let number = sequence[..split].parse::<i32>().ok()?;
    let insertion_code = sequence[split..].chars().next();
    Some((
        ResidueKey {
            chain,
            number,
            insertion_code,
        },
        name,
    ))
}

fn parse_link(line: &str) -> Option<DeclaredLink> {
    Some(DeclaredLink {
        first_atom: field(line, 12, 16).trim().to_string(),
        first: ResidueKey {
            chain: field(line, 21, 22).trim().to_string(),
            number: decode_residue_number(field(line, 22, 26))?,
            insertion_code: char_field(line, 26),
        },
        second_atom: field(line, 42, 46).trim().to_string(),
        second: ResidueKey {
            chain: field(line, 51, 52).trim().to_string(),
            number: decode_residue_number(field(line, 52, 56))?,
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

/// Bulk monovalent ions: removed and re-added by solvation.
pub(crate) fn is_free_ion(name: &str) -> bool {
    matches!(
        name.to_ascii_uppercase().as_str(),
        "NA" | "NA+" | "CL" | "CL-" | "K" | "K+"
    )
}

/// Structural divalent metal ions kept with Li/Merz 12-6 parameters.
pub(crate) fn is_metal_ion(name: &str) -> bool {
    matches!(
        name.to_ascii_uppercase().as_str(),
        "MG" | "CA" | "ZN" | "MN" | "FE2" | "CU" | "CO" | "NI" | "CD" | "HG"
    )
}

pub(crate) const NUCLEIC_RESIDUES: &[&str] = &["DA", "DC", "DG", "DT", "A", "C", "G", "U"];
