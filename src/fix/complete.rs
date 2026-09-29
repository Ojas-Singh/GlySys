//! Template selection, modified-residue replacement and heavy-atom rebuilding.

use std::collections::{HashMap, HashSet, VecDeque};

use super::geometry::{Superposition, V, add, distance, normalize, place, scale, sub, v};
use super::work::{Origin, ResidueKind, WAtom, WResidue, Work};
use crate::forcefield::{Template, TemplateSet};

/// Terminal state of a polymer residue.
pub(crate) fn terminal_flags(work: &Work, index: usize) -> (bool, bool) {
    let residue = &work.residues[index];
    let n_terminal = residue.prev.is_none() && !residue.open_before;
    let c_terminal = residue.next.is_none() && !residue.open_after;
    (n_terminal, c_terminal)
}

/// Force-field template for a protein or nucleic-acid residue.
pub(crate) fn residue_template<'a>(
    templates: &'a TemplateSet,
    work: &Work,
    index: usize,
) -> Option<(&'a Template, Option<String>)> {
    let residue = &work.residues[index];
    let (first, last) = terminal_flags(work, index);
    match residue.kind {
        ResidueKind::Protein => {
            let amber = match residue.name.as_str() {
                "NH2" => "NHE".to_string(),
                _ => residue
                    .variant
                    .clone()
                    .unwrap_or_else(|| residue.name.clone()),
            };
            if residue.is_cap() {
                let cap = if amber == "ACE" {
                    templates.protein(&amber, true, false)
                } else {
                    templates.protein(&amber, false, true)
                };
                return cap.map(|template| (template, None));
            }
            // A cap is a bonded neighbour, not a chain end.
            let first = first && !residue.prev.is_some_and(|p| work.residues[p].is_cap());
            let last = last && !residue.next.is_some_and(|n| work.residues[n].is_cap());
            if let Some(template) = templates.protein(&amber, first, last) {
                return Some((template, None));
            }
            // Terminal forms exist only for the common protonation states:
            // use the charged standard form, else the internal template.
            let standard = match amber.as_str() {
                "ASH" => Some("ASP"),
                "GLH" => Some("GLU"),
                "LYN" => Some("LYS"),
                "CYM" => Some("CYS"),
                _ => None,
            };
            if let Some(standard) = standard
                && let Some(template) = templates.protein(standard, first, last)
            {
                return Some((
                    template,
                    Some(format!(
                        "{}: no terminal {amber} template; used the terminal {standard} form",
                        residue.label()
                    )),
                ));
            }
            templates.protein(&amber, false, false).map(|template| {
                (
                    template,
                    Some(format!(
                        "{}: no terminal {amber} template; built as an internal residue",
                        residue.label()
                    )),
                )
            })
        }
        ResidueKind::Nucleic => {
            // A deposited 5' phosphate keeps the internal template atoms.
            let five_prime = first && residue.atom("P").is_none();
            templates
                .nucleic(&residue.name, five_prime, last)
                .map(|template| (template, None))
        }
        _ => None,
    }
}

/// Heavy-atom names of a template.
fn heavy_names(template: &Template) -> HashSet<&str> {
    template
        .atoms
        .iter()
        .filter(|atom| atom.element != 1)
        .map(|atom| atom.name.as_str())
        .collect()
}

/// Replace a modified residue by its standard parent.  Returns removed atoms.
pub(crate) fn replace_with_parent(
    residue: &mut WResidue,
    templates: &TemplateSet,
) -> Option<Vec<String>> {
    let parent = residue.parent.clone()?;
    let (kind, template) = if let Some(template) = templates.protein(&parent, false, false) {
        (ResidueKind::Protein, template)
    } else {
        (
            ResidueKind::Nucleic,
            templates.nucleic(&parent, false, false)?,
        )
    };
    let allowed = heavy_names(template);
    // Selenomethionine: keep the selenium position as sulfur at C-S length.
    if residue.name == "MSE"
        && let Some(cg) = residue.position("CG")
        && let Some(selenium) = residue.atoms.iter_mut().find(|atom| atom.name == "SE")
    {
        if let Some(direction) = normalize(sub(selenium.position, cg)) {
            selenium.position = add(cg, scale(direction, 1.81));
        }
        selenium.name = "SD".into();
        selenium.element = "S".into();
    }
    let mut removed = Vec::new();
    residue.atoms.retain(|atom| {
        let keep =
            !atom.is_hydrogen() && (allowed.contains(atom.name.as_str()) || atom.name == "OXT");
        if !keep && !atom.is_hydrogen() {
            removed.push(atom.name.clone());
        }
        keep
    });
    residue.name = parent;
    residue.kind = kind;
    residue.variant = None;
    Some(removed)
}

/// Outcome of completing one residue.
#[derive(Default)]
pub(crate) struct Completion {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub unresolved: Vec<String>,
}

/// Build the template's missing heavy atoms from local geometry.
///
/// Each missing atom is placed from its bonded parent and the parent's
/// already-placed neighbours, reproducing the template's local bond length,
/// angle and torsion (a natural-extension build), so existing coordinates are
/// never moved.  Trigonal centres bonded to the neighbouring residue (the
/// carbonyl O of a peptide) are built in the actual peptide plane.
pub(crate) fn complete_residue(
    residue: &mut WResidue,
    template: &Template,
    previous_link: Option<V>,
    next_link: Option<V>,
    drop_extra: bool,
) -> Completion {
    let mut completion = Completion::default();
    let names = template
        .atoms
        .iter()
        .enumerate()
        .map(|(index, atom)| (atom.name.as_str(), index))
        .collect::<HashMap<_, _>>();
    // Remove heavy atoms the template does not define (and stray hydrogens).
    let keep_phosphate_oxygen = template
        .head
        .is_some_and(|head| template.atoms[head].name == "P");
    residue.atoms.retain(|atom| {
        let known =
            names.contains_key(atom.name.as_str()) || (keep_phosphate_oxygen && atom.name == "OP3");
        if !known && drop_extra && !atom.is_hydrogen() {
            completion.removed.push(atom.name.clone());
        }
        known || !drop_extra
    });

    let mut adjacency = vec![Vec::new(); template.atoms.len()];
    for &[a, b] in &template.bonds {
        adjacency[a].push(b);
        adjacency[b].push(a);
    }
    let mut placed: Vec<Option<V>> = template
        .atoms
        .iter()
        .map(|atom| residue.position(&atom.name))
        .collect();
    let external = |index: usize| -> Option<V> {
        if Some(index) == template.head {
            previous_link
        } else if Some(index) == template.tail {
            next_link
        } else {
            None
        }
    };
    let missing = template
        .atoms
        .iter()
        .enumerate()
        .filter(|(index, atom)| atom.element != 1 && placed[*index].is_none())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return completion;
    }
    let present_heavy = placed
        .iter()
        .enumerate()
        .filter(|(index, position)| position.is_some() && template.atoms[*index].element != 1)
        .count();
    if present_heavy == 0 {
        completion.unresolved = missing
            .iter()
            .map(|&i| template.atoms[i].name.clone())
            .collect();
        return completion;
    }

    // Breadth-first growth from atoms that already have coordinates.
    let mut queue = VecDeque::new();
    for &index in &missing {
        queue.push_back(index);
    }
    let mut stalled = 0;
    while let Some(index) = queue.pop_front() {
        if placed[index].is_some() {
            continue;
        }
        match grow_atom(template, &adjacency, &placed, index, &external) {
            Some(position) => {
                placed[index] = Some(position);
                stalled = 0;
            }
            None => {
                queue.push_back(index);
                stalled += 1;
                if stalled > queue.len() {
                    break;
                }
            }
        }
    }
    // Anything still missing: rigid superposition of the whole template.
    let remaining = missing
        .iter()
        .copied()
        .filter(|&index| placed[index].is_none())
        .collect::<Vec<_>>();
    if !remaining.is_empty() {
        let pairs = placed
            .iter()
            .enumerate()
            .filter_map(|(index, position)| {
                let position = (*position)?;
                (template.atoms[index].element != 1)
                    .then_some((v(template.atoms[index].position), position))
            })
            .collect::<Vec<_>>();
        let (from, to): (Vec<V>, Vec<V>) = pairs.into_iter().unzip();
        if let Some(fit) = Superposition::fit(&from, &to) {
            for index in remaining {
                placed[index] = Some(fit.apply(v(template.atoms[index].position)));
            }
        } else {
            completion.unresolved = remaining
                .iter()
                .map(|&index| template.atoms[index].name.clone())
                .collect();
        }
    }
    for &index in &missing {
        if let Some(position) = placed[index] {
            let atom = &template.atoms[index];
            completion.added.push(atom.name.clone());
            residue.atoms.push(WAtom {
                name: atom.name.clone(),
                element: element_symbol(atom.element).to_string(),
                position,
                occupancy: 1.0,
                b_factor: 0.0,
                serial: None,
                origin: if residue.modelled {
                    Origin::Modelled
                } else {
                    Origin::Rebuilt
                },
            });
        }
    }
    order_like_template(residue, template);
    completion
}

fn grow_atom(
    template: &Template,
    adjacency: &[Vec<usize>],
    placed: &[Option<V>],
    index: usize,
    external: &dyn Fn(usize) -> Option<V>,
) -> Option<V> {
    let atom_template = v(template.atoms[index].position);
    // Prefer a heavy parent; hydrogens are never used as references.
    let parent = adjacency[index]
        .iter()
        .copied()
        .filter(|&neighbor| template.atoms[neighbor].element != 1)
        .find(|&neighbor| placed[neighbor].is_some())?;
    let parent_actual = placed[parent]?;
    let parent_template = v(template.atoms[parent].position);
    let bond = distance(atom_template, parent_template);
    let references = adjacency[parent]
        .iter()
        .copied()
        .filter(|&neighbor| neighbor != index && template.atoms[neighbor].element != 1)
        .filter_map(|neighbor| placed[neighbor].map(|position| (neighbor, position)))
        .collect::<Vec<_>>();
    let heavy_degree = adjacency[parent]
        .iter()
        .filter(|&&neighbor| template.atoms[neighbor].element != 1)
        .count()
        + usize::from(
            external(parent).is_some()
                || Some(parent) == template.head
                || Some(parent) == template.tail,
        );
    let total_degree = adjacency[parent].len()
        + usize::from(Some(parent) == template.head || Some(parent) == template.tail);

    // Trigonal centre whose third substituent is in the neighbouring residue.
    if let Some(link) = external(parent)
        && references.len() == 1
        && total_degree == 3
        && heavy_degree == 3
    {
        let a = normalize(sub(parent_actual, references[0].1))?;
        let b = normalize(sub(parent_actual, link))?;
        let direction = normalize(add(a, b))?;
        return Some(add(parent_actual, scale(direction, bond)));
    }
    if references.len() >= 2 {
        let from = [
            parent_template,
            v(template.atoms[references[0].0].position),
            v(template.atoms[references[1].0].position),
        ];
        let to = [parent_actual, references[0].1, references[1].1];
        // Superimpose the local frame (parent + two neighbours).
        let fit = Superposition::fit(&from, &to)?;
        let candidate = fit.apply(atom_template);
        let direction = normalize(sub(candidate, parent_actual))?;
        return Some(add(parent_actual, scale(direction, bond)));
    }
    if references.len() == 1 {
        let (first, first_actual) = references[0];
        let first_template = v(template.atoms[first].position);
        let bond_angle = super::geometry::angle(atom_template, parent_template, first_template);
        // A torsion reference two bonds away, inside the template.
        let torsion_reference = adjacency[first]
            .iter()
            .copied()
            .filter(|&n| n != parent && template.atoms[n].element != 1)
            .find_map(|n| placed[n].map(|position| (n, position)));
        if let Some((second, second_actual)) = torsion_reference {
            let torsion = super::geometry::dihedral(
                atom_template,
                parent_template,
                first_template,
                v(template.atoms[second].position),
            );
            return Some(place(
                second_actual,
                first_actual,
                parent_actual,
                bond,
                bond_angle,
                torsion,
            ));
        }
    }
    None
}

/// Keep residue atoms in template order (heavy atoms first as in the input).
fn order_like_template(residue: &mut WResidue, template: &Template) {
    let order = template
        .atoms
        .iter()
        .enumerate()
        .map(|(index, atom)| (atom.name.clone(), index))
        .collect::<HashMap<_, _>>();
    residue
        .atoms
        .sort_by_key(|atom| order.get(&atom.name).copied().unwrap_or(usize::MAX));
}

pub(crate) fn element_symbol(atomic_number: u8) -> &'static str {
    match atomic_number {
        1 => "H",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        11 => "NA",
        12 => "MG",
        15 => "P",
        16 => "S",
        17 => "CL",
        19 => "K",
        20 => "CA",
        26 => "FE",
        29 => "CU",
        30 => "ZN",
        34 => "SE",
        35 => "BR",
        53 => "I",
        _ => "X",
    }
}
