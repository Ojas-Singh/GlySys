//! Carbohydrate residues identified from their atoms, not their names.
//!
//! A residue's heavy-atom graph and the configuration of its stereocentres,
//! read from the coordinates, select the GLYCAM residue type; the bonds it
//! makes to other residues select the linkage form.  Atoms that are not part
//! of the GLYCAM residue split off as the GLYCAM residue for an aglycone
//! (`ROH`, `OME`) or an O-substituent (`SO3`, `MEX`, `ACX`).  Residue and atom
//! names are only a tie-breaker, so a CCD, CHARMM or GLYCAM-named input and
//! an input whose names contradict its coordinates all resolve the same way.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::forcefield::{Template, TemplateSet};
use crate::model::Vec3;
use crate::pdb::PdbResidue;

/// Stereocentres where the GLYCAM library's own coordinates contradict the
/// residue they name, found by reading every template with crabWURCS:
/// GLYCAM-Web's galactofuranose forms (`xLU`, `xLD`) are built with C5
/// inverted, the L-altro configuration.  The centre does not decide the
/// residue (GLYCAM has no altrofuranose and the charges are the Galf ones);
/// an input that follows the library there is reported.
/// (residue suffix, atom, what an input with the library's configuration is)
const LIBRARY_STEREO_ERRORS: &[(&str, &str, &str)] = &[
    (
        "LU",
        "C5",
        "beta-L-altrofuranose, the C5 epimer of beta-D-galactofuranose",
    ),
    (
        "LD",
        "C5",
        "alpha-L-altrofuranose, the C5 epimer of alpha-D-galactofuranose",
    ),
    // Named beta-D-GlcpN and charged differently from YN (alpha), but built alpha.
    (
        "Yn",
        "C1",
        "alpha-D-glucosamine, while this GLYCAM residue is the beta anomer",
    ),
];

/// The residue a wwPDB chemical component names, as a GLYCAM residue code
/// without its linkage letter.  Only used when the coordinates cannot decide
/// (the anomeric oxygen is missing or a centre is too distorted to read).
const CCD_HINTS: &[(&str, &str)] = &[
    ("NAG", "YB"),
    ("NDG", "YA"),
    ("MAN", "MA"),
    ("BMA", "MB"),
    ("GLC", "GA"),
    ("BGC", "GB"),
    ("GAL", "LB"),
    ("GLA", "LA"),
    ("FUC", "fA"),
    ("FUL", "fB"),
    ("XYS", "XA"),
    ("XYP", "XB"),
    ("SIA", "SA"),
    ("SLB", "SB"),
    ("NGC", "GL"),
    ("A2G", "VA"),
    ("NGA", "VB"),
    ("RAM", "hA"),
    ("RM4", "hB"),
    ("FRU", "CU"),
    ("BDF", "CB"),
    ("GCU", "ZA"),
    ("BDP", "ZB"),
    ("IDR", "uA"),
    ("ADA", "OA"),
    ("GTR", "OB"),
    ("MMA", "MA"),
    ("MBG", "LB"),
    ("AMG", "LA"),
    ("GYP", "GA"),
    ("MFU", "fA"),
    ("MFB", "fB"),
    ("MAG", "YB"),
    ("MGC", "VA"),
    ("GZL", "LU"),
    ("ARA", "aA"),
    ("ARB", "aB"),
    ("AHR", "aD"),
    ("FUB", "aU"),
    ("KDN", "KN"),
    ("KDO", "KO"),
    ("Z9N", "MB"),
];

/// Mappings examined per residue type (graph automorphisms are few).
const MAX_MAPPINGS: usize = 64;
/// A chiral volume below this (A^3) is too flat to read a configuration from.
const MIN_CHIRAL_VOLUME: f64 = 0.3;

/// The GLYCAM template for a residue and the residues split off from it.
#[derive(Debug, Clone)]
pub(crate) struct Identified {
    pub template: String,
    /// (index into the residue's atoms, GLYCAM atom name) for the heavy atoms
    /// of the GLYCAM residue.
    pub atoms: Vec<(usize, String)>,
    pub fragments: Vec<Fragment>,
    /// The anomeric configuration could not be read (no atom on the
    /// anomeric carbon) and the residue name decided it.
    pub anomer_from_name: bool,
    /// Stereocentres that follow a known error of the GLYCAM library.
    pub stereo_notes: Vec<String>,
    /// Nothing was bonded to the anomeric carbon, so the ROH oxygen was
    /// placed for the chosen anomer.
    pub rebuilt_anomeric_oxygen: bool,
}

/// An aglycone or substituent carried by the same input residue.
#[derive(Debug, Clone)]
pub(crate) struct Fragment {
    /// GLYCAM residue name: ROH, OME, SO3, MEX or ACX.
    pub residue: &'static str,
    pub atoms: Vec<(usize, String)>,
    /// Atoms the input lacks, placed here: (GLYCAM name, position).
    pub built: Vec<(String, Vec3)>,
    /// GLYCAM names of the bonded pair: (carbohydrate atom, fragment atom).
    pub bond: (String, String),
}

#[derive(Debug)]
pub(crate) struct CarbohydrateLibrary {
    cores: Vec<Core>,
}

/// One GLYCAM residue type with all of its linkage forms.
#[derive(Debug)]
struct Core {
    /// (template name, indices of the linked oxygens)
    forms: Vec<(String, BTreeSet<usize>)>,
    form_info: HashMap<String, FormInfo>,
    atoms: Vec<CoreAtom>,
    neighbors: Vec<Vec<usize>>,
    anomeric: usize,
    /// The anomeric carbon bonds to an atom outside the residue (false when
    /// the residue holds its own glycosidic oxygen).
    open_anomeric: bool,
    centres: Vec<Centre>,
    elements: BTreeMap<u8, usize>,
    /// Atoms in matching order (breadth first from the anomeric carbon).
    order: Vec<usize>,
}

/// What separates forms with the same heavy atoms and configuration.
#[derive(Debug, Clone, Copy)]
struct FormInfo {
    /// Has an amine nitrogen (bonded to no sp2 carbon or sulfur).
    amine: bool,
    /// The amine is an ammonium (Amber type N3).
    ammonium: bool,
    /// A carboxyl group is protonated.
    carboxylic_acid: bool,
    /// Hydrogens on carbon typed against GLYCAM's own rule (Hc, H1, H2 for
    /// zero, one, two bonded O/N); a duplicate entry with fewer is preferred.
    typing_errors: usize,
    /// Taken from GLYCAM-Web's library rather than GLYCAM06j-1.
    glycam_web: bool,
}

#[derive(Debug)]
struct CoreAtom {
    name: String,
    element: u8,
    /// An exocyclic oxygen that is a hydroxyl or a linkage site.
    linkable: bool,
}

#[derive(Debug, Clone, Copy)]
enum Reference {
    Atom(usize),
    /// The atom bonded to the anomeric carbon from outside the residue.
    Outside,
}

#[derive(Debug)]
struct Centre {
    atom: usize,
    references: [Reference; 3],
    sign: f64,
    /// The library's sign here is wrong: what an input with it is.
    library_error: Option<&'static str>,
}

impl CarbohydrateLibrary {
    pub(crate) fn new(templates: &TemplateSet) -> Self {
        let mut grouped: BTreeMap<(String, Vec<(String, u8)>), Vec<&Template>> = BTreeMap::new();
        for template in templates.glycans() {
            let Some(suffix) = template.name.get(1..) else {
                continue;
            };
            let mut heavy = template
                .atoms
                .iter()
                .filter(|atom| atom.element != 1)
                .map(|atom| (atom.name.clone(), atom.element))
                .collect::<Vec<_>>();
            heavy.sort();
            grouped
                .entry((suffix.to_string(), heavy))
                .or_default()
                .push(template);
        }
        let mut cores = Vec::new();
        for forms in grouped.into_values() {
            if let Some(core) = Core::from_forms(&forms, |name| templates.is_glycam_web(name)) {
                cores.push(core);
            }
        }
        Self { cores }
    }

    /// Identify `residue`.  `outside[i]` holds the positions of atoms in other
    /// residues bonded to `residue.atoms[i]`.
    pub(crate) fn identify(
        &self,
        residue: &PdbResidue,
        outside: &[Vec<Vec3>],
    ) -> std::result::Result<Identified, String> {
        let input = InputGraph::new(residue);
        if input.atoms.len() < 5 {
            return Err("too few heavy atoms for a carbohydrate residue".into());
        }
        let mut found: Vec<Candidate> = Vec::new();
        let mut reasons: BTreeSet<String> = BTreeSet::new();
        for (core_index, core) in self.cores.iter().enumerate() {
            if core
                .elements
                .iter()
                .any(|(element, count)| input.elements.get(element).copied().unwrap_or(0) < *count)
            {
                continue;
            }
            for mapping in core.mappings(&input) {
                match core.evaluate(&input, &mapping, residue, outside) {
                    Ok(mut candidate) => {
                        candidate.core = core_index;
                        candidate.index = found.len();
                        found.push(candidate);
                    }
                    Err(reason) => {
                        reasons.insert(reason);
                    }
                }
            }
        }
        if found.is_empty() {
            return Err(if reasons.is_empty() {
                "no GLYCAM residue has this heavy-atom graph (unknown monosaccharide or missing atoms)"
                    .into()
            } else {
                reasons.into_iter().take(3).collect::<Vec<_>>().join("; ")
            });
        }
        // Configurations read from the coordinates beat a name.  Forms that
        // remain (same atoms and configuration) differ in protonation or are
        // duplicate library entries: the residue name, hydrogens in the input
        // (else the ionization at pH 7), correct hydrogen typing, and
        // GLYCAM06j-1 over GLYCAM-Web decide, in that order.
        let name = residue.reference.name.as_str();
        let named_ccd = CCD_HINTS
            .iter()
            .find(|(code, _)| *code == name)
            .map(|(_, suffix)| *suffix);
        let named_glycam = self
            .cores
            .iter()
            .any(|core| core.forms.iter().any(|(form, _)| form == name));
        let hint = named_ccd.or_else(|| named_glycam.then(|| name.get(1..)).flatten());
        let protons = ProtonEvidence::read(residue, &input, &found, &self.cores);
        found.sort_by_key(|candidate| {
            let info = self.cores[candidate.core].form_info[&candidate.form];
            let hinted = candidate.form == name || Some(&candidate.form[1..]) == hint;
            (
                candidate.undetermined && !hinted,
                candidate.form != name,
                !hinted,
                candidate.stereo_notes.len(),
                protons.penalty(candidate, info),
                info.typing_errors,
                info.glycam_web,
                std::cmp::Reverse(candidate.name_agreement),
                candidate.form.len(),
                candidate.form.clone(),
            )
        });
        let best = &found[0];
        let readings = found
            .iter()
            .map(|other| other.form.as_str())
            .collect::<BTreeSet<_>>();
        let hinted = best.form == name || Some(&best.form[1..]) == hint;
        if best.undetermined && !hinted && hint.is_some() {
            return Err(format!(
                "nothing is bonded to the anomeric carbon or a centre is too distorted to read, and the only fitting GLYCAM residue ({}) is not the one the name {name} means",
                best.form
            ));
        }
        if best.undetermined && !hinted && readings.len() > 1 {
            let mut candidates = found
                .iter()
                .map(|other| other.form.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            candidates.truncate(6);
            return Err(format!(
                "nothing is bonded to the anomeric carbon (missing O1?), so the residue cannot be read from the coordinates alone (fits {}); name it with its GLYCAM code or repair it",
                candidates.join(", ")
            ));
        }
        let core = &self.cores[best.core];
        let mut fragments = best.fragments.clone();
        let rebuilt_anomeric_oxygen = core.open_anomeric && !best.anomeric_bonded;
        if rebuilt_anomeric_oxygen {
            let position = core
                .anomeric_oxygen(&best.mapping, &input)
                .ok_or("cannot place the missing anomeric oxygen")?;
            fragments.push(Fragment {
                residue: "ROH",
                atoms: Vec::new(),
                built: vec![("O1".into(), position)],
                bond: (core.atoms[core.anomeric].name.clone(), "O1".into()),
            });
        }
        Ok(Identified {
            template: best.form.clone(),
            atoms: best
                .mapping
                .iter()
                .enumerate()
                .map(|(core_atom, &local)| (input.atoms[local], core.atoms[core_atom].name.clone()))
                .collect(),
            fragments,
            anomer_from_name: best.undetermined,
            stereo_notes: best.stereo_notes.clone(),
            rebuilt_anomeric_oxygen,
        })
    }
}

#[derive(Debug)]
struct Candidate {
    /// Position in the candidate list (for per-candidate evidence).
    index: usize,
    core: usize,
    form: String,
    mapping: Vec<usize>,
    fragments: Vec<Fragment>,
    undetermined: bool,
    /// Something (an aglycone in the residue, or another residue) is
    /// bonded to the anomeric carbon.
    anomeric_bonded: bool,
    name_agreement: usize,
    stereo_notes: Vec<String>,
}

impl Core {
    fn from_forms(forms: &[&Template], glycam_web: impl Fn(&str) -> bool) -> Option<Self> {
        // The form with the fewest linkages carries hydroxyl hydrogens, so
        // its geometry and hydrogen counts describe the residue best.
        let linked = |template: &Template| {
            let heavy = heavy_graph(template);
            heavy.iter().filter(|atom| atom.linked).count()
        };
        let representative = forms.iter().min_by_key(|template| linked(template))?;
        let graph = heavy_graph(representative);
        let ring_oxygen = |index: usize| {
            graph[index]
                .neighbors
                .iter()
                .any(|&next| graph[next].element == 8 && graph[next].neighbors.len() == 2)
        };
        let open = graph
            .iter()
            .enumerate()
            .filter(|(_, atom)| atom.open_carbon)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        // A residue that carries its own glycosidic oxygen (the fructose of
        // sucrose, 2CU) has no open carbon: its anomeric carbon is the ring
        // carbon next to the ring oxygen that holds a linked oxygen.
        let closed = graph
            .iter()
            .enumerate()
            .filter(|(index, atom)| {
                atom.element == 6
                    && ring_oxygen(*index)
                    && atom.neighbors.iter().any(|&next| graph[next].linked)
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let (anomeric, open_anomeric) = match (open.as_slice(), closed.as_slice()) {
            ([single], _) => (*single, true),
            ([], [single]) => (*single, false),
            _ => return None,
        };
        // A glycosyl residue: the anomeric carbon is in a ring through an oxygen.
        if !ring_oxygen(anomeric) {
            return None;
        }
        let index_of = graph
            .iter()
            .enumerate()
            .map(|(index, atom)| (atom.name.clone(), index))
            .collect::<HashMap<_, _>>();
        let mut all_forms = Vec::new();
        let mut form_info = HashMap::new();
        let mut linkable = vec![false; graph.len()];
        for template in forms {
            form_info.insert(
                template.name.clone(),
                form_info_of(template, glycam_web(&template.name)),
            );
            let form_graph = heavy_graph(template);
            let mut linked_set = BTreeSet::new();
            for atom in &form_graph {
                let &index = index_of.get(&atom.name)?;
                if atom.linked {
                    linked_set.insert(index);
                    linkable[index] = true;
                }
                if atom.hydroxyl {
                    linkable[index] = true;
                }
            }
            all_forms.push((template.name.clone(), linked_set));
        }
        all_forms.sort();
        let neighbors = graph
            .iter()
            .map(|atom| atom.neighbors.clone())
            .collect::<Vec<_>>();
        let mut centres = Vec::new();
        for (index, atom) in graph.iter().enumerate() {
            if atom.element != 6 || atom.sp2 || atom.hydrogens.len() > 1 {
                continue;
            }
            let open = index == anomeric && open_anomeric;
            if atom.neighbors.len() + atom.hydrogens.len() + usize::from(open) != 4 {
                continue;
            }
            let mut heavy = atom.neighbors.clone();
            heavy.sort_unstable();
            let (references, sign) = if heavy.len() >= 3 {
                let references = [
                    Reference::Atom(heavy[0]),
                    Reference::Atom(heavy[1]),
                    Reference::Atom(heavy[2]),
                ];
                let volume = chiral_volume(
                    atom.position,
                    graph[heavy[0]].position,
                    graph[heavy[1]].position,
                    graph[heavy[2]].position,
                );
                (references, volume)
            } else if heavy.len() == 2 && open && atom.hydrogens.len() == 1 {
                // Swapping the outside atom for the hydrogen inverts the
                // handedness of the reference triple.
                let volume = -chiral_volume(
                    atom.position,
                    graph[heavy[0]].position,
                    graph[heavy[1]].position,
                    atom.hydrogens[0],
                );
                (
                    [
                        Reference::Atom(heavy[0]),
                        Reference::Atom(heavy[1]),
                        Reference::Outside,
                    ],
                    volume,
                )
            } else {
                continue;
            };
            if sign.abs() < 1.0e-3 {
                continue;
            }
            let suffix = representative.name.get(1..).unwrap_or_default();
            let library_error = LIBRARY_STEREO_ERRORS
                .iter()
                .find(|(residue, centre, _)| *residue == suffix && *centre == atom.name)
                .map(|(_, _, meaning)| *meaning);
            centres.push(Centre {
                atom: index,
                references,
                sign: sign.signum(),
                library_error,
            });
        }
        let mut elements = BTreeMap::new();
        for atom in &graph {
            *elements.entry(atom.element).or_insert(0) += 1;
        }
        let mut order = vec![anomeric];
        let mut seen = vec![false; graph.len()];
        seen[anomeric] = true;
        let mut cursor = 0;
        while cursor < order.len() {
            let current = order[cursor];
            cursor += 1;
            for &next in &neighbors[current] {
                if !seen[next] {
                    seen[next] = true;
                    order.push(next);
                }
            }
        }
        if order.len() != graph.len() {
            return None; // disconnected template
        }
        Some(Self {
            forms: all_forms,
            form_info,
            atoms: graph
                .iter()
                .enumerate()
                .map(|(index, atom)| CoreAtom {
                    name: atom.name.clone(),
                    element: atom.element,
                    linkable: linkable[index],
                })
                .collect(),
            neighbors,
            anomeric,
            open_anomeric,
            centres,
            elements,
            order,
        })
    }

    /// Where the anomeric oxygen of this form goes when the input lacks it:
    /// the free tetrahedral position on the anomeric carbon that gives the
    /// form's configuration.
    fn anomeric_oxygen(&self, mapping: &[usize], input: &InputGraph) -> Option<Vec3> {
        let centre = input.positions[mapping[self.anomeric]];
        let bonded = self.neighbors[self.anomeric]
            .iter()
            .map(|&core_atom| input.positions[mapping[core_atom]])
            .collect::<Vec<_>>();
        let units = bonded
            .iter()
            .map(|p| normalized([p.x - centre.x, p.y - centre.y, p.z - centre.z]))
            .collect::<Option<Vec<_>>>()?;
        let away = normalized([
            -units.iter().map(|u| u[0]).sum::<f64>(),
            -units.iter().map(|u| u[1]).sum::<f64>(),
            -units.iter().map(|u| u[2]).sum::<f64>(),
        ])?;
        let place = |direction: [f64; 3]| Vec3 {
            x: centre.x + 1.43 * direction[0],
            y: centre.y + 1.43 * direction[1],
            z: centre.z + 1.43 * direction[2],
        };
        if units.len() != 2 {
            return Some(place(away));
        }
        // Two free positions either side of the plane of the two bonds.
        let normal = normalized(cross(units[0], units[1]))?;
        let (cos, sin) = (54.75f64.to_radians().cos(), 54.75f64.to_radians().sin());
        let centre_ref = self
            .centres
            .iter()
            .find(|centre_item| centre_item.atom == self.anomeric)?;
        [1.0, -1.0].into_iter().find_map(|side| {
            let direction = normalized([
                cos * away[0] + side * sin * normal[0],
                cos * away[1] + side * sin * normal[1],
                cos * away[2] + side * sin * normal[2],
            ])?;
            let candidate = place(direction);
            let position = |reference: Reference| match reference {
                Reference::Atom(core_atom) => input.positions[mapping[core_atom]],
                Reference::Outside => candidate,
            };
            let volume = chiral_volume(
                centre,
                position(centre_ref.references[0]),
                position(centre_ref.references[1]),
                position(centre_ref.references[2]),
            );
            (volume.signum() == centre_ref.sign).then_some(candidate)
        })
    }

    /// Induced subgraph isomorphisms of this residue type into the input.
    fn mappings(&self, input: &InputGraph) -> Vec<Vec<usize>> {
        let mut found = Vec::new();
        let mut mapping = vec![usize::MAX; self.atoms.len()];
        let mut used = vec![false; input.atoms.len()];
        self.extend(input, 0, &mut mapping, &mut used, &mut found);
        found
    }

    fn extend(
        &self,
        input: &InputGraph,
        depth: usize,
        mapping: &mut Vec<usize>,
        used: &mut Vec<bool>,
        found: &mut Vec<Vec<usize>>,
    ) {
        if found.len() >= MAX_MAPPINGS {
            return;
        }
        if depth == self.order.len() {
            found.push(mapping.clone());
            return;
        }
        let atom = self.order[depth];
        let element = self.atoms[atom].element;
        let degree = self.neighbors[atom].len();
        'candidates: for local in 0..input.atoms.len() {
            if used[local]
                || input.elements_by_atom[local] != element
                || input.neighbors[local].len() < degree
            {
                continue;
            }
            // Every placed template atom must be bonded to `local` exactly
            // when it is bonded to `atom` (an induced match).
            for (other, &placed) in mapping.iter().enumerate() {
                if placed == usize::MAX {
                    continue;
                }
                let template_bond = self.neighbors[atom].contains(&other);
                let input_bond = input.neighbors[local].contains(&placed);
                if template_bond != input_bond {
                    continue 'candidates;
                }
            }
            mapping[atom] = local;
            used[local] = true;
            self.extend(input, depth + 1, mapping, used, found);
            mapping[atom] = usize::MAX;
            used[local] = false;
        }
    }

    fn evaluate(
        &self,
        input: &InputGraph,
        mapping: &[usize],
        residue: &PdbResidue,
        outside: &[Vec<Vec3>],
    ) -> std::result::Result<Candidate, String> {
        let mut owner = vec![usize::MAX; input.atoms.len()];
        for (core_atom, &local) in mapping.iter().enumerate() {
            owner[local] = core_atom;
        }
        // Atoms outside the GLYCAM residue, grouped into connected fragments.
        let mut fragment_of = vec![usize::MAX; input.atoms.len()];
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for start in 0..input.atoms.len() {
            if owner[start] != usize::MAX || fragment_of[start] != usize::MAX {
                continue;
            }
            let id = groups.len();
            let mut stack = vec![start];
            fragment_of[start] = id;
            let mut members = Vec::new();
            while let Some(local) = stack.pop() {
                members.push(local);
                for &next in &input.neighbors[local] {
                    if owner[next] == usize::MAX && fragment_of[next] == usize::MAX {
                        fragment_of[next] = id;
                        stack.push(next);
                    }
                }
            }
            groups.push(members);
        }
        let outside_of = |local: usize| &outside[input.atoms[local]];
        let mut fragments = Vec::new();
        // The atom bonded to each core atom from outside the GLYCAM residue.
        let mut partner: Vec<Vec<Vec3>> = mapping
            .iter()
            .map(|&local| outside_of(local).clone())
            .collect();
        for members in &groups {
            let mut attachments = Vec::new();
            for &local in members {
                for &next in &input.neighbors[local] {
                    if owner[next] != usize::MAX {
                        attachments.push((local, owner[next]));
                    }
                }
            }
            let &[(fragment_atom, core_atom)] = attachments.as_slice() else {
                return Err(format!(
                    "{} atoms not in any GLYCAM residue are bonded to it at {} points",
                    members.len(),
                    attachments.len()
                ));
            };
            if members.iter().any(|local| !outside_of(*local).is_empty()) {
                return Err("an aglycone or substituent bonds to another residue".into());
            }
            let on_anomeric_oxygen = !self.open_anomeric
                && self.neighbors[self.anomeric].contains(&core_atom)
                && self.atoms[core_atom].linkable;
            let fragment = if core_atom == self.anomeric {
                classify_aglycone(input, members, fragment_atom)
            } else if on_anomeric_oxygen {
                // A group on the anomeric oxygen is an aglycone (OME, not
                // MEX on a 1-linked form): the open form reads it.
                None
            } else if self.atoms[core_atom].linkable {
                classify_substituent(input, members, fragment_atom)
            } else {
                None
            }
            .ok_or_else(|| {
                format!(
                    "a group of {} heavy atoms on {} has no GLYCAM residue",
                    members.len(),
                    self.atoms[core_atom].name
                )
            })?;
            partner[core_atom].push(input.positions[fragment_atom]);
            let (residue_name, named, fragment_bond_atom) = fragment;
            fragments.push(Fragment {
                residue: residue_name,
                built: Vec::new(),
                atoms: named
                    .into_iter()
                    .map(|(local, name)| (input.atoms[local], name.to_string()))
                    .collect(),
                bond: (
                    self.atoms[core_atom].name.clone(),
                    fragment_bond_atom.to_string(),
                ),
            });
        }
        let mut linked = BTreeSet::new();
        for (core_atom, partners) in partner.iter().enumerate() {
            if partners.is_empty() || core_atom == self.anomeric {
                continue;
            }
            if !self.atoms[core_atom].linkable {
                return Err(format!(
                    "{} is bonded to another residue, which no GLYCAM form allows",
                    self.atoms[core_atom].name
                ));
            }
            linked.insert(core_atom);
        }
        if partner[self.anomeric].len() > usize::from(self.open_anomeric) {
            return Err("the anomeric carbon has more than one outside bond".into());
        }
        let form = self
            .forms
            .iter()
            .find(|(_, set)| *set == linked)
            .map(|(name, _)| name.clone())
            .ok_or_else(|| {
                format!(
                    "GLYCAM has {} but no form linked at {}",
                    self.forms[0].0,
                    linked
                        .iter()
                        .map(|index| self.atoms[*index].name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
        // Without an atom on the anomeric carbon the residue is not fully
        // read even when no centre needs it: an aldose that lost O1 has the
        // atom graph of a ketose, so neither reading may win on its own.
        let mut undetermined = self.open_anomeric && partner[self.anomeric].is_empty();
        let mut stereo_notes = Vec::new();
        for centre in &self.centres {
            let position = |reference: Reference| match reference {
                Reference::Atom(core_atom) => Some(input.positions[mapping[core_atom]]),
                Reference::Outside => partner[centre.atom].first().copied(),
            };
            let (Some(a), Some(b), Some(c)) = (
                position(centre.references[0]),
                position(centre.references[1]),
                position(centre.references[2]),
            ) else {
                undetermined = true;
                continue;
            };
            let volume = chiral_volume(input.positions[mapping[centre.atom]], a, b, c);
            if volume.abs() < MIN_CHIRAL_VOLUME {
                undetermined = true; // too distorted to read
                continue;
            }
            match centre.library_error {
                Some(meaning) if volume.signum() == centre.sign => stereo_notes.push(format!(
                    "{} is in the configuration of GLYCAM's library coordinates, which are inverted there: the coordinates are {meaning}",
                    self.atoms[centre.atom].name
                )),
                Some(_) => {}
                None if volume.signum() != centre.sign => {
                    return Err(format!("{} configuration differs", self.forms[0].0));
                }
                None => {}
            }
        }
        let name_agreement = mapping
            .iter()
            .enumerate()
            .filter(|(core_atom, local)| {
                residue.atoms[input.atoms[**local]].name == self.atoms[*core_atom].name
            })
            .count();
        Ok(Candidate {
            index: usize::MAX,
            core: usize::MAX,
            form,
            mapping: mapping.to_vec(),
            fragments,
            undetermined,
            anomeric_bonded: !partner[self.anomeric].is_empty(),
            name_agreement,
            stereo_notes,
        })
    }
}

fn form_info_of(template: &Template, glycam_web: bool) -> FormInfo {
    let mut neighbors = vec![Vec::new(); template.atoms.len()];
    for &[first, second] in &template.bonds {
        neighbors[first].push(second);
        neighbors[second].push(first);
    }
    let atom = |index: usize| &template.atoms[index];
    let mut info = FormInfo {
        amine: false,
        ammonium: false,
        carboxylic_acid: false,
        typing_errors: 0,
        glycam_web,
    };
    for (index, item) in template.atoms.iter().enumerate() {
        match item.element {
            7 => {
                let conjugated = neighbors[index].iter().any(|&next| {
                    (atom(next).element == 6 && atom(next).atom_type == "C")
                        || atom(next).element == 16
                });
                if !conjugated {
                    info.amine = true;
                    info.ammonium |= item.atom_type == "N3";
                }
            }
            8 => {
                let on_carboxyl = neighbors[index]
                    .iter()
                    .any(|&next| atom(next).element == 6 && atom(next).atom_type == "C");
                let protonated = neighbors[index].iter().any(|&next| atom(next).element == 1);
                info.carboxylic_acid |= on_carboxyl && protonated;
            }
            1 => {
                let Some(&carbon) = neighbors[index]
                    .iter()
                    .find(|&&next| atom(next).element == 6)
                else {
                    continue;
                };
                let polar = neighbors[carbon]
                    .iter()
                    .filter(|&&next| matches!(atom(next).element, 7 | 8))
                    .count();
                let expected = match polar {
                    0 => "Hc",
                    1 => "H1",
                    _ => "H2",
                };
                if atom(carbon).atom_type != "C" && item.atom_type != expected {
                    info.typing_errors += 1;
                }
            }
            _ => {}
        }
    }
    info
}

/// Hydrogens the input itself puts on amine nitrogens and carboxyl oxygens.
struct ProtonEvidence {
    any_hydrogens: bool,
    /// Per candidate: most hydrogens on one amine N, and whether a carboxyl
    /// oxygen carries one.
    per_candidate: Vec<(usize, bool)>,
}

impl ProtonEvidence {
    fn read(
        residue: &PdbResidue,
        input: &InputGraph,
        candidates: &[Candidate],
        cores: &[Core],
    ) -> Self {
        let hydrogens = residue
            .atoms
            .iter()
            .filter(|atom| atomic_number(&atom.element) == 1)
            .map(|atom| atom.position)
            .collect::<Vec<_>>();
        let count_near = |position: Vec3| {
            hydrogens
                .iter()
                .filter(|hydrogen| hydrogen.distance2(position) < 1.25 * 1.25)
                .count()
        };
        let per_candidate = candidates
            .iter()
            .map(|candidate| {
                let core = &cores[candidate.core];
                let mut amine_h = 0;
                let mut acid = false;
                for (core_atom, &local) in candidate.mapping.iter().enumerate() {
                    let position = input.positions[local];
                    match core.atoms[core_atom].element {
                        7 => amine_h = amine_h.max(count_near(position)),
                        8 if core.neighbors[core_atom].len() == 1 => {
                            let carbon = core.neighbors[core_atom][0];
                            let carboxyl = core.atoms[carbon].element == 6
                                && core.neighbors[carbon]
                                    .iter()
                                    .filter(|next| core.atoms[**next].element == 8)
                                    .count()
                                    == 2
                                && core.neighbors[carbon].len() == 3;
                            acid |= carboxyl && count_near(position) > 0;
                        }
                        _ => {}
                    }
                }
                (amine_h, acid)
            })
            .collect();
        Self {
            any_hydrogens: !hydrogens.is_empty(),
            per_candidate,
        }
    }

    fn penalty(&self, candidate: &Candidate, info: FormInfo) -> usize {
        let index = candidate.index;
        let (amine_h, acid) = self.per_candidate[index];
        let (want_ammonium, want_acid) = if self.any_hydrogens {
            (amine_h >= 3, acid)
        } else {
            (true, false) // pH 7: amines protonated, carboxyl groups not
        };
        usize::from(info.amine && info.ammonium != want_ammonium)
            + usize::from(info.carboxylic_acid != want_acid)
    }
}

type Classified = (&'static str, Vec<(usize, &'static str)>, &'static str);

/// O1 (ROH) or O-methyl (OME) on the anomeric carbon.
fn classify_aglycone(input: &InputGraph, members: &[usize], attached: usize) -> Option<Classified> {
    let element = |local: usize| input.elements_by_atom[local];
    match members.len() {
        1 if element(attached) == 8 => Some(("ROH", vec![(attached, "O1")], "O1")),
        2 if element(attached) == 8 => {
            let carbon = *members.iter().find(|local| **local != attached)?;
            (element(carbon) == 6).then(|| ("OME", vec![(attached, "O"), (carbon, "CH3")], "O"))
        }
        _ => None,
    }
}

/// O-sulfate (SO3), O-methyl (MEX) or O-acetyl (ACX) on a hydroxyl oxygen.
fn classify_substituent(
    input: &InputGraph,
    members: &[usize],
    attached: usize,
) -> Option<Classified> {
    let element = |local: usize| input.elements_by_atom[local];
    let degree = |local: usize| input.neighbors[local].len();
    match members.len() {
        1 if element(attached) == 6 => Some(("MEX", vec![(attached, "CH3")], "CH3")),
        4 if element(attached) == 16 => {
            let oxygens = members
                .iter()
                .copied()
                .filter(|local| *local != attached)
                .collect::<Vec<_>>();
            oxygens
                .iter()
                .all(|local| element(*local) == 8 && degree(*local) == 1)
                .then(|| {
                    (
                        "SO3",
                        vec![
                            (attached, "S1"),
                            (oxygens[0], "O1"),
                            (oxygens[1], "O2"),
                            (oxygens[2], "O3"),
                        ],
                        "S1",
                    )
                })
        }
        3 if element(attached) == 6 => {
            let others = members
                .iter()
                .copied()
                .filter(|local| *local != attached)
                .collect::<Vec<_>>();
            let oxygen = *others.iter().find(|local| element(**local) == 8)?;
            let methyl = *others.iter().find(|local| element(**local) == 6)?;
            (degree(oxygen) == 1 && degree(methyl) == 1).then(|| {
                (
                    "ACX",
                    vec![(attached, "C1A"), (oxygen, "O1A"), (methyl, "C2A")],
                    "C1A",
                )
            })
        }
        _ => None,
    }
}

/// Heavy atoms of a template with their bonding.
struct GraphAtom {
    name: String,
    element: u8,
    position: Vec3,
    neighbors: Vec<usize>,
    hydrogens: Vec<Vec3>,
    sp2: bool,
    /// The one carbon with a free valence: the anomeric carbon.
    open_carbon: bool,
    /// An exocyclic oxygen without hydrogen typed as an ether: a linkage.
    linked: bool,
    hydroxyl: bool,
}

fn heavy_graph(template: &Template) -> Vec<GraphAtom> {
    let heavy = template
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, atom)| atom.element != 1)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let local = heavy
        .iter()
        .enumerate()
        .map(|(position, &index)| (index, position))
        .collect::<HashMap<_, _>>();
    let mut atoms = heavy
        .iter()
        .map(|&index| {
            let atom = &template.atoms[index];
            GraphAtom {
                name: atom.name.clone(),
                element: atom.element,
                position: atom.position,
                neighbors: Vec::new(),
                hydrogens: Vec::new(),
                // GLYCAM types sp2 carbons (carbonyl, carboxylate) "C".
                sp2: atom.atom_type == "C",
                open_carbon: false,
                linked: false,
                hydroxyl: false,
            }
        })
        .collect::<Vec<_>>();
    for &[first, second] in &template.bonds {
        match (local.get(&first), local.get(&second)) {
            (Some(&a), Some(&b)) => {
                atoms[a].neighbors.push(b);
                atoms[b].neighbors.push(a);
            }
            (Some(&a), None) => atoms[a].hydrogens.push(template.atoms[second].position),
            (None, Some(&b)) => atoms[b].hydrogens.push(template.atoms[first].position),
            (None, None) => {}
        }
    }
    for (position, &index) in heavy.iter().enumerate() {
        let atom_type = template.atoms[index].atom_type.as_str();
        let atom = &mut atoms[position];
        let valence = atom.neighbors.len() + atom.hydrogens.len();
        atom.open_carbon = atom.element == 6 && !atom.sp2 && valence == 3;
        atom.linked = atom.element == 8
            && atom_type == "Os"
            && atom.neighbors.len() == 1
            && atom.hydrogens.is_empty();
        atom.hydroxyl = atom.element == 8 && atom.neighbors.len() == 1 && atom.hydrogens.len() == 1;
    }
    atoms
}

/// Heavy atoms of an input residue bonded by distance.
struct InputGraph {
    /// Indices into the residue's atoms.
    atoms: Vec<usize>,
    positions: Vec<Vec3>,
    elements_by_atom: Vec<u8>,
    neighbors: Vec<Vec<usize>>,
    elements: BTreeMap<u8, usize>,
}

impl InputGraph {
    fn new(residue: &PdbResidue) -> Self {
        let mut atoms = Vec::new();
        let mut positions = Vec::new();
        let mut elements_by_atom = Vec::new();
        for (index, atom) in residue.atoms.iter().enumerate() {
            let element = atomic_number(&atom.element);
            if element == 1 || element == 0 {
                continue;
            }
            atoms.push(index);
            positions.push(atom.position);
            elements_by_atom.push(element);
        }
        let mut neighbors = vec![Vec::new(); atoms.len()];
        for first in 0..atoms.len() {
            for second in first + 1..atoms.len() {
                let limit =
                    inter_residue_bond_limit(elements_by_atom[first], elements_by_atom[second]);
                if positions[first].distance2(positions[second]) < limit * limit {
                    neighbors[first].push(second);
                    neighbors[second].push(first);
                }
            }
        }
        // A distorted deposited model can bring a 1-3 pair within bonding
        // distance (an acetamide O near N); no monosaccharide has a
        // three-membered ring, so drop the longest edge of any triangle.
        loop {
            let mut longest: Option<(f64, usize, usize)> = None;
            for a in 0..atoms.len() {
                for &b in &neighbors[a] {
                    for &c in &neighbors[b] {
                        if b > a && c > b && neighbors[c].contains(&a) {
                            for (x, y) in [(a, b), (b, c), (a, c)] {
                                let d = positions[x].distance2(positions[y]);
                                if longest.is_none_or(|(best, _, _)| d > best) {
                                    longest = Some((d, x, y));
                                }
                            }
                        }
                    }
                }
            }
            let Some((_, x, y)) = longest else {
                break;
            };
            neighbors[x].retain(|other| *other != y);
            neighbors[y].retain(|other| *other != x);
        }
        let mut elements = BTreeMap::new();
        for element in &elements_by_atom {
            *elements.entry(*element).or_insert(0) += 1;
        }
        Self {
            atoms,
            positions,
            elements_by_atom,
            neighbors,
            elements,
        }
    }
}

/// Bond-length cutoff between atoms of different residues.
pub(crate) fn inter_residue_bond_limit(first: u8, second: u8) -> f64 {
    covalent_radius(first) + covalent_radius(second) + 0.25
}

pub(crate) fn covalent_radius(element: u8) -> f64 {
    match element {
        1 => 0.31,
        5 => 0.84,
        6 => 0.76,
        7 => 0.71,
        8 => 0.66,
        9 => 0.57,
        14 => 1.11,
        15 => 1.07,
        16 => 1.05,
        17 => 1.02,
        34 => 1.20,
        35 => 1.20,
        53 => 1.39,
        _ => 1.2,
    }
}

pub(crate) fn atomic_number(element: &str) -> u8 {
    match element.trim().to_ascii_uppercase().as_str() {
        "H" | "D" => 1,
        "B" => 5,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        "NA" => 11,
        "MG" => 12,
        "SI" => 14,
        "P" => 15,
        "S" => 16,
        "CL" => 17,
        "K" => 19,
        "CA" => 20,
        "MN" => 25,
        "FE" => 26,
        "CO" => 27,
        "NI" => 28,
        "CU" => 29,
        "ZN" => 30,
        "SE" => 34,
        "BR" => 35,
        "CD" => 48,
        "I" => 53,
        "HG" => 80,
        _ => 0,
    }
}

fn normalized(v: [f64; 3]) -> Option<[f64; 3]> {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (length > 1.0e-9).then(|| [v[0] / length, v[1] / length, v[2] / length])
}

fn cross(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

fn chiral_volume(centre: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let u = [a.x - centre.x, a.y - centre.y, a.z - centre.z];
    let v = [b.x - centre.x, b.y - centre.y, b.z - centre.z];
    let w = [c.x - centre.x, c.y - centre.y, c.z - centre.z];
    u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdb::PdbAtom;
    use crate::report::ResidueRef;

    fn element_symbol(element: u8) -> &'static str {
        match element {
            6 => "C",
            7 => "N",
            8 => "O",
            15 => "P",
            16 => "S",
            _ => "X",
        }
    }

    /// A template's heavy atoms as a PDB residue, with an outside atom on the
    /// anomeric carbon and on every linked oxygen.
    fn residue_from_template(
        template: &Template,
        rename: impl Fn(&str) -> String,
    ) -> (PdbResidue, Vec<Vec<Vec3>>) {
        let graph = heavy_graph(template);
        let mut atoms = Vec::new();
        let mut outside = Vec::new();
        for atom in &graph {
            atoms.push(PdbAtom {
                serial: atoms.len() as u32 + 1,
                name: rename(&atom.name),
                residue_name: "UNK".into(),
                chain: "A".into(),
                residue_number: 1,
                insertion_code: None,
                element: element_symbol(atom.element).into(),
                occupancy: 1.0,
                b_factor: 0.0,
                position: atom.position,
            });
            let mut partners = Vec::new();
            if atom.open_carbon || atom.linked {
                // Continue away from the bonded heavy atoms and hydrogens.
                let mut direction = [0.0; 3];
                for other in atom
                    .neighbors
                    .iter()
                    .map(|index| graph[*index].position)
                    .chain(atom.hydrogens.iter().copied())
                {
                    let d = [
                        other.x - atom.position.x,
                        other.y - atom.position.y,
                        other.z - atom.position.z,
                    ];
                    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    for k in 0..3 {
                        direction[k] -= d[k] / n;
                    }
                }
                let n = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
                partners.push(Vec3 {
                    x: atom.position.x + 1.43 * direction[0] / n,
                    y: atom.position.y + 1.43 * direction[1] / n,
                    z: atom.position.z + 1.43 * direction[2] / n,
                });
            }
            outside.push(partners);
        }
        let residue = PdbResidue {
            reference: ResidueRef {
                chain: "A".into(),
                name: "UNK".into(),
                number: 1,
                insertion_code: None,
            },
            atoms,
        };
        (residue, outside)
    }

    #[test]
    fn every_glycam_residue_is_read_back_from_its_own_coordinates() {
        let mut templates = TemplateSet::load().unwrap();
        let parameters = crate::forcefield::ParameterSet::load().unwrap();
        templates.retain_parameterized_glycans(|template| parameters.covers(template));
        let library = CarbohydrateLibrary::new(&templates);
        let known = library
            .cores
            .iter()
            .flat_map(|core| core.forms.iter().map(|(name, _)| name.clone()))
            .collect::<BTreeSet<_>>();
        let mut failures = Vec::new();
        let mut checked = 0;
        for name in &known {
            let name = name.as_str();
            let template = templates.glycan(name).unwrap();
            checked += 1;
            // Unnamed, the coordinates alone must resolve to a residue; named,
            // the name must pick this form among chemically identical ones.
            let (mut residue, outside) = residue_from_template(template, |name| format!("X{name}"));
            if let Err(reason) = library.identify(&residue, &outside) {
                failures.push(format!("{name} unnamed: {reason}"));
            }
            residue.reference.name = name.to_string();
            match library.identify(&residue, &outside) {
                Ok(found) if found.template == name => {}
                Ok(found) => failures.push(format!("{name} -> {}", found.template)),
                Err(reason) => failures.push(format!("{name}: {reason}")),
            }
        }
        eprintln!("checked {checked}, failures {}", failures.len());
        for failure in &failures {
            eprintln!("  {failure}");
        }
        assert!(failures.is_empty());
    }
}
