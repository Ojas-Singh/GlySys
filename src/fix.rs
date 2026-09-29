//! Structure repair: a pure-Rust counterpart of OpenMM PDBFixer.
//!
//! The fixer reads any PDB model, resolves alternate locations, replaces
//! modified residues by their standard parents, models residues that SEQRES
//! lists but the coordinates lack, rebuilds missing heavy atoms, relieves the
//! clashes this creates, assigns protonation states for a pH and adds
//! hydrogens to proteins, nucleic acids, glycans, waters and — when their
//! Chemical Component Dictionary definition is supplied — ligands.  Nothing
//! is parameterized, so unusual chemistry is kept instead of rejected.
//!
//! The command line and the browser worker share this entry point, so both
//! always apply the same repairs.

mod chemistry;
mod complete;
mod components;
mod geometry;
mod hydrogens;
mod loops;
mod output;
mod protonation;
mod relax;
mod work;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use sha2::{Digest, Sha256};

pub use components::{Component, ComponentAtom, ComponentBond, ComponentLibrary};
pub use loops::MissingResidues;
pub use output::Naming;
pub use work::ResidueKind;

use crate::forcefield::{Template, TemplateSet};
use crate::pdb::{self, PdbAtom, PdbResidue};
use crate::report::ResidueRef;
use crate::{BuildError, ProtonationOverrides, Result, SystemBuilder};
use geometry::{Grid, V, distance, p};
use hydrogens::{Environment, HTemplate, Phase};
use work::{Origin, WResidue, Work};

/// What the fixer is allowed to change.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FixOptions {
    /// PDB MODEL number.
    pub model: u32,
    /// Preferred alternate location; otherwise `A`, then highest occupancy.
    pub altloc: Option<char>,
    /// pH used for side-chain protonation states.
    pub ph: f64,
    /// Model residues listed in SEQRES but absent from the coordinates.
    pub missing_residues: MissingResidues,
    /// Replace modified residues (MSE, SEP, PTR, ...) by their standard parent.
    pub replace_nonstandard: bool,
    /// Rebuild missing heavy atoms of standard residues.
    pub add_missing_atoms: bool,
    pub add_hydrogens: bool,
    pub keep_water: bool,
    /// Keep ligands, ions and other heterogens.
    pub keep_heterogens: bool,
    /// Keep carbohydrate residues (glycans).
    pub keep_glycans: bool,
    /// Relieve clashes of rebuilt and modelled atoms.
    pub relax: bool,
    pub naming: Naming,
    pub protonation: ProtonationOverrides,
}

impl Default for FixOptions {
    fn default() -> Self {
        Self {
            model: 1,
            altloc: None,
            ph: 7.0,
            missing_residues: MissingResidues::All,
            replace_nonstandard: true,
            add_missing_atoms: true,
            add_hydrogens: true,
            keep_water: true,
            keep_heterogens: true,
            keep_glycans: true,
            relax: true,
            naming: Naming::Pdb,
            protonation: ProtonationOverrides::default(),
        }
    }
}

impl FixOptions {
    fn validate(&self) -> Result<()> {
        if self.model == 0 {
            return Err(BuildError::InvalidOption(
                "PDB model numbers start at 1".into(),
            ));
        }
        if !self.ph.is_finite() || !(0.0..=14.0).contains(&self.ph) {
            return Err(BuildError::InvalidOption(
                "pH must be between 0 and 14".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ResidueCounts {
    pub protein: usize,
    pub nucleic: usize,
    pub glycan: usize,
    pub water: usize,
    pub ion: usize,
    pub ligand: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MissingSegment {
    pub chain: String,
    /// Observed residue before the segment (`None` at the N-terminus).
    pub after: Option<String>,
    /// Observed residue after the segment (`None` at the C-terminus).
    pub before: Option<String>,
    pub residues: Vec<String>,
    pub modelled: bool,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReplacedResidue {
    pub residue: String,
    pub from: String,
    pub to: String,
    pub removed_atoms: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ResidueAtoms {
    pub residue: String,
    pub atoms: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProtonationChoice {
    pub residue: String,
    pub state: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HeterogenSummary {
    pub name: String,
    pub kind: ResidueKind,
    pub count: usize,
    /// `ccd`, `glycam`, `template`, `water` or `none`.
    pub hydrogens: String,
}

/// Everything the fixer changed, for users and for downstream tools.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FixReport {
    pub glysys_version: String,
    pub input_sha256: String,
    pub output_sha256: String,
    pub options: FixOptions,
    pub chains: Vec<String>,
    pub residue_counts: ResidueCounts,
    pub atoms_in: usize,
    pub atoms_out: usize,
    pub heavy_atoms_added: usize,
    pub hydrogens_added: usize,
    pub residues_added: usize,
    pub missing_residues: Vec<MissingSegment>,
    pub replaced_residues: Vec<ReplacedResidue>,
    pub rebuilt_atoms: Vec<ResidueAtoms>,
    pub removed_atoms: Vec<ResidueAtoms>,
    pub removed_residues: BTreeMap<String, usize>,
    pub protonation: Vec<ProtonationChoice>,
    pub disulfides: Vec<String>,
    pub chain_breaks: Vec<String>,
    pub heterogens: Vec<HeterogenSummary>,
    /// Residue names whose CCD definition would enable hydrogens/replacement.
    pub components_missing: Vec<String>,
    pub unresolved: Vec<String>,
    /// Heavy-atom contacts under 2.2 Å involving rebuilt or modelled atoms.
    pub remaining_clashes: usize,
    /// Inter-residue contacts under 1.5 Å involving a hydrogen, after relaxation.
    pub hydrogen_contacts: usize,
    pub warnings: Vec<String>,
}

/// A repaired structure and the report describing what was changed.
#[derive(Debug, Clone)]
pub struct FixedStructure {
    pub pdb: String,
    pub report: FixReport,
}

/// Reusable structure fixer with loaded residue libraries.
#[derive(Debug)]
pub struct StructureFixer {
    options: FixOptions,
    templates: TemplateSet,
    components: ComponentLibrary,
}

impl SystemBuilder {
    /// Repair a PDB held in memory with the builder's model/altloc/protonation choices.
    pub fn fix_pdb_str(&self, contents: &str) -> Result<FixedStructure> {
        let options = FixOptions {
            model: self.options().model,
            altloc: self.options().altloc,
            protonation: self.options().protonation.clone(),
            ..FixOptions::default()
        };
        StructureFixer::new(options)?.fix_pdb_str(contents)
    }
}

impl StructureFixer {
    pub fn new(options: FixOptions) -> Result<Self> {
        options.validate()?;
        Ok(Self {
            options,
            templates: TemplateSet::load()?,
            components: ComponentLibrary::new(),
        })
    }

    pub fn options(&self) -> &FixOptions {
        &self.options
    }

    pub fn with_components(mut self, components: ComponentLibrary) -> Self {
        self.components = components;
        self
    }

    pub fn components_mut(&mut self) -> &mut ComponentLibrary {
        &mut self.components
    }

    /// Residue names in `contents` whose CCD definitions the fixer can use:
    /// ligands and modified residues without a built-in template.
    pub fn component_requests(&self, contents: &str) -> Result<Vec<String>> {
        let raw = pdb::read_model(contents, self.options.model, self.options.altloc)?;
        let (work, _) = work::load(raw, &self.templates, &self.components);
        let mut names = work
            .residues
            .iter()
            .filter(|residue| {
                matches!(residue.kind, ResidueKind::Ligand | ResidueKind::Glycan)
                    && self.components.get(&residue.input_name).is_none()
            })
            .map(|residue| residue.input_name.to_ascii_uppercase())
            .filter(|name| {
                !name.is_empty()
                    && name.len() <= 5
                    && name.chars().all(|c| c.is_ascii_alphanumeric())
            })
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        Ok(names)
    }

    pub fn fix_pdb(&self, path: impl AsRef<Path>) -> Result<FixedStructure> {
        let path = path.as_ref();
        let contents =
            std::fs::read_to_string(path).map_err(crate::error::read_error(path.to_path_buf()))?;
        self.fix_pdb_str(&contents)
    }

    pub fn fix_pdb_str(&self, contents: &str) -> Result<FixedStructure> {
        let options = &self.options;
        let templates = &self.templates;
        let raw = pdb::read_model(contents, options.model, options.altloc)?;
        let atoms_in = raw.residues.iter().map(|r| r.atoms.len()).sum();
        let mut report = FixReport {
            glysys_version: env!("CARGO_PKG_VERSION").into(),
            input_sha256: format!("{:x}", Sha256::digest(contents.as_bytes())),
            output_sha256: String::new(),
            options: options.clone(),
            chains: Vec::new(),
            residue_counts: ResidueCounts::default(),
            atoms_in,
            atoms_out: 0,
            heavy_atoms_added: 0,
            hydrogens_added: 0,
            residues_added: 0,
            missing_residues: Vec::new(),
            replaced_residues: Vec::new(),
            rebuilt_atoms: Vec::new(),
            removed_atoms: Vec::new(),
            removed_residues: BTreeMap::new(),
            protonation: Vec::new(),
            disulfides: Vec::new(),
            chain_breaks: Vec::new(),
            heterogens: Vec::new(),
            components_missing: Vec::new(),
            unresolved: Vec::new(),
            remaining_clashes: 0,
            hydrogen_contacts: 0,
            warnings: raw
                .warnings
                .iter()
                .map(|warning| match warning {
                    crate::BuildWarning::AlternateLocationSelected(message) => message.clone(),
                    other => format!("{other:?}"),
                })
                .collect(),
        };
        let (mut work, notes) = work::load(raw, templates, &self.components);
        if notes.aliased_atoms > 0 {
            report.warnings.push(format!(
                "{} atom names were normalized to wwPDB conventions (e.g. OT1/OT2, O1P, CD of ILE)",
                notes.aliased_atoms
            ));
        }
        if notes.dropped_hydrogens > 0 {
            report.warnings.push(format!(
                "{} input hydrogens on standard residues and waters were rebuilt",
                notes.dropped_hydrogens
            ));
        }

        self.apply_removal_policy(&mut work, &mut report);
        if options.replace_nonstandard {
            for residue in &mut work.residues {
                if residue.kind != ResidueKind::Ligand || residue.parent.is_none() {
                    continue;
                }
                let from = residue.input_name.clone();
                let label = residue.label();
                if let Some(removed) = complete::replace_with_parent(residue, templates) {
                    report.replaced_residues.push(ReplacedResidue {
                        residue: label,
                        from,
                        to: residue.name.clone(),
                        removed_atoms: removed,
                    });
                }
            }
        }
        work::link_polymers(&mut work);

        let planned = self.plan_missing_residues(&mut work, &mut report);
        if options.add_missing_atoms {
            self.complete_residues(&mut work, &mut report, false);
        }
        if !planned.is_empty() {
            self.build_missing_residues(&mut work, planned, &mut report);
            for residue in &mut work.residues {
                residue.open_before = false;
                residue.open_after = false;
            }
            work::link_polymers(&mut work);
            self.complete_residues(&mut work, &mut report, true);
        }
        work::prune_polymer_bonds(&mut work);
        self.report_chain_breaks(&work, &mut report);

        if options.relax
            && work
                .residues
                .iter()
                .any(|r| r.atoms.iter().any(|a| a.movable()))
        {
            let map = self.template_map(&work);
            let mut model = relax::build_model(&work, &map, &[]);
            relax::scan_side_chains(&mut model, &work, &map);
            relax::minimize(&mut model, 600);
            relax::store(&model, &mut work);
            report.remaining_clashes = relax::remaining_clashes(&model);
        }

        for decision in protonation::assign(&mut work, options.ph, &options.protonation.residues) {
            let residue = &work.residues[decision.residue];
            report.protonation.push(ProtonationChoice {
                residue: residue.label(),
                state: decision.variant,
                reason: decision.reason,
            });
        }
        let index_of = work.index_of();
        for ((a, _), (b, _)) in &work.bonds {
            let (Some(&a), Some(&b)) = (index_of.get(a), index_of.get(b)) else {
                continue;
            };
            let (ra, rb) = (&work.residues[a], &work.residues[b]);
            if ra.variant.as_deref() == Some("CYX") && rb.variant.as_deref() == Some("CYX") {
                report
                    .disulfides
                    .push(format!("{} - {}", ra.label(), rb.label()));
            }
        }

        let mut glycam_names = HashMap::new();
        if options.add_hydrogens {
            glycam_names = self.add_hydrogens(&mut work, &mut report);
            if options.relax {
                let mut model = relax::build_hydrogen_model(&work);
                relax::minimize(&mut model, 300);
                relax::store(&model, &mut work);
            }
            report.hydrogen_contacts = relax::hydrogen_clashes(&work);
        }
        for (index, residue) in work.residues.iter_mut().enumerate() {
            if residue.template.is_none()
                && let Some(name) = glycam_names.get(&index)
            {
                residue.template = Some(name.clone());
            }
        }

        self.summarize(&work, &mut report);
        let remarks = vec![
            format!("GlySys {} structure repair", env!("CARGO_PKG_VERSION")),
            format!(
                "pH {:.1}; missing residues: {:?}; heavy atoms added {}; hydrogens added {}",
                options.ph,
                options.missing_residues,
                report.heavy_atoms_added,
                report.hydrogens_added
            ),
        ];
        let text = output::write(&work, options.naming, &glycam_names, &remarks);
        report.atoms_out = work.residues.iter().map(|r| r.atoms.len()).sum();
        report.output_sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
        Ok(FixedStructure { pdb: text, report })
    }

    fn apply_removal_policy(&self, work: &mut Work, report: &mut FixReport) {
        let options = &self.options;
        work.residues.retain(|residue| {
            let keep = match residue.kind {
                ResidueKind::Water => options.keep_water,
                ResidueKind::Ion => options.keep_heterogens,
                ResidueKind::Ligand => options.keep_heterogens || residue.parent.is_some(),
                ResidueKind::Glycan => options.keep_glycans,
                _ => true,
            };
            if !keep {
                *report
                    .removed_residues
                    .entry(residue.input_name.clone())
                    .or_default() += 1;
            }
            keep
        });
        let remaining = work.residues.iter().map(|r| r.uid).collect::<HashSet<_>>();
        work.bonds
            .retain(|((a, _), (b, _))| remaining.contains(a) && remaining.contains(b));
    }

    fn plan_missing_residues(&self, work: &mut Work, report: &mut FixReport) -> Vec<loops::Gap> {
        let mode = self.options.missing_residues;
        let mut planned = Vec::new();
        for gap in loops::find_gaps(work) {
            let label = |index: Option<usize>| index.map(|i| work.residues[i].label());
            let mut segment = MissingSegment {
                chain: gap.chain.clone(),
                after: label(gap.after),
                before: label(gap.before),
                residues: gap.names.clone(),
                modelled: false,
                note: String::new(),
            };
            let protein_names = gap
                .names
                .iter()
                .all(|name| chemistry::AMINO_ACIDS.contains(&name.as_str()));
            let protein_flanks = [gap.after, gap.before]
                .iter()
                .flatten()
                .all(|&i| work.residues[i].kind == ResidueKind::Protein);
            let internal = gap.after.is_some() && gap.before.is_some();
            if mode == MissingResidues::None || !self.options.add_missing_atoms {
                segment.note = "not modelled (disabled)".into();
            } else if !protein_names || !protein_flanks {
                segment.note = "not modelled: only protein segments are built".into();
            } else if !internal && mode != MissingResidues::All {
                segment.note = "terminal residues not modelled (internal gaps only)".into();
            } else if let (Some(a), Some(b)) = (gap.after, gap.before)
                && work::bonded_in_chain(&work.residues[a], &work.residues[b])
            {
                segment.note =
                    "not modelled: residue numbering skips but the chain is continuous".into();
            } else if let (Some(a), Some(b)) = (gap.after, gap.before)
                && let (Some(ca_a), Some(ca_b)) = (
                    work.residues[a].position("CA"),
                    work.residues[b].position("CA"),
                )
                && distance(ca_a, ca_b) > 3.8 * (gap.names.len() + 1) as f64 + 0.5
            {
                segment.note = format!(
                    "not modelled: {} residues cannot span {:.1} Å",
                    gap.names.len(),
                    distance(ca_a, ca_b)
                );
            } else {
                if let Some(a) = gap.after {
                    work.residues[a].open_after = true;
                }
                if let Some(b) = gap.before {
                    work.residues[b].open_before = true;
                }
                segment.modelled = true;
                planned.push(gap);
            }
            report.missing_residues.push(segment);
        }
        planned
    }

    fn build_missing_residues(
        &self,
        work: &mut Work,
        planned: Vec<loops::Gap>,
        report: &mut FixReport,
    ) {
        let mut positions = Vec::new();
        for residue in &work.residues {
            for atom in &residue.atoms {
                if !atom.is_hydrogen() {
                    positions.push(atom.position);
                }
            }
        }
        let mut grid = Grid::new(4.0);
        for (index, position) in positions.iter().enumerate() {
            grid.insert(index, *position);
        }
        let center = loops::centroid(&positions);
        // Insertions keyed by the residue index they precede (or follow).
        let mut before = HashMap::<usize, Vec<WResidue>>::new();
        let mut after = HashMap::<usize, Vec<WResidue>>::new();
        let segments = report
            .missing_residues
            .iter()
            .enumerate()
            .filter(|(_, segment)| segment.modelled)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        for (gap, segment_index) in planned.into_iter().zip(segments) {
            let built = match (gap.after, gap.before) {
                (Some(a), Some(b)) => loops::build_internal(
                    &work.residues[a],
                    &work.residues[b],
                    &gap.names,
                    &grid,
                    &positions,
                    (a as u64) << 20 | b as u64,
                )
                .map(|(backbone, rmsd)| {
                    if rmsd > 0.5 {
                        report.missing_residues[segment_index].note =
                            format!("loop closure deviation {rmsd:.2} Å before relaxation");
                    }
                    backbone
                }),
                (Some(a), None) => {
                    loops::build_c_tail(&work.residues[a], &gap.names, &grid, &positions, center)
                }
                (None, Some(b)) => {
                    loops::build_n_tail(&work.residues[b], &gap.names, &grid, &positions, center)
                }
                (None, None) => None,
            };
            let Some(backbones) = built else {
                let segment = &mut report.missing_residues[segment_index];
                segment.modelled = false;
                segment.note = "not modelled: flanking backbone atoms are missing".into();
                if let Some(a) = gap.after {
                    work.residues[a].open_after = false;
                }
                if let Some(b) = gap.before {
                    work.residues[b].open_before = false;
                }
                continue;
            };
            let numbers = self.gap_numbers(work, &gap, backbones.len());
            let mut residues = Vec::new();
            for ((name, backbone), (number, icode)) in gap.names.iter().zip(&backbones).zip(numbers)
            {
                let mut residue =
                    loops::modelled_residue(work.next_uid, &gap.chain, number, name, backbone);
                residue.icode = icode;
                work.next_uid += 1;
                residues.push(residue);
            }
            report.residues_added += residues.len();
            match (gap.after, gap.before) {
                (Some(a), _) => after.entry(a).or_default().extend(residues),
                (None, Some(b)) => before.entry(b).or_default().extend(residues),
                _ => {}
            }
        }
        let old = std::mem::take(&mut work.residues);
        for (index, residue) in old.into_iter().enumerate() {
            if let Some(inserted) = before.remove(&index) {
                work.residues.extend(inserted);
            }
            // A deposited C-terminal OXT is no longer terminal.
            work.residues.push(residue);
            if let Some(inserted) = after.remove(&index) {
                work.residues.extend(inserted);
            }
        }
    }

    fn gap_numbers(&self, work: &Work, gap: &loops::Gap, count: usize) -> Vec<(i32, Option<char>)> {
        if let Some(numbers) = &gap.numbers
            && numbers.len() == count
        {
            return numbers.clone();
        }
        match (gap.after, gap.before) {
            (Some(a), Some(b)) => {
                let (first, last) = (work.residues[a].number, work.residues[b].number);
                if last - first > count as i32 {
                    (1..=count as i32).map(|k| (first + k, None)).collect()
                } else {
                    // Numbering leaves no room: use insertion codes.
                    (0..count)
                        .map(|k| (first, char::from_u32('A' as u32 + (k % 26) as u32)))
                        .collect()
                }
            }
            (None, Some(b)) => {
                let first = work.residues[b].number;
                (0..count as i32)
                    .map(|k| (first - count as i32 + k, None))
                    .collect()
            }
            (Some(a), None) => {
                let last = work.residues[a].number;
                (1..=count as i32).map(|k| (last + k, None)).collect()
            }
            (None, None) => Vec::new(),
        }
    }

    fn complete_residues(&self, work: &mut Work, report: &mut FixReport, only_modelled: bool) {
        for index in 0..work.residues.len() {
            let residue = &work.residues[index];
            if !residue.is_polymer() || (only_modelled && !residue.modelled) {
                continue;
            }
            let Some((template, note)) = complete::residue_template(&self.templates, work, index)
            else {
                report
                    .unresolved
                    .push(format!("{}: no residue template", residue.label()));
                continue;
            };
            if let Some(note) = note {
                report.warnings.push(note);
            }
            let (link_prev, link_next) = match residue.kind {
                ResidueKind::Nucleic => ("O3'", "P"),
                _ => ("C", "N"),
            };
            let previous = residue
                .prev
                .and_then(|p| work.residues[p].position(link_prev));
            let next = residue
                .next
                .and_then(|n| work.residues[n].position(link_next));
            let label = residue.label();
            let modelled = residue.modelled;
            let residue = &mut work.residues[index];
            residue.template = Some(template.name.clone());
            let completion = complete::complete_residue(residue, template, previous, next, true);
            if !completion.added.is_empty() {
                report.heavy_atoms_added += completion.added.len();
                if !modelled {
                    report.rebuilt_atoms.push(ResidueAtoms {
                        residue: label.clone(),
                        atoms: completion.added,
                    });
                }
            }
            if !completion.removed.is_empty() {
                report.removed_atoms.push(ResidueAtoms {
                    residue: label.clone(),
                    atoms: completion.removed,
                });
            }
            if !completion.unresolved.is_empty() {
                report.unresolved.push(format!(
                    "{label}: could not place {}",
                    completion.unresolved.join(", ")
                ));
            }
        }
    }

    fn report_chain_breaks(&self, work: &Work, report: &mut FixReport) {
        for pair in work.residues.windows(2) {
            if work::chain_gap(&pair[0], &pair[1]) && !pair[0].ter_after {
                report
                    .chain_breaks
                    .push(format!("{} | {}", pair[0].label(), pair[1].label()));
            }
        }
        if !report.chain_breaks.is_empty() {
            report.warnings.push(format!(
                "{} chain breaks remain; each side was capped as a charged terminus",
                report.chain_breaks.len()
            ));
        }
    }

    fn template_map(&self, work: &Work) -> HashMap<usize, &Template> {
        (0..work.residues.len())
            .filter(|&index| work.residues[index].is_polymer())
            .filter_map(|index| {
                complete::residue_template(&self.templates, work, index).map(|(t, _)| (index, t))
            })
            .collect()
    }

    fn external_neighbors(work: &Work, index: usize) -> HashMap<String, Vec<V>> {
        let residue = &work.residues[index];
        let mut external = HashMap::<String, Vec<V>>::new();
        let mut push = |name: &str, position: Option<V>| {
            if let Some(position) = position {
                external.entry(name.to_string()).or_default().push(position);
            }
        };
        if let Some(prev) = residue.prev {
            push(
                "N",
                work.residues[prev]
                    .position("C")
                    .filter(|_| residue.atom("N").is_some()),
            );
            push(
                "P",
                work.residues[prev]
                    .position("O3'")
                    .filter(|_| residue.atom("P").is_some()),
            );
        }
        if let Some(next) = residue.next {
            push(
                "C",
                work.residues[next]
                    .position("N")
                    .filter(|_| residue.atom("C").is_some()),
            );
            push(
                "O3'",
                work.residues[next]
                    .position("P")
                    .filter(|_| residue.atom("O3'").is_some()),
            );
        }
        let index_of = work.index_of();
        for ((uid_a, atom_a), (uid_b, atom_b)) in &work.bonds {
            for ((this_uid, this_atom), (other_uid, other_atom)) in [
                ((uid_a, atom_a), (uid_b, atom_b)),
                ((uid_b, atom_b), (uid_a, atom_a)),
            ] {
                if *this_uid == residue.uid
                    && let Some(&other) = index_of.get(other_uid)
                {
                    push(this_atom, work.residues[other].position(other_atom));
                }
            }
        }
        external
    }

    #[allow(clippy::needless_range_loop)]
    fn add_hydrogens(&self, work: &mut Work, report: &mut FixReport) -> HashMap<usize, String> {
        let mut environment = Environment::new();
        for residue in &work.residues {
            for atom in &residue.atoms {
                if atom.is_hydrogen() {
                    environment.add_hydrogen(atom.position, false);
                } else {
                    let acceptor = match atom.element.as_str() {
                        "O" => true,
                        "N" => nitrogen_acceptor(residue, &atom.name),
                        _ => false,
                    };
                    environment.add_heavy(atom.position, &atom.element, acceptor);
                }
            }
        }
        // Hydrogen sources per residue.
        let mut sources: Vec<Option<(HTemplate, HashSet<String>)>> =
            Vec::with_capacity(work.residues.len());
        let mut glycam = HashMap::new();
        let mut heterogen_source = BTreeMap::<(String, ResidueKind), (usize, String)>::new();
        let declared = declared_bonds(work);
        for index in 0..work.residues.len() {
            let residue = &work.residues[index];
            let source = match residue.kind {
                ResidueKind::Protein | ResidueKind::Nucleic => {
                    complete::residue_template(&self.templates, work, index)
                        .map(|(template, _)| (HTemplate::from_amber(template), HashSet::new()))
                }
                ResidueKind::Glycan => None,
                ResidueKind::Ligand => self.component_template(work, index),
                _ => None,
            };
            sources.push(source);
        }
        // Glycans: GLYCAM templates, with CCD definitions as a fallback.
        let mut environment_residues = Vec::new();
        let glycan_indices = (0..work.residues.len())
            .filter(|&i| work.residues[i].kind == ResidueKind::Glycan)
            .collect::<Vec<_>>();
        if !glycan_indices.is_empty() {
            environment_residues = work
                .residues
                .iter()
                .map(|residue| pdb_residue(residue, false))
                .collect::<Vec<_>>();
        }
        for &index in &glycan_indices {
            let neighbors = nearby_residues(work, index, &environment_residues);
            match glycam_hydrogens(
                &self.templates,
                &work.residues[index],
                &declared,
                &neighbors,
            ) {
                Some((name, atoms, keep)) => {
                    // Deposited hydrogens under other naming schemes are
                    // replaced so every glycan hydrogen has its GLYCAM name.
                    let before = work.residues[index].atoms.len();
                    work.residues[index]
                        .atoms
                        .retain(|atom| !atom.is_hydrogen() || keep.contains(&atom.name));
                    let replaced = before - work.residues[index].atoms.len();
                    report.hydrogens_added += atoms.len().saturating_sub(replaced);
                    for atom in &atoms {
                        environment.add_hydrogen(atom.position, false);
                    }
                    work.residues[index].atoms.extend(atoms);
                    glycam.insert(index, name);
                    let residue = &work.residues[index];
                    heterogen_source
                        .entry((residue.input_name.clone(), residue.kind))
                        .or_insert((0, "glycam".into()))
                        .0 += 1;
                }
                None => {
                    sources[index] = self.component_template(work, index);
                }
            }
        }
        // Ligand hydrogens that the CCD definition does not name are dropped
        // and rebuilt under the definition's names.
        for index in 0..work.residues.len() {
            if let Some((template, _)) = &sources[index]
                && matches!(
                    work.residues[index].kind,
                    ResidueKind::Ligand | ResidueKind::Glycan
                )
            {
                let names = template.names.iter().collect::<HashSet<_>>();
                work.residues[index]
                    .atoms
                    .retain(|atom| !atom.is_hydrogen() || names.contains(&atom.name));
            }
        }
        for phase in [Phase::Fixed, Phase::Rotors] {
            for index in 0..work.residues.len() {
                let Some((template, skip)) = &sources[index] else {
                    continue;
                };
                let external = Self::external_neighbors(work, index);
                let added = hydrogens::place_residue(
                    &mut work.residues[index],
                    template,
                    &external,
                    skip,
                    &mut environment,
                    phase,
                );
                report.hydrogens_added += added;
            }
        }
        for index in 0..work.residues.len() {
            let residue = &work.residues[index];
            match residue.kind {
                ResidueKind::Water => {
                    let Some(oxygen) = residue.position("O") else {
                        continue;
                    };
                    let [h1, h2] = hydrogens::place_water(oxygen, &mut environment);
                    for (name, position) in [("H1", h1), ("H2", h2)] {
                        work.residues[index].atoms.push(work::WAtom {
                            name: name.into(),
                            element: "H".into(),
                            position,
                            occupancy: 1.0,
                            b_factor: 0.0,
                            serial: None,
                            origin: Origin::Hydrogen,
                        });
                    }
                    report.hydrogens_added += 2;
                }
                ResidueKind::Ligand | ResidueKind::Glycan if !glycam.contains_key(&index) => {
                    let how = if sources[index].is_some() {
                        "ccd"
                    } else {
                        "none"
                    };
                    heterogen_source
                        .entry((residue.input_name.clone(), residue.kind))
                        .or_insert((0, how.into()))
                        .0 += 1;
                    if sources[index].is_none()
                        && !report.components_missing.contains(&residue.input_name)
                        && residue.atoms.iter().filter(|a| !a.is_hydrogen()).count() > 1
                    {
                        report.components_missing.push(residue.input_name.clone());
                    }
                }
                ResidueKind::Ion => {
                    heterogen_source
                        .entry((residue.input_name.clone(), residue.kind))
                        .or_insert((0, "none".into()))
                        .0 += 1;
                }
                _ => {}
            }
        }
        report.heterogens = heterogen_source
            .into_iter()
            .map(|((name, kind), (count, hydrogens))| HeterogenSummary {
                name,
                kind,
                count,
                hydrogens,
            })
            .collect();
        if !report.components_missing.is_empty() {
            report.warnings.push(format!(
                "no hydrogens were added to {} (no chemical component definition available)",
                report.components_missing.join(", ")
            ));
        }
        glycam
    }

    /// Hydrogen template from a CCD definition whose heavy atoms match.
    fn component_template(
        &self,
        work: &Work,
        index: usize,
    ) -> Option<(HTemplate, HashSet<String>)> {
        let residue = &work.residues[index];
        let component = self.components.get(&residue.input_name)?;
        for atom in residue.atoms.iter().filter(|atom| !atom.is_hydrogen()) {
            let defined = component.atom(&atom.name)?;
            if defined.element != atom.element
                && !(defined.element.len() == 1 && atom.element.starts_with(&defined.element))
            {
                return None;
            }
        }
        let template = HTemplate::from_component(component)?;
        // Leaving hydrogens are absent where the parent is bonded elsewhere.
        let external = Self::external_neighbors(work, index);
        let mut skip = HashSet::new();
        // Parents next to an absent (non-leaving) heavy atom have an unknown
        // valence geometry: leave their hydrogens off rather than guess.
        let truncated = component
            .bonds
            .iter()
            .flat_map(|bond| [(&bond.first, &bond.second), (&bond.second, &bond.first)])
            .filter(|(atom, other)| {
                residue.atom(atom).is_some()
                    && component.atom(other).is_some_and(|other| {
                        other.element != "H"
                            && !other.leaving
                            && residue.atom(&other.name).is_none()
                    })
            })
            .map(|(atom, _)| atom.clone())
            .collect::<HashSet<_>>();
        for bond in &component.bonds {
            for (h, parent) in [(&bond.first, &bond.second), (&bond.second, &bond.first)] {
                let Some(h_atom) = component.atom(h) else {
                    continue;
                };
                if h_atom.element != "H" {
                    continue;
                }
                let parent_missing = residue.atom(parent).is_none();
                if parent_missing
                    || truncated.contains(parent)
                    || (h_atom.leaving && external.contains_key(parent.as_str()))
                {
                    skip.insert(h.clone());
                }
            }
        }
        Some((template, skip))
    }

    fn summarize(&self, work: &Work, report: &mut FixReport) {
        let mut counts = ResidueCounts::default();
        let mut chains = Vec::<String>::new();
        for residue in &work.residues {
            match residue.kind {
                ResidueKind::Protein => counts.protein += 1,
                ResidueKind::Nucleic => counts.nucleic += 1,
                ResidueKind::Glycan => counts.glycan += 1,
                ResidueKind::Water => counts.water += 1,
                ResidueKind::Ion => counts.ion += 1,
                ResidueKind::Ligand => counts.ligand += 1,
            }
            if !chains.contains(&residue.chain) {
                chains.push(residue.chain.clone());
            }
        }
        report.residue_counts = counts;
        report.chains = chains;
    }
}

fn nitrogen_acceptor(residue: &WResidue, atom: &str) -> bool {
    match residue.kind {
        ResidueKind::Protein => {
            residue.name == "HIS"
                && match residue.variant.as_deref() {
                    Some("HID") => atom == "NE2",
                    Some("HIP") => false,
                    _ => atom == "ND1",
                }
        }
        ResidueKind::Nucleic => match residue.name.as_str() {
            "A" | "DA" => matches!(atom, "N1" | "N3" | "N7"),
            "G" | "DG" => matches!(atom, "N3" | "N7"),
            "C" | "DC" => atom == "N3",
            _ => false,
        },
        ResidueKind::Water => false,
        _ => true,
    }
}

fn pdb_residue(residue: &WResidue, keep_hydrogens: bool) -> PdbResidue {
    PdbResidue {
        reference: ResidueRef {
            chain: residue.chain.clone(),
            name: residue.input_name.clone(),
            number: residue.number,
            insertion_code: residue.icode,
        },
        atoms: residue
            .atoms
            .iter()
            .filter(|atom| keep_hydrogens || !atom.is_hydrogen())
            .map(|atom| PdbAtom {
                serial: atom.serial.unwrap_or(0),
                name: atom.name.clone(),
                residue_name: residue.input_name.clone(),
                chain: residue.chain.clone(),
                residue_number: residue.number,
                insertion_code: residue.icode,
                element: atom.element.clone(),
                occupancy: atom.occupancy,
                b_factor: atom.b_factor,
                position: p(atom.position),
            })
            .collect(),
    }
}

fn declared_bonds(work: &Work) -> Vec<crate::prepare::DeclaredAtomBond> {
    let index_of = work.index_of();
    work.bonds
        .iter()
        .filter_map(|((a, atom_a), (b, atom_b))| {
            let ra = &work.residues[*index_of.get(a)?];
            let rb = &work.residues[*index_of.get(b)?];
            Some((
                (
                    ra.chain.clone(),
                    ra.number,
                    normalize_glycan_atom(&ra.input_name, atom_a),
                ),
                (
                    rb.chain.clone(),
                    rb.number,
                    normalize_glycan_atom(&rb.input_name, atom_b),
                ),
            ))
        })
        .collect()
}

fn normalize_glycan_atom(residue: &str, atom: &str) -> String {
    if matches!(residue, "NAG" | "NDG") {
        match atom {
            "C7" => return "C2N".into(),
            "O7" => return "O2N".into(),
            "C8" => return "CME".into(),
            _ => {}
        }
    }
    atom.to_string()
}

/// Residues (as PDB views) with any heavy atom within 10 Å of `index`.
fn nearby_residues(work: &Work, index: usize, all: &[PdbResidue]) -> Vec<PdbResidue> {
    let center = loops::centroid(
        &work.residues[index]
            .atoms
            .iter()
            .map(|atom| atom.position)
            .collect::<Vec<_>>(),
    );
    all.iter()
        .enumerate()
        .filter(|(other, residue)| {
            *other == index
                || residue
                    .atoms
                    .iter()
                    .any(|atom| distance(geometry::v(atom.position), center) < 12.0)
        })
        .map(|(_, residue)| residue.clone())
        .collect()
}

/// GLYCAM06j-1 hydrogens for a carbohydrate residue.
fn glycam_hydrogens(
    templates: &TemplateSet,
    residue: &WResidue,
    declared: &[crate::prepare::DeclaredAtomBond],
    neighbors: &[PdbResidue],
) -> Option<(String, Vec<work::WAtom>, HashSet<String>)> {
    let mut view = pdb_residue(residue, true);
    for atom in &mut view.atoms {
        atom.name = normalize_glycan_atom(&residue.input_name, &atom.name);
    }
    let name =
        crate::prepare::infer_glycam_template_name(&view, neighbors, declared, templates).ok()?;
    let template = templates.glycan(&name)?;
    let heavy_known = view
        .atoms
        .iter()
        .filter(|atom| !matches!(atom.element.as_str(), "H" | "D"))
        .all(|atom| template.atom(&atom.name).is_some());
    if !heavy_known {
        return None;
    }
    let mut environment = neighbors.to_vec();
    if let Some(own) = environment
        .iter_mut()
        .find(|candidate| candidate.reference == view.reference)
    {
        *own = view.clone();
    }
    let positions = crate::prepare::glycan_hydrogen_positions(template, &view, &environment);
    let present = residue
        .atoms
        .iter()
        .map(|atom| atom.name.as_str())
        .collect::<HashSet<_>>();
    let keep = template
        .atoms
        .iter()
        .filter(|atom| atom.element == 1 && present.contains(atom.name.as_str()))
        .map(|atom| atom.name.clone())
        .collect::<HashSet<_>>();
    let mut atoms = positions
        .into_iter()
        .filter(|(index, _)| !present.contains(template.atoms[*index].name.as_str()))
        .map(|(index, position)| {
            (
                index,
                work::WAtom {
                    name: template.atoms[index].name.clone(),
                    element: "H".into(),
                    position: geometry::v(position),
                    occupancy: 1.0,
                    b_factor: 0.0,
                    serial: None,
                    origin: Origin::Hydrogen,
                },
            )
        })
        .collect::<Vec<_>>();
    atoms.sort_by_key(|(index, _)| *index);
    Some((
        name,
        atoms.into_iter().map(|(_, atom)| atom).collect(),
        keep,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIPEPTIDE_WITH_GAP: &str = include_str!("../tests/fixtures/dipeptide.pdb");

    #[test]
    fn fixes_the_dipeptide_fixture() {
        let fixer = StructureFixer::new(FixOptions::default()).unwrap();
        let fixed = fixer.fix_pdb_str(DIPEPTIDE_WITH_GAP).unwrap();
        assert!(fixed.report.hydrogens_added > 0);
        assert!(fixed.pdb.contains("ATOM"));
    }
}
