use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::forcefield::{ParameterSet, Template, TemplateSet};
use crate::model::{Angle, Atom, Bond, Dihedral, PreparedSystem, Residue, System, Vec3};
use crate::pdb::{self, PROTEIN_RESIDUES, ParsedPdb, PdbResidue};
use crate::report::BuildReport;
use crate::{BuildError, BuildOptions, BuildWarning, Result};

/// Reusable builder containing validated options and immutable force-field data.
#[derive(Debug)]
pub struct SystemBuilder {
    options: BuildOptions,
    templates: TemplateSet,
    parameters: ParameterSet,
}

impl SystemBuilder {
    pub fn new(options: BuildOptions) -> Result<Self> {
        options.validate()?;
        Ok(Self {
            options,
            templates: TemplateSet::load()?,
            parameters: ParameterSet::load()?,
        })
    }

    pub fn options(&self) -> &BuildOptions {
        &self.options
    }

    pub fn prepare_pdb(&self, path: impl AsRef<Path>) -> Result<PreparedSystem> {
        let path = path.as_ref();
        let contents =
            std::fs::read_to_string(path).map_err(crate::error::read_error(path.to_path_buf()))?;
        self.prepare_pdb_str(&contents)
    }

    /// Parse, classify, and validate an input without solvating or writing files.
    pub fn inspect_pdb(&self, path: impl AsRef<Path>) -> Result<BuildReport> {
        let path = path.as_ref();
        let contents =
            std::fs::read_to_string(path).map_err(crate::error::read_error(path.to_path_buf()))?;
        let parsed = pdb::parse(&contents, &self.options)?;
        let mut warnings = parsed.warnings.clone();
        let protonation_decisions = self.protonation_decisions(&parsed, &mut warnings);
        let unsupported_residues = parsed
            .residues
            .iter()
            .filter(|residue| {
                let name = residue.reference.name.as_str();
                !PROTEIN_RESIDUES.contains(&name)
                    && !pdb::is_water(name)
                    && !pdb::is_free_ion(name)
                    && self.templates.glycan(name).is_none()
                    && !matches!(
                        name,
                        "NAG"
                            | "NDG"
                            | "MAN"
                            | "BMA"
                            | "GLC"
                            | "BGC"
                            | "GAL"
                            | "GLA"
                            | "FUC"
                            | "FUL"
                            | "XYS"
                            | "XYP"
                            | "SIA"
                            | "SLB"
                    )
            })
            .map(|residue| residue.reference.clone())
            .collect::<Vec<_>>();
        let mut missing_heavy_atoms = Vec::new();
        let mut solute_atoms = 0;
        let mut solute_charge = 0.0;
        if unsupported_residues.is_empty() {
            match self.parameterize(&parsed, &protonation_decisions, &mut warnings) {
                Ok(system) => {
                    solute_atoms = system.solute_atom_count;
                    solute_charge = system.charge();
                }
                Err(BuildError::MissingHeavyAtoms { residue, atoms }) => {
                    missing_heavy_atoms.push(format!("{residue}: {atoms}"));
                }
                Err(error) => return Err(error),
            }
        }
        Ok(BuildReport {
            input_sha256: format!("{:x}", Sha256::digest(contents.as_bytes())),
            force_field_versions: crate::report::force_field_versions(),
            output_sha256: BTreeMap::new(),
            options: self.options.clone(),
            selected_model: self.options.model,
            chains: parsed.chains,
            glycans: parsed.glycans,
            protonation_decisions,
            unsupported_residues,
            missing_heavy_atoms,
            warnings,
            solute_atoms,
            total_atoms: solute_atoms,
            residues: parsed.residues.len(),
            waters: 0,
            sodium_ions: 0,
            chloride_ions: 0,
            solute_charge,
            total_charge: solute_charge,
            box_angstrom: [0.0; 3],
        })
    }

    pub fn prepare_pdb_str(&self, contents: &str) -> Result<PreparedSystem> {
        let parsed = pdb::parse(contents, &self.options)?;
        let input_sha256 = format!("{:x}", Sha256::digest(contents.as_bytes()));
        let mut warnings = parsed.warnings.clone();
        let protonation_decisions = self.protonation_decisions(&parsed, &mut warnings);
        let mut system = self.parameterize(&parsed, &protonation_decisions, &mut warnings)?;
        let solute_charge = system.charge();
        let rounded_charge = solute_charge.round();
        if (solute_charge - rounded_charge).abs() > 1.0e-3 {
            return Err(BuildError::NonIntegralCharge(solute_charge));
        }
        if self.options.add_water {
            crate::solvate::solvate_and_ionize(
                &mut system,
                &self.templates,
                &self.parameters,
                &self.options,
            )?;
        }
        let report = BuildReport {
            input_sha256,
            force_field_versions: crate::report::force_field_versions(),
            output_sha256: BTreeMap::new(),
            options: self.options.clone(),
            selected_model: self.options.model,
            chains: parsed.chains,
            glycans: parsed.glycans,
            protonation_decisions,
            unsupported_residues: Vec::new(),
            missing_heavy_atoms: Vec::new(),
            warnings,
            solute_atoms: system.solute_atom_count,
            total_atoms: system.atoms.len(),
            residues: system.residues.len(),
            waters: system.water_residue_count,
            sodium_ions: system.sodium_count,
            chloride_ions: system.chloride_count,
            solute_charge,
            total_charge: system.charge(),
            box_angstrom: system.box_angstrom,
        };
        Ok(PreparedSystem { system, report })
    }

    fn protonation_decisions(
        &self,
        parsed: &ParsedPdb,
        warnings: &mut Vec<BuildWarning>,
    ) -> Vec<String> {
        let acceptors = parsed
            .residues
            .iter()
            .flat_map(|residue| &residue.atoms)
            .filter(|atom| matches!(atom.element.as_str(), "O" | "N" | "S"))
            .collect::<Vec<_>>();
        let mut decisions = Vec::new();
        for residue in &parsed.residues {
            if residue.reference.name != "HIS" {
                continue;
            }
            let selector = selector(residue);
            let state = self
                .options
                .protonation
                .residues
                .get(&selector)
                .cloned()
                .unwrap_or_else(|| {
                    let nd1 = residue.atoms.iter().find(|atom| atom.name == "ND1");
                    let ne2 = residue.atoms.iter().find(|atom| atom.name == "NE2");
                    let closest = |source: Option<&crate::pdb::PdbAtom>| {
                        source
                            .map(|source| {
                                acceptors
                                    .iter()
                                    .filter(|target| target.serial != source.serial)
                                    .map(|target| source.position.distance2(target.position))
                                    .fold(f64::INFINITY, f64::min)
                            })
                            .unwrap_or(f64::INFINITY)
                    };
                    if closest(nd1) + 0.04 < closest(ne2) {
                        "HID".to_string()
                    } else {
                        "HIE".to_string()
                    }
                });
            decisions.push(format!("{selector}={state}"));
            warnings.push(BuildWarning::HistidineStateInferred(format!(
                "{selector} selected as {state}"
            )));
        }
        decisions
    }

    fn parameterize(
        &self,
        parsed: &ParsedPdb,
        protonation: &[String],
        warnings: &mut Vec<BuildWarning>,
    ) -> Result<System> {
        let mut kept = parsed
            .residues
            .iter()
            .filter(|residue| {
                let remove = pdb::is_water(&residue.reference.name)
                    || pdb::is_free_ion(&residue.reference.name);
                if remove {
                    warnings.push(BuildWarning::ExistingSolventRemoved(format!(
                        "{}",
                        residue.reference
                    )));
                }
                !remove
            })
            .cloned()
            .collect::<Vec<_>>();
        let declared_bonds = declared_atom_bonds(parsed);
        normalize_special_residues(&mut kept, &declared_bonds, protonation, warnings);
        normalize_disulfides(&mut kept, &parsed.ssbonds);
        let terminal = protein_terminal_flags(&kept);

        let mut atoms = Vec::new();
        let mut residues = Vec::new();
        let mut bonds = Vec::<[usize; 2]>::new();
        let mut serial_to_atom = HashMap::new();
        let mut residue_atom = HashMap::<(usize, String), usize>::new();

        for (residue_index, residue) in kept.iter().enumerate() {
            let name = residue.reference.name.as_str();
            let (n_terminal, c_terminal) =
                terminal.get(&residue_index).copied().unwrap_or_default();
            let template = if PROTEIN_RESIDUES.contains(&name) {
                self.templates.protein(name, n_terminal, c_terminal)
            } else if let Some(template) = self.templates.glycan(name) {
                Some(template)
            } else {
                let glycam_name =
                    infer_glycam_template_name(residue, &kept, &declared_bonds, &self.templates)?;
                warnings.push(BuildWarning::GlycanNameNormalized(format!(
                    "{} -> {glycam_name}",
                    residue.reference
                )));
                self.templates.glycan(&glycam_name)
            }
            .ok_or_else(|| BuildError::UnsupportedResidue {
                residue: residue.reference.to_string(),
                reason: "no ff14SB or GLYCAM06j-1 residue template".into(),
            })?;
            let first_atom = atoms.len();
            let actual = residue
                .atoms
                .iter()
                .map(|atom| (atom.name.as_str(), atom))
                .collect::<HashMap<_, _>>();
            let generated_glycan_hydrogens = if PROTEIN_RESIDUES.contains(&name) {
                HashMap::new()
            } else {
                glycan_hydrogen_positions(template, residue, &kept)
            };
            let missing = template
                .atoms
                .iter()
                .filter(|atom| atom.element != 1 && !actual.contains_key(atom.name.as_str()))
                .map(|atom| atom.name.clone())
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(BuildError::MissingHeavyAtoms {
                    residue: residue.reference.to_string(),
                    atoms: missing.join(", "),
                });
            }
            for actual_atom in &residue.atoms {
                if actual_atom.element != "H"
                    && actual_atom.element != "D"
                    && template.atom(&actual_atom.name).is_none()
                {
                    return Err(BuildError::UnsupportedResidue {
                        residue: residue.reference.to_string(),
                        reason: format!(
                            "heavy atom {} is not present in template {}",
                            actual_atom.name, template.name
                        ),
                    });
                }
            }
            for (template_atom_index, template_atom) in template.atoms.iter().enumerate() {
                let position = if let Some(atom) = actual.get(template_atom.name.as_str()) {
                    atom.position
                } else if let Some(position) = generated_glycan_hydrogens.get(&template_atom_index)
                {
                    *position
                } else if let Some(position) = peptide_amide_hydrogen_position(
                    &kept,
                    residue_index,
                    template,
                    template_atom_index,
                ) {
                    position
                } else {
                    hydrogen_transform(template, residue, template_atom_index)
                        .ok_or_else(|| {
                            BuildError::InvalidPdb(format!(
                                "cannot construct local geometry for hydrogen {} in {}",
                                template_atom.name, residue.reference
                            ))
                        })?
                        .apply(template_atom.position)
                };
                let (radius, epsilon) = self.parameters.nonbonded(&template_atom.atom_type)?;
                let atom_index = atoms.len();
                atoms.push(Atom {
                    name: template_atom.name.clone(),
                    atom_type: template_atom.atom_type.clone(),
                    element: template_atom.element,
                    residue: residue_index,
                    charge: template_atom.charge,
                    mass: self
                        .parameters
                        .mass(&template_atom.atom_type, template_atom.element),
                    radius,
                    epsilon,
                    position,
                });
                residue_atom.insert((residue_index, template_atom.name.clone()), atom_index);
                if let Some(actual) = actual.get(template_atom.name.as_str()) {
                    serial_to_atom.insert(actual.serial, atom_index);
                }
            }
            for [first, second] in &template.bonds {
                bonds.push([first_atom + first, first_atom + second]);
            }
            residues.push(Residue {
                name: parameterized_residue_name(name, template),
                number: residue.reference.number,
                insertion_code: residue.reference.insertion_code,
                chain: residue.reference.chain.clone(),
                first_atom,
                atom_count: template.atoms.len(),
                component: 0,
            });
        }

        // Standard peptide bonds.
        for (first_index, pair) in kept.windows(2).enumerate() {
            let second_index = first_index + 1;
            if pair[0].reference.chain == pair[1].reference.chain
                && PROTEIN_RESIDUES.contains(&pair[0].reference.name.as_str())
                && PROTEIN_RESIDUES.contains(&pair[1].reference.name.as_str())
                && let (Some(&first), Some(&second)) = (
                    residue_atom.get(&(first_index, "C".into())),
                    residue_atom.get(&(second_index, "N".into())),
                )
                && atoms[first].position.distance2(atoms[second].position) <= 2.0f64.powi(2)
            {
                bonds.push([first, second]);
            }
        }
        // PDB-declared bonds and conservative carbohydrate/attachment distance bonds.
        for (first_serial, second_serial) in &parsed.conect {
            if let (Some(&first), Some(&second)) = (
                serial_to_atom.get(first_serial),
                serial_to_atom.get(second_serial),
            ) {
                bonds.push([first, second]);
            }
        }
        for link in &parsed.links {
            let first_residue = find_residue_index(&kept, &link.first);
            let second_residue = find_residue_index(&kept, &link.second);
            if let (Some(first_residue), Some(second_residue)) = (first_residue, second_residue)
                && let (Some(&first), Some(&second)) = (
                    residue_atom.get(&(first_residue, link.first_atom.clone())),
                    residue_atom.get(&(second_residue, link.second_atom.clone())),
                )
            {
                bonds.push([first, second]);
            }
        }
        infer_cross_residue_bonds(&kept, &residue_atom, &atoms, &mut bonds);
        for (first, second) in &parsed.ssbonds {
            if let (Some(first_residue), Some(second_residue)) = (
                find_residue_index(&kept, first),
                find_residue_index(&kept, second),
            ) && let (Some(&first), Some(&second)) = (
                residue_atom.get(&(first_residue, "SG".into())),
                residue_atom.get(&(second_residue, "SG".into())),
            ) {
                bonds.push([first, second]);
            }
        }
        for bond in &mut bonds {
            if bond[0] > bond[1] {
                bond.swap(0, 1);
            }
        }
        bonds.sort_unstable();
        bonds.dedup();

        let (component_count, component_by_residue) =
            molecular_components(kept.len(), &bonds, &atoms);
        for (residue, component) in residues.iter_mut().zip(component_by_residue) {
            residue.component = component;
        }
        let (bonds, angles, dihedrals, exclusions) =
            enumerate_parameters(&atoms, &bonds, &self.parameters)?;
        let atom_count = atoms.len();
        Ok(System {
            atoms,
            residues,
            bonds,
            angles,
            dihedrals,
            exclusions,
            box_angstrom: [0.0; 3],
            component_count,
            solute_atom_count: atom_count,
            water_residue_count: 0,
            sodium_count: 0,
            chloride_count: 0,
        })
    }
}

fn parameterized_residue_name(input_name: &str, template: &Template) -> String {
    if PROTEIN_RESIDUES.contains(&input_name) {
        input_name.to_string()
    } else {
        template.name.clone()
    }
}

fn normalize_disulfides(
    residues: &mut [PdbResidue],
    declared: &[(pdb::ResidueKey, pdb::ResidueKey)],
) {
    let declared = declared
        .iter()
        .flat_map(|(first, second)| [first, second])
        .map(|residue| {
            (
                residue.chain.as_str(),
                residue.number,
                residue.insertion_code,
            )
        })
        .collect::<HashSet<_>>();
    let cysteines = residues
        .iter()
        .enumerate()
        .filter(|(_, residue)| residue.reference.name == "CYS")
        .filter_map(|(index, residue)| {
            let sulfur = residue.atoms.iter().find(|atom| atom.name == "SG")?;
            Some((index, sulfur.position))
        })
        .collect::<Vec<_>>();
    let mut bridged = HashSet::new();
    for (position, &(first, first_sulfur)) in cysteines.iter().enumerate() {
        for &(second, second_sulfur) in cysteines.iter().skip(position + 1) {
            if first_sulfur.distance2(second_sulfur) <= 2.3f64.powi(2) {
                bridged.insert(first);
                bridged.insert(second);
            }
        }
    }
    for (index, residue) in residues.iter_mut().enumerate() {
        let reference = &residue.reference;
        if reference.name == "CYS"
            && (bridged.contains(&index)
                || declared.contains(&(
                    reference.chain.as_str(),
                    reference.number,
                    reference.insertion_code,
                )))
        {
            residue.reference.name = "CYX".into();
        }
    }
}

fn selector(residue: &PdbResidue) -> String {
    format!(
        "{}:{}{}",
        residue.reference.chain,
        residue.reference.number,
        residue.reference.insertion_code.unwrap_or(' ')
    )
    .trim_end()
    .to_string()
}

type AtomLocator = (String, i32, String);
type DeclaredAtomBond = (AtomLocator, AtomLocator);

fn declared_atom_bonds(parsed: &ParsedPdb) -> Vec<DeclaredAtomBond> {
    let by_serial = parsed
        .residues
        .iter()
        .flat_map(|residue| {
            residue.atoms.iter().map(move |atom| {
                (
                    atom.serial,
                    (
                        residue.reference.chain.clone(),
                        residue.reference.number,
                        atom.name.clone(),
                    ),
                )
            })
        })
        .collect::<HashMap<_, _>>();
    let mut result = parsed
        .conect
        .iter()
        .filter_map(|(first, second)| {
            Some((
                by_serial.get(first)?.clone(),
                by_serial.get(second)?.clone(),
            ))
        })
        .collect::<Vec<_>>();
    result.extend(parsed.links.iter().map(|link| {
        (
            (
                link.first.chain.clone(),
                link.first.number,
                link.first_atom.clone(),
            ),
            (
                link.second.chain.clone(),
                link.second.number,
                link.second_atom.clone(),
            ),
        )
    }));
    result
}

fn normalize_special_residues(
    residues: &mut [PdbResidue],
    bonds: &[DeclaredAtomBond],
    protonation: &[String],
    warnings: &mut Vec<BuildWarning>,
) {
    for residue in residues {
        let selector = selector(residue);
        if residue.reference.name == "HIS" {
            residue.reference.name = protonation
                .iter()
                .find_map(|decision| decision.strip_prefix(&(selector.clone() + "=")))
                .unwrap_or("HIE")
                .to_string();
        }
        let attachment_atom = bonds.iter().find_map(|(first, second)| {
            if first.0 == residue.reference.chain && first.1 == residue.reference.number {
                Some(first.2.as_str())
            } else if second.0 == residue.reference.chain && second.1 == residue.reference.number {
                Some(second.2.as_str())
            } else {
                None
            }
        });
        let renamed = match (residue.reference.name.as_str(), attachment_atom) {
            ("ASN", Some("ND2")) => Some("NLN"),
            ("SER", Some("OG")) => Some("OLS"),
            ("THR", Some("OG1")) => Some("OLT"),
            ("HYP", Some("OD1")) => Some("OLP"),
            _ => None,
        };
        if let Some(renamed) = renamed {
            warnings.push(BuildWarning::GlycanNameNormalized(format!(
                "{} -> {renamed}",
                residue.reference
            )));
            residue.reference.name = renamed.into();
        }
    }
}

fn protein_terminal_flags(residues: &[PdbResidue]) -> HashMap<usize, (bool, bool)> {
    let mut by_chain: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, residue) in residues.iter().enumerate() {
        if PROTEIN_RESIDUES.contains(&residue.reference.name.as_str()) {
            by_chain
                .entry(&residue.reference.chain)
                .or_default()
                .push(index);
        }
    }
    let mut result = HashMap::new();
    for indices in by_chain.values() {
        for (position, &index) in indices.iter().enumerate() {
            result.insert(index, (position == 0, position + 1 == indices.len()));
        }
    }
    result
}

fn infer_glycam_template_name(
    residue: &PdbResidue,
    all: &[PdbResidue],
    bonds: &[DeclaredAtomBond],
    templates: &TemplateSet,
) -> Result<String> {
    let suffixes: &[&str] = match residue.reference.name.as_str() {
        "NAG" => &["YB"],
        "NDG" => &["YA"],
        "MAN" => &["MA"],
        "BMA" => &["MB"],
        "GLC" => &["GA"],
        "BGC" => &["GB"],
        "GAL" => &["LA", "LB"],
        "GLA" => &["LA"],
        "FUC" => &["fA"],
        "FUL" => &["fB"],
        "XYS" | "XYP" => &["XB"],
        "SIA" | "SLB" => &["SA", "SB"],
        other if other.len() == 3 && templates.glycan(other).is_some() => return Ok(other.into()),
        _ => &[],
    };
    if suffixes.is_empty() {
        return Err(BuildError::UnsupportedResidue {
            residue: residue.reference.to_string(),
            reason: "carbohydrate component has no GLYCAM06j-1 mapping".into(),
        });
    }
    let actual_names = residue
        .atoms
        .iter()
        .filter(|atom| atom.element != "H" && atom.element != "D")
        .map(|atom| atom.name.as_str())
        .collect::<HashSet<_>>();
    let linked_positions = bonds
        .iter()
        .filter_map(|(first, second)| {
            let endpoint = if first.0 == residue.reference.chain
                && first.1 == residue.reference.number
                && first.2.starts_with('O')
                && (second.0 != residue.reference.chain || second.1 != residue.reference.number)
            {
                Some(first)
            } else if second.0 == residue.reference.chain
                && second.1 == residue.reference.number
                && second.2.starts_with('O')
                && (first.0 != residue.reference.chain || first.1 != residue.reference.number)
            {
                Some(second)
            } else {
                None
            }?;
            endpoint.2[1..].parse::<u8>().ok()
        })
        .collect::<BTreeSet<_>>();
    let _ = all;
    let mut candidates = templates
        .glycan_names()
        .filter(|name| suffixes.iter().any(|suffix| name.ends_with(suffix)))
        .filter_map(|name| {
            let template = templates.glycan(name)?;
            let template_heavy = template
                .atoms
                .iter()
                .filter(|atom| atom.element != 1)
                .map(|atom| atom.name.as_str())
                .collect::<HashSet<_>>();
            if !actual_names.is_subset(&template_heavy) {
                return None;
            }
            let substituted = template
                .atoms
                .iter()
                .filter(|atom| atom.atom_type == "Os")
                .filter_map(|atom| atom.name.strip_prefix('O')?.parse::<u8>().ok())
                .filter(|position| *position != 5)
                .collect::<BTreeSet<_>>();
            let mismatch = substituted.symmetric_difference(&linked_positions).count();
            Some((mismatch, name.to_string()))
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates
        .first()
        .map(|(_, name)| name.clone())
        .ok_or_else(|| BuildError::UnsupportedResidue {
            residue: residue.reference.to_string(),
            reason: "no GLYCAM template matches its heavy-atom names and linkages".into(),
        })
}

#[derive(Clone, Copy)]
struct ResidueTransform {
    source_origin: Vec3,
    target_origin: Vec3,
    source_basis: [[f64; 3]; 3],
    target_basis: [[f64; 3]; 3],
}

impl ResidueTransform {
    fn from_atom_indices(
        template: &Template,
        residue: &PdbResidue,
        anchor_indices: &[usize],
    ) -> Result<Self> {
        let actual = residue
            .atoms
            .iter()
            .map(|atom| (atom.name.as_str(), atom.position))
            .collect::<HashMap<_, _>>();
        let anchors = anchor_indices
            .iter()
            .filter_map(|index| template.atoms.get(*index))
            .filter(|atom| atom.element != 1 && actual.contains_key(atom.name.as_str()))
            .collect::<Vec<_>>();
        if anchors.is_empty() {
            return Err(BuildError::MissingHeavyAtoms {
                residue: residue.reference.to_string(),
                atoms: "at least one coordinate anchor is required".into(),
            });
        }
        if anchors.len() == 1 {
            return Ok(Self {
                source_origin: anchors[0].position,
                target_origin: actual[anchors[0].name.as_str()],
                source_basis: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                target_basis: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            });
        }
        if anchors.len() == 2 {
            return Ok(Self {
                source_origin: anchors[0].position,
                target_origin: actual[anchors[0].name.as_str()],
                source_basis: basis_from_two(anchors[0].position, anchors[1].position),
                target_basis: basis_from_two(
                    actual[anchors[0].name.as_str()],
                    actual[anchors[1].name.as_str()],
                ),
            });
        }
        for second in 1..anchors.len() {
            for third in second + 1..anchors.len() {
                if let (Some(source_basis), Some(target_basis)) = (
                    basis(
                        anchors[0].position,
                        anchors[second].position,
                        anchors[third].position,
                    ),
                    basis(
                        actual[anchors[0].name.as_str()],
                        actual[anchors[second].name.as_str()],
                        actual[anchors[third].name.as_str()],
                    ),
                ) {
                    return Ok(Self {
                        source_origin: anchors[0].position,
                        target_origin: actual[anchors[0].name.as_str()],
                        source_basis,
                        target_basis,
                    });
                }
            }
        }
        Err(BuildError::InvalidPdb(format!(
            "{} has collinear template anchors",
            residue.reference
        )))
    }

    fn apply(self, point: Vec3) -> Vec3 {
        let delta = [
            point.x - self.source_origin.x,
            point.y - self.source_origin.y,
            point.z - self.source_origin.z,
        ];
        let local = [
            dot(delta, self.source_basis[0]),
            dot(delta, self.source_basis[1]),
            dot(delta, self.source_basis[2]),
        ];
        Vec3 {
            x: self.target_origin.x
                + local[0] * self.target_basis[0][0]
                + local[1] * self.target_basis[1][0]
                + local[2] * self.target_basis[2][0],
            y: self.target_origin.y
                + local[0] * self.target_basis[0][1]
                + local[1] * self.target_basis[1][1]
                + local[2] * self.target_basis[2][1],
            z: self.target_origin.z
                + local[0] * self.target_basis[0][2]
                + local[1] * self.target_basis[1][2]
                + local[2] * self.target_basis[2][2],
        }
    }
}

/// Build the local template frame around a hydrogen's bonded heavy atom.
///
/// A residue-wide transform is wrong whenever the experimental side-chain
/// rotamer differs from the library rotamer.  Walking the template bond graph
/// produces anchors at the parent atom and its nearest heavy-atom neighbors,
/// preserving the Amber/GLYCAM bond length and local tetrahedral/planar
/// geometry in the actual PDB conformation.
fn hydrogen_transform(
    template: &Template,
    residue: &PdbResidue,
    hydrogen_index: usize,
) -> Option<ResidueTransform> {
    if template.atoms.get(hydrogen_index)?.element != 1 {
        return None;
    }
    let actual_names = residue
        .atoms
        .iter()
        .map(|atom| atom.name.as_str())
        .collect::<HashSet<_>>();
    let mut adjacency = vec![Vec::new(); template.atoms.len()];
    for &[first, second] in &template.bonds {
        adjacency[first].push(second);
        adjacency[second].push(first);
    }
    let parent = adjacency[hydrogen_index]
        .iter()
        .copied()
        .find(|index| template.atoms[*index].element != 1)?;
    let mut queue = VecDeque::from([parent]);
    let mut visited = HashSet::from([hydrogen_index, parent]);
    let mut anchors = Vec::new();
    while let Some(index) = queue.pop_front() {
        let atom = &template.atoms[index];
        if atom.element != 1 && actual_names.contains(atom.name.as_str()) {
            anchors.push(index);
        }
        for &neighbor in &adjacency[index] {
            if visited.insert(neighbor) {
                queue.push_back(neighbor);
            }
        }
    }
    ResidueTransform::from_atom_indices(template, residue, &anchors).ok()
}

/// Construct missing glycan hydrogens from the actual heavy-atom coordination.
///
/// A library-wide rigid transform can point a hydrogen into the ring when a
/// carbohydrate puckering angle differs from the PREP conformer.  The local
/// rules below preserve each template bond length while deriving directions
/// from the experimental heavy atoms.  One-heavy-neighbor groups (OH, CH3)
/// retain their template cone geometry and sample its free torsion to avoid
/// nonbonded heavy-atom clashes.
fn glycan_hydrogen_positions(
    template: &Template,
    residue: &PdbResidue,
    residues: &[PdbResidue],
) -> HashMap<usize, Vec3> {
    let actual = residue
        .atoms
        .iter()
        .map(|atom| (atom.name.as_str(), atom.position))
        .collect::<HashMap<_, _>>();
    let environment = residues
        .iter()
        .flat_map(|item| item.atoms.iter())
        .filter(|atom| atom.element != "H" && atom.element != "D")
        .map(|atom| atom.position)
        .collect::<Vec<_>>();
    let mut adjacency = vec![Vec::new(); template.atoms.len()];
    for &[first, second] in &template.bonds {
        adjacency[first].push(second);
        adjacency[second].push(first);
    }
    let mut by_parent = BTreeMap::<usize, Vec<usize>>::new();
    for (index, atom) in template.atoms.iter().enumerate() {
        if atom.element == 1
            && !actual.contains_key(atom.name.as_str())
            && let Some(parent) = adjacency[index]
                .iter()
                .copied()
                .find(|neighbor| template.atoms[*neighbor].element != 1)
        {
            by_parent.entry(parent).or_default().push(index);
        }
    }

    let mut positions = HashMap::new();
    for (parent, hydrogens) in by_parent {
        let Some(&parent_position) = actual.get(template.atoms[parent].name.as_str()) else {
            continue;
        };
        let heavy_neighbors = adjacency[parent]
            .iter()
            .copied()
            .filter(|neighbor| template.atoms[*neighbor].element != 1)
            .filter_map(|neighbor| {
                actual
                    .get(template.atoms[neighbor].name.as_str())
                    .copied()
                    .map(|position| (neighbor, position))
            })
            .collect::<Vec<_>>();
        let mut coordination_neighbors = heavy_neighbors.clone();
        if heavy_neighbors.is_empty() {
            let mut external = environment
                .iter()
                .copied()
                .filter_map(|position| {
                    let distance2 = parent_position.distance2(position);
                    (distance2 > 0.8f64.powi(2) && distance2 < 1.85f64.powi(2))
                        .then_some((distance2, position))
                })
                .collect::<Vec<_>>();
            external.sort_by(|left, right| left.0.total_cmp(&right.0));
            if let Some((_, position)) = external.first() {
                coordination_neighbors.push((usize::MAX, *position));
            }
        }
        if hydrogens.len() == 1 && heavy_neighbors.len() >= 2 {
            // Linked GLYCAM templates omit the substituent belonging to the
            // adjacent residue.  Recover that heavy neighbor from its
            // covalent-distance coordinate so anomeric CH geometry uses all
            // three substituents, not only the two atoms in this template.
            for &position in &environment {
                let distance2 = parent_position.distance2(position);
                if distance2 > 0.8f64.powi(2)
                    && distance2 < 1.85f64.powi(2)
                    && coordination_neighbors
                        .iter()
                        .all(|(_, known)| known.distance2(position) > 1.0e-8)
                {
                    coordination_neighbors.push((usize::MAX, position));
                }
            }
        }

        let placed = if coordination_neighbors.len() == 1 {
            place_one_heavy_neighbor_group(
                template,
                parent,
                &hydrogens,
                coordination_neighbors[0],
                parent_position,
                &environment,
            )
        } else if hydrogens.len() == 1 && coordination_neighbors.len() >= 2 {
            place_single_coordinated_hydrogen(
                template,
                parent,
                hydrogens[0],
                &coordination_neighbors,
                parent_position,
            )
            .map(|position| vec![(hydrogens[0], position)])
        } else if hydrogens.len() == 2 && heavy_neighbors.len() == 2 {
            place_tetrahedral_pair(
                template,
                parent,
                &hydrogens,
                &heavy_neighbors,
                parent_position,
            )
        } else {
            None
        };

        if let Some(placed) = placed {
            positions.extend(placed);
        } else {
            for hydrogen in hydrogens {
                if let Some(transform) = hydrogen_transform(template, residue, hydrogen) {
                    positions.insert(hydrogen, transform.apply(template.atoms[hydrogen].position));
                }
            }
        }
    }
    positions
}

fn place_single_coordinated_hydrogen(
    template: &Template,
    parent: usize,
    hydrogen: usize,
    heavy_neighbors: &[(usize, Vec3)],
    parent_position: Vec3,
) -> Option<Vec3> {
    let direction = if heavy_neighbors.len() >= 3 {
        // Three substituents define the face opposite the missing vertex of
        // the local tetrahedron.  Point away from that face; this remains
        // stable when ring puckering distorts the individual bond angles.
        let first = heavy_neighbors[0].1;
        let second = heavy_neighbors[1].1;
        let third = heavy_neighbors[2].1;
        let mut normal = normalize(cross(
            [second.x - first.x, second.y - first.y, second.z - first.z],
            [third.x - first.x, third.y - first.y, third.z - first.z],
        ))?;
        let toward_face = [
            (first.x + second.x + third.x) / 3.0 - parent_position.x,
            (first.y + second.y + third.y) / 3.0 - parent_position.y,
            (first.z + second.z + third.z) / 3.0 - parent_position.z,
        ];
        if dot(normal, toward_face) > 0.0 {
            for value in &mut normal {
                *value = -*value;
            }
        }
        normal
    } else {
        let mut opposite = [0.0; 3];
        for (_, position) in heavy_neighbors {
            let neighbor = direction(parent_position, *position)?;
            for axis in 0..3 {
                opposite[axis] -= neighbor[axis];
            }
        }
        normalize(opposite)?
    };
    let length = template.atoms[hydrogen]
        .position
        .distance2(template.atoms[parent].position)
        .sqrt();
    Some(offset(parent_position, direction, length))
}

fn place_tetrahedral_pair(
    template: &Template,
    parent: usize,
    hydrogens: &[usize],
    heavy_neighbors: &[(usize, Vec3)],
    parent_position: Vec3,
) -> Option<Vec<(usize, Vec3)>> {
    let first = direction(parent_position, heavy_neighbors[0].1)?;
    let second = direction(parent_position, heavy_neighbors[1].1)?;
    let center = normalize([
        -first[0] - second[0],
        -first[1] - second[1],
        -first[2] - second[2],
    ])?;
    let normal = normalize(cross(first, second))?;
    let half_angle_cosine = ((1.0 + dot(first, second)) / 2.0).sqrt();
    let center_weight = (1.0 / (3.0 * half_angle_cosine)).clamp(0.0, 1.0);
    let normal_weight = (1.0 - center_weight * center_weight).sqrt();

    let template_first = direction(
        template.atoms[parent].position,
        template.atoms[heavy_neighbors[0].0].position,
    )?;
    let template_second = direction(
        template.atoms[parent].position,
        template.atoms[heavy_neighbors[1].0].position,
    )?;
    let template_normal = normalize(cross(template_first, template_second))?;
    let mut result = Vec::new();
    for (ordinal, &hydrogen) in hydrogens.iter().enumerate() {
        let template_hydrogen = direction(
            template.atoms[parent].position,
            template.atoms[hydrogen].position,
        )?;
        let signed = dot(template_hydrogen, template_normal);
        let sign = if signed.abs() > 1.0e-6 {
            signed.signum()
        } else if ordinal == 0 {
            1.0
        } else {
            -1.0
        };
        let direction = normalize([
            center_weight * center[0] + sign * normal_weight * normal[0],
            center_weight * center[1] + sign * normal_weight * normal[1],
            center_weight * center[2] + sign * normal_weight * normal[2],
        ])?;
        let length = template.atoms[hydrogen]
            .position
            .distance2(template.atoms[parent].position)
            .sqrt();
        result.push((hydrogen, offset(parent_position, direction, length)));
    }
    Some(result)
}

fn place_one_heavy_neighbor_group(
    template: &Template,
    parent: usize,
    hydrogens: &[usize],
    heavy_neighbor: (usize, Vec3),
    parent_position: Vec3,
    environment: &[Vec3],
) -> Option<Vec<(usize, Vec3)>> {
    let actual_axis = direction(parent_position, heavy_neighbor.1)?;
    let template_axis = if let Some(neighbor) = template.atoms.get(heavy_neighbor.0) {
        direction(template.atoms[parent].position, neighbor.position)?
    } else {
        [1.0, 0.0, 0.0]
    };
    let actual_basis = perpendicular_basis(actual_axis)?;
    let template_basis = perpendicular_basis(template_axis)?;
    let direct_neighbors = [parent_position, heavy_neighbor.1];
    let mut best: Option<(f64, Vec<(usize, Vec3)>)> = None;

    // Fifteen-degree increments are sufficient to resolve OH and methyl
    // clashes while keeping the result deterministic.
    for step in 0..24 {
        let rotation = step as f64 * std::f64::consts::TAU / 24.0;
        let mut candidate = Vec::new();
        for &hydrogen in hydrogens {
            let template_direction = direction(
                template.atoms[parent].position,
                template.atoms[hydrogen].position,
            )?;
            // GLYCAM hydroxyl and aliphatic one-neighbor groups are locally
            // tetrahedral: H-parent-heavy is about 109.47 degrees.  Use that
            // chemically defined cone angle directly; PREP Cartesian
            // reconstruction is only needed to preserve relative azimuths.
            let axial: f64 = -1.0 / 3.0;
            let template_x = dot(template_direction, template_basis[0]);
            let template_y = dot(template_direction, template_basis[1]);
            let angle = template_y.atan2(template_x) + rotation;
            let radial = (1.0 - axial * axial).sqrt();
            let direction = [
                axial * actual_axis[0]
                    + radial
                        * (angle.cos() * actual_basis[0][0] + angle.sin() * actual_basis[1][0]),
                axial * actual_axis[1]
                    + radial
                        * (angle.cos() * actual_basis[0][1] + angle.sin() * actual_basis[1][1]),
                axial * actual_axis[2]
                    + radial
                        * (angle.cos() * actual_basis[0][2] + angle.sin() * actual_basis[1][2]),
            ];
            let length = template.atoms[hydrogen]
                .position
                .distance2(template.atoms[parent].position)
                .sqrt();
            candidate.push((hydrogen, offset(parent_position, direction, length)));
        }
        let score = candidate
            .iter()
            .map(|(_, position)| {
                environment
                    .iter()
                    .filter(|other| {
                        direct_neighbors
                            .iter()
                            .all(|direct| direct.distance2(**other) > 1.0e-8)
                    })
                    .map(|other| {
                        let distance2 = position.distance2(*other).max(0.25);
                        1.0 / distance2.powi(6)
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score < *best_score)
        {
            best = Some((score, candidate));
        }
    }
    best.map(|(_, positions)| positions)
}

fn direction(origin: Vec3, point: Vec3) -> Option<[f64; 3]> {
    normalize([point.x - origin.x, point.y - origin.y, point.z - origin.z])
}

fn offset(origin: Vec3, direction: [f64; 3], distance: f64) -> Vec3 {
    Vec3 {
        x: origin.x + distance * direction[0],
        y: origin.y + distance * direction[1],
        z: origin.z + distance * direction[2],
    }
}

fn perpendicular_basis(axis: [f64; 3]) -> Option<[[f64; 3]; 2]> {
    let reference = if axis[2].abs() < 0.8 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let first = normalize(cross(axis, reference))?;
    let second = normalize(cross(axis, first))?;
    Some([first, second])
}

/// Place an internal peptide N-H from the actual C(i-1)-N(i)-CA(i) plane.
///
/// The amide nitrogen is trigonal planar.  The hydrogen direction is the
/// outward bisector opposite the unit vectors toward the previous carbonyl
/// carbon and the current alpha carbon.  This is the bond-length/angle
/// construction used by general hydrogen builders such as Hydride, and avoids
/// the false short H--C contacts produced by residue-local template alignment.
fn peptide_amide_hydrogen_position(
    residues: &[PdbResidue],
    residue_index: usize,
    template: &Template,
    hydrogen_index: usize,
) -> Option<Vec3> {
    let hydrogen = template.atoms.get(hydrogen_index)?;
    if hydrogen.element != 1 || hydrogen.name != "H" || residue_index == 0 {
        return None;
    }
    let residue = &residues[residue_index];
    let previous = &residues[residue_index - 1];
    if residue.reference.chain != previous.reference.chain
        || !PROTEIN_RESIDUES.contains(&residue.reference.name.as_str())
        || !PROTEIN_RESIDUES.contains(&previous.reference.name.as_str())
    {
        return None;
    }
    let nitrogen = residue.atoms.iter().find(|atom| atom.name == "N")?.position;
    let alpha = residue
        .atoms
        .iter()
        .find(|atom| atom.name == "CA")?
        .position;
    let carbonyl = previous
        .atoms
        .iter()
        .find(|atom| atom.name == "C")?
        .position;
    if nitrogen.distance2(carbonyl) > 2.0f64.powi(2) {
        return None;
    }
    let toward_carbonyl = normalize([
        carbonyl.x - nitrogen.x,
        carbonyl.y - nitrogen.y,
        carbonyl.z - nitrogen.z,
    ])?;
    let toward_alpha = normalize([
        alpha.x - nitrogen.x,
        alpha.y - nitrogen.y,
        alpha.z - nitrogen.z,
    ])?;
    let direction = normalize([
        -toward_carbonyl[0] - toward_alpha[0],
        -toward_carbonyl[1] - toward_alpha[1],
        -toward_carbonyl[2] - toward_alpha[2],
    ])?;
    let template_nitrogen = template.atom("N")?.1.position;
    let bond_length = hydrogen.position.distance2(template_nitrogen).sqrt();
    Some(Vec3 {
        x: nitrogen.x + direction[0] * bond_length,
        y: nitrogen.y + direction[1] * bond_length,
        z: nitrogen.z + direction[2] * bond_length,
    })
}

fn basis_from_two(origin: Vec3, second: Vec3) -> [[f64; 3]; 3] {
    let first = normalize([
        second.x - origin.x,
        second.y - origin.y,
        second.z - origin.z,
    ])
    .unwrap_or([1.0, 0.0, 0.0]);
    let reference = if first[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let second = normalize(cross(reference, first)).unwrap_or([0.0, 1.0, 0.0]);
    let third = cross(first, second);
    [first, second, third]
}

fn basis(origin: Vec3, second: Vec3, third: Vec3) -> Option<[[f64; 3]; 3]> {
    let first = normalize([
        second.x - origin.x,
        second.y - origin.y,
        second.z - origin.z,
    ])?;
    let raw = [third.x - origin.x, third.y - origin.y, third.z - origin.z];
    let projection = dot(raw, first);
    let second = normalize([
        raw[0] - projection * first[0],
        raw[1] - projection * first[1],
        raw[2] - projection * first[2],
    ])?;
    let third = cross(first, second);
    Some([first, second, third])
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length = dot(vector, vector).sqrt();
    (length > 1e-8).then(|| [vector[0] / length, vector[1] / length, vector[2] / length])
}

fn dot(first: [f64; 3], second: [f64; 3]) -> f64 {
    first[0] * second[0] + first[1] * second[1] + first[2] * second[2]
}

fn cross(first: [f64; 3], second: [f64; 3]) -> [f64; 3] {
    [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
}

fn find_residue_index(residues: &[PdbResidue], key: &crate::pdb::ResidueKey) -> Option<usize> {
    residues.iter().position(|residue| {
        residue.reference.chain == key.chain
            && residue.reference.number == key.number
            && residue.reference.insertion_code == key.insertion_code
    })
}

fn infer_cross_residue_bonds(
    residues: &[PdbResidue],
    residue_atom: &HashMap<(usize, String), usize>,
    atoms: &[Atom],
    bonds: &mut Vec<[usize; 2]>,
) {
    for first_residue in 0..residues.len() {
        for second_residue in first_residue + 1..residues.len() {
            for (first_name, second_name) in [
                ("O1", "C1"),
                ("O2", "C1"),
                ("O3", "C1"),
                ("O4", "C1"),
                ("O6", "C1"),
                ("O8", "C2"),
                ("ND2", "C1"),
                ("OG", "C1"),
                ("OG1", "C1"),
                ("OD1", "C1"),
                ("SG", "SG"),
            ] {
                for ((left_residue, left_name), (right_residue, right_name)) in [
                    ((first_residue, first_name), (second_residue, second_name)),
                    ((first_residue, second_name), (second_residue, first_name)),
                ] {
                    let Some(&left) = residue_atom.get(&(left_residue, left_name.into())) else {
                        continue;
                    };
                    let Some(&right) = residue_atom.get(&(right_residue, right_name.into())) else {
                        continue;
                    };
                    let cutoff: f64 = if first_name == "SG" { 2.3 } else { 1.9 };
                    if atoms[left].position.distance2(atoms[right].position) <= cutoff.powi(2) {
                        bonds.push([left, right]);
                    }
                }
            }
        }
    }
}

fn molecular_components(
    residue_count: usize,
    bonds: &[[usize; 2]],
    atoms: &[Atom],
) -> (usize, Vec<usize>) {
    let mut parent = (0..residue_count).collect::<Vec<_>>();
    fn root(parent: &mut [usize], value: usize) -> usize {
        if parent[value] != value {
            parent[value] = root(parent, parent[value]);
        }
        parent[value]
    }
    for [first, second] in bonds {
        let first = atoms[*first].residue;
        let second = atoms[*second].residue;
        let first_root = root(&mut parent, first);
        let second_root = root(&mut parent, second);
        parent[first_root] = second_root;
    }
    let mut remap = HashMap::new();
    let mut components = Vec::new();
    for residue in 0..residue_count {
        let root = root(&mut parent, residue);
        let next = remap.len();
        components.push(*remap.entry(root).or_insert(next));
    }
    (remap.len(), components)
}

type Enumerated = (Vec<Bond>, Vec<Angle>, Vec<Dihedral>, Vec<BTreeSet<usize>>);

fn enumerate_parameters(
    atoms: &[Atom],
    raw_bonds: &[[usize; 2]],
    parameters: &ParameterSet,
) -> Result<Enumerated> {
    let mut adjacency = vec![Vec::new(); atoms.len()];
    let mut bonds = Vec::new();
    for &[first, second] in raw_bonds {
        adjacency[first].push(second);
        adjacency[second].push(first);
        let parameter = parameters.bond(&atoms[first].atom_type, &atoms[second].atom_type)?;
        bonds.push(Bond {
            atoms: [first, second],
            force: parameter.force,
            length: parameter.length,
        });
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    let mut angles = Vec::new();
    for center in 0..atoms.len() {
        for left_index in 0..adjacency[center].len() {
            for right_index in left_index + 1..adjacency[center].len() {
                let left = adjacency[center][left_index];
                let right = adjacency[center][right_index];
                let parameter = parameters.angle(
                    &atoms[left].atom_type,
                    &atoms[center].atom_type,
                    &atoms[right].atom_type,
                )?;
                angles.push(Angle {
                    atoms: [left, center, right],
                    force: parameter.force,
                    radians: parameter.degrees.to_radians(),
                });
            }
        }
    }
    let mut proper_keys = BTreeSet::new();
    let mut dihedrals = Vec::new();
    for &[second, third] in raw_bonds {
        for &first in &adjacency[second] {
            if first == third {
                continue;
            }
            for &fourth in &adjacency[third] {
                if fourth == second || fourth == first {
                    continue;
                }
                let key = if first < fourth {
                    [first, second, third, fourth]
                } else {
                    [fourth, third, second, first]
                };
                if !proper_keys.insert(key) {
                    continue;
                }
                for parameter in parameters.dihedrals([
                    &atoms[key[0]].atom_type,
                    &atoms[key[1]].atom_type,
                    &atoms[key[2]].atom_type,
                    &atoms[key[3]].atom_type,
                ])? {
                    dihedrals.push(Dihedral {
                        atoms: key,
                        force: parameter.force,
                        periodicity: parameter.periodicity,
                        phase: parameter.phase_degrees.to_radians(),
                        improper: false,
                        scee: 1.2,
                        scnb: 2.0,
                    });
                }
            }
        }
    }
    for center in 0..atoms.len() {
        if adjacency[center].len() < 3 {
            continue;
        }
        for first in 0..adjacency[center].len() - 2 {
            for second in first + 1..adjacency[center].len() - 1 {
                for third in second + 1..adjacency[center].len() {
                    let peripheral = [
                        adjacency[center][first],
                        adjacency[center][second],
                        adjacency[center][third],
                    ];
                    if let Some(parameter_terms) = parameters.improper([
                        &atoms[peripheral[0]].atom_type,
                        &atoms[peripheral[1]].atom_type,
                        &atoms[center].atom_type,
                        &atoms[peripheral[2]].atom_type,
                    ]) {
                        for parameter in parameter_terms {
                            dihedrals.push(Dihedral {
                                atoms: [peripheral[0], peripheral[1], center, peripheral[2]],
                                force: parameter.force,
                                periodicity: parameter.periodicity,
                                phase: parameter.phase_degrees.to_radians(),
                                improper: true,
                                scee: 1.2,
                                scnb: 2.0,
                            });
                        }
                    }
                }
            }
        }
    }
    let mut exclusions = vec![BTreeSet::new(); atoms.len()];
    for (start, excluded) in exclusions.iter_mut().enumerate() {
        let mut queue = VecDeque::from([(start, 0usize)]);
        let mut visited = HashSet::from([start]);
        while let Some((current, depth)) = queue.pop_front() {
            if depth == 3 {
                continue;
            }
            for &neighbor in &adjacency[current] {
                if visited.insert(neighbor) {
                    excluded.insert(neighbor);
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }
    }
    Ok((bonds, angles, dihedrals, exclusions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::ResidueRef;

    #[test]
    fn hydrogen_placement_follows_the_local_side_chain() {
        let templates = TemplateSet::load().unwrap();
        let template = templates.protein("LEU", false, false).unwrap();
        let cd1 = template.atom("CD1").unwrap().0;
        let hydrogen = template
            .bonds
            .iter()
            .find_map(|bond| {
                let other = if bond[0] == cd1 {
                    bond[1]
                } else if bond[1] == cd1 {
                    bond[0]
                } else {
                    return None;
                };
                (template.atoms[other].element == 1).then_some(other)
            })
            .unwrap();
        let mut residue = PdbResidue {
            reference: ResidueRef {
                chain: "A".into(),
                name: "LEU".into(),
                number: 1,
                insertion_code: None,
            },
            atoms: template
                .atoms
                .iter()
                .enumerate()
                .filter(|(_, atom)| atom.element != 1)
                .map(|(index, atom)| crate::pdb::PdbAtom {
                    serial: index as u32 + 1,
                    name: atom.name.clone(),
                    residue_name: "LEU".into(),
                    chain: "A".into(),
                    residue_number: 1,
                    insertion_code: None,
                    element: "C".into(),
                    position: atom.position,
                })
                .collect(),
        };
        let moved_parent = residue
            .atoms
            .iter_mut()
            .find(|atom| atom.name == "CD1")
            .unwrap();
        moved_parent.position.x += 4.0;
        moved_parent.position.y -= 2.0;
        let parent = moved_parent.position;

        let transform = hydrogen_transform(template, &residue, hydrogen).unwrap();
        let placed = transform.apply(template.atoms[hydrogen].position);
        let expected = template.atoms[hydrogen]
            .position
            .distance2(template.atoms[cd1].position)
            .sqrt();
        assert!((placed.distance2(parent).sqrt() - expected).abs() < 1.0e-8);
    }

    #[test]
    fn preserves_cysteine_and_glycosylated_protein_residue_names() {
        let templates = TemplateSet::load().unwrap();
        assert_eq!(
            parameterized_residue_name("CYX", templates.protein("CYX", false, false).unwrap()),
            "CYX"
        );
        assert_eq!(
            parameterized_residue_name("NLN", templates.protein("NLN", false, false).unwrap()),
            "NLN"
        );
    }
}
