use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::carbohydrate::CarbohydrateLibrary;
use crate::fix::ComponentLibrary;
use crate::forcefield::{ParameterSet, Template, TemplateSet};
use crate::ligand::{ExplicitTerms, canonical};
use crate::model::{Angle, Atom, Bond, Dihedral, ParameterizedSystem, Residue, System, Vec3};
use crate::pdb::{self, PROTEIN_RESIDUES, ParsedPdb, PdbResidue};
use crate::report::BuildReport;
use crate::smirnoff::ForceField;
use crate::{BuildError, BuildOptions, BuildWarning, Result, Structure};

/// Reusable builder containing validated options and immutable force-field data.
pub struct SystemBuilder {
    options: BuildOptions,
    templates: TemplateSet,
    parameters: ParameterSet,
    /// GLYCAM residue types indexed by atom graph and stereochemistry.
    carbohydrates: CarbohydrateLibrary,
    /// Chemical Component Dictionary definitions for small molecules.
    components: ComponentLibrary,
    small_molecules: std::sync::OnceLock<std::result::Result<ForceField, String>>,
}

impl std::fmt::Debug for SystemBuilder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SystemBuilder")
            .field("options", &self.options)
            .field("components", &self.components.len())
            .finish_non_exhaustive()
    }
}

impl SystemBuilder {
    pub fn new(options: BuildOptions) -> Result<Self> {
        options.validate()?;
        let mut templates = TemplateSet::load()?;
        let parameters = ParameterSet::load()?;
        // GLYCAM-Web's library holds entries that need parameters GlySys does
        // not ship (ring conformers typed CG/OS/OH, ammonium N3 angles); they
        // are never chosen.
        templates.retain_parameterized_glycans(|template| parameters.covers(template));
        let carbohydrates = CarbohydrateLibrary::new(&templates);
        Ok(Self {
            options,
            templates,
            parameters,
            carbohydrates,
            components: ComponentLibrary::new(),
            small_molecules: std::sync::OnceLock::new(),
        })
    }

    /// Chemical component definitions used to parameterize ligands with
    /// OpenFF Sage 2.2.1 and AM1-BCC charges.
    pub fn with_components(mut self, components: ComponentLibrary) -> Self {
        self.components = components;
        self
    }

    pub fn components_mut(&mut self) -> &mut ComponentLibrary {
        &mut self.components
    }

    fn small_molecule_force_field(&self) -> Result<&ForceField> {
        self.small_molecules
            .get_or_init(|| ForceField::sage().map_err(|error| error.0))
            .as_ref()
            .map_err(|error| BuildError::ForceField(error.clone()))
    }

    pub fn options(&self) -> &BuildOptions {
        &self.options
    }

    /// Chemical component IDs in `contents` (ligands, cofactors, modified
    /// residues) whose CCD definitions preparation and repair can use.
    pub fn component_requests(&self, contents: &str) -> Result<Vec<String>> {
        crate::StructureFixer::new(crate::FixOptions {
            model: self.options.model,
            altloc: self.options.altloc,
            ..crate::FixOptions::default()
        })?
        .component_requests(contents)
    }

    pub fn prepare_pdb(&self, path: impl AsRef<Path>) -> Result<ParameterizedSystem> {
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
        self.inspect_pdb_str(&contents)
    }

    /// Inspect in-memory PDB text without solvating or writing files.
    pub fn inspect_pdb_str(&self, contents: &str) -> Result<BuildReport> {
        let parsed = pdb::parse(contents, &self.options)?;
        let mut warnings = parsed.warnings.clone();
        let protonation_decisions = self.protonation_decisions(&parsed, &mut warnings);
        // A residue is unsupported when no template, chemical component or
        // carbohydrate reading covers it; the carbohydrate reading says why.
        let carbohydrate_errors = self.carbohydrate_diagnostics(&parsed);
        let mut unsupported_residues = Vec::new();
        for residue in &parsed.residues {
            let name = residue.reference.name.as_str();
            let covered = PROTEIN_RESIDUES.contains(&name)
                || pdb::NUCLEIC_RESIDUES.contains(&name)
                || pdb::is_water(name)
                || pdb::is_free_ion(name)
                || pdb::is_metal_ion(name)
                || self.templates.glycan(name).is_some()
                || self.components.get(name).is_some();
            if let Some(reason) = carbohydrate_errors.get(&residue_key(residue))
                && (!covered || reason.contains(crate::carbohydrate::LIBRARY_REFUSAL))
            {
                warnings.push(BuildWarning::UnsupportedResidue(format!(
                    "{}: {reason}",
                    residue.reference
                )));
                unsupported_residues.push(residue.reference.clone());
            }
        }
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
            chains: parsed.chains.clone(),
            glycans: parsed.glycans.clone(),
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
            calcium_ions: 0,
            magnesium_ions: 0,
            solute_charge,
            total_charge: solute_charge,
            box_angstrom: [0.0; 3],
        })
    }

    pub fn prepare_pdb_str(&self, contents: &str) -> Result<ParameterizedSystem> {
        let input_sha256 = format!("{:x}", Sha256::digest(contents.as_bytes()));
        if self.options.repair {
            let (repaired, note) = self.repair(contents)?;
            // The repaired file holds only the selected model and altloc.
            let mut options = self.options.clone();
            options.model = 1;
            options.altloc = None;
            let structure = crate::read_pdb_str(&repaired, &options)?;
            let mut prepared = self.prepare_structure_with_hash(&structure, input_sha256)?;
            prepared
                .report
                .warnings
                .insert(0, BuildWarning::StructureRepaired(note));
            return Ok(prepared);
        }
        let structure = crate::read_pdb_str(contents, &self.options)?;
        self.prepare_structure_with_hash(&structure, input_sha256)
    }

    /// Run the structure fixer as preparation does: internal gaps only,
    /// crystal water removed, every other repair enabled.
    fn repair(&self, contents: &str) -> Result<(String, String)> {
        let fixer = crate::StructureFixer::new(crate::FixOptions {
            model: self.options.model,
            altloc: self.options.altloc,
            protonation: self.options.protonation.clone(),
            missing_residues: crate::MissingResidues::Internal,
            keep_water: false,
            ..crate::FixOptions::default()
        })?
        .with_components(self.components.clone());
        let fixed = fixer.fix_pdb_str(contents)?;
        let report = &fixed.report;
        let note = format!(
            "structure repaired: {} heavy atoms rebuilt, {} residues modelled, {} modified residues replaced, {} hydrogens added",
            report.heavy_atoms_added,
            report.residues_added,
            report.replaced_residues.len(),
            report.hydrogens_added
        );
        Ok((fixed.pdb, note))
    }

    /// Parameterize an editable structure without serializing and reparsing it.
    pub fn prepare_structure(&self, structure: &Structure) -> Result<ParameterizedSystem> {
        let input_sha256 = format!(
            "{:x}",
            Sha256::digest(crate::write_pdb_string(structure).as_bytes())
        );
        self.prepare_structure_with_hash(structure, input_sha256)
    }

    fn prepare_structure_with_hash(
        &self,
        structure: &Structure,
        input_sha256: String,
    ) -> Result<ParameterizedSystem> {
        let parsed = &structure.parsed;
        let mut warnings = parsed.warnings.clone();
        let protonation_decisions = self.protonation_decisions(parsed, &mut warnings);
        let mut system = self.parameterize(parsed, &protonation_decisions, &mut warnings)?;
        let solute_charge = system.charge();
        let rounded_charge = solute_charge.round();
        // GLYCAM charges sum to an integer once each O-substituent's
        // correction is applied (`apply_substituent_charges`); a fractional
        // total means a residue was given the wrong linkage form.
        let charge_tolerance = 1.0e-3;
        if (solute_charge - rounded_charge).abs() > charge_tolerance {
            // Name the residues whose charge is fractional (usually a glycan
            // template chosen without one of its linkages).
            let fractional = system
                .residues
                .iter()
                .filter_map(|residue| {
                    let charge: f64 = system.atoms[residue.atom_range()]
                        .iter()
                        .map(|a| a.charge)
                        .sum();
                    ((charge - charge.round()).abs() > 0.01).then(|| {
                        format!(
                            "{}:{}{} {} {charge:+.3}",
                            residue.chain,
                            residue.number,
                            residue.insertion_code.map(String::from).unwrap_or_default(),
                            residue.name
                        )
                    })
                })
                .take(12)
                .collect::<Vec<_>>();
            let detail = if fractional.is_empty() {
                String::new()
            } else {
                format!(" (fractional residues: {})", fractional.join(", "))
            };
            return Err(BuildError::NonIntegralCharge(solute_charge, detail));
        }
        if self.options.add_water {
            crate::solvate::solvate_and_ionize(
                &mut system,
                &self.templates,
                &self.parameters,
                &self.options,
            )?;
        }
        if self.options.ecc {
            let scaled = crate::ecc::apply(
                &mut system,
                |name| PROTEIN_RESIDUES.contains(&name),
                &mut warnings,
            );
            warnings.push(BuildWarning::ChargeScaling(format!(
                "electronic continuum correction (prosECCo75 conventions, ions v20221028): {} ion(s) with scaled-charge models, {} charged carbohydrate block(s) at {} of their charge; water is TIP3P",
                scaled.ions,
                scaled.groups,
                crate::ecc::SCALE
            )));
        }
        let report = BuildReport {
            input_sha256,
            force_field_versions: crate::report::force_field_versions(),
            output_sha256: BTreeMap::new(),
            options: self.options.clone(),
            selected_model: self.options.model,
            chains: parsed.chains.clone(),
            glycans: parsed.glycans.clone(),
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
            calcium_ions: system.calcium_count,
            magnesium_ions: system.magnesium_count,
            solute_charge,
            total_charge: system.charge(),
            box_angstrom: system.box_angstrom,
        };
        Ok(ParameterizedSystem {
            system,
            report,
            metadata: structure.metadata().clone(),
        })
    }

    /// Why each residue that is not a GLYCAM carbohydrate could not be read
    /// as one (see `resolve_carbohydrates`).
    fn carbohydrate_diagnostics(&self, parsed: &ParsedPdb) -> HashMap<pdb::ResidueKey, String> {
        let mut kept = parsed
            .residues
            .iter()
            .filter(|residue| {
                !pdb::is_water(&residue.reference.name)
                    && !pdb::is_free_ion(&residue.reference.name)
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut declared = declared_atom_bonds(parsed);
        let mut inferred = inferred_glycosylation_links(&kept, &declared);
        let mut links = parsed.links.clone();
        resolve_carbohydrates(
            &mut kept,
            &mut declared,
            &mut links,
            &mut inferred,
            &self.carbohydrates,
            &mut Vec::new(),
        )
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
        let mut declared_bonds = declared_atom_bonds(parsed);
        let mut inferred_links = inferred_glycosylation_links(&kept, &declared_bonds);
        let mut links = parsed.links.clone();
        let carbohydrate_errors = resolve_carbohydrates(
            &mut kept,
            &mut declared_bonds,
            &mut links,
            &mut inferred_links,
            &self.carbohydrates,
            warnings,
        );
        for link in &inferred_links {
            declared_bonds.push((
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
            ));
        }
        normalize_special_residues(&mut kept, &declared_bonds, protonation, warnings);
        normalize_disulfides(&mut kept, &parsed.ssbonds);
        let terminal = protein_terminal_flags(&kept);
        let nucleic = nucleic_terminal_flags(&kept);
        for (index, residue) in kept.iter_mut().enumerate() {
            // Amber has no 5'-phosphate terminal residues: drop the phosphate.
            if nucleic.get(&index).is_some_and(|(five, _)| *five)
                && residue.atoms.iter().any(|atom| atom.name == "P")
            {
                residue.atoms.retain(|atom| {
                    !matches!(
                        atom.name.as_str(),
                        "P" | "OP1" | "OP2" | "OP3" | "HOP3" | "HOP2"
                    )
                });
                warnings.push(BuildWarning::TerminalPhosphateRemoved(format!(
                    "{}",
                    residue.reference
                )));
            }
        }

        let mut atoms = Vec::new();
        let mut residues = Vec::new();
        let mut bonds = Vec::<[usize; 2]>::new();
        let mut serial_to_atom = HashMap::new();
        let mut residue_atom = HashMap::<(usize, String), usize>::new();
        let mut explicit = ExplicitTerms::default();

        for (residue_index, residue) in kept.iter().enumerate() {
            let name = residue.reference.name.as_str();
            let (n_terminal, c_terminal) =
                terminal.get(&residue_index).copied().unwrap_or_default();
            let is_nucleic = pdb::NUCLEIC_RESIDUES.contains(&name);
            let is_ion = pdb::is_metal_ion(name);
            // A GLYCAM code is used only when the residue's heavy atoms fit
            // it; otherwise a same-named CCD component (e.g. ASA) wins.
            let glycam_fits = |template: &Template| {
                residue
                    .atoms
                    .iter()
                    .filter(|atom| atom.element != "H" && atom.element != "D")
                    .all(|atom| template.atom(&atom.name).is_some())
            };
            let chosen = if PROTEIN_RESIDUES.contains(&name) {
                Ok(self.templates.protein(name, n_terminal, c_terminal))
            } else if is_nucleic {
                let (five, three) = nucleic.get(&residue_index).copied().unwrap_or((true, true));
                Ok(self.templates.nucleic(name, five, three))
            } else if is_ion {
                Ok(self.templates.ion(&name.to_ascii_uppercase()))
            } else if let Some(reason) = carbohydrate_errors
                .get(&residue_key(residue))
                .filter(|reason| reason.contains(crate::carbohydrate::LIBRARY_REFUSAL))
            {
                // GLYCAM-Web's inverted galactofuranose: never prepared by name.
                Err(BuildError::UnsupportedResidue {
                    residue: residue.reference.to_string(),
                    reason: reason.clone(),
                })
            } else if let Some(template) = self.templates.glycan(name) {
                if glycam_fits(template) || self.components.get(name).is_none() {
                    Ok(Some(template))
                } else {
                    Err(BuildError::UnsupportedResidue {
                        residue: residue.reference.to_string(),
                        reason: format!("atoms do not match GLYCAM residue {name}"),
                    })
                }
            } else {
                Err(BuildError::UnsupportedResidue {
                    residue: residue.reference.to_string(),
                    reason: carbohydrate_errors
                        .get(&residue_key(residue))
                        .cloned()
                        .unwrap_or_else(|| {
                            "no ff14SB or GLYCAM06j-1 residue template, and not a carbohydrate"
                                .into()
                        }),
                })
            };
            let template = match chosen {
                Ok(template) => template,
                Err(error) => {
                    // Anything else with a chemical component definition
                    // is a small molecule for the generic force field.
                    let Some(component) = self.components.get(name) else {
                        return Err(error);
                    };
                    let first_atom = atoms.len();
                    let ligand = crate::ligand::parameterize(
                        residue,
                        component,
                        self.small_molecule_force_field()?,
                        residue_index,
                        &mut explicit,
                        first_atom,
                    )?;
                    for (offset, &source) in ligand.order.iter().enumerate() {
                        serial_to_atom.insert(residue.atoms[source].serial, first_atom + offset);
                        residue_atom.insert(
                            (residue_index, ligand.atoms[offset].name.clone()),
                            first_atom + offset,
                        );
                    }
                    bonds.extend(
                        ligand
                            .bonds
                            .iter()
                            .map(|&[a, b]| [first_atom + a, first_atom + b]),
                    );
                    residues.push(Residue {
                        name: residue.reference.name.clone(),
                        number: residue.reference.number,
                        insertion_code: residue.reference.insertion_code,
                        chain: residue.reference.chain.clone(),
                        first_atom,
                        atom_count: ligand.atoms.len(),
                        component: 0,
                    });
                    warnings.push(BuildWarning::SmallMoleculeParameterized(format!(
                        "{}: OpenFF Sage 2.2.1 with AM1-BCC charges, net charge {}",
                        residue.reference, ligand.charge
                    )));
                    atoms.extend(ligand.atoms);
                    continue;
                }
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
            let generated_glycan_hydrogens =
                if PROTEIN_RESIDUES.contains(&name) || is_nucleic || is_ion {
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
            // Experimental PDB models often omit a side-chain while retaining
            // a chemically usable backbone (5KZC contains exactly this case
            // for VAL H:211).  Rebuild side-chain heavy atoms and a terminal
            // OXT from the template frame.  Missing peptide backbone N/CA/C/O
            // atoms remain hard errors because inventing peptide connectivity
            // would hide a genuinely incomplete model.
            let protein_heavy_transform = if !missing.is_empty()
                && (PROTEIN_RESIDUES.contains(&name) || is_nucleic)
                && missing
                    .iter()
                    .all(|atom| !matches!(atom.as_str(), "N" | "CA" | "C" | "O"))
            {
                let anchors = template
                    .atoms
                    .iter()
                    .enumerate()
                    .filter(|(_, atom)| {
                        atom.element != 1 && actual.contains_key(atom.name.as_str())
                    })
                    .map(|(index, _)| index)
                    .collect::<Vec<_>>();
                ResidueTransform::from_atom_indices(template, residue, &anchors).ok()
            } else {
                None
            };
            if !missing.is_empty() && protein_heavy_transform.is_none() {
                return Err(BuildError::MissingHeavyAtoms {
                    residue: residue.reference.to_string(),
                    atoms: missing.join(", "),
                });
            }
            if protein_heavy_transform.is_some() {
                warnings.push(BuildWarning::ProteinHeavyAtomsReconstructed(format!(
                    "{}: {}",
                    residue.reference,
                    missing.join(", ")
                )));
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
                        .or(protein_heavy_transform)
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
        // Nucleic-acid phosphodiester bonds.
        for (first_index, pair) in kept.windows(2).enumerate() {
            let second_index = first_index + 1;
            if nucleic.get(&first_index).is_some_and(|(_, three)| !three)
                && nucleic.get(&second_index).is_some_and(|(five, _)| !five)
                && let (Some(&first), Some(&second)) = (
                    residue_atom.get(&(first_index, "O3'".into())),
                    residue_atom.get(&(second_index, "P".into())),
                )
                && pair[0].reference.chain == pair[1].reference.chain
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
        for link in links.iter().chain(&inferred_links) {
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
        // Explicit connectivity is authoritative.  GLYCAM/GMML GAG exports
        // provide it for glycosidic and sulfate bonds; adding distance-based
        // contacts on top of that can create false O...O bonds.
        if parsed.conect.is_empty() && parsed.links.is_empty() {
            infer_cross_residue_bonds(&kept, &residue_atom, &atoms, &mut bonds);
        }
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
        // Metal ions use a nonbonded model: coordination LINK/CONECT records
        // are not covalent bonds.
        bonds.retain(|[first, second]| {
            !is_metal_element(atoms[*first].element) && !is_metal_element(atoms[*second].element)
        });
        for bond in &mut bonds {
            if bond[0] > bond[1] {
                bond.swap(0, 1);
            }
        }
        bonds.sort_unstable();
        bonds.dedup();
        apply_substituent_charges(&mut atoms, &residues, &bonds, &self.templates);

        let (component_count, component_by_residue) =
            molecular_components(kept.len(), &bonds, &atoms);
        order_molecular_components(&mut atoms, &mut residues, &mut bonds, &component_by_residue);
        let (bonds, angles, dihedrals, exclusions) =
            enumerate_parameters(&atoms, &bonds, &self.parameters, &explicit)?;
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
            calcium_count: 0,
            magnesium_count: 0,
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

pub(crate) type AtomLocator = (String, i32, String);
pub(crate) type DeclaredAtomBond = (AtomLocator, AtomLocator);

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

fn is_metal_element(element: u8) -> bool {
    matches!(element, 12 | 20 | 25 | 26 | 27 | 28 | 29 | 30 | 48 | 80)
}

/// (5'-terminal, 3'-terminal) for nucleotides, from O3'-P connectivity.
fn nucleic_terminal_flags(residues: &[PdbResidue]) -> HashMap<usize, (bool, bool)> {
    let position = |residue: &PdbResidue, name: &str| {
        residue
            .atoms
            .iter()
            .find(|atom| atom.name == name)
            .map(|atom| atom.position)
    };
    let mut flags = HashMap::new();
    for (index, residue) in residues.iter().enumerate() {
        if pdb::NUCLEIC_RESIDUES.contains(&residue.reference.name.as_str()) {
            flags.insert(index, (true, true));
        }
    }
    for index in 1..residues.len() {
        let (previous, current) = (&residues[index - 1], &residues[index]);
        if !flags.contains_key(&(index - 1))
            || !flags.contains_key(&index)
            || previous.reference.chain != current.reference.chain
        {
            continue;
        }
        if let (Some(o3), Some(p)) = (position(previous, "O3'"), position(current, "P"))
            && o3.distance2(p) < 2.2 * 2.2
        {
            flags.get_mut(&(index - 1)).unwrap().1 = false;
            flags.get_mut(&index).unwrap().0 = false;
        }
    }
    flags
}

fn protein_terminal_flags(residues: &[PdbResidue]) -> HashMap<usize, (bool, bool)> {
    let position = |residue: &PdbResidue, name: &str| {
        residue
            .atoms
            .iter()
            .find(|atom| atom.name == name)
            .map(|atom| atom.position)
    };
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
        // A missing peptide bond (C-N beyond 2 Å) splits the chain into
        // separately capped segments, as the structure fixer does.
        let mut segments: Vec<Vec<usize>> = Vec::new();
        for &index in indices {
            let continues = segments
                .last()
                .and_then(|segment| segment.last())
                .is_some_and(|&previous| {
                    previous + 1 == index
                        && match (
                            position(&residues[previous], "C"),
                            position(&residues[index], "N"),
                        ) {
                            (Some(c), Some(n)) => c.distance2(n) <= 2.0 * 2.0,
                            _ => true,
                        }
                });
            if continues {
                segments.last_mut().unwrap().push(index);
            } else {
                segments.push(vec![index]);
            }
        }
        for segment in segments {
            for (position, &index) in segment.iter().enumerate() {
                result.insert(index, (position == 0, position + 1 == segment.len()));
            }
        }
    }
    result
}

fn residue_key(residue: &PdbResidue) -> pdb::ResidueKey {
    pdb::ResidueKey {
        chain: residue.reference.chain.clone(),
        number: residue.reference.number,
        insertion_code: residue.reference.insertion_code,
    }
}

/// Read every carbohydrate residue from its atoms (see [`crate::carbohydrate`]):
/// rename it and its atoms to the GLYCAM residue its graph, stereocentres and
/// linkages fit, and split off an aglycone (ROH, OME) or O-substituents
/// (SO3, MEX, ACX) into their own GLYCAM residues.  Links and declared bonds
/// are carried over to the new names, and bonds to other residues found from
/// the coordinates are added as links.  Returns, for residues that could not
/// be read, why.
fn resolve_carbohydrates(
    residues: &mut Vec<PdbResidue>,
    declared: &mut [DeclaredAtomBond],
    links: &mut [pdb::DeclaredLink],
    inferred: &mut Vec<pdb::DeclaredLink>,
    library: &CarbohydrateLibrary,
    warnings: &mut Vec<BuildWarning>,
) -> HashMap<pdb::ResidueKey, String> {
    use crate::carbohydrate::{atomic_number, inter_residue_bond_limit};

    let candidate = |residue: &PdbResidue| {
        let name = residue.reference.name.as_str();
        !PROTEIN_RESIDUES.contains(&name)
            && !pdb::NUCLEIC_RESIDUES.contains(&name)
            && !pdb::is_water(name)
            && !pdb::is_free_ion(name)
            && !pdb::is_metal_ion(name)
            && !matches!(name, "ROH" | "OME" | "SO3" | "MEX" | "ACX" | "TBT")
    };
    let bonding = |residue: &PdbResidue| {
        let name = residue.reference.name.as_str();
        !pdb::is_water(name) && !pdb::is_free_ion(name) && !pdb::is_metal_ion(name)
    };
    // Heavy atoms of every residue that can bond, on a 2.5 A grid.
    let cell = |position: Vec3| {
        (
            (position.x / 2.5).floor() as i64,
            (position.y / 2.5).floor() as i64,
            (position.z / 2.5).floor() as i64,
        )
    };
    let mut grid: HashMap<(i64, i64, i64), Vec<(usize, usize)>> = HashMap::new();
    for (residue_index, residue) in residues.iter().enumerate() {
        if !bonding(residue) {
            continue;
        }
        for (atom_index, atom) in residue.atoms.iter().enumerate() {
            let element = atomic_number(&atom.element);
            if element > 1 && !is_metal_element(element) {
                grid.entry(cell(atom.position))
                    .or_default()
                    .push((residue_index, atom_index));
            }
        }
    }
    let locate = |residues: &[PdbResidue], key: (&str, i32, &str)| {
        residues
            .iter()
            .enumerate()
            .find_map(|(residue_index, residue)| {
                (residue.reference.chain == key.0 && residue.reference.number == key.1)
                    .then(|| {
                        residue
                            .atoms
                            .iter()
                            .position(|atom| atom.name == key.2)
                            .map(|atom_index| (residue_index, atom_index))
                    })
                    .flatten()
            })
    };

    // Bonds between residues: (residue, atom) pairs, from coordinates and records.
    let mut cross = BTreeSet::<((usize, usize), (usize, usize))>::new();
    for (residue_index, residue) in residues.iter().enumerate() {
        if !candidate(residue) {
            continue;
        }
        for (atom_index, atom) in residue.atoms.iter().enumerate() {
            let element = atomic_number(&atom.element);
            if element <= 1 || is_metal_element(element) {
                continue;
            }
            let (cx, cy, cz) = cell(atom.position);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        for &(other_residue, other_atom) in
                            grid.get(&(cx + dx, cy + dy, cz + dz)).into_iter().flatten()
                        {
                            if other_residue == residue_index {
                                continue;
                            }
                            let other = &residues[other_residue].atoms[other_atom];
                            let limit =
                                inter_residue_bond_limit(element, atomic_number(&other.element));
                            if atom.position.distance2(other.position) < limit * limit {
                                let pair =
                                    ((residue_index, atom_index), (other_residue, other_atom));
                                cross.insert(if pair.0 < pair.1 {
                                    pair
                                } else {
                                    (pair.1, pair.0)
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    for (first, second) in declared.iter() {
        if first.0 == second.0 && first.1 == second.1 {
            continue;
        }
        if let (Some(a), Some(b)) = (
            locate(residues, (&first.0, first.1, &first.2)),
            locate(residues, (&second.0, second.1, &second.2)),
        ) && (candidate(&residues[a.0]) || candidate(&residues[b.0]))
        {
            cross.insert(if a < b { (a, b) } else { (b, a) });
        }
    }
    // GLYCAM-Web builds galactofuranose as L-altrofuranose: rebuild those
    // residues from the corrected templates before they are read.
    crate::carbohydrate::rebuild_glycam_web_galactofuranoses(residues, &cross, library, warnings);
    let mut outside: HashMap<usize, Vec<Vec<Vec3>>> = HashMap::new();
    for &((ra, aa), (rb, ab)) in &cross {
        for (own, own_atom, other, other_atom) in [(ra, aa, rb, ab), (rb, ab, ra, aa)] {
            let position = residues[other].atoms[other_atom].position;
            let slots = outside
                .entry(own)
                .or_insert_with(|| vec![Vec::new(); residues[own].atoms.len()]);
            if slots[own_atom]
                .iter()
                .all(|known| known.distance2(position) > 1.0e-8)
            {
                slots[own_atom].push(position);
            }
        }
    }

    // Identify, then rebuild the residue list.
    let mut errors = HashMap::new();
    let mut located: HashMap<(usize, usize), (pdb::ResidueKey, String)> = HashMap::new();
    let mut rebuilt = Vec::with_capacity(residues.len());
    let mut changed = false;
    for (residue_index, residue) in residues.iter().enumerate() {
        let original_key = residue_key(residue);
        let keep =
            |rebuilt: &mut Vec<PdbResidue>,
             located: &mut HashMap<(usize, usize), (pdb::ResidueKey, String)>| {
                for (atom_index, atom) in residue.atoms.iter().enumerate() {
                    located.insert(
                        (residue_index, atom_index),
                        (original_key.clone(), atom.name.clone()),
                    );
                }
                rebuilt.push(residue.clone());
            };
        if !candidate(residue) {
            keep(&mut rebuilt, &mut located);
            continue;
        }
        let empty = vec![Vec::new(); residue.atoms.len()];
        let bonded = outside.get(&residue_index).unwrap_or(&empty);
        let identified = match library.identify(residue, bonded) {
            Ok(identified) => identified,
            Err(reason) => {
                errors.insert(original_key.clone(), reason);
                keep(&mut rebuilt, &mut located);
                continue;
            }
        };
        for note in &identified.stereo_notes {
            warnings.push(BuildWarning::GlycanStereochemistry(format!(
                "{} ({}): {note}",
                residue.reference, identified.template
            )));
        }
        let name = residue.reference.name.as_str();
        let renamed = identified
            .atoms
            .iter()
            .any(|(atom_index, new)| residue.atoms[*atom_index].name != *new);
        if identified.template == name && !renamed && identified.fragments.is_empty() {
            keep(&mut rebuilt, &mut located);
            continue;
        }
        changed = true;
        // Hydrogens are kept only when the input already used this GLYCAM
        // residue's naming; otherwise they are rebuilt from the template.
        let same_naming = !renamed && identified.template.get(1..) == name.get(1..);
        let mut sugar = PdbResidue {
            reference: crate::report::ResidueRef {
                name: identified.template.clone(),
                ..residue.reference.clone()
            },
            atoms: Vec::new(),
        };
        for (atom_index, new) in &identified.atoms {
            let mut atom = residue.atoms[*atom_index].clone();
            atom.name = new.clone();
            atom.residue_name = identified.template.clone();
            located.insert(
                (residue_index, *atom_index),
                (original_key.clone(), new.clone()),
            );
            sugar.atoms.push(atom);
        }
        if same_naming {
            for (atom_index, atom) in residue.atoms.iter().enumerate() {
                if matches!(atom.element.trim(), "H" | "D") {
                    let mut atom = atom.clone();
                    atom.residue_name = identified.template.clone();
                    located.insert(
                        (residue_index, atom_index),
                        (original_key.clone(), atom.name.clone()),
                    );
                    sugar.atoms.push(atom);
                }
            }
        }
        let mut used_codes = residues
            .iter()
            .filter(|other| {
                other.reference.chain == residue.reference.chain
                    && other.reference.number == residue.reference.number
            })
            .filter_map(|other| other.reference.insertion_code)
            .collect::<HashSet<_>>();
        let mut parts = Vec::new();
        for fragment in &identified.fragments {
            let code = ['R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                .into_iter()
                .find(|code| !used_codes.contains(code))
                .unwrap_or('Z');
            used_codes.insert(code);
            let key = pdb::ResidueKey {
                chain: residue.reference.chain.clone(),
                number: residue.reference.number,
                insertion_code: Some(code),
            };
            let mut part = PdbResidue {
                reference: crate::report::ResidueRef {
                    chain: key.chain.clone(),
                    name: fragment.residue.into(),
                    number: key.number,
                    insertion_code: Some(code),
                },
                atoms: Vec::new(),
            };
            for (atom_index, new) in &fragment.atoms {
                let mut atom = residue.atoms[*atom_index].clone();
                atom.name = new.clone();
                atom.residue_name = fragment.residue.into();
                atom.insertion_code = Some(code);
                located.insert((residue_index, *atom_index), (key.clone(), new.clone()));
                part.atoms.push(atom);
            }
            for (new, position) in &fragment.built {
                let template = residue.atoms.first().cloned();
                if let Some(mut atom) = template {
                    atom.serial = 0;
                    atom.name = new.clone();
                    atom.element = "O".into();
                    atom.residue_name = fragment.residue.into();
                    atom.insertion_code = Some(code);
                    atom.position = *position;
                    part.atoms.push(atom);
                }
            }
            inferred.push(pdb::DeclaredLink {
                first: original_key.clone(),
                first_atom: fragment.bond.0.clone(),
                second: key,
                second_atom: fragment.bond.1.clone(),
            });
            parts.push(part);
        }
        let mut note = format!(
            "{}: read from its atoms as GLYCAM {}",
            residue.reference, identified.template
        );
        if !identified.fragments.is_empty() {
            note.push_str(&format!(
                " + {}",
                identified
                    .fragments
                    .iter()
                    .map(|fragment| fragment.residue)
                    .collect::<Vec<_>>()
                    .join(" + ")
            ));
        }
        let renames = identified
            .atoms
            .iter()
            .filter(|(atom_index, new)| residue.atoms[*atom_index].name != **new)
            .map(|(atom_index, new)| format!("{} -> {new}", residue.atoms[*atom_index].name))
            .collect::<Vec<_>>();
        if !renames.is_empty() {
            note.push_str(&format!("; atoms {}", renames.join(", ")));
        }
        if identified.rebuilt_anomeric_oxygen {
            note.push_str(&format!(
                "; nothing was bonded to the anomeric carbon: its oxygen was rebuilt as ROH for the anomer {}",
                if identified.anomer_from_name {
                    "the residue name gives"
                } else {
                    "the coordinates give"
                }
            ));
        } else if identified.anomer_from_name {
            note.push_str("; a centre was too distorted to read, so the residue name decided");
        }
        warnings.push(BuildWarning::GlycanNameNormalized(note));
        // Aglycones precede their residue, as GLYCAM-Web writes ROH.
        let (before, after): (Vec<_>, Vec<_>) = parts
            .into_iter()
            .partition(|part| matches!(part.reference.name.as_str(), "ROH" | "OME"));
        rebuilt.extend(before);
        rebuilt.push(sugar);
        rebuilt.extend(after);
    }
    if !changed && cross.is_empty() {
        return errors;
    }

    // Carry records over to the new names, and add the bonds read from the
    // coordinates for carbohydrate residues.
    let translate = |key: &pdb::ResidueKey, atom: &str| {
        residues
            .iter()
            .enumerate()
            .find(|(_, residue)| residue_key(residue) == *key)
            .and_then(|(residue_index, residue)| {
                let atom_index = residue.atoms.iter().position(|item| item.name == atom)?;
                located.get(&(residue_index, atom_index)).cloned()
            })
    };
    for link in links.iter_mut().chain(inferred.iter_mut()) {
        if let Some((key, atom)) = translate(&link.first, &link.first_atom) {
            link.first = key;
            link.first_atom = atom;
        }
        if let Some((key, atom)) = translate(&link.second, &link.second_atom) {
            link.second = key;
            link.second_atom = atom;
        }
    }
    for (first, second) in declared.iter_mut() {
        for end in [first, second] {
            if let Some((residue_index, atom_index)) = locate(residues, (&end.0, end.1, &end.2))
                && let Some((key, atom)) = located.get(&(residue_index, atom_index))
            {
                end.0 = key.chain.clone();
                end.1 = key.number;
                end.2 = atom.clone();
            }
        }
    }
    let known = links
        .iter()
        .chain(inferred.iter())
        .map(|link| {
            let a = (link.first.clone(), link.first_atom.clone());
            let b = (link.second.clone(), link.second_atom.clone());
            if (
                a.0.chain.as_str(),
                a.0.number,
                a.0.insertion_code,
                a.1.as_str(),
            ) <= (
                b.0.chain.as_str(),
                b.0.number,
                b.0.insertion_code,
                b.1.as_str(),
            ) {
                (a, b)
            } else {
                (b, a)
            }
        })
        .collect::<HashSet<_>>();
    for ((ra, aa), (rb, ab)) in cross {
        if !candidate(&residues[ra]) && !candidate(&residues[rb]) {
            continue;
        }
        let (Some(a), Some(b)) = (located.get(&(ra, aa)), located.get(&(rb, ab))) else {
            continue;
        };
        let (a, b) = if (
            a.0.chain.as_str(),
            a.0.number,
            a.0.insertion_code,
            a.1.as_str(),
        ) <= (
            b.0.chain.as_str(),
            b.0.number,
            b.0.insertion_code,
            b.1.as_str(),
        ) {
            (a.clone(), b.clone())
        } else {
            (b.clone(), a.clone())
        };
        if !known.contains(&(a.clone(), b.clone())) {
            inferred.push(pdb::DeclaredLink {
                first: a.0,
                first_atom: a.1,
                second: b.0,
                second_atom: b.1,
            });
        }
    }
    *residues = rebuilt;
    errors
}

/// GLYCAM's charge rule for O-substituents.  A linked oxygen carries the
/// -0.194 that balances the +0.194 of a glycosyl residue; SO3 (-0.837), MEX
/// (+0.233) and ACX (+0.186) differ from that, so the oxygen they replace the
/// hydrogen of absorbs the difference (+0.031, -0.039, +0.008) and the pair
/// carries the group's formal charge.
fn apply_substituent_charges(
    atoms: &mut [Atom],
    residues: &[Residue],
    bonds: &[[usize; 2]],
    templates: &TemplateSet,
) {
    for (index, residue) in residues.iter().enumerate() {
        let formal = match residue.name.as_str() {
            "SO3" => -1.0,
            "MEX" | "ACX" => 0.0,
            _ => continue,
        };
        let Some(template) = templates.glycan(&residue.name) else {
            continue;
        };
        let group: f64 = template.atoms.iter().map(|atom| atom.charge).sum();
        let correction = formal - group + 0.194;
        let range = residue.atom_range();
        let oxygen = bonds.iter().find_map(|&[first, second]| {
            let (own, other) = if range.contains(&first) {
                (first, second)
            } else if range.contains(&second) {
                (second, first)
            } else {
                return None;
            };
            let _ = own;
            (!range.contains(&other) && atoms[other].element == 8 && atoms[other].residue != index)
                .then_some(other)
        });
        if let Some(oxygen) = oxygen {
            atoms[oxygen].charge += correction;
        }
    }
}

/// Protein-glycan bonds present in the coordinates but not declared by
/// LINK/CONECT records (common in older entries).
fn inferred_glycosylation_links(
    residues: &[PdbResidue],
    declared: &[DeclaredAtomBond],
) -> Vec<pdb::DeclaredLink> {
    let mut links = Vec::new();
    for protein in residues {
        let attachment = match protein.reference.name.as_str() {
            "ASN" | "NLN" => "ND2",
            "SER" | "OLS" => "OG",
            "THR" | "OLT" => "OG1",
            "HYP" | "OLP" => "OD1",
            _ => continue,
        };
        // Only inter-residue records count; CONECT often lists intra-residue bonds.
        let inter = |(a, b): &&DeclaredAtomBond| a.0 != b.0 || a.1 != b.1;
        let already = declared.iter().filter(inter).any(|(a, b)| {
            (a.0 == protein.reference.chain && a.1 == protein.reference.number && a.2 == attachment)
                || (b.0 == protein.reference.chain
                    && b.1 == protein.reference.number
                    && b.2 == attachment)
        });
        let Some(anchor) = protein.atoms.iter().find(|atom| atom.name == attachment) else {
            continue;
        };
        if already {
            continue;
        }
        for sugar in residues {
            if PROTEIN_RESIDUES.contains(&sugar.reference.name.as_str()) {
                continue;
            }
            // Bonded geometry, or a reducing-end sugar (no O1, C1 otherwise
            // unbonded) left within 3 Å of the attachment atom by a
            // low-resolution model.
            let reducing_end = sugar.atoms.iter().all(|atom| atom.name != "O1")
                && !declared.iter().filter(inter).any(|(a, b)| {
                    (a.0 == sugar.reference.chain && a.1 == sugar.reference.number && a.2 == "C1")
                        || (b.0 == sugar.reference.chain
                            && b.1 == sugar.reference.number
                            && b.2 == "C1")
                });
            let limit = if reducing_end { 3.0 } else { 1.75 };
            if let Some(c1) = sugar.atoms.iter().find(|atom| atom.name == "C1")
                && c1.position.distance2(anchor.position) < limit * limit
                && !links.iter().any(|link: &pdb::DeclaredLink| {
                    link.second.chain == sugar.reference.chain
                        && link.second.number == sugar.reference.number
                })
            {
                links.push(pdb::DeclaredLink {
                    first: pdb::ResidueKey {
                        chain: protein.reference.chain.clone(),
                        number: protein.reference.number,
                        insertion_code: protein.reference.insertion_code,
                    },
                    first_atom: attachment.into(),
                    second: pdb::ResidueKey {
                        chain: sugar.reference.chain.clone(),
                        number: sugar.reference.number,
                        insertion_code: sugar.reference.insertion_code,
                    },
                    second_atom: "C1".into(),
                });
            }
        }
    }
    // Glycosidic bonds: a carbohydrate carbon within 1.65 Å of an oxygen or
    // nitrogen of another carbohydrate residue.
    let is_sugar = |residue: &PdbResidue| {
        let name = residue.reference.name.as_str();
        !PROTEIN_RESIDUES.contains(&name)
            && !pdb::NUCLEIC_RESIDUES.contains(&name)
            && !pdb::is_water(name)
            && residue.atoms.iter().any(|atom| atom.name == "C1")
            && residue
                .atoms
                .iter()
                .any(|atom| atom.name == "O5" || atom.name == "O6")
    };
    let key = |residue: &PdbResidue| pdb::ResidueKey {
        chain: residue.reference.chain.clone(),
        number: residue.reference.number,
        insertion_code: residue.reference.insertion_code,
    };
    for (i, first) in residues.iter().enumerate() {
        if !is_sugar(first) {
            continue;
        }
        for carbon in first.atoms.iter().filter(|atom| atom.element == "C") {
            for (j, second) in residues.iter().enumerate() {
                if i == j || !is_sugar(second) {
                    continue;
                }
                for other in second
                    .atoms
                    .iter()
                    .filter(|atom| matches!(atom.element.as_str(), "O" | "N"))
                {
                    if carbon.position.distance2(other.position) >= 1.65 * 1.65 {
                        continue;
                    }
                    let declared_already = declared.iter().any(|(a, b)| {
                        (a.0 == first.reference.chain
                            && a.1 == first.reference.number
                            && a.2 == carbon.name
                            && b.0 == second.reference.chain
                            && b.1 == second.reference.number
                            && b.2 == other.name)
                            || (b.0 == first.reference.chain
                                && b.1 == first.reference.number
                                && b.2 == carbon.name
                                && a.0 == second.reference.chain
                                && a.1 == second.reference.number
                                && a.2 == other.name)
                    });
                    if !declared_already {
                        links.push(pdb::DeclaredLink {
                            first: key(first),
                            first_atom: carbon.name.clone(),
                            second: key(second),
                            second_atom: other.name.clone(),
                        });
                    }
                }
            }
        }
    }
    links
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
pub(crate) fn glycan_hydrogen_positions(
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

/// Amber topology files require every covalent molecule to occupy one
/// contiguous atom span. Deposited GLYCAM GAG PDBs can interleave sulfate and
/// terminal-cap residues with the carbohydrate chain, so normalize the
/// internal ordering after connectivity has been resolved. All atom-indexed
/// data is remapped together; coordinates and molecular connectivity remain
/// unchanged.
fn order_molecular_components(
    atoms: &mut Vec<Atom>,
    residues: &mut Vec<Residue>,
    bonds: &mut Vec<[usize; 2]>,
    component_by_residue: &[usize],
) {
    let mut residue_order = (0..residues.len()).collect::<Vec<_>>();
    residue_order.sort_by_key(|&residue| (component_by_residue[residue], residue));

    let mut residue_remap = vec![0usize; residues.len()];
    let mut atom_remap = vec![0usize; atoms.len()];
    let old_atoms = std::mem::take(atoms);
    let old_residues = std::mem::take(residues);
    let mut grouped_atoms = Vec::with_capacity(old_atoms.len());
    let mut grouped_residues = Vec::with_capacity(old_residues.len());

    for old_residue_index in residue_order {
        let new_residue_index = grouped_residues.len();
        residue_remap[old_residue_index] = new_residue_index;
        let old_residue = &old_residues[old_residue_index];
        let first_atom = grouped_atoms.len();
        for old_atom_index in
            old_residue.first_atom..old_residue.first_atom + old_residue.atom_count
        {
            atom_remap[old_atom_index] = grouped_atoms.len();
            let mut atom = old_atoms[old_atom_index].clone();
            atom.residue = new_residue_index;
            grouped_atoms.push(atom);
        }
        let mut residue = old_residue.clone();
        residue.first_atom = first_atom;
        residue.component = component_by_residue[old_residue_index];
        grouped_residues.push(residue);
    }

    for bond in &mut *bonds {
        bond[0] = atom_remap[bond[0]];
        bond[1] = atom_remap[bond[1]];
        if bond[0] > bond[1] {
            bond.swap(0, 1);
        }
    }
    bonds.sort_unstable();
    bonds.dedup();
    *atoms = grouped_atoms;
    *residues = grouped_residues;
}

type Enumerated = (Vec<Bond>, Vec<Angle>, Vec<Dihedral>, Vec<BTreeSet<usize>>);

fn enumerate_parameters(
    atoms: &[Atom],
    raw_bonds: &[[usize; 2]],
    parameters: &ParameterSet,
    explicit: &ExplicitTerms,
) -> Result<Enumerated> {
    // Terms touching a small molecule come from its own assignment; a term
    // spanning a small molecule and anything else would be a covalent link
    // the generic force field cannot parameterize.
    let covalent = |indices: &[usize]| -> Result<bool> {
        let inside = indices.iter().filter(|&&i| explicit.contains(i)).count();
        if inside == 0 {
            Ok(false)
        } else if inside == indices.len() {
            Ok(true)
        } else {
            Err(BuildError::UnsupportedResidue {
                residue: format!(
                    "atoms {}",
                    indices
                        .iter()
                        .map(|i| atoms[*i].name.clone())
                        .collect::<Vec<_>>()
                        .join("-")
                ),
                reason: "covalently bound small molecules are not supported".into(),
            })
        }
    };
    let mut adjacency = vec![Vec::new(); atoms.len()];
    let mut bonds = Vec::new();
    for &[first, second] in raw_bonds {
        adjacency[first].push(second);
        adjacency[second].push(first);
        if covalent(&[first, second])? {
            let bond = explicit
                .bonds
                .get(&canonical([first, second]))
                .ok_or_else(|| {
                    BuildError::ForceField(format!(
                        "no small-molecule bond {}-{}",
                        atoms[first].name, atoms[second].name
                    ))
                })?;
            bonds.push(*bond);
            continue;
        }
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
                if covalent(&[left, center, right])? {
                    let angle = explicit
                        .angles
                        .get(&canonical([left, center, right]))
                        .ok_or_else(|| {
                            BuildError::ForceField(format!(
                                "no small-molecule angle at {}",
                                atoms[center].name
                            ))
                        })?;
                    angles.push(*angle);
                    continue;
                }
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
                if covalent(&key)? {
                    let terms = explicit.propers.get(&canonical(key)).ok_or_else(|| {
                        BuildError::ForceField(format!(
                            "no small-molecule torsion at {}",
                            atoms[key[1]].name
                        ))
                    })?;
                    dihedrals.extend(terms.iter().copied());
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
                        scee: parameter.scee,
                        scnb: parameter.scnb,
                    });
                }
            }
        }
    }
    dihedrals.extend(explicit.impropers.iter().copied());
    for center in 0..atoms.len() {
        if adjacency[center].len() < 3 || explicit.contains(center) {
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
                                scee: parameter.scee,
                                scnb: parameter.scnb,
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
                    occupancy: 1.0,
                    b_factor: 0.0,
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
