//! Backend-independent, versioned scoring contracts. GPU layouts are private adapters.
use crate::geometry::{RigidTransform, TorsionUpdate, cross, dot, rotate_torsion, sub};
use crate::{AtomGroupMask, EnergyComponents, EnergyError, EnergyEvaluator, EnergyOptions, Result};
use glysys::{ParameterizedSystem, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub const MODEL_VERSION: &str = "amber-glycam-v2";
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentRole {
    Receptor,
    Glycan,
    Ligand,
    Water,
    Ion,
    Cofactor,
    Unknown,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalAnnotation {
    pub atom: usize,
    pub role: ComponentRole,
    pub element: u8,
    pub atom_type: String,
    pub formal_charge: Option<i32>,
    pub aromatic: Option<bool>,
    pub donor: Option<bool>,
    pub acceptor: Option<bool>,
    pub hydrogen_parent: Option<usize>,
    pub parameter_source: String,
    pub source_atom_id: Option<glysys::AtomId>,
    pub occupancy: Option<f64>,
    pub b_factor: Option<f64>,
    pub generated: Option<bool>,
    pub residue_name: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Boundary {
    NonPeriodic,
    Periodic { box_angstrom: [f64; 3] },
}
#[derive(Debug, Clone)]
pub struct GeometryGroup {
    pub kind: String,
    pub atoms: Vec<usize>,
}
#[derive(Debug, Clone)]
pub struct PreparedScene {
    pub system: Arc<ParameterizedSystem>,
    pub annotations: Vec<ChemicalAnnotation>,
    pub groups: BTreeMap<String, Vec<usize>>,
    pub options: EnergyOptions,
    pub boundary: Boundary,
    pub fingerprint: String,
    pub geometry_groups: Vec<GeometryGroup>,
}
pub fn component_roles(system: &ParameterizedSystem) -> Vec<ComponentRole> {
    let glycans: BTreeSet<_> = system
        .metadata()
        .glycan_trees
        .iter()
        .flat_map(|t| t.residue_ids.iter())
        .collect();
    system
        .atoms()
        .iter()
        .map(|a| {
            let r = &system.residues()[a.residue_index()];
            let id = glysys::ResidueId {
                chain: r.chain().into(),
                number: r.number(),
                insertion_code: r.insertion_code(),
            };
            if glycans.contains(&id) {
                ComponentRole::Glycan
            } else if matches!(r.name(), "HOH" | "WAT" | "TIP3" | "SOL") {
                ComponentRole::Water
            } else if r.atom_range().len() == 1
                && matches!(
                    a.element(),
                    3 | 9 | 11 | 12 | 17 | 19 | 20 | 25 | 26 | 27 | 28 | 29 | 30 | 35 | 53
                )
            {
                ComponentRole::Ion
            } else if matches!(
                r.name(),
                "ALA"
                    | "ARG"
                    | "ASN"
                    | "ASP"
                    | "ASH"
                    | "CYS"
                    | "CYM"
                    | "CYX"
                    | "GLN"
                    | "GLU"
                    | "GLH"
                    | "GLY"
                    | "HIS"
                    | "HID"
                    | "HIE"
                    | "HIP"
                    | "ILE"
                    | "LEU"
                    | "LYS"
                    | "LYN"
                    | "MET"
                    | "PHE"
                    | "PRO"
                    | "SER"
                    | "THR"
                    | "TRP"
                    | "TYR"
                    | "VAL"
                    | "ACE"
                    | "NME"
            ) {
                ComponentRole::Receptor
            } else {
                ComponentRole::Unknown
            }
        })
        .collect()
}
fn invalid(message: &str) -> EnergyError {
    EnergyError::InvalidConfiguration(message.into())
}
impl PreparedScene {
    pub fn new(
        system: Arc<ParameterizedSystem>,
        options: EnergyOptions,
        boundary: Boundary,
    ) -> Result<Self> {
        if !matches!(boundary, Boundary::NonPeriodic) {
            return Err(invalid(
                "periodic scoring is not implemented; a prepared water box does not enable PME",
            ));
        }
        EnergyEvaluator::new(&system, options.clone())?;
        let roles = component_roles(&system);
        let mut parents = vec![None; system.atom_count()];
        for b in system.bonds() {
            let [a, b] = b.atoms();
            if system.atoms()[a].element() == 1 {
                parents[a] = Some(b);
            }
            if system.atoms()[b].element() == 1 {
                parents[b] = Some(a);
            }
        }
        let mut annotations: Vec<ChemicalAnnotation> = system
            .atoms()
            .iter()
            .enumerate()
            .map(|(i, a)| ChemicalAnnotation {
                atom: i,
                role: roles[i],
                element: a.element(),
                atom_type: a.atom_type().into(),
                formal_charge: None,
                aromatic: None,
                donor: None,
                acceptor: None,
                hydrogen_parent: parents[i],
                source_atom_id: None,
                occupancy: None,
                b_factor: None,
                generated: None,
                residue_name: system.residues()[a.residue_index()].name().into(),
                parameter_source: "prepared topology; original parameter file not recorded".into(),
            })
            .collect();
        let mut groups = BTreeMap::new();
        for (name, role) in [
            ("receptor", ComponentRole::Receptor),
            ("glycan", ComponentRole::Glycan),
            ("water", ComponentRole::Water),
            ("ion", ComponentRole::Ion),
        ] {
            groups.insert(
                name.into(),
                roles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| (*r == role).then_some(i))
                    .collect(),
            );
        }
        let mut geometry_groups = Vec::new();
        for residue in system.residues() {
            let rings: Vec<Vec<&str>> = match residue.name() {
                "PHE" | "TYR" => vec![vec!["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]],
                "TRP" => vec![
                    vec!["CG", "CD1", "NE1", "CE2", "CD2"],
                    vec!["CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"],
                ],
                "HIS" | "HID" | "HIE" | "HIP" => vec![vec!["CG", "ND1", "CE1", "NE2", "CD2"]],
                _ => Vec::new(),
            };
            for ring in rings {
                let atoms = ring
                    .iter()
                    .filter_map(|name| {
                        residue
                            .atom_range()
                            .find(|&i| system.atoms()[i].name() == *name)
                    })
                    .collect::<Vec<_>>();
                if atoms.len() == ring.len() {
                    for &i in &atoms {
                        annotations[i].aromatic = Some(true);
                    }
                    geometry_groups.push(GeometryGroup {
                        kind: "aromatic_ring".into(),
                        atoms,
                    });
                }
            }
        }
        for bond in system.bonds() {
            let [a, b] = bond.atoms();
            for (heavy, h) in [(a, b), (b, a)] {
                if system.atoms()[h].element() == 1
                    && matches!(system.atoms()[heavy].element(), 7 | 8 | 16)
                {
                    annotations[heavy].donor = Some(true);
                }
            }
        }
        let mut scene = Self {
            system,
            annotations,
            groups,
            options,
            boundary,
            fingerprint: String::new(),
            geometry_groups,
        };
        scene.refresh_fingerprint()?;
        Ok(scene)
    }
    pub fn with_source(mut self, source: &glysys::Structure) -> Result<Self> {
        let mut lookup = BTreeMap::new();
        for a in source.iter_atoms() {
            let key = (
                a.residue.chain.clone(),
                a.residue.number,
                a.residue.insertion_code,
                a.name.to_owned(),
            );
            if lookup
                .insert(key, (a.id, a.occupancy, a.b_factor))
                .is_some()
            {
                return Err(invalid("ambiguous source atom identity"));
            }
        }
        for annotation in &mut self.annotations {
            let atom = &self.system.atoms()[annotation.atom];
            let residue = &self.system.residues()[atom.residue_index()];
            let key = (
                residue.chain().to_owned(),
                residue.number(),
                residue.insertion_code(),
                atom.name().to_owned(),
            );
            if let Some(&(id, occupancy, b_factor)) = lookup.get(&key) {
                annotation.source_atom_id = Some(id);
                annotation.occupancy = Some(occupancy);
                annotation.b_factor = Some(b_factor);
                annotation.generated = Some(false);
            } else {
                annotation.generated = Some(true);
            }
        }
        self.refresh_fingerprint()?;
        Ok(self)
    }
    pub fn set_group(&mut self, name: String, indices: Vec<usize>) -> Result<()> {
        if indices.iter().any(|&i| i >= self.system.atom_count())
            || indices.iter().collect::<BTreeSet<_>>().len() != indices.len()
        {
            return Err(invalid("invalid or repeated group atom"));
        }
        self.groups.insert(name, indices);
        self.refresh_fingerprint()
    }
    pub fn refresh_fingerprint(&mut self) -> Result<()> {
        use sha2::{Digest, Sha256};
        // Full immutable chemistry/options/reference coordinates; not just atom count.
        let bytes = format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}",
            self.system, self.options, self.boundary, self.annotations, self.groups
        );
        self.fingerprint = format!("{:x}", Sha256::digest(bytes.as_bytes()));
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub struct Pose {
    pub id: u64,
    pub conformer_id: Option<String>,
    pub coordinates: Vec<Vec3>,
    pub transform: RigidTransform,
    pub transformed_atoms: Vec<usize>,
    pub torsions: Vec<TorsionUpdate>,
}
impl Pose {
    pub fn cartesian(id: u64, coordinates: Vec<Vec3>) -> Self {
        Self {
            id,
            conformer_id: None,
            coordinates,
            transform: Default::default(),
            transformed_atoms: Vec::new(),
            torsions: Vec::new(),
        }
    }
    pub fn materialize(&self) -> Result<Vec<Vec3>> {
        let mut p = self.coordinates.clone();
        for t in &self.torsions {
            rotate_torsion(&mut p, t)?;
        }
        for &i in &self.transformed_atoms {
            if i >= p.len() {
                return Err(invalid("pose atom index"));
            }
            p[i] = self.transform.apply(p[i]);
        }
        Ok(p)
    }
}
#[derive(Debug, Clone, Default)]
pub struct PoseBatch {
    pub poses: Vec<Pose>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Unit {
    KcalPerMol,
    Dimensionless,
    Angstrom,
    Radian,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pattern {
    Bonded,
    Pair,
    AtomLocal,
    MultiStage,
    Restraint,
    Prior,
    Feature,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermDescriptor {
    pub id: String,
    pub version: u32,
    pub unit: Unit,
    pub pattern: Pattern,
    pub dependencies: Vec<String>,
    pub differentiable: bool,
    pub cpu: bool,
    pub webgpu: bool,
}
#[derive(Debug, Clone)]
pub struct TermValue {
    pub raw: f64,
    pub weighted: f64,
    pub unit: Unit,
}
#[derive(Debug, Clone)]
pub struct TermOutput {
    pub value: f64,
    pub gradient: Option<Vec<Vec3>>,
}
/// Native extensions are explicit compiled terms, never arbitrary shader strings.
pub trait ScoreTerm: Send + Sync {
    fn descriptor(&self) -> TermDescriptor;
    fn evaluate(
        &self,
        scene: &PreparedScene,
        coordinates: &[Vec3],
        gradient: bool,
    ) -> Result<TermOutput>;
}
#[derive(Clone)]
pub struct ScoreModel {
    pub id: String,
    pub version: u32,
    pub purpose: String,
    pub weights: BTreeMap<String, f64>,
    pub interaction: Option<(String, String)>,
    pub extensions: Vec<Arc<dyn ScoreTerm>>,
}
const NAMES: [&str; 9] = [
    "bonds",
    "angles",
    "proper_torsions",
    "improper_torsions",
    "van_der_waals",
    "electrostatics",
    "generalized_born",
    "surface_area",
    "restraints",
];
fn values(c: EnergyComponents) -> [f64; 9] {
    [
        c.bonds,
        c.angles,
        c.proper_torsions,
        c.improper_torsions,
        c.van_der_waals,
        c.electrostatics,
        c.generalized_born,
        c.surface_area,
        c.restraints,
    ]
}
impl ScoreModel {
    pub fn amber() -> Self {
        Self {
            id: MODEL_VERSION.into(),
            version: 2,
            purpose: "potential_energy".into(),
            weights: NAMES.iter().map(|n| (n.to_string(), 1.)).collect(),
            interaction: None,
            extensions: Vec::new(),
        }
    }
    pub fn interaction(first: &str, second: &str) -> Self {
        let mut m = Self::amber();
        m.purpose = "cross_interaction_energy".into();
        m.interaction = Some((first.into(), second.into()));
        m
    }
    pub fn descriptors(&self) -> Vec<TermDescriptor> {
        NAMES
            .iter()
            .enumerate()
            .map(|(i, n)| TermDescriptor {
                id: n.to_string(),
                version: 1,
                unit: Unit::KcalPerMol,
                pattern: match i {
                    0..=3 => Pattern::Bonded,
                    4 | 5 => Pattern::Pair,
                    6 | 7 => Pattern::MultiStage,
                    _ => Pattern::Restraint,
                },
                dependencies: if i == 6 || i == 7 {
                    vec!["born_radii".into()]
                } else {
                    Vec::new()
                },
                differentiable: true,
                cpu: true,
                webgpu: true,
            })
            .chain(self.extensions.iter().map(|t| t.descriptor()))
            .collect()
    }
}
#[derive(Debug, Clone, Default)]
pub struct EvaluationRequest {
    pub gradients: bool,
    pub forces: bool,
    pub pose_derivatives: bool,
    pub feature_distance: Option<f64>,
    pub feature_limit: usize,
    pub per_term_gradients: bool,
    pub components: Option<BTreeSet<String>>,
}
#[derive(Debug, Clone)]
pub struct PairFeature {
    pub atoms: [usize; 2],
    pub distance_angstrom: f64,
}
#[derive(Debug, Clone)]
pub struct PoseDerivatives {
    pub translation: Vec3,
    pub rotation: Vec3,
    pub torsions: Vec<f64>,
}
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub candidate_id: u64,
    pub total: f64,
    pub terms: BTreeMap<String, TermValue>,
    pub gradients: Option<Vec<Vec3>>,
    pub forces: Option<Vec<Vec3>>,
    pub pose_derivatives: Option<PoseDerivatives>,
    pub features: Vec<PairFeature>,
    pub term_gradients: BTreeMap<String, Vec<Vec3>>,
    pub backend: String,
    pub model_version: String,
}
/// A small compiled registry, not a general expression language. Dependencies
/// name resident intermediate stages; terms remain separate in returned values.
#[derive(Debug, Clone)]
pub struct EvaluationPlan {
    pub model_id: String,
    pub model_version: u32,
    pub terms: Vec<TermDescriptor>,
    pub intermediates: BTreeSet<String>,
    pub derivatives: bool,
    pub feature_only: bool,
}
impl EvaluationPlan {
    pub fn compile(model: &ScoreModel, request: &EvaluationRequest) -> Result<Self> {
        let descriptors = model.descriptors();
        let mut ids = BTreeSet::new();
        for d in &descriptors {
            if !ids.insert(d.id.clone()) || d.version == 0 {
                return Err(invalid("duplicate or unversioned scoring term"));
            }
        }
        if request
            .components
            .as_ref()
            .is_some_and(|s| s.iter().any(|id| !ids.contains(id)))
        {
            return Err(invalid("unknown requested component"));
        }
        let derivatives = request.gradients
            || request.forces
            || request.pose_derivatives
            || request.per_term_gradients;
        let terms = descriptors
            .into_iter()
            .filter(|d| {
                model.weights.get(&d.id).copied().unwrap_or(0.) != 0.
                    || request
                        .components
                        .as_ref()
                        .is_none_or(|s| s.contains(&d.id))
            })
            .collect::<Vec<_>>();
        for d in &terms {
            if derivatives
                && model.weights.get(&d.id).copied().unwrap_or(0.) != 0.
                && !d.differentiable
            {
                return Err(invalid("nondifferentiable objective term"));
            }
            if !d.cpu {
                return Err(invalid("a CPU reference is required for every term"));
            }
        }
        let intermediates = terms
            .iter()
            .flat_map(|d| d.dependencies.iter().cloned())
            .collect();
        let feature_only = terms.iter().all(|d| d.pattern == Pattern::Feature);
        Ok(Self {
            model_id: model.id.clone(),
            model_version: model.version,
            terms,
            intermediates,
            derivatives,
            feature_only,
        })
    }
}

pub struct PreparedEvaluator {
    pub scene: PreparedScene,
    pub model: ScoreModel,
}
impl PreparedEvaluator {
    pub fn new(mut scene: PreparedScene, model: ScoreModel) -> Result<Self> {
        scene.refresh_fingerprint()?;
        let descriptors = model.descriptors();
        let mut ids = BTreeSet::new();
        for d in &descriptors {
            if !ids.insert(d.id.clone()) {
                return Err(invalid("duplicate scoring term"));
            }
        }
        for (id, w) in &model.weights {
            if !w.is_finite() || !ids.contains(id) {
                return Err(invalid("unknown term or nonfinite weight"));
            }
        }
        if let Some((a, b)) = &model.interaction {
            let first = scene
                .groups
                .get(a)
                .ok_or_else(|| invalid("unknown interaction group"))?;
            let second = scene
                .groups
                .get(b)
                .ok_or_else(|| invalid("unknown interaction group"))?;
            if first.iter().any(|i| second.contains(i)) {
                return Err(invalid("overlapping interaction groups"));
            }
        }
        Ok(Self { scene, model })
    }
    pub fn evaluate(
        &self,
        batch: &PoseBatch,
        request: &EvaluationRequest,
    ) -> Result<Vec<EvaluationResult>> {
        EvaluationPlan::compile(&self.model, request)?;
        batch
            .poses
            .iter()
            .map(|p| self.evaluate_pose(p, request))
            .collect()
    }
    fn evaluate_pose(&self, pose: &Pose, request: &EvaluationRequest) -> Result<EvaluationResult> {
        let coordinates = pose.materialize()?;
        if coordinates.len() != self.scene.system.atom_count()
            || coordinates
                .iter()
                .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
        {
            return Err(invalid("invalid pose coordinates"));
        }
        let plan = EvaluationPlan::compile(&self.model, request)?;
        let derivative = request.gradients || request.forces || request.pose_derivatives;
        let evaluator = EnergyEvaluator::new(&self.scene.system, self.scene.options.clone())?;
        let e = if plan.feature_only {
            crate::EnergyResult {
                components: Default::default(),
                gradients: None,
            }
        } else if let Some((a, b)) = &self.model.interaction {
            let n = self.scene.system.atom_count();
            let a = AtomGroupMask::from_indices(n, self.scene.groups[a].iter().copied());
            let b = AtomGroupMask::from_indices(n, self.scene.groups[b].iter().copied());
            let c = evaluator.interaction_energy(&coordinates, &a, &b)?;
            crate::EnergyResult {
                components: EnergyComponents {
                    van_der_waals: c.van_der_waals,
                    electrostatics: c.electrostatics,
                    ..Default::default()
                },
                gradients: None,
            }
        } else {
            evaluator.energy(&coordinates)?
        };
        let mut gradient = e.gradients;
        let weights = NAMES.map(|n| self.model.weights.get(n).copied().unwrap_or(0.));
        if derivative {
            gradient = Some(if let Some((a, b)) = &self.model.interaction {
                let n = coordinates.len();
                evaluator.interaction_gradient(
                    &coordinates,
                    &AtomGroupMask::from_indices(n, self.scene.groups[a].iter().copied()),
                    &AtomGroupMask::from_indices(n, self.scene.groups[b].iter().copied()),
                    [weights[4], weights[5]],
                )?
            } else {
                evaluator.weighted_gradient(&coordinates, weights)?
            });
        }
        let mut term_gradients = BTreeMap::new();
        if request.per_term_gradients {
            for (i, n) in NAMES.iter().enumerate() {
                let mut w = [0.; 9];
                w[i] = 1.;
                let g = if let Some((a, b)) = &self.model.interaction {
                    evaluator.interaction_gradient(
                        &coordinates,
                        &AtomGroupMask::from_indices(
                            coordinates.len(),
                            self.scene.groups[a].iter().copied(),
                        ),
                        &AtomGroupMask::from_indices(
                            coordinates.len(),
                            self.scene.groups[b].iter().copied(),
                        ),
                        [w[4], w[5]],
                    )?
                } else {
                    evaluator.weighted_gradient(&coordinates, w)?
                };
                term_gradients.insert(n.to_string(), g);
            }
        }
        let mut terms = BTreeMap::new();
        let mut total = 0.;
        for (name, raw) in NAMES.iter().zip(values(e.components)) {
            let weighted = raw * self.model.weights.get(*name).copied().unwrap_or(0.);
            total += weighted;
            terms.insert(
                name.to_string(),
                TermValue {
                    raw,
                    weighted,
                    unit: Unit::KcalPerMol,
                },
            );
        }
        for term in &self.model.extensions {
            let d = term.descriptor();
            let weight = self.model.weights.get(&d.id).copied().unwrap_or(0.);
            if derivative && weight != 0. && !d.differentiable {
                return Err(invalid("requested derivative of nondifferentiable term"));
            }
            let out = term.evaluate(
                &self.scene,
                &coordinates,
                (derivative && weight != 0.) || (request.per_term_gradients && d.differentiable),
            )?;
            if let Some(g) = &out.gradient {
                if g.len() != coordinates.len()
                    || g.iter()
                        .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
                {
                    return Err(invalid("invalid term gradient"));
                }
                if request.per_term_gradients {
                    term_gradients.insert(d.id.clone(), g.clone());
                }
            } else if request.per_term_gradients && d.differentiable {
                return Err(invalid("term omitted requested component gradient"));
            }
            if !out.value.is_finite() {
                return Err(invalid("nonfinite extension value"));
            }
            if d.unit != Unit::KcalPerMol
                && weight != 0.
                && self.model.purpose == "potential_energy"
            {
                return Err(invalid(
                    "dimensionless feature cannot be added to physical energy",
                ));
            }
            total += weight * out.value;
            if derivative && weight != 0. {
                let g = out
                    .gradient
                    .ok_or_else(|| invalid("term omitted requested gradient"))?;
                if g.len() != coordinates.len() {
                    return Err(invalid("term gradient dimension"));
                }
                for (a, b) in gradient.as_mut().unwrap().iter_mut().zip(g) {
                    *a = crate::geometry::add(*a, crate::geometry::scale(b, weight));
                }
            }
            terms.insert(
                d.id,
                TermValue {
                    raw: out.value,
                    weighted: weight * out.value,
                    unit: d.unit,
                },
            );
        }
        let mut features = Vec::new();
        if let Some(cutoff) = request.feature_distance {
            if !cutoff.is_finite() || cutoff <= 0. || request.feature_limit == 0 {
                return Err(invalid(
                    "feature request needs positive radius and explicit capacity",
                ));
            }
            for a in 0..coordinates.len() {
                for b in a + 1..coordinates.len() {
                    let v = sub(coordinates[a], coordinates[b]);
                    let d = dot(v, v).sqrt();
                    if d <= cutoff {
                        if features.len() == request.feature_limit {
                            return Err(invalid("feature capacity exceeded"));
                        }
                        features.push(PairFeature {
                            atoms: [a, b],
                            distance_angstrom: d,
                        });
                    }
                }
            }
        }
        let pd = if request.pose_derivatives {
            let g = gradient.as_ref().unwrap();
            let mut t = Vec3 {
                x: 0.,
                y: 0.,
                z: 0.,
            };
            let mut r = t;
            for &i in &pose.transformed_atoms {
                t = crate::geometry::add(t, g[i]);
                r = crate::geometry::add(
                    r,
                    cross(sub(coordinates[i], pose.transform.translation), g[i]),
                );
            }
            let mut torsions = Vec::new();
            for torsion in &pose.torsions {
                let origin = coordinates[torsion.axis[0]];
                let axis = crate::geometry::unit(sub(coordinates[torsion.axis[1]], origin))?;
                torsions.push(
                    torsion
                        .moving
                        .iter()
                        .map(|&i| dot(g[i], cross(axis, sub(coordinates[i], origin))))
                        .sum(),
                );
            }
            Some(PoseDerivatives {
                translation: t,
                rotation: r,
                torsions,
            })
        } else {
            None
        };
        if let Some(selected) = &request.components {
            if selected.iter().any(|id| !terms.contains_key(id)) {
                return Err(invalid("unknown requested component"));
            }
            terms.retain(|id, _| selected.contains(id));
        }
        let forces = request.forces.then(|| {
            gradient
                .as_ref()
                .unwrap()
                .iter()
                .map(|g| crate::geometry::scale(*g, -1.))
                .collect()
        });
        Ok(EvaluationResult {
            candidate_id: pose.id,
            total,
            terms,
            gradients: request.gradients.then(|| gradient.unwrap()),
            forces,
            pose_derivatives: pd,
            features,
            term_gradients,
            backend: "cpu".into(),
            model_version: self.model.id.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scene() -> PreparedScene {
        let system = glysys::SystemBuilder::new(glysys::BuildOptions {
            add_water: false,
            add_ions: false,
            ..Default::default()
        })
        .unwrap()
        .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))
        .unwrap();
        PreparedScene::new(
            Arc::new(system),
            EnergyOptions::default(),
            Boundary::NonPeriodic,
        )
        .unwrap()
    }
    #[test]
    fn weighted_terms_and_forces_match_finite_difference() {
        let scene = scene();
        let mut model = ScoreModel::amber();
        for (i, n) in NAMES.iter().enumerate() {
            model.weights.insert(n.to_string(), (i + 1) as f64 * 0.13);
        }
        let evaluator = PreparedEvaluator::new(scene, model).unwrap();
        let coordinates = evaluator.scene.system.coordinates();
        let request = EvaluationRequest {
            gradients: true,
            forces: true,
            per_term_gradients: true,
            ..Default::default()
        };
        let out = evaluator
            .evaluate(
                &PoseBatch {
                    poses: vec![Pose::cartesian(7, coordinates.clone())],
                },
                &request,
            )
            .unwrap()
            .remove(0);
        assert_eq!(out.candidate_id, 7);
        assert_eq!(out.term_gradients.len(), 9);
        for atom in [0, 3, coordinates.len() - 1] {
            let mut a = coordinates.clone();
            let mut b = coordinates.clone();
            a[atom].x += 1e-5;
            b[atom].x -= 1e-5;
            let vals = evaluator
                .evaluate(
                    &PoseBatch {
                        poses: vec![Pose::cartesian(0, a), Pose::cartesian(1, b)],
                    },
                    &Default::default(),
                )
                .unwrap();
            let fd = (vals[0].total - vals[1].total) / 2e-5;
            let g = out.gradients.as_ref().unwrap()[atom].x;
            assert!((fd - g).abs() < 1e-3, "{fd} {g}");
            assert_eq!(out.forces.as_ref().unwrap()[atom].x, -g);
        }
    }
    struct CountFeature;
    impl ScoreTerm for CountFeature {
        fn descriptor(&self) -> TermDescriptor {
            TermDescriptor {
                id: "test.atom_count".into(),
                version: 1,
                unit: Unit::Dimensionless,
                pattern: Pattern::Feature,
                dependencies: Vec::new(),
                differentiable: false,
                cpu: true,
                webgpu: false,
            }
        }
        fn evaluate(&self, _: &PreparedScene, p: &[Vec3], _: bool) -> Result<TermOutput> {
            Ok(TermOutput {
                value: p.len() as f64,
                gradient: None,
            })
        }
    }
    #[test]
    fn feature_extension_does_not_change_physical_total() {
        let scene = scene();
        let pose = Pose::cartesian(1, scene.system.coordinates());
        let mut model = ScoreModel::amber();
        model.extensions.push(Arc::new(CountFeature));
        let evaluator = PreparedEvaluator::new(scene, model).unwrap();
        let out = evaluator
            .evaluate(&PoseBatch { poses: vec![pose] }, &Default::default())
            .unwrap()
            .remove(0);
        assert_eq!(out.terms["test.atom_count"].weighted, 0.);
        assert!(out.terms["test.atom_count"].raw > 0.);
    }
    #[test]
    fn configuration_changes_invalidate_fingerprint() {
        let mut a = scene();
        let first = a.fingerprint.clone();
        a.options.cutoff = Some(4.);
        a.refresh_fingerprint().unwrap();
        assert_ne!(a.fingerprint, first);
        let second = a.fingerprint.clone();
        a.set_group("focus".into(), vec![0, 1]).unwrap();
        assert_ne!(a.fingerprint, second);
        assert!(
            PreparedScene::new(
                a.system.clone(),
                a.options.clone(),
                Boundary::Periodic {
                    box_angstrom: [10.; 3]
                }
            )
            .is_err()
        );
    }
    #[test]
    fn cross_interaction_gradient_matches_difference() {
        let mut scene = scene();
        scene.set_group("a".into(), vec![0, 1]).unwrap();
        scene.set_group("b".into(), vec![4, 5, 6]).unwrap();
        let eval = PreparedEvaluator::new(scene, ScoreModel::interaction("a", "b")).unwrap();
        let mut a = eval.scene.system.coordinates();
        let mut b = a.clone();
        let out = eval
            .evaluate(
                &PoseBatch {
                    poses: vec![Pose::cartesian(0, a.clone())],
                },
                &EvaluationRequest {
                    gradients: true,
                    ..Default::default()
                },
            )
            .unwrap();
        a[0].x += 1e-5;
        b[0].x -= 1e-5;
        let v = eval
            .evaluate(
                &PoseBatch {
                    poses: vec![Pose::cartesian(0, a), Pose::cartesian(1, b)],
                },
                &Default::default(),
            )
            .unwrap();
        assert!(
            ((v[0].total - v[1].total) / 2e-5 - out[0].gradients.as_ref().unwrap()[0].x).abs()
                < 1e-3
        );
    }
    #[test]
    fn feature_only_plan_has_no_physical_stages() {
        let scene = scene();
        let pose = Pose::cartesian(9, scene.system.coordinates());
        let mut model = ScoreModel::amber();
        model.weights.clear();
        model.purpose = "geometric_features".into();
        model.extensions.push(Arc::new(CountFeature));
        let request = EvaluationRequest {
            components: Some(["test.atom_count".to_string()].into_iter().collect()),
            ..Default::default()
        };
        let plan = EvaluationPlan::compile(&model, &request).unwrap();
        assert!(plan.feature_only);
        assert!(plan.intermediates.is_empty());
        let result = PreparedEvaluator::new(scene, model)
            .unwrap()
            .evaluate(&PoseBatch { poses: vec![pose] }, &request)
            .unwrap();
        assert_eq!(result[0].total, 0.);
        assert_eq!(result[0].terms.len(), 1);
    }
    #[test]
    fn sparse_features_require_bounded_capacity() {
        let scene = scene();
        let batch = PoseBatch {
            poses: vec![Pose::cartesian(1, scene.system.coordinates())],
        };
        let eval = PreparedEvaluator::new(scene, ScoreModel::amber()).unwrap();
        assert!(
            eval.evaluate(
                &batch,
                &EvaluationRequest {
                    feature_distance: Some(100.),
                    feature_limit: 1,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}
