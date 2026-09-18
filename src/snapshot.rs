//! Lossless preparation handoff. PDB is a visualization format, not a topology.
use crate::{BuildError, BuildReport, ParameterizedSystem, Result, SystemMetadata, model::System};

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    version: u32,
    system: System,
    report: BuildReport,
    metadata: SystemMetadata,
}

impl ParameterizedSystem {
    /// Preserve exact parameters, atom order, solvent and box across workers.
    pub fn snapshot_json(&self) -> Result<String> {
        serde_json::to_string(&Snapshot {
            version: 1,
            system: self.system.clone(),
            report: self.report.clone(),
            metadata: self.metadata.clone(),
        })
        .map_err(|e| BuildError::Serialization(e.to_string()))
    }

    /// Import only a structurally valid parameterized snapshot. This does not
    /// perceive chemistry or supply missing parameters from coordinates.
    pub fn from_snapshot_json(json: &str) -> Result<Self> {
        let invalid = || BuildError::InvalidOption("invalid prepared-system snapshot".into());
        if json.len() > 256 * 1024 * 1024 {
            return Err(invalid());
        }
        let snapshot: Snapshot =
            serde_json::from_str(json).map_err(|e| BuildError::Serialization(e.to_string()))?;
        let s = &snapshot.system;
        let n = s.atoms.len();
        if snapshot.version != 1
            || n == 0
            || s.exclusions.len() != n
            || s.solute_atom_count > n
            || s.component_count == 0
            || s.box_angstrom.iter().any(|x| !x.is_finite() || *x < 0.)
            || snapshot.report.total_atoms != n
            || snapshot.report.residues != s.residues.len()
            || snapshot.report.solute_atoms != s.solute_atom_count
            || snapshot.report.box_angstrom != s.box_angstrom
            || snapshot.report.waters != s.water_residue_count
            || snapshot.report.sodium_ions != s.sodium_count
            || snapshot.report.chloride_ions != s.chloride_count
        {
            return Err(invalid());
        }
        let mut end = 0usize;
        let mut components = std::collections::BTreeSet::new();
        for (index, r) in s.residues.iter().enumerate() {
            if r.first_atom != end || r.atom_count == 0 || r.component >= s.component_count {
                return Err(invalid());
            }
            end = end.checked_add(r.atom_count).ok_or_else(invalid)?;
            if end > n
                || s.atoms[r.first_atom..end]
                    .iter()
                    .any(|a| a.residue != index)
            {
                return Err(invalid());
            }
            components.insert(r.component);
        }
        if end != n || components.len() != s.component_count {
            return Err(invalid());
        }
        for (i, a) in s.atoms.iter().enumerate() {
            if a.name.is_empty()
                || a.atom_type.is_empty()
                || a.element == 0
                || ![
                    a.charge,
                    a.mass,
                    a.radius,
                    a.epsilon,
                    a.position.x,
                    a.position.y,
                    a.position.z,
                ]
                .iter()
                .all(|x| x.is_finite())
                || a.mass <= 0.
                || a.radius < 0.
                || a.epsilon < 0.
                || s.exclusions[i]
                    .iter()
                    .any(|&j| j >= n || j == i || !s.exclusions[j].contains(&i))
            {
                return Err(invalid());
            }
        }
        let indices = |atoms: &[usize]| {
            atoms.iter().all(|&i| i < n)
                && atoms
                    .iter()
                    .enumerate()
                    .all(|(i, a)| !atoms[..i].contains(a))
        };
        if s.bonds.iter().any(|b| {
            !indices(&b.atoms)
                || !b.force.is_finite()
                || b.force < 0.
                || !b.length.is_finite()
                || b.length <= 0.
        }) || s.angles.iter().any(|a| {
            !indices(&a.atoms)
                || !a.force.is_finite()
                || a.force < 0.
                || !a.radians.is_finite()
                || a.radians <= 0.
                || a.radians > std::f64::consts::PI
        }) || s.dihedrals.iter().any(|d| {
            !indices(&d.atoms)
                || ![d.force, d.phase, d.scee, d.scnb]
                    .iter()
                    .all(|x| x.is_finite())
                || d.scee <= 0.
                || d.scnb <= 0.
                || d.periodicity == 0
        }) {
            return Err(invalid());
        }
        Ok(Self {
            system: snapshot.system,
            report: snapshot.report,
            metadata: snapshot.metadata,
        })
    }
}
