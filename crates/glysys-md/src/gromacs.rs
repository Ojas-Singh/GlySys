//! Strict, dependency-free parsing of the supported GOTW `.mdp` recipe.
//!
//! This is deliberately a settings resolver, not a second topology builder.
//! A GlySys run still consumes a lossless `system.snapshot.json`, because a
//! GROMACS topology cannot safely be reconstructed from text without the
//! original parameter provenance.  The resolver makes the physical choices in
//! an `.mdp` explicit and refuses settings that would silently change the
//! Hamiltonian or integrator.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use glysys_dynamics::{
    ConstraintModel, ElectrostaticsModel, Ensemble, PressureCoupling, SimulationProtocol,
    SimulationStage, SolventModel, Thermostat, ThermostatGroup,
};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedMdp {
    pub source: String,
    pub integrator: String,
    pub timestep_ps: f64,
    pub steps: usize,
    pub energy_output_interval_steps: Option<usize>,
    pub compressed_coordinate_interval_steps: Option<usize>,
    pub constraints: String,
    pub cutoff_scheme: String,
    pub cutoff_angstrom: f64,
    pub coulomb_type: String,
    pub vdw_type: String,
    pub vdw_modifier: String,
    pub dispersion_correction: String,
    pub thermostat: String,
    pub temperature_groups: Vec<String>,
    pub temperature_tau_ps: Vec<f64>,
    pub reference_temperature_k: Vec<f64>,
    pub pressure_coupling: String,
    pub pressure_coupling_type: String,
    pub pressure_tau_ps: Option<f64>,
    pub reference_pressure_bar: Option<f64>,
    pub compressibility_bar_inverse: Vec<f64>,
    pub pbc: String,
    pub com_mode: String,
    pub com_groups: Vec<String>,
    /// Every parsed key/value is retained for audit output.  This makes a
    /// resolved run reproducible without depending on the source file later.
    pub raw: BTreeMap<String, String>,
}

fn value<'a>(raw: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str> {
    raw.get(key)
        .map(String::as_str)
        .ok_or_else(|| anyhow::anyhow!("GROMACS .mdp is missing required key '{key}'"))
}

fn parse_f64(raw: &BTreeMap<String, String>, key: &str) -> Result<f64> {
    value(raw, key)?
        .parse::<f64>()
        .with_context(|| format!("GROMACS .mdp key '{key}' must be a finite number"))
}

fn parse_usize(raw: &BTreeMap<String, String>, key: &str) -> Result<usize> {
    value(raw, key)?
        .parse::<usize>()
        .with_context(|| format!("GROMACS .mdp key '{key}' must be a non-negative integer"))
}

fn optional_positive_usize(raw: &BTreeMap<String, String>, key: &str) -> Result<Option<usize>> {
    let Some(value) = raw.get(key) else {
        return Ok(None);
    };
    let parsed = value
        .parse::<usize>()
        .with_context(|| format!("GROMACS .mdp key '{key}' must be a non-negative integer"))?;
    Ok((parsed > 0).then_some(parsed))
}

fn parse_list<T: std::str::FromStr>(raw: &BTreeMap<String, String>, key: &str) -> Result<Vec<T>>
where
    T::Err: std::fmt::Display,
{
    value(raw, key)?
        .split_whitespace()
        .map(|item| item.parse::<T>().map_err(|e| anyhow::anyhow!("{key}: {e}")))
        .collect()
}

fn parse_words(raw: &BTreeMap<String, String>, key: &str) -> Result<Vec<String>> {
    Ok(value(raw, key)?
        .split_whitespace()
        .map(str::to_owned)
        .collect())
}

fn equal_f64(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1.0e-9 * a.abs().max(b.abs()).max(1.0)
}

/// Parse a GROMACS parameter file and enforce the subset whose semantics are
/// needed by the GOTW compatibility profile.
pub fn parse(path: &Path) -> Result<ResolvedMdp> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading GROMACS mdp {}", path.display()))?;
    let mut raw = BTreeMap::new();
    for (line_number, line) in text.lines().enumerate() {
        let line = line.split(';').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            bail!("GROMACS .mdp line {} has no '='", line_number + 1);
        };
        // GROMACS accepts both hyphenated and underscore spellings for
        // several keys (the GOTW files use `tau_t`, `ref_t`, and
        // `comm_mode`). Normalize those aliases once so the resolver cannot
        // accidentally parse one spelling while ignoring the other.
        let key = key.trim().to_ascii_lowercase().replace('_', "-");
        let value = value.trim().to_owned();
        if key.is_empty() || value.is_empty() {
            bail!(
                "GROMACS .mdp line {} has an empty key or value",
                line_number + 1
            );
        }
        if raw.insert(key.clone(), value).is_some() {
            bail!("GROMACS .mdp contains duplicate key '{key}'");
        }
    }

    let integrator = value(&raw, "integrator")?.to_ascii_lowercase();
    if integrator != "md" {
        bail!(
            "unsupported GOTW integrator '{integrator}'; GlySys compatibility requires leap-frog md"
        );
    }
    let timestep_ps = parse_f64(&raw, "dt")?;
    let steps = parse_usize(&raw, "nsteps")?;
    if !timestep_ps.is_finite() || timestep_ps <= 0. || steps == 0 {
        bail!("GROMACS dt and nsteps must be positive and finite");
    }
    let energy_output_interval_steps = optional_positive_usize(&raw, "nstenergy")?;
    let compressed_coordinate_interval_steps = optional_positive_usize(&raw, "nstxout-compressed")?;

    let constraints = value(&raw, "constraints")?.to_ascii_lowercase();
    if constraints != "h-bonds" {
        bail!(
            "unsupported constraints='{constraints}'; the compatibility profile requires h-bonds with SETTLE/LINCS validation"
        );
    }
    let cutoff_scheme = value(&raw, "cutoff-scheme")?.to_ascii_lowercase();
    if cutoff_scheme != "verlet" {
        bail!(
            "unsupported cutoff-scheme='{cutoff_scheme}'; the compatibility profile requires Verlet lists"
        );
    }
    let pbc = value(&raw, "pbc")?.to_ascii_lowercase();
    if pbc != "xyz" {
        bail!("unsupported pbc='{pbc}'; only orthorhombic xyz is currently supported");
    }
    let cutoff_nm = parse_f64(&raw, "rcoulomb")?;
    let rvdw_nm = parse_f64(&raw, "rvdw").unwrap_or(cutoff_nm);
    let rlist_nm = parse_f64(&raw, "rlist").unwrap_or(cutoff_nm);
    if !equal_f64(cutoff_nm, rvdw_nm) || !equal_f64(cutoff_nm, rlist_nm) {
        bail!("rcoulomb, rvdw, and rlist must match for the current shared cutoff implementation");
    }
    let cutoff_angstrom = cutoff_nm * 10.0;
    if !cutoff_angstrom.is_finite() || cutoff_angstrom <= 0. {
        bail!("GROMACS cutoff must be positive and finite");
    }

    let coulomb_type = value(&raw, "coulombtype")?.to_ascii_lowercase();
    if coulomb_type != "pme" {
        bail!("unsupported coulombtype='{coulomb_type}'; use the explicit PME path for GOTW");
    }
    let vdw_type = value(&raw, "vdwtype")?.to_ascii_lowercase();
    if vdw_type != "cut-off" && vdw_type != "cutoff" {
        bail!("unsupported vdwtype='{vdw_type}' for the GOTW compatibility profile");
    }
    let vdw_modifier = value(&raw, "vdw-modifier")?.to_ascii_lowercase();
    if vdw_modifier != "none" {
        bail!(
            "unsupported vdw-modifier='{vdw_modifier}'; switching is intentionally disabled in the first GOTW profile"
        );
    }
    let dispersion_correction = value(&raw, "dispcorr")?.to_ascii_lowercase();
    if dispersion_correction != "enerpres" {
        bail!("unsupported DispCorr='{dispersion_correction}'; the GOTW profile requires EnerPres");
    }

    let thermostat = value(&raw, "tcoupl")?.to_ascii_lowercase();
    if thermostat != "nose-hoover" {
        bail!("unsupported tcoupl='{thermostat}'; the GOTW profile requires Nose-Hoover");
    }
    let temperature_groups = parse_words(&raw, "tc-grps")?;
    let temperature_tau_ps = parse_list(&raw, "tau-t")?;
    let reference_temperature_k = parse_list(&raw, "ref-t")?;
    if temperature_groups.is_empty()
        || temperature_groups.len() != temperature_tau_ps.len()
        || temperature_groups.len() != reference_temperature_k.len()
    {
        bail!("tc-grps, tau-t, and ref-t must contain the same nonzero number of groups");
    }

    let pressure_coupling = value(&raw, "pcoupl")?.to_ascii_lowercase();
    if pressure_coupling != "parrinello-rahman" {
        bail!(
            "unsupported pcoupl='{pressure_coupling}'; the GOTW profile requires Parrinello-Rahman"
        );
    }
    let pressure_coupling_type = value(&raw, "pcoupltype")?.to_ascii_lowercase();
    if pressure_coupling_type != "isotropic" {
        bail!(
            "unsupported pcoupltype='{pressure_coupling_type}'; only isotropic coupling is in the first GOTW profile"
        );
    }
    let pressure_tau_ps = Some(parse_f64(&raw, "tau-p")?);
    let reference_pressure_bar = Some(parse_f64(&raw, "ref-p")?);
    let compressibility_bar_inverse: Vec<f64> = parse_list(&raw, "compressibility")?;
    if compressibility_bar_inverse.is_empty()
        || compressibility_bar_inverse
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.)
    {
        bail!("compressibility must contain positive finite values");
    }
    let com_mode = value(&raw, "comm-mode")?.to_ascii_lowercase();
    let com_groups = parse_words(&raw, "comm-grps")?;
    if com_mode != "linear" {
        bail!("unsupported comm-mode='{com_mode}'; the GOTW profile requires linear COM removal");
    }

    Ok(ResolvedMdp {
        source: path.display().to_string(),
        integrator,
        timestep_ps,
        steps,
        energy_output_interval_steps,
        compressed_coordinate_interval_steps,
        constraints,
        cutoff_scheme,
        cutoff_angstrom,
        coulomb_type,
        vdw_type,
        vdw_modifier,
        dispersion_correction,
        thermostat,
        temperature_groups,
        temperature_tau_ps,
        reference_temperature_k,
        pressure_coupling,
        pressure_coupling_type,
        pressure_tau_ps,
        reference_pressure_bar,
        compressibility_bar_inverse,
        pbc,
        com_mode,
        com_groups,
        raw,
    })
}

impl ResolvedMdp {
    /// Convert the resolved recipe to the versioned engine contract.  The
    /// resulting protocol intentionally retains PME/Nose–Hoover/Parrinello–
    /// Rahman as explicit choices; current drivers report a capability error
    /// until those kernels pass their independent validation gates.
    pub fn to_protocol(&self) -> Result<SimulationProtocol> {
        let timestep_fs = self.timestep_ps * 1000.0;
        if !timestep_fs.is_finite() || timestep_fs > 2.0 {
            bail!("GOTW timestep {timestep_fs:.6} fs exceeds GlySys's validated 2 fs limit");
        }
        let protocol = SimulationProtocol {
            temperature_k: self.reference_temperature_k[0],
            timestep_fs,
            equilibration_steps: 0,
            production_steps: self.steps,
            save_every: self
                .compressed_coordinate_interval_steps
                .or(self.energy_output_interval_steps)
                .unwrap_or(1),
            minimization_iterations: 0,
            solvent: SolventModel::Explicit,
            equilibration_ensemble: Ensemble::Nvt,
            production_ensemble: Ensemble::Npt,
            pressure_bar: self.reference_pressure_bar.unwrap_or(1.0),
            constraints: ConstraintModel::Settle,
            thermostat: Thermostat::NoseHoover,
            electrostatics: ElectrostaticsModel::Pme,
            pressure_coupling: PressureCoupling::ParrinelloRahman,
            cutoff_angstrom: Some(self.cutoff_angstrom),
            dispersion_correction: true,
            thermostat_groups: self
                .temperature_groups
                .iter()
                .zip(self.temperature_tau_ps.iter())
                .zip(self.reference_temperature_k.iter())
                .map(
                    |((name, tau_ps), reference_temperature_k)| ThermostatGroup {
                        name: name.clone(),
                        tau_ps: *tau_ps,
                        reference_temperature_k: *reference_temperature_k,
                    },
                )
                .collect(),
            pressure_tau_ps: self.pressure_tau_ps,
            pressure_compressibility_bar_inverse: self.compressibility_bar_inverse.clone(),
            com_mode: Some(self.com_mode.clone()),
            com_groups: self.com_groups.clone(),
            stages: Some(vec![SimulationStage {
                id: "production".into(),
                ensemble: Ensemble::Npt,
                steps: self.steps,
                barostat_adaptation: false,
            }]),
            ..SimulationProtocol::default()
        };
        protocol
            .validate()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(protocol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn resolves_gotw_recipe_without_reinterpreting_units() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "integrator = md\ndt = 0.002\nnsteps = 250\nconstraints = h-bonds\ncutoff-scheme = Verlet\npbc = xyz\nrcoulomb = 0.9\nrvdw = 0.9\nrlist = 0.9\ncoulombtype = PME\nvdwtype = Cut-off\nvdw-modifier = None\nDispCorr = EnerPres\ntcoupl = Nose-Hoover\ntc-grps = WAT System_&_!WAT\ntau-t = 1 1\nref-t = 300 300\npcoupl = Parrinello-Rahman\npcoupltype = isotropic\ntau-p = 5\nref-p = 1\ncompressibility = 4.5e-5\ncomm-mode = linear\ncomm-grps = WAT System_&_!WAT").unwrap();
        let resolved = parse(file.path()).unwrap();
        assert!((resolved.cutoff_angstrom - 9.0).abs() < 1e-12);
        assert_eq!(resolved.steps, 250);
        let protocol = resolved.to_protocol().unwrap();
        assert_eq!(protocol.electrostatics, ElectrostaticsModel::Pme);
        assert_eq!(
            protocol.pressure_coupling,
            PressureCoupling::ParrinelloRahman
        );
        assert_eq!(protocol.thermostat, Thermostat::NoseHoover);
        assert_eq!(protocol.save_every, 1);
    }

    #[test]
    fn rejects_recipe_that_would_change_the_hamiltonian() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "integrator = md\ndt = 0.002\nnsteps = 2\nconstraints = h-bonds\ncutoff-scheme = Verlet\npbc = xyz\nrcoulomb = 0.9\nrvdw = 0.9\nrlist = 0.9\ncoulombtype = Cut-off\nvdwtype = Cut-off\nvdw-modifier = None\nDispCorr = EnerPres\ntcoupl = Nose-Hoover\ntc-grps = System\ntau-t = 1\nref-t = 300\npcoupl = Parrinello-Rahman\npcoupltype = isotropic\ntau-p = 5\nref-p = 1\ncompressibility = 4.5e-5\ncomm-mode = linear\ncomm-grps = System").unwrap();
        let error = parse(file.path()).unwrap_err().to_string();
        assert!(error.contains("coulombtype"), "{error}");
    }

    #[test]
    fn accepts_gromacs_underscore_aliases_used_by_gotw() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "integrator = md\ndt = 0.002\nnsteps = 2\nconstraints = h-bonds\ncutoff_scheme = Verlet\npbc = xyz\nrcoulomb = 0.9\nrvdw = 0.9\nrlist = 0.9\ncoulombtype = PME\nvdwtype = Cut-off\nvdw_modifier = None\nDispCorr = EnerPres\ntcoupl = Nose-Hoover\ntc-grps = WAT System_&_!WAT\ntau_t = 1 1\nref_t = 300 300\npcoupl = Parrinello-Rahman\npcoupltype = isotropic\ntau_p = 5\nref_p = 1\ncompressibility = 4.5e-5\ncomm_mode = linear\ncomm_grps = WAT System_&_!WAT").unwrap();
        let resolved = parse(file.path()).unwrap();
        assert_eq!(resolved.temperature_groups.len(), 2);
        assert_eq!(resolved.com_mode, "linear");
        assert_eq!(resolved.raw["tau-t"], "1 1");
    }

    #[test]
    fn carries_gotw_output_cadence_into_protocol() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "integrator = md\ndt = 0.002\nnsteps = 5000\nnstenergy = 5000\nnstxout-compressed = 5000\nconstraints = h-bonds\ncutoff-scheme = Verlet\npbc = xyz\nrcoulomb = 0.9\nrvdw = 0.9\nrlist = 0.9\ncoulombtype = PME\nvdwtype = Cut-off\nvdw-modifier = None\nDispCorr = EnerPres\ntcoupl = Nose-Hoover\ntc-grps = System\ntau-t = 1\nref-t = 300\npcoupl = Parrinello-Rahman\npcoupltype = isotropic\ntau-p = 5\nref-p = 1\ncompressibility = 4.5e-5\ncomm-mode = linear\ncomm-grps = System").unwrap();
        let resolved = parse(file.path()).unwrap();
        assert_eq!(resolved.energy_output_interval_steps, Some(5000));
        assert_eq!(resolved.compressed_coordinate_interval_steps, Some(5000));
        assert_eq!(resolved.to_protocol().unwrap().save_every, 5000);
    }
}
