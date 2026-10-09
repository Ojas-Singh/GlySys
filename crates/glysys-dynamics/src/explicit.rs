//! Explicit-water PBC dynamics: minimization, NVE, NVT, and NPT.
//!
//! Coordinates live in *unwrapped* space for the whole run; wrapping happens
//! only for pair-list builds and exported frames. Thermostat noise and the
//! barostat use independent deterministic streams so barostat schedule
//! changes never perturb thermostat reproducibility.
use super::accumulators::{Accumulator, WaterOccupancy};
use super::coupling::{CouplingPlan, CouplingState};
use super::settle::SettleWaters;
use super::{
    ACCEL, BAR_PER_KCAL_MOL_A3, ConstraintModel, ElectrostaticsModel, Ensemble, KB,
    LangevinDiscretization, PressureCoupling, Thermostat, VelocityConvention, normal,
};
use super::{Error, Result, SimulationProtocol, SimulationState, SolventModel, TrajectoryChunk};
use super::{frame_from_state_wrapped, invalid};
use glysys::{ParameterizedSystem, Vec3};
use glysys_energy::HarmonicRestraint;
use glysys_energy::pbc::{
    BoxVectors, NonbondedElectrostatics, PbcForceField, PbcNeighborList, ReactionField,
    classify_waters, molecules,
};
use glysys_energy::pbc_cluster::ClusterPairEngine;
use glysys_energy::pme::{PmeEngine, PmeParameters};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::sync::Arc;

pub const SKIN_ANGSTROM: f64 = 1.5;
/// Fewest atoms a parallel task of a per-atom loop takes; such loops cost a
/// few nanoseconds per atom, less than handing out a task does.
const PER_ATOM_TASK: usize = 512;
/// Below this many atoms the per-atom loops of a step run on the calling
/// thread: waking the pool for one of them takes longer than the loop.
const PARALLEL_ATOMS: usize = 32_768;
/// 1 fs for flexible waters, 2 fs with SETTLE. Larger requests are rejected
/// rather than silently integrated.
pub fn max_timestep_fs(constraints: ConstraintModel) -> f64 {
    match constraints {
        ConstraintModel::None => 1.0,
        ConstraintModel::Settle | ConstraintModel::HBonds => 2.0,
    }
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
    }
}

fn scale(a: Vec3, s: f64) -> Vec3 {
    Vec3 {
        x: a.x * s,
        y: a.y * s,
        z: a.z * s,
    }
}

fn norm2(a: Vec3) -> f64 {
    a.x * a.x + a.y * a.y + a.z * a.z
}

fn finite(a: &Vec3) -> bool {
    a.x.is_finite() && a.y.is_finite() && a.z.is_finite()
}

fn random_velocity(rng: &mut u64, s: f64) -> Vec3 {
    Vec3 {
        x: normal(rng) * s,
        y: normal(rng) * s,
        z: normal(rng) * s,
    }
}

pub fn cutoff_angstrom(protocol: &SimulationProtocol) -> f64 {
    protocol.cutoff_angstrom.unwrap_or(9.0)
}

/// GROMACS defaults of `ewald-rtol`, `fourierspacing` and `pme-order`.
pub const DEFAULT_EWALD_TOLERANCE: f64 = 1e-5;
pub const DEFAULT_FOURIER_SPACING_ANGSTROM: f64 = 1.2;
pub const DEFAULT_PME_ORDER: usize = 4;

/// Mesh Ewald settings of a protocol for a prepared system. The grid comes
/// from the box the system was prepared with, so a run and its restarts use
/// the same one whatever a barostat has done to the box since.
pub fn pme_parameters(
    system: &ParameterizedSystem,
    protocol: &SimulationProtocol,
) -> Result<PmeParameters> {
    let prepared_box = BoxVectors::from_system(system).map_err(Error::Energy)?;
    PmeParameters::for_box(
        &prepared_box,
        cutoff_angstrom(protocol),
        protocol.ewald_tolerance.unwrap_or(DEFAULT_EWALD_TOLERANCE),
        protocol
            .fourier_spacing_angstrom
            .unwrap_or(DEFAULT_FOURIER_SPACING_ANGSTROM),
        protocol.pme_order.unwrap_or(DEFAULT_PME_ORDER),
    )
    .map_err(Error::Energy)
}

/// Electrostatics of one run: the reaction-field pair function, or the mesh
/// Ewald engine with its grids and work buffers.
enum Electrostatics {
    ReactionField(ReactionField),
    Pme(Box<PmeEngine>),
}

impl Electrostatics {
    fn new(system: &ParameterizedSystem, protocol: &SimulationProtocol) -> Result<Self> {
        match protocol.electrostatics {
            ElectrostaticsModel::ReactionField => Ok(Self::ReactionField(rf_backend(protocol)?)),
            ElectrostaticsModel::Pme => {
                if protocol.has_npt() && !protocol.uses_leapfrog() {
                    return Err(invalid(
                        "PME at constant pressure needs the Nose-Hoover leap-frog integrator with Parrinello-Rahman; the Monte Carlo barostat evaluates reaction field only",
                    ));
                }
                let engine = PmeEngine::new(system, pme_parameters(system, protocol)?)
                    .map_err(Error::Energy)?;
                Ok(Self::Pme(Box::new(engine)))
            }
        }
    }

    /// The reaction-field function, for the Monte Carlo barostat and the
    /// pressure probe, which are written for it.
    fn reaction_field(&self) -> Result<ReactionField> {
        match self {
            Self::ReactionField(backend) => Ok(*backend),
            Self::Pme(_) => Err(invalid("this path evaluates reaction field only")),
        }
    }

    /// Energy, gradient and virial on the f64 reference path.
    fn reference(
        &mut self,
        field: &PbcForceField,
        coordinates: &[Vec3],
        box_vec: &BoxVectors,
        pairs: &[(usize, usize)],
        protocol: &SimulationProtocol,
        dispersion_coefficient: f64,
    ) -> Result<glysys_energy::pbc::PbcEnergy> {
        let cutoff = cutoff_angstrom(protocol);
        match self {
            Self::ReactionField(backend) => field
                .evaluate_with_dispersion_coefficient(
                    coordinates,
                    box_vec,
                    pairs,
                    backend,
                    cutoff,
                    dispersion_coefficient,
                    protocol.dispersion_correction,
                )
                .map_err(Error::Energy),
            Self::Pme(engine) => {
                let mut energy = field
                    .evaluate_pme(engine, coordinates, box_vec, pairs, cutoff)
                    .map_err(Error::Energy)?;
                if protocol.dispersion_correction {
                    energy.components.dispersion_correction =
                        dispersion_coefficient / box_vec.volume();
                }
                Ok(energy)
            }
        }
    }
}

pub fn rf_backend(protocol: &SimulationProtocol) -> Result<ReactionField> {
    if protocol.electrostatics != ElectrostaticsModel::ReactionField {
        return Err(invalid(
            "this path evaluates reaction-field electrostatics only",
        ));
    }
    ReactionField::new(
        cutoff_angstrom(protocol),
        protocol.rf_dielectric.unwrap_or(78.5),
    )
    .map_err(Error::Energy)
}

fn electrostatics_kind(protocol: &SimulationProtocol) -> NonbondedElectrostatics {
    match protocol.electrostatics {
        ElectrostaticsModel::ReactionField => NonbondedElectrostatics::ReactionField {
            cutoff_angstrom: cutoff_angstrom(protocol),
            solvent_dielectric: protocol.rf_dielectric.unwrap_or(78.5),
        },
        ElectrostaticsModel::Pme => NonbondedElectrostatics::Pme {
            // Fingerprint label only: the settings that fix alpha and the
            // grid (cutoff, tolerance, spacing, order, prepared box) are
            // hashed with the protocol and the system.
            alpha_per_angstrom: 0.35,
            grid: [0, 0, 0],
            interpolation_order: 4,
        },
    }
}

/// Cluster-pair engine for the dynamics force loop, in reaction-field or
/// PME mode. Setting the environment variable `GLYSYS_CPU_REFERENCE_PAIRS`
/// keeps the f64 reference pair loop (for validation runs).
fn cluster_engine(
    system: &ParameterizedSystem,
    protocol: &SimulationProtocol,
) -> Result<Option<ClusterPairEngine>> {
    if reference_pairs_requested() {
        return Ok(None);
    }
    match protocol.electrostatics {
        ElectrostaticsModel::ReactionField => ClusterPairEngine::new(
            system,
            cutoff_angstrom(protocol),
            SKIN_ANGSTROM,
            protocol.rf_dielectric.unwrap_or(78.5),
        ),
        ElectrostaticsModel::Pme => ClusterPairEngine::new_pme(
            system,
            cutoff_angstrom(protocol),
            SKIN_ANGSTROM,
            pme_parameters(system, protocol)?.alpha_per_angstrom,
        ),
    }
    .map(Some)
    .map_err(Error::Energy)
}

#[cfg(not(target_arch = "wasm32"))]
fn reference_pairs_requested() -> bool {
    std::env::var_os("GLYSYS_CPU_REFERENCE_PAIRS").is_some()
}

#[cfg(target_arch = "wasm32")]
fn reference_pairs_requested() -> bool {
    false
}

/// Covalent bonds involving solute hydrogen, constrained alongside SETTLE
/// waters so 2 fs steps stay stable. Targets come from the minimized geometry.
fn solute_h_bonds(system: &ParameterizedSystem, waters: &[[usize; 3]]) -> Vec<(usize, usize, f64)> {
    let in_water: std::collections::HashSet<usize> = waters.iter().flatten().copied().collect();
    system
        .bonds()
        .iter()
        .filter_map(|b| {
            let [a, c] = b.atoms();
            (!in_water.contains(&a)
                && !in_water.contains(&c)
                && (system.atoms()[a].element() == 1 || system.atoms()[c].element() == 1))
                .then_some((a, c, b.length()))
        })
        .collect()
}

/// Per-water equilibrium targets from the force-field topology.
fn water_targets(
    system: &ParameterizedSystem,
    waters: &[[usize; 3]],
) -> Result<Vec<(f64, f64, f64)>> {
    waters
        .iter()
        .map(|w| glysys_energy::pbc::water_equilibrium(system, *w).map_err(Error::Energy))
        .collect()
}

/// Solute heavy-atom positional restraints at the prepared geometry.
pub(crate) fn solute_restraints(
    system: &ParameterizedSystem,
    force: f64,
) -> Vec<HarmonicRestraint> {
    if force <= 0. {
        return Vec::new();
    }
    let waters: std::collections::HashSet<usize> =
        classify_waters(system).into_iter().flatten().collect();
    system
        .atoms()
        .iter()
        .enumerate()
        .filter(|(i, a)| !waters.contains(i) && a.element() > 1)
        .map(|(i, a)| HarmonicRestraint {
            atom: i,
            reference: a.position(),
            force,
        })
        .collect()
}

fn fingerprint(
    system: &ParameterizedSystem,
    protocol: &SimulationProtocol,
    restraints: usize,
) -> String {
    // Keep the original digest byte-for-byte for v1 NVE/NVT checkpoints. The
    // corrected model intentionally gets a new digest because its volume
    // measure, dispersion term, and/or stage schedule are part of the
    // physical state.
    if model_version(protocol) == super::EXPLICIT_MODEL_VERSION {
        return format!(
            "{:x}",
            Sha256::digest(
                format!(
                    "{}:{:?}:{:?}:{restraints}:{system:?}",
                    super::EXPLICIT_MODEL_VERSION,
                    electrostatics_kind(protocol),
                    protocol.constraints,
                )
                .as_bytes(),
            )
        );
    }
    // LF-middle benchmark windows legitimately extend the production-stage
    // length while retaining the exact step-zero state. The trajectory plan
    // is not part of the physical state identity for this model; hash the
    // physical recipe and integrator settings, but not stage lengths or
    // coordinate-output cadence.
    if model_version(protocol) == super::EXPLICIT_LF_MIDDLE_MODEL_VERSION {
        return format!(
            "{:x}",
            Sha256::digest(
                format!(
                    "{}:{:?}:{:?}:{:?}:dt={}:T={}:friction={}:cutoff={:?}:rf={:?}:dispersion={}:restraints={restraints}:{system:?}",
                    model_version(protocol),
                    electrostatics_kind(protocol),
                    protocol.constraints,
                    protocol.thermostat,
                    protocol.timestep_fs,
                    protocol.temperature_k,
                    protocol.friction_per_ps,
                    protocol.cutoff_angstrom,
                    protocol.rf_dielectric,
                    protocol.dispersion_correction,
                )
                .as_bytes(),
            )
        );
    }
    // The leap-frog model adds its coupling intervals and time step: they
    // set the equations of motion as much as the coupling times do.
    let leapfrog = if protocol.uses_leapfrog() {
        format!(
            "dt={}:T={}:P={}:cutoff={:?}:rf={:?}:tc-interval={:?}:pc-interval={:?}:com-interval={:?}:ewald={:?}:grid={:?}:pme-order={:?}:",
            protocol.timestep_fs,
            protocol.temperature_k,
            protocol.pressure_bar,
            protocol.cutoff_angstrom,
            protocol.rf_dielectric,
            protocol.temperature_coupling_interval,
            protocol.pressure_coupling_interval,
            protocol.com_removal_interval,
            protocol.ewald_tolerance,
            protocol.fourier_spacing_angstrom,
            protocol.pme_order,
        )
    } else {
        String::new()
    };
    format!(
        "{:x}",
        Sha256::digest(
                format!(
                "{}:{:?}:{:?}:pressure-coupling={:?}:thermostat={:?}:dispersion={}:stages={:?}:thermostat-groups={:?}:pressure-tau={:?}:compressibility={:?}:com-mode={:?}:com-groups={:?}:{leapfrog}{restraints}:{system:?}",
                model_version(protocol),
                electrostatics_kind(protocol),
                protocol.constraints,
                protocol.pressure_coupling,
                protocol.thermostat,
                protocol.dispersion_correction,
                protocol.execution_stages(),
                protocol.thermostat_groups,
                protocol.pressure_tau_ps,
                protocol.pressure_compressibility_bar_inverse,
                protocol.com_mode,
                protocol.com_groups,
            )
            .as_bytes()
        )
    )
}

pub fn model_version(protocol: &SimulationProtocol) -> &'static str {
    if protocol.uses_leapfrog() {
        super::EXPLICIT_LEAPFROG_MODEL_VERSION
    } else if protocol.langevin_discretization == LangevinDiscretization::LfMiddle {
        super::EXPLICIT_LF_MIDDLE_MODEL_VERSION
    } else if protocol.has_npt() || protocol.dispersion_correction || protocol.stages.is_some() {
        super::EXPLICIT_NPT_MODEL_VERSION
    } else {
        super::EXPLICIT_MODEL_VERSION
    }
}

pub fn kinetic_temperature(kinetic_kcal_mol: f64, degrees_of_freedom: usize) -> f64 {
    if degrees_of_freedom == 0 {
        return 0.;
    }
    2. * kinetic_kcal_mol / (degrees_of_freedom as f64 * KB)
}

pub fn kinetic_energy(masses: &[f64], velocities: &[Vec3]) -> f64 {
    masses
        .iter()
        .zip(velocities)
        .map(|(m, v)| m * (v.x * v.x + v.y * v.y + v.z * v.z) / (2. * ACCEL))
        .sum()
}

/// Bussi-Parrinello canonical velocity rescaling (JCP 127, 014102, 2007),
/// integrated with the mean-preserving Euler discretization of its kinetic-
/// energy SDE: `dK = (K0-K) dt/tau + 2 sqrt(K K0 dt/(Nf tau)) R`. The mean
/// kinetic energy is exactly preserved by construction, and the stochastic
/// term gives the correct canonical fluctuations to O(dt); unlike a recalled
/// closed-form propagator this cannot double-count the retained energy or
/// blow up for small dt. Returns the scale factor applied (1 on any fault).
pub fn vrescale_factor(
    rng: &mut u64,
    kinetic: f64,
    target_temperature_k: f64,
    degrees_of_freedom: usize,
    dt_ps: f64,
    tau_ps: f64,
) -> f64 {
    let nf = degrees_of_freedom.max(1) as f64;
    if !(kinetic.is_finite() && kinetic > 0.) || !(target_temperature_k > 0.) {
        return 1.;
    }
    let k0 = nf * KB * target_temperature_k / 2.;
    let h = dt_ps / tau_ps.max(dt_ps);
    let dk = (k0 - kinetic) * h + 2. * (kinetic * k0 * h / nf).sqrt() * normal(rng);
    let k_target = kinetic + dk;
    if !k_target.is_finite() || k_target <= 0. {
        return 1.;
    }
    (k_target / kinetic).sqrt()
}

pub fn pressure_bar(n_atoms: usize, temperature_k: f64, virial: f64, volume_a3: f64) -> f64 {
    if volume_a3 <= 0. {
        return 0.;
    }
    (n_atoms as f64 * KB * temperature_k + virial / 3.) / volume_a3 * BAR_PER_KCAL_MOL_A3
}

/// Pressure for a constrained molecular system from the intermolecular
/// configurational virial and center-of-mass kinetic energy. Internal bonds,
/// angles, torsions, and rigid-water constraint forces do not contribute to
/// the molecular volume derivative because a barostat translates each whole
/// molecule without changing its internal geometry.
///
/// `pair_virial` is \(\sum r_{ij}\cdot F_{ij}\) over intermolecular pairs,
/// `molecular_kinetic` is the kinetic energy of molecule COM motion in
/// kcal/mol, and `dispersion_coefficient` is the homogeneous LJ correction
/// coefficient for `U_disp = C_disp / V`.
pub fn molecular_pressure_bar_from_pair_virial(
    pair_virial: f64,
    molecular_kinetic: f64,
    volume_a3: f64,
    dispersion_coefficient: f64,
    dispersion_enabled: bool,
) -> f64 {
    if !pair_virial.is_finite()
        || !molecular_kinetic.is_finite()
        || !volume_a3.is_finite()
        || volume_a3 <= 0.
        || !dispersion_coefficient.is_finite()
    {
        return 0.;
    }
    let kinetic = 2. * molecular_kinetic / (3. * volume_a3);
    let configurational = pair_virial / (3. * volume_a3);
    let dispersion = if dispersion_enabled {
        dispersion_coefficient / volume_a3.powi(2)
    } else {
        0.
    };
    (kinetic + configurational + dispersion) * BAR_PER_KCAL_MOL_A3
}

/// Apply an isotropic molecule-preserving volume scale.  The arithmetic
/// centroid convention is the one used by OpenMM's rigid-molecule barostat;
/// all atoms in a covalent/constraint group receive the same translation, so
/// internal distances and velocities are unchanged.
pub fn scale_molecules(
    coordinates: &[Vec3],
    molecules: &[Vec<usize>],
    linear_scale: f64,
) -> Result<Vec<Vec3>> {
    if !linear_scale.is_finite() || linear_scale <= 0. {
        return Err(invalid("molecule scale must be finite and positive"));
    }
    let mut scaled = coordinates.to_vec();
    for molecule in molecules {
        if molecule.is_empty() {
            continue;
        }
        let mut center = Vec3 {
            x: 0.,
            y: 0.,
            z: 0.,
        };
        for &atom in molecule {
            let Some(position) = coordinates.get(atom) else {
                return Err(invalid("molecule atom index out of range"));
            };
            center = add(center, *position);
        }
        center = scale(center, 1. / molecule.len() as f64);
        let shift = scale(center, linear_scale - 1.);
        for &atom in molecule {
            scaled[atom] = add(coordinates[atom], shift);
        }
    }
    Ok(scaled)
}

/// Dimensionful Monte Carlo barostat work in kcal/mol.  Keeping this pure
/// makes the molecule-count Jacobian and pressure conversion independently
/// testable without constructing an evaluator or consuming RNG state.
pub fn barostat_work(
    current_energy: f64,
    proposed_energy: f64,
    current_volume: f64,
    proposed_volume: f64,
    pressure_bar: f64,
    temperature_k: f64,
    molecule_count: usize,
) -> Result<f64> {
    if ![
        current_energy,
        proposed_energy,
        current_volume,
        proposed_volume,
        pressure_bar,
        temperature_k,
    ]
    .iter()
    .all(|value| value.is_finite())
        || current_volume <= 0.
        || proposed_volume <= 0.
        || temperature_k <= 0.
        || pressure_bar <= 0.
        || molecule_count == 0
    {
        return Err(invalid("invalid barostat work inputs"));
    }
    let pv = pressure_bar * (proposed_volume - current_volume) / BAR_PER_KCAL_MOL_A3;
    Ok(proposed_energy - current_energy + pv
        - molecule_count as f64 * KB * temperature_k * (proposed_volume / current_volume).ln())
}

/// Decide a barostat move from a supplied log-uniform variate.  A variate of
/// zero is accepted for every finite work value; the open interval is useful
/// for deterministic tests and does not alter the runtime two-draw stream.
pub fn barostat_accept(work: f64, temperature_k: f64, log_uniform: f64) -> Result<bool> {
    if !work.is_finite()
        || !temperature_k.is_finite()
        || temperature_k <= 0.
        || !log_uniform.is_finite()
        || log_uniform > 0.
    {
        return Err(invalid("invalid barostat acceptance inputs"));
    }
    Ok(log_uniform < (0.0f64).min(-work / (KB * temperature_k)))
}

/// Molecular pressure estimator matching OpenMM's barostat diagnostic. The
/// kinetic contribution uses molecule center-of-mass velocities and the
/// configurational contribution is a reversible finite difference under the
/// same rigid-centroid scaling used by the Monte Carlo move. This method is
/// intentionally an explicit probe and does not mutate coordinates, pairs,
/// or RNG state.
pub fn molecular_pressure_bar(
    field: &PbcForceField,
    backend: &ReactionField,
    protocol: &SimulationProtocol,
    box_vec: &BoxVectors,
    coordinates: &[Vec3],
    velocities: &[Vec3],
    masses: &[f64],
    molecules: &[Vec<usize>],
) -> Result<f64> {
    if coordinates.len() != masses.len() || velocities.len() != masses.len() {
        return Err(invalid("pressure probe state dimensions do not match"));
    }
    let volume = box_vec.volume();
    let perturb = 1.0e-3;
    let scaled_energy = |factor: f64| -> Result<f64> {
        let trial_box = BoxVectors {
            x: box_vec.x * factor,
            y: box_vec.y * factor,
            z: box_vec.z * factor,
        };
        let mut trial = coordinates.to_vec();
        for molecule in molecules {
            if molecule.is_empty() {
                continue;
            }
            let mut center = Vec3 {
                x: 0.,
                y: 0.,
                z: 0.,
            };
            for &atom in molecule {
                center = add(center, coordinates[atom]);
            }
            center = scale(center, 1. / molecule.len() as f64);
            let shift = scale(center, factor - 1.);
            for &atom in molecule {
                trial[atom] = add(coordinates[atom], shift);
            }
        }
        let wrapped: Vec<Vec3> = trial.iter().map(|p| trial_box.wrap(*p)).collect();
        let pairs = PbcNeighborList::build(
            &wrapped,
            &trial_box,
            cutoff_angstrom(protocol),
            SKIN_ANGSTROM,
        )
        .map_err(Error::Energy)?;
        Ok(field
            .evaluate_with_dispersion(
                &trial,
                &trial_box,
                &pairs.pairs,
                backend,
                cutoff_angstrom(protocol),
                protocol.dispersion_correction,
            )?
            .components
            .total())
    };
    let e_plus = scaled_energy(1. + perturb)?;
    let e_minus = scaled_energy(1. - perturb)?;
    let d_volume = volume * ((1. + perturb).powi(3) - (1. - perturb).powi(3));
    let d_u_d_v = (e_plus - e_minus) / d_volume;
    let mut k_com = 0.;
    for molecule in molecules {
        if molecule.is_empty() {
            continue;
        }
        let mut momentum = Vec3 {
            x: 0.,
            y: 0.,
            z: 0.,
        };
        let mut mass = 0.;
        for &atom in molecule {
            momentum = add(momentum, scale(velocities[atom], masses[atom]));
            mass += masses[atom];
        }
        if mass > 0. {
            k_com += norm2(scale(momentum, 1. / mass)) * mass / (2. * ACCEL);
        }
    }
    Ok((2. * k_com / (3. * volume) - d_u_d_v) * BAR_PER_KCAL_MOL_A3)
}

/// Coupling set-up for a leap-frog protocol; `None` for the other
/// integrators.
fn coupling_plan(
    atom_count: usize,
    waters: &[[usize; 3]],
    h_bonds: &[(usize, usize, f64)],
    protocol: &SimulationProtocol,
) -> Result<Option<Arc<CouplingPlan>>> {
    if !protocol.uses_leapfrog() {
        return Ok(None);
    }
    let mut is_water = vec![false; atom_count];
    let mut constraints = vec![0u32; atom_count];
    for water in waters {
        for &atom in water {
            is_water[atom] = true;
            constraints[atom] += 2;
        }
    }
    for &(a, b, _) in h_bonds {
        constraints[a] += 1;
        constraints[b] += 1;
    }
    CouplingPlan::new(protocol, &is_water, &constraints).map(|plan| Some(Arc::new(plan)))
}

/// First atom of each atom's molecule: the origin its constraint forces are
/// measured from in the virial, so the sum does not depend on where the
/// unwrapped molecule has drifted.
fn molecule_anchors(atom_count: usize, molecules: &[Vec<usize>]) -> Vec<u32> {
    let mut anchors: Vec<u32> = (0..atom_count as u32).collect();
    for molecule in molecules {
        if let Some(&first) = molecule.iter().min() {
            for &atom in molecule {
                anchors[atom] = first as u32;
            }
        }
    }
    anchors
}

/// Subtract the center-of-mass velocity of each group from its atoms.
fn remove_com_motion(groups: &[Vec<usize>], masses: &[f64], velocities: &mut [Vec3]) {
    for group in groups {
        let mut momentum = Vec3 {
            x: 0.,
            y: 0.,
            z: 0.,
        };
        let mut mass = 0.;
        for &atom in group {
            momentum = add(momentum, scale(velocities[atom], masses[atom]));
            mass += masses[atom];
        }
        if mass <= 0. {
            continue;
        }
        let drift = scale(momentum, -1. / mass);
        for &atom in group {
            velocities[atom] = add(velocities[atom], drift);
        }
    }
}

fn schema_version(protocol: &SimulationProtocol) -> u32 {
    match model_version(protocol) {
        version if version == super::EXPLICIT_LEAPFROG_MODEL_VERSION => 3,
        version if version == super::EXPLICIT_NPT_MODEL_VERSION => 2,
        _ => 1,
    }
}

fn dof_count(atom_count: usize, settle: &Option<SettleWaters>) -> usize {
    let constrained = settle.as_ref().map(|s| s.constraint_count()).unwrap_or(0);
    (3 * atom_count).saturating_sub(constrained).max(1)
}

/// Constrained degrees of freedom for reporting, without building a
/// simulation or evaluating energies. GPU-installed states use this so their
/// trajectory frames report the same kinetic temperature as CPU frames.
pub fn degrees_of_freedom(
    system: &ParameterizedSystem,
    protocol: &SimulationProtocol,
) -> Result<usize> {
    let masses: Vec<_> = system.atoms().iter().map(|a| a.mass()).collect();
    let waters = classify_waters(system);
    let h_bonds = solute_h_bonds(system, &waters);
    let targets = water_targets(system, &waters)?;
    if let Some(plan) = coupling_plan(system.atom_count(), &waters, &h_bonds, protocol)? {
        return Ok(plan.degrees_of_freedom().round() as usize);
    }
    let settle = match protocol.constraints {
        ConstraintModel::None => None,
        ConstraintModel::Settle => Some(SettleWaters::from_equilibrium(
            waters,
            targets,
            h_bonds,
            &masses,
            system.atom_count(),
        )?),
        ConstraintModel::HBonds => {
            return Err(invalid("explicit solvent requires constraints=settle"));
        }
    };
    Ok(dof_count(system.atom_count(), &settle))
}

pub fn density_g_ml(total_amu: f64, volume_a3: f64) -> f64 {
    if volume_a3 <= 0. {
        return 0.;
    }
    total_amu * 1.660_539_066_60 / volume_a3
}

pub struct ExplicitSimulation<'a> {
    field: PbcForceField<'a>,
    masses: Vec<f64>,
    settle: Option<SettleWaters>,
    waters: Vec<[usize; 3]>,
    molecules: Vec<Vec<usize>>,
    electrostatics: Electrostatics,
    pairs: PbcNeighborList,
    /// Single-precision cluster-pair engine for the dynamics force loop.
    /// `None` keeps the f64 reference pair list for every evaluation.
    cluster: Option<ClusterPairEngine>,
    /// Set by [`Self::advance`] for steps whose energy, virial and pressure
    /// are not observed (no frame, stage boundary or chunk end follows).
    force_only: bool,
    /// Homogeneous long-range LJ correction, precomputed per topology and
    /// cutoff. It is a volume-only term and therefore adds no Cartesian
    /// force in the fixed-box integrator.
    dispersion_coefficient: f64,
    /// Its counterpart for a pressure computed from forces
    /// ([`PbcForceField::dispersion_pressure_coefficient`]).
    dispersion_pressure_coefficient: f64,
    /// Unconstrained velocity DOF: 3N minus one per distance constraint.
    degrees_of_freedom: usize,
    /// Temperature groups and coupling intervals of the leap-frog
    /// integrator; `None` for the other integrators.
    coupling_plan: Option<Arc<CouplingPlan>>,
    /// See [`molecule_anchors`].
    constraint_anchor: Vec<u32>,
    /// Whether `state.virial_kcal_mol` holds the full virial of the stored
    /// gradient (force-only evaluations leave the pair part out).
    virial_current: bool,
    scratch: LeapfrogScratch,
    /// `-ACCEL / mass` of every atom: gradient to acceleration.
    kick_per_ps: Vec<f64>,
    pub state: SimulationState,
}

/// Buffers of the leap-frog step, kept between steps: the new velocities,
/// the positions before and after the constraints.
#[derive(Default)]
struct LeapfrogScratch {
    velocities: Vec<Vec3>,
    trial: Vec<Vec3>,
    coordinates: Vec<Vec3>,
}

impl<'a> ExplicitSimulation<'a> {
    pub fn into_owned(self) -> ExplicitSimulation<'static> {
        ExplicitSimulation {
            field: self.field.into_owned(),
            masses: self.masses,
            settle: self.settle,
            waters: self.waters,
            molecules: self.molecules,
            electrostatics: self.electrostatics,
            pairs: self.pairs,
            cluster: self.cluster,
            force_only: self.force_only,
            dispersion_coefficient: self.dispersion_coefficient,
            dispersion_pressure_coefficient: self.dispersion_pressure_coefficient,
            degrees_of_freedom: self.degrees_of_freedom,
            coupling_plan: self.coupling_plan,
            constraint_anchor: self.constraint_anchor,
            virial_current: self.virial_current,
            scratch: self.scratch,
            kick_per_ps: self.kick_per_ps,
            state: self.state,
        }
    }

    pub fn new(system: &'a ParameterizedSystem, protocol: SimulationProtocol) -> Result<Self> {
        Self::new_with_progress(system, protocol, |_, _, _, _, _| {})
    }

    /// Construct an explicit simulation while reporting bounded preparation
    /// and minimization milestones.  The callback is deliberately a small
    /// engine-level contract rather than a browser type: native callers can
    /// log it, while the WASM adapter can forward it to the worker without
    /// changing the numerical path.
    pub fn new_with_progress<F>(
        system: &'a ParameterizedSystem,
        protocol: SimulationProtocol,
        progress: F,
    ) -> Result<Self>
    where
        F: FnMut(&str, usize, usize, f64, f64),
    {
        Self::construct(system, protocol, None, progress)
    }
    /// Finalize the same physical state after an external evaluator has
    /// completed the shared preparation minimizer. Constraint initialization
    /// and thermal velocities remain identical to the synchronous constructor.
    pub fn from_minimized(
        system: &'a ParameterizedSystem,
        protocol: SimulationProtocol,
        coordinates: Vec<Vec3>,
    ) -> Result<Self> {
        Self::construct(system, protocol, Some(coordinates), |_, _, _, _, _| {})
    }
    fn construct<F>(
        system: &'a ParameterizedSystem,
        protocol: SimulationProtocol,
        minimized: Option<Vec<Vec3>>,
        mut progress: F,
    ) -> Result<Self>
    where
        F: FnMut(&str, usize, usize, f64, f64),
    {
        progress("prepare", 0, 1, 0., 0.);
        protocol.validate()?;
        if protocol.solvent != SolventModel::Explicit {
            return Err(invalid("explicit driver needs solvent=explicit"));
        }
        if protocol.has_npt()
            && protocol.pressure_coupling == PressureCoupling::ParrinelloRahman
            && !protocol.uses_leapfrog()
        {
            return Err(invalid(
                "Parrinello-Rahman requires the Nose-Hoover leap-frog integrator",
            ));
        }
        if protocol.timestep_fs > max_timestep_fs(protocol.constraints) {
            return Err(invalid(format!(
                "timestep {:.2} fs exceeds the {:?} limit of {:.1} fs",
                protocol.timestep_fs,
                protocol.constraints,
                max_timestep_fs(protocol.constraints)
            )));
        }
        let box_vec = BoxVectors::from_system(system).map_err(Error::Energy)?;
        let cutoff = cutoff_angstrom(&protocol);
        if cutoff >= 0.5 * box_vec.x.min(box_vec.y).min(box_vec.z) {
            return Err(invalid("cutoff must be below half the shortest box edge"));
        }
        if system.atoms().is_empty()
            || system
                .atoms()
                .iter()
                .any(|a| !a.mass().is_finite() || a.mass() <= 0.)
        {
            return Err(invalid("finite positive masses required"));
        }
        let restraints = solute_restraints(system, protocol.restraint_force);
        let field = PbcForceField::new(system, restraints.clone()).map_err(Error::Energy)?;
        let dispersion_coefficient = if protocol.dispersion_correction {
            field
                .dispersion_coefficient(cutoff)
                .map_err(Error::Energy)?
        } else {
            0.0
        };
        let dispersion_pressure_coefficient = if protocol.dispersion_correction {
            field
                .dispersion_pressure_coefficient(cutoff)
                .map_err(Error::Energy)?
        } else {
            0.0
        };
        let mut electrostatics = Electrostatics::new(system, &protocol)?;
        let waters = classify_waters(system);
        if waters.is_empty() {
            return Err(invalid(
                "explicit solvent needs classified TIP3P waters; prepare with solvation",
            ));
        }
        let molecules = molecules(system);
        let masses: Vec<_> = system.atoms().iter().map(|a| a.mass()).collect();
        let externally_minimized = minimized.is_some();
        let mut coordinates = minimized.unwrap_or_else(|| system.coordinates());
        if coordinates.len() != system.atom_count()
            || coordinates
                .iter()
                .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
        {
            return Err(invalid("invalid minimized coordinates"));
        }
        progress("parameterize", 1, 1, 0., 0.);
        // Flexible minimization first; constraints engage at dynamics start,
        // matching the OpenMM validation protocol.
        if protocol.minimization_iterations > 0 && !externally_minimized {
            progress(
                "minimize-pre",
                0,
                protocol.minimization_iterations.min(500),
                0.,
                0.,
            );
            coordinates = minimize(
                &field,
                &box_vec,
                &mut electrostatics,
                coordinates,
                &protocol,
                dispersion_coefficient,
                &mut progress,
            )?;
        }
        let h_bonds = solute_h_bonds(system, &waters);
        let targets = water_targets(system, &waters)?;
        let coupling_plan = coupling_plan(system.atom_count(), &waters, &h_bonds, &protocol)?;
        let settle = match protocol.constraints {
            ConstraintModel::None => None,
            ConstraintModel::Settle => Some(
                SettleWaters::from_equilibrium(
                    waters.clone(),
                    targets,
                    h_bonds,
                    &masses,
                    system.atom_count(),
                )
                .map_err(|e| invalid(format!("invalid constraint topology: {e}")))?,
            ),
            ConstraintModel::HBonds => {
                return Err(invalid("explicit solvent requires constraints=settle"));
            }
        };
        if let Some(settle) = &settle {
            settle.constrain_positions(&mut coordinates)?;
        }
        let wrapped: Vec<Vec3> = coordinates.iter().map(|p| box_vec.wrap(*p)).collect();
        let pairs = PbcNeighborList::build(&wrapped, &box_vec, cutoff, SKIN_ANGSTROM)
            .map_err(Error::Energy)?;
        let energy = electrostatics.reference(
            &field,
            &coordinates,
            &box_vec,
            &pairs.pairs,
            &protocol,
            dispersion_coefficient,
        )?;
        let mut rng = protocol.seed;
        let mut velocities = masses
            .iter()
            .map(|m| random_velocity(&mut rng, (KB * protocol.temperature_k * ACCEL / m).sqrt()))
            .collect::<Vec<_>>();
        if let Some(settle) = &settle {
            settle.constrain_velocities(&coordinates, &mut velocities)?;
        }
        if let Some(plan) = coupling_plan.as_ref().filter(|plan| plan.com_interval > 0) {
            remove_com_motion(&plan.com_groups, &masses, &mut velocities);
        }
        let barostat_rng = protocol_seed_barostat(protocol_seed(&protocol));
        let has_npt = protocol.has_npt();
        let degrees_of_freedom = coupling_plan.as_ref().map_or_else(
            || dof_count(system.atom_count(), &settle),
            |plan| plan.degrees_of_freedom().round() as usize,
        );
        let coupling = coupling_plan
            .as_ref()
            .map(|plan| CouplingState::new(plan.groups.len()));
        let constraint_anchor = molecule_anchors(system.atom_count(), &molecules);
        let resident_rng = (protocol.thermostat == Thermostat::Langevin).then(|| {
            super::resident_rng::ResidentThermostatRng::seeded(protocol.seed, system.atom_count())
        });
        let velocity_convention = VelocityConvention::for_protocol(&protocol);
        let cluster = cluster_engine(system, &protocol)?;
        let mut sim = Self {
            field,
            kick_per_ps: masses.iter().map(|m| -ACCEL / m).collect(),
            masses,
            settle,
            waters,
            molecules,
            electrostatics,
            pairs,
            cluster,
            force_only: false,
            dispersion_coefficient,
            dispersion_pressure_coefficient,
            degrees_of_freedom,
            coupling_plan,
            constraint_anchor,
            virial_current: true,
            scratch: LeapfrogScratch::default(),
            state: SimulationState {
                schema_version: schema_version(&protocol),
                model_version: model_version(&protocol).into(),
                velocity_convention,
                degrees_of_freedom,
                system_fingerprint: String::new(),
                protocol,
                step: 0,
                reference_coordinates: coordinates.clone(),
                coordinates,
                velocities,
                gradient: energy.gradients,
                potential_energy: energy.components.total(),
                rng_state: rng,
                resident_rng,
                integrator_phase: "ready".into(),
                box_angstrom: box_vec.as_array(),
                virial_kcal_mol: energy.virial,
                pressure_bar: 0.,
                pressure_estimator: "atomic-virial".into(),
                barostat_rng,
                barostat_volume_width: if has_npt { box_vec.volume() * 0.01 } else { 0. },
                barostat_attempts: 0,
                barostat_accepts: 0,
                barostat_window_attempts: 0,
                barostat_window_accepts: 0,
                barostat_step_counter: 0,
                barostat_frozen: false,
                water_occupancy: None,
                coupling,
            },
        };
        sim.state.system_fingerprint =
            fingerprint(system, &sim.state.protocol, sim.field_restraint_count());
        sim.state.pressure_bar = pressure_bar(
            sim.masses.len(),
            sim.state.protocol.temperature_k,
            energy.virial,
            box_vec.volume(),
        );
        if let Some(settle) = &sim.settle {
            settle.constrain_velocities(&sim.state.coordinates, &mut sim.state.velocities)?;
        }
        progress("ready", 1, 1, sim.state.potential_energy, 0.);
        Ok(sim)
    }

    fn field_restraint_count(&self) -> usize {
        self.masses.len().saturating_sub(self.waters.len() * 3)
    }

    pub fn restore(system: &'a ParameterizedSystem, state: SimulationState) -> Result<Self> {
        state.protocol.validate()?;
        if let Some(rng) = &state.resident_rng {
            rng.validate(system.atom_count())?;
            if state.protocol.thermostat != Thermostat::Langevin {
                return Err(invalid("resident RNG requires a Langevin thermostat"));
            }
        }
        let expected_model = model_version(&state.protocol);
        let expected_schema = schema_version(&state.protocol);
        if state.schema_version != expected_schema
            || state.model_version != expected_model
            || state.velocity_convention != VelocityConvention::for_protocol(&state.protocol)
            || state.integrator_phase != "ready"
            || state.coordinates.len() != system.atom_count()
            || state.velocities.len() != system.atom_count()
            || state.gradient.len() != system.atom_count()
            || state.step > state.protocol.total_steps()
        {
            return Err(invalid("incompatible or invalid explicit checkpoint"));
        }
        if state.protocol.timestep_fs > max_timestep_fs(state.protocol.constraints) {
            return Err(invalid("checkpoint timestep exceeds the constraint limit"));
        }
        let box_vec = BoxVectors::new(
            state.box_angstrom[0],
            state.box_angstrom[1],
            state.box_angstrom[2],
        )
        .map_err(Error::Energy)?;
        let restraints = solute_restraints(system, state.protocol.restraint_force);
        let field = PbcForceField::new(system, restraints).map_err(Error::Energy)?;
        let dispersion_coefficient = if state.protocol.dispersion_correction {
            field
                .dispersion_coefficient(cutoff_angstrom(&state.protocol))
                .map_err(Error::Energy)?
        } else {
            0.0
        };
        let dispersion_pressure_coefficient = if state.protocol.dispersion_correction {
            field
                .dispersion_pressure_coefficient(cutoff_angstrom(&state.protocol))
                .map_err(Error::Energy)?
        } else {
            0.0
        };
        let mut electrostatics = Electrostatics::new(system, &state.protocol)?;
        let waters = classify_waters(system);
        let masses: Vec<_> = system.atoms().iter().map(|a| a.mass()).collect();
        let h_bonds = solute_h_bonds(system, &waters);
        let targets = water_targets(system, &waters)?;
        let coupling_plan =
            coupling_plan(system.atom_count(), &waters, &h_bonds, &state.protocol)?;
        if coupling_plan.as_ref().map(|plan| plan.groups.len())
            != state.coupling.as_ref().map(|c| c.thermostat_velocity.len())
            || state.coupling.as_ref().is_some_and(|c| {
                c.thermostat_position.len() != c.thermostat_velocity.len()
                    || c.thermostat_velocity.iter().any(|v| !v.is_finite())
                    || c.box_velocity.iter().any(|v| !v.is_finite())
            })
        {
            return Err(invalid(
                "checkpoint coupling variables do not match the protocol",
            ));
        }
        let settle = match state.protocol.constraints {
            ConstraintModel::None => None,
            ConstraintModel::Settle => Some(SettleWaters::from_equilibrium(
                waters.clone(),
                targets,
                h_bonds,
                &masses,
                system.atom_count(),
            )?),
            ConstraintModel::HBonds => {
                return Err(invalid("explicit solvent requires constraints=settle"));
            }
        };
        // Reference re-evaluation guards chemistry mismatches.
        let wrapped: Vec<Vec3> = state.coordinates.iter().map(|p| box_vec.wrap(*p)).collect();
        let pairs = PbcNeighborList::build(
            &wrapped,
            &box_vec,
            cutoff_angstrom(&state.protocol),
            SKIN_ANGSTROM,
        )
        .map_err(Error::Energy)?;
        let reference = electrostatics.reference(
            &field,
            &state.coordinates,
            &box_vec,
            &pairs.pairs,
            &state.protocol,
            dispersion_coefficient,
        )?;
        if (reference.components.total() - state.potential_energy).abs()
            > 1e-3 + 1e-4 * reference.components.total().abs()
        {
            return Err(invalid(
                "checkpoint energy does not match prepared chemistry",
            ));
        }
        let degrees_of_freedom = coupling_plan.as_ref().map_or_else(
            || dof_count(system.atom_count(), &settle),
            |plan| plan.degrees_of_freedom().round() as usize,
        );
        let cluster = cluster_engine(system, &state.protocol)?;
        let molecules = molecules(system);
        let constraint_anchor = molecule_anchors(system.atom_count(), &molecules);
        let leapfrog = coupling_plan.is_some();
        let mut sim = Self {
            field,
            kick_per_ps: masses.iter().map(|m| -ACCEL / m).collect(),
            masses,
            settle,
            waters,
            molecules,
            electrostatics,
            pairs,
            cluster,
            force_only: false,
            dispersion_coefficient,
            dispersion_pressure_coefficient,
            degrees_of_freedom,
            coupling_plan,
            constraint_anchor,
            // A checkpoint may follow a force-only step. The barostat needs
            // the virial of the stored forces, so take it from the
            // evaluation above.
            virial_current: leapfrog,
            scratch: LeapfrogScratch::default(),
            state,
        };
        if leapfrog {
            sim.state.virial_kcal_mol = reference.virial;
        }
        let expected = fingerprint(system, &sim.state.protocol, sim.field_restraint_count());
        if sim.state.system_fingerprint != expected {
            return Err(invalid("checkpoint chemistry fingerprint mismatch"));
        }
        if sim.state.barostat_volume_width <= 0. && sim.state.protocol.has_npt() {
            sim.state.barostat_volume_width = box_vec.volume() * 0.01;
        }
        Ok(sim)
    }

    fn segment_info(&self) -> (usize, String, Ensemble, bool, usize) {
        self.state.protocol.stage_info(self.state.step)
    }

    fn segment_ensemble(&self) -> Ensemble {
        self.segment_info().2
    }

    fn refresh_pairs(&mut self) -> Result<()> {
        let box_vec = self.box_vectors()?;
        let wrapped: Vec<Vec3> = self
            .state
            .coordinates
            .iter()
            .map(|p| box_vec.wrap(*p))
            .collect();
        if self.pairs.box_changed(&box_vec) || self.pairs.needs_rebuild(&wrapped) {
            self.pairs = PbcNeighborList::build(
                &wrapped,
                &box_vec,
                cutoff_angstrom(&self.state.protocol),
                SKIN_ANGSTROM,
            )
            .map_err(Error::Energy)?;
        }
        Ok(())
    }

    fn box_vectors(&self) -> Result<BoxVectors> {
        BoxVectors::new(
            self.state.box_angstrom[0],
            self.state.box_angstrom[1],
            self.state.box_angstrom[2],
        )
        .map_err(Error::Energy)
    }

    /// Transactional step: errors leave committed state and RNG untouched.
    pub fn step(&mut self) -> Result<()> {
        if self.coupling_plan.is_some() {
            self.leapfrog_step()?;
            self.state.step += 1;
            return Ok(());
        }
        if self.cluster.is_none() || self.state.protocol.has_npt() {
            self.refresh_pairs()?;
        }
        let ensemble = self.segment_ensemble();
        let adaptation_enabled = self.segment_info().3;
        // A volume attempt is the only operation after integration that can
        // fail while holding a newly evaluated trial state.  Snapshot at the
        // bounded attempt boundary so an evaluator/capacity error restores
        // the complete pre-step state (including both RNG streams) without
        // cloning the solvated system on ordinary steps.
        let barostat_snapshot = (ensemble == Ensemble::Npt
            && self.state.barostat_step_counter.saturating_add(1)
                % self.state.protocol.barostat_interval
                == 0)
            .then(|| (self.state.clone(), self.pairs.clone()));
        match ensemble {
            Ensemble::Nve => self.nve_step(),
            Ensemble::Nvt => self.nvt_step(),
            Ensemble::Npt => {
                if !adaptation_enabled {
                    // Production starts with the absolute proposal width
                    // learned during equilibration.  Mark it frozen before
                    // the first production attempt so a restart at a stage
                    // boundary cannot adapt it accidentally.
                    self.state.barostat_frozen = true;
                }
                self.nvt_step()?;
                self.state.barostat_step_counter =
                    self.state.barostat_step_counter.saturating_add(1);
                if self.state.barostat_step_counter % self.state.protocol.barostat_interval == 0 {
                    if let Err(error) = self.try_barostat(adaptation_enabled) {
                        if let Some((state, pairs)) = barostat_snapshot {
                            self.state = state;
                            self.pairs = pairs;
                        }
                        return Err(error);
                    }
                }
                Ok(())
            }
        }?;
        self.state.step += 1;
        let box_vec = self.box_vectors()?;
        let (pressure, estimator) = if ensemble == Ensemble::Npt
            && (self
                .state
                .step
                .is_multiple_of(self.state.protocol.save_every)
                || self
                    .state
                    .barostat_step_counter
                    .is_multiple_of(self.state.protocol.barostat_interval))
        {
            match self.electrostatics.reaction_field().and_then(|backend| {
                molecular_pressure_bar(
                    &self.field,
                    &backend,
                    &self.state.protocol,
                    &box_vec,
                    &self.state.coordinates,
                    &self.state.velocities,
                    &self.masses,
                    &self.molecules,
                )
            }) {
                Ok(value) => (value, "molecular-finite-difference"),
                Err(_) => (
                    pressure_bar(
                        self.masses.len(),
                        self.state.protocol.temperature_k,
                        self.state.virial_kcal_mol,
                        box_vec.volume(),
                    ),
                    "atomic-virial",
                ),
            }
        } else {
            (
                pressure_bar(
                    self.masses.len(),
                    self.state.protocol.temperature_k,
                    self.state.virial_kcal_mol,
                    box_vec.volume(),
                ),
                "atomic-virial",
            )
        };
        self.state.pressure_bar = pressure;
        self.state.pressure_estimator = estimator.into();
        Ok(())
    }

    /// Velocity Verlet with constraints: the NVE validation stage.
    ///
    /// Velocity Verlet with analytic SETTLE positions and the matching
    /// closed-form RATTLE velocity projection. Positions are constrained
    /// after the drift and velocities after the final kick, exactly as in the
    /// OpenMM reference integrator. Solute X-H bonds use the bounded general
    /// SHAKE/RATTLE solver.
    fn nve_step(&mut self) -> Result<()> {
        let dt = self.state.protocol.timestep_fs * 0.001;
        let prev_coords = self.state.coordinates.clone();
        let mut next_coords = prev_coords.clone();
        let mut next_vel = self.state.velocities.clone();
        for (i, m) in self.masses.iter().enumerate() {
            next_vel[i] = add(
                next_vel[i],
                scale(self.state.gradient[i], -0.5 * dt * ACCEL / m),
            );
            next_coords[i] = add(next_coords[i], scale(next_vel[i], dt));
        }
        if let Some(settle) = &self.settle {
            settle.settle_positions(
                &prev_coords,
                &next_coords.clone(),
                &mut next_coords,
                &self.masses,
            )?;
            settle.shake_solute_positions(&mut next_coords)?;
            // Position SHAKE/SETTLE supplies a constraint impulse. Rebuild
            // the half-step velocity from the committed displacement before
            // the second force kick; otherwise the position correction is
            // absent from the momentum update and rigid-water NVE drifts.
            settle.velocity_from_displacement(
                &prev_coords,
                &next_coords,
                &mut next_vel,
                1. / dt,
            )?;
        }
        let energy = self.dynamics_energy(&next_coords)?;
        for (i, m) in self.masses.iter().enumerate() {
            next_vel[i] = add(
                next_vel[i],
                scale(energy.gradients[i], -0.5 * dt * ACCEL / m),
            );
        }
        if let Some(settle) = &self.settle {
            settle.constrain_velocities(&next_coords, &mut next_vel)?;
        }
        self.commit(
            next_coords,
            next_vel,
            energy.components.total(),
            energy.gradients,
            energy.virial,
        )
    }

    /// BAOAB Langevin with SETTLE positions and RATTLE velocity projection
    /// after each constrained drift. The stochastic thermostat acts on the
    /// unconstrained momenta, then the exact water projection restores the
    /// rigid manifold without a frame-rotation heuristic.
    fn nvt_step(&mut self) -> Result<()> {
        if self.state.protocol.langevin_discretization == LangevinDiscretization::LfMiddle {
            return self.nvt_lf_middle_step();
        }
        let dt = self.state.protocol.timestep_fs * 0.001;
        let decay = (-self.state.protocol.friction_per_ps * dt).exp();
        let prev_coords = self.state.coordinates.clone();
        let mut next_coords = prev_coords.clone();
        let mut next_vel = self.state.velocities.clone();
        let mut rng = self.state.rng_state;
        let mut resident_rng = self.state.resident_rng.clone();
        // B + A halves.
        for (i, m) in self.masses.iter().enumerate() {
            next_vel[i] = add(
                next_vel[i],
                scale(self.state.gradient[i], -0.5 * dt * ACCEL / m),
            );
            next_coords[i] = add(next_coords[i], scale(next_vel[i], 0.5 * dt));
        }
        if let Some(settle) = &self.settle {
            settle.settle_positions(
                &prev_coords,
                &next_coords.clone(),
                &mut next_coords,
                &self.masses,
            )?;
            settle.shake_solute_positions(&mut next_coords)?;
            settle.velocity_from_displacement(
                &prev_coords,
                &next_coords,
                &mut next_vel,
                2. / dt,
            )?;
        }
        let midpoint_coords = next_coords.clone();
        // O thermostat plus velocity constraints.
        if self.state.protocol.thermostat == Thermostat::VRescale {
            // Bussi-Parrinello rescaling on the half-step velocities: exact
            // canonical sampling that holds target temperature with
            // constraints, where pure OU equilibrates below target.
            let kinetic = kinetic_energy(&self.masses, &next_vel);
            let tau = 1. / self.state.protocol.friction_per_ps.max(1e-6);
            let alpha = vrescale_factor(
                &mut rng,
                kinetic,
                self.state.protocol.temperature_k,
                self.degrees_of_freedom,
                dt,
                tau,
            );
            for i in 0..next_vel.len() {
                next_vel[i] = scale(next_vel[i], alpha);
                next_coords[i] = add(next_coords[i], scale(next_vel[i], 0.5 * dt));
            }
        } else {
            for (i, m) in self.masses.iter().enumerate() {
                let sigma = ((1. - decay * decay) * KB * self.state.protocol.temperature_k * ACCEL
                    / m)
                    .sqrt();
                let noise = if let Some(stream) = &mut resident_rng {
                    scale(stream.normal3(i), sigma)
                } else {
                    random_velocity(&mut rng, sigma)
                };
                next_vel[i] = add(scale(next_vel[i], decay), noise);
                next_coords[i] = add(next_coords[i], scale(next_vel[i], 0.5 * dt));
            }
        }
        if let Some(settle) = &self.settle {
            settle.settle_positions(
                &midpoint_coords,
                &next_coords.clone(),
                &mut next_coords,
                &self.masses,
            )?;
            settle.shake_solute_positions(&mut next_coords)?;
            settle.velocity_from_displacement(
                &midpoint_coords,
                &next_coords,
                &mut next_vel,
                2. / dt,
            )?;
        }
        let energy = self.dynamics_energy(&next_coords)?;
        if !energy.components.total().is_finite() {
            return Err(invalid("nonfinite explicit energy; checkpoint retained"));
        }
        for (i, m) in self.masses.iter().enumerate() {
            next_vel[i] = add(
                next_vel[i],
                scale(energy.gradients[i], -0.5 * dt * ACCEL / m),
            );
            let displacement = add(next_coords[i], scale(self.state.coordinates[i], -1.));
            if !finite(&next_coords[i]) || !finite(&next_vel[i]) || norm2(displacement) > 1. {
                return Err(invalid(
                    "unstable explicit step (>1 A); checkpoint retained",
                ));
            }
        }
        if let Some(settle) = &self.settle {
            settle.constrain_velocities(&next_coords, &mut next_vel)?;
        }
        self.state.rng_state = rng;
        self.state.resident_rng = resident_rng;
        self.commit(
            next_coords,
            next_vel,
            energy.components.total(),
            energy.gradients,
            energy.virial,
        )
    }

    /// OpenMM LangevinMiddle ordering for constrained NVT. Velocities are
    /// centered at the committed positions and receive no legacy half-step
    /// interpretation or final force kick.
    fn nvt_lf_middle_step(&mut self) -> Result<()> {
        let dt = self.state.protocol.timestep_fs * 0.001;
        let decay = (-self.state.protocol.friction_per_ps * dt).exp();
        let old_coords = self.state.coordinates.clone();
        let mut trial_coords = old_coords.clone();
        let mut next_vel = self.state.velocities.clone();
        let mut rng = self.state.rng_state;
        let mut resident_rng = self.state.resident_rng.clone();

        // Full kick, followed by the velocity projection at the old positions.
        // Per-atom updates are independent; the parallel loops perform the
        // same arithmetic as the serial reference in any schedule.
        next_vel
            .par_iter_mut()
            .zip(self.state.gradient.par_iter())
            .zip(self.masses.par_iter())
            .for_each(|((velocity, gradient), mass)| {
                *velocity = add(*velocity, scale(*gradient, -dt * ACCEL / mass));
            });
        if let Some(constraints) = &self.settle {
            constraints.constrain_velocities(&old_coords, &mut next_vel)?;
        }

        // First half drift, OU thermostat, second half drift.
        let temperature = self.state.protocol.temperature_k;
        if let Some(stream) = &mut resident_rng {
            trial_coords
                .par_iter_mut()
                .zip(next_vel.par_iter_mut())
                .zip(self.masses.par_iter())
                .zip(stream.words.par_iter_mut())
                .for_each(|(((position, velocity), mass), word)| {
                    *position = add(*position, scale(*velocity, 0.5 * dt));
                    let sigma = ((1. - decay * decay) * KB * temperature * ACCEL / mass).sqrt();
                    let noise = scale(super::resident_rng::normal3_word(word), sigma);
                    *velocity = add(scale(*velocity, decay), noise);
                    *position = add(*position, scale(*velocity, 0.5 * dt));
                });
        } else {
            for i in 0..trial_coords.len() {
                trial_coords[i] = add(trial_coords[i], scale(next_vel[i], 0.5 * dt));
            }
            for (i, mass) in self.masses.iter().enumerate() {
                let sigma = ((1. - decay * decay) * KB * temperature * ACCEL / mass).sqrt();
                let noise = random_velocity(&mut rng, sigma);
                next_vel[i] = add(scale(next_vel[i], decay), noise);
                trial_coords[i] = add(trial_coords[i], scale(next_vel[i], 0.5 * dt));
            }
        }

        let mut next_coords = trial_coords.clone();
        if let Some(constraints) = &self.settle {
            constraints.settle_positions(
                &old_coords,
                &trial_coords,
                &mut next_coords,
                &self.masses,
            )?;
            constraints.shake_solute_positions_from_old(&old_coords, &mut next_coords)?;
            // Add the constraint impulse to the trial velocity. This is the
            // LF-middle centered-velocity convention (not BAOAB RATTLE).
            next_vel
                .par_iter_mut()
                .zip(next_coords.par_iter())
                .zip(trial_coords.par_iter())
                .for_each(|((velocity, next), trial)| {
                    *velocity = add(*velocity, scale(add(*next, scale(*trial, -1.)), 1. / dt));
                });
        }

        let energy = self.dynamics_energy(&next_coords)?;
        if !energy.components.total().is_finite() {
            return Err(invalid("nonfinite explicit energy; checkpoint retained"));
        }
        let unstable = next_coords
            .par_iter()
            .zip(next_vel.par_iter())
            .zip(old_coords.par_iter())
            .any(|((next, velocity), old)| {
                !finite(next) || !finite(velocity) || norm2(add(*next, scale(*old, -1.))) > 1.
            });
        if unstable {
            return Err(invalid(
                "unstable LF-middle step (>1 A); checkpoint retained",
            ));
        }
        self.state.rng_state = rng;
        self.state.resident_rng = resident_rng;
        self.commit(
            next_coords,
            next_vel,
            energy.components.total(),
            energy.gradients,
            energy.virial,
        )
    }

    /// Leap-frog with Nose–Hoover and Parrinello–Rahman coupling; the
    /// equations are in [`super::coupling`]. The stored velocities belong to
    /// the half step before the stored positions. One step:
    ///
    /// 1. on a coupling step, advance the thermostat frictions from the
    ///    group temperatures and the box velocities from the last pressure;
    /// 2. advance velocities and positions, constrain the positions and add
    ///    the constraint displacement to the velocities;
    /// 3. when the virial of the stored forces is known, compute the
    ///    pressure of this step from it, the constraint forces and the mean
    ///    of the two half-step kinetic energies;
    /// 4. on a barostat step, scale the box and the coordinates;
    /// 5. evaluate the forces at the new positions.
    fn leapfrog_step(&mut self) -> Result<()> {
        // The buffers leave `self` for the step, so the force evaluation can
        // borrow the simulation while they hold the new coordinates.
        let mut scratch = std::mem::take(&mut self.scratch);
        let result = self.leapfrog_step_in(&mut scratch);
        self.scratch = scratch;
        result
    }

    fn leapfrog_step_in(&mut self, scratch: &mut LeapfrogScratch) -> Result<()> {
        let Some(plan) = self.coupling_plan.clone() else {
            return Err(invalid("leap-frog step without a coupling plan"));
        };
        let dt = self.state.protocol.timestep_fs * 0.001;
        let step = self.state.step;
        let ensemble = self.segment_ensemble();
        let old_box = self.box_vectors()?;
        let lengths = old_box.as_array();
        let mut coupling = self
            .state
            .coupling
            .clone()
            .unwrap_or_else(|| CouplingState::new(plan.groups.len()));

        let mut friction = vec![0.; plan.groups.len()];
        if ensemble != Ensemble::Nve && CouplingPlan::acts_on(step, plan.temperature_interval) {
            let dt_t = plan.temperature_interval as f64 * dt;
            let mut kinetic = vec![0.; plan.groups.len()];
            for ((mass, velocity), group) in self
                .masses
                .iter()
                .zip(&self.state.velocities)
                .zip(&plan.group_of_atom)
            {
                kinetic[*group as usize] += mass * norm2(*velocity) / (2. * ACCEL);
            }
            let temperatures: Vec<f64> = kinetic
                .iter()
                .zip(&plan.groups)
                .map(|(k, group)| 2. * k / (group.degrees_of_freedom * KB))
                .collect();
            plan.advance_thermostats(&mut coupling, &temperatures, dt_t);
            for (f, velocity) in friction.iter_mut().zip(&coupling.thermostat_velocity) {
                *f = 0.5 * dt_t * velocity;
            }
        }

        // `drag` is the box-velocity term of the velocity update over the
        // barostat time step; the same number is the relative box change.
        let mut drag = [0.; 3];
        if ensemble == Ensemble::Npt {
            if CouplingPlan::acts_on(step, plan.pressure_interval) {
                let dt_p = plan.pressure_interval as f64 * dt;
                // The first step of a run has no pressure yet.
                if step > 0 && coupling.pressure_step == Some(step - 1) {
                    plan.accelerate_box(
                        &mut coupling.box_velocity,
                        lengths,
                        coupling.pressure_bar,
                        dt_p,
                    );
                }
                for d in 0..3 {
                    drag[d] = dt_p * coupling.box_velocity[d] / lengths[d];
                }
            }
        } else {
            coupling.box_velocity = [0.; 3];
        }
        // GROMACS warns above 1% per coupling step (a system far from
        // equilibrium under this barostat); ten times that is a runaway box.
        if drag.iter().any(|d| !d.is_finite() || d.abs() > 0.1) {
            return Err(invalid(
                "the barostat changed the box by more than 10% in one coupling step; checkpoint retained",
            ));
        }

        let atoms = self.masses.len();
        let zero = Vec3 {
            x: 0.,
            y: 0.,
            z: 0.,
        };
        for buffer in [&mut scratch.velocities, &mut scratch.trial, &mut scratch.coordinates] {
            buffer.resize(atoms, zero);
        }
        let LeapfrogScratch {
            velocities: next_vel,
            trial,
            coordinates: next_coords,
        } = scratch;
        let old_coords = &self.state.coordinates;
        {
            let velocities = &self.state.velocities;
            let gradient = &self.state.gradient;
            let kicks = &self.kick_per_ps;
            let groups = &plan.group_of_atom;
            // per group: the factors of the old velocity and of the kick
            let factors: Vec<([f64; 3], f64)> = friction
                .iter()
                .map(|f| {
                    let inverse = 1. / (1. + f);
                    (
                        [
                            (1. - f - drag[0]) * inverse,
                            (1. - f - drag[1]) * inverse,
                            (1. - f - drag[2]) * inverse,
                        ],
                        dt * inverse,
                    )
                })
                .collect();
            let factors = &factors;
            let advance = |(i, (velocity, position)): (usize, (&mut Vec3, &mut Vec3))| {
                let (keep, kick) = factors[groups[i] as usize];
                let kick = kick * kicks[i];
                let (v, g) = (velocities[i], gradient[i]);
                *velocity = Vec3 {
                    x: v.x * keep[0] + kick * g.x,
                    y: v.y * keep[1] + kick * g.y,
                    z: v.z * keep[2] + kick * g.z,
                };
                *position = add(old_coords[i], scale(*velocity, dt));
            };
            if atoms >= PARALLEL_ATOMS {
                next_vel
                    .par_iter_mut()
                    .zip(trial.par_iter_mut())
                    .enumerate()
                    .with_min_len(PER_ATOM_TASK)
                    .for_each(advance);
            } else {
                next_vel.iter_mut().zip(trial.iter_mut()).enumerate().for_each(advance);
            }
        }

        next_coords.copy_from_slice(trial);
        let constrained = self.settle.is_some();
        if let Some(constraints) = &self.settle {
            constraints.settle_positions(old_coords, trial, next_coords, &self.masses)?;
            constraints.shake_solute_positions_from_old(old_coords, next_coords)?;
        }
        // One pass adds the constraint displacement to the velocities and
        // finds the largest move of the step.
        let inverse_dt = 1. / dt;
        let correct = |(((velocity, next), unconstrained), old): (
            ((&mut Vec3, &Vec3), &Vec3),
            &Vec3,
        )| {
            if constrained {
                *velocity = add(
                    *velocity,
                    scale(add(*next, scale(*unconstrained, -1.)), inverse_dt),
                );
            }
            if finite(next) && finite(velocity) {
                norm2(add(*next, scale(*old, -1.)))
            } else {
                f64::INFINITY
            }
        };
        let largest_move = if atoms >= PARALLEL_ATOMS {
            next_vel
                .par_iter_mut()
                .zip(next_coords.par_iter())
                .zip(trial.par_iter())
                .zip(old_coords.par_iter())
                .with_min_len(PER_ATOM_TASK)
                .map(correct)
                .reduce(|| 0., f64::max)
        } else {
            next_vel
                .iter_mut()
                .zip(next_coords.iter())
                .zip(trial.iter())
                .zip(old_coords.iter())
                .map(correct)
                .fold(0., f64::max)
        };
        if !(largest_move <= 1.) {
            return Err(invalid(
                "unstable leap-frog step (>1 A); checkpoint retained",
            ));
        }
        let mut constraint_virial = 0.;
        if constrained && self.virial_current {
            // sum of r . G with G = m (x_constrained - x_free) / dt^2, summed
            // in atom order so the result does not depend on the thread
            // schedule
            for (i, mass) in self.masses.iter().enumerate() {
                let moved = add(next_coords[i], scale(trial[i], -1.));
                let arm = add(
                    old_coords[i],
                    scale(old_coords[self.constraint_anchor[i] as usize], -1.),
                );
                constraint_virial += mass * (arm.x * moved.x + arm.y * moved.y + arm.z * moved.z);
            }
            constraint_virial /= dt * dt * ACCEL;
        }

        let mut pressure = None;
        if self.virial_current {
            let kinetic = 0.5
                * (kinetic_energy(&self.masses, &self.state.velocities)
                    + kinetic_energy(&self.masses, next_vel));
            let volume = old_box.volume();
            let mut value =
                (2. * kinetic + self.state.virial_kcal_mol + constraint_virial) / (3. * volume);
            if self.state.protocol.dispersion_correction {
                value += self.dispersion_pressure_coefficient / (volume * volume);
            }
            let bar = value * BAR_PER_KCAL_MOL_A3;
            coupling.pressure_bar = bar;
            coupling.pressure_step = Some(step);
            pressure = Some(bar);
        }

        let new_box = if drag == [0.; 3] {
            old_box
        } else {
            let scale_by = [1. + drag[0], 1. + drag[1], 1. + drag[2]];
            for p in next_coords.iter_mut() {
                p.x *= scale_by[0];
                p.y *= scale_by[1];
                p.z *= scale_by[2];
            }
            BoxVectors::new(
                lengths[0] * scale_by[0],
                lengths[1] * scale_by[1],
                lengths[2] * scale_by[2],
            )
            .map_err(Error::Energy)?
        };
        if cutoff_angstrom(&self.state.protocol)
            >= 0.5 * new_box.x.min(new_box.y).min(new_box.z)
        {
            return Err(invalid(
                "the box shrank below twice the cutoff; checkpoint retained",
            ));
        }

        if plan.com_interval > 0 && step.is_multiple_of(plan.com_interval) {
            remove_com_motion(&plan.com_groups, &self.masses, next_vel);
        }

        let virial_was_current = self.virial_current;
        let energy = match self.dynamics_energy_in(next_coords, &new_box) {
            Ok(energy)
                if energy.components.total().is_finite()
                    && energy.gradients.len() == atoms
                    && energy.gradients.iter().all(finite) =>
            {
                energy
            }
            Ok(_) => {
                self.virial_current = virial_was_current;
                return Err(invalid("nonfinite explicit energy; checkpoint retained"));
            }
            Err(error) => {
                self.virial_current = virial_was_current;
                return Err(error);
            }
        };
        // The new state takes the buffers and leaves the old ones for the
        // next step.
        std::mem::swap(&mut self.state.coordinates, next_coords);
        std::mem::swap(&mut self.state.velocities, next_vel);
        self.state.potential_energy = energy.components.total();
        self.state.gradient = energy.gradients;
        self.state.virial_kcal_mol = energy.virial;
        self.state.box_angstrom = new_box.as_array();
        self.state.coupling = Some(coupling);
        if let Some(bar) = pressure {
            self.state.pressure_bar = bar;
            self.state.pressure_estimator = "atomic-virial-with-constraints".into();
        }
        Ok(())
    }

    /// Isotropic Monte Carlo barostat. Volume moves use their own stream;
    /// restraint forces are external and excluded from the virial by design.
    fn try_barostat(&mut self, adaptation_enabled: bool) -> Result<()> {
        let box_vec = self.box_vectors()?;
        let volume = box_vec.volume();
        if self.state.barostat_volume_width <= 0. {
            self.state.barostat_volume_width = volume * 0.01;
        }
        // OpenMM uses a symmetric uniform volume proposal. Consume both
        // draws before validation so replays preserve the RNG schedule.
        let proposal_u = super::uniform(&mut self.state.barostat_rng);
        let accept_u = super::uniform(&mut self.state.barostat_rng);
        let delta_volume = (2. * proposal_u - 1.) * self.state.barostat_volume_width;
        let proposed_volume = volume + delta_volume;
        self.state.barostat_attempts = self.state.barostat_attempts.saturating_add(1);
        self.state.barostat_window_attempts = self.state.barostat_window_attempts.saturating_add(1);
        if proposed_volume <= 0. {
            return Err(invalid("barostat proposed a nonpositive volume"));
        }
        let linear = (proposed_volume / volume).powf(1. / 3.);
        let proposed_box = BoxVectors {
            x: box_vec.x * linear,
            y: box_vec.y * linear,
            z: box_vec.z * linear,
        };
        if cutoff_angstrom(&self.state.protocol)
            >= 0.5 * proposed_box.x.min(proposed_box.y).min(proposed_box.z)
        {
            return Err(invalid(
                "barostat proposal reaches the cutoff/box geometry boundary",
            ));
        }
        // Translate each molecule's centroid while leaving internal geometry
        // untouched. This is the convention used by OpenMM and preserves
        // rigid water constraints without a repair impulse.
        let proposed_coords = scale_molecules(&self.state.coordinates, &self.molecules, linear)?;
        let wrapped: Vec<Vec3> = proposed_coords
            .iter()
            .map(|p| proposed_box.wrap(*p))
            .collect();
        let pairs = PbcNeighborList::build(
            &wrapped,
            &proposed_box,
            cutoff_angstrom(&self.state.protocol),
            SKIN_ANGSTROM,
        )
        .map_err(Error::Energy)?;
        let energy = self.electrostatics.reference(
            &self.field,
            &proposed_coords,
            &proposed_box,
            &pairs.pairs,
            &self.state.protocol,
            self.dispersion_coefficient,
        )?;
        let delta_g = barostat_work(
            self.state.potential_energy,
            energy.components.total(),
            volume,
            proposed_volume,
            self.state.protocol.pressure_bar,
            self.state.protocol.temperature_k,
            self.molecules.len(),
        )?;
        let accept = barostat_accept(delta_g, self.state.protocol.temperature_k, accept_u.ln())?;
        if accept {
            self.state.coordinates = proposed_coords;
            self.state.box_angstrom = proposed_box.as_array();
            self.pairs = pairs;
            self.state.potential_energy = energy.components.total();
            self.state.gradient = energy.gradients;
            self.state.virial_kcal_mol = energy.virial;
            self.state.barostat_accepts = self.state.barostat_accepts.saturating_add(1);
            self.state.barostat_window_accepts =
                self.state.barostat_window_accepts.saturating_add(1);
        }
        if adaptation_enabled
            && !self.state.barostat_frozen
            && self.state.barostat_window_attempts >= 10
        {
            let attempts = self.state.barostat_window_attempts;
            let accepts = self.state.barostat_window_accepts;
            if accepts * 4 < attempts {
                self.state.barostat_volume_width /= 1.1;
            } else if accepts * 4 > attempts * 3 {
                self.state.barostat_volume_width =
                    (self.state.barostat_volume_width * 1.1).min(volume * 0.3);
            }
            self.state.barostat_window_attempts = 0;
            self.state.barostat_window_accepts = 0;
        } else if !adaptation_enabled {
            self.state.barostat_frozen = true;
        }
        Ok(())
    }

    /// Forces and observables for a dynamics step at `coords`: the cluster
    /// engine when enabled (NPT keeps the reference list, which its
    /// barostat proposals share), otherwise the f64 reference evaluator.
    fn dynamics_energy(&mut self, coords: &[Vec3]) -> Result<glysys_energy::pbc::PbcEnergy> {
        let box_vec = self.box_vectors()?;
        self.dynamics_energy_in(coords, &box_vec)
    }

    /// Energy and gradient of `coords` in `box_vec`, which a barostat step
    /// may have changed from the committed box.
    fn dynamics_energy_in(
        &mut self,
        coords: &[Vec3],
        box_vec: &BoxVectors,
    ) -> Result<glysys_energy::pbc::PbcEnergy> {
        // The Monte Carlo barostat evaluates trial boxes with the reference
        // pair list; the leap-frog barostat changes the box a little at a
        // time, which the cluster lists follow.
        if (!self.state.protocol.has_npt() || self.coupling_plan.is_some())
            && let Some(engine) = &mut self.cluster
        {
            let energy = match &mut self.electrostatics {
                Electrostatics::ReactionField(_) => self.field.evaluate_with_cluster(
                    engine,
                    coords,
                    box_vec,
                    self.dispersion_coefficient,
                    self.state.protocol.dispersion_correction,
                    !self.force_only,
                ),
                Electrostatics::Pme(pme) => self.field.evaluate_with_cluster_pme(
                    engine,
                    pme,
                    coords,
                    box_vec,
                    self.dispersion_coefficient,
                    self.state.protocol.dispersion_correction,
                    !self.force_only,
                ),
            }
            .map_err(Error::Energy)?;
            self.virial_current = !self.force_only;
            return Ok(energy);
        }
        self.refresh_pairs_in(coords, box_vec)?;
        let energy = self.electrostatics.reference(
            &self.field,
            coords,
            box_vec,
            &self.pairs.pairs,
            &self.state.protocol,
            self.dispersion_coefficient,
        )?;
        self.virial_current = true;
        Ok(energy)
    }

    fn refresh_pairs_in(&mut self, coords: &[Vec3], box_vec: &BoxVectors) -> Result<()> {
        let wrapped: Vec<Vec3> = coords.iter().map(|p| box_vec.wrap(*p)).collect();
        // Displacement-gated rebuild: forces never use stale pairs, and
        // quiescent systems skip the rebuild entirely.
        if self.pairs.box_changed(box_vec) || self.pairs.needs_rebuild(&wrapped) {
            self.pairs = PbcNeighborList::build(
                &wrapped,
                box_vec,
                cutoff_angstrom(&self.state.protocol),
                SKIN_ANGSTROM,
            )
            .map_err(Error::Energy)?;
        }
        Ok(())
    }

    fn commit(
        &mut self,
        coords: Vec<Vec3>,
        velocities: Vec<Vec3>,
        energy: f64,
        gradient: Vec<Vec3>,
        virial: f64,
    ) -> Result<()> {
        if !energy.is_finite()
            || gradient.len() != self.masses.len()
            || gradient.iter().any(|g| !finite(g))
        {
            return Err(invalid("nonfinite explicit energy or gradient"));
        }
        self.state.coordinates = coords;
        self.state.velocities = velocities;
        self.state.potential_energy = energy;
        self.state.gradient = gradient;
        self.state.virial_kcal_mol = virial;
        Ok(())
    }

    /// Check external geometry without evaluating forces or projecting the
    /// sampled frame. Production and opt-in reference checks share these
    /// existing f32 constraint residual bounds.
    fn validate_external_state(
        &self,
        coordinates: &[Vec3],
        velocities: &[Vec3],
        step: usize,
    ) -> Result<()> {
        let total_steps = self.state.protocol.total_steps();
        if coordinates.len() != self.masses.len()
            || velocities.len() != self.masses.len()
            || step > total_steps
            || coordinates.iter().any(|p| !finite(p))
            || velocities.iter().any(|v| !finite(v))
        {
            return Err(invalid("invalid external dynamics state"));
        }
        if let Some(settle) = &self.settle {
            let position_error = settle.max_violation(&coordinates);
            let velocity_error = settle.max_velocity_violation(&coordinates, &velocities);
            if !position_error.is_finite()
                || !velocity_error.is_finite()
                || position_error > 5e-4
                || (self.state.velocity_convention == VelocityConvention::LegacyFullStep
                    && velocity_error > 5e-3)
            {
                return Err(invalid(format!(
                    "external constrained state residuals {position_error:.3e} A/{velocity_error:.3e} A/ps ({:?} velocity convention)",
                    self.state.velocity_convention
                )));
            }
        }
        Ok(())
    }

    /// Install backend-evaluated forces and energy without running a CPU
    /// reference force calculation. Geometry checks never modify the frame.
    pub fn install_evaluated_state(
        &mut self,
        coordinates: Vec<Vec3>,
        velocities: Vec<Vec3>,
        step: usize,
        energy: f64,
        gradient: Vec<Vec3>,
        virial: f64,
    ) -> Result<()> {
        self.validate_external_state(&coordinates, &velocities, step)?;
        if !virial.is_finite() {
            return Err(invalid("nonfinite external virial"));
        }
        let volume = self.box_vectors()?.volume();
        self.commit(coordinates, velocities, energy, gradient, virial)?;
        self.state.step = step;
        self.state.pressure_bar = pressure_bar(
            self.masses.len(),
            self.state.protocol.temperature_k,
            virial,
            volume,
        );
        self.state.pressure_estimator = "atomic-virial".into();
        self.state.integrator_phase = "ready".into();
        Ok(())
    }

    pub fn replace_dynamics_state(
        &mut self,
        coordinates: Vec<Vec3>,
        velocities: Vec<Vec3>,
        step: usize,
    ) -> Result<()> {
        self.validate_external_state(&coordinates, &velocities, step)?;
        let box_vec = self.box_vectors()?;
        self.refresh_pairs_in(&coordinates, &box_vec)?;
        let energy = self.electrostatics.reference(
            &self.field,
            &coordinates,
            &box_vec,
            &self.pairs.pairs,
            &self.state.protocol,
            self.dispersion_coefficient,
        )?;
        self.state.coordinates = coordinates;
        self.state.velocities = velocities;
        self.state.step = step;
        self.state.potential_energy = energy.components.total();
        self.state.gradient = energy.gradients;
        self.state.virial_kcal_mol = energy.virial;
        self.state.pressure_bar = pressure_bar(
            self.masses.len(),
            self.state.protocol.temperature_k,
            energy.virial,
            box_vec.volume(),
        );
        self.state.pressure_estimator = "atomic-virial".into();
        self.state.integrator_phase = "ready".into();
        Ok(())
    }

    /// Install a leap-frog state advanced by an external evaluator (the GPU
    /// engine): positions, the velocities half a step behind them, the box
    /// and the coupling variables, with the energy, gradient and full virial
    /// of the new positions.
    #[allow(clippy::too_many_arguments)]
    pub fn install_leapfrog_state(
        &mut self,
        coordinates: Vec<Vec3>,
        velocities: Vec<Vec3>,
        step: usize,
        energy: f64,
        gradient: Vec<Vec3>,
        virial: f64,
        box_angstrom: [f64; 3],
        coupling: CouplingState,
    ) -> Result<()> {
        let Some(plan) = &self.coupling_plan else {
            return Err(invalid("leap-frog state for another integrator"));
        };
        if coupling.thermostat_velocity.len() != plan.groups.len()
            || coupling.thermostat_position.len() != plan.groups.len()
            || !virial.is_finite()
            || !coupling.pressure_bar.is_finite()
            || !box_angstrom.iter().all(|b| b.is_finite() && *b > 0.)
            || coupling
                .thermostat_velocity
                .iter()
                .chain(&coupling.thermostat_position)
                .chain(&coupling.box_velocity)
                .any(|value| !value.is_finite())
        {
            return Err(invalid("invalid external leap-frog state"));
        }
        self.validate_external_state(&coordinates, &velocities, step)?;
        self.commit(coordinates, velocities, energy, gradient, virial)?;
        self.state.box_angstrom = box_angstrom;
        self.state.step = step;
        if coupling.pressure_step.is_some() {
            self.state.pressure_bar = coupling.pressure_bar;
            self.state.pressure_estimator = "atomic-virial-with-constraints".into();
        }
        self.state.coupling = Some(coupling);
        self.state.integrator_phase = "ready".into();
        self.virial_current = true;
        Ok(())
    }

    /// Homogeneous long-range Lennard-Jones energy of the current box,
    /// kcal/mol; zero when the protocol leaves the correction out.
    pub fn dispersion_energy(&self) -> f64 {
        if self.state.protocol.dispersion_correction {
            self.dispersion_coefficient / self.state.box_angstrom.iter().product::<f64>()
        } else {
            0.
        }
    }

    /// Temperature groups and coupling intervals of the leap-frog integrator;
    /// `None` when the protocol uses another integrator.
    pub fn coupling_plan(&self) -> Option<&CouplingPlan> {
        self.coupling_plan.as_deref()
    }

    /// Coefficient of the long-range dispersion term in a pressure computed
    /// from forces, kcal Å³/mol; zero when the protocol leaves the correction
    /// out.
    pub fn dispersion_pressure_coefficient(&self) -> f64 {
        if self.state.protocol.dispersion_correction {
            self.dispersion_pressure_coefficient
        } else {
            0.
        }
    }

    pub fn masses(&self) -> &[f64] {
        &self.masses
    }

    pub fn advance(&mut self, steps: usize) -> Result<TrajectoryChunk> {
        let first_step = self.state.step;
        let end = (self.state.step + steps.min(100)).min(self.state.protocol.total_steps());
        let mut frames = Vec::new();
        while self.state.step < end {
            let next = self.state.step + 1;
            // Only steps whose observables are reported need energies; the
            // integrator itself consumes forces alone.
            self.force_only = next != end
                && !next.is_multiple_of(self.state.protocol.save_every)
                && !self.state.protocol.is_stage_boundary(next)
                && next != self.state.protocol.total_steps()
                && !self
                    .coupling_plan
                    .as_ref()
                    .is_some_and(|plan| plan.needs_pressure(next));
            let result = self.step();
            self.force_only = false;
            result?;
            if self
                .state
                .step
                .is_multiple_of(self.state.protocol.save_every)
                || self.state.protocol.is_stage_boundary(self.state.step)
                || self.state.step == self.state.protocol.total_steps()
            {
                self.observe();
                frames.push(self.frame());
            }
        }
        Ok(TrajectoryChunk {
            first_step,
            last_step: self.state.step,
            frames,
        })
    }

    /// Feed saved production frames to the occupancy accumulator.
    fn observe(&mut self) {
        let (stage_index, _, _, _, _) = self.state.protocol.stage_info(self.state.step);
        if stage_index + 1 < self.state.protocol.execution_stages().len() {
            return;
        }
        let box_vec = match self.box_vectors() {
            Ok(b) => b,
            Err(_) => return,
        };
        let oxygen: Vec<Vec3> = self
            .waters
            .iter()
            .filter_map(|w| self.state.coordinates.get(w[0]).map(|p| box_vec.wrap(*p)))
            .collect();
        let entry = self
            .state
            .water_occupancy
            .get_or_insert_with(|| WaterOccupancy::new(box_vec.as_array(), 1.0));
        entry.observe(&oxygen, box_vec.as_array());
    }

    pub fn frame(&self) -> super::TrajectoryFrame {
        let mut frame = frame_from_state_wrapped(&self.state, &self.masses, &self.molecules);
        frame.temperature_k = kinetic_temperature(
            kinetic_energy(&self.masses, &self.state.velocities),
            self.degrees_of_freedom,
        );
        let total_amu: f64 = self.masses.iter().sum();
        let volume =
            self.state.box_angstrom[0] * self.state.box_angstrom[1] * self.state.box_angstrom[2];
        frame.density_g_ml = density_g_ml(total_amu, volume);
        frame.segment = self.state.protocol.stage_info(self.state.step).1;
        frame
    }

    /// Live neighbor pairs (validation diagnostics).
    pub fn live_pairs(&self) -> &[(usize, usize)] {
        &self.pairs.pairs
    }

    /// Number of pairs in the live neighbor list (diagnostics/progress).
    pub fn pair_count(&self) -> usize {
        self.pairs.pairs.len()
    }

    /// Worst constraint residual in angstrom, if constraints are active.
    pub fn constraint_violation(&self) -> Option<f64> {
        self.settle
            .as_ref()
            .map(|s| s.max_violation(&self.state.coordinates))
    }

    /// Worst velocity-constraint residual in A/ps, if constraints are active.
    pub fn velocity_constraint_violation(&self) -> Option<f64> {
        self.settle
            .as_ref()
            .map(|s| s.max_velocity_violation(&self.state.coordinates, &self.state.velocities))
    }

    pub fn wrapped_coordinates(&self) -> Result<Vec<Vec3>> {
        let box_vec = self.box_vectors()?;
        Ok(self
            .state
            .coordinates
            .iter()
            .map(|p| box_vec.wrap(*p))
            .collect())
    }

    pub fn whole_molecule_coordinates(&self) -> Result<Vec<Vec3>> {
        let box_vec = self.box_vectors()?;
        Ok(box_vec.wrap_molecules(&self.state.coordinates, &self.molecules))
    }
}

fn protocol_seed(protocol: &SimulationProtocol) -> u64 {
    protocol.seed
}

fn protocol_seed_barostat(seed: u64) -> u64 {
    seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678_9ABC_DEF0)
        | 1
}

/// Consume one value from the barostat stream using the same integer
/// generator as the CPU sampler.  GPU orchestration uses this adapter when a
/// device cannot yet encode a complete trial transaction; exposing the
/// primitive keeps the proposal/acceptance stream identical across backends.
pub fn next_barostat_uniform(state: &mut u64) -> f64 {
    super::uniform(state)
}

fn minimize<F>(
    field: &PbcForceField,
    box_vec: &BoxVectors,
    electrostatics: &mut Electrostatics,
    initial: Vec<Vec3>,
    protocol: &SimulationProtocol,
    dispersion_coefficient: f64,
    progress: &mut F,
) -> Result<Vec<Vec3>>
where
    F: FnMut(&str, usize, usize, f64, f64),
{
    // Steepest-descent pre-relaxation first: solvated boxes start with
    // clashing contacts whose huge forces stall quasi-Newton line searches.
    let coords = initial;
    // The objective uses the physical cutoff, not the neighbor-list radius.
    // Rebuilding before a trial exceeds the skin preserves that objective and
    // avoids box-sized lists (and their memory cost) during line searches.
    let cutoff = cutoff_angstrom(protocol);
    let wrapped: Vec<Vec3> = coords.iter().map(|p| box_vec.wrap(*p)).collect();
    let min_skin = SKIN_ANGSTROM;
    let mut pairs =
        PbcNeighborList::build(&wrapped, box_vec, cutoff, min_skin).map_err(Error::Energy)?;
    let flat: Vec<_> = coords.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let mut optimizer = crate::minimization::PreparationMinimizer::new(
        &flat,
        protocol.minimization_iterations,
        true,
    )?;
    while let Some(point) = optimizer.request() {
        let coords: Vec<_> = point
            .chunks_exact(3)
            .map(|p| Vec3 {
                x: p[0],
                y: p[1],
                z: p[2],
            })
            .collect();
        let wrapped: Vec<Vec3> = coords.iter().map(|p| box_vec.wrap(*p)).collect();
        if pairs.needs_rebuild(&wrapped) {
            pairs = PbcNeighborList::build(&wrapped, box_vec, cutoff, min_skin)
                .map_err(Error::Energy)?;
        }
        let energy = electrostatics.reference(
            field,
            &coords,
            box_vec,
            &pairs.pairs,
            protocol,
            dispersion_coefficient,
        )?;
        let submitted = optimizer.submit(
            energy.components.total(),
            energy
                .gradients
                .iter()
                .flat_map(|p| [p.x, p.y, p.z])
                .collect(),
        )?;
        for event in submitted {
            progress(
                event.stage,
                event.completed,
                event.total,
                event.energy,
                event.max_gradient,
            );
        }
    }
    let outcome = optimizer
        .outcome()
        .ok_or_else(|| invalid("L-BFGS stopped without a completed optimization state"))?;
    Ok(outcome
        .chunks_exact(3)
        .map(|p| Vec3 {
            x: p[0],
            y: p[1],
            z: p[2],
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glysys::{BuildOptions, SystemBuilder};

    fn solvated_dipeptide() -> ParameterizedSystem {
        solvated_dipeptide_padded(6.0)
    }

    fn solvated_dipeptide_padded(padding: f64) -> ParameterizedSystem {
        let pdb = include_str!("../../../tests/fixtures/dipeptide.pdb");
        let options = BuildOptions {
            add_water: true,
            add_ions: false,
            padding_angstrom: padding,
            ..Default::default()
        };
        SystemBuilder::new(options)
            .unwrap()
            .prepare_pdb_str(pdb)
            .unwrap()
    }

    fn nve_protocol() -> SimulationProtocol {
        SimulationProtocol {
            solvent: SolventModel::Explicit,
            equilibration_ensemble: Ensemble::Nve,
            production_ensemble: Ensemble::Nve,
            equilibration_steps: 0,
            production_steps: 200,
            save_every: 50,
            minimization_iterations: 50,
            timestep_fs: 1.0,
            constraints: ConstraintModel::None,
            // Tiny test box: production cutoffs need bigger boxes.
            cutoff_angstrom: Some(4.0),
            seed: 7,
            ..Default::default()
        }
    }

    #[test]
    fn pressure_conversion_matches_independent_dimensions() {
        // bar = (kcal/mol / N_A) * 1e30 Å^3/m^3 * 1e5 Pa/bar / 1e5 Pa/bar.
        // The factor is bar per (kcal/mol/Å^3), not the inverse and not 10x
        // smaller. Use exact independent constants rather than KB.
        let derived = 4184.0 * 1e25 / 6.02214076e23;
        assert!(
            (BAR_PER_KCAL_MOL_A3 - derived).abs() < 1.,
            "{} vs {derived}",
            BAR_PER_KCAL_MOL_A3
        );

        // Ideal-gas check pV = N kB T in a 1000 Å^3 cell.
        let pressure = pressure_bar(100, 300., 0., 1000.);
        let expected = 100. * KB * 300. / 1000. * derived;
        assert!((pressure - expected).abs() < 1e-8);

        // The constrained molecular form gives the same ideal-gas pressure
        // when one molecule carries 3/2 kT of COM kinetic energy.
        let molecular =
            molecular_pressure_bar_from_pair_virial(0., 1.5 * KB * 300., 1000., 0., false);
        assert!((molecular - KB * 300. / 1000. * derived).abs() < 1e-8);

        // U_disp = C_disp/V contributes +C*C_disp/V² to pressure.
        let dispersion = molecular_pressure_bar_from_pair_virial(0., 0., 1000., 2., true);
        assert!((dispersion - 2. / 1_000_000. * derived).abs() < 1e-8);
    }

    #[test]
    fn molecule_scaling_preserves_internal_geometry() {
        let coordinates = vec![
            Vec3 {
                x: 1.,
                y: 2.,
                z: 3.,
            },
            Vec3 {
                x: 2.,
                y: 2.,
                z: 3.,
            },
            Vec3 {
                x: 9.,
                y: 4.,
                z: 1.,
            },
        ];
        let scaled = scale_molecules(&coordinates, &[vec![0, 1], vec![2]], 1.2).unwrap();
        let diff = |a: Vec3, b: Vec3| Vec3 {
            x: a.x - b.x,
            y: a.y - b.y,
            z: a.z - b.z,
        };
        let d0 = norm2(diff(scaled[1], scaled[0]));
        let d1 = norm2(diff(coordinates[1], coordinates[0]));
        assert!((d0 - d1).abs() < 1e-12);
        assert!((scaled[2].x - 10.8).abs() < 1e-12);
        assert!((scaled[2].y - 4.8).abs() < 1e-12);
        assert!((scaled[2].z - 1.2).abs() < 1e-12);
        let center_x = 1.5;
        assert!((scaled[0].x - (coordinates[0].x + (1.2 - 1.) * center_x)).abs() < 1e-12);
    }

    #[test]
    fn barostat_work_uses_molecule_count_and_bar_units() {
        let work = barostat_work(0., 0., 1000., 1100., 1., 300., 2).unwrap();
        let expected = 1. * 100. / BAR_PER_KCAL_MOL_A3 - 2. * KB * 300. * (1.1f64).ln();
        assert!((work - expected).abs() < 1e-12);
        let atom_count_work = 1. * 100. / BAR_PER_KCAL_MOL_A3 - 20. * KB * 300. * (1.1f64).ln();
        assert!((work - atom_count_work).abs() > 1e-3);
        assert!(barostat_accept(-1., 300., -1e-12).unwrap());
        assert!(!barostat_accept(100., 300., 0.).unwrap());
    }

    #[test]
    fn explicit_rejects_unphysical_setup() {
        let system = solvated_dipeptide();
        // Implicit solvent flag routes elsewhere.
        let mut bad = nve_protocol();
        bad.solvent = SolventModel::Implicit;
        assert!(ExplicitSimulation::new(&system, bad).is_err());
        // 2 fs flexible water is rejected, not silently integrated.
        let mut bad = nve_protocol();
        bad.timestep_fs = 2.0;
        assert!(ExplicitSimulation::new(&system, bad).is_err());
    }

    #[test]
    fn nve_drift_stays_bounded_without_thermostat() {
        // Production-like 9 A cutoff needs a box with room to spare.
        let system = solvated_dipeptide_padded(9.0);
        let mut protocol = nve_protocol();
        protocol.cutoff_angstrom = Some(9.0);
        protocol.production_steps = 100;
        let mut sim = ExplicitSimulation::new(&system, protocol).unwrap();
        let kinetic = |sim: &ExplicitSimulation| {
            sim.masses
                .iter()
                .zip(&sim.state.velocities)
                .map(|(m, v)| m * norm2(*v) / (2. * ACCEL))
                .sum::<f64>()
        };
        let initial_total = sim.state.potential_energy + kinetic(&sim);
        for _ in 0..100 {
            sim.step().unwrap();
        }
        let total = sim.state.potential_energy + kinetic(&sim);
        let drift = ((total - initial_total) / sim.masses.len() as f64).abs();
        assert!(drift < 0.02, "per-atom drift {drift}");
    }

    #[test]
    fn settle_nve_drift_stays_bounded() {
        // Constrained 2 fs steps must remain much closer to energy
        // conservation than the earlier rotation heuristic. The external
        // OpenMM reference remains the scientific parity gate.
        let system = solvated_dipeptide_padded(9.0);
        let mut protocol = nve_protocol();
        protocol.cutoff_angstrom = Some(9.0);
        protocol.production_steps = 100;
        protocol.timestep_fs = 2.0;
        protocol.constraints = ConstraintModel::Settle;
        let mut sim = ExplicitSimulation::new(&system, protocol).unwrap();
        let kinetic = |sim: &ExplicitSimulation| {
            sim.masses
                .iter()
                .zip(&sim.state.velocities)
                .map(|(m, v)| m * norm2(*v) / (2. * ACCEL))
                .sum::<f64>()
        };
        let initial_total = sim.state.potential_energy + kinetic(&sim);
        for _ in 0..100 {
            sim.step().unwrap();
            assert!(
                sim.settle
                    .as_ref()
                    .unwrap()
                    .max_violation(&sim.state.coordinates)
                    < 1e-8,
                "SETTLE position residual exceeded 1e-8 A"
            );
            assert!(
                sim.settle
                    .as_ref()
                    .unwrap()
                    .max_velocity_violation(&sim.state.coordinates, &sim.state.velocities)
                    < 1e-6,
                "RATTLE velocity residual exceeded 1e-6 A/ps"
            );
        }
        let total = sim.state.potential_energy + kinetic(&sim);
        let drift = ((total - initial_total) / sim.masses.len() as f64).abs();
        assert!(drift < 0.005, "per-atom drift {drift}");
    }

    #[test]
    fn settle_nvt_holds_target_temperature() {
        // The missing coverage that allowed the 125 K stall: constrained NVT
        // must hold the target temperature, not freeze rotation out (RATTLE
        // deletion) or heat out of control (radial ratchet).
        let system = solvated_dipeptide();
        let mut protocol = nve_protocol();
        protocol.equilibration_ensemble = Ensemble::Nvt;
        protocol.production_ensemble = Ensemble::Nvt;
        protocol.timestep_fs = 2.0;
        protocol.constraints = ConstraintModel::Settle;
        protocol.friction_per_ps = 1.0;
        protocol.production_steps = 400;
        protocol.save_every = 2;
        let mut sim = ExplicitSimulation::new(&system, protocol).unwrap();
        let mut temps = Vec::new();
        for _ in 0..400 {
            sim.step().unwrap();
            if sim.state.step % 2 == 0 {
                temps.push(sim.frame().temperature_k);
            }
        }
        let tail = &temps[temps.len() / 2..];
        let mean = tail.iter().sum::<f64>() / tail.len() as f64;
        assert!(
            (mean - 300.).abs() < 80.,
            "NVT mean temperature {mean} K over last {} steps",
            tail.len() * 2
        );
    }

    #[test]
    fn settle_holds_water_geometry_in_nvt() {
        let system = solvated_dipeptide();
        let mut protocol = nve_protocol();
        protocol.equilibration_ensemble = Ensemble::Nvt;
        protocol.production_ensemble = Ensemble::Nvt;
        protocol.timestep_fs = 2.0;
        protocol.constraints = ConstraintModel::Settle;
        protocol.friction_per_ps = 1.0;
        let mut sim = ExplicitSimulation::new(&system, protocol).unwrap();
        for _ in 0..100 {
            sim.step().unwrap();
        }
        let settle = sim.settle.as_ref().unwrap();
        assert!(settle.max_violation(&sim.state.coordinates) < 1e-8);
        assert!(sim.state.pressure_bar.is_finite());
        assert!(sim.frame().density_g_ml > 0.5 && sim.frame().density_g_ml < 1.5);
    }

    #[test]
    fn occupancy_accumulates_over_production() {
        let system = solvated_dipeptide();
        let mut protocol = nve_protocol();
        protocol.production_steps = 100;
        protocol.save_every = 10;
        let mut sim = ExplicitSimulation::new(&system, protocol).unwrap();
        let chunk = sim.advance(100).unwrap();
        assert!(!chunk.frames.is_empty());
        let occ = sim.state.water_occupancy.as_ref().unwrap();
        assert_eq!(occ.frames_observed(), chunk.frames.len() as u64);
        assert!(occ.counts.iter().sum::<u32>() > 0);
    }

    #[test]
    fn npt_barostat_keeps_density_sane() {
        let system = solvated_dipeptide();
        let mut protocol = nve_protocol();
        protocol.equilibration_ensemble = Ensemble::Nvt;
        protocol.production_ensemble = Ensemble::Npt;
        protocol.timestep_fs = 2.0;
        protocol.constraints = ConstraintModel::Settle;
        protocol.friction_per_ps = 1.0;
        protocol.pressure_bar = 1.0;
        protocol.barostat_interval = 5;
        protocol.production_steps = 100;
        let initial_volume = protocol_seed_volume(&system);
        let mut sim = ExplicitSimulation::new(&system, protocol).unwrap();
        for _ in 0..100 {
            sim.step().unwrap();
        }
        let volume: f64 = sim.state.box_angstrom.iter().product();
        assert!(volume.is_finite() && volume > 0.);
        // Isotropic moves scale the box; tight acceptance is not required,
        // but the box must stay physical and the density water-like.
        assert!((volume / initial_volume - 1.).abs() < 0.25);
        let density = sim.frame().density_g_ml;
        assert!(density > 0.5 && density < 1.5, "density {density}");
        assert!(sim.state.pressure_bar.is_finite());
        assert_eq!(sim.state.barostat_step_counter, 100);
        assert_eq!(sim.state.barostat_attempts, 20);
        assert!(sim.state.barostat_accepts <= sim.state.barostat_attempts);
        assert!(
            sim.settle
                .as_ref()
                .unwrap()
                .max_violation(&sim.state.coordinates)
                < 1e-8
        );
    }

    fn protocol_seed_volume(system: &ParameterizedSystem) -> f64 {
        let b = system.box_angstrom();
        b[0] * b[1] * b[2]
    }

    #[test]
    fn checkpoint_round_trip_with_box() {
        let system = solvated_dipeptide();
        let mut sim = ExplicitSimulation::new(&system, nve_protocol()).unwrap();
        for _ in 0..10 {
            sim.step().unwrap();
        }
        let json = serde_json::to_string(&sim.state).unwrap();
        let restored: SimulationState = serde_json::from_str(&json).unwrap();
        let again = ExplicitSimulation::restore(&system, restored).unwrap();
        assert_eq!(again.state.coordinates, sim.state.coordinates);
        assert_eq!(again.state.box_angstrom, sim.state.box_angstrom);
    }

    #[test]
    fn lf_middle_checkpoint_identity_allows_benchmark_schedule_extension_only() {
        let system = solvated_dipeptide();
        let protocol = SimulationProtocol {
            solvent: SolventModel::Explicit,
            constraints: ConstraintModel::Settle,
            thermostat: Thermostat::Langevin,
            langevin_discretization: LangevinDiscretization::LfMiddle,
            timestep_fs: 2.0,
            temperature_k: 300.0,
            friction_per_ps: 1.0,
            cutoff_angstrom: Some(9.0),
            rf_dielectric: Some(78.5),
            equilibration_steps: 4000,
            production_steps: 150_000,
            save_every: 2500,
            ..Default::default()
        };
        let original = fingerprint(&system, &protocol, 0);
        let mut extended = protocol.clone();
        extended.equilibration_steps = 0;
        extended.production_steps = super::super::MAX_NATIVE_STEPS;
        extended.save_every = usize::MAX;
        assert_eq!(original, fingerprint(&system, &extended, 0));
        extended.temperature_k += 1.0;
        assert_ne!(original, fingerprint(&system, &extended, 0));
    }

    fn leapfrog_protocol(steps: usize) -> SimulationProtocol {
        SimulationProtocol {
            solvent: SolventModel::Explicit,
            equilibration_ensemble: Ensemble::Nvt,
            production_ensemble: Ensemble::Nvt,
            equilibration_steps: 0,
            production_steps: steps,
            save_every: 50,
            minimization_iterations: 50,
            timestep_fs: 2.0,
            constraints: ConstraintModel::Settle,
            thermostat: Thermostat::NoseHoover,
            thermostat_groups: ["WAT", "System_&_!WAT"]
                .into_iter()
                .map(|name| crate::ThermostatGroup {
                    name: name.into(),
                    tau_ps: 0.2,
                    reference_temperature_k: 300.0,
                })
                .collect(),
            com_mode: Some("linear".into()),
            com_groups: vec!["WAT".into(), "System_&_!WAT".into()],
            cutoff_angstrom: Some(4.0),
            seed: 7,
            ..Default::default()
        }
    }

    fn leapfrog_npt_protocol(steps: usize) -> SimulationProtocol {
        SimulationProtocol {
            production_ensemble: Ensemble::Npt,
            pressure_coupling: PressureCoupling::ParrinelloRahman,
            pressure_bar: 1.0,
            pressure_tau_ps: Some(2.0),
            pressure_compressibility_bar_inverse: vec![4.5e-5],
            dispersion_correction: true,
            ..leapfrog_protocol(steps)
        }
    }

    #[test]
    fn coupling_choices_that_do_not_belong_together_are_rejected() {
        let system = solvated_dipeptide();
        let mut monte_carlo = leapfrog_npt_protocol(10);
        monte_carlo.pressure_coupling = PressureCoupling::MonteCarlo;
        assert!(ExplicitSimulation::new(&system, monte_carlo).is_err());
        let mut langevin = leapfrog_npt_protocol(10);
        langevin.thermostat = Thermostat::Langevin;
        assert!(ExplicitSimulation::new(&system, langevin).is_err());
        let mut unknown_group = leapfrog_protocol(10);
        unknown_group.thermostat_groups[1].name = "Protein".into();
        assert!(ExplicitSimulation::new(&system, unknown_group).is_err());
    }

    #[test]
    fn leapfrog_nose_hoover_holds_the_target_temperature() {
        let system = solvated_dipeptide();
        // 601 steps: the last one removes center-of-mass motion.
        let mut sim = ExplicitSimulation::new(&system, leapfrog_protocol(601)).unwrap();
        assert_eq!(sim.state.velocity_convention, VelocityConvention::LeapfrogHalfStep);
        let mut temperatures = Vec::new();
        for _ in 0..601 {
            sim.step().unwrap();
            temperatures.push(sim.frame().temperature_k);
        }
        let tail = &temperatures[temperatures.len() / 2..];
        let mean = tail.iter().sum::<f64>() / tail.len() as f64;
        assert!((mean - 300.).abs() < 30., "mean temperature {mean} K");
        let coupling = sim.state.coupling.as_ref().unwrap();
        assert!(coupling.thermostat_velocity.iter().all(|v| v.is_finite() && v.abs() < 50.));
        assert!(
            sim.settle
                .as_ref()
                .unwrap()
                .max_violation(&sim.state.coordinates)
                < 1e-4
        );
        // Neither center-of-mass group is left with a net velocity.
        let plan = sim.coupling_plan.clone().unwrap();
        for group in &plan.com_groups {
            let mut momentum = Vec3 {
                x: 0.,
                y: 0.,
                z: 0.,
            };
            let mut mass = 0.;
            for &atom in group {
                momentum = add(momentum, scale(sim.state.velocities[atom], sim.masses[atom]));
                mass += sim.masses[atom];
            }
            assert!(norm2(scale(momentum, 1. / mass)).sqrt() < 1e-9);
        }
    }

    #[test]
    fn leapfrog_nose_hoover_conserves_its_extended_energy() {
        // Production-like 9 A cutoff, as for the NVE drift test above.
        let system = solvated_dipeptide_padded(9.0);
        let mut protocol = leapfrog_protocol(200);
        protocol.cutoff_angstrom = Some(9.0);
        protocol.temperature_coupling_interval = Some(1);
        protocol.com_mode = Some("none".into());
        protocol.com_groups.clear();
        let mut sim = ExplicitSimulation::new(&system, protocol).unwrap();
        let plan = sim.coupling_plan.clone().unwrap();
        // Kinetic energy at a full step: the mean of the two half steps.
        let mut previous = kinetic_energy(&sim.masses, &sim.state.velocities);
        let mut series = Vec::new();
        for _ in 0..200 {
            let potential = sim.state.potential_energy;
            let thermostat = plan.thermostat_energy(sim.state.coupling.as_ref().unwrap());
            sim.step().unwrap();
            let next = kinetic_energy(&sim.masses, &sim.state.velocities);
            series.push(potential + 0.5 * (previous + next) + thermostat);
            previous = next;
        }
        let head = series[..20].iter().sum::<f64>() / 20.;
        let tail = series[series.len() - 20..].iter().sum::<f64>() / 20.;
        let drift = ((tail - head) / sim.masses.len() as f64).abs();
        assert!(drift < 0.02, "per-atom drift of the extended energy {drift}");
    }

    #[test]
    fn leapfrog_parrinello_rahman_moves_the_box_towards_the_target_pressure() {
        let system = solvated_dipeptide();
        let initial_volume = protocol_seed_volume(&system);
        let mut sim = ExplicitSimulation::new(&system, leapfrog_npt_protocol(400)).unwrap();
        let mut pressures = Vec::new();
        for _ in 0..400 {
            sim.step().unwrap();
            if sim.state.step % 10 == 1 {
                pressures.push(sim.state.pressure_bar);
            }
        }
        let volume: f64 = sim.state.box_angstrom.iter().product();
        let density = sim.frame().density_g_ml;
        assert!(density > 0.85 && density < 1.15, "density {density}");
        assert!((volume / initial_volume - 1.).abs() > 1e-4, "the box did not move");
        assert!(pressures.iter().all(|p| p.is_finite()));
        eprintln!(
            "volume ratio {:.4}, density {density:.3}, first pressures {:?}, last {:?}",
            volume / initial_volume,
            &pressures[..4],
            &pressures[pressures.len() - 4..]
        );
        assert_eq!(sim.state.pressure_estimator, "atomic-virial-with-constraints");
        // The box keeps its shape under isotropic coupling.
        let initial = system.box_angstrom();
        let ratio = sim.state.box_angstrom[0] / initial[0];
        for d in 1..3 {
            assert!((sim.state.box_angstrom[d] / initial[d] / ratio - 1.).abs() < 1e-9);
        }
        assert!(
            sim.settle
                .as_ref()
                .unwrap()
                .max_violation(&sim.state.coordinates)
                < 1e-3
        );
    }

    #[test]
    fn leapfrog_pressure_matches_the_volume_derivative_of_the_energy() {
        // The virial the barostat uses, checked against a finite difference:
        // with every atom scaled affinely, dU/dV = -(virial of all forces)/3V.
        let system = solvated_dipeptide();
        let mut protocol = leapfrog_npt_protocol(20);
        protocol.dispersion_correction = false;
        let sim = ExplicitSimulation::new(&system, protocol).unwrap();
        let box_vec = sim.box_vectors().unwrap();
        let energy_at = |factor: f64| {
            let scaled_box =
                BoxVectors::new(box_vec.x * factor, box_vec.y * factor, box_vec.z * factor).unwrap();
            let coordinates: Vec<Vec3> =
                sim.state.coordinates.iter().map(|p| scale(*p, factor)).collect();
            let wrapped: Vec<Vec3> = coordinates.iter().map(|p| scaled_box.wrap(*p)).collect();
            let pairs = PbcNeighborList::build(&wrapped, &scaled_box, 4.0, SKIN_ANGSTROM).unwrap();
            sim.field
                .evaluate(
                    &coordinates,
                    &scaled_box,
                    &pairs.pairs,
                    &sim.electrostatics.reaction_field().unwrap(),
                    4.0,
                )
                .unwrap()
                .components
                .total()
        };
        let h: f64 = 1e-5;
        let volume = box_vec.volume();
        let d_volume = volume * ((1. + h).powi(3) - (1. - h).powi(3));
        let finite_difference = -(energy_at(1. + h) - energy_at(1. - h)) / d_volume;
        let from_virial = sim.state.virial_kcal_mol / (3. * volume);
        assert!(
            (finite_difference - from_virial).abs() < 2e-3 * from_virial.abs().max(1e-3),
            "-dU/dV {finite_difference} against virial/3V {from_virial}"
        );
    }

    #[test]
    fn leapfrog_checkpoint_restores_the_coupling_variables() {
        let system = solvated_dipeptide();
        let mut sim = ExplicitSimulation::new(&system, leapfrog_npt_protocol(60)).unwrap();
        // 23 steps: the last one is force-only, so the restored run has to
        // recover the virial itself.
        sim.advance(23).unwrap();
        let json = serde_json::to_string(&sim.state).unwrap();
        let restored: SimulationState = serde_json::from_str(&json).unwrap();
        let mut again = ExplicitSimulation::restore(&system, restored).unwrap();
        assert_eq!(again.state.coupling, sim.state.coupling);
        assert_eq!(again.state.box_angstrom, sim.state.box_angstrom);
        sim.advance(17).unwrap();
        again.advance(17).unwrap();
        assert_eq!(again.state.step, 40);
        let worst = sim
            .state
            .coordinates
            .iter()
            .zip(&again.state.coordinates)
            .map(|(a, b)| norm2(add(*a, scale(*b, -1.))).sqrt())
            .fold(0., f64::max);
        assert!(worst < 1e-3, "restored run departs by {worst} A");
        let (a, b) = (
            sim.state.coupling.as_ref().unwrap(),
            again.state.coupling.as_ref().unwrap(),
        );
        assert!((a.box_velocity[0] - b.box_velocity[0]).abs() < 1e-6);
        assert!((a.pressure_bar - b.pressure_bar).abs() < 1.0);
    }

    #[test]
    fn pme_leapfrog_conserves_its_extended_energy_and_restarts() {
        let system = solvated_dipeptide_padded(9.0);
        let mut protocol = leapfrog_protocol(120);
        protocol.cutoff_angstrom = Some(9.0);
        protocol.electrostatics = ElectrostaticsModel::Pme;
        protocol.temperature_coupling_interval = Some(1);
        protocol.com_mode = Some("none".into());
        protocol.com_groups.clear();
        let parameters = pme_parameters(&system, &protocol).unwrap();
        // GROMACS defaults: erfc(alpha rc) = 1e-5 and a grid no coarser than 1.2 A
        assert!((parameters.alpha_per_angstrom - 0.3468).abs() < 2e-3);
        assert_eq!(parameters.interpolation_order, 4);
        for (points, length) in parameters.grid.iter().zip(system.box_angstrom()) {
            assert!(length / *points as f64 <= 1.2 + 1e-9);
        }
        let mut sim = ExplicitSimulation::new(&system, protocol).unwrap();
        let plan = sim.coupling_plan.clone().unwrap();
        let mut previous = kinetic_energy(&sim.masses, &sim.state.velocities);
        let mut series = Vec::new();
        for step in 0..120 {
            if step == 60 {
                let json = serde_json::to_string(&sim.state).unwrap();
                let restored: SimulationState = serde_json::from_str(&json).unwrap();
                let again = ExplicitSimulation::restore(&system, restored).unwrap();
                assert!((again.state.potential_energy - sim.state.potential_energy).abs() < 1e-6);
            }
            let potential = sim.state.potential_energy;
            let thermostat = plan.thermostat_energy(sim.state.coupling.as_ref().unwrap());
            sim.step().unwrap();
            let next = kinetic_energy(&sim.masses, &sim.state.velocities);
            series.push(potential + 0.5 * (previous + next) + thermostat);
            previous = next;
        }
        let head = series[..20].iter().sum::<f64>() / 20.;
        let tail = series[series.len() - 20..].iter().sum::<f64>() / 20.;
        let drift = ((tail - head) / sim.masses.len() as f64).abs();
        assert!(drift < 0.02, "per-atom drift of the extended energy {drift}");
    }

    #[test]
    fn force_based_pressure_takes_the_virial_of_the_dispersion_tail() {
        // One Lennard-Jones class with N atoms: the tail energy per volume is
        // (8/3) pi N^2 eps sigma^3 [ (1/3)(sigma/rc)^9 - (sigma/rc)^3 ] / V and
        // the tail pressure (16/3) pi N^2 eps sigma^3 [ (2/3)(sigma/rc)^9 -
        // (sigma/rc)^3 ] / V^2 (Allen and Tildesley), with N^2 counted as the
        // N (N + 1) / 2 pairs of the energy correction.
        let system = solvated_dipeptide();
        let field = PbcForceField::new(&system, Vec::new()).unwrap();
        let cutoff = 9.0;
        let energy = field.dispersion_coefficient(cutoff).unwrap();
        let pressure = field.dispersion_pressure_coefficient(cutoff).unwrap();
        // dominated by the attractive term, for which the factor is two
        assert!(energy < 0. && pressure < 0.);
        let ratio = pressure / energy;
        assert!(ratio > 1.99 && ratio < 2.0, "tail pressure / tail energy = {ratio}");
        // and exactly 4 E12 + 2 E6 with E12 + E6 the energy coefficient at
        // two cutoffs, which separates the two powers of the cutoff
        let (e1, e2) = (energy, field.dispersion_coefficient(2. * cutoff).unwrap());
        // E12 scales as rc^-9 and E6 as rc^-3
        let e12 = (e2 - e1 / 8.) / (1. / 512. - 1. / 8.);
        let e6 = e1 - e12;
        assert!((pressure - (4. * e12 + 2. * e6)).abs() < 1e-9 * pressure.abs());
    }

    #[test]
    fn pme_with_the_monte_carlo_barostat_is_refused() {
        let system = solvated_dipeptide();
        let mut protocol = nve_protocol();
        protocol.equilibration_ensemble = Ensemble::Npt;
        protocol.production_ensemble = Ensemble::Npt;
        protocol.timestep_fs = 2.0;
        protocol.constraints = ConstraintModel::Settle;
        protocol.friction_per_ps = 1.0;
        protocol.electrostatics = ElectrostaticsModel::Pme;
        assert!(ExplicitSimulation::new(&system, protocol).is_err());
    }
}
