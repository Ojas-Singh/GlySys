//! Explicit-water PBC dynamics: minimization, NVE, NVT, and NPT.
//!
//! Coordinates live in *unwrapped* space for the whole run; wrapping happens
//! only for pair-list builds and exported frames. Thermostat noise and the
//! barostat use independent deterministic streams so barostat schedule
//! changes never perturb thermostat reproducibility.
use super::accumulators::{Accumulator, WaterOccupancy};
use super::settle::SettleWaters;
use super::{ACCEL, ConstraintModel, Ensemble, KB, KBAR_PER_KCAL_MOL_A3, Thermostat, normal};
use super::{Error, Result, SimulationProtocol, SimulationState, SolventModel, TrajectoryChunk};
use super::{frame_from_state, invalid};
use glysys::{ParameterizedSystem, Vec3};
use glysys_energy::HarmonicRestraint;
use glysys_energy::pbc::{
    BoxVectors, NonbondedElectrostatics, PbcForceField, PbcNeighborList,
    ReactionField, classify_waters, molecules,
};
use sha2::{Digest, Sha256};

pub const SKIN_ANGSTROM: f64 = 1.5;
/// 1 fs for flexible waters, 2 fs with SETTLE. Larger requests are rejected
/// rather than silently integrated.
pub fn max_timestep_fs(constraints: ConstraintModel) -> f64 {
    match constraints {
        ConstraintModel::None => 1.0,
        ConstraintModel::Settle => 2.0,
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

pub fn rf_backend(protocol: &SimulationProtocol) -> Result<ReactionField> {
    ReactionField::new(
        cutoff_angstrom(protocol),
        protocol.rf_dielectric.unwrap_or(78.5),
    )
    .map_err(Error::Energy)
}

fn electrostatics_kind(protocol: &SimulationProtocol) -> NonbondedElectrostatics {
    NonbondedElectrostatics::ReactionField {
        cutoff_angstrom: cutoff_angstrom(protocol),
        solvent_dielectric: protocol.rf_dielectric.unwrap_or(78.5),
    }
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
fn solute_restraints(system: &ParameterizedSystem, force: f64) -> Vec<HarmonicRestraint> {
    if force <= 0. {
        return Vec::new();
    }
    let waters: std::collections::HashSet<usize> = classify_waters(system)
        .into_iter()
        .flatten()
        .collect();
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
    format!(
        "{:x}",
        Sha256::digest(
            format!(
                "{}:{:?}:{:?}:{restraints}:{system:?}",
                super::EXPLICIT_MODEL_VERSION,
                electrostatics_kind(protocol),
                protocol.constraints,
            )
            .as_bytes()
        )
    )
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
    (n_atoms as f64 * KB * temperature_k + virial / 3.) / volume_a3 * KBAR_PER_KCAL_MOL_A3
}

fn dof_count(atom_count: usize, settle: &Option<SettleWaters>) -> usize {
    let constrained = settle
        .as_ref()
        .map(|s| s.constraint_count())
        .unwrap_or(0);
    (3 * atom_count).saturating_sub(constrained).max(1)
}

pub fn density_g_ml(total_amu: f64, volume_a3: f64) -> f64 {
    if volume_a3 <= 0. {
        return 0.;
    }
    total_amu * 1.660_539_066_60 / volume_a3
}

pub struct ExplicitSimulation<'a> {
    system: &'a ParameterizedSystem,
    field: PbcForceField<'a>,
    masses: Vec<f64>,
    settle: Option<SettleWaters>,
    waters: Vec<[usize; 3]>,
    molecules: Vec<Vec<usize>>,
    backend: ReactionField,
    pairs: PbcNeighborList,
    /// Unconstrained velocity DOF: 3N minus one per distance constraint.
    degrees_of_freedom: usize,
    pub state: SimulationState,
}

impl<'a> ExplicitSimulation<'a> {
    pub fn new(system: &'a ParameterizedSystem, protocol: SimulationProtocol) -> Result<Self> {
        protocol.validate()?;
        if protocol.solvent != SolventModel::Explicit {
            return Err(invalid("explicit driver needs solvent=explicit"));
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
        if cutoff + SKIN_ANGSTROM >= 0.5 * box_vec.x.min(box_vec.y).min(box_vec.z) {
            return Err(invalid(
                "cutoff plus skin must be below half the shortest box edge",
            ));
        }
        if system.atoms().is_empty()
            || system.atoms().iter().any(|a| !a.mass().is_finite() || a.mass() <= 0.)
        {
            return Err(invalid("finite positive masses required"));
        }
        let restraints = solute_restraints(system, protocol.restraint_force);
        let field = PbcForceField::new(system, restraints.clone()).map_err(Error::Energy)?;
        let backend = rf_backend(&protocol)?;
        let waters = classify_waters(system);
        if waters.is_empty() {
            return Err(invalid(
                "explicit solvent needs classified TIP3P waters; prepare with solvation",
            ));
        }
        let molecules = molecules(system);
        let masses: Vec<_> = system.atoms().iter().map(|a| a.mass()).collect();
        let mut coordinates = system.coordinates();
        // Flexible minimization first; constraints engage at dynamics start,
        // matching the OpenMM validation protocol.
        if protocol.minimization_iterations > 0 {
            coordinates = minimize(&field, &box_vec, &backend, coordinates, &protocol)?;
        }
        let h_bonds = solute_h_bonds(system, &waters);
        let targets = water_targets(system, &waters)?;
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
        };
        if let Some(settle) = &settle {
            settle.constrain_positions(&mut coordinates)?;
        }
        let wrapped: Vec<Vec3> = coordinates.iter().map(|p| box_vec.wrap(*p)).collect();
        let pairs = PbcNeighborList::build(&wrapped, &box_vec, cutoff, SKIN_ANGSTROM)
            .map_err(Error::Energy)?;
        let energy = field
            .evaluate(&coordinates, &box_vec, &pairs.pairs, &backend, cutoff_angstrom(&protocol))
            .map_err(Error::Energy)?;
        let mut rng = protocol.seed;
        let velocities = masses
            .iter()
            .map(|m| random_velocity(&mut rng, (KB * protocol.temperature_k * ACCEL / m).sqrt()))
            .collect::<Vec<_>>();
        let barostat_rng = protocol_seed_barostat(protocol_seed(&protocol));
        let degrees_of_freedom = dof_count(system.atom_count(), &settle);
        let mut sim = Self {
            system,
            field,
            masses,
            settle,
            waters,
            molecules,
            backend,
            pairs,
            degrees_of_freedom,
            state: SimulationState {
                schema_version: 1,
                model_version: super::EXPLICIT_MODEL_VERSION.into(),
                system_fingerprint: String::new(),
                protocol,
                step: 0,
                reference_coordinates: coordinates.clone(),
                coordinates,
                velocities,
                gradient: energy.gradients,
                potential_energy: energy.components.total(),
                rng_state: rng,
                integrator_phase: "ready".into(),
                box_angstrom: box_vec.as_array(),
                virial_kcal_mol: energy.virial,
                pressure_bar: 0.,
                barostat_rng,
                water_occupancy: None,
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
        Ok(sim)
    }

    fn field_restraint_count(&self) -> usize {
        self.system
            .atoms()
            .len()
            .saturating_sub(self.waters.len() * 3)
    }

    pub fn restore(system: &'a ParameterizedSystem, state: SimulationState) -> Result<Self> {
        state.protocol.validate()?;
        if state.schema_version != 1
            || state.model_version != super::EXPLICIT_MODEL_VERSION
            || state.integrator_phase != "ready"
            || state.coordinates.len() != system.atom_count()
            || state.velocities.len() != system.atom_count()
            || state.gradient.len() != system.atom_count()
            || state.step > state.protocol.equilibration_steps + state.protocol.production_steps
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
        let backend = rf_backend(&state.protocol)?;
        let waters = classify_waters(system);
        let masses: Vec<_> = system.atoms().iter().map(|a| a.mass()).collect();
        let h_bonds = solute_h_bonds(system, &waters);
        let targets = water_targets(system, &waters)?;
        let settle = match state.protocol.constraints {
            ConstraintModel::None => None,
            ConstraintModel::Settle => Some(SettleWaters::from_equilibrium(
                waters.clone(),
                targets,
                h_bonds,
                &masses,
                system.atom_count(),
            )?),
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
        let reference = field
            .evaluate(&state.coordinates, &box_vec, &pairs.pairs, &backend, cutoff_angstrom(&state.protocol))
            .map_err(Error::Energy)?;
        if (reference.components.total() - state.potential_energy).abs()
            > 1e-3 + 1e-4 * reference.components.total().abs()
        {
            return Err(invalid("checkpoint energy does not match prepared chemistry"));
        }
        let degrees_of_freedom = dof_count(system.atom_count(), &settle);
        let sim = Self {
            system,
            field,
            masses,
            settle,
            waters,
            molecules: molecules(system),
            backend,
            pairs,
            degrees_of_freedom,
            state,
        };
        let expected =
            fingerprint(system, &sim.state.protocol, sim.field_restraint_count());
        if sim.state.system_fingerprint != expected {
            return Err(invalid("checkpoint chemistry fingerprint mismatch"));
        }
        Ok(sim)
    }

    fn segment_ensemble(&self) -> Ensemble {
        if self.state.step < self.state.protocol.equilibration_steps {
            self.state.protocol.equilibration_ensemble
        } else {
            self.state.protocol.production_ensemble
        }
    }

    fn refresh_pairs(&mut self) -> Result<()> {
        let box_vec = self.box_vectors()?;
        let wrapped: Vec<Vec3> = self.state.coordinates.iter().map(|p| box_vec.wrap(*p)).collect();
        if self.pairs.needs_rebuild(&wrapped) {
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
        self.refresh_pairs()?;
        match self.segment_ensemble() {
            Ensemble::Nve => self.nve_step(),
            Ensemble::Nvt => self.nvt_step(),
            Ensemble::Npt => {
                self.nvt_step()?;
                if self.state.step % self.state.protocol.barostat_interval == 0 {
                    self.try_barostat()?;
                }
                Ok(())
            }
        }?;
        self.state.step += 1;
        let box_vec = self.box_vectors()?;
        self.state.pressure_bar = pressure_bar(
            self.masses.len(),
            self.state.protocol.temperature_k,
            self.state.virial_kcal_mol,
            box_vec.volume(),
        );
        Ok(())
    }

    /// Velocity Verlet with constraints: the NVE validation stage.
    ///
    /// Waters move as rigid bodies (analytic-SETTLE-style): positions
    /// translate/rotate rigidly, and half-step velocities rotate by the same
    /// rotation, which keeps them geometrically consistent without deleting
    /// anything. A full-step RATTLE projection after the second kick then
    /// removes only genuine radial slosh — including the fresh kick
    /// contribution, which would otherwise accumulate into a radial ratchet
    /// — while tangential (rotational) components pass through untouched.
    /// Solute X-H bonds use SHAKE/RATTLE throughout (slow rotation makes
    /// projection loss negligible there).
    fn nve_step(&mut self) -> Result<()> {
        let dt = self.state.protocol.timestep_fs * 0.001;
        let prev_coords = self.state.coordinates.clone();
        let mut next_coords = prev_coords.clone();
        let mut next_vel = self.state.velocities.clone();
        for (i, m) in self.masses.iter().enumerate() {
            next_vel[i] = add(next_vel[i], scale(self.state.gradient[i], -0.5 * dt * ACCEL / m));
            next_coords[i] = add(next_coords[i], scale(next_vel[i], dt));
        }
        if let Some(settle) = &self.settle {
            settle.rigid_water_positions(&prev_coords, &next_coords.clone(), &mut next_coords, &self.masses)?;
            settle.shake_solute_positions(&mut next_coords)?;
            settle.rotate_water_velocities(&prev_coords, &next_coords, &mut next_vel, &self.masses)?;
        }
        self.refresh_pairs_for(&next_coords)?;
        let box_vec = self.box_vectors()?;
        let energy = self.field.evaluate(&next_coords, &box_vec, &self.pairs.pairs, &self.backend, cutoff_angstrom(&self.state.protocol)).map_err(Error::Energy)?;
        for (i, m) in self.masses.iter().enumerate() {
            next_vel[i] = add(next_vel[i], scale(energy.gradients[i], -0.5 * dt * ACCEL / m));
        }
        if let Some(settle) = &self.settle {
            settle.constrain_velocities(&next_coords, &mut next_vel)?;
        }
        self.commit(next_coords, next_vel, energy.components.total(), energy.gradients, energy.virial)
    }

    /// BAOAB Langevin with constraints after every position and velocity update.
    ///
    /// Waters move rigidly with SETTLE-style velocity rotation at the full
    /// step; a full-step RATTLE after the second kick removes fresh radial
    /// slosh (which would otherwise ratchet) while leaving rotation
    /// untouched. The mid-step projection covers solute bonds only.
    fn nvt_step(&mut self) -> Result<()> {
        let dt = self.state.protocol.timestep_fs * 0.001;
        let decay = (-self.state.protocol.friction_per_ps * dt).exp();
        let prev_coords = self.state.coordinates.clone();
        let mut next_coords = prev_coords.clone();
        let mut next_vel = self.state.velocities.clone();
        let mut rng = self.state.rng_state;
        // B + A halves.
        for (i, m) in self.masses.iter().enumerate() {
            next_vel[i] = add(next_vel[i], scale(self.state.gradient[i], -0.5 * dt * ACCEL / m));
            next_coords[i] = add(next_coords[i], scale(next_vel[i], 0.5 * dt));
        }
        if let Some(settle) = &self.settle {
            settle.rigid_water_positions(&prev_coords, &next_coords.clone(), &mut next_coords, &self.masses)?;
            settle.shake_solute_positions(&mut next_coords)?;
        }
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
                next_vel[i] = add(scale(next_vel[i], decay), random_velocity(&mut rng, sigma));
                next_coords[i] = add(next_coords[i], scale(next_vel[i], 0.5 * dt));
            }
        }
        if let Some(settle) = &self.settle {
            settle.rigid_water_positions(&prev_coords, &next_coords.clone(), &mut next_coords, &self.masses)?;
            settle.shake_solute_positions(&mut next_coords)?;
            settle.project_solute_velocities(&next_coords, &mut next_vel)?;
        }
        if let Some(settle) = &self.settle {
            settle.rotate_water_velocities(&prev_coords, &next_coords, &mut next_vel, &self.masses)?;
        }
        self.refresh_pairs_for(&next_coords)?;
        let box_vec = self.box_vectors()?;
        let energy = self.field.evaluate(&next_coords, &box_vec, &self.pairs.pairs, &self.backend, cutoff_angstrom(&self.state.protocol)).map_err(Error::Energy)?;
        if !energy.components.total().is_finite() {
            return Err(invalid("nonfinite explicit energy; checkpoint retained"));
        }
        for (i, m) in self.masses.iter().enumerate() {
            next_vel[i] = add(next_vel[i], scale(energy.gradients[i], -0.5 * dt * ACCEL / m));
            let displacement = add(next_coords[i], scale(self.state.coordinates[i], -1.));
            if !finite(&next_coords[i]) || !finite(&next_vel[i]) || norm2(displacement) > 1. {
                return Err(invalid("unstable explicit step (>1 A); checkpoint retained"));
            }
        }
        if let Some(settle) = &self.settle {
            settle.constrain_velocities(&next_coords, &mut next_vel)?;
        }
        self.state.rng_state = rng;
        self.commit(next_coords, next_vel, energy.components.total(), energy.gradients, energy.virial)
    }

    /// Isotropic Monte Carlo barostat. Volume moves use their own stream;
    /// restraint forces are external and excluded from the virial by design.
    fn try_barostat(&mut self) -> Result<()> {
        use super::normal as gaussian;
        let box_vec = self.box_vectors()?;
        let volume = box_vec.volume();
        // Gaussian log-volume proposal, sigma tuned for water boxes.
        let delta = gaussian(&mut self.state.barostat_rng) * 0.005;
        let proposed_volume = volume * delta.exp();
        let linear = (proposed_volume / volume).powf(1. / 3.);
        let proposed_box = BoxVectors {
            x: box_vec.x * linear,
            y: box_vec.y * linear,
            z: box_vec.z * linear,
        };
        if cutoff_angstrom(&self.state.protocol) + SKIN_ANGSTROM
            >= 0.5 * proposed_box.x.min(proposed_box.y).min(proposed_box.z)
        {
            return Ok(());
        }
        let mut proposed_coords: Vec<Vec3> = self
            .state
            .coordinates
            .iter()
            .map(|p| scale(*p, linear))
            .collect();
        // Isotropic scaling stretches constrained bonds by the same factor;
        // re-project to the constraint manifold before evaluating, or the
        // accepted state would carry a permanent violation (rigid updates
        // preserve whatever lengths they are given, so nothing later would
        // repair it). Velocities are re-projected on acceptance below.
        if let Some(settle) = &self.settle {
            settle.constrain_positions(&mut proposed_coords)?;
        }
        let wrapped: Vec<Vec3> = proposed_coords.iter().map(|p| proposed_box.wrap(*p)).collect();
        let pairs = PbcNeighborList::build(
            &wrapped,
            &proposed_box,
            cutoff_angstrom(&self.state.protocol),
            SKIN_ANGSTROM,
        )
        .map_err(Error::Energy)?;
        let energy = self.field.evaluate(&proposed_coords, &proposed_box, &pairs.pairs, &self.backend, cutoff_angstrom(&self.state.protocol)).map_err(Error::Energy)?;
        let n = self.masses.len() as f64;
        let beta = 1. / (KB * self.state.protocol.temperature_k);
        // Pressure-volume work in kcal/mol: P_bar * dV_A3 / KBAR.
        let pv = self.state.protocol.pressure_bar * (proposed_volume - volume) / KBAR_PER_KCAL_MOL_A3;
        let delta_g = energy.components.total() - self.state.potential_energy + pv
            - n * KB * self.state.protocol.temperature_k * (proposed_volume / volume).ln();
        let accept = delta_g <= 0.
            || {
                let u = super::uniform(&mut self.state.barostat_rng);
                u < (-beta * delta_g).exp()
            };
        if accept {
            self.state.coordinates = proposed_coords;
            if let Some(settle) = &self.settle {
                settle.constrain_velocities(
                    &self.state.coordinates,
                    &mut self.state.velocities,
                )?;
            }
            self.state.box_angstrom = proposed_box.as_array();
            self.pairs = pairs;
            self.state.potential_energy = energy.components.total();
            self.state.gradient = energy.gradients;
            self.state.virial_kcal_mol = energy.virial;
        }
        Ok(())
    }

    fn refresh_pairs_for(&mut self, coords: &[Vec3]) -> Result<()> {
        let box_vec = self.box_vectors()?;
        let wrapped: Vec<Vec3> = coords.iter().map(|p| box_vec.wrap(*p)).collect();
        // Displacement-gated rebuild: forces never use stale pairs, and
        // quiescent systems skip the rebuild entirely.
        if self.pairs.needs_rebuild(&wrapped) {
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

    pub fn advance(&mut self, steps: usize) -> Result<TrajectoryChunk> {
        let first_step = self.state.step;
        let end = (self.state.step + steps.min(100)).min(
            self.state.protocol.equilibration_steps + self.state.protocol.production_steps,
        );
        let mut frames = Vec::new();
        while self.state.step < end {
            self.step()?;
            if self.state.step.is_multiple_of(self.state.protocol.save_every)
                || self.state.step == self.state.protocol.equilibration_steps
                || self.state.step
                    == self.state.protocol.equilibration_steps
                        + self.state.protocol.production_steps
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
        if self.state.step <= self.state.protocol.equilibration_steps {
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
        let entry = self.state.water_occupancy.get_or_insert_with(|| {
            WaterOccupancy::new(box_vec.as_array(), 1.0)
        });
        entry.observe(&oxygen, box_vec.as_array());
    }

    pub fn frame(&self) -> super::TrajectoryFrame {
        let mut frame = frame_from_state(&self.state, &self.masses);
        frame.temperature_k = kinetic_temperature(
            kinetic_energy(&self.masses, &self.state.velocities),
            self.degrees_of_freedom,
        );
        let total_amu: f64 = self.masses.iter().sum();
        let volume = self.state.box_angstrom[0] * self.state.box_angstrom[1] * self.state.box_angstrom[2];
        frame.density_g_ml = density_g_ml(total_amu, volume);
        frame.segment = if self.state.step <= self.state.protocol.equilibration_steps {
            "equilibration"
        } else {
            "production"
        }
        .into();
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
    seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x1234_5678_9ABC_DEF0) | 1
}

/// Robust pre-relaxation for clash-heavy solvated starts. Fixed small
/// steps with backtracking; stops when the max force is tame enough for LBFGS
/// or the iteration budget is spent.
fn steepest_descent(
    field: &PbcForceField,
    box_vec: &BoxVectors,
    backend: &ReactionField,
    mut coords: Vec<Vec3>,
    protocol: &SimulationProtocol,
) -> Result<Vec<Vec3>> {
    let cutoff = cutoff_angstrom(protocol);
    let mut step = 0.01;
    for _ in 0..500 {
        let wrapped: Vec<Vec3> = coords.iter().map(|p| box_vec.wrap(*p)).collect();
        let pairs = PbcNeighborList::build(&wrapped, box_vec, cutoff, SKIN_ANGSTROM)
            .map_err(Error::Energy)?;
        let energy = field
            .evaluate(&coords, box_vec, &pairs.pairs, backend, cutoff)
            .map_err(Error::Energy)?;
        let gmax = energy
            .gradients
            .iter()
            .map(|g| (g.x * g.x + g.y * g.y + g.z * g.z).sqrt())
            .fold(0f64, f64::max);
        if !gmax.is_finite() {
            return Err(invalid("nonfinite gradient during minimization"));
        }
        if gmax < 50. {
            return Ok(coords);
        }
        // Backtracking line search on the steepest-descent direction.
        let current = energy.components.total();
        let mut accepted = false;
        for _ in 0..20 {
            let trial: Vec<Vec3> = coords
                .iter()
                .zip(&energy.gradients)
                .map(|(p, g)| Vec3 {
                    x: p.x - step * g.x,
                    y: p.y - step * g.y,
                    z: p.z - step * g.z,
                })
                .collect();
            let w: Vec<Vec3> = trial.iter().map(|p| box_vec.wrap(*p)).collect();
            let pairs = PbcNeighborList::build(&w, box_vec, cutoff, SKIN_ANGSTROM)
                .map_err(Error::Energy)?;
            let trial_energy = field
                .evaluate(&trial, box_vec, &pairs.pairs, backend, cutoff)
                .map_err(Error::Energy)?
                .components
                .total();
            if trial_energy.is_finite() && trial_energy < current {
                coords = trial;
                step = (step * 1.5).min(0.05);
                accepted = true;
                break;
            }
            step *= 0.5;
            if step < 1e-8 {
                break;
            }
        }
        if !accepted {
            return Ok(coords);
        }
    }
    Ok(coords)
}

fn minimize(
    field: &PbcForceField,
    box_vec: &BoxVectors,
    backend: &ReactionField,
    initial: Vec<Vec3>,
    protocol: &SimulationProtocol,
) -> Result<Vec<Vec3>> {
    // Steepest-descent pre-relaxation first: solvated boxes start with
    // clashing contacts whose huge forces stall quasi-Newton line searches.
    let coords = steepest_descent(field, box_vec, backend, initial, protocol)?;
    // LBFGS runs on a FIXED pair list with a generous skin: rebuilding pairs
    // inside the line search makes the objective discontinuous and stalls
    // convergence. The list is still validity-checked every evaluation.
    let cutoff = cutoff_angstrom(protocol);
    let wrapped: Vec<Vec3> = coords.iter().map(|p| box_vec.wrap(*p)).collect();
    let min_edge = box_vec.x.min(box_vec.y).min(box_vec.z);
    let min_skin = (0.5 * min_edge - cutoff - 0.05).max(0.5);
    let mut pairs = PbcNeighborList::build(&wrapped, box_vec, cutoff, min_skin)
        .map_err(Error::Energy)?;
    let config = glysys_opt::LbfgsConfig {
        max_iterations: protocol.minimization_iterations,
        ..Default::default()
    };
    let flat: Vec<_> = coords.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let mut optimizer = glysys_opt::resumable::LbfgsState::new(&flat, &config)?;
    let mut evals = 0usize;
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
            pairs =
                PbcNeighborList::build(&wrapped, box_vec, cutoff, min_skin)
                    .map_err(Error::Energy)?;
        }
        let energy = field
            .evaluate(&coords, box_vec, &pairs.pairs, backend, cutoff)
            .map_err(Error::Energy)?;
        evals += 1;
        if std::env::var("GLYSYS_MIN_TRACE").is_ok() && evals % 50 == 0 {
            let gmax = energy
                .gradients
                .iter()
                .map(|g| (g.x * g.x + g.y * g.y + g.z * g.z).sqrt())
                .fold(0f64, f64::max);
            eprintln!("min eval {evals}: E={} gmax={gmax:.3}", energy.components.total());
        }
        optimizer.submit(
            energy.components.total(),
            energy
                .gradients
                .iter()
                .flat_map(|p| [p.x, p.y, p.z])
                .collect(),
        )?;
    }
    Ok(optimizer
        .outcome()
        .unwrap()
        .point
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
        // Same bar as the flexible path: constrained 2 fs steps must not
        // pump or drain total energy. Guards the rigid-body update against
        // both the RATTLE-rotation-damping and velocity-rotation-ratchet
        // failure modes (each missed by gigabytes of passing unit tests).
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
        }
        let total = sim.state.potential_energy + kinetic(&sim);
        let drift = ((total - initial_total) / sim.masses.len() as f64).abs();
        assert!(drift < 0.05, "per-atom drift {drift}");
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
        assert!(sim.settle.as_ref().unwrap().max_violation(&sim.state.coordinates) < 1e-8);
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
}
