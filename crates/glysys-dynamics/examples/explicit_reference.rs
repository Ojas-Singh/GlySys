#![recursion_limit = "256"]

//! Export the solvated-dipeptide PBC reference case for
//! `benchmarks/openmm_pbc_rf.py`.
//!
//! The checker rebuilds the identical Amber chemistry from the exported
//! prmtop with CutoffPeriodic + reaction field (no switching, RF dielectric
//! matched) and validates, in layers:
//!   1. single-point total energy, per-term components (via force groups,
//!      with an LJ-only leg isolating electrostatics), and forces;
//!   2. NVE total-energy drift for SETTLE (2 fs) and flexible (1 fs) legs,
//!      plus SETTLE drift convergence over 0.5/1.0/2.0 fs at fixed physical
//!      time (second-order integrators: drift should scale ~dt^2);
//!   3. NVT ensemble means with full series so the checker can form
//!      autocorrelation-aware standard errors instead of fixed floors.
//!
//! Snapshots carry minimized coordinates, post-NVE coordinates, and the
//! production-only NVT series (equilibration is not ensemble data).
use glysys::{BuildOptions, SystemBuilder};
use glysys_dynamics::explicit::{ExplicitSimulation, kinetic_energy, molecular_pressure_bar};
use glysys_dynamics::{ConstraintModel, Ensemble, SimulationProtocol, SolventModel, Thermostat};
use glysys_energy::pbc::{
    BoxVectors, PbcEnergy, PbcForceField, PbcNeighborList, ReactionField, molecules,
};

fn masses_of(system: &glysys::ParameterizedSystem) -> Vec<f64> {
    system.atoms().iter().map(|a| a.mass()).collect()
}

fn total_energy(sim: &ExplicitSimulation, masses: &[f64]) -> f64 {
    sim.state.potential_energy + kinetic_energy(masses, &sim.state.velocities)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Independent single-point evaluation at the given coordinates, using only
/// public force-field API (separate path from the simulation's cached state).
fn energy_at(
    system: &glysys::ParameterizedSystem,
    coords: &[glysys::Vec3],
) -> Result<PbcEnergy, Box<dyn std::error::Error>> {
    let box_vec = BoxVectors::from_system(system)?;
    let field = PbcForceField::new(system, vec![])?;
    let backend = ReactionField::new(9.0, 78.5)?;
    let wrapped: Vec<_> = coords.iter().map(|p| box_vec.wrap(*p)).collect();
    let pairs = PbcNeighborList::build(&wrapped, &box_vec, 9.0, 1.5)?;
    Ok(field.evaluate(coords, &box_vec, &pairs.pairs, &backend, 9.0)?)
}

fn components_at(
    system: &glysys::ParameterizedSystem,
    coords: &[glysys::Vec3],
) -> Result<glysys_energy::EnergyComponents, Box<dyn std::error::Error>> {
    Ok(energy_at(system, coords)?.components)
}

/// Independent molecular-pressure probe for the parity harness.  This is the
/// same finite-difference, whole-molecule scaling used by the NPT driver; the
/// atomic virial remains a separate diagnostic in the exported fixture.
fn molecular_pressure_at(
    system: &glysys::ParameterizedSystem,
    protocol: &SimulationProtocol,
    coords: &[glysys::Vec3],
    velocities: &[glysys::Vec3],
    masses: &[f64],
) -> Result<f64, Box<dyn std::error::Error>> {
    let box_vec = BoxVectors::from_system(system)?;
    let field = PbcForceField::new(system, vec![])?;
    let backend = ReactionField::new(9.0, 78.5)?;
    Ok(molecular_pressure_bar(
        &field,
        &backend,
        protocol,
        &box_vec,
        coords,
        velocities,
        masses,
        &molecules(system),
    )?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let system = SystemBuilder::new(BuildOptions {
        add_water: true,
        add_ions: false,
        padding_angstrom: 9.0,
        ..Default::default()
    })?
    .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))?;
    let masses = masses_of(&system);
    let n_atoms = masses.len();
    // The full defaults are suitable for a release benchmark. Shorter legs
    // are useful when iterating on kernels on a CPU-only validation host.
    let nve_steps = env_usize("GLYSYS_REF_NVE_STEPS", 200);
    let convergence_window_ps = env_f64("GLYSYS_REF_CONVERGENCE_PS", 0.4);
    let nvt_equilibration_steps = env_usize("GLYSYS_REF_NVT_EQUILIBRATION", 4000);
    let nvt_production_steps = env_usize("GLYSYS_REF_NVT_PRODUCTION", 4000);
    let minimization_iterations = env_usize("GLYSYS_REF_MINIMIZATION", 200);
    let base = SimulationProtocol {
        solvent: SolventModel::Explicit,
        cutoff_angstrom: Some(9.0),
        rf_dielectric: Some(78.5),
        constraints: ConstraintModel::Settle,
        timestep_fs: 2.0,
        minimization_iterations,
        seed: 11,
        ..Default::default()
    };
    // Snapshot A: minimized geometry for single-point parity.
    let nve = SimulationProtocol {
        equilibration_ensemble: Ensemble::Nve,
        production_ensemble: Ensemble::Nve,
        equilibration_steps: 0,
        production_steps: nve_steps,
        save_every: (nve_steps / 20).max(1),
        friction_per_ps: 0.0,
        ..base.clone()
    };
    let mut sim = ExplicitSimulation::new(&system, nve.clone())?;
    let energy_a = sim.state.potential_energy;
    let velocities_a = sim.state.velocities.clone();
    let forces_a: Vec<[f64; 3]> = sim
        .state
        .gradient
        .iter()
        .map(|g| [-g.x, -g.y, -g.z])
        .collect();
    let components_a = components_at(&system, &sim.state.coordinates)?;
    let energy_probe_a = energy_at(&system, &sim.state.coordinates)?;
    let virial_a = energy_probe_a.virial;
    let molecular_pressure_a = molecular_pressure_at(
        &system,
        &nve,
        &sim.state.coordinates,
        &velocities_a,
        &masses,
    )?;
    if std::env::var("GLYSYS_REF_STATIC_ONLY").is_ok() {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "boxAngstrom": system.box_angstrom(),
                "files": system.bundle_strings()?,
                "snapshotA": sim.state.coordinates,
                "energyA": energy_a,
                "forcesA": forces_a,
                "velocitiesA": velocities_a,
                "componentsA": components_a,
                "virialA": virial_a,
                "virialTermsA": energy_probe_a.virial_terms,
                "virialPairSplitA": energy_probe_a.virial_pair_split,
                "molecularPressureA": molecular_pressure_a,
            }))?
        );
        return Ok(());
    }
    let mut drift = vec![sim.state.potential_energy];
    let mut total_drift = vec![total_energy(&sim, &masses)];
    let mut max_constraint_residual = sim.constraint_violation().unwrap_or(0.0);
    let mut max_velocity_constraint_residual = sim.velocity_constraint_violation().unwrap_or(0.0);
    let mut snapshot_step1 = None;
    let mut velocities_step1 = None;
    for i in 0..nve_steps {
        sim.step()?;
        max_constraint_residual =
            max_constraint_residual.max(sim.constraint_violation().unwrap_or(0.0));
        max_velocity_constraint_residual = max_velocity_constraint_residual
            .max(sim.velocity_constraint_violation().unwrap_or(0.0));
        if i == 0 {
            snapshot_step1 = Some(sim.state.coordinates.clone());
            velocities_step1 = Some(sim.state.velocities.clone());
        }
        if (i + 1) % (nve_steps / 20).max(1) == 0 {
            drift.push(sim.state.potential_energy);
            total_drift.push(total_energy(&sim, &masses));
        }
    }
    let snapshot_b = sim.state.coordinates.clone();
    let velocities_b = sim.state.velocities.clone();
    let energy_b = sim.state.potential_energy;
    let forces_b: Vec<[f64; 3]> = sim
        .state
        .gradient
        .iter()
        .map(|g| [-g.x, -g.y, -g.z])
        .collect();
    let components_b = components_at(&system, &snapshot_b)?;
    let energy_probe_b = energy_at(&system, &snapshot_b)?;
    let virial_b = energy_probe_b.virial;
    let molecular_pressure_b =
        molecular_pressure_at(&system, &nve, &snapshot_b, &velocities_b, &masses)?;
    if std::env::var("GLYSYS_REF_NVE_ONLY").is_ok() {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "boxAngstrom": system.box_angstrom(),
                "files": system.bundle_strings()?,
                "snapshotA": sim.state.reference_coordinates,
                "energyA": energy_a,
                "forcesA": forces_a,
                "velocitiesA": velocities_a,
                "componentsA": components_a,
                "virialA": virial_a,
                "snapshotB": snapshot_b,
                "energyB": energy_b,
                "forcesB": forces_b,
                "velocitiesB": velocities_b,
                "componentsB": components_b,
                "virialB": virial_b,
                "virialTermsB": energy_probe_b.virial_terms,
                "virialPairSplitB": energy_probe_b.virial_pair_split,
                "molecularPressureA": molecular_pressure_a,
                "molecularPressureB": molecular_pressure_b,
                "maxConstraintResidualA": max_constraint_residual,
                "maxVelocityConstraintResidualAps": max_velocity_constraint_residual,
                "nveTotalDrift": total_drift,
                "snapshotStep1": snapshot_step1,
                "velocitiesStep1": velocities_step1,
            }))?
        );
        return Ok(());
    }
    // Flexible NVE leg (1 fs max for flexible waters): isolates constraint
    // effects by comparing against OpenMM with constraints=None.
    let flex_proto = SimulationProtocol {
        solvent: SolventModel::Explicit,
        cutoff_angstrom: Some(9.0),
        rf_dielectric: Some(78.5),
        constraints: ConstraintModel::None,
        timestep_fs: 1.0,
        minimization_iterations,
        seed: 11,
        equilibration_ensemble: Ensemble::Nve,
        production_ensemble: Ensemble::Nve,
        equilibration_steps: 0,
        production_steps: nve_steps,
        save_every: (nve_steps / 20).max(1),
        friction_per_ps: 0.0,
        ..Default::default()
    };
    let mut flex_sim = ExplicitSimulation::new(&system, flex_proto)?;
    let mut flex_total = vec![total_energy(&flex_sim, &masses)];
    for i in 0..nve_steps {
        flex_sim.step()?;
        if (i + 1) % (nve_steps / 20).max(1) == 0 {
            flex_total.push(total_energy(&flex_sim, &masses));
        }
    }
    // Timestep convergence: a fixed physical window at three step sizes.
    let mut convergence = Vec::new();
    for dt in [0.5, 1.0, 2.0] {
        let proto = SimulationProtocol {
            timestep_fs: dt,
            equilibration_ensemble: Ensemble::Nve,
            production_ensemble: Ensemble::Nve,
            production_steps: (convergence_window_ps / (dt * 0.001)).round() as usize,
            save_every: (convergence_window_ps / (dt * 0.001)).round() as usize,
            friction_per_ps: 0.0,
            ..base.clone()
        };
        let mut s = ExplicitSimulation::new(&system, proto)?;
        let e0 = total_energy(&s, &masses);
        let mut maximum_drift = 0f64;
        while s.state.step < s.state.protocol.production_steps {
            s.step()?;
            maximum_drift = maximum_drift.max((total_energy(&s, &masses) - e0).abs());
        }
        let e1 = total_energy(&s, &masses);
        convergence.push(serde_json::json!({
            "timestepFs": dt,
            "steps": s.state.step,
            "driftPerAtom": (e1 - e0) / n_atoms as f64,
            "maxDriftPerAtom": maximum_drift / n_atoms as f64,
        }));
    }
    // NVT statistics with a fixed seed (streams differ from OpenMM; compare stats).
    // Lengths are set by equilibration physics, not convenience: v-rescale
    // relaxes with tau = 1/friction = 1 ps, so 500 steps (1 ps) of
    // equilibration leaves a measurable cold transient from the minimized
    // start (observed: T climbing 241 -> 281 K across production with 1 ps
    // of equilibration). 4000 steps (8 ps = 8 tau) equilibrate: production
    // quarters then plateau instead of climbing. 4000 production steps
    // (8 ps, 400 frames) give block-averaged SEMs that resolve a genuine
    // thermostat bias from sampling noise.
    let nvt = SimulationProtocol {
        equilibration_ensemble: Ensemble::Nvt,
        production_ensemble: Ensemble::Nvt,
        thermostat: Thermostat::Langevin,
        equilibration_steps: nvt_equilibration_steps,
        production_steps: nvt_production_steps,
        save_every: (nvt_production_steps / 400).max(1),
        friction_per_ps: 1.0,
        ..base.clone()
    };
    let mut sim = ExplicitSimulation::new(&system, nvt.clone())?;
    // `advance` commits at most 100 steps per call; loop to the full segment.
    let mut temps = Vec::new();
    let mut pes = Vec::new();
    let mut kes = Vec::new();
    let mut produced = 0usize;
    let total = nvt.equilibration_steps + nvt.production_steps;
    while produced < total {
        let chunk = sim.advance(total - produced)?;
        if chunk.last_step == produced {
            break;
        }
        produced = chunk.last_step;
        // Production frames only: equilibration is not ensemble data, and
        // including the cold-start transient would bias the OpenMM comparison.
        for f in chunk
            .frames
            .iter()
            .filter(|f| f.step > nvt.equilibration_steps)
        {
            temps.push(f.temperature_k);
            pes.push(f.potential_energy);
            kes.push(f.kinetic_energy);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    // Optional constant-pressure leg used by the disposable OpenMM oracle.
    // Keeping it behind an environment switch avoids making the ordinary
    // force/NVE/NVT reference export pay for a second long trajectory.
    let npt_report = if std::env::var("GLYSYS_REF_NPT").is_ok() {
        let npt_warmup_steps = env_usize("GLYSYS_REF_NPT_WARMUP", 5000);
        let npt_equilibration_steps = env_usize("GLYSYS_REF_NPT_EQUILIBRATION", 5000);
        let npt_production_steps = env_usize("GLYSYS_REF_NPT_PRODUCTION", 10000);
        let npt_pressure_bar = env_f64("GLYSYS_REF_NPT_PRESSURE_BAR", 1.0);
        let npt_protocol = SimulationProtocol {
            solvent: SolventModel::Explicit,
            cutoff_angstrom: Some(9.0),
            rf_dielectric: Some(78.5),
            constraints: ConstraintModel::Settle,
            thermostat: Thermostat::Langevin,
            timestep_fs: 2.0,
            friction_per_ps: 1.0,
            pressure_bar: npt_pressure_bar,
            dispersion_correction: true,
            stages: Some(vec![
                glysys_dynamics::SimulationStage {
                    id: "nvt-warmup".into(),
                    ensemble: Ensemble::Nvt,
                    steps: npt_warmup_steps,
                    barostat_adaptation: false,
                },
                glysys_dynamics::SimulationStage {
                    id: "npt-equilibration".into(),
                    ensemble: Ensemble::Npt,
                    steps: npt_equilibration_steps,
                    barostat_adaptation: true,
                },
                glysys_dynamics::SimulationStage {
                    id: "production".into(),
                    ensemble: Ensemble::Npt,
                    steps: npt_production_steps,
                    barostat_adaptation: false,
                },
            ]),
            equilibration_steps: 0,
            production_steps: 0,
            save_every: env_usize(
                "GLYSYS_REF_NPT_SAVE_EVERY",
                (npt_production_steps / 200).max(1),
            ),
            minimization_iterations,
            seed: 11,
            ..Default::default()
        };
        let npt_dispersion_coefficient =
            PbcForceField::new(&system, vec![])?.dispersion_coefficient(9.0)?;
        let mut npt_sim = ExplicitSimulation::new(&system, npt_protocol.clone())?;
        let mut temperature = Vec::new();
        let mut potential = Vec::new();
        let mut volume = Vec::new();
        let mut density = Vec::new();
        let mut pressure = Vec::new();
        let total_steps = npt_protocol.total_steps();
        for _ in 0..total_steps {
            npt_sim.step()?;
            let frame = npt_sim.frame();
            if frame.segment == "production" && frame.step % npt_protocol.save_every == 0 {
                temperature.push(frame.temperature_k);
                potential.push(frame.potential_energy);
                volume.push(frame.box_angstrom.iter().product());
                density.push(frame.density_g_ml);
                pressure.push(frame.pressure_bar);
            }
        }
        Some(serde_json::json!({
            "pressureBar": npt_pressure_bar,
            "warmupSteps": npt_warmup_steps,
            "equilibrationSteps": npt_equilibration_steps,
            "productionSteps": npt_production_steps,
            "saveEvery": npt_protocol.save_every,
            "temperatureSeries": temperature,
            "potentialSeries": potential,
            "volumeSeries": volume,
            "densitySeries": density,
            "pressureSeries": pressure,
            "meanTemperatureK": mean(&temperature),
            "meanPotentialEnergy": mean(&potential),
            "meanVolumeA3": mean(&volume),
            "meanDensityGMl": mean(&density),
            "meanPressureBar": mean(&pressure),
            "barostatAttempts": npt_sim.state.barostat_attempts,
            "barostatAccepts": npt_sim.state.barostat_accepts,
            "barostatWidthA3": npt_sim.state.barostat_volume_width,
            "model": glysys_dynamics::EXPLICIT_NPT_MODEL_VERSION,
            "dispersionCorrection": true,
            "dispersionCoefficientKcalA3": npt_dispersion_coefficient,
        }))
    } else {
        None
    };
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "schemaVersion": 2,
            "fixture": "solvated-dipeptide",
            "model": "tip3p-rf-md-v1",
            "electrostatics": {"method": "reaction-field", "cutoffAngstrom": 9.0, "solventDielectric": 78.5},
            "constraints": "settle",
            "timestepFs": 2.0,
            "temperatureK": 300.0,
            "frictionPerPs": 1.0,
            "barostatInterval": 25,
            "boxAngstrom": system.box_angstrom(),
            "atomCount": n_atoms,
            "files": system.bundle_strings()?,
            "snapshotA": sim.state.reference_coordinates,
            "energyA": energy_a,
            "forcesA": forces_a,
            "velocitiesA": velocities_a,
            "componentsA": components_a,
            "virialA": virial_a,
            "virialTermsA": energy_probe_a.virial_terms,
            "virialPairSplitA": energy_probe_a.virial_pair_split,
            "snapshotB": snapshot_b,
            "energyB": energy_b,
            "forcesB": forces_b,
            "velocitiesB": velocities_b,
            "componentsB": components_b,
            "virialB": virial_b,
            "virialTermsB": energy_probe_b.virial_terms,
            "virialPairSplitB": energy_probe_b.virial_pair_split,
            "molecularPressureA": molecular_pressure_a,
            "molecularPressureB": molecular_pressure_b,
            "maxConstraintResidualA": max_constraint_residual,
            "maxVelocityConstraintResidualAps": max_velocity_constraint_residual,
            "nveDrift": drift,
            "nveTotalDrift": total_drift,
            "flexNve": {"timestepFs": 1.0, "totalDrift": flex_total},
            "nveConvergence": convergence,
            "nvt": {
                "seed": nvt.seed,
                "equilibrationSteps": nvt.equilibration_steps,
                "productionSteps": nvt.production_steps,
                "meanTemperatureK": mean(&temps),
                "meanPotentialEnergy": mean(&pes),
                "meanKineticEnergy": mean(&kes),
                "frames": temps.len(),
                "tempSeries": temps,
                "peSeries": pes,
                "keSeries": kes,
            },
            "npt": npt_report,
        }))?
    );
    Ok(())
}
