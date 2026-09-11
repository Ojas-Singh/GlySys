//! Export a solvated dipeptide PBC reference case for `benchmarks/openmm_pbc_rf.py`.
//!
//! Snapshots carry our minimized coordinates, post-NVE coordinates, NVE drift
//! series, and NVT ensemble statistics. The script rebuilds the identical
//! Amber chemistry from the exported prmtop and checks energies, forces,
//! drift bounds, and ensemble statistics on OpenMM's Reference platform.
use glysys::{BuildOptions, SystemBuilder};
use glysys_dynamics::explicit::ExplicitSimulation;
use glysys_dynamics::{ConstraintModel, Ensemble, SimulationProtocol, SolventModel};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let system = SystemBuilder::new(BuildOptions {
        add_water: true,
        add_ions: false,
        padding_angstrom: 9.0,
        ..Default::default()
    })?
    .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))?;
    let base = SimulationProtocol {
        solvent: SolventModel::Explicit,
        cutoff_angstrom: Some(9.0),
        rf_dielectric: Some(78.5),
        constraints: ConstraintModel::Settle,
        timestep_fs: 2.0,
        minimization_iterations: 200,
        seed: 11,
        ..Default::default()
    };
    // Snapshot A: minimized geometry for single-point parity.
    let nve = SimulationProtocol {
        equilibration_ensemble: Ensemble::Nve,
        production_ensemble: Ensemble::Nve,
        equilibration_steps: 0,
        production_steps: 200,
        save_every: 10,
        friction_per_ps: 0.0,
        ..base.clone()
    };
    let mut sim = ExplicitSimulation::new(&system, nve)?;
    let energy_a = sim.state.potential_energy;
    let forces_a: Vec<[f64; 3]> = sim
        .state
        .gradient
        .iter()
        .map(|g| [-g.x, -g.y, -g.z])
        .collect();
    let mut drift = vec![sim.state.potential_energy];
    for i in 0..200 {
        sim.step()?;
        if (i + 1) % 10 == 0 {
            drift.push(sim.state.potential_energy);
        }
    }
    let snapshot_b = sim.state.coordinates.clone();
    let energy_b = sim.state.potential_energy;
    let forces_b: Vec<[f64; 3]> = sim
        .state
        .gradient
        .iter()
        .map(|g| [-g.x, -g.y, -g.z])
        .collect();
    // NVT statistics with a fixed seed (streams differ from OpenMM; compare stats).
    let nvt = SimulationProtocol {
        equilibration_ensemble: Ensemble::Nvt,
        production_ensemble: Ensemble::Nvt,
        equilibration_steps: 500,
        production_steps: 2000,
        save_every: 10,
        friction_per_ps: 1.0,
        ..base.clone()
    };
    let mut sim = ExplicitSimulation::new(&system, nvt)?;
    // `advance` commits at most 100 steps per call; loop to the full segment.
    let mut temps = Vec::new();
    let mut pes = Vec::new();
    let mut produced = 0usize;
    while produced < 2500 {
        let chunk = sim.advance(2500 - produced)?;
        if chunk.last_step == produced {
            break;
        }
        produced = chunk.last_step;
        // Production frames only: equilibration is not ensemble data, and
        // including the cold-start transient would bias the OpenMM comparison.
        temps.extend(
            chunk
                .frames
                .iter()
                .filter(|f| f.step > 500)
                .map(|f| f.temperature_k),
        );
        pes.extend(
            chunk
                .frames
                .iter()
                .filter(|f| f.step > 500)
                .map(|f| f.potential_energy),
        );
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "schemaVersion": 1,
            "fixture": "solvated-dipeptide",
            "model": "tip3p-rf-md-v1",
            "electrostatics": {"method": "reaction-field", "cutoffAngstrom": 9.0, "solventDielectric": 78.5},
            "constraints": "settle",
            "timestepFs": 2.0,
            "temperatureK": 300.0,
            "frictionPerPs": 1.0,
            "files": system.bundle_strings()?,
            "snapshotA": sim.state.reference_coordinates,
            "energyA": energy_a,
            "forcesA": forces_a,
            "snapshotB": snapshot_b,
            "energyB": energy_b,
            "forcesB": forces_b,
            "nveDrift": drift,
            "nvt": {"meanTemperatureK": mean(&temps), "meanPotentialEnergy": mean(&pes), "frames": temps.len()},
        }))?
    );
    Ok(())
}
