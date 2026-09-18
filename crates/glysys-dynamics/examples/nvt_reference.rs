//! Longer independent NVT replicates from an existing, externally checked
//! reference snapshot. Reuses the runtime integrator and emits the same oracle
//! input schema; run openmm_pbc_rf.py INPUT --nvt-only on the output.
use glysys::{BuildOptions, SystemBuilder, Vec3};
use glysys_dynamics::{
    ConstraintModel, Ensemble, SimulationProtocol, SolventModel, Thermostat,
    explicit::ExplicitSimulation,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let mut input: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        args.get(1).ok_or("reference JSON required")?,
    )?)?;
    let seed: u64 = args
        .get(2)
        .ok_or("explicit replicate seed required")?
        .parse()?;
    let production: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(16000);
    let system = SystemBuilder::new(BuildOptions {
        add_water: true,
        add_ions: false,
        padding_angstrom: 9.,
        ..Default::default()
    })?
    .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))?;
    if input["files"]["system.prmtop"].as_str()
        != system
            .bundle_strings()?
            .get("system.prmtop")
            .map(String::as_str)
    {
        return Err("reference chemistry differs from current fixture preparation".into());
    }
    let coordinates: Vec<Vec3> = serde_json::from_value(input["snapshotA"].clone())?;
    let velocities: Vec<Vec3> = serde_json::from_value(input["velocitiesA"].clone())?;
    let forces: Vec<[f64; 3]> = serde_json::from_value(input["forcesA"].clone())?;
    let protocol = SimulationProtocol {
        solvent: SolventModel::Explicit,
        cutoff_angstrom: Some(9.),
        rf_dielectric: Some(78.5),
        constraints: ConstraintModel::Settle,
        thermostat: Thermostat::Langevin,
        equilibration_ensemble: Ensemble::Nvt,
        production_ensemble: Ensemble::Nvt,
        timestep_fs: 2.,
        temperature_k: 300.,
        friction_per_ps: 1.,
        minimization_iterations: 0,
        equilibration_steps: 4000,
        production_steps: production,
        save_every: (production / 400).max(1),
        seed,
        ..Default::default()
    };
    let mut initial =
        ExplicitSimulation::from_minimized(&system, protocol.clone(), coordinates.clone())?.state;
    // Restore the actual validated coordinates, velocities and starting forces,
    // rather than rerun minimization or project a different initial sample.
    initial.coordinates = coordinates;
    initial.velocities = velocities;
    initial.gradient = forces
        .iter()
        .map(|p| Vec3 {
            x: -p[0],
            y: -p[1],
            z: -p[2],
        })
        .collect();
    initial.potential_energy = input["energyA"].as_f64().ok_or("missing energy")?;
    let mut simulation = ExplicitSimulation::restore(&system, initial)?;
    let mut temperatures = Vec::new();
    let mut potential = Vec::new();
    let mut kinetic = Vec::new();
    let total = protocol.equilibration_steps + protocol.production_steps;
    while simulation.state.step < total {
        let chunk = simulation.advance(total - simulation.state.step)?;
        for frame in chunk
            .frames
            .into_iter()
            .filter(|f| f.step > protocol.equilibration_steps)
        {
            temperatures.push(frame.temperature_k);
            potential.push(frame.potential_energy);
            kinetic.push(frame.kinetic_energy);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    input["nvt"] = serde_json::json!({"seed":seed,"equilibrationSteps":protocol.equilibration_steps,"productionSteps":production,
        "meanTemperatureK":mean(&temperatures),"meanPotentialEnergy":mean(&potential),"meanKineticEnergy":mean(&kinetic),
        "frames":temperatures.len(),"tempSeries":temperatures,"peSeries":potential,"keSeries":kinetic});
    println!("{}", serde_json::to_string(&input)?);
    Ok(())
}
