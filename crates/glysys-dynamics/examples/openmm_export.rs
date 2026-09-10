use glysys::{BuildOptions, SystemBuilder};
use glysys_dynamics::{CpuSimulation, SimulationProtocol, normal_noise};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        ..Default::default()
    })?
    .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))?;
    let mut cases = Vec::new();
    for friction in [0., 1.] {
        let mut simulation = CpuSimulation::new(
            &system,
            SimulationProtocol {
                minimization_iterations: 30,
                equilibration_steps: 0,
                production_steps: 20,
                friction_per_ps: friction,
                ..Default::default()
            },
        )?;
        let initial = simulation.state.clone();
        let mut rng = initial.rng_state;
        let noise = normal_noise(&mut rng, system.atom_count() * 20);
        simulation.advance(20)?;
        cases.push(serde_json::json!({"initial":initial,"final":simulation.state,"noise":noise}));
    }
    let gb: Vec<_> = system
        .atoms()
        .iter()
        .map(|a| [a.charge(), a.gb_radius(), a.gb_screen()])
        .collect();
    println!(
        "{}",
        serde_json::to_string(
            &serde_json::json!({"schemaVersion":1,"fixture":"dipeptide","files":system.bundle_strings()?,"gbParameters":gb,"cases":cases})
        )?
    );
    Ok(())
}
