//! Emit matched coordinates, topology, components and derivatives for independent engines.
use glysys::{BuildOptions, SystemBuilder};
use glysys_energy::{EnergyEvaluator, EnergyOptions};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = std::env::args().nth(1).unwrap_or("dipeptide".into());
    let source = match fixture.as_str() {
        "dipeptide" => include_str!("../../../tests/fixtures/dipeptide.pdb"),
        "glycan" => include_str!("../../../tests/fixtures/glycan.pdb"),
        _ => return Err("unknown fixture".into()),
    };
    let system = SystemBuilder::new(BuildOptions {
        add_water: false,
        add_ions: false,
        ..Default::default()
    })?
    .prepare_pdb_str(source)?;
    let coordinates = system.coordinates();
    let mut evaluations = Vec::new();
    for obc in [false, true] {
        let options = EnergyOptions {
            obc2: obc.then(Default::default),
            ..Default::default()
        };
        let e = EnergyEvaluator::new(&system, options)?.energy_and_gradient(&coordinates)?;
        evaluations.push(
            serde_json::json!({"obc2":obc,"components":e.components,"gradients":e.gradients}),
        );
    }
    println!(
        "{}",
        serde_json::json!({"schemaVersion":1,"fixture":fixture,"modelVersion":"amber-glycam-v2","coordinates":coordinates,"exclusions":system.exclusions(),"oneFour":system.dihedrals().iter().filter(|t|!t.is_improper()).map(|t|serde_json::json!({"atoms":t.atoms(),"scee":t.electrostatic_14_scale(),"scnb":t.lennard_jones_14_scale()})).collect::<Vec<_>>(),"gbParameters":system.atoms().iter().map(|a|[a.charge(),a.gb_radius(),a.gb_screen()]).collect::<Vec<_>>(),"files":system.bundle_strings()?,"evaluations":evaluations})
    );
    Ok(())
}
