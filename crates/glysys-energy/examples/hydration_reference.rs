use glysys::{BuildOptions, SystemBuilder, Vec3, read_pdb_str};
use glysys_energy::hydration::{HydrationProvider, HydrationRequest, PhysicalProbe};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pdb = include_str!("../../../benchmarks/hydration/1UBQ.pdb");
    let options = BuildOptions {
        add_water: false,
        add_ions: false,
        ..Default::default()
    };
    let raw = read_pdb_str(pdb, &options)?;
    let waters: Vec<_> = raw
        .atoms()
        .into_iter()
        .filter(|a| a.residue_name == "HOH" && a.element == "O")
        .map(|a| a.position)
        .collect();
    let mut receptor = raw.clone();
    receptor.remove_residues_named(&["HOH"]);
    let system = SystemBuilder::new(options)?.prepare_structure(&receptor)?;
    let probe = PhysicalProbe::new(&system)?;
    // Fixed region selected before site prediction, including a deposited water.
    let o = waters[0];
    let mut output = Vec::new();
    for spacing in [1., 0.5] {
        let request = HydrationRequest {
            minimum: Vec3 {
                x: o.x - 2.,
                y: o.y - 2.,
                z: o.z - 2.,
            },
            maximum: Vec3 {
                x: o.x + 2.,
                y: o.y + 2.,
                z: o.z + 2.,
            },
            spacing,
            orientations: 96,
            max_sites: 10,
            cutoff: None,
            method: None,
            chemical_potential: None,
            gc_steps: None,
            gc_seed: None,
        };
        let started = std::time::Instant::now();
        let field = probe.predict(&request)?;
        let distances: Vec<_> = field
            .sites
            .iter()
            .map(|s| {
                waters
                    .iter()
                    .map(|w| {
                        ((s.position.x - w.x).powi(2)
                            + (s.position.y - w.y).powi(2)
                            + (s.position.z - w.z).powi(2))
                        .sqrt()
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        output.push(serde_json::json!({"spacingAngstrom":spacing,"seconds":started.elapsed().as_secs_f64(),"sites":field.sites.len(),"nearestDepositedWaterAngstrom":distances,"field":field}));
    }
    println!(
        "{}",
        serde_json::to_string(
            &serde_json::json!({"schemaVersion":1,"structure":"1UBQ","selection":"4 angstrom box about first deposited water; location-informed validation, not blind whole-protein prediction","results":output})
        )?
    );
    Ok(())
}
