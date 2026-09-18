//! A force-free rotor isolates constraint/thermostat errors from intermolecular forces.
use glysys::{BuildOptions, ParameterizedSystem, SystemBuilder};
use glysys_dynamics::{
    ConstraintModel, Ensemble, SimulationProtocol, SolventModel, Thermostat,
    explicit::ExplicitSimulation,
};

fn water() -> ParameterizedSystem {
    let system = SystemBuilder::new(BuildOptions::default())
        .unwrap()
        .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))
        .unwrap();
    let mut snapshot: serde_json::Value =
        serde_json::from_str(&system.snapshot_json().unwrap()).unwrap();
    let residue = snapshot["system"]["residues"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "WAT")
        .unwrap()
        .clone();
    let first = residue["first_atom"].as_u64().unwrap() as usize;
    let mut atoms = snapshot["system"]["atoms"].as_array().unwrap()[first..first + 3].to_vec();
    for a in &mut atoms {
        a["residue"] = 0.into();
    }
    snapshot["system"]["atoms"] = atoms.into();
    for term in ["bonds", "angles", "dihedrals"] {
        let terms: Vec<_> = snapshot["system"][term]
            .as_array()
            .unwrap()
            .iter()
            .filter(|t| {
                t["atoms"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|i| (first..first + 3).contains(&(i.as_u64().unwrap() as usize)))
            })
            .map(|t| {
                let mut t = t.clone();
                for i in t["atoms"].as_array_mut().unwrap() {
                    *i = (i.as_u64().unwrap() - first as u64).into();
                }
                t
            })
            .collect();
        snapshot["system"][term] = terms.into();
    }
    let mut residue = residue;
    residue["first_atom"] = 0.into();
    residue["component"] = 0.into();
    snapshot["system"]["residues"] = serde_json::json!([residue]);
    snapshot["system"]["exclusions"] = serde_json::json!([[1, 2], [0, 2], [0, 1]]);
    for (name, value) in [
        ("component_count", 1),
        ("solute_atom_count", 0),
        ("water_residue_count", 1),
        ("sodium_count", 0),
        ("chloride_count", 0),
    ] {
        snapshot["system"][name] = value.into();
    }
    snapshot["system"]["box_angstrom"] = serde_json::json!([40., 40., 40.]);
    for (name, value) in [
        ("total_atoms", 3),
        ("residues", 1),
        ("solute_atoms", 0),
        ("waters", 1),
        ("sodium_ions", 0),
        ("chloride_ions", 0),
    ] {
        snapshot["report"][name] = value.into();
    }
    snapshot["report"]["box_angstrom"] = serde_json::json!([40., 40., 40.]);
    snapshot["metadata"] = serde_json::json!({"protein_chains":[],"glycan_trees":[],"glycosylation_sites":[],"residue_annotations":[]});
    ParameterizedSystem::from_snapshot_json(&snapshot.to_string()).unwrap()
}

#[test]
fn force_free_langevin_water_has_canonical_kinetic_energy() {
    let system = water();
    let protocol = SimulationProtocol {
        solvent: SolventModel::Explicit,
        constraints: ConstraintModel::Settle,
        thermostat: Thermostat::Langevin,
        equilibration_ensemble: Ensemble::Nvt,
        production_ensemble: Ensemble::Nvt,
        timestep_fs: 2.,
        friction_per_ps: 1.,
        temperature_k: 300.,
        minimization_iterations: 0,
        equilibration_steps: 10000,
        production_steps: 1000000,
        save_every: 100,
        cutoff_angstrom: Some(9.),
        seed: 11,
        ..Default::default()
    };
    let mut simulation = ExplicitSimulation::new(&system, protocol).unwrap();
    let mut sum = 0.;
    let mut count = 0;
    while simulation.state.step < 1010000 {
        let chunk = simulation.advance(100).unwrap();
        for frame in chunk.frames.into_iter().filter(|f| f.step > 10000) {
            sum += frame.kinetic_energy;
            count += 1;
        }
    }
    // Six unconstrained translational/rotational DOF, with COM thermalized.
    let expected = 3. * 0.00198720425864083 * 300.;
    let relative_error = (sum / count as f64 / expected - 1.).abs();
    eprintln!(
        "force-free water mean KE={} expected={expected} relative error={relative_error}",
        sum / count as f64
    );
    assert!(
        relative_error < 0.03,
        "force-free rotor thermostat bias {relative_error}"
    );
}
