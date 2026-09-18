use glysys::{BuildOptions, ParameterizedSystem, SystemBuilder};

#[test]
fn snapshot_preserves_solvated_topology_and_rejects_invalid_indices() {
    let system = SystemBuilder::new(BuildOptions::default())
        .unwrap()
        .prepare_pdb_str(include_str!("fixtures/dipeptide.pdb"))
        .unwrap();
    let json = system.snapshot_json().unwrap();
    let restored = ParameterizedSystem::from_snapshot_json(&json).unwrap();
    assert_eq!(system.coordinates(), restored.coordinates());
    assert_eq!(system.box_angstrom(), restored.box_angstrom());
    assert_eq!(
        system.bundle_strings().unwrap(),
        restored.bundle_strings().unwrap()
    );
    assert!(restored.report.waters > 0);
    let mut corrupt: serde_json::Value = serde_json::from_str(&json).unwrap();
    corrupt["system"]["bonds"][0]["atoms"][0] = serde_json::json!(system.atom_count());
    assert!(ParameterizedSystem::from_snapshot_json(&corrupt.to_string()).is_err());
    let mut corrupt: serde_json::Value = serde_json::from_str(&json).unwrap();
    corrupt["version"] = serde_json::json!(2);
    assert!(ParameterizedSystem::from_snapshot_json(&corrupt.to_string()).is_err());
}
