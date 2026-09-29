//! Print AM1 Mulliken and AM1-BCC charges for CCD definitions (validation aid).
//!
//! Usage: cargo run --release --example ligand_charges -- HEM.cif [more.cif ...]
use glysys::{ComponentLibrary, SmallMoleculeForceField, am1_mulliken, molecule_from_component};

fn main() {
    let force_field = SmallMoleculeForceField::sage().expect("Sage force field");
    let mut out = serde_json::Map::new();
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).expect("read CIF");
        let mut library = ComponentLibrary::new();
        let ids = library.add_cif(&text).expect("parse CIF");
        for id in ids {
            let component = library.get(&id).unwrap();
            let entry = match molecule_from_component(component) {
                Ok(molecule) => {
                    let elements = molecule.atoms.iter().map(|a| a.element).collect::<Vec<_>>();
                    let positions = molecule
                        .atoms
                        .iter()
                        .map(|a| a.position)
                        .collect::<Vec<_>>();
                    let charge = molecule.atoms.iter().map(|a| a.formal_charge).sum::<i32>();
                    let started = std::time::Instant::now();
                    let am1 = am1_mulliken(&elements, &positions, charge);
                    let t_am1 = started.elapsed().as_secs_f64();
                    let bcc = force_field.am1bcc_charges(&molecule);
                    let t_bcc = started.elapsed().as_secs_f64();
                    let assignment = force_field.assign(&molecule).map(|a| {
                        serde_json::json!({
                            "bonds": a.bonds.iter().map(|b| (b.0.to_vec(), b.3.clone())).collect::<Vec<_>>(),
                            "angles": a.angles.iter().map(|b| (b.0.to_vec(), b.3.clone())).collect::<Vec<_>>(),
                            "torsions": a.torsions.iter().map(|t| (t.0.to_vec(), t.5.clone(), t.4)).collect::<Vec<_>>(),
                            "vdw": a.vdw.iter().map(|v| v.2.clone()).collect::<Vec<_>>(),
                        })
                    }).map_err(|e| e.0);
                    eprintln!(
                        "{id}: am1 {t_am1:.2}s bcc {:.2}s assign {:.2}s",
                        t_bcc - t_am1,
                        started.elapsed().as_secs_f64() - t_bcc
                    );
                    let names = molecule
                        .atoms
                        .iter()
                        .map(|a| a.name.clone())
                        .collect::<Vec<_>>();
                    serde_json::json!({
                        "names": names,
                        "charge": charge,
                        "am1": am1.as_ref().map(|(q, _)| q.clone()).map_err(|e| e.0.clone()),
                        "hof": am1.as_ref().map(|(_, h)| *h).ok(),
                        "am1bcc": bcc.map_err(|e| e.0),
                        "sage": assignment,
                        "seconds": started.elapsed().as_secs_f64(),
                    })
                }
                Err(error) => serde_json::json!({ "error": error.0 }),
            };
            out.insert(id, entry);
        }
    }
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
