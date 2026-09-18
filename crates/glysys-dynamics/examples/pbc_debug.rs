use glysys::{BuildOptions, SystemBuilder};
use glysys_dynamics::explicit::ExplicitSimulation;
use glysys_dynamics::{ConstraintModel, Ensemble, SimulationProtocol, SolventModel};
use glysys_energy::pbc::{BoxVectors, PbcForceField, PbcNeighborList, ReactionField};
fn main() {
    let pdb = include_str!("../../../tests/fixtures/dipeptide.pdb");
    let options = BuildOptions {
        add_water: true,
        add_ions: false,
        padding_angstrom: 9.0,
        ..Default::default()
    };
    let system = SystemBuilder::new(options)
        .unwrap()
        .prepare_pdb_str(pdb)
        .unwrap();
    println!("atoms: {}", system.atom_count());
    let protocol = SimulationProtocol {
        solvent: SolventModel::Explicit,
        equilibration_ensemble: Ensemble::Nve,
        production_ensemble: Ensemble::Nve,
        equilibration_steps: 0,
        production_steps: 200,
        save_every: 200,
        minimization_iterations: 200,
        timestep_fs: 1.0,
        friction_per_ps: 0.0,
        constraints: ConstraintModel::Settle,
        cutoff_angstrom: Some(9.0),
        seed: 11,
        ..Default::default()
    };
    let mut sim = ExplicitSimulation::new(&system, protocol).unwrap();
    for _ in 0..200 {
        sim.step().unwrap();
    }
    // Decompose the virial on the live state with a fresh list.
    let field = PbcForceField::new(&system, vec![]).unwrap();
    let rf = ReactionField::new(9.0, 78.5).unwrap();
    let boxv = BoxVectors::new(
        sim.state.box_angstrom[0],
        sim.state.box_angstrom[1],
        sim.state.box_angstrom[2],
    )
    .unwrap();
    let w: Vec<_> = sim
        .state
        .coordinates
        .iter()
        .map(|p| boxv.wrap(*p))
        .collect();
    let pairs = PbcNeighborList::build(&w, &boxv, 9.0, 1.5).unwrap();
    let e = field
        .evaluate(&sim.state.coordinates, &boxv, &pairs.pairs, &rf, 9.0)
        .unwrap();
    println!("virial_terms bond/angle/tors/pair: {:.1?}", e.virial_terms);
    println!("pair virial LJ/elec: {:.1?}", e.virial_pair_split);
    let c = e.components;
    println!(
        "E bond={:.1} angle={:.1} tors={:.1} vdw={:.1} elec={:.1} total={:.1}",
        c.bonds,
        c.angles,
        c.proper_torsions + c.improper_torsions,
        c.van_der_waals,
        c.electrostatics,
        c.total()
    );
    println!(
        "virial total: {:.1}, PE: {:.1}",
        e.virial,
        e.components.total()
    );
    println!("box: {:.2?} V={:.0}", boxv.as_array(), boxv.volume());
    // Unwrapped coordinate spread (diffusion check).
    let (mut lo, mut hi) = (1e9f64, -1e9f64);
    for p in &sim.state.coordinates {
        lo = lo.min(p.x).min(p.y).min(p.z);
        hi = hi.max(p.x).max(p.y).max(p.z);
    }
    println!("unwrapped coord range: [{lo:.1}, {hi:.1}]");
}
