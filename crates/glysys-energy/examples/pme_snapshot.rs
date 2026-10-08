//! Evaluate a prepared periodic system with particle-mesh Ewald on the f64
//! reference path and write energies, the virial and all gradients as JSON,
//! for comparison with an independent engine (`benchmarks/openmm_pme.py`).
//!
//! ```sh
//! cargo run --release -p glysys-energy --example pme_snapshot -- \
//!     system.snapshot.json --order 5 --out glysys_pme.json
//! ```
//!
//! Options: `--cutoff A` (default 9), `--alpha 1/A` or `--rtol X` (default
//! 1e-5, the GROMACS rule), `--grid NX,NY,NZ` or `--spacing A` (default
//! 1.2), `--order N` (default 4), `--out FILE` (default stdout).
use glysys::{ParameterizedSystem, Vec3};
use glysys_energy::pbc::{BoxVectors, ElectrostaticsBackend, PbcForceField, PbcNeighborList};
use glysys_energy::pme::{
    DEFAULT_EWALD_RTOL, DEFAULT_FOURIER_SPACING_ANGSTROM, DEFAULT_INTERPOLATION_ORDER, PmeEngine,
    PmeParameters, ewald_coefficient,
};

/// Bonded terms, Lennard-Jones and the plain 1-4 Coulomb exceptions only.
struct NoElectrostatics;

impl ElectrostaticsBackend for NoElectrostatics {
    fn name(&self) -> &'static str {
        "none"
    }

    fn pair(&self, _r: f64, _qq: f64) -> (f64, f64) {
        (0., 0.)
    }
}

struct Options {
    snapshot: String,
    cutoff: f64,
    alpha: Option<f64>,
    rtol: f64,
    grid: Option<[usize; 3]>,
    spacing: f64,
    order: usize,
    out: Option<String>,
}

fn parse_options() -> Result<Options, Box<dyn std::error::Error>> {
    let mut options = Options {
        snapshot: String::new(),
        cutoff: 9.0,
        alpha: None,
        rtol: DEFAULT_EWALD_RTOL,
        grid: None,
        spacing: DEFAULT_FOURIER_SPACING_ANGSTROM,
        order: DEFAULT_INTERPOLATION_ORDER,
        out: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--cutoff" => options.cutoff = value()?.parse()?,
            "--alpha" => options.alpha = Some(value()?.parse()?),
            "--rtol" => options.rtol = value()?.parse()?,
            "--spacing" => options.spacing = value()?.parse()?,
            "--order" => options.order = value()?.parse()?,
            "--out" => options.out = Some(value()?),
            "--grid" => {
                let points: Vec<usize> = value()?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
                options.grid = Some(points.try_into().map_err(|_| "--grid needs NX,NY,NZ")?);
            }
            _ if options.snapshot.is_empty() && !arg.starts_with("--") => options.snapshot = arg,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if options.snapshot.is_empty() {
        return Err("usage: pme_snapshot SNAPSHOT.json [options]".into());
    }
    Ok(options)
}

fn rows(values: &[Vec3]) -> Vec<[f64; 3]> {
    values.iter().map(|p| [p.x, p.y, p.z]).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options()?;
    let system =
        ParameterizedSystem::from_snapshot_json(&std::fs::read_to_string(&options.snapshot)?)?;
    let box_vec = BoxVectors::from_system(&system)?;
    let coordinates = system.coordinates();
    let cutoff = options.cutoff;
    let mut parameters = PmeParameters::for_box(
        &box_vec,
        cutoff,
        options.rtol,
        options.spacing,
        options.order,
    )?;
    if let Some(alpha) = options.alpha {
        parameters.alpha_per_angstrom = alpha;
    }
    if let Some(grid) = options.grid {
        parameters.grid = grid;
    }
    let parameters = PmeParameters::new(
        parameters.alpha_per_angstrom,
        parameters.grid,
        parameters.interpolation_order,
    )?;

    let field = PbcForceField::new(&system, vec![])?;
    let mut engine = PmeEngine::new(&system, parameters)?;
    let wrapped: Vec<Vec3> = coordinates.iter().map(|p| box_vec.wrap(*p)).collect();
    let list = PbcNeighborList::build(&wrapped, &box_vec, cutoff, 0.)?;
    let full = field.evaluate_pme(&mut engine, &coordinates, &box_vec, &list.pairs, cutoff)?;
    // The same evaluation without regular-pair electrostatics isolates the
    // part that depends on the Ewald settings (direct space + long range).
    let rest = field.evaluate(
        &coordinates,
        &box_vec,
        &list.pairs,
        &NoElectrostatics,
        cutoff,
    )?;
    let ewald_gradients: Vec<[f64; 3]> = full
        .gradients
        .iter()
        .zip(&rest.gradients)
        .map(|(g, r)| [g.x - r.x, g.y - r.y, g.z - r.z])
        .collect();
    let terms = engine.last_terms();
    let ewald_energy = full.components.electrostatics - rest.components.electrostatics;
    let report = serde_json::json!({
        "schemaVersion": 1,
        "snapshot": options.snapshot,
        "atoms": system.atom_count(),
        "boxAngstrom": box_vec.as_array(),
        "cutoffAngstrom": cutoff,
        "pme": {
            "alphaPerAngstrom": parameters.alpha_per_angstrom,
            "grid": parameters.grid,
            "interpolationOrder": parameters.interpolation_order,
            "ewaldRtol": glysys_energy::pme::erfc(parameters.alpha_per_angstrom * cutoff),
            "gromacsAlphaPerAngstrom": ewald_coefficient(cutoff, options.rtol)?,
            "excludedPairs": engine.excluded_pair_count(),
        },
        "units": {"energy": "kcal/mol", "length": "angstrom", "gradients": "dE/dx"},
        "components": full.components,
        "total": full.components.total(),
        "longRange": terms,
        // Regular-pair direct space plus long range: everything but the
        // plain 1-4 Coulomb exceptions that `components.electrostatics` holds.
        "ewaldEnergy": ewald_energy,
        "directEnergy": ewald_energy
            - (terms.reciprocal + terms.self_energy + terms.excluded_pairs + terms.background),
        "virial": full.virial,
        "virialTerms": full.virial_terms,
        "virialPairSplit": full.virial_pair_split,
        "coordinates": rows(&coordinates),
        "gradients": rows(&full.gradients),
        "ewaldGradients": ewald_gradients,
    });
    let text = serde_json::to_string(&report)?;
    match &options.out {
        Some(path) => std::fs::write(path, text)?,
        None => println!("{text}"),
    }
    eprintln!(
        "{} atoms, cutoff {cutoff} A, alpha {:.9} 1/A, grid {:?}, order {}: total {:.6} kcal/mol, \
         electrostatics {:.6} (long range {:.6})",
        system.atom_count(),
        parameters.alpha_per_angstrom,
        parameters.grid,
        parameters.interpolation_order,
        full.components.total(),
        full.components.electrostatics,
        terms.reciprocal + terms.self_energy + terms.excluded_pairs + terms.background,
    );
    Ok(())
}
