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
//! 1.2), `--order N` (default 4), `--out FILE` (default stdout),
//! `--coordinates FILE` (JSON with a `coordinates` array of `[x, y, z]` in
//! angstrom replacing the snapshot's, e.g. a minimized configuration).
//!
//! `--timing N` instead times the dynamics path over N calls, on as many
//! threads as `RAYON_NUM_THREADS` allows: the cluster-pair kernel in
//! reaction-field and PME mode and the long-range PME evaluation, and
//! compares the single-precision cluster path with the f64 reference.
use glysys::{ParameterizedSystem, Vec3};
use glysys_energy::pbc::{BoxVectors, ElectrostaticsBackend, PbcForceField, PbcNeighborList};
use glysys_energy::pbc_cluster::ClusterPairEngine;
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
    coordinates: Option<String>,
    timing: usize,
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
        coordinates: None,
        timing: 0,
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
            "--coordinates" => options.coordinates = Some(value()?),
            "--timing" => options.timing = value()?.parse()?,
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

fn zeros(n: usize) -> Vec<Vec3> {
    vec![
        Vec3 {
            x: 0.,
            y: 0.,
            z: 0.
        };
        n
    ]
}

/// Median and minimum wall time of `call` in milliseconds over `repeats`
/// calls; the minimum is the least disturbed by other load on the machine.
fn time_ms(repeats: usize, mut call: impl FnMut()) -> serde_json::Value {
    let mut samples: Vec<f64> = (0..repeats.max(1))
        .map(|_| {
            let start = std::time::Instant::now();
            call();
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    serde_json::json!({"median": samples[samples.len() / 2], "min": samples[0]})
}

fn run_timing(
    system: &ParameterizedSystem,
    box_vec: &BoxVectors,
    coordinates: &[Vec3],
    cutoff: f64,
    parameters: PmeParameters,
    repeats: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = system.atom_count();
    let skin = glysys_energy::pbc::DEFAULT_SKIN_ANGSTROM;
    let alpha = parameters.alpha_per_angstrom;
    let mut reaction_field = ClusterPairEngine::new(
        system,
        cutoff,
        skin,
        glysys_energy::pbc::DEFAULT_RF_DIELECTRIC,
    )?;
    let mut ewald = ClusterPairEngine::new_pme(system, cutoff, skin, alpha)?;
    let mut pme = PmeEngine::new(system, parameters)?;
    let mut gradients = zeros(n);
    let mut timings = serde_json::Map::new();
    for (name, engine) in [
        ("clusterReactionField", &mut reaction_field),
        ("clusterPme", &mut ewald),
    ] {
        // The first call builds the pair lists; later calls reuse them.
        engine.evaluate_into(coordinates, box_vec, &mut gradients, true)?;
        let mut entry = serde_json::Map::new();
        for (key, observables) in [("forcesMs", false), ("observablesMs", true)] {
            let ms = time_ms(repeats, || {
                engine
                    .evaluate_into(coordinates, box_vec, &mut gradients, observables)
                    .unwrap();
            });
            entry.insert(key.into(), ms);
        }
        let rebuild = time_ms(repeats.div_ceil(10), || {
            engine.invalidate();
            engine
                .evaluate_into(coordinates, box_vec, &mut gradients, false)
                .unwrap();
        });
        entry.insert("forcesWithListRebuildMs".into(), rebuild);
        entry.insert("listedEntries".into(), engine.list_len().into());
        timings.insert(name.into(), entry.into());
    }
    pme.evaluate_into(coordinates, box_vec, &mut gradients, true)?;
    let mut entry = serde_json::Map::new();
    for (key, observables) in [("forcesMs", false), ("observablesMs", true)] {
        let ms = time_ms(repeats, || {
            pme.evaluate_into(coordinates, box_vec, &mut gradients, observables)
                .unwrap();
        });
        entry.insert(key.into(), ms);
    }
    timings.insert("pmeLongRange".into(), entry.into());

    // Single-precision cluster path against the f64 reference path.
    let field = PbcForceField::new(system, vec![])?;
    let wrapped: Vec<Vec3> = coordinates.iter().map(|p| box_vec.wrap(*p)).collect();
    let list = PbcNeighborList::build(&wrapped, box_vec, cutoff, 0.)?;
    let reference = field.evaluate_pme(&mut pme, coordinates, box_vec, &list.pairs, cutoff)?;
    let rest = field.evaluate(coordinates, box_vec, &list.pairs, &NoElectrostatics, cutoff)?;
    let fast = field.evaluate_with_cluster_pme(
        &mut ewald,
        &mut pme,
        coordinates,
        box_vec,
        0.,
        false,
        true,
    )?;
    let step = time_ms(repeats, || {
        field
            .evaluate_with_cluster_pme(&mut ewald, &mut pme, coordinates, box_vec, 0., false, false)
            .unwrap();
    });
    let (mut error2, mut total2, mut ewald2) = (0., 0., 0.);
    for ((f, r), o) in fast
        .gradients
        .iter()
        .zip(&reference.gradients)
        .zip(&rest.gradients)
    {
        for (got, want, other) in [(f.x, r.x, o.x), (f.y, r.y, o.y), (f.z, r.z, o.z)] {
            error2 += (got - want).powi(2);
            total2 += want * want;
            ewald2 += (want - other).powi(2);
        }
    }
    let report = serde_json::json!({
        "schemaVersion": 1,
        "atoms": n,
        "threads": rayon::current_num_threads(),
        "repeats": repeats,
        "cutoffAngstrom": cutoff,
        "skinAngstrom": skin,
        "pme": {
            "alphaPerAngstrom": alpha,
            "grid": parameters.grid,
            "interpolationOrder": parameters.interpolation_order,
        },
        "ewaldPairError": ewald.ewald_pair_error(),
        "timings": timings,
        "forceStepMs": step,
        "clusterVersusReference": {
            "energyDifference": fast.components.total() - reference.components.total(),
            "electrostaticsDifference":
                fast.components.electrostatics - reference.components.electrostatics,
            "virialDifference": fast.virial - reference.virial,
            "forceRmsDifference": (error2 / (3 * n) as f64).sqrt(),
            "forceRms": (total2 / (3 * n) as f64).sqrt(),
            "ewaldForceRms": (ewald2 / (3 * n) as f64).sqrt(),
        },
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options()?;
    let mut system =
        ParameterizedSystem::from_snapshot_json(&std::fs::read_to_string(&options.snapshot)?)?;
    if let Some(path) = &options.coordinates {
        let file: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let rows: Vec<[f64; 3]> = serde_json::from_value(file["coordinates"].clone())?;
        let replaced: Vec<Vec3> = rows
            .iter()
            .map(|p| Vec3 {
                x: p[0],
                y: p[1],
                z: p[2],
            })
            .collect();
        system.set_coordinates(&replaced)?;
    }
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

    if options.timing > 0 {
        return run_timing(
            &system,
            &box_vec,
            &coordinates,
            cutoff,
            parameters,
            options.timing,
        );
    }

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
