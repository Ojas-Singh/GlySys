//! Canonical PBC validation: the same `pbc.wgsl` kernels the browser will
//! run, exercised natively (here on lavapipe software Vulkan) against two
//! independent CPU oracles.
//!
//! Oracles: (a) the runtime `PbcNeighborList`/`PbcForceField` (validates the
//! GPU math), and (b) a harness-local exact O(N^2) reference with hand-written
//! Amber formulas (validates the CPU list end to end and guards against the
//! list and the kernels sharing one cutoff bug). Nothing here is
//! performance evidence; software execution proves shader/layout/dispatch,
//! PBC, reduction, and determinism correctness only.
//!
//! Environment: default wgpu adapter selection (lavapipe `lvp` on headless
//! CI; any real adapter also works for these correctness tests). For a
//! forced software device: `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`.
use glysys::{BuildOptions, ParameterizedSystem, SystemBuilder, Vec3};
use glysys_dynamics::explicit::{ExplicitSimulation, kinetic_energy};
use glysys_dynamics::{ConstraintModel, Ensemble, SimulationProtocol, SolventModel};
use glysys_energy::pbc::{
    BoxVectors, NonbondedElectrostatics, PbcForceField, PbcNeighborList, ReactionField,
    classify_waters, water_equilibrium,
};
use glysys_gpu::pbc::{PbcPacking, ResidentPbc};
use glysys_gpu::{GpuContext, GpuContextOptions};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, OnceLock};

const COULOMB: f64 = 332.063_713_299;

// lavapipe is stable for the individual kernels, but concurrent wgpu device
// teardown in one integration binary can race inside the software Vulkan
// driver.  Serialize device-owning tests while keeping the production code
// and browser execution fully concurrent.
static GPU_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn gpu_test_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU_TEST_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("GPU test lock poisoned")
}

#[path = "support/neighbor_reuse.rs"]
mod neighbor_reuse;

// ---------------------------------------------------------------------------
// Exact O(N^2) reference (independent formulas, deterministic index order).
// ---------------------------------------------------------------------------

struct RefParams {
    q: Vec<f64>,
    sig: Vec<f64>,
    eps: Vec<f64>,
    excl: Vec<BTreeSet<usize>>,
    one_four: BTreeMap<(usize, usize), (f64, f64)>,
}

fn ref_params(system: &ParameterizedSystem) -> RefParams {
    let q = system.atoms().iter().map(|a| a.charge()).collect();
    let sig = system
        .atoms()
        .iter()
        .map(|a| a.lennard_jones_radius())
        .collect();
    let eps = system
        .atoms()
        .iter()
        .map(|a| a.lennard_jones_epsilon())
        .collect();
    let excl = system.exclusions().to_vec();
    let mut one_four = BTreeMap::new();
    for t in system.dihedrals().iter().filter(|t| !t.is_improper()) {
        let ids = t.atoms();
        let key = (ids[0].min(ids[3]), ids[0].max(ids[3]));
        one_four
            .entry(key)
            .or_insert((t.electrostatic_14_scale(), t.lennard_jones_14_scale()));
    }
    RefParams {
        q,
        sig,
        eps,
        excl,
        one_four,
    }
}

fn min_image(a: Vec3, b: Vec3, box_xyz: &[f64; 3]) -> Vec3 {
    Vec3 {
        x: a.x - b.x - box_xyz[0] * ((a.x - b.x) / box_xyz[0]).round(),
        y: a.y - b.y - box_xyz[1] * ((a.y - b.y) / box_xyz[1]).round(),
        z: a.z - b.z - box_xyz[2] * ((a.z - b.z) / box_xyz[2]).round(),
    }
}

fn wrap(p: Vec3, box_xyz: &[f64; 3]) -> Vec3 {
    Vec3 {
        x: p.x - box_xyz[0] * (p.x / box_xyz[0]).floor(),
        y: p.y - box_xyz[1] * (p.y / box_xyz[1]).floor(),
        z: p.z - box_xyz[2] * (p.z / box_xyz[2]).floor(),
    }
}

/// All i<j pairs within `limit` (exact, no cells).
fn exact_pairs(coords: &[Vec3], box_xyz: &[f64; 3], limit: f64) -> Vec<(u32, u32)> {
    let w: Vec<Vec3> = coords.iter().map(|p| wrap(*p, box_xyz)).collect();
    let mut out = Vec::new();
    for i in 0..w.len() {
        for j in (i + 1)..w.len() {
            let d = min_image(w[i], w[j], box_xyz);
            if d.x * d.x + d.y * d.y + d.z * d.z <= limit * limit {
                out.push((i as u32, j as u32));
            }
        }
    }
    out
}

/// Exact LJ+RF energy and gradients over explicit pairs (hand formulas).
fn exact_energy(
    coords: &[Vec3],
    box_xyz: &[f64; 3],
    pairs: &[(u32, u32)],
    cutoff: f64,
    krf: f64,
    crf: f64,
    rp: &RefParams,
) -> (f64, f64, Vec<[f64; 3]>) {
    let w: Vec<Vec3> = coords.iter().map(|p| wrap(*p, box_xyz)).collect();
    let mut grad = vec![[0.; 3]; w.len()];
    let (mut lj_tot, mut rf_tot) = (0., 0.);
    for &(a, b) in pairs {
        let (a, b) = (a as usize, b as usize);
        let key = (a.min(b), a.max(b));
        let scale = rp.one_four.get(&key).copied();
        if rp.excl[a].contains(&b) && scale.is_none() {
            continue;
        }
        let (scee, scnb) = scale.unwrap_or((1., 1.));
        let d = min_image(w[a], w[b], box_xyz);
        let r2 = d.x * d.x + d.y * d.y + d.z * d.z;
        if r2 > cutoff * cutoff {
            continue;
        }
        let r = r2.sqrt().max(1e-8);
        let sig = rp.sig[a] + rp.sig[b];
        let eps = (rp.eps[a] * rp.eps[b]).sqrt() / scnb;
        let u = (sig / r).powi(6);
        lj_tot += eps * (u * u - 2. * u);
        let qq = COULOMB * rp.q[a] * rp.q[b] / scee;
        // 1-4 exceptions bypass reaction-field screening (OpenMM convention:
        // plain Coulomb), exactly like the runtime engine under test.
        let (ecoul, dcoul_dr) = if scale.is_some() {
            (qq / r, -qq / (r * r))
        } else {
            (
                qq * (1. / r + krf * r * r - crf),
                qq * (-1. / (r * r) + 2. * krf * r),
            )
        };
        rf_tot += ecoul;
        let flj = 12. * eps * (u - u * u) / r;
        let fmag = (flj + dcoul_dr) / r;
        for (k, (atom, s)) in [(a, 1.), (b, -1.)].iter().enumerate() {
            let _ = k;
            grad[*atom][0] += s * fmag * d.x;
            grad[*atom][1] += s * fmag * d.y;
            grad[*atom][2] += s * fmag * d.z;
        }
    }
    (lj_tot, rf_tot, grad)
}

// ---------------------------------------------------------------------------
// Fixtures and drivers.
// ---------------------------------------------------------------------------

fn solvated_system(padding: f64) -> ParameterizedSystem {
    solvated_system_with_ions(padding, false)
}

fn solvated_system_with_ions(padding: f64, add_ions: bool) -> ParameterizedSystem {
    SystemBuilder::new(BuildOptions {
        add_water: true,
        add_ions,
        padding_angstrom: padding,
        ..Default::default()
    })
    .unwrap()
    .prepare_pdb_str(include_str!("../../../tests/fixtures/dipeptide.pdb"))
    .unwrap()
}

/// Minimized coordinates through the production dynamics path (LBFGS).
/// The raw builder output contains overlapping atoms (individual |LJ| terms
/// up to ~5e4 kcal/mol cancelling to ~1e3); no f32 engine can sum those
/// precisely, and production always minimizes first. Precision legs therefore
/// run on minimized states; the raw state keeps a finiteness-only leg.
fn minimized_coords(system: &ParameterizedSystem) -> Vec<Vec3> {
    let protocol = SimulationProtocol {
        solvent: SolventModel::Explicit,
        constraints: ConstraintModel::None,
        timestep_fs: 1.0,
        minimization_iterations: 100,
        equilibration_ensemble: Ensemble::Nve,
        production_ensemble: Ensemble::Nve,
        equilibration_steps: 0,
        production_steps: 1,
        save_every: 1,
        friction_per_ps: 0.0,
        seed: 7,
        cutoff_angstrom: Some(4.0),
        rf_dielectric: Some(78.5),
        ..Default::default()
    };
    let sim = ExplicitSimulation::new(system, protocol).unwrap();
    sim.state.coordinates.clone()
}

fn xorshift(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

/// Deterministic pseudo-random coordinates in the box (topology untouched;
/// only nonbonded paths are exercised).
fn random_coords(n: usize, box_xyz: [f64; 3], seed: u64) -> Vec<Vec3> {
    let mut s = seed;
    (0..n)
        .map(|_| Vec3 {
            x: xorshift(&mut s) * box_xyz[0],
            y: xorshift(&mut s) * box_xyz[1],
            z: xorshift(&mut s) * box_xyz[2],
        })
        .collect()
}

struct GpuCtx {
    gpu: ResidentPbc,
    #[allow(dead_code)]
    packing: PbcPacking,
    box_vec: BoxVectors,
    box_f32: [f32; 3],
    max_pairs: u32,
}

async fn setup(system: &ParameterizedSystem, cutoff: f64, skin: f64) -> GpuCtx {
    let packing = PbcPacking::new(system, cutoff, skin).unwrap();
    let backend = NonbondedElectrostatics::ReactionField {
        cutoff_angstrom: cutoff,
        solvent_dielectric: 78.5,
    };
    // Generous pair bound for validation sizes; overflow is asserted absent.
    let max_pairs = (system.atom_count() as u32 * 256).max(4096);
    let context = GpuContext::new(GpuContextOptions::default()).await.unwrap();
    let gpu = ResidentPbc::with_context(&context, &packing, &backend, max_pairs)
        .await
        .unwrap();
    let box_vec = BoxVectors::from_system(system).unwrap();
    let b = box_vec.as_array();
    GpuCtx {
        gpu,
        packing,
        box_vec,
        box_f32: [b[0] as f32, b[1] as f32, b[2] as f32],
        max_pairs,
    }
}

#[test]
fn pbc_packing_uses_the_supplied_box_for_cell_dimensions() {
    let system = solvated_system(6.0);
    let compact = PbcPacking::new_with_box(&system, [22.0, 22.0, 22.0], 4.0, 1.5).unwrap();
    let expanded = PbcPacking::new_with_box(&system, [28.0, 28.0, 28.0], 4.0, 1.5).unwrap();
    assert_eq!(compact.dims, [system.atom_count() as u32, 4, 4, 4]);
    assert_eq!(expanded.dims, [system.atom_count() as u32, 5, 5, 5]);
    assert_eq!(compact.params, expanded.params);
    assert_eq!(compact.specials.len(), expanded.specials.len());
    assert!(
        compact
            .specials
            .iter()
            .zip(&expanded.specials)
            .all(|(a, b)| {
                a.other == b.other && a.scee == b.scee && a.scnb == b.scnb && a.spare == b.spare
            })
    );
    assert_eq!(compact.bonded, expanded.bonded);
}

fn cpu_list(coords: &[Vec3], box_vec: &BoxVectors, cutoff: f64, skin: f64) -> Vec<(u32, u32)> {
    let wrapped: Vec<Vec3> = coords.iter().map(|p| box_vec.wrap(*p)).collect();
    let list = PbcNeighborList::build(&wrapped, box_vec, cutoff, skin).unwrap();
    let mut pairs: Vec<(u32, u32)> = list
        .pairs
        .iter()
        .map(|&(a, b)| (a as u32, b as u32))
        .collect();
    pairs.sort_unstable();
    pairs
}

fn check_pairs_exact(got: &[(u32, u32)], want: &[(u32, u32)], tag: &str) {
    let mut got = got.to_vec();
    got.sort_unstable();
    assert_eq!(
        got.len(),
        want.len(),
        "{tag}: pair count {} != {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "{tag}: pair {i} differs: {g:?} != {w:?}");
    }
}

fn e_tol(expected: f64) -> f64 {
    1e-3 + 1e-4 * expected.abs()
}

/// Full-force f32 tolerance. CPU-vs-exact checks remain near machine
/// precision; this accounts for GPU arithmetic in bonded plus nonbonded
/// accumulation and rejects sign/topology-class errors decisively.
fn gpu_g_tol(expected: f64) -> f64 {
    0.05 + 0.003 * expected.abs()
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[test]
fn pbc_neighbor_list_matches_cpu_exact() {
    pollster::block_on(async {
        let _guard = gpu_test_guard();
        // Realistic fixture plus randomized coordinate sets on the same
        // topology (cells, wrap, and stencil paths all vary).
        let system = solvated_system(6.0);
        let cutoff = 4.0;
        let skin = 1.5;
        let ctx = setup(&system, cutoff, skin).await;
        let box_arr = ctx.box_vec.as_array();
        let cases: Vec<(&str, Vec<Vec3>)> = vec![
            ("prepared", system.coordinates()),
            ("random7", random_coords(system.atom_count(), box_arr, 7)),
            ("random99", random_coords(system.atom_count(), box_arr, 99)),
        ];
        for (tag, coords) in cases {
            ctx.gpu.set_coordinates(&coords, ctx.box_f32, false);
            let gpu = ctx.gpu.neighbor_list().await.unwrap();
            assert!(
                gpu.count <= ctx.max_pairs,
                "{tag}: pair overflow {}/max",
                gpu.count
            );
            let want = cpu_list(&coords, &ctx.box_vec, cutoff, skin)
                .into_iter()
                .collect::<Vec<_>>();
            check_pairs_exact(&gpu.pairs, &want, tag);
            // Cross-check the CPU list itself against the exact reference.
            let exact = exact_pairs(&coords, &box_arr, cutoff + skin)
                .into_iter()
                .collect::<Vec<_>>();
            check_pairs_exact(&want, &exact, &format!("cpu-vs-exact {tag}"));
        }
        eprintln!("adapter: {}", ctx.gpu.adapter_info.name);
    });
}

#[test]
fn pbc_neighbor_list_boundary_cases() {
    pollster::block_on(async {
        let _guard = gpu_test_guard();
        // Adversarial placements: cell-boundary snapping, cutoff/limit
        // straddles, and box-edge wrapping (exact ties avoided: round-half
        // conventions need not agree, and both images are then valid).
        let system = solvated_system(6.0);
        let cutoff = 4.0;
        let skin = 1.5;
        let limit = cutoff + skin;
        let ctx = setup(&system, cutoff, skin).await;
        let box_arr = ctx.box_vec.as_array();
        let n = system.atom_count();
        // Exact-fit cell size, mirroring the packing.
        let nx = ((box_arr[0] / limit).floor() as usize).max(1);
        let cx = box_arr[0] / nx as f64;
        let mut rng = 12345u64;
        let jitter = |s: &mut u64| (xorshift(s) - 0.5) * 1e-3;
        // (a) every atom snapped near a cell boundary plane.
        let snapped: Vec<Vec3> = (0..n)
            .map(|i| {
                let k = (i % nx) as f64;
                Vec3 {
                    x: (k * cx + 1e-7 + jitter(&mut rng)).rem_euclid(box_arr[0]),
                    y: xorshift(&mut rng) * box_arr[1],
                    z: xorshift(&mut rng) * box_arr[2],
                }
            })
            .collect();
        // (b) box-edge wrap pair + cutoff straddle pair.
        let mut edged = random_coords(n, box_arr, 5);
        edged[0] = Vec3 {
            x: 1e-7,
            y: 1.0,
            z: 1.0,
        };
        edged[1] = Vec3 {
            x: box_arr[0] - 2e-7,
            y: 1.0,
            z: 1.0,
        };
        edged[2] = Vec3 {
            x: 5.0,
            y: 5.0,
            z: 5.0,
        };
        edged[3] = Vec3 {
            x: 5.0 + cutoff - 1e-4,
            y: 5.0,
            z: 5.0,
        };
        edged[4] = Vec3 {
            x: 8.0,
            y: 8.0,
            z: 8.0,
        };
        edged[5] = Vec3 {
            x: 8.0 + limit - 1e-4,
            y: 8.0,
            z: 8.0,
        };
        for (tag, coords) in [("snapped", snapped), ("edged", edged)] {
            ctx.gpu.set_coordinates(&coords, ctx.box_f32, false);
            let gpu = ctx.gpu.neighbor_list().await.unwrap();
            assert!(gpu.count <= ctx.max_pairs, "{tag}: overflow");
            let want = cpu_list(&coords, &ctx.box_vec, cutoff, skin);
            check_pairs_exact(&gpu.pairs, &want, &format!("gpu {tag}"));
            let exact = exact_pairs(&coords, &box_arr, limit);
            check_pairs_exact(&want, &exact, &format!("cpu {tag}"));
        }
    });
}

#[test]
fn pbc_energy_matches_cpu_and_exact() {
    pollster::block_on(async {
        let _guard = gpu_test_guard();
        let system = solvated_system(6.0);
        let cutoff: f64 = 4.0;
        let skin = 1.5;
        let e: f64 = 78.5;
        let krf = (e - 1.) / (2. * e + 1.) / cutoff.powi(3);
        let crf = 3. * e / (2. * e + 1.) / cutoff;
        let ctx = setup(&system, cutoff, skin).await;
        let box_arr = ctx.box_vec.as_array();
        let rp = ref_params(&system);
        let field = PbcForceField::new(&system, vec![]).unwrap();
        let backend = ReactionField::new(cutoff, e).unwrap();
        let minimized = minimized_coords(&system);
        let mut jittered = minimized.clone();
        let mut rng = 0x1234_5678_9abc_def0u64;
        for v in jittered.iter_mut() {
            // Deterministic +-0.3 A jitter: stays physical, moves atoms
            // across cell/neighborhood boundaries for traversal coverage.
            v.x += (xorshift(&mut rng) - 0.5) * 0.6;
            v.y += (xorshift(&mut rng) - 0.5) * 0.6;
            v.z += (xorshift(&mut rng) - 0.5) * 0.6;
        }
        // Raw builder output: finiteness + overflow-free execution only (see
        // `minimized_coords` for why precision is not asserted here).
        {
            ctx.gpu
                .set_coordinates(&system.coordinates(), ctx.box_f32, true);
            let raw = ctx.gpu.energy_and_forces(false).await.unwrap();
            assert!(!raw.neighbor_overflow, "prepared: neighbor overflow");
            assert!(
                raw.lj.is_finite() && raw.rf.is_finite(),
                "prepared: finite energies"
            );
        }
        for (tag, coords) in [("minimized", minimized), ("jittered", jittered)] {
            ctx.gpu.set_coordinates(&coords, ctx.box_f32, true);
            let got = ctx.gpu.energy_and_forces(true).await.unwrap();
            assert!(!got.neighbor_overflow, "{tag}: neighbor overflow");
            // Runtime CPU oracle (same pair list the dynamics uses).
            let clist = cpu_list(&coords, &ctx.box_vec, cutoff, skin);
            let cpairs: Vec<(usize, usize)> = clist
                .iter()
                .map(|&(a, b)| (a as usize, b as usize))
                .collect();
            let cpu = field
                .evaluate(&coords, &ctx.box_vec, &cpairs, &backend, cutoff)
                .unwrap();
            assert!(
                (got.virial.unwrap() - cpu.virial).abs() <= e_tol(cpu.virial),
                "{tag}: GPU periodic virial {} vs CPU {}",
                got.virial.unwrap(),
                cpu.virial
            );
            // Independent exact oracle (own pair enumeration + formulas).
            let xpairs = exact_pairs(&coords, &box_arr, cutoff + skin);
            let (xlj, xrf, xgrad) = exact_energy(&coords, &box_arr, &xpairs, cutoff, krf, crf, &rp);
            // CPU list path vs exact math (tight: same f64 arithmetic).
            assert!(
                (cpu.components.van_der_waals - xlj).abs() <= 1e-9 * xlj.abs().max(1.),
                "{tag}: cpu LJ vs exact"
            );
            assert!(
                (cpu.components.electrostatics - xrf).abs() <= 1e-9 * xrf.abs().max(1.),
                "{tag}: cpu RF vs exact"
            );
            // GPU kernels vs both oracles.
            assert!(
                (got.lj - cpu.components.van_der_waals).abs()
                    <= e_tol(cpu.components.van_der_waals),
                "{tag}: gpu LJ {} vs cpu {}",
                got.lj,
                cpu.components.van_der_waals
            );
            assert!(
                (got.rf - cpu.components.electrostatics).abs()
                    <= e_tol(cpu.components.electrostatics),
                "{tag}: gpu RF {} vs cpu {}",
                got.rf,
                cpu.components.electrostatics
            );
            assert!((got.lj - xlj).abs() <= e_tol(xlj), "{tag}: gpu LJ vs exact");
            // The GPU evaluator now includes bonded terms and nonbonded
            // forces in one resident gradient array.
            assert!(
                (got.bonds - cpu.components.bonds).abs() <= e_tol(cpu.components.bonds),
                "{tag}: gpu bonds {} vs cpu {}",
                got.bonds,
                cpu.components.bonds
            );
            // Angles are the most cancellation-sensitive bonded component in
            // f32; molecule-centered coordinates keep this at roughly 1%.
            let angle_tolerance = 0.01 + 0.01 * cpu.components.angles.abs();
            assert!(
                (got.angles - cpu.components.angles).abs() <= angle_tolerance,
                "{tag}: gpu angles {} vs cpu {}",
                got.angles,
                cpu.components.angles
            );
            assert!(
                (got.proper_torsions - cpu.components.proper_torsions).abs() <= 1e-4,
                "{tag}: gpu proper torsions {} vs cpu {}",
                got.proper_torsions,
                cpu.components.proper_torsions
            );
            assert!(
                (got.improper_torsions - cpu.components.improper_torsions).abs() <= 1e-4,
                "{tag}: gpu improper torsions {} vs cpu {}",
                got.improper_torsions,
                cpu.components.improper_torsions
            );
            let bonded_baseline = field
                .evaluate(&coords, &ctx.box_vec, &[], &backend, cutoff)
                .unwrap();
            let full: Vec<[f64; 3]> = cpu.gradients.iter().map(|g| [g.x, g.y, g.z]).collect();
            let nb: Vec<[f64; 3]> = cpu
                .gradients
                .iter()
                .zip(bonded_baseline.gradients.iter())
                .map(|(g, b)| [g.x - b.x, g.y - b.y, g.z - b.z])
                .collect();
            let gg = got.gradients.unwrap();
            let mut worst = 0f64;
            let mut bad: Vec<(f64, usize, usize, f32, f64)> = vec![];
            for (i, (g, c)) in gg.iter().zip(full.iter()).enumerate() {
                for (axis, (a, e)) in [g[0], g[1], g[2]].iter().zip(c.iter()).enumerate() {
                    let d = (*a as f64 - e).abs();
                    worst = worst.max(d / gpu_g_tol(*e).max(1e-12));
                    if d > gpu_g_tol(*e) {
                        bad.push((d, i, axis, *a, *e));
                    }
                }
            }
            bad.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
            assert!(
                bad.is_empty(),
                "{tag}: {} force comps fail; worst: {:?}",
                bad.len(),
                &bad[..bad.len().min(10)]
            );
            // Exact-oracle gradient agreement on a sparse subset.
            for i in (0..nb.len()).step_by(37) {
                for (axis, e) in [xgrad[i][0], xgrad[i][1], xgrad[i][2]].iter().enumerate() {
                    let c = nb[i][axis];
                    assert!(
                        (c - e).abs() <= 1e-9 * e.abs().max(1.),
                        "{tag}: cpu vs exact grad {i}/{axis}"
                    );
                }
            }
            eprintln!("{tag}: worst force tol ratio {worst:.3}");
        }
    });
}

#[test]
fn pbc_eval_is_bitwise_deterministic() {
    pollster::block_on(async {
        let _guard = gpu_test_guard();
        let system = solvated_system(6.0);
        let ctx = setup(&system, 4.0, 1.5).await;
        let coords = random_coords(system.atom_count(), ctx.box_vec.as_array(), 7);
        ctx.gpu.set_coordinates(&coords, ctx.box_f32, true);
        let a = ctx.gpu.energy_and_forces(true).await.unwrap();
        // Re-upload before the second eval: back-to-back compute submits
        // without an intervening transfer hang software Vulkan (llvmpipe);
        // real WebGPU queues serialize correctly, and the extra upload is
        // harmless (identical bytes).
        ctx.gpu.set_coordinates(&coords, ctx.box_f32, true);
        let b = ctx.gpu.energy_and_forces(true).await.unwrap();
        assert_eq!(a.lj.to_bits(), b.lj.to_bits(), "LJ totals differ");
        assert_eq!(a.rf.to_bits(), b.rf.to_bits(), "RF totals differ");
        assert_eq!(
            a.gradients.unwrap(),
            b.gradients.unwrap(),
            "gradients differ"
        );
    });
}

#[test]
fn resident_settle_nve_tracks_cpu_step() {
    pollster::block_on(async {
        let _guard = gpu_test_guard();
        // This is a correctness test for the complete resident path, not a
        // performance measurement: one initial force evaluation seeds the
        // Verlet kick, then positions, velocities, SETTLE/RATTLE, and force
        // buffers remain on the device between explicit state samples.
        let system = solvated_system(6.0);
        let protocol = SimulationProtocol {
            solvent: SolventModel::Explicit,
            constraints: ConstraintModel::Settle,
            timestep_fs: 1.0,
            minimization_iterations: 20,
            equilibration_ensemble: Ensemble::Nve,
            production_ensemble: Ensemble::Nve,
            equilibration_steps: 0,
            production_steps: 6,
            save_every: 1,
            friction_per_ps: 0.0,
            cutoff_angstrom: Some(4.0),
            rf_dielectric: Some(78.5),
            seed: 23,
            ..Default::default()
        };
        let mut cpu = ExplicitSimulation::new(&system, protocol).unwrap();
        let coordinates = cpu.state.coordinates.clone();
        let velocities = cpu.state.velocities.clone();
        let mut gpu = setup(&system, 4.0, 1.5).await.gpu;
        gpu.initialize_dynamics(
            &coordinates,
            &velocities,
            [
                system.box_angstrom()[0] as f32,
                system.box_angstrom()[1] as f32,
                system.box_angstrom()[2] as f32,
            ],
            0.001,
        )
        .unwrap();
        let initial = gpu.energy_and_forces(true).await.unwrap();
        assert!(!initial.neighbor_overflow);
        let mut max_position_error = 0.0f64;
        let mut max_velocity_error = 0.0f64;
        for _ in 0..6 {
            gpu.dynamics_step().await.unwrap();
            assert_eq!(gpu.dynamics_status().await.unwrap(), None);
            cpu.step().unwrap();
            let (gpu_coords, gpu_velocities) = gpu
                .read_dynamics_state([
                    system.box_angstrom()[0] as f32,
                    system.box_angstrom()[1] as f32,
                    system.box_angstrom()[2] as f32,
                ])
                .await
                .unwrap();
            assert!(
                max_water_position_error(&system, &gpu_coords) < 5e-5,
                "GPU SETTLE position residual"
            );
            assert!(
                max_water_velocity_error(&system, &gpu_coords, &gpu_velocities) < 5e-4,
                "GPU RATTLE velocity residual"
            );
            for (a, b) in gpu_coords.iter().zip(&cpu.state.coordinates) {
                let d = Vec3 {
                    x: a.x - b.x,
                    y: a.y - b.y,
                    z: a.z - b.z,
                };
                max_position_error = max_position_error.max(dot3(d).sqrt());
            }
            for (a, b) in gpu_velocities.iter().zip(&cpu.state.velocities) {
                let d = Vec3 {
                    x: a.x - b.x,
                    y: a.y - b.y,
                    z: a.z - b.z,
                };
                max_velocity_error = max_velocity_error.max(dot3(d).sqrt());
            }
            let checked = gpu.energy_and_forces(true).await.unwrap();
            assert!(!checked.neighbor_overflow);
            assert!(checked.gradients.is_some());
        }
        // f32 WGSL versus f64 CPU differences are expected, but a wrong
        // constraint ordering or stale force buffer is orders of magnitude
        // larger than these bounds.
        assert!(
            max_position_error < 5e-3,
            "position error {max_position_error}"
        );
        assert!(
            max_velocity_error < 5e-1,
            "velocity error {max_velocity_error}"
        );
        let box_xyz = [
            system.box_angstrom()[0] as f32,
            system.box_angstrom()[1] as f32,
            system.box_angstrom()[2] as f32,
        ];
        let mut batched = setup(&system, 4.0, 1.5).await.gpu;
        batched
            .initialize_dynamics(&coordinates, &velocities, box_xyz, 0.001)
            .unwrap();
        batched.energy_and_forces(true).await.unwrap();
        for size in [1, 2, 3] {
            batched.dynamics_steps(size).await.unwrap();
        }
        assert_eq!(batched.dynamics_status().await.unwrap(), None);
        let a = gpu.read_dynamics_checkpoint(box_xyz).await.unwrap();
        let b = batched.read_dynamics_checkpoint(box_xyz).await.unwrap();
        for (a, b) in a
            .coordinates
            .iter()
            .chain(&a.velocities)
            .zip(b.coordinates.iter().chain(&b.velocities))
        {
            assert!(
                (a.x - b.x)
                    .abs()
                    .max((a.y - b.y).abs())
                    .max((a.z - b.z).abs())
                    < 1e-6,
                "batch size changed deterministic NVE"
            );
        }
    });
}

#[test]
fn resident_settle_nve_stays_stable_over_long_window() {
    pollster::block_on(async {
        let _guard = gpu_test_guard();
        // The default remains a quick regression.  The environment knobs let
        // the same lasting test exercise the browser's 2 fs/9 A horizon on a
        // validation host without maintaining a second dynamics harness.
        let steps = std::env::var("GLYSYS_GPU_LONG_STEPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(200);
        let cutoff = std::env::var("GLYSYS_GPU_LONG_CUTOFF")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(4.0);
        let timestep_fs = std::env::var("GLYSYS_GPU_LONG_TIMESTEP_FS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(1.0);
        let padding = std::env::var("GLYSYS_GPU_LONG_PADDING")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(6.0);
        let with_ions = std::env::var("GLYSYS_GPU_LONG_IONS").is_ok();
        let system = solvated_system_with_ions(padding, with_ions);
        let protocol = SimulationProtocol {
            solvent: SolventModel::Explicit,
            constraints: ConstraintModel::Settle,
            timestep_fs,
            minimization_iterations: 20,
            equilibration_ensemble: Ensemble::Nve,
            production_ensemble: Ensemble::Nve,
            equilibration_steps: 0,
            production_steps: steps,
            save_every: (steps / 5).max(1),
            friction_per_ps: 0.0,
            cutoff_angstrom: Some(cutoff),
            rf_dielectric: Some(78.5),
            seed: 29,
            ..Default::default()
        };
        let cpu = ExplicitSimulation::new(&system, protocol).unwrap();
        let coordinates = cpu.state.coordinates.clone();
        let velocities = cpu.state.velocities.clone();
        let box_xyz = [
            system.box_angstrom()[0] as f32,
            system.box_angstrom()[1] as f32,
            system.box_angstrom()[2] as f32,
        ];
        let mut gpu = setup(&system, cutoff, 1.5).await.gpu;
        gpu.initialize_dynamics(
            &coordinates,
            &velocities,
            box_xyz,
            (timestep_fs * 0.001) as f32,
        )
        .unwrap();
        let initial = gpu.energy_and_forces(true).await.unwrap();
        let masses: Vec<f64> = system.atoms().iter().map(|a| a.mass()).collect();
        let initial_total = initial.lj
            + initial.rf
            + initial.bonds
            + initial.angles
            + initial.proper_torsions
            + initial.improper_torsions
            + kinetic_energy(&masses, &velocities);
        for step in 0..steps {
            gpu.dynamics_step().await.unwrap();
            if (step + 1) % (steps / 5).max(1) == 0 {
                assert_eq!(
                    gpu.dynamics_status().await.unwrap(),
                    None,
                    "step {}",
                    step + 1
                );
            }
        }
        let final_energy = gpu.energy_and_forces(true).await.unwrap();
        let (_, final_velocities) = gpu.read_dynamics_state(box_xyz).await.unwrap();
        let final_total = final_energy.lj
            + final_energy.rf
            + final_energy.bonds
            + final_energy.angles
            + final_energy.proper_torsions
            + final_energy.improper_torsions
            + kinetic_energy(&masses, &final_velocities);
        let drift_per_atom = (final_total - initial_total).abs() / masses.len() as f64;
        assert!(drift_per_atom < 0.05, "resident NVE drift {drift_per_atom}");
        let checkpoint = gpu.read_dynamics_checkpoint(box_xyz).await.unwrap();
        assert!(max_water_position_error(&system, &checkpoint.coordinates) < 5e-5);
        assert!(
            max_water_velocity_error(&system, &checkpoint.coordinates, &checkpoint.velocities)
                < 5e-4
        );
    });
}

fn dot3(v: Vec3) -> f64 {
    v.x * v.x + v.y * v.y + v.z * v.z
}

fn max_water_position_error(system: &ParameterizedSystem, coordinates: &[Vec3]) -> f64 {
    classify_waters(system)
        .into_iter()
        .filter_map(|water| {
            let [o, h1, h2] = water;
            let (oh1, oh2, hh) = water_equilibrium(system, water).ok()?;
            let distance = |a: Vec3, b: Vec3| {
                dot3(Vec3 {
                    x: a.x - b.x,
                    y: a.y - b.y,
                    z: a.z - b.z,
                })
                .sqrt()
            };
            Some(
                (distance(coordinates[o], coordinates[h1]) - oh1)
                    .abs()
                    .max((distance(coordinates[o], coordinates[h2]) - oh2).abs())
                    .max((distance(coordinates[h1], coordinates[h2]) - hh).abs()),
            )
        })
        .fold(0.0, f64::max)
}

fn max_water_velocity_error(
    system: &ParameterizedSystem,
    coordinates: &[Vec3],
    velocities: &[Vec3],
) -> f64 {
    classify_waters(system)
        .into_iter()
        .map(|[o, h1, h2]| {
            let terms = [
                (
                    velocities[h1],
                    velocities[o],
                    coordinates[h1],
                    coordinates[o],
                ),
                (
                    velocities[h2],
                    velocities[o],
                    coordinates[h2],
                    coordinates[o],
                ),
                (
                    velocities[h2],
                    velocities[h1],
                    coordinates[h2],
                    coordinates[h1],
                ),
            ];
            terms
                .into_iter()
                .map(|(va, vb, pa, pb)| {
                    let rel = Vec3 {
                        x: va.x - vb.x,
                        y: va.y - vb.y,
                        z: va.z - vb.z,
                    };
                    let d = Vec3 {
                        x: pa.x - pb.x,
                        y: pa.y - pb.y,
                        z: pa.z - pb.z,
                    };
                    (rel.x * d.x + rel.y * d.y + rel.z * d.z).abs()
                })
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

#[test]
fn resident_langevin_nvt_keeps_rigid_water_stable() {
    pollster::block_on(async {
        let _guard = gpu_test_guard();
        // Keep the default small and quick; optional knobs reproduce the
        // browser's longer explicit NVT run on a software adapter.
        let steps = std::env::var("GLYSYS_GPU_NVT_STEPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(120);
        let cutoff = std::env::var("GLYSYS_GPU_NVT_CUTOFF")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(4.0);
        let timestep_fs = std::env::var("GLYSYS_GPU_NVT_TIMESTEP_FS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(1.0);
        let padding = std::env::var("GLYSYS_GPU_NVT_PADDING")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(6.0);
        let with_ions = std::env::var("GLYSYS_GPU_NVT_IONS").is_ok();
        let system = solvated_system_with_ions(padding, with_ions);
        let protocol = SimulationProtocol {
            solvent: SolventModel::Explicit,
            constraints: ConstraintModel::Settle,
            timestep_fs,
            minimization_iterations: 20,
            equilibration_ensemble: Ensemble::Nvt,
            production_ensemble: Ensemble::Nvt,
            equilibration_steps: 0,
            production_steps: steps,
            save_every: (steps / 6).max(1),
            friction_per_ps: 1.0,
            cutoff_angstrom: Some(cutoff),
            rf_dielectric: Some(78.5),
            seed: 31,
            ..Default::default()
        };
        let cpu = ExplicitSimulation::new(&system, protocol).unwrap();
        let coordinates = cpu.state.coordinates.clone();
        let velocities = cpu.state.velocities.clone();
        let box_xyz = [
            system.box_angstrom()[0] as f32,
            system.box_angstrom()[1] as f32,
            system.box_angstrom()[2] as f32,
        ];
        let mut gpu = setup(&system, cutoff, 1.5).await.gpu;
        gpu.initialize_dynamics(
            &coordinates,
            &velocities,
            box_xyz,
            (timestep_fs * 0.001) as f32,
        )
        .unwrap();
        gpu.energy_and_forces(true).await.unwrap();
        let masses: Vec<f64> = system.atoms().iter().map(|a| a.mass()).collect();
        let mut temperatures = Vec::new();
        for step in 0..steps {
            gpu.dynamics_step_nvt(300.0, 1.0).await.unwrap();
            if (step + 1) % (steps / 6).max(1) == 0 {
                assert_eq!(
                    gpu.dynamics_status().await.unwrap(),
                    None,
                    "step {}",
                    step + 1
                );
                let (_, v) = gpu.read_dynamics_state(box_xyz).await.unwrap();
                let ke = kinetic_energy(&masses, &v);
                // Fully rigid waters remove three scalar DOF each. The test
                // only needs a stability bound; CPU/OpenMM statistical parity
                // is measured by the layered benchmark harness.
                let waters = glysys_energy::pbc::classify_waters(&system).len();
                let dof = 3 * masses.len() - 3 * waters;
                temperatures.push(2.0 * ke / (dof as f64 * 0.00198720425864083));
            }
        }
        let (_, v) = gpu.read_dynamics_state(box_xyz).await.unwrap();
        assert!(
            v.iter()
                .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
        );
        let mean = temperatures.iter().sum::<f64>() / temperatures.len() as f64;
        assert!(mean > 150.0 && mean < 500.0, "NVT mean temperature {mean}");
        assert!(temperatures.iter().all(|t| *t > 20.0 && *t < 1000.0));
    });
}

#[test]
fn resident_nvt_checkpoint_restart_reproduces_rng_stream() {
    pollster::block_on(async {
        let _guard = gpu_test_guard();
        let system = solvated_system(6.0);
        let protocol = SimulationProtocol {
            solvent: SolventModel::Explicit,
            constraints: ConstraintModel::Settle,
            thermostat: glysys_dynamics::Thermostat::Langevin,
            timestep_fs: 1.0,
            minimization_iterations: 20,
            equilibration_ensemble: Ensemble::Nvt,
            production_ensemble: Ensemble::Nvt,
            equilibration_steps: 0,
            production_steps: 16,
            save_every: 8,
            friction_per_ps: 1.0,
            cutoff_angstrom: Some(4.0),
            rf_dielectric: Some(78.5),
            seed: 41,
            ..Default::default()
        };
        let mut cpu = ExplicitSimulation::new(&system, protocol).unwrap();
        let coordinates = cpu.state.coordinates.clone();
        let velocities = cpu.state.velocities.clone();
        let box_xyz = [
            system.box_angstrom()[0] as f32,
            system.box_angstrom()[1] as f32,
            system.box_angstrom()[2] as f32,
        ];
        let mut uninterrupted = setup(&system, 4.0, 1.5).await.gpu;
        uninterrupted
            .initialize_dynamics(&coordinates, &velocities, box_xyz, 0.001)
            .unwrap();
        uninterrupted.energy_and_forces(true).await.unwrap();
        for _ in 0..4 {
            uninterrupted.dynamics_step_nvt(300.0, 1.0).await.unwrap();
        }
        let checkpoint = uninterrupted
            .read_dynamics_checkpoint(box_xyz)
            .await
            .unwrap();
        let evaluated = uninterrupted.read_dynamics_observables(true).await.unwrap();
        cpu.install_evaluated_state(
            checkpoint.coordinates.clone(),
            checkpoint.velocities.clone(),
            4,
            evaluated.lj
                + evaluated.rf
                + evaluated.bonds
                + evaluated.angles
                + evaluated.proper_torsions
                + evaluated.improper_torsions,
            evaluated
                .gradients
                .unwrap()
                .iter()
                .map(|g| Vec3 {
                    x: g[0] as f64,
                    y: g[1] as f64,
                    z: g[2] as f64,
                })
                .collect(),
            evaluated.virial.unwrap(),
        )
        .unwrap();
        cpu.state.resident_rng = Some(glysys_dynamics::resident_rng::ResidentThermostatRng {
            version: 1,
            words: checkpoint.rng_words.clone(),
        });

        let mut resumed = setup(&system, 4.0, 1.5).await.gpu;
        resumed.set_timestep(0.001).unwrap();
        resumed
            .set_dynamics_state(&checkpoint, box_xyz, true)
            .unwrap();
        resumed.energy_and_forces(true).await.unwrap();
        for _ in 0..4 {
            uninterrupted.dynamics_step_nvt(300.0, 1.0).await.unwrap();
        }
        // The resumed path uses one submission; the reference uses four.
        // This simultaneously guards RNG/restart and submission-size invariance.
        resumed.dynamics_steps_nvt(4, 300.0, 1.0).await.unwrap();
        cpu.advance(4).unwrap();
        let a = uninterrupted
            .read_dynamics_checkpoint(box_xyz)
            .await
            .unwrap();
        let b = resumed.read_dynamics_checkpoint(box_xyz).await.unwrap();
        assert_eq!(a.rng_words, b.rng_words, "RNG checkpoint state diverged");
        assert_eq!(
            a.rng_words,
            cpu.state.resident_rng.as_ref().unwrap().words,
            "CPU fallback changed the resident RNG stream"
        );
        for (gpu, cpu) in a.coordinates.iter().zip(&cpu.state.coordinates) {
            assert!(
                (gpu.x - cpu.x)
                    .abs()
                    .max((gpu.y - cpu.y).abs())
                    .max((gpu.z - cpu.z).abs())
                    < 1e-3,
                "CPU fallback NVT position parity"
            );
        }
        let max_coord = a
            .coordinates
            .iter()
            .zip(&b.coordinates)
            .flat_map(|(pa, pb)| {
                [
                    (pa.x - pb.x).abs(),
                    (pa.y - pb.y).abs(),
                    (pa.z - pb.z).abs(),
                ]
            })
            .fold(0.0, f64::max);
        let max_velocity = a
            .velocities
            .iter()
            .zip(&b.velocities)
            .flat_map(|(va, vb)| {
                [
                    (va.x - vb.x).abs(),
                    (va.y - vb.y).abs(),
                    (va.z - vb.z).abs(),
                ]
            })
            .fold(0.0, f64::max);
        assert!(max_coord < 1e-5, "restart coordinate error {max_coord:.3e}");
        assert!(
            max_velocity < 1e-3,
            "restart velocity error {max_velocity:.3e}"
        );
    });
}
