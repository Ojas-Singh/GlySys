# Molecular scoring reference corpus, schema 1

The initial corpus contains a dipeptide and a standalone glycan, with and without
OBC2. `manifest.json` records original structures, full coordinate/topology exports,
checksums, model version, units, settings and independent evaluation results.
Coordinates in the JSON are unrounded; the Amber coordinate file is archival.

Reproduce from the GlySys workspace:

```sh
cargo run -p glysys-energy --example reference_export -- dipeptide > /tmp/dipeptide.json
cargo run -p glysys-energy --example reference_export -- glycan > /tmp/glycan.json
python benchmarks/openmm_reference.py /tmp/dipeptide.json
python benchmarks/openmm_reference.py /tmp/glycan.json
```

Use an isolated Python environment with `openmm==8.1.1` and `numpy<2` for the
recorded references. The evaluator uses OpenMM's Reference platform, NoCutoff,
no constraints, matched charges/LJ/1–4 parameters, and explicit native OBC2
radii/screens. This independently checks energy and gradient formulas and Amber
export; it does **not** independently establish chemical perception or parameter
assignment. The older `tests/fixtures/glycan.prmtop` omits a cap bond present in
GlySys preparation and must not be treated as an identical-topology reference.
OpenMM's default finite-cutoff reaction-field convention is not a reference for
the current hard-cutoff Coulomb evaluator.

These comparisons exposed and now cover the torsion angle convention, OBC2
radius-offset descreening, Amber signed torsion indices, GLYCAM SCEE/SCNB values,
and multiline torsion parameter parsing. The largest errors in this initial
corpus are below 2e-6 kcal/mol and 2e-6 kcal/mol/angstrom. They are small-system
correctness evidence, not docking or binding-affinity validation.

Future curated complexes must preserve assembly, atom correspondence, glycan
stereochemistry, waters, metals, structure quality, conditions, source/license,
and uncertainty. Unknown fields stay null. Pose-ranking labels, affinity values
and nonbinding evidence are separate; missing affinity is not nonbinding.
Keep each complex's conformers/decoys in one split, grouped further by receptor
family and glycan similarity. Do not train on this tiny formula-check corpus.

Performance records must identify physical GPU/browser/driver, initialized WASM
thread count, model/plan/workload IDs, candidate count, cold/warm state, peak buffer
bytes, preparation/transforms/sterics/priors/scoring/refinement/transfers/output
seconds, component/ranking errors and sampling acceptance/population/ESS metrics.
Software Vulkan is correctness evidence only. No performance thresholds have yet
been established on physical Apple, integrated or discrete GPUs.

## Native session and GOTW qualification

The native `glysys-md` adapter is intentionally independent of the browser:

```sh
cargo run -p glysys-md -- devices
cargo run -p glysys-md -- verify-gromacs --input prepared
cargo run -p glysys-md -- benchmark --input prepared --backend cpu --threads 48 --steps 10000
```

Use `resolve-mdp` to capture a GROMACS recipe before a qualification run. It
retains PME, Nose–Hoover, Parrinello–Rahman, coupling groups, COM groups, and
the exact raw keys in an audit JSON. The native driver rejects those choices
until their independent gates pass; it never substitutes reaction field or a
different thermostat. The opt-in four-replica Slurm adapter lives in
`GlycoShape-Cookbook/API/GOTW_Scripts/gotw_glysys_iridis6.sh` and requires a
lossless `system.snapshot.json` in every replica. The existing GROMACS runner
remains the production reference during qualification.

## Periodic explicit-water (PBC) OpenMM parity

Fixture: `tests/fixtures/dipeptide.pdb` solvated in a 9 A TIP3P box
(1139 atoms), parameterized once by GlySys; the Amber prmtop is exported
and OpenMM rebuilds the identical chemistry from it
(`benchmarks/openmm_pbc_rf.py`).

Method matching (all verified, not assumed): CutoffPeriodic, 9 A cutoff,
no switching, reaction-field dielectric 78.5, no dispersion correction,
no COM removal, rigid TIP3P on both sides. OpenMM's `rigidWater` lists
only the two O-H bonds (2 constraints per water); the checker adds the
H-H constraint explicitly so both sides sample the fully rigid ensemble
(true DOF 3N - 3N_water = 2298 here; using OpenMM's bookkeeping count
would bias every temperature by ~42 K). 1-4 exception pairs use plain
Coulomb on both sides (OpenMM exception convention); applying
reaction-field screening to them errs by 61 kcal/mol on this fixture.

Run:

```sh
cargo run -p glysys-dynamics --example explicit_reference > /tmp/pbc.json
/path/to/openmm-env/bin/python benchmarks/openmm_pbc_rf.py /tmp/pbc.json
```

Use a disposable environment (`openmm==8.1.1`, `numpy<2`); OpenMM is a
validation reference only, never a runtime dependency. The checker exits
nonzero on any failure and always prints the full report JSON.

Layers and tolerances (rationale, not round numbers):

- Layer 1 (statics): total energy, per-term components (bond, angle,
  torsion, LJ isolated by a zero-charge leg, RF electrostatics), and
  forces on minimized and post-NVE snapshots. Tolerances 0.05 kcal/mol
  (energy/term) and 0.02 kcal/mol/A (forces): far above f64 path noise
  (observed agreement 0.00000/2e-6) so any failure is a real formula,
  cutoff, exclusion, 1-4, minimum-image, or RF-convention bug.
- Layer 2 (NVE): total-energy drift per atom over matched windows,
  flexible-water leg (isolates constraints), dt-convergence legs at
  fixed physical time (second-order integrators must scale ~dt^2;
  guards against dt-independent leaks), and a cross-code magnitude
  ratio band [0.1, 10]. Tolerance 0.05/atom over 0.4 ps (~8% of kT per
  atom): short-window drift conflates truncation with chaotic
  trajectory divergence, so the scaling law is the correctness signal
  and the absolute number is a regression tripwire, not a proof.
- Layer 3 (NVT): production temperature, potential, and kinetic means
  with block-averaged (autocorrelation-aware) standard errors;
  tolerances are 3x combined SEM with floors (8 K, 10/2 kcal/mol)
  covering representation differences no statistic can see. A
  production-halves stationarity check fails drifting runs outright:
  a transient cannot validate a thermostat. NVT lengths are set by
  equilibration physics (8 ps = 8 tau at friction 1/ps); 1 ps of
  equilibration leaves a measurable cold transient from a minimized
  start and was the first thing this harness caught.

### Current results (solvated dipeptide, 1139 atoms, 9 A cutoff + RF 78.5)

- Layer 1: minimized snapshot A differs from OpenMM by `7.13e-7`
  kcal/mol in total energy and `2.90e-6` kcal/mol/A in the largest force
  component; post-NVE snapshot B differs by `4.40e-7` and `2.03e-6`.
  Bond, angle, torsion, LJ (zero-charge leg), and RF components each agree
  below `1.7e-6` kcal/mol. The independent periodic virial check agrees by
  `0.121` and `0.017` kcal/mol on A and B (scaled tolerances 1.262 and
  1.061). Static energies, forces, decompositions, and pair virials pass.
- Layer 2: rigid-water 2 fs NVE drift is `0.002603` kcal/mol/atom versus
  OpenMM `0.000533` (ratio 3.22, inside the declared `[0.1, 10]` band), a
  roughly 20-fold improvement over the former rotation path. The flexible
  1 fs leg is `0.02823` versus `0.01372` kcal/mol/atom. Fixed-time
  convergence is monotonic: GlySys `3.72e-5`, `2.41e-4`, `1.72e-3` at
  0.5, 1, and 2 fs (log slope 2.76); OpenMM gives `1.27e-4`, `1.92e-4`,
  `5.33e-4`. Maximum position and velocity constraint residuals are
  `1.0e-10` A and `5.8e-14` A/ps.
- Layer 3: after 8 ps equilibration and 8 ps production, OpenMM is
  `299.28±0.93` K versus GlySys `297.96±1.83` K; potential energy is
  `-3537.41±5.28` versus `-3535.32±6.57` kcal/mol; kinetic energy is
  `680.37±2.12` versus `677.38±4.16` kcal/mol. The block-SEM comparisons,
  target-temperature check, and production-halves stationarity checks pass.
- The GPU PBC path uses the same canonical `crates/glysys-gpu/src/pbc.wgsl`
  for native and browser execution. On lavapipe, exact neighbor-list tests,
  independent O(N²) energy/force oracles, bitwise repeated evaluation,
  resident constrained NVE, long-window NVE stability, resident Langevin
  NVT, and checkpoint/RNG restart all pass. This validates shader compilation,
  layouts, dispatches, reductions, PBC logic, and constraints; software
  Vulkan is not hardware performance evidence.

The former drift came from applying a position correction and then retaining
the unconstrained trial half-step velocity. That omits the constraint impulse;
the old frame-rotation repair also introduced a non-Hamiltonian deformation
before RATTLE. The replacement is the direct OpenMM SETTLE position solve
(including its H-H quadratic correction), velocity reconstruction from the
constrained displacement, and the closed-form three-mass RATTLE velocity
projection. CPU and WGSL use the same equations. The old rotation heuristic
is gone.

Other fixes caught by this harness: 1-4 pairs bypass reaction-field screening
(plain Coulomb, OpenMM exception convention); fully rigid TIP3P requires the
explicit H-H constraint in the OpenMM oracle; NVT needs approximately 8 ps of
equilibration from a minimized start; GPU 1-4 specials are bidirectional; and
nonbonded GPU force checks subtract the bonded baseline. The pressure
conversion constant is correct: `BAR_PER_KCAL_MOL_A3 = 69476.95457` from
`(4184 J/kcal)·10^25/(N_A)`; the focused dimensional test and independent
periodic virial comparison pass.

### CPU NPT reference

New explicit protocols can opt into the corrected molecule-preserving
Monte-Carlo barostat and the homogeneous OpenMM long-range LJ correction. The
Rust reference exporter keeps this leg optional so ordinary force/NVE/NVT
fixtures stay fast:

```sh
GLYSYS_REF_NPT=1 \\
GLYSYS_REF_NPT_EQUILIBRATION=50000 \\
GLYSYS_REF_NPT_PRODUCTION=250000 \\
cargo run -p glysys-dynamics --example explicit_reference > /tmp/pbc-npt.json
/path/to/openmm-env/bin/python benchmarks/openmm_npt.py /tmp/pbc-npt.json
```

`openmm_npt.py` is a disposable OpenMM Reference-platform oracle (the current
recorded run used OpenMM 8.1.1). It
rebuilds the exported Amber topology with CutoffPeriodic reaction field,
fully constrained TIP3P water, the requested pressure and barostat interval,
and the same dispersion-correction setting. It reports temperature using
the actual constrained degrees of freedom, finite-sample means, and volume and
density differences. Runs with fewer than 20 saved production samples are
marked `inconclusive`; they are useful smoke tests but do not establish NPT
statistics. OpenMM and GlySys use independent Monte-Carlo streams, so parity
is assessed from block/statistical observables rather than identical volume
trajectories. Capacity or cutoff-geometry failures are execution errors, not
ordinary rejected moves.

The NPT model is versioned (`tip3p-rf-md-npt-v2`). Historical v1 NVE/NVT
checkpoints remain readable; old NPT state is never silently resumed under the
new sampler. GPU execution currently uses the resident periodic evaluator for
trial scoring and falls back before a box/cell-capacity transition; no physical
GPU performance claim is made from lavapipe.
