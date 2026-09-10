# Development repair and browser laboratory

## ReGlyco repair deployed 2026-09-08

Development manifest: `0.1.0-repair-20260908`, web assets
`/mnt/glycoshape/GlycoShapeCE/web-development-repair-20260908`.
Rollback service override:
`/var/lib/glycoshape-ce/rollbacks/repair-20260908/telemetry.conf`.
Release content SHA-256 remains
`8f01d40ea9b22abd0588dce7fbcd76766899ce959a091861a23c5d0aee8a7876`.

Implemented: sampler-owned energy evaluator, batched independent-chain GPU
surrogates with exact delayed-acceptance correction, cached normalized priors,
score-only sampler geometry (no Cookbook freezing materialization), deferred
output structures, and stage-level compute reporting. Sampler version is
`model-da-mh-v3`. CPU runs use one-stage exact MH; GPU-qualified runs use two-stage
MH. The target and attempted-step residence semantics are unchanged, but GPU
mixing and seeded transitions can differ. GPU surrogate errors never substitute
for the exact target energy in the final acceptance correction.

Tests: analytic delayed-acceptance detailed balance; native molecular GPU
sampling dispatch for both energy objectives with independently recomputed
export energies; native steric seeded transition parity; browser CPU and
unavailable-GPU build and both ensemble modes; artifact checksums and release
GPU-exclusion audit. Native GPU tests used software Vulkan, not physical GPU
performance evidence. Live development smoke tests passed.

One cold-worker CPU browser smoke benchmark (single-site fixture, 100 frames,
4 chains, 250 burn-in steps and thinning 50, seed 23): prior foundation 2175 ms,
repair 1429 ms. Both performed 6000 attempts with acceptance 0.9425. This is a
small single-run measurement under concurrent development load, not a hardware
speedup claim or a comparison with the released historical sampler.

Remaining repair work: main build/collection energy ownership migration;
GPU VMM/polish and pool coverage; comprehensive timings and large fixture/ESS corpus;
Safari physical-device reproduction and performance validation.

## Laboratory development deployment (2026-09-08)

Physical rigid-TIP3P probe CPU and GPU evaluators added. Formula uses shipped
GLYCAM OW radius 1.7683 and epsilon 0.1520; water charges -0.834/+0.417 and rigid
0.9572 Å / 104.52° geometry. Outputs are exploratory interaction minima, not
occupancy, confidence or water displacement free energy. CPU/GPU component
parity and rigid water/translation tests pass.

CPU and resident WebGPU OBC2 BAOAB implementations now run beneath the browser
laboratory. CPU minimization precedes dynamics. GPU checkpoints are checked
against CPU component energies and gradients; initial trajectory parity is
required. Auto additionally measures the initial batch benefit. Browser storage
holds full frames and restart states; the viewer loads only its selected frame.
Energy, temperature, aligned heavy-atom RMSD and supported glycan linkage torsion
plots share the playback cursor. Downloadable ZIP contains full-precision JSON
trajectory, topology and restart; energy CSV is separate.

Validation on 2026-09-08: native energy/dynamics/GPU suite passed (31 tests;
three pre-existing ignored tests). This includes deterministic cell-stream steric
traversal and flexible updates. Browser CPU and unavailable-GPU hydration,
imported/deposited providers, dynamics, cancellation and exact restart passed.
Checkpoint RNG states serialize as decimal strings to preserve all 64 bits.
Completed checkpoints can be resumed without corrupting diagnostics. The default
10 ps browser protocol completed 20,000 steps and exported 200 full-precision
frames; plots, playback, ZIP export/import and completed restart passed.

Matched OpenMM 8.1.1 BAOAB checks with identical stochastic increments agree
within approximately 2e-9 Angstrom in coordinates over 20 steps with and without
friction. Inputs and results are in benchmarks/dynamics. Harmonic NVE drift,
thermostat temperature and transactional rollback tests also pass. These checks
do not establish long-time molecular equilibrium statistics. GPU tests use
software Vulkan; physical Safari/GPU performance remains unmeasured.

The hydration recovery fixture is deliberately location-informed (a region near
a deposited 1UBQ water), not a blind accuracy benchmark. Predictions remain
independent probe alternatives, not an equilibrated water network. GPU dynamics
uses implicit OBC2 only. NPT and explicit solvent are excluded.

Recent ReGlyco additions isolate qualification by energy plan, split the 256 MiB
geometry/energy budget, and release initialization resources before sampler
ownership. Steric receptor cell streams preserve original atom traversal order;
flexible atoms are maintained separately. Larger cutoffs retain exact full scans.

## Reproducible checks

- Native: `CARGO_BUILD_JOBS=2 cargo test -p glysys-dynamics -p glysys-energy -p glysys-gpu -- --test-threads=1`.
- ReGlyco: `cargo test -p reglyco-ensemble --features webgpu --test statistical_gpu -- --test-threads=1`.
- CE browser scripts: `scripts/test-glysys-lab-browser.cjs`,
  `scripts/test-glysys-lab-restart.cjs`, `scripts/test-glysys-lab-ui.cjs`.
  Run with Node and installed Playwright (or `PLAYWRIGHT_MODULE` pointing to its
  module), a staged `web/dist`, and `LAB_TEST_URL` (default loopback 16969).
- Artifact audit: `python3 scripts/check-laboratory-deployment.py <webroot> gpu`
  for development; use `cpu` for release exclusion.

Hydration-only UI builds set `VITE_GLYSYS_DYNAMICS=false`; the subsequent
laboratory development build sets it to `true`. Both also require the explicit
development profile and WebGPU flag. Release excludes laboratory imports,
artifacts and manifest even when both feature flags are accidentally enabled.

## Current deployment and rollback

The hydration update was staged and deployed first, followed by the dynamics
update. Current 6969 assets are
`/mnt/glycoshape/GlycoShapeCE/web-development-dynamics-20260908`; the ReGlyco
manifest is `0.1.0-laboratory-20260908`. Hydration's intermediate deployment is
`web-development-hydration-20260908`. Each copied artifact was checksum-verified
before service activation. Release build artifact-exclusion checks passed.

Rollback overrides are stored in
`/var/lib/glycoshape-ce/rollbacks/dynamics-20260908/telemetry.conf` (back to
hydration) and `rollbacks/hydration-20260908/telemetry.conf` (back to the prior
repair). Restore the appropriate file to the service drop-in, daemon-reload and
restart only `glycoshape-ce.service`. The release service and its index checksum
remain unchanged.

This is a development milestone, not completion of the entire performance and
scientific acceptance matrix. Remaining work includes physical Safari/GPU
qualification, large multi-site benchmarks and effective-samples-per-second,
longer multi-seed molecular thermostat statistics, GPU VMM/polish/pool coverage,
and migration of the remaining build/collection compatibility evaluator owner.
Minimization in the GlySys laboratory currently runs on CPU; dynamics integration
and force batches can use GPU with CPU checks. No NPT or explicit-solvent MD is
exposed.
