# ReGlyco development scoring foundation

Deployed 2026-09-08 at https://dev.glycoshape.io/reglyco, served by CE on
loopback port 6969. Engine manifest: `0.1.0-foundation-20260908`.
Release port 8989 remains unchanged and CPU-only.

This is an initial implementation milestone, **not completion of the full
approved scoring/GPU plan**. See [scoring-foundation.md](scoring-foundation.md)
for contracts, scientific corrections and remaining implementation priorities.

## Available for development testing

- Auto / CPU / GPU controls, with CPU numerical qualification and fallback.
- Existing interaction, full molecular-mechanics and OBC2 energy/gradient GPU
  kernels, plus a separate score-only specialization.
- Resident attachment transforms and traversal-compatible steric batches in
  Cookbook populations and independent sampled-ensemble chains. Boundary cases
  use CPU checks. Auto includes required CPU materialization overhead.
- Backend-independent named scoring/pose contracts, an initial compiled term
  registry, reference derivatives/features, topology-derived torsion trees and
  a resident GPU adapter. ReGlyco still has a specialized energy client.
- Corrected Amber torsion/OBC2 conventions, parameter continuations and GLYCAM
  scaling, atom roles and generated-hydrogen coordinate/export handling.
- Explicit Sampled and Conformer collection modes. Statistical burn-in/thinning
  counts attempted transitions and retains rejected-state residence. Collections
  permit minimization/repair without claiming equilibrium populations.
- CPU-checked per-frame energy and geometry records, including after relaxation.
  Verification reuses original prepared chemistry; it does not re-parameterize
  hydrogen-enriched outputs. PDB coordinate rounding is explicitly distinguished
  from the full-precision coordinates used for reported energies.
- Versioned local reports and checksummed CPU/GPU single/threaded artifacts.
  Detailed driver errors remain in reports; the UI shows a short fallback status.

VMM values/gates/polish, compatible-pool/targeted-repair acceleration, fully
resident refinement/post-relaxation, complete shared scoring-client integration,
spatial/tiled optimizations and broader failure testing remain unfinished.
There is no docking/design UI or new public telemetry.

## Validation performed

- 17 energy unit tests, three pose/export integration tests and three resident
  GPU/scoring tests passed.
- 21 ReGlyco ensemble and 21 workflow tests passed, including native GPU parity.
- Five-seed independent-chain CPU/GPU comparisons retained identical transitions
  and emitted structures for the tested fixture. A minimized-collection regression
  checks retained hydrogens and energy consistency with the prepared topology.
- Independent OpenMM 8.1.1 Reference-platform checks passed for two small fixtures,
  with/without OBC2. Errors were below 2e-6 in the recorded energy/gradient units.
  These use matched exported parameters and do not independently validate all
  chemical perception or parameter assignment. See the benchmark corpus README.
- Six CE UI/profile tests and TypeScript checks passed.
- Packaged browser workers passed build, sampled ensemble, minimized collection
  and post-relaxation jobs; CPU override and unavailable-GPU fallback were checked.
  Per-frame energies matched across CPU and fallback paths.
- The live HTTPS UI and both ensemble modes passed browser checks after deployment.
  All seven development manifest entries passed checksum verification.
- A release build with full profile and the GPU flag still excluded GPU artifacts,
  manifest entries, imports and controls. CPU WASM dependencies exclude wgpu and
  glysys-gpu. The live release index SHA-256 remained
  `8f01d40ea9b22abd0588dce7fbcd76766899ce959a091861a23c5d0aee8a7876`.

The server has no usable physical GPU. Native GPU tests used software Vulkan;
browser tests verified fallback. This is correctness evidence, not proof of the
3x end-to-end or 10x kernel targets. Physical Apple Silicon, integrated and discrete
GPU/browser benchmarks, larger fixtures, distribution/ESS studies and systematic
mid-job device-loss/memory/cancellation tests remain required. WebGPU runs on the
visitor's GPU, so the server hardware does not prevent user testing.

## Deployment and rollback

Current assets:
`/mnt/glycoshape/GlycoShapeCE/web-development-foundation-20260908`.
Previous assets remain at `web-development-webgpu-20260907`, including cached
chunks retained for existing pages. The service drop-in backup is:
`/var/lib/glycoshape-ce/rollbacks/foundation-20260908/telemetry.conf`.

To restore the previous development preview:

```sh
sudo cp /var/lib/glycoshape-ce/rollbacks/foundation-20260908/telemetry.conf /etc/systemd/system/glycoshape-ce.service.d/telemetry.conf
sudo systemctl daemon-reload
sudo systemctl restart glycoshape-ce.service
```

Telemetry environment/database settings were preserved. The temporary server on
16969 was stopped after verification; the release service was not restarted.
