# Molecular scoring foundation, development milestone

This work keeps preparation in GlySys, search/sampling in ReGlyco, reusable
optimizers in glysys-opt, and GPU execution optional. It does not implement
docking, design, additional empirical attractions or public telemetry.

## Scientific reference changes (model v2)

- Generated hydrogens are reconstructed in topology-derived parent frames when
  source heavy-atom coordinates change; missing heavy atoms are rejected.
  Evaluated/generated hydrogens are retained in exported energy-model frames
  and optimized structures instead of being discarded during coordinate updates.
- Receptor, glycan, water, ion and unknown roles replace the complement-of-glycan
  assumption for cross interaction scoring.
- Proper/improper torsions use the Amber dihedral convention. OBC2 descreening
  uses the offset intrinsic radius. Amber export correctly encodes suppressed
  1–4 interactions/impropers even when signed indices would otherwise be zero.
- Parameter loading preserves GLYCAM SCEE/SCNB, multiline torsion continuations
  and explicit pattern overrides.
- Circular mixture densities normalize component weights and von Mises factors,
  use stable log-sum-exp and expose analytic angular derivatives. Component
  probability bounds use numerical circular integration. Historical backfill
  retains a separately named Gaussian approximation.
- Sampled ensembles use attempted-transition burn-in/thinning, explicit proposal
  corrections and rejected-state residence. Steric mode targets the prior
  conditioned on sterics; interaction mode adds the cross-energy Boltzmann
  factor; full energy uses the prior for proposals only. This is a finite
  represented-state model, not a demonstrated solution-equilibrium population.
- Conformer collections retain optimization/repair without population claims.
  Exported ensemble coordinates are preserved; CPU recomputation updates their
  energies, torsions, prior values and steric checks, including after relaxation.
  Verification and relaxation reuse chemistry prepared from the original assets;
  generated output hydrogens are mapped to that topology, not re-parameterized.

## Public contracts

`glysys-energy::scoring` provides PreparedScene, Pose/PoseBatch, ScoreModel,
versioned TermDescriptor, EvaluationPlan, EvaluationRequest/Result and a CPU
PreparedEvaluator. `glysys-gpu::scoring::PreparedGpuEvaluator` adapts resident
Amber kernels to the same named values/candidate IDs. Unsupported advanced
requests use the CPU evaluator and identify that backend explicitly. It is a
low-level execution adapter; callers still own qualification and failure policy.

Raw/weighted term values remain distinct. Gradients and negative-gradient forces,
Cartesian/pose derivatives, per-term reference derivatives and bounded pair
features can be requested independently. A compiled native extension and a
feature-only plan demonstrate extensibility without changing search/report code.
The registry is deliberately small, not a runtime expression language.

Prepared chemistry retains the original parameterized system, stable source
mapping where supplied, generated-atom identity, explicit roles/groups, unknown
chemical annotations, hydrogen-parent links, known protein aromatic rings and
source occupancy/B factors. Unknown parameter-file provenance is labelled as
unknown rather than fabricated. Nonperiodic scoring is explicit; periodic
preparation is not treated as support for PME. Scene fingerprints cover system,
coordinates, parameters, settings and groups at evaluator construction.

The topology-derived kinematic tree partitions rigid fragments and derives
hydrogen-inclusive torsion subtrees; ring bonds and inconsistent tree orientation
are rejected. Pose derivatives are checked against Cartesian finite differences.
Free-pose primitives are available in the library, but no docking workflow exists.

## GPU coverage in this development milestone

- Existing resident interaction/full-energy/OBC2 kernels now have separate
  score-only specialization and corrected physical conventions.
- A resident receptor/conformer attachment library supports compact candidate
  genes, attachment transforms, rotamer updates and traversal-compatible steric
  screening in Cookbook populations and independent sampled-ensemble chains.
- Original early-exit order is retained. Near-cutoff and acceptance-boundary
  steric decisions use CPU reference checks. Auto includes required CPU
  conformation materialization in the 20% qualification threshold.
- Geometry qualification/failure is local to a search session and does not
  disable unrelated energy work. GPU input ranges and memory/device limits are
  checked. Existing energy allocation retries shrink batches; geometry allocation
  failure currently falls back to CPU.

## Remaining implementation work

This is an initial foundation milestone, not completion of the entire approved
plan. The following are still required:

- Integrate the new public scoring/pose adapter throughout ReGlyco rather than
  keeping its existing specialized energy client. Replace remaining thread-local
  energy lifecycle state with a single explicit job evaluator session.
- Complete chemical provenance/connectivity annotations (formal charges,
  protonation/template source, glycan linkage/stereochemistry/ring semantics),
  receptor-state handles and versioned general directional feature schemas.
- Move VMM values/gates/polish and compatible-pool/targeted-repair work to useful
  GPU batches. Current normalized VMM calculations remain CPU reference work.
- Replace serial per-candidate steric pair traversal with exact spatial indexing
  that preserves traversal-sensitive semantics. Avoid CPU conformation rebuilding
  for Cookbook freezing by returning compact GPU geometry summaries.
- Shared pair tiles, compact candidate updates for full force-field scoring,
  mixed-term plans, fully resident refinement/post-relaxation, unified buffer
  budgeting, geometry batch shrinking and complete checkpoint/device-loss tests.
- Larger independent chemistry/reference fixtures, model-distribution tests and
  effective sample sizes. The small analytic MH test and fixture checks are not
  comprehensive ensemble validation.
- Physical-browser qualification and cold/warm threaded-WASM benchmarks on
  Apple Silicon, integrated graphics and discrete GPUs. The server's software
  Vulkan tests establish correctness only; no 3x/10x performance claim is made.

`benchmarks/manifest.json` and `benchmarks/README.md` describe the checksummed
OpenMM reference corpus and its limits. Future ranking/affinity/nonbinding labels
remain separate and unknown values stay unknown.
