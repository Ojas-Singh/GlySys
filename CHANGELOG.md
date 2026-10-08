# Changelog

## Unreleased

- Explicit-solvent dynamics on the CPU gains what the GOTW (GROMACS) recipe
  uses: smooth particle-mesh Ewald electrostatics (`glysys_energy::pme`, a PME
  mode of the cluster-pair kernel), Nose–Hoover temperature coupling per group
  and isotropic Parrinello–Rahman pressure coupling in a leap-frog integrator
  that follows GROMACS' `md`, with center-of-mass motion removed per group.
  Checked against OpenMM (PME energies and forces) and GROMACS (thermostat
  response, density).
- `glysys-md --gromacs-mdp` can be given once per file: a minimisation and one
  stage per dynamics file. `--trajectory dcd`, `--trajectory-atoms solute` and
  `--checkpoint-ps` write compact output for long runs.
- GROMACS topology: 1-4 pairs carry their own scale factors (function 2).
  They were written with function 1, so GROMACS applied the ff14SB factors
  (Lennard-Jones 0.5, electrostatics 1/1.2) to GLYCAM06 pairs, which are not
  scaled. The Amber files and the native engines were not affected. The
  hydrogens of a water are now excluded from each other as well.
- Faster explicit CPU steps: force buffers combined over the slots a chunk
  writes, pair lists kept across small box changes and rebuilt faster, SETTLE
  solved in place, step buffers reused.
- A barostat pressure computed from forces takes the virial of the truncated
  Lennard-Jones tail (`PbcForceField::dispersion_pressure_coefficient`).

- Add `glysysbuilder fix` / `StructureFixer`, a PDBFixer-equivalent structure
  repair: alternate-location and microheterogeneity resolution, modified-residue
  replacement (MODRES, PDBFixer table, CCD parents), missing-residue modelling
  from REMARK 465 or SEQRES with loop closure and terminal growth, missing
  heavy-atom reconstruction with clash relief, pH-dependent protonation
  (disulfides, metal sites, histidine tautomers, glycosylated residues), and
  hydrogens for proteins, nucleic acids, glycans, waters and CCD-defined
  ligands, with a JSON report of every change.
- Add Amber OL15 DNA and OL3 RNA residue templates (used by the fixer).
- PDB reading: hybrid-36 serials and residue numbers, duplicate serials,
  microheterogeneous residues, TER-separated segments, element inference for
  two-letter elements, and REMARK 465/SEQRES/MODRES records.
- Add the `benchmarks/fixer` comparison with OpenMM PDBFixer.
- `prepare` parameterizes small molecules with OpenFF Sage 2.2.1 and AM1-BCC
  charges (SMIRKS/SMARTS engine, MDL and AM1-BCC aromaticity, a native AM1
  implementation), validated against the OpenFF Toolkit, AmberTools and
  OpenFF Interchange.
- `prepare` supports DNA/RNA (OL15/OL3), keeps structural divalent metal ions
  (Li/Merz 12-6), caps chain breaks, honours protonation states written as
  hydrogens (e.g. by `fix`), infers undeclared glycosidic and N/O-glycan
  bonds, represents free reducing sugars with GLYCAM's ROH, and names the
  residues responsible for a fractional total charge.
- `prepare --fix` / `BuildOptions::repair` repairs the input first;
  `SystemBuilder::component_requests` lists the CCD definitions it can use.
- Add `ParameterizedSystem::solute` (the solvated system's dry solute).
- The aligned heavy-atom RMSD of a trajectory frame is taken over the solute
  only (`analysis::rmsd_atoms`): water and single-atom ions are left out. In a
  water box it used to measure the solvent's diffusion. Add
  `ParameterizedSystem::solute_atom_count`.
- Fix a data race in the tiled explicit kernels that made them wrong on Metal
  (Safari): the sorted slots of four atoms shared one `vec4` and were stored
  one component at a time, so slots were lost, far exclusions (disulfides,
  glycan and other inter-residue links) were not applied, and bonded atoms
  repelled each other. Each atom now has its own element. Other devices were
  not affected.
- In a browser, `GpuContext::uncaptured_errors` keeps what the device raised
  outside an error scope (a shader the browser's compiler rejected leaves an
  invalid pipeline whose passes silently do nothing).
- Add `glysys_gpu::selftest::bit_patterns`: a device self-test that copies
  integer bit patterns through float lanes the way the kernels do (buffer
  loads, rebuilt locals, workgroup memory, `select`, store and rewrite) and
  reports the ones that come back different.
- A failed solute constraint on the GPU names its constraint group and says
  when the reported error is saturated; the group index no longer spills into
  the error kind for solutes with more than 4,095 groups.
- The no-cutoff energy no longer allocates every atom pair (large systems
  ran out of memory, notably in WebAssembly) and is about twice as fast,
  with bit-identical results.
- Fix 1-4 interactions for atoms that are also 1-3 neighbours across a
  five-membered ring (proline, histidine, tryptophan, furanoses, nucleic-acid
  sugars): they are now excluded as in tleap, in the Amber and GROMACS writers
  and in the CPU and GPU energy code.
- Explicit GPU dynamics use a new tiled nonbonded engine by default
  (`PbcKernel::Tiles`): Hilbert-ordered 32-atom blocks with bounding-box tile
  lists rebuilt on the GPU, full-list tiles without atomics, and deterministic
  64-bit fixed-point bonded/energy accumulation. Integration packets are
  encoded in one compute pass with up to two in flight, and energies are
  evaluated only on each packet's final step. The CSR and fixed-row kernels stay selectable
  through `ResidentPbc::with_context_kernel`; `with_context_variant(true)`
  still selects fixed rows.
- Implicit GPU LF-middle dynamics use 32×32 OBC2 tiles for Born radii, their
  adjoints and forces, with per-term bonded forces.
- `GpuContext` turns off wgpu's indirect-dispatch validation, which re-encoded
  every indirect dispatch on native backends (browsers validate on their own).
- CPU explicit dynamics evaluate reaction-field nonbonded forces with an f32
  8-atom cluster-pair engine (explicit AVX2 with a bit-identical portable
  fallback, reused Verlet cluster lists); SETTLE and the LF-middle updates run
  in parallel, and unobserved steps skip energies.
- CPU implicit dynamics evaluate the no-cutoff Lennard-Jones/Coulomb and OBC2
  forces with an f32 engine (AVX2 with a bit-identical portable fallback;
  results do not depend on the thread count). Set
  `GLYSYS_CPU_REFERENCE_PAIRS` to use the f64 reference forces in dynamics.
- Torsion gradients use the closed-form expression instead of dual numbers
  (same angle bits, gradients equal to ~1e-9), roughly halving bonded cost.
- Implicit checkpoint restore compares forces with a scale-aware tolerance, so
  single-precision GPU checkpoints restore.
- `benchmarks/openmm_1crn_dynamics.py` accepts `--allow-unpinned-openmm` and
  the CUDA platform (mixed precision); `aggregate_1crn_replicas.py` accepts
  `--expected-backend CPU`; `openmm_force_snapshot.py` and
  `compare_force_snapshots.py` accept `--allow-unpinned-openmm`.
- crabWURCS is now taken from its GitHub repository at tag `v0.3.1` instead of
  a sibling `../crabWURCS` checkout, so clean clones and CI build without it;
  published crates still depend on crabwurcs 0.3.1 from crates.io.
- Add `benchmarks/openmm_replica_control.py` (the replica-aggregation gate
  applied to two independent replica sets, e.g. OpenMM against OpenMM) and the
  1 ns explicit validation protocol
  `benchmarks/dynamics/1crn-explicit-lf-middle-2fs-1ns-validation.json`.

## 0.1.2 — 2026-09-25

- Add explicit LF-middle NVT integration, hydrogen-bond constraints, and v3
  checkpoint velocity-convention validation while preserving legacy BAOAB
  checkpoint behavior.
- Add native benchmark controls, output-boundary scheduling, scalar-only
  thermodynamic observations, and GPU execution/readback diagnostics.
- Improve resident explicit/implicit GPU execution and explicit neighbor/force
  evaluation; add WASM-compatible RNG dependencies for browser builds.
- Add reproducible 1CRN OpenMM comparison and trajectory-validation scripts.

### Validation status

The full explicit 1CRN GPU run completed 0.308 ns at 2 fs without CPU fallback,
and the tested initial-state energy/force comparisons passed. The independent
production-statistics check did not meet its predeclared margins (temperature
mean difference 3.86 K; potential-energy mean difference 0.00891 kcal/mol per
atom). Treat dynamics as experimental and do not interpret this release as
OpenMM-equivalent. The latest short explicit GPU benchmark median was 219.77
ns/day versus 600.53 ns/day for OpenMM 8.1.1 OpenCL mixed precision.
