# Changelog

## Unreleased

Carbohydrates are read from their atoms, and GLYCAM-Web's residues are added.
On the 2,542 reference lectin-glycan sites of SugarGLIDER, `inspect` now
parameterizes 91.8% with GLYCAM (0.1.2: 48.0%), 99.3% with `prepare --fix`.

Output changes:

- Carbohydrate residues get the GLYCAM residue their coordinates fit, with
  GLYCAM atom names, whatever they were called. A GLYCAM name that
  contradicts the coordinates is corrected (a beta GlcNAc named `0YA` becomes
  `0YB`); 0.1.2 picked the alpha template for `GAL` on a tie and the beta one
  for `XYS` (alpha in the CCD).
- Methyl glycosides split into the sugar and `OME`, sulfate, methyl and
  acetyl groups into `SO3`, `MEX` and `ACX`, with GLYCAM's charge correction on
  the oxygen they replace the hydrogen of. Sulfated GAGs now sum to an
  integer charge (a heparin fragment was -8.124), and the 0.15 e tolerance
  for them is gone.
- PREP templates are rebuilt with correct torsions: every internal coordinate
  was placed 180 degrees off its torsion (ring-closing bonds up to 4.9 A),
  which moved hydrogens built from template geometry.
- A free reducing end without O1 gets it back as `ROH`, for the anomer the
  residue name gives.
- GLYCAM-Web's galactofuranose templates (`xLU`, `xLD`) are corrected when
  they are loaded: as shipped they are beta/alpha-L-altrofuranose (C5
  inverted, and C1 anomeric relative to that wrong C5, so both centres differ
  from the residue named; crabWURCS reads `LAltf`). After the correction
  crabWURCS reads `0LU` as `DGalfb` and `0LD` as `DGalfa`, and a deposited
  D-galactofuranose gets the anomer its coordinates have. An input built by
  GLYCAM-Web (beta/alpha-L-altrofuranose under a Galf name) is refused with
  the reason, also when its name is a GLYCAM template: preparing it by name
  would simulate the wrong sugar. It cannot be inverted in place (C1 needs the
  residue rebuilt about its glycosidic bond); rebuild it with corrected
  templates.

### Changes

- `src/carbohydrate.rs`: carbohydrate identification by induced subgraph
  match, stereocentre configuration, linkage form and aglycone/substituent
  split. It replaces the 12-name CCD table in `prepare` and in the fixer.
- GLYCAM-Web's residues missing from GLYCAM06j-1 (538, from GMML's
  `GLYCAM_06k_gmml2.lib`), and its corrections of `3dR`, `4AE` and `WdR`; see
  `data/amber/PROVENANCE.md`. Templates GLYCAM_06j.dat cannot parameterize
  completely are never chosen.
- Fructose of sucrose and raffinose (`2CU`, the residue holds its own
  glycosidic oxygen) and the other closed anomeric forms.
- New warnings: `glycan_stereochemistry` (an input follows a GLYCAM library
  coordinate error, e.g. GLYCAM-Web galactofuranose with C5 inverted) and
  `unsupported_residue` (why `inspect` lists a residue).
- `inspect` decides support with the same carbohydrate reading as `prepare`.
- Removed: the reducing-O1 splitter that moved any atom named O1 (a ketose's
  C1 hydroxyl) into `ROH`.

## 0.1.2 - 2026-10-15

Everything since 0.1.1 (2026-09-24). The main additions are a
PDBFixer-equivalent structure fixer (`glysysbuilder fix`), small molecules
with OpenFF Sage 2.2.1 and AM1-BCC charges, DNA/RNA and structural metal ions
in `prepare`, and explicit-solvent dynamics with the GOTW (GROMACS) recipe,
particle-mesh Ewald with Nose-Hoover and Parrinello-Rahman coupling, on the
CPU and resident on the GPU. GlySys now builds against crabWURCS 0.3.3.

Output changes (the same input can give different files than 0.1.1):

- Ion counts follow SLTCAP and ions are spread through the solvent with
  `--seed` selecting the placement.
- GROMACS topologies write 1-4 pairs with their own scale factors
  (function 2) and exclude water hydrogens from each other.
- 1-4 interactions across five-membered rings are excluded as in tleap.
- Glycans in the preparation report are read from the coordinates
  (crabWURCS 0.3.3), and their GLYCAM names carry the reducing end.

### Changes

- crabWURCS 0.3.3 (tag `v0.3.3`). The glycans of the report are read from
  the coordinates: every residue whose atoms form a sugar ring is a sugar
  whatever it is called, and its configuration, anomer and substituents are
  those of its atoms, with the residue name used only where the atoms cannot
  tell. The report changes where a file's residue names and atoms disagree:
  GLYCAM-coded L-iduronic acid is `LIdopA` (was `DIdopA`), an `NDG`-named
  Asn-linked GlcNAc reads `b` (`example/glycoprotein.pdb`), and a GlycoShape
  Level 1 file whose reducing residue keeps the other anomer's code reads the
  anomer of its atoms. Glycans that were split at a residue crabWURCS did not
  know as a sugar (`RHM`, GLYCAM letter codes, `SO3`) are one glycan, with
  the attachment site of its root. GLYCAM-coded and wwPDB-coded files of the
  same structure now give the same glycans.
- crabWURCS 0.3.2 (tag `v0.3.2`). The GLYCAM name of a glycan in the report
  carries its reducing end when the structure declares it (`DGlcpNAcb1-OH`,
  was `DGlcpNAc`), lists substituents the GLYCAM way (`DNeup5Ac[9A]`,
  `DGalp[3S,6S]`), names furanoses and non-default configurations
  (`LAraf`, `DGalf`, `LTalp`) and the configuration of N-sulfated
  glucosamine (`DGlcpNS`). The WURCS of the report is unchanged. crabWURCS
  0.3.2 trims the records it hands to its PDB parser, which rejects an atom
  record that ends before the element columns, so GlySys now writes the
  element it reads for an atom into the view of the file crabWURCS receives
  when the input leaves those columns blank; before, padding the line was
  enough.
- Ions are spread through the solvent. Each ion replaces a water taken in a
  seeded pseudo-random order, at least 5 Å from the solute and from every
  other ion (by the minimum image). Ions used to go where the solute's own
  electrostatic potential was lowest (sodium) or highest (chloride), without
  the potential of the ions already placed: around a charged solute every
  counter-ion started on its surface and every co-ion in one cluster in the
  far corner of the box (17 chloride ions 4.6 Å apart for a glycan of charge
  −4). The numbers of ions are unchanged; `--seed` now selects the placement.
- The GOTW recipe runs on the GPU: particle-mesh Ewald (direct-space term in
  the tile kernel, reciprocal space on a mesh transformed on the device,
  corrections for excluded pairs), and the leap-frog integrator with
  Nose–Hoover and Parrinello–Rahman coupling, the pressure from the atomic
  virial with constraint forces, and center-of-mass removal per group, all
  resident on the device (`glysys-md run --backend vulkan --gromacs-mdp ...`).
  The device follows the CPU integrator step for step (box, pressure and
  thermostat variables to 4-6 digits over 200 steps on the same mesh).
  Molecules are wrapped into the box on the device, so a long run keeps
  single-precision bond lengths. `pmeGrid` in a protocol sets the PME mesh
  (GROMACS `fourier-nx/ny/nz`); the device rounds each axis up to a power of
  two.
- Ion counts follow SLTCAP (Schmit et al., J. Chem. Theory Comput. 2018, 14,
  1823). The salt is counted from the water molecules of the box (N0 pairs at
  the requested concentration) and a solute of charge Q gets
  sqrt(N0² + Q²/4) ∓ Q/2 ions of each sign, so the box matches a bath of that
  concentration. Preparation used to neutralize the solute and then add one
  pair per volume of the box as built, which gave about 10% more salt than
  asked for (the box is built slightly under liquid density and holds the
  solute too) and more again around a charged solute: a glycan of charge −4
  in 5,291 waters at 0.2 M now gets 21 Na⁺ and 17 Cl⁻ instead of 25 and 21.
  A neutral solute gets N0 pairs; without salt only the neutralizing
  counter-ions are added, as before.
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
- The preparation report lists a branched GLYCAM-Web glycan (letter-coded
  branch residues written as ATOM records) as one glycan.
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
- crabWURCS is now taken from its GitHub repository at a release tag instead
  of a sibling `../crabWURCS` checkout, so clean clones and CI build without
  it; the published crates depend on the matching crabwurcs release on
  crates.io (0.3.3 for this release).
- Add `benchmarks/openmm_replica_control.py` (the replica-aggregation gate
  applied to two independent replica sets, e.g. OpenMM against OpenMM) and the
  1 ns explicit validation protocol
  `benchmarks/dynamics/1crn-explicit-lf-middle-2fs-1ns-validation.json`.

### Earlier in this cycle (2026-09-25)

- Add explicit LF-middle NVT integration, hydrogen-bond constraints, and v3
  checkpoint velocity-convention validation while preserving legacy BAOAB
  checkpoint behavior.
- Add native benchmark controls, output-boundary scheduling, scalar-only
  thermodynamic observations, and GPU execution/readback diagnostics.
- Improve resident explicit/implicit GPU execution and explicit neighbor/force
  evaluation; add WASM-compatible RNG dependencies for browser builds.
- Add reproducible 1CRN OpenMM comparison and trajectory-validation scripts.

### Validation status (2026-09-25, LF-middle integrator)

The full explicit 1CRN GPU run completed 0.308 ns at 2 fs without CPU fallback,
and the tested initial-state energy/force comparisons passed. The independent
production-statistics check did not meet its predeclared margins (temperature
mean difference 3.86 K; potential-energy mean difference 0.00891 kcal/mol per
atom). Treat dynamics as experimental and do not interpret this release as
OpenMM-equivalent. The latest short explicit GPU benchmark median was 219.77
ns/day versus 600.53 ns/day for OpenMM 8.1.1 OpenCL mixed precision.
