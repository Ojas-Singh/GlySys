# GlySys

GlySys is a pure-Rust molecular-system and Amber/GLYCAM parameterization
library. It includes the `glysysbuilder` system-preparation CLI and the
native `glysys-md` simulation runner.
It repairs raw PDB entries (a PDBFixer-equivalent `fix` command), adds
force-field hydrogens, parameterizes proteins with ff14SB, carbohydrates with
GLYCAM06j-1, DNA/RNA with OL15/OL3, structural metal ions with Li/Merz 12-6
parameters and other small molecules (ligands, cofactors) with the generic
OpenFF Sage 2.2.1 force field and AM1-BCC charges, solvates with TIP3P, adds
neutralizing ions and 0.15 M NaCl, and writes files for OpenMM and GROMACS.

The workspace also contains reusable AGPL-3.0-only libraries:

- `glysys-energy` evaluates Amber/GLYCAM bonded, nonbonded, restraint, and
  OBC2 GBSA terms in kcal/mol, with Cartesian gradients in kcal/mol/Å.
- `glysys-opt` provides deterministic seeded genetic search and L-BFGS
  minimization independently of any molecular representation.
- `glysys-runtime` provides the backend-neutral execution sessions shared by
  native MD, browser laboratory bindings, hydration, scoring, and ReGlyco.
  One coordinator-owned `GpuContext` supplies resident wgpu resources and the
  canonical WGSL kernels; CPU fallback and v3 checkpoints are handled at this
  boundary. See [`docs/runtime-sessions.md`](docs/runtime-sessions.md).

No AmberTools, acpype, Python, GROMACS, or OpenMM executable is invoked at
runtime. The exact public-domain AmberTools 23.6 parameter subset is embedded
in the crate.

## CLI

```console
glysysbuilder fix 1abc.pdb -o fixed.pdb --report fix-report.json
glysysbuilder fix 1abc.pdb -o fixed.pdb --ph 5.5 --missing-residues internal --remove-water
glysysbuilder inspect input.pdb
glysysbuilder prepare 1abc.pdb --fix --output prepared   # repair a raw PDB entry first
glysysbuilder prepare input.pdb --output prepared
glysysbuilder prepare input.pdb -o prepared --padding 12 --salt 0.15 --seed 7
glysysbuilder prepare input.pdb -o prepared --protonation A:42=HID
glysysbuilder prepare input.pdb -o dry --no-water
glysysbuilder prepare input.pdb -o water-only --no-ions

# Native MD on a prepared snapshot
glysys-md benchmark --input prepared --backend auto --threads 48 --steps 1000
glysys-md run --input prepared --output replica-01 --backend cpu --threads 48
glysys-md resume --input prepared --output replica-01 --backend cpu --threads 48
glysys-md devices
glysys-md resolve-mdp --mdp production.mdp --protocol-out resolved.protocol.json
glysys-md verify-gromacs --input prepared
```

`glysys-md` is the native session adapter; it does not require WASM or a
browser. `--threads` is per process, so a Slurm replica should receive only
the CPUs in its allocation. The default GPU policy is adaptive and queries
legal wgpu limits; `--gpu-memory low-memory` restores the historical 256 MiB
cap, while `--gpu-memory-mib N` sets an explicit aggregate budget. `devices`
prints adapter type, features, and limits so a Vulkan device can be audited
before a benchmark.

The current validated explicit periodic model is reaction-field with the
existing constrained Langevin/v-rescale and Monte Carlo paths. The protocol
and GROMACS resolver preserve PME, Nose–Hoover, and Parrinello–Rahman as
explicit choices, but those choices stop with a capability error until the
independent PME/coupling parity gates pass. GlySys therefore cannot yet claim
to replace the GOTW GROMACS production recipe; use the opt-in native Slurm
adapter in `GlycoShape-Cookbook/API/GOTW_Scripts` for qualification runs.

The output bundle contains:

- `system.prmtop` and `system.inpcrd` for OpenMM's Amber readers
- `system.top` and `system.gro` for GROMACS
- `system.snapshot.json` for lossless native GlySys execution
- `manifest.json` with options, provenance, detected glycans, system counts,
  charge, box dimensions, and warnings

`resolve-mdp` records the supported GOTW settings without changing units or
silently mapping PME/Nose–Hoover/Parrinello–Rahman to reaction field. Those
models remain capability-gated until their independent parity tests pass.
`verify-gromacs` is a strict, read-only cross-check: it compares the emitted
`.gro` coordinates/box and `.top` atom order, interactions, exclusions,
1–4 pairs, and SETTLE declarations with the lossless snapshot before a native
run. It does not attempt to reconstruct chemistry from a text topology.

## 1CRN dynamics qualification

Latest diagnostic screen (2026-10-06): Intel i7-12700H with an RTX 3060 Laptop
GPU on Windows 11, OpenMM 8.6.1 in mixed precision, the prepared systems and
2 fs LF-middle protocols described below, and one paired 8-second window per
engine after minimization and warmup (ns/day):

| Model | GlySys Vulkan | OpenMM CUDA | OpenMM OpenCL | GlySys CPU | OpenMM CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Explicit TIP3P, reaction field | 555 | 541 | 386 | 11.0 / 32.7 / 32.8 / 34.2 (1 / 6 / 14 / 20 threads) | 8.0 / 17.3 / 28.9 / 31.6 |
| Implicit OBC2 | 1834 | 1499 | 1194 | 43.5 / 130.9 / 134.4 (1 / 6 / 14 threads) | 29.6 / 101.9 / 83.6 |

Accuracy on the same machine, against OpenMM with unchanged margins (3 K and
0.005 kcal/mol/atom): end-of-run forces from every engine match OpenMM
Reference (normalized RMS 4.7e-6 explicit CPU, 5.3e-5 explicit Vulkan, 7.3e-6
implicit CPU, 1.4e-5 implicit Vulkan). Two explicit Vulkan 0.308 ns replicas
pass every trajectory check (0.17 K, 0.00019 kcal/mol/atom). A 1 ns explicit CPU
run (0.06 K, 0.00044 kcal/mol/atom) and a 2 ns implicit CPU run (0.13 K, 0.0018)
pass mean equivalence with 20 ps blocks; at the predeclared 10 ps blocks the
explicit run's energy correlation is 0.304 (rule ≤ 0.3) and the implicit run's
single-run stationarity intervals, like OpenMM's own, exceed the margin. Three
2 ns implicit Vulkan replicas differ from OpenMM by 0.00185 kcal/mol/atom, less
than two independent OpenMM replica sets differ from each other (0.00256). See
[`performance-progress.md`](benchmarks/dynamics/performance-progress.md). These
short screens are not the qualification matrix below, which pins OpenMM 8.1.1,
runs on an RX 7800 XT, and uses repeated order-alternated windows. Older short
benchmarks used different measurement boundaries and 0.5 fs protocols, so they
are not carried forward as current throughput claims.

The new qualification uses the prepared 12,132-atom explicit TIP3P system
(12 Å padding, 0.15 M salt, periodic reaction field, 9 Å cutoff) and the
642-atom implicit OBC2 system. Both use NVT, 2 fs, hydrogen constraints,
300 K, 1/ps friction, 8 ps equilibration, and 0.3 ns production. OpenMM 8.1.1
is the pinned development/reference dependency; GlySys runtime remains Rust
and wgpu. Primary targets are Vulkan for GlySys GPU and OpenCL mixed precision
for OpenMM GPU, plus CPU at 1, 6, and 12 threads.

| Model | GlySys CPU ns/day (1 / 6 / 12 threads) | GlySys Vulkan ns/day | OpenMM CPU ns/day (1 / 6 / 12 threads) | OpenMM OpenCL ns/day | Median ratio gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Explicit TIP3P, reaction field | pending qualification | 219.77 (3×5 s diagnostic) | pending qualification | 600.53 (3×5 s diagnostic) | 36.6% diagnostic; not passed |
| Implicit OBC2 | pending qualification | pending qualification | pending qualification | pending qualification | not passed |

The 2026-09-25 explicit screen used the then opt-in cooperative kernel, fixed
640-entry neighbor rows, 2 fs LF-middle NVT, 2,000 warmup steps, and three
synchronized 5-second windows. Its median was 219.77 ns/day on the RX 7800 XT
Vulkan backend; the paired OpenMM 8.1.1 OpenCL mixed-precision median was
600.53 ns/day (36.6%).
An earlier fixed-row screen measured 247.28 versus 663.20 ns/day; these short
screens vary with run conditions and are diagnostics, not the full matrix.

A separate 0.308 ns explicit GPU trajectory completed all 154,000 steps with
no CPU fallback. Including initialization and scheduled output, its measured
rate was 198.82 ns/day. Static energy/force checks passed, but the predeclared
production comparison did not: mean temperature differed by 3.86 K (3 K
margin), and mean potential energy differed by 0.00891 kcal/mol/atom (0.005
margin). This implementation therefore remains experimental for dynamics; the
trajectory result is not described as OpenMM-equivalent. Raw data and the
failed comparison are preserved under
`D:\GlySys-performance-20260925\phase-e-explicit-fixed640-scalar-rerun`.

Do not treat “pending” as zero or as an extrapolated result. The runner
alternates engine order, starts repeats from the same step-zero
coordinates/velocities, excludes setup and warmup from its timed window,
records actual completed steps, and preserves raw JSON/logs plus binary,
source-diff, protocol, and input hashes.
See the qualification protocols and
[`benchmark_1crn_matrix.py`](benchmarks/dynamics/benchmark_1crn_matrix.py),
[`openmm_1crn_dynamics.py`](benchmarks/openmm_1crn_dynamics.py), and
[`validate_1crn_dynamics.py`](benchmarks/validate_1crn_dynamics.py).

Configuration can be saved as TOML or JSON and supplied with `--config`:

```toml
padding_angstrom = 12.0
salt_molar = 0.15
seed = 7
model = 1
add_water = true
add_ions = true

[protonation.residues]
"A:42" = "HID"
```

## Structure repair (`fix`)

`glysysbuilder fix` (and `StructureFixer` in the library) repairs a PDB model
without parameterizing it, so ligands, metals and unusual chemistry are kept
instead of rejected:

- selects one model and one alternate location per residue, including
  microheterogeneous sites (two residue types at one position);
- replaces modified residues by their standard parent (`MSE` → `MET`,
  `SEP` → `SER`, `PTR` → `TYR`, …) using MODRES records, PDBFixer's
  substitution table and the Chemical Component Dictionary; hydroxyproline and
  GLYCAM glycosylated residues are kept;
- models residues that are absent from the coordinates, taking their exact
  numbering from `REMARK 465` when present and otherwise aligning `SEQRES` as
  PDBFixer does; internal gaps are closed by cyclic coordinate descent and
  termini are grown by beam search, and numbering gaps without a physical
  chain break are left alone;
- rebuilds missing heavy atoms from each atom's local template geometry and
  relieves the resulting clashes by a torsion scan and restrained L-BFGS
  minimization in which only new atoms move;
- assigns protonation states for a pH (default 7): disulfides, metal-bound
  cysteine thiolates, histidine tautomers from hydrogen bonds and metal
  coordination, and N-/O-glycosylated residues;
- adds hydrogens to proteins, DNA/RNA (Amber OL15/OL3 templates), glycans
  (GLYCAM06j-1 names), waters (oriented to hydrogen bond) and ligands, then
  relaxes hydrogens that touch other atoms.

Ligand chemistry comes from the wwPDB Chemical Component Dictionary (CCD). The
CLI downloads the definitions it needs from the RCSB (only component codes are
sent) and caches them in `~/.cache/glysys/ccd`; `--offline` and `--ccd FILE`
use local definitions only. Library callers pass definitions through
`ComponentLibrary::add_cif`, and `StructureFixer::component_requests` lists the
codes a structure needs. Output uses wwPDB names by default; `--naming amber`
writes Amber/GLYCAM residue names (`HID`, `CYX`, `NLN`, `4YB`). Every change is
listed in the JSON report.

[`benchmarks/fixer`](benchmarks/fixer) compares `fix` with OpenMM PDBFixer on 30
PDB entries covering alternate locations, NMR models, modified residues,
ligands, metals, nucleic acids, glycoproteins and a 58,000-atom assembly.

## Generic force field for everything else (`prepare`)

Residues that no ff14SB, GLYCAM06j-1, OL15/OL3 or ion template covers — drug
ligands, cofactors, buffer molecules, modified residues kept as they are — are
parameterized with **OpenFF Sage 2.2.1** (SMIRNOFF: every bond, angle, torsion,
improper and van der Waals parameter is assigned by SMIRKS matching, including
MDL aromaticity) and **AM1-BCC** partial charges computed by GlySys's own AM1
implementation (MOPAC parameters) plus the original AM1-BCC bond charge
corrections. Sage is designed to be combined with Amber ff14SB, and its 1-4
scaling (0.5 / 0.8333) is Amber's.

Chemistry (bond orders, formal charges, hydrogens) comes from each residue's
Chemical Component Dictionary definition, fetched like `fix` does. Ligands
must carry their hydrogens: `prepare --fix` (or `BuildOptions::repair`) runs
the structure fixer first. Charges are computed at the deposited geometry
without optimization, and the CCD protonation state is used as-is.

Validation (`scripts/data`, 80 CCD components from the fixer benchmark):

- Sage assignments are identical to the OpenFF Toolkit's `label_molecules` for
  every bond, angle, torsion, improper and vdW term (≈17,000 terms);
- AM1 heats of formation agree with AmberTools `sqm` to 0.06 kcal/mol
  (median), and AM1-BCC charges with `antechamber -c bcc` to 0.005 e RMS
  (median; 78/80 molecules within 0.05 e on every atom — nitrate and the flavin
  of FAD differ by AmberTools' resonance-form choices);
- OpenMM energies of GlySys prmtops equal OpenFF Interchange energies for the
  same charges within 0.003 kcal/mol.

Not covered: ligands containing metals (heme and other organometallics),
ligands covalently bonded to the protein, and ligands whose deposited
coordinates are incomplete; these are reported per residue.

## Rust API

```rust,no_run
use glysys::{BuildOptions, SystemBuilder};

let builder = SystemBuilder::new(BuildOptions::default())?;
let system = builder.prepare_pdb("input.pdb")?;
system.write_bundle("prepared")?;
# Ok::<(), glysys::BuildError>(())
```

`ParameterizedSystem::coordinates` and `set_coordinates` provide a checked
coordinate-update path for energy minimizers. `Structure::update_from_parameterized`
copies minimized coordinates back without a file-format round trip.

## Input contract

`prepare` accepts PDB files whose heavy atoms are complete (use `--fix` for
raw PDB entries). It supports DNA and RNA (Amber OL15/OL3; a 5'-terminal
phosphate is removed), structural Mg/Ca/Zn/Mn/Fe(II)/Cu/Co/Ni/Cd/Hg ions
(Li/Merz 12-6 compromise set; coordination bonds are not modelled), small
molecules via the generic force field above, and
standard proteins, standalone GLYCAM-compatible glycans, noncovalent
lectin–glycan complexes, and existing Asn/Ser/Thr/Hyp glycosylation.
Metal-containing ligands, covalently bound ligands, missing heavy atoms and
ambiguous covalent chemistry are rejected with residue-level diagnostics.

Input waters and free ions are removed and rebuilt. Protein hydrogens are
regenerated from the selected Amber templates. Supplied glycan hydrogens whose
names are compatible with the selected GLYCAM template retain their original
coordinates; only absent glycan hydrogens are reconstructed.

## Licensing

This project is dual licensed.

The public source code is available under the **GNU Affero General Public
License v3.0 only (AGPL-3.0-only)**. See [`LICENSE`](LICENSE).

Organizations that require terms other than the AGPL—for example for
proprietary integration or commercial redistribution—may obtain a separate
commercial licence from the copyright holder. See
[`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md).

A separate commercial licence does not change the AGPL rights granted to users
of the public version.

Third-party dependencies remain subject to their respective licences.

Versions released before the AGPL transition remain available under their
original MIT terms.

## Data licensing

AmberTools states that force-field parameter
files in `dat/leap` are in the public domain. See
[`data/amber/PROVENANCE.md`](data/amber/PROVENANCE.md) for the pinned subset.

OpenFF Sage 2.2.1 parameters (`data/openff`) are CC-BY-4.0 (Open Force Field
Initiative); the AM1-BCC corrections come from openff-recharge (MIT) and the
AM1 parameters from MOPAC (Apache-2.0). See
[`data/openff/PROVENANCE.md`](data/openff/PROVENANCE.md).

The modified-residue substitution table in `src/fix/chemistry.rs` is adapted
from OpenMM PDBFixer (MIT licence, Copyright (c) 2013-2025 Stanford University
and the Authors). Chemical Component Dictionary definitions are downloaded at
run time and are not redistributed.
