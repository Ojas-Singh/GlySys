# GlySys

GlySys is a pure-Rust molecular-system and Amber/GLYCAM parameterization
library. It includes the `glysysbuilder` system-preparation CLI and the
native `glysys-md` simulation runner.
It reads complete-heavy-atom PDB structures, adds force-field hydrogens,
parameterizes proteins with ff14SB and carbohydrates with GLYCAM06j-1,
solvates with TIP3P, adds neutralizing ions and 0.15 M NaCl, and writes files
for OpenMM and GROMACS.

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
glysysbuilder inspect input.pdb
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

Version 0.1 accepts PDB files whose heavy atoms are complete. It supports
standard proteins, standalone GLYCAM-compatible glycans, noncovalent
lectin–glycan complexes, and existing Asn/Ser/Thr/Hyp glycosylation.
Unsupported ligands, nucleic acids, lipids, metals, missing heavy atoms, and
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
