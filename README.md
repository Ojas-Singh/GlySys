# GlySys

GlySys is a pure-Rust molecular-system and Amber/GLYCAM parameterization
library. It includes the `glysysbuilder` system-preparation CLI.
It reads complete-heavy-atom PDB structures, adds force-field hydrogens,
parameterizes proteins with ff14SB and carbohydrates with GLYCAM06j-1,
solvates with TIP3P, adds neutralizing ions and 0.15 M NaCl, and writes files
for OpenMM and GROMACS.

The workspace also contains two reusable MIT-licensed libraries:

- `glysys-energy` evaluates Amber/GLYCAM bonded, nonbonded, restraint, and
  OBC2 GBSA terms in kcal/mol, with Cartesian gradients in kcal/mol/Å.
- `glysys-opt` provides deterministic seeded genetic search and L-BFGS
  minimization independently of any molecular representation.

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
```

The output bundle contains:

- `system.prmtop` and `system.inpcrd` for OpenMM's Amber readers
- `system.top` and `system.gro` for GROMACS
- `manifest.json` with options, provenance, detected glycans, system counts,
  charge, box dimensions, and warnings

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

## Data licensing

The Rust source is MIT licensed. AmberTools states that force-field parameter
files in `dat/leap` are in the public domain. See
[`data/amber/PROVENANCE.md`](data/amber/PROVENANCE.md) for the pinned subset.
