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
