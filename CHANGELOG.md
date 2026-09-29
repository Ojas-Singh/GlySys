# Changelog

## Unreleased

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
