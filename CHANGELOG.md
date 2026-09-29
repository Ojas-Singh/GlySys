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
- `prepare` parameterizes small molecules with OpenFF Sage 2.2.1 and AM1-BCC
  charges (SMIRKS/SMARTS engine, MDL and AM1-BCC aromaticity, a native AM1
  implementation), validated against the OpenFF Toolkit, AmberTools and
  OpenFF Interchange.
- `prepare` supports DNA/RNA (OL15/OL3), keeps structural divalent metal ions
  (Li/Merz 12-6), caps chain breaks, honours protonation states written as
  hydrogens (e.g. by `fix`), infers undeclared glycosidic and N/O-glycan
  bonds, represents free reducing sugars with GLYCAM's ROH, and names the
  residues responsible for a fractional total charge.
- `prepare --fix` / `BuildOptions::repair` repairs the input first.
- Fix 1-4 interactions for atoms that are also 1-3 neighbours across a
  five-membered ring (proline, histidine, tryptophan, furanoses, nucleic-acid
  sugars): they are now excluded as in tleap, in the Amber and GROMACS writers
  and in the CPU and GPU energy code.

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
