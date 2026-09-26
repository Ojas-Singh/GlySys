# Changelog

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
