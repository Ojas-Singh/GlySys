#!/usr/bin/env python3
"""Particle-mesh Ewald parity against OpenMM's Reference platform.

Input: JSON from

    cargo run --release -p glysys-energy --example pme_snapshot -- \
        system.snapshot.json --order 5 --out glysys_pme.json

Mode 1 (`--prmtop system.prmtop`): build the same system in OpenMM with
`nonbondedMethod=PME`, the GlySys cutoff, no constraints, flexible water, no
switching function and no dispersion correction, force OpenMM onto exactly
the Ewald coefficient and grid GlySys used (`setPMEParameters`), and compare
energies and forces in double precision. OpenMM's B-spline order is fixed at
5, so the GlySys run must use `--order 5`. Positions come from the GlySys
JSON, so both engines see identical coordinates.

Besides the totals, the comparison isolates the part that depends on the
Ewald settings (regular-pair direct space + reciprocal + self + exclusion
corrections) by subtracting, in both engines, an evaluation without regular
pair electrostatics (in OpenMM: particle charges zeroed, exception charge
products kept). A prepared but unminimized system can carry Lennard-Jones
clashes that dominate the total force RMS; the isolated comparison is not
affected by them.

Mode 2 (`--reference converged.json`): compare the Ewald part of one GlySys
evaluation with another one (finer grid, larger cutoff), to measure the
accuracy of a production setting.

Mode 3 (`--prmtop system.prmtop --minimize relaxed.json`): relax the
configuration with OpenMM's minimizer (CPU platform, same PME settings) and
write it for `pme_snapshot --coordinates relaxed.json`, so that the
comparisons above can also be made on a configuration without clashes.

Exits nonzero when a tolerance in TOL is exceeded; always prints the report.
"""
import argparse
import json
import sys

import numpy as np

TOL = {
    # Total potential energy, kcal/mol per atom.
    "energy_per_atom_kcal_mol": 1e-4,
    # RMS force difference over RMS force, total and Ewald part.
    "force_rms_ratio": 1e-4,
}


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def compare_forces(ours, theirs):
    delta = ours - theirs
    return {
        "rmsDifference": rms(delta),
        "rmsReference": rms(theirs),
        "rmsRatio": rms(delta) / rms(theirs),
        "maxAbsDifference": float(np.abs(delta).max()),
    }


def build_system(prmtop, run, zero_particle_charges=False, zero_exception_charges=False):
    """The GlySys Hamiltonian in OpenMM, with PME forced onto the GlySys settings."""
    import openmm as mm
    from openmm import app, unit

    system = prmtop.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=run["cutoffAngstrom"] / 10.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    nb.setUseSwitchingFunction(False)
    nb.setUseDispersionCorrection(False)
    nb.setPMEParameters(run["pme"]["alphaPerAngstrom"] * 10.0, *run["pme"]["grid"])
    if zero_particle_charges:
        for i in range(nb.getNumParticles()):
            _, sigma, epsilon = nb.getParticleParameters(i)
            nb.setParticleParameters(i, 0.0, sigma, epsilon)
    if zero_exception_charges:
        for i in range(nb.getNumExceptions()):
            a, b, _, sigma, epsilon = nb.getExceptionParameters(i)
            nb.setExceptionParameters(i, a, b, 0.0, sigma, epsilon)
    for force in system.getForces():
        group = {"HarmonicBondForce": 0, "HarmonicAngleForce": 1,
                 "PeriodicTorsionForce": 2, "NonbondedForce": 3}.get(
                     force.__class__.__name__, 4)
        force.setForceGroup(group)
    return system, nb


def make_context(system, run, platform_name, threads=None):
    import openmm as mm

    platform = mm.Platform.getPlatformByName(platform_name)
    properties = {"Threads": str(threads)} if platform_name == "CPU" and threads else {}
    context = mm.Context(system, mm.VerletIntegrator(0.001), platform, properties)
    lx, ly, lz = (v / 10.0 for v in run["boxAngstrom"])
    context.setPeriodicBoxVectors(mm.Vec3(lx, 0, 0), mm.Vec3(0, ly, 0), mm.Vec3(0, 0, lz))
    context.setPositions(np.array(run["coordinates"]) / 10.0)
    return context


def minimize(run, prmtop_path, out_path, iterations, threads):
    """Relax the configuration and write it for `pme_snapshot --coordinates`."""
    import openmm as mm
    from openmm import app, unit

    kcal = unit.kilocalories_per_mole
    system, _ = build_system(app.AmberPrmtopFile(prmtop_path), run)
    context = make_context(system, run, "CPU", threads)
    before = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kcal)
    mm.LocalEnergyMinimizer.minimize(context, tolerance=10.0, maxIterations=iterations)
    # Positions are not wrapped, so molecules stay whole.
    state = context.getState(getEnergy=True, getPositions=True)
    after = state.getPotentialEnergy().value_in_unit(kcal)
    coordinates = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    report = {
        "schemaVersion": 1,
        "mode": "minimize",
        "openmmVersion": mm.__version__,
        "iterations": iterations,
        "energyBeforeKcalMol": before,
        "energyAfterKcalMol": after,
        "boxAngstrom": run["boxAngstrom"],
        "failures": [],
    }
    with open(out_path, "w") as handle:
        json.dump({**report, "coordinates": coordinates.tolist()}, handle)
    return report


def openmm_parity(run, prmtop_path, platform_name, threads):
    import openmm as mm
    from openmm import app, unit

    pme = run["pme"]
    if pme["interpolationOrder"] != 5:
        sys.exit("OpenMM interpolates with order 5: rerun pme_snapshot with --order 5")
    n = run["atoms"]
    alpha_nm = pme["alphaPerAngstrom"] * 10.0
    grid = pme["grid"]
    kcal = unit.kilocalories_per_mole
    prmtop = app.AmberPrmtopFile(prmtop_path)

    def build(**zeroed):
        return build_system(prmtop, run, **zeroed)

    def evaluate(system, nb, groups=(0, 1, 2, 3)):
        context = make_context(system, run, platform_name, threads)
        out = {"pmeInContext": list(nb.getPMEParametersInContext(context))}
        for g in groups:
            out[g] = context.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(kcal)
        state = context.getState(getEnergy=True, getForces=True)
        out["total"] = state.getPotentialEnergy().value_in_unit(kcal)
        out["forces"] = state.getForces(asNumpy=True).value_in_unit(kcal / unit.angstrom)
        del context
        return out

    assert prmtop.topology.getNumAtoms() == n, "prmtop and snapshot differ in atom count"
    full = evaluate(*build())
    used_alpha, *used_grid = full["pmeInContext"]
    if abs(used_alpha - alpha_nm) > 1e-12 * alpha_nm or list(used_grid) != list(grid):
        sys.exit(f"OpenMM used alpha {used_alpha} grid {used_grid}, not {alpha_nm} {grid}")
    # Lennard-Jones and exceptions only, then Lennard-Jones only.
    rest = evaluate(*build(zero_particle_charges=True), groups=(3,))
    lj = evaluate(*build(zero_particle_charges=True, zero_exception_charges=True), groups=(3,))

    c = run["components"]
    ours_forces = -np.array(run["gradients"])
    ours_ewald_forces = -np.array(run["ewaldGradients"])
    theirs_ewald_forces = full["forces"] - rest["forces"]
    one_four_ours = c["electrostatics"] - run["ewaldEnergy"]
    rows = {
        "bonds": (c["bonds"], full[0]),
        "angles": (c["angles"], full[1]),
        "torsions": (c["proper_torsions"] + c["improper_torsions"], full[2]),
        "lennardJones": (c["van_der_waals"], lj[3]),
        "oneFourCoulomb": (one_four_ours, rest[3] - lj[3]),
        "ewald": (run["ewaldEnergy"], full[3] - rest[3]),
        "total": (run["total"], full["total"]),
    }
    report = {
        "schemaVersion": 1,
        "mode": "openmm",
        "openmmVersion": mm.__version__,
        "platform": platform_name,
        "atoms": n,
        "cutoffAngstrom": run["cutoffAngstrom"],
        "pme": {"alphaPerNm": used_alpha, "grid": list(used_grid), "order": 5},
        "tolerances": TOL,
        "energiesKcalMol": {
            name: {"glysys": ours, "openmm": theirs, "difference": ours - theirs}
            for name, (ours, theirs) in rows.items()
        },
        "energyDifferencePerAtom": (run["total"] - full["total"]) / n,
        "ewaldEnergyDifferencePerAtom": (rows["ewald"][0] - rows["ewald"][1]) / n,
        "forcesTotal": compare_forces(ours_forces, full["forces"]),
        "forcesEwald": compare_forces(ours_ewald_forces, theirs_ewald_forces),
    }
    failures = []
    if abs(report["energyDifferencePerAtom"]) > TOL["energy_per_atom_kcal_mol"]:
        failures.append("total energy per atom")
    if abs(report["ewaldEnergyDifferencePerAtom"]) > TOL["energy_per_atom_kcal_mol"]:
        failures.append("Ewald energy per atom")
    for key in ("forcesTotal", "forcesEwald"):
        if report[key]["rmsRatio"] > TOL["force_rms_ratio"]:
            failures.append(f"{key} RMS ratio")
    report["failures"] = failures
    return report


def glysys_convergence(run, reference):
    """The Ewald part of `run` against a converged GlySys evaluation."""
    n = run["atoms"]
    if reference["atoms"] != n or not np.allclose(run["coordinates"], reference["coordinates"], atol=0):
        sys.exit("the two evaluations are not of the same configuration")
    forces = compare_forces(-np.array(run["ewaldGradients"]), -np.array(reference["ewaldGradients"]))
    return {
        "schemaVersion": 1,
        "mode": "convergence",
        "atoms": n,
        "run": {"cutoffAngstrom": run["cutoffAngstrom"], **run["pme"]},
        "reference": {"cutoffAngstrom": reference["cutoffAngstrom"], **reference["pme"]},
        "ewaldEnergyKcalMol": {
            "run": run["ewaldEnergy"],
            "reference": reference["ewaldEnergy"],
            "difference": run["ewaldEnergy"] - reference["ewaldEnergy"],
            "differencePerAtom": (run["ewaldEnergy"] - reference["ewaldEnergy"]) / n,
        },
        "forcesEwald": forces,
        "failures": [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("glysys_json")
    parser.add_argument("--prmtop", help="Amber topology of the same system (OpenMM parity)")
    parser.add_argument("--reference", help="converged pme_snapshot JSON (accuracy of a setting)")
    parser.add_argument("--platform", default="Reference", choices=["Reference", "CPU"])
    parser.add_argument("--minimize", metavar="OUT_JSON",
                        help="with --prmtop: write an OpenMM-minimized configuration instead")
    parser.add_argument("--iterations", type=int, default=2000, help="minimizer iterations")
    parser.add_argument("--threads", type=int, default=8, help="threads of the CPU platform")
    parser.add_argument("--report", help="also write the report JSON to this file")
    args = parser.parse_args()
    if bool(args.prmtop) == bool(args.reference):
        parser.error("give exactly one of --prmtop and --reference")
    run = json.load(open(args.glysys_json))
    if args.minimize:
        report = minimize(run, args.prmtop, args.minimize, args.iterations, args.threads)
    elif args.prmtop:
        report = openmm_parity(run, args.prmtop, args.platform, args.threads)
    else:
        report = glysys_convergence(run, json.load(open(args.reference)))
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        open(args.report, "w").write(text + "\n")
    if report["failures"]:
        sys.exit("FAILED: " + ", ".join(report["failures"]))


if __name__ == "__main__":
    main()
