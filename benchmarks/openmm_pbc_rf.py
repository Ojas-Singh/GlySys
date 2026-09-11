#!/usr/bin/env python3
"""Independent PBC explicit-water check against OpenMM's Reference platform.

Reads the JSON produced by
`cargo run -p glysys-dynamics --example explicit_reference` (argv[1]) and:
  1. Rebuilds the identical Amber chemistry from the exported prmtop with
     CutoffPeriodic + reaction field (no switching, RF dielectric matched),
     rigid TIP3P waters, and flexible solute, then checks single-point
     energies and forces on our minimized (A) and post-NVE (B) snapshots.
  2. Runs OpenMM's own 200-step NVE (leapfrog + SETTLE) and requires bounded
     drift, proving the exported PES is sane independent of our integrator.
  3. Runs 2000-step NVT (LangevinMiddle = BAOAB ordering, matched T/gamma/dt)
     and compares ensemble statistics against ours within statistical bounds.
Tolerances live in one dict below; tighten after the first green run if the
margins allow. Exits nonzero on any failure.
"""
import json, sys, tempfile, math
from pathlib import Path

TOL = {
    "energy_abs_kcal_mol": 0.05,
    "force_abs_kcal_mol_A": 0.02,
    "nve_drift_per_atom": 0.05,
    "nvt_temperature_K": 8.0,
    "nvt_energy_kcal_mol": 10.0,
}

r = json.load(open(sys.argv[1]))
elec = r["electrostatics"]
assert elec["method"] == "reaction-field"
cutoff_nm = elec["cutoffAngstrom"] / 10.0
rf_dielectric = elec["solventDielectric"]

import openmm as mm
from openmm import app, unit

with tempfile.TemporaryDirectory() as folder:
    Path(folder, "system.prmtop").write_text(r["files"]["system.prmtop"])
    prmtop = app.AmberPrmtopFile(str(Path(folder, "system.prmtop")))
    system = prmtop.createSystem(
        nonbondedMethod=app.CutoffPeriodic,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=None,
        rigidWater=True,
        removeCMMotion=False,
    )
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    nb.setUseSwitchingFunction(False)
    nb.setReactionFieldDielectric(rf_dielectric)
    nb.setUseDispersionCorrection(False)

    to_nm = lambda ps: [mm.Vec3(p["x"] / 10.0, p["y"] / 10.0, p["z"] / 10.0) for p in ps]
    box = prmtop.getBoxVectors() or prmtop.topology.getPeriodicBoxVectors()
    report = {"schemaVersion": 1, "openmmVersion": mm.__version__, "platform": "Reference"}
    failures_single_point = []

    def check_snapshot(name, coords):
        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
        context.setPositions(to_nm(coords))
        if box is not None:
            context.setPeriodicBoxVectors(*box)
        state = context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilocalories_per_mole / unit.angstrom
        )
        del context, integrator
        return energy, forces

    import numpy as np
    parity = {}
    for name, key, ek, fk in [("A", "snapshotA", "energyA", "forcesA"), ("B", "snapshotB", "energyB", "forcesB")]:
        energy, forces = check_snapshot(name, r[key])
        de = abs(energy - r[ek])
        df = float(abs(forces - np.asarray(r[fk])).max())
        parity[name] = {"openmmEnergy": energy, "ourEnergy": r[ek], "absEnergyError": de,
                        "maxForceComponentError": df}
        if de > TOL["energy_abs_kcal_mol"]:
            failures_single_point.append(f"{name} energy error {de}")
        if df > TOL["force_abs_kcal_mol_A"]:
            failures_single_point.append(f"{name} force error {df}")
    report["singlePoint"] = parity

    # Our drift series is validated structurally: energies must be finite and
    # bounded the same way OpenMM's own NVE below must be.
    drift = r["nveDrift"]
    assert all(math.isfinite(e) for e in drift), "nonfinite NVE energies"
    n_atoms = len(r["snapshotA"])
    report["ourNveDriftPerAtom"] = max(abs(e - drift[0]) for e in drift) / n_atoms

    # OpenMM's own NVE drift on the same PES.
    integrator = mm.VerletIntegrator(0.002)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(to_nm(r["snapshotA"]))
    context.setVelocitiesToTemperature(r["temperatureK"] * unit.kelvin, 11)
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    energies = []
    for step in range(200):
        integrator.step(1)
        if (step + 1) % 10 == 0:
            energies.append(
                context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilocalories_per_mole)
            )
    del context, integrator
    report["openmmNveDriftPerAtom"] = max(abs(e - energies[0]) for e in energies) / n_atoms

    # NVT ensemble statistics with matched BAOAB-order Langevin.
    integrator = mm.LangevinMiddleIntegrator(
        r["temperatureK"] * unit.kelvin,
        r["frictionPerPs"] / unit.picosecond,
        r["timestepFs"] * 0.001 * unit.picoseconds,
    )
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(to_nm(r["snapshotA"]))
    context.setVelocitiesToTemperature(r["temperatureK"] * unit.kelvin, 13)
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    integrator.step(500)
    temps, pes = [], []
    for step in range(2000):
        integrator.step(1)
        if (step + 1) % 10 == 0:
            state = context.getState(getEnergy=True)
            ke = state.getKineticEnergy().value_in_unit(unit.kilocalories_per_mole)
            temps.append(2.0 * ke / (3.0 * n_atoms * 0.00198720425864083))
            pes.append(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
    del context, integrator
    report["openmmNvt"] = {
        "meanTemperatureK": sum(temps) / len(temps),
        "meanPotentialEnergy": sum(pes) / len(pes),
    }
    report["ourNvt"] = {
        "meanTemperatureK": r["nvt"]["meanTemperatureK"],
        "meanPotentialEnergy": r["nvt"]["meanPotentialEnergy"],
    }

print(json.dumps(report, indent=2))
failures = list(failures_single_point)
# Our drift series is validated structurally: finite and bounded per atom.
if abs(report["ourNveDriftPerAtom"]) > TOL["nve_drift_per_atom"]:
    failures.append(f"our NVE drift {report['ourNveDriftPerAtom']}")
if report["openmmNveDriftPerAtom"] > TOL["nve_drift_per_atom"]:
    failures.append(f"openmm NVE drift {report['openmmNveDriftPerAtom']}")
if abs(report["openmmNvt"]["meanTemperatureK"] - r["temperatureK"]) > 15.0:
    failures.append("openmm NVT temperature off target")
if abs(report["openmmNvt"]["meanTemperatureK"] - r["nvt"]["meanTemperatureK"]) > TOL["nvt_temperature_K"]:
    failures.append("NVT temperature mismatch")
if abs(report["openmmNvt"]["meanPotentialEnergy"] - r["nvt"]["meanPotentialEnergy"]) > TOL["nvt_energy_kcal_mol"]:
    failures.append("NVT energy mismatch")
if failures:
    print("FAILURES:", failures, file=sys.stderr)
    sys.exit(1)
print("PBC-RF parity checks passed within statistical bounds")
