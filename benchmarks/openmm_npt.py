#!/usr/bin/env python3
"""Small disposable OpenMM oracle for the optional GlySys NPT export.

The script deliberately uses only OpenMM's Reference platform.  It rebuilds
the Amber topology from the exported prmtop, installs the supplied GlySys
coordinates/velocities, and compares production means from the same
CutoffPeriodic + reaction-field + dispersion model.  OpenMM does not expose
Monte Carlo barostat acceptance counters, so those are reported for GlySys
only; volume, density, energy and temperature are the parity observables.
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit


if len(sys.argv) != 2:
    raise SystemExit("usage: openmm_npt.py EXPORT.json")

data = json.loads(Path(sys.argv[1]).read_text())
npt = data.get("npt")
if not npt:
    raise SystemExit("export has no npt section; set GLYSYS_REF_NPT=1")

with tempfile.TemporaryDirectory() as folder:
    prmtop_path = Path(folder) / "system.prmtop"
    prmtop_path.write_text(data["files"]["system.prmtop"])
    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    system = prmtop.createSystem(
        nonbondedMethod=app.CutoffPeriodic,
        nonbondedCutoff=float(data["electrostatics"]["cutoffAngstrom"]) / 10.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=False,
        removeCMMotion=False,
    )
    # Close the TIP3P triangle exactly as the GlySys constraint model does.
    for residue in prmtop.topology.residues():
        if residue.name not in ("HOH", "WAT"):
            continue
        hydrogens = [a.index for a in residue.atoms() if a.element and a.element.symbol == "H"]
        if len(hydrogens) != 2:
            raise RuntimeError(f"unsupported water residue {residue}")
        d = 2 * 0.09572 * math.sin(math.radians(104.52) / 2)
        # `d` is already in nanometres (the O-H target above is in nm).
        system.addConstraint(hydrogens[0], hydrogens[1], d * unit.nanometer)
    nonbonded = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    nonbonded.setUseSwitchingFunction(False)
    nonbonded.setReactionFieldDielectric(float(data["electrostatics"]["solventDielectric"]))
    dispersion_enabled = bool(npt.get("dispersionCorrection", True))
    nonbonded.setUseDispersionCorrection(dispersion_enabled)

    def openmm_dispersion_coefficient(force, cutoff_nm):
        """OpenMM's NonbondedForceImpl::calcDispersionCorrection formula."""
        classes = {}
        for index in range(force.getNumParticles()):
            _, sigma, epsilon = force.getParticleParameters(index)
            key = (
                sigma.value_in_unit(unit.nanometer),
                epsilon.value_in_unit(unit.kilojoule_per_mole),
            )
            classes[key] = classes.get(key, 0) + 1
        entries = list(classes.items())
        sum1 = sum2 = 0.0
        for i, ((sigma, epsilon), count_i) in enumerate(entries):
            count = count_i * (count_i + 1) / 2.0
            sigma6 = sigma ** 6
            sum1 += count * epsilon * sigma6 * sigma6
            sum2 += count * epsilon * sigma6
            for (sigma_j, epsilon_j), count_j in entries[:i]:
                pair_sigma = 0.5 * (sigma + sigma_j)
                pair_epsilon = math.sqrt(epsilon * epsilon_j)
                pair_count = count_i * count_j
                pair_sigma6 = pair_sigma ** 6
                sum1 += pair_count * pair_epsilon * pair_sigma6 * pair_sigma6
                sum2 += pair_count * pair_epsilon * pair_sigma6
        n_particles = float(force.getNumParticles())
        normalization = n_particles * (n_particles + 1.0) / 2.0
        sum1 /= normalization
        sum2 /= normalization
        coefficient_kj_nm3 = 8.0 * n_particles * n_particles * math.pi * (
            sum1 / (9.0 * cutoff_nm ** 9) - sum2 / (3.0 * cutoff_nm ** 3)
        )
        # kJ/mol nm^3 -> kcal/mol A^3.
        return coefficient_kj_nm3 / 4.184 * 1000.0

    cutoff_nm = float(data["electrostatics"]["cutoffAngstrom"]) / 10.0
    openmm_dispersion = openmm_dispersion_coefficient(nonbonded, cutoff_nm)
    glysys_dispersion = float(npt.get("dispersionCoefficientKcalA3", float("nan")))
    if dispersion_enabled and not math.isfinite(glysys_dispersion):
        raise SystemExit("GlySys NPT export is missing its dispersion coefficient")
    # Amber prmtop stores parameters at lower precision than the in-memory
    # builder; the resulting coefficient difference is sub-millikcal at the
    # fixture volume. Keep a tight absolute gate while allowing that rounding.
    if dispersion_enabled and abs(openmm_dispersion - glysys_dispersion) > 1e-2:
        raise SystemExit(
            f"dispersion coefficient mismatch {openmm_dispersion} vs {glysys_dispersion}"
        )

    box = data["boxAngstrom"]
    vectors = (
        mm.Vec3(float(box[0]) / 10.0, 0, 0),
        mm.Vec3(0, float(box[1]) / 10.0, 0),
        mm.Vec3(0, 0, float(box[2]) / 10.0),
    )
    temperature = float(data["temperatureK"])
    pressure = float(npt["pressureBar"])
    interval = int(data.get("barostatInterval", 25))
    barostat = mm.MonteCarloBarostat(pressure * unit.bar, temperature * unit.kelvin, interval)
    barostat.setRandomNumberSeed(11)
    system.addForce(barostat)
    integrator = mm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        float(data["frictionPerPs"]) / unit.picosecond,
        float(data["timestepFs"]) * 0.001 * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(11)
    reference_platform = mm.Platform.getPlatformByName("Reference")
    context = mm.Context(system, integrator, reference_platform)
    context.setPeriodicBoxVectors(*vectors)
    context.setPositions(
        [mm.Vec3(p["x"] / 10.0, p["y"] / 10.0, p["z"] / 10.0) for p in data["snapshotA"]]
    )
    if data.get("velocitiesA"):
        context.setVelocities(
            [mm.Vec3(v["x"] / 10.0, v["y"] / 10.0, v["z"] / 10.0) for v in data["velocitiesA"]]
        )
    else:
        context.setVelocitiesToTemperature(temperature * unit.kelvin, 11)

    warmup = int(npt.get("warmupSteps", 0))
    equil = int(npt["equilibrationSteps"])
    production = int(npt["productionSteps"])
    sample_every = max(1, int(npt.get("saveEvery", 25)))
    integrator.step(warmup)
    integrator.step(equil)
    temps, energies, volumes, densities, pressures = [], [], [], [], []
    # The exported system has the same explicit water triangle and solute
    # X-H constraints as GlySys.  Kinetic temperature must use the actual
    # constrained coordinate count; 3N would bias every rigid-water result.
    degrees_of_freedom = max(1, 3 * system.getNumParticles() - system.getNumConstraints())
    masses = sum(system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles()))

    # Stable covalent components are the molecule groups used by the
    # molecule-preserving barostat and pressure estimator.  This is built from
    # topology bonds, so waters and isolated ions are single groups while a
    # covalently attached glycan remains connected to its receptor.
    parent = list(range(system.getNumParticles()))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for bond in prmtop.topology.bonds():
        union(bond[0].index, bond[1].index)
    molecule_map = {}
    for index in range(system.getNumParticles()):
        molecule_map.setdefault(find(index), []).append(index)
    molecule_groups = list(molecule_map.values())
    pressure_probe_integrator = mm.VerletIntegrator(0.001)
    pressure_probe_context = mm.Context(system, pressure_probe_integrator, reference_platform)
    pressure_conversion = 4184.0 * 1e25 / 6.02214076e23

    def molecular_pressure(positions_a, velocities_a, box_a):
        """OpenMM-side copy of GlySys's non-mutating molecular pressure probe."""
        volume_a3 = float(np.prod(box_a))

        def trial_energy(factor):
            trial = positions_a.copy()
            for group in molecule_groups:
                center = positions_a[group].mean(axis=0)
                trial[group] += (factor - 1.0) * center
            trial_box = box_a * factor
            pressure_probe_context.setPeriodicBoxVectors(
                mm.Vec3(trial_box[0] / 10.0, 0, 0),
                mm.Vec3(0, trial_box[1] / 10.0, 0),
                mm.Vec3(0, 0, trial_box[2] / 10.0),
            )
            pressure_probe_context.setPositions(
                [mm.Vec3(p[0] / 10.0, p[1] / 10.0, p[2] / 10.0) for p in trial]
            )
            return pressure_probe_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilocalories_per_mole
            )

        perturb = 1.0e-3
        e_plus = trial_energy(1.0 + perturb)
        e_minus = trial_energy(1.0 - perturb)
        d_u_d_v = (e_plus - e_minus) / (
            volume_a3 * ((1.0 + perturb) ** 3 - (1.0 - perturb) ** 3)
        )
        kinetic_com = 0.0
        masses_array = np.array(
            [system.getParticleMass(i).value_in_unit(unit.dalton)
             for i in range(system.getNumParticles())], dtype=float
        )
        for group in molecule_groups:
            mass = masses_array[group]
            total_mass = float(mass.sum())
            velocity = (velocities_a[group] * mass[:, None]).sum(axis=0) / total_mass
            kinetic_com += total_mass * float(np.dot(velocity, velocity)) / (2.0 * 418.4)
        return (2.0 * kinetic_com / (3.0 * volume_a3) - d_u_d_v) * pressure_conversion
    for step in range(production):
        integrator.step(1)
        if (step + 1) % sample_every:
            continue
        state = context.getState(getEnergy=True, getPositions=True, getVelocities=True)
        pe = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        ke = state.getKineticEnergy().value_in_unit(unit.kilocalories_per_mole)
        temp = 2.0 * ke / (degrees_of_freedom * 0.00198720425864083)
        bx, by, bz = state.getPeriodicBoxVectors()
        volume = (bx[0] * by[1] * bz[2]).value_in_unit(unit.nanometer**3) * 1000.0
        density = masses * 1.66053906660 / volume
        positions_a = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=float) * 10.0
        velocities_a = np.asarray(state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond), dtype=float) * 10.0
        box_a = np.array([
            bx[0].value_in_unit(unit.nanometer),
            by[1].value_in_unit(unit.nanometer),
            bz[2].value_in_unit(unit.nanometer),
        ], dtype=float) * 10.0
        pressure_value = molecular_pressure(positions_a, velocities_a, box_a)
        temps.append(temp)
        energies.append(pe)
        volumes.append(volume)
        densities.append(density)
        pressures.append(pressure_value)

def mean(values):
    return float(np.mean(values)) if values else float("nan")


def finite_series(name, values):
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise SystemExit(f"OpenMM NPT oracle produced no finite {name} samples")


for name, values in (("temperature", temps), ("potential energy", energies),
                     ("volume", volumes), ("density", densities),
                     ("molecular pressure", pressures)):
    finite_series(name, values)

if not all(math.isfinite(float(npt.get(key, float("nan")))) for key in
           ("meanTemperatureK", "meanPotentialEnergy", "meanVolumeA3", "meanDensityGMl", "meanPressureBar")):
    raise SystemExit("GlySys NPT export contains non-finite summary values")

report = {
    "openmmVersion": mm.__version__,
    "platform": "Reference",
    "samples": len(temps),
    "meanTemperatureK": mean(temps),
    "meanPotentialEnergy": mean(energies),
    "meanVolumeA3": mean(volumes),
        "meanDensityGMl": mean(densities),
        "meanPressureBar": mean(pressures),
        "dispersionCoefficientKcalA3": openmm_dispersion,
        "dispersionCoefficientDifferenceKcalA3": abs(openmm_dispersion - glysys_dispersion),
        "glysys": {
        "meanTemperatureK": npt["meanTemperatureK"],
        "meanPotentialEnergy": npt["meanPotentialEnergy"],
        "meanVolumeA3": npt["meanVolumeA3"],
        "meanDensityGMl": npt["meanDensityGMl"],
        "meanPressureBar": npt["meanPressureBar"],
        "barostatAttempts": npt["barostatAttempts"],
        "barostatAccepts": npt["barostatAccepts"],
        "dispersionCorrection": dispersion_enabled,
            "degreesOfFreedom": degrees_of_freedom,
            "dispersionCoefficientKcalA3": glysys_dispersion,
        },
}
report["differences"] = {
    "temperatureK": abs(report["meanTemperatureK"] - report["glysys"]["meanTemperatureK"]),
    "potentialEnergyKcalMolPerAtom": abs(report["meanPotentialEnergy"] - report["glysys"]["meanPotentialEnergy"]) / len(data["snapshotA"]),
    "volumeFraction": abs(report["meanVolumeA3"] - report["glysys"]["meanVolumeA3"]) / report["meanVolumeA3"],
    "densityGMl": abs(report["meanDensityGMl"] - report["glysys"]["meanDensityGMl"]),
    "pressureBar": abs(report["meanPressureBar"] - report["glysys"]["meanPressureBar"]),
}
report["productionTimePs"] = float(npt["productionSteps"]) * float(data["timestepFs"]) * 0.001
if report["samples"] < 20:
    report["status"] = "inconclusive"
    report["inconclusiveReason"] = "fewer than 20 production samples; extend the run before judging NPT statistics"
elif report["productionTimePs"] < 100.0:
    report["status"] = "inconclusive"
    report["inconclusiveReason"] = "less than 100 ps of production; finite-volume pressure and density means are underpowered"
else:
    report["status"] = "ok"
print(json.dumps(report, indent=2))
