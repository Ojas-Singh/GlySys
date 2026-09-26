#!/usr/bin/env python3
"""Run an independent 1CRN NVT leg from a GlySys step-zero checkpoint.

This is a validation/benchmark helper, not a GlySys runtime dependency. It
uses the Amber topology exported by GlySys and the exact minimized starting
coordinates/velocities serialized by ``glysys-md run`` before step 1.
"""
import argparse
import json
import math
import time
from pathlib import Path

import openmm as mm
from openmm import app, unit
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path,
                        help="GlySys prepared system directory")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="step-zero GlySys checkpoint.json")
    parser.add_argument("--protocol", required=True, type=Path,
                        help="matching benchmarks/dynamics protocol JSON")
    parser.add_argument("--mode", choices=("explicit", "implicit"), required=True)
    parser.add_argument("--platform", choices=("OpenCL", "CPU", "Reference"), default="OpenCL")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--threads", type=int, default=None,
                        help="OpenMM CPU platform thread count")
    parser.add_argument("--seed", type=int, default=1701,
                        help="LangevinMiddleIntegrator random seed")
    parser.add_argument("--equilibration-steps", type=int, default=None)
    parser.add_argument("--production-steps", type=int, default=None,
                        help="optional short-run override for smoke tests")
    parser.add_argument("--benchmark-seconds", type=float, default=None,
                        help="run a synchronized core-throughput window instead of a trajectory")
    parser.add_argument("--warmup-steps", type=int, default=2000,
                        help="steps excluded before a timed benchmark window")
    parser.add_argument("--observables-every", "--sample-steps", dest="observables_every",
                        type=int, default=100,
                        help="production scalar-observation interval")
    parser.add_argument("--coordinates-every", type=int, default=2500,
                        help="production structural-diagnostic interval")
    parser.add_argument("--compare-glysys-checkpoint", type=Path, default=None,
                        help="optionally compare final OpenMM subsystem temperatures with a GlySys checkpoint at the same step")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def make_system(prmtop, mode, protocol):
    if mode == "implicit":
        system = prmtop.createSystem(
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            rigidWater=False,
            implicitSolvent=app.OBC2,
            soluteDielectric=1.0,
            solventDielectric=78.5,
            removeCMMotion=False,
        )
        gbsa = next(force for force in system.getForces()
                    if isinstance(force, mm.GBSAOBCForce))
        # OpenMM's Amber default is 2.25936 kJ/mol/nm^2; GlySys uses exactly
        # 0.00542 kcal/mol/A^2 (2.267728 kJ/mol/nm^2).
        gbsa.setSurfaceAreaEnergy(
            0.00542 * unit.kilocalories_per_mole / unit.angstrom**2
        )
        return system

    cutoff = protocol["cutoffAngstrom"] * unit.angstrom
    system = prmtop.createSystem(
        nonbondedMethod=app.CutoffPeriodic,
        nonbondedCutoff=cutoff,
        constraints=app.HBonds,
        rigidWater=False,
        removeCMMotion=False,
    )
    nonbonded = next(force for force in system.getForces()
                     if isinstance(force, mm.NonbondedForce))
    nonbonded.setReactionFieldDielectric(protocol["rfDielectric"])
    nonbonded.setUseDispersionCorrection(False)
    # GlySys SETTLE mode also SHAKE/RATTLE-constrains solute X-H bonds. OpenMM
    # HBonds matches those plus both water O-H edges; add the water H-H edge
    # so the rigid-water constraint count is exactly three per TIP3P molecule.
    hh_nm = 2.0 * 0.09572 * math.sin(math.radians(104.52) / 2.0)
    for residue in prmtop.topology.residues():
        if residue.name not in ("HOH", "WAT", "TIP3"):
            continue
        atoms = list(residue.atoms())
        oxygens = [atom.index for atom in atoms
                   if atom.element is not None and atom.element.symbol == "O"]
        hydrogens = [atom.index for atom in atoms
                     if atom.element is not None and atom.element.symbol == "H"]
        if len(oxygens) != 1 or len(hydrogens) != 2:
            raise ValueError(f"water residue {residue} does not have O + 2 H atoms")
        system.addConstraint(hydrogens[0], hydrogens[1], hh_nm * unit.nanometer)
    return system


def temperature_k(state, system, mode):
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    constraints = system.getNumConstraints()
    dof = 3 * system.getNumParticles() - constraints
    if mode == "explicit":
        # removeCMMotion=False on both systems.
        pass
    return 2.0 * kinetic / (dof * unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin))


def subsystem_temperatures(velocities_nm_ps, system, topology):
    """Instantaneous kinetic temperatures for solute, water, and ions.

    Constraint DOFs are subtracted within their owning group; COM DOFs are
    retained, matching the qualification's removeCMMotion=False convention.
    """
    solvent_names = {"HOH", "WAT", "TIP3"}
    ion_names = {"NA", "CL", "SOD", "CLA"}
    groups = {"solute": [], "water": [], "ions": []}
    atom_group = {}
    for atom in topology.atoms():
        name = atom.residue.name.upper()
        group = ("water" if name in solvent_names else
                 "ions" if name in ion_names else "solute")
        groups[group].append(atom.index)
        atom_group[atom.index] = group

    constraint_counts = {name: 0 for name in groups}
    for constraint_index in range(system.getNumConstraints()):
        atom_a, atom_b, _ = system.getConstraintParameters(constraint_index)
        group_a = atom_group[atom_a]
        group_b = atom_group[atom_b]
        if group_a == group_b:
            constraint_counts[group_a] += 1

    result = {}
    gas_constant = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin
    )
    velocities_nm_ps = np.asarray(velocities_nm_ps, dtype=np.float64)
    for name, indices in groups.items():
        masses = np.array([
            system.getParticleMass(index).value_in_unit(unit.dalton)
            for index in indices
        ], dtype=np.float64)
        dof = 3 * len(indices) - constraint_counts[name]
        kinetic = 0.5 * np.sum(masses[:, None] * velocities_nm_ps[indices] ** 2)
        result[name] = {
            "atoms": len(indices),
            "constraints": constraint_counts[name],
            "degreesOfFreedom": dof,
            "temperatureK": float(2.0 * kinetic / (dof * gas_constant)),
        }
    return result


def protein_heavy_indices(topology):
    nonprotein = {"HOH", "WAT", "TIP3", "NA", "CL", "SOD", "CLA"}
    return [atom.index for atom in topology.atoms()
            if atom.residue.name not in nonprotein
            and atom.element is not None and atom.element.symbol != "H"]


def structural_metrics(reference, positions):
    reference = np.asarray(reference, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    ref_centered = reference - reference.mean(axis=0)
    pos_centered = positions - positions.mean(axis=0)
    left, _, right = np.linalg.svd(pos_centered.T @ ref_centered)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    aligned = pos_centered @ left @ correction @ right
    rmsd = np.sqrt(np.mean(np.sum((aligned - ref_centered) ** 2, axis=1)))
    rg = np.sqrt(np.mean(np.sum(pos_centered ** 2, axis=1)))
    return float(rmsd), float(rg)


def main():
    args = parse_args()
    if mm.__version__ != "8.1.1":
        raise RuntimeError(f"expected OpenMM 8.1.1, found {mm.__version__}")
    if int(np.__version__.split(".")[0]) >= 2:
        raise RuntimeError("this validated environment requires numpy<2")
    protocol = json.loads(args.protocol.read_text())
    if args.equilibration_steps is not None:
        protocol["equilibrationSteps"] = args.equilibration_steps
    if args.production_steps is not None:
        protocol["productionSteps"] = args.production_steps
    checkpoint = json.loads(args.checkpoint.read_text())["state"]
    if checkpoint["step"] != 0:
        raise ValueError(
            "OpenMM comparison requires the preserved step-zero reference checkpoint; "
            f"{args.checkpoint} is at step {checkpoint['step']}"
        )
    prmtop = app.AmberPrmtopFile(str(args.input / "system.prmtop"))
    system = make_system(prmtop, args.mode, protocol)
    heavy_indices = protein_heavy_indices(prmtop.topology)
    if not heavy_indices:
        raise ValueError("prepared topology has no protein heavy atoms")
    reference = np.array([
        [checkpoint["coordinates"][i][axis] for axis in ("x", "y", "z")]
        for i in heavy_indices
    ], dtype=np.float64)
    if args.observables_every <= 0 or args.coordinates_every <= 0:
        raise ValueError("observation intervals must be positive")

    platform = mm.Platform.getPlatformByName(args.platform)
    properties = {}
    if args.platform == "OpenCL":
        properties = {"DeviceIndex": args.device_index, "Precision": "mixed"}
    elif args.platform == "CPU" and args.threads is not None:
        properties = {"Threads": str(args.threads)}
    integrator = mm.LangevinMiddleIntegrator(
        protocol["temperatureK"] * unit.kelvin,
        protocol["frictionPerPs"] / unit.picosecond,
        protocol["timestepFs"] * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(args.seed)
    context = mm.Context(system, integrator, platform, properties)
    positions = checkpoint["coordinates"]
    velocities = checkpoint["velocities"]
    to_vec3 = lambda rows, scale: [
        mm.Vec3(row["x"] * scale, row["y"] * scale, row["z"] * scale)
        for row in rows
    ]
    context.setPositions(to_vec3(positions, 0.1))
    box = checkpoint["boxAngstrom"]
    if args.mode == "explicit":
        vectors = tuple(mm.Vec3(*(value * 0.1 if axis == j else 0.0
                                  for axis in range(3)))
                        for j, value in enumerate(box))
        context.setPeriodicBoxVectors(*vectors)
    context.setVelocities(to_vec3(velocities, 0.1))
    context.applyConstraints(1e-5)
    context.applyVelocityConstraints(1e-5)

    if args.benchmark_seconds is not None:
        if not math.isfinite(args.benchmark_seconds) or args.benchmark_seconds <= 0:
            raise ValueError("--benchmark-seconds must be finite and positive")
        if args.warmup_steps < 0:
            raise ValueError("--warmup-steps must be nonnegative")
        warmup_started = time.perf_counter()
        if args.warmup_steps:
            integrator.step(args.warmup_steps)
        # A state request completes queued OpenMM work before the timed window.
        context.getState(getPositions=True)
        warmup_seconds = time.perf_counter() - warmup_started
        completed = 0
        started = time.perf_counter()
        while time.perf_counter() - started < args.benchmark_seconds:
            elapsed_before = time.perf_counter() - started
            remaining = max(args.benchmark_seconds - elapsed_before, 0.0)
            if completed == 0:
                # Calibrate cheaply before choosing a bounded work chunk. A
                # fixed 1,000-step batch can overshoot a five-second CPU
                # window by many seconds on explicit solvent.
                count = 8
            else:
                observed_rate = completed / max(elapsed_before, 1e-12)
                count = max(1, min(1000, int(observed_rate * remaining * 0.9)))
            integrator.step(count)
            completed += count
            # Bound queued OpenCL work. Without an in-window synchronization,
            # the Python loop can enqueue many seconds of GPU work quickly and
            # the final readback would make the nominal window misleading.
            context.getState(getPositions=True, getVelocities=False,
                             getForces=False, getEnergy=False)
        # Include completion of all submitted GPU work in the measurement.
        context.getState(getPositions=True)
        elapsed = time.perf_counter() - started
        simulated_ns = completed * protocol["timestepFs"] * 1e-6
        try:
            device = platform.getPropertyValue(context, "DeviceName")
        except Exception:
            device = args.platform
        result = {
            "schemaVersion": 1,
            "measurement": "core-throughput",
            "openmmVersion": mm.__version__,
            "mode": args.mode,
            "platform": args.platform,
            "device": device,
            "precision": properties.get("Precision", "n/a"),
            "atoms": system.getNumParticles(),
            "constraints": system.getNumConstraints(),
            "temperatureK": protocol["temperatureK"],
            "timestepFs": protocol["timestepFs"],
            "frictionPerPs": protocol["frictionPerPs"],
            "randomNumberSeed": args.seed,
            "warmupSteps": args.warmup_steps,
            "warmupSeconds": warmup_seconds,
            "steps": completed,
            "simulatedNs": simulated_ns,
            "simulationSeconds": elapsed,
            "stepsPerSecond": completed / max(elapsed, 1e-12),
            "nsPerDay": simulated_ns / max(elapsed, 1e-12) * 86400.0,
            "synchronizationEverySteps": "adaptive; maximum 1000",
            "targetWindowSeconds": args.benchmark_seconds,
            "synchronizationPolicy": "final position readback included before stopping timer",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        del context, integrator
        return

    initial_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilocalories_per_mole)
    glysys_initial_energy = checkpoint["potentialEnergy"]
    started = time.perf_counter()
    equilibration_steps = protocol["equilibrationSteps"]
    if equilibration_steps:
        integrator.step(equilibration_steps)
    equilibration_seconds = time.perf_counter() - started

    production_steps = protocol["productionSteps"]
    sample_steps = args.observables_every
    samples = []
    structure_samples = []
    completed = 0
    production_started = time.perf_counter()
    while completed < production_steps:
        count = min(sample_steps, production_steps - completed)
        integrator.step(count)
        completed += count
        structure_due = (completed % args.coordinates_every == 0
                         or completed == production_steps)
        state = context.getState(getEnergy=True, getVelocities=True,
                                 getPositions=structure_due)
        samples.append({
            "step": completed,
            "absoluteStep": equilibration_steps + completed,
            "timePs": completed * protocol["timestepFs"] * 0.001,
            "potentialEnergyKcalMol": state.getPotentialEnergy().value_in_unit(
                unit.kilocalories_per_mole),
            "kineticEnergyKcalMol": state.getKineticEnergy().value_in_unit(
                unit.kilocalories_per_mole),
            "temperatureK": temperature_k(state, system, args.mode),
        })
        if structure_due:
            positions_angstrom = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            rmsd, rg = structural_metrics(reference, positions_angstrom[heavy_indices])
            structure_samples.append({
                "step": completed,
                "absoluteStep": equilibration_steps + completed,
                "proteinHeavyRmsdAngstrom": rmsd,
                "proteinHeavyRgAngstrom": rg,
            })
        if len(samples) == 1 or completed == production_steps or len(samples) % 10 == 0:
            print(
                f"OpenMM {args.mode} {args.platform} step={completed}/{production_steps}",
                flush=True,
            )
    production_seconds = time.perf_counter() - production_started
    simulated_ns = production_steps * protocol["timestepFs"] * 1e-6
    try:
        device = platform.getPropertyValue(context, "DeviceName")
    except Exception:
        device = args.platform
    result = {
        "schemaVersion": 1,
        "openmmVersion": mm.__version__,
        "mode": args.mode,
        "platform": args.platform,
        "device": device,
        "precision": properties.get("Precision", "n/a"),
        "atoms": system.getNumParticles(),
        "constraints": system.getNumConstraints(),
        "temperatureK": protocol["temperatureK"],
        "timestepFs": protocol["timestepFs"],
        "frictionPerPs": protocol["frictionPerPs"],
        "randomNumberSeed": args.seed,
        "thermostat": "langevin-middle",
        "equilibrationSteps": equilibration_steps,
        "equilibrationPs": equilibration_steps * protocol["timestepFs"] * 0.001,
        "equilibrationSeconds": equilibration_seconds,
        "productionSteps": production_steps,
        "productionNs": simulated_ns,
        "productionSeconds": production_seconds,
        "stepsPerSecond": production_steps / max(production_seconds, 1e-12),
        "nsPerDay": simulated_ns / max(production_seconds, 1e-12) * 86400.0,
        "observablesEvery": args.observables_every,
        "coordinatesEvery": args.coordinates_every,
        "proteinHeavyAtomCount": len(heavy_indices),
        "proteinHeavyAtomIndices": heavy_indices,
        "initialPotentialEnergyKcalMol": initial_energy,
        "glysysInitialPotentialEnergyKcalMol": glysys_initial_energy,
        "initialEnergyDifferenceKcalMol": initial_energy - glysys_initial_energy,
        "samples": samples,
        "structureSamples": structure_samples,
    }
    if args.compare_glysys_checkpoint is not None:
        glysys_checkpoint = json.loads(args.compare_glysys_checkpoint.read_text())["state"]
        final_step = equilibration_steps + production_steps
        if glysys_checkpoint["step"] != final_step:
            raise ValueError(
                "GlySys diagnostic checkpoint must match the OpenMM final step "
                f"({final_step}), got {glysys_checkpoint['step']}"
            )
        openmm_velocities = state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )
        glysys_velocities = np.array([
            [row[axis] * 0.1 for axis in ("x", "y", "z")]
            for row in glysys_checkpoint["velocities"]
        ], dtype=np.float64)
        result["finalSubsystemTemperatures"] = {
            "step": final_step,
            "openmm": subsystem_temperatures(openmm_velocities, system, prmtop.topology),
            "glysys": subsystem_temperatures(glysys_velocities, system, prmtop.topology),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    del context, integrator


if __name__ == "__main__":
    main()
