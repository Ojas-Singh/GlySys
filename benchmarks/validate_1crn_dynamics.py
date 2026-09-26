#!/usr/bin/env python3
"""Validate paired 1CRN NVT observables with predeclared equivalence margins."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def block_stats(values, block_size=50):
    count = len(values) // block_size
    if count < 4:
        raise ValueError(
            f"need >=4 complete 10 ps blocks ({block_size} samples each), got {count}"
        )
    blocks = [
        float(np.mean(values[i * block_size:(i + 1) * block_size]))
        for i in range(count)
    ]
    mean = float(np.mean(values))
    sem = float(np.std(blocks, ddof=1) / math.sqrt(count))
    if count > 2 and np.std(blocks[:-1]) > 0 and np.std(blocks[1:]) > 0:
        lag1 = float(np.corrcoef(blocks[:-1], blocks[1:])[0, 1])
    else:
        lag1 = 0.0
    return {
        "mean": mean,
        "sem": sem,
        "blocks": count,
        "blockSamples": block_size,
        "blockDurationPs": 10.0,
        "lagOneBlockCorrelation": lag1,
    }


def equivalence(left, right, margin):
    difference = abs(left["mean"] - right["mean"])
    ci_half_width = 1.96 * math.hypot(left["sem"], right["sem"])
    ci_width = 2.0 * ci_half_width
    if ci_width > margin:
        status = "inconclusive"
    elif difference + ci_half_width <= margin:
        status = "pass"
    elif difference - ci_half_width > margin:
        status = "fail"
    else:
        status = "inconclusive"
    return {
        "absoluteMeanDifference": difference,
        "equivalenceMargin": margin,
        "combined95PercentCiHalfWidth": ci_half_width,
        "combined95PercentCiWidth": ci_width,
        "status": status,
    }


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--glysys-run", required=True, type=Path)
    parser.add_argument("--openmm-result", required=True, type=Path)
    parser.add_argument("--expected-backend", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run = json.loads((args.glysys_run / "run.json").read_text())
    openmm = json.loads(args.openmm_result.read_text())
    protocol = run["protocol"]
    if run["backend"] != args.expected_backend or run["fallbackReason"] is not None:
        raise ValueError(
            f"GlySys qualification must complete on {args.expected_backend} without fallback; "
            f"got backend={run['backend']}, fallback={run['fallbackReason']}"
        )
    if openmm["mode"] != protocol["solvent"]:
        raise ValueError("GlySys and OpenMM solvent modes differ")
    if openmm["productionSteps"] != protocol["productionSteps"]:
        raise ValueError("GlySys and OpenMM production step counts differ")
    for key in ("temperatureK", "timestepFs", "frictionPerPs"):
        if not math.isclose(openmm[key], protocol[key], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"GlySys and OpenMM {key} differ")
    if protocol["langevinDiscretization"] != "lf-middle":
        raise ValueError("qualification requires explicit LF-middle selection")
    if protocol["thermostat"] != "langevin" or openmm["thermostat"] != "langevin-middle":
        raise ValueError("the validated comparison requires matching Langevin dynamics")
    expected_eq_ps = protocol["equilibrationSteps"] * protocol["timestepFs"] * 0.001
    if abs(openmm["equilibrationPs"] - expected_eq_ps) > 1e-9:
        raise ValueError("GlySys and OpenMM equilibration durations differ")
    if not math.isclose(run["simulatedPs"],
                        (protocol["equilibrationSteps"] + protocol["productionSteps"])
                        * protocol["timestepFs"] * 0.001,
                        rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("GlySys run did not complete the requested equilibration + production")

    observable_path = args.glysys_run / "observables.jsonl"
    if not observable_path.is_file():
        raise ValueError("GlySys run lacks observables.jsonl; rerun with --observables-every 100")
    glysys_observables = [
        json.loads(line) for line in observable_path.read_text().splitlines() if line.strip()
    ]
    production_start = protocol["equilibrationSteps"]
    glysys_observables = [
        item for item in glysys_observables
        if item["segment"] == "production" and item["step"] > production_start
    ]
    openmm_samples = openmm["samples"]
    if not openmm_samples:
        raise ValueError("OpenMM result has no production scalar samples")
    if openmm["observablesEvery"] != 100 or openmm["coordinatesEvery"] != 2500:
        raise ValueError("OpenMM sampling cadence differs from the qualification protocol")
    if run.get("observablesEvery") != 100 or protocol["saveEvery"] != 2500:
        raise ValueError("GlySys sampling cadence differs from the qualification protocol")

    # 2 fs * 100 steps = 0.2 ps/sample; 50 samples form a predeclared 10 ps block.
    block_size = max(1, round(10.0 / (protocol["timestepFs"] * 0.001 * 100)))
    scalar_fields = {
        "temperatureK": ("temperatureK", "temperatureK", 3.0),
        "potentialEnergyKcalMolPerAtom": (
            "potentialEnergyKcalMol", "potentialEnergyKcalMol",
            0.005 * openmm["atoms"],
        ),
    }
    glysys_by_step = {item["step"]: item for item in glysys_observables}
    openmm_by_step = {item["absoluteStep"]: item for item in openmm_samples}
    common_steps = sorted(set(glysys_by_step) & set(openmm_by_step))
    if len(common_steps) < block_size * 4:
        raise ValueError("too few aligned observations for four 10 ps uncertainty blocks")

    scalar_results = {}
    statuses = []
    for label, (glysys_key, openmm_key, margin) in scalar_fields.items():
        left_values = [glysys_by_step[step][glysys_key] for step in common_steps]
        right_values = [openmm_by_step[step][openmm_key] for step in common_steps]
        if label == "potentialEnergyKcalMolPerAtom":
            left_values = [value / openmm["atoms"] for value in left_values]
            right_values = [value / openmm["atoms"] for value in right_values]
            margin /= openmm["atoms"]
        left = block_stats(left_values, block_size)
        right = block_stats(right_values, block_size)
        comparison = equivalence(left, right, margin)
        if max(abs(left["lagOneBlockCorrelation"]), abs(right["lagOneBlockCorrelation"])) > 0.3:
            comparison["status"] = "inconclusive"
            comparison["reason"] = "residual 10 ps block correlation exceeds 0.3"
        midpoint_left = len(left_values) // 2
        midpoint_right = len(right_values) // 2
        glysys_stationarity = equivalence(
            block_stats(left_values[:midpoint_left], block_size),
            block_stats(left_values[midpoint_left:], block_size),
            margin,
        )
        openmm_stationarity = equivalence(
            block_stats(right_values[:midpoint_right], block_size),
            block_stats(right_values[midpoint_right:], block_size),
            margin,
        )
        if glysys_stationarity["status"] != "pass" or openmm_stationarity["status"] != "pass":
            statuses.append("inconclusive" if "inconclusive" in (
                glysys_stationarity["status"], openmm_stationarity["status"]
            ) else "fail")
        scalar_results[label] = {
            "glysys": left,
            "openmm": right,
            "equivalence": comparison,
            "firstHalfVsSecondHalfStationarity": {
                "glysys": glysys_stationarity,
                "openmm": openmm_stationarity,
            },
        }
        statuses.append(comparison["status"])

    # Structural diagnostics are reported, not used as short-trajectory pass/fail gates.
    checkpoint = json.loads(
        (args.glysys_run / "step-zero-reference-checkpoint.json").read_text()
    )["state"]
    heavy_indices = openmm["proteinHeavyAtomIndices"]
    reference = np.array([
        [checkpoint["coordinates"][i][axis] for axis in ("x", "y", "z")]
        for i in heavy_indices
    ], dtype=np.float64)
    glysys_frames = []
    with (args.glysys_run / "trajectory.jsonl").open() as trajectory:
        for line in trajectory:
            if not line.strip():
                continue
            frame = json.loads(line)
            if frame["segment"] != "production" or frame["step"] <= production_start:
                continue
            if frame["step"] % 2500 != 0 and frame["step"] != run["totalSteps"]:
                continue
            positions = np.array([
                [frame["coordinates"][i][axis] for axis in ("x", "y", "z")]
                for i in heavy_indices
            ], dtype=np.float64)
            rmsd, rg = structural_metrics(reference, positions)
            glysys_frames.append({"absoluteStep": frame["step"], "rmsdAngstrom": rmsd,
                                  "rgAngstrom": rg})
    openmm_structures = openmm["structureSamples"]
    glysys_rg = [item["rgAngstrom"] for item in glysys_frames]
    openmm_rg = [item["proteinHeavyRgAngstrom"] for item in openmm_structures]
    glysys_rmsd = [item["rmsdAngstrom"] for item in glysys_frames]
    openmm_rmsd = [item["proteinHeavyRmsdAngstrom"] for item in openmm_structures]

    energy_difference = openmm["initialEnergyDifferenceKcalMol"]
    static_limit = max(0.05, openmm["atoms"] * 1e-5)
    static_check = {
        "differenceKcalMol": energy_difference,
        "limitKcalMol": static_limit,
        "status": "pass" if abs(energy_difference) <= static_limit else "fail",
    }
    statuses.append(static_check["status"])
    overall = "pass" if all(status == "pass" for status in statuses) else (
        "fail" if "fail" in statuses else "inconclusive"
    )
    report = {
        "schemaVersion": 2,
        "status": overall,
        "mode": protocol["solvent"],
        "glysysActualBackend": run["backend"],
        "timestepFs": protocol["timestepFs"],
        "glysysBackend": run["adapter"],
        "openmmPlatform": openmm["platform"],
        "openmmDevice": openmm["device"],
        "alignedScalarObservations": len(common_steps),
        "blockDurationPs": 10.0,
        "equivalenceMargins": {
            "meanTemperatureK": 3.0,
            "meanPotentialEnergyKcalMolPerAtom": 0.005,
        },
        "staticEnergyCheck": static_check,
        "productionMeanEquivalence": scalar_results,
        "structuralDiagnostics": {
            "glysysSamples": len(glysys_frames),
            "openmmSamples": len(openmm_structures),
            "glysysMeanProteinHeavyRmsdAngstrom": float(np.mean(glysys_rmsd)) if glysys_rmsd else None,
            "openmmMeanProteinHeavyRmsdAngstrom": float(np.mean(openmm_rmsd)) if openmm_rmsd else None,
            "glysysMeanProteinHeavyRgAngstrom": float(np.mean(glysys_rg)) if glysys_rg else None,
            "openmmMeanProteinHeavyRgAngstrom": float(np.mean(openmm_rg)) if openmm_rg else None,
            "interpretation": "diagnostic only; 0.3 ns does not establish conformational convergence",
        },
        "passed": overall == "pass",
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)
    if overall != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
