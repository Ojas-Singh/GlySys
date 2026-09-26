#!/usr/bin/env python3
"""Pool predeclared 10 ps block statistics across independent 1CRN replicas."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def block_stats_replicas(replicas, block_size, block_duration_ps):
    blocks = []
    correlations = []
    values = []
    for replica in replicas:
        array = np.asarray(replica, dtype=np.float64)
        if len(array) % block_size:
            raise ValueError("replica sample count is not divisible by the 10 ps block size")
        values.extend(array.tolist())
        local_blocks = [
            float(np.mean(array[start:start + block_size]))
            for start in range(0, len(array), block_size)
        ]
        blocks.extend(local_blocks)
        if len(local_blocks) > 2 and np.std(local_blocks[:-1]) > 0 and np.std(local_blocks[1:]) > 0:
            correlations.append(float(np.corrcoef(local_blocks[:-1], local_blocks[1:])[0, 1]))
        else:
            correlations.append(0.0)
    block_array = np.asarray(blocks, dtype=np.float64)
    sem = float(np.std(block_array, ddof=1) / math.sqrt(len(block_array)))
    return {
        "mean": float(np.mean(values)),
        "sem": sem,
        "blocks": len(blocks),
        "blockSamples": block_size,
        "blockDurationPs": block_duration_ps,
        "maxAbsoluteLagOneBlockCorrelation": max(map(abs, correlations), default=0.0),
        "lagOneBlockCorrelationByReplica": correlations,
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


def read_pair(run_path, openmm_path):
    run = json.loads((run_path / "run.json").read_text())
    openmm = json.loads(openmm_path.read_text())
    protocol = run["protocol"]
    if run["backend"] != "GPU" or run["fallbackReason"] is not None:
        raise ValueError(f"{run_path} is not a no-fallback GPU qualification")
    if protocol["solvent"] not in ("explicit", "implicit") or openmm["mode"] != protocol["solvent"]:
        raise ValueError(f"solvent model mismatch in {run_path}")
    if protocol["langevinDiscretization"] != "lf-middle" or openmm["thermostat"] != "langevin-middle":
        raise ValueError("replicas must use the qualified LF-middle protocol")
    for key in ("temperatureK", "timestepFs", "frictionPerPs"):
        if not math.isclose(openmm[key], protocol[key], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"OpenMM and GlySys {key} differ in {run_path}")
    if openmm["productionSteps"] != protocol["productionSteps"]:
        raise ValueError(f"production step counts differ in {run_path}")
    if openmm["observablesEvery"] != 100 or openmm["coordinatesEvery"] != 2500:
        raise ValueError("OpenMM sampling cadence differs from the qualification protocol")
    if run.get("observablesEvery") != 100 or protocol["saveEvery"] != 2500:
        raise ValueError("GlySys sampling cadence differs from the qualification protocol")
    try:
        glysys_seed = int(protocol["seed"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid GlySys random seed in {run_path}") from error
    if int(openmm["randomNumberSeed"]) != glysys_seed:
        raise ValueError(f"engine seeds differ in {run_path}")
    expected_ps = protocol["equilibrationSteps"] * protocol["timestepFs"] * 0.001
    if not math.isclose(openmm["equilibrationPs"], expected_ps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"equilibration durations differ in {run_path}")
    if run["simulatedPs"] != (
        protocol["equilibrationSteps"] + protocol["productionSteps"]
    ) * protocol["timestepFs"] * 0.001:
        raise ValueError(f"incomplete GlySys trajectory in {run_path}")

    start = protocol["equilibrationSteps"]
    glysys = [
        json.loads(line) for line in (run_path / "observables.jsonl").read_text().splitlines()
        if line.strip()
    ]
    glysys = [sample for sample in glysys
              if sample["segment"] == "production" and sample["step"] > start]
    openmm_samples = openmm["samples"]
    left = {sample["step"]: sample for sample in glysys}
    right = {sample["absoluteStep"]: sample for sample in openmm_samples}
    common = sorted(set(left) & set(right))
    if len(common) != protocol["productionSteps"] // 100:
        raise ValueError(f"replica has incomplete or misaligned scalar samples: {run_path}")

    energy_difference = openmm["initialEnergyDifferenceKcalMol"]
    static_limit = max(0.05, openmm["atoms"] * 1e-5)
    if abs(energy_difference) > static_limit:
        raise ValueError(f"initial-state energy parity failed in {run_path}")
    return {
        "mode": protocol["solvent"],
        "seed": glysys_seed,
        "runPath": str(run_path),
        "openmmResult": str(openmm_path),
        "glysysTemperature": [left[step]["temperatureK"] for step in common],
        "openmmTemperature": [right[step]["temperatureK"] for step in common],
        "glysysEnergyPerAtom": [
            left[step]["potentialEnergyKcalMol"] / openmm["atoms"] for step in common
        ],
        "openmmEnergyPerAtom": [
            right[step]["potentialEnergyKcalMol"] / openmm["atoms"] for step in common
        ],
        "glysysInitialEnergyDifferenceKcalMol": energy_difference,
        "initialEnergyLimitKcalMol": static_limit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glysys-run", required=True, action="append", type=Path)
    parser.add_argument("--openmm-result", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-duration-ps", type=float, default=10.0)
    args = parser.parse_args()
    if len(args.glysys_run) != len(args.openmm_result):
        raise ValueError("provide one OpenMM result for every GlySys run")
    if not 2 <= len(args.glysys_run) <= 3:
        raise ValueError("aggregate two or three independent replicas")

    replicas = [read_pair(run_path, openmm_path)
                for run_path, openmm_path in zip(args.glysys_run, args.openmm_result)]
    seeds = [replica["seed"] for replica in replicas]
    if len(set(seeds)) != len(seeds):
        raise ValueError("replica seeds must be unique")
    modes = {replica["mode"] for replica in replicas}
    if len(modes) != 1:
        raise ValueError("all replicas must use the same solvent model")
    mode = modes.pop()
    protocol = json.loads((args.glysys_run[0] / "run.json").read_text())["protocol"]
    if not math.isfinite(args.block_duration_ps) or args.block_duration_ps <= 0:
        raise ValueError("block duration must be positive and finite")
    first_run = json.loads((args.glysys_run[0] / "run.json").read_text())
    sample_interval_ps = protocol["timestepFs"] * 0.001 * first_run["observablesEvery"]
    exact_block_size = args.block_duration_ps / sample_interval_ps
    block_size = round(exact_block_size)
    if block_size < 1 or not math.isclose(exact_block_size, block_size, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("block duration must be a whole multiple of the scalar-observation interval")
    for run_path in args.glysys_run:
        run = json.loads((run_path / "run.json").read_text())
        sample_count = protocol["productionSteps"] // run["observablesEvery"]
        if sample_count % block_size or (sample_count // 2) % block_size:
            raise ValueError("block duration must divide both production and half-production sample counts")
    specifications = {
        "temperatureK": ("glysysTemperature", "openmmTemperature", 3.0),
        "potentialEnergyKcalMolPerAtom": ("glysysEnergyPerAtom", "openmmEnergyPerAtom", 0.005),
    }
    results = {}
    statuses = []
    for label, (left_key, right_key, margin) in specifications.items():
        glysys = [replica[left_key] for replica in replicas]
        openmm = [replica[right_key] for replica in replicas]
        left = block_stats_replicas(glysys, block_size, args.block_duration_ps)
        right = block_stats_replicas(openmm, block_size, args.block_duration_ps)
        mean_equivalence = equivalence(left, right, margin)
        if max(left["maxAbsoluteLagOneBlockCorrelation"],
               right["maxAbsoluteLagOneBlockCorrelation"]) > 0.3:
            mean_equivalence["status"] = "inconclusive"
            mean_equivalence["reason"] = (
                f"residual {args.block_duration_ps:g} ps block correlation exceeds 0.3"
            )

        left_first = [values[:len(values) // 2] for values in glysys]
        left_second = [values[len(values) // 2:] for values in glysys]
        right_first = [values[:len(values) // 2] for values in openmm]
        right_second = [values[len(values) // 2:] for values in openmm]
        stationarity = equivalence(
            block_stats_replicas(left_first, block_size, args.block_duration_ps),
            block_stats_replicas(left_second, block_size, args.block_duration_ps),
            margin,
        )
        openmm_stationarity = equivalence(
            block_stats_replicas(right_first, block_size, args.block_duration_ps),
            block_stats_replicas(right_second, block_size, args.block_duration_ps),
            margin,
        )
        for check in (stationarity, openmm_stationarity):
            if check["combined95PercentCiWidth"] > margin:
                check["status"] = "inconclusive"
        results[label] = {
            "glysys": left,
            "openmm": right,
            "equivalence": mean_equivalence,
            "firstHalfVsSecondHalfStationarity": {
                "glysys": stationarity,
                "openmm": openmm_stationarity,
            },
        }
        statuses.extend([
            mean_equivalence["status"], stationarity["status"], openmm_stationarity["status"]
        ])

    static_checks = [{
        "seed": replica["seed"],
        "differenceKcalMol": replica["glysysInitialEnergyDifferenceKcalMol"],
        "limitKcalMol": replica["initialEnergyLimitKcalMol"],
        "status": "pass",
    } for replica in replicas]
    statuses.extend(check["status"] for check in static_checks)
    report = {
        "schemaVersion": 1,
        "status": "pass" if all(status == "pass" for status in statuses) else (
            "fail" if "fail" in statuses else "inconclusive"
        ),
        "mode": mode,
        "timestepFs": protocol["timestepFs"],
        "replicaCount": len(replicas),
        "seeds": seeds,
        "blockDurationPs": args.block_duration_ps,
        "equivalenceMargins": {
            "meanTemperatureK": 3.0,
            "meanPotentialEnergyKcalMolPerAtom": 0.005,
        },
        "staticEnergyChecks": static_checks,
        "productionMeanEquivalence": results,
        "replicas": [{key: replica[key] for key in (
            "seed", "runPath", "openmmResult"
        )} for replica in replicas],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
