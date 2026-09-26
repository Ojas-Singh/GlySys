#!/usr/bin/env python3
"""Compare exported GlySys force arrays against OpenMM's independent snapshot."""
import argparse
import json
from pathlib import Path

import numpy as np


def metrics(actual, reference):
    delta = actual - reference
    norm = np.linalg.norm(reference)
    return {
        "normalizedRms": float(np.linalg.norm(delta) / max(norm, 1e-30)),
        "maxAbsoluteComponent": float(np.max(np.abs(delta))),
        "componentsOutsideTolerance": int(np.count_nonzero(
            np.abs(delta) > 0.02 + 0.001 * np.abs(reference)
        )),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openmm", required=True, type=Path)
    parser.add_argument("--glysys", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    openmm = json.loads(args.openmm.read_text())
    glysys = json.loads(args.glysys.read_text())
    if openmm["openmmVersion"] != "8.1.1":
        raise ValueError("force comparison requires pinned OpenMM 8.1.1")
    if openmm["atoms"] != glysys["atoms"]:
        raise ValueError("atom counts differ")
    if openmm["mode"] != glysys["mode"]:
        raise ValueError("solvent models differ")
    if openmm["checkpointStep"] != glysys["checkpointStep"]:
        raise ValueError("checkpoint steps differ")
    reference = np.asarray(openmm["forcesKcalMolAngstrom"], dtype=np.float64)
    comparisons = {}
    if "gpuForcesKcalMolAngstrom" in glysys:
        gpu = np.asarray(glysys["gpuForcesKcalMolAngstrom"], dtype=np.float64)
        if reference.shape != gpu.shape:
            raise ValueError("GPU force array shape differs")
        comparisons["glysysGpuForceComparison"] = metrics(gpu, reference)
    if "glysysCpuForcesKcalMolAngstrom" in glysys:
        cpu = np.asarray(glysys["glysysCpuForcesKcalMolAngstrom"], dtype=np.float64)
        if reference.shape != cpu.shape:
            raise ValueError("CPU force array shape differs")
        comparisons["glysysCpuF64ForceComparison"] = metrics(cpu, reference)
    if not comparisons:
        raise ValueError("GlySys file contains no recognized force array")
    energy_delta = abs(
        float(glysys["potentialEnergyKcalMol"])
        - float(openmm["potentialEnergyKcalMol"])
    )
    result = {
        "schemaVersion": 1,
        "openmmPlatform": openmm["platform"],
        "openmmPrecision": openmm["precision"],
        "atoms": int(openmm["atoms"]),
        "checkpointSha256": openmm["checkpointSha256"],
        "checkpointStep": openmm["checkpointStep"],
        "potentialEnergyAbsoluteDifferenceKcalMol": energy_delta,
        "potentialEnergyLimitKcalMol": max(0.05, 1e-5 * openmm["atoms"]),
        **comparisons,
        "forceTolerance": {
            "normalizedRmsMaximum": 1e-3,
            "componentAbsolutePlusRelative": "0.02 + 0.001 * abs(reference)",
        },
    }
    for key in comparisons:
        row = result[key]
        row["status"] = (
            "pass"
            if row["normalizedRms"] <= 1e-3
            and row["componentsOutsideTolerance"] == 0
            else "fail"
        )
    result["potentialEnergyStatus"] = (
        "pass" if energy_delta <= result["potentialEnergyLimitKcalMol"] else "fail"
    )
    result["overallStatus"] = (
        "pass"
        if result["potentialEnergyStatus"] == "pass"
        and all(result[key]["status"] == "pass" for key in comparisons)
        else "fail"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
