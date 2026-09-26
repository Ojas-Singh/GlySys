"""Export the force stored at a GlySys runtime checkpoint for OpenMM comparison."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("cpu", "gpu"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    raw = args.checkpoint.read_bytes()
    checkpoint = json.loads(raw)
    state = checkpoint["state"]
    gradient = state["gradient"]
    coordinates = state["coordinates"]
    if len(gradient) != len(coordinates):
        raise ValueError("checkpoint gradient and coordinate counts differ")
    forces = [
        [-float(row[axis]) for axis in ("x", "y", "z")]
        for row in gradient
    ]
    result = {
        "schemaVersion": 1,
        "backend": args.backend,
        "mode": state["protocol"]["solvent"],
        "atoms": len(forces),
        "checkpointSha256": hashlib.sha256(raw).hexdigest(),
        "checkpointStep": state["step"],
        "modelVersion": state["modelVersion"],
        "potentialEnergyKcalMol": state["potentialEnergy"],
    }
    if args.backend == "gpu":
        result["gpuForcesKcalMolAngstrom"] = forces
    else:
        result["glysysCpuForcesKcalMolAngstrom"] = forces

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({key: value for key, value in result.items() if "Forces" not in key}, indent=2))


if __name__ == "__main__":
    main()
