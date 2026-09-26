#!/usr/bin/env python3
"""Export an independent OpenMM force/energy snapshot for a GlySys checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import openmm as mm
from openmm import app, unit
import numpy as np

from openmm_1crn_dynamics import make_system


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--mode", choices=("explicit", "implicit"), required=True)
    parser.add_argument("--platform", choices=("Reference", "CPU", "OpenCL"), default="Reference")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--gpu-f32-centered-input",
        action="store_true",
        help=(
            "Evaluate at the exact centered f32 coordinate/box representation "
            "uploaded by the explicit GlySys GPU path"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if mm.__version__ != "8.1.1":
        raise RuntimeError(f"expected OpenMM 8.1.1, found {mm.__version__}")
    protocol = json.loads(args.protocol.read_text())
    checkpoint_raw = args.checkpoint.read_bytes()
    checkpoint = json.loads(checkpoint_raw)["state"]
    prmtop = app.AmberPrmtopFile(str(args.input / "system.prmtop"))
    system = make_system(prmtop, args.mode, protocol)
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    platform = mm.Platform.getPlatformByName(args.platform)
    properties = {}
    if args.platform == "OpenCL":
        properties = {"DeviceIndex": args.device_index, "Precision": "mixed"}
    elif args.platform == "CPU" and args.threads:
        properties = {"Threads": str(args.threads)}
    context = mm.Context(system, integrator, platform, properties)

    positions = checkpoint["coordinates"]
    box = checkpoint.get("boxAngstrom")
    if args.gpu_f32_centered_input:
        if args.mode != "explicit" or box is None:
            raise ValueError("--gpu-f32-centered-input requires an explicit checkpoint box")
        box = np.asarray(box, dtype=np.float32).astype(np.float64)
        half_box_f64 = 0.5 * box
        raw_positions = np.asarray(
            [[row["x"], row["y"], row["z"]] for row in positions],
            dtype=np.float64,
        )
        positions = (raw_positions - half_box_f64).astype(np.float32).astype(np.float64)
    context.setPositions([
        mm.Vec3(*(coordinate * 0.1 for coordinate in row))
        if args.gpu_f32_centered_input
        else mm.Vec3(row["x"] * 0.1, row["y"] * 0.1, row["z"] * 0.1)
        for row in positions
    ])
    if args.mode == "explicit":
        if box is None:
            raise ValueError("explicit checkpoint is missing boxAngstrom")
        vectors = tuple(
            mm.Vec3(*(box[axis] * 0.1 if axis == j else 0.0 for axis in range(3)))
            for j in range(3)
        )
        context.setPeriodicBoxVectors(*vectors)

    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(
        unit.kilocalories_per_mole / unit.angstrom), dtype=np.float64)
    result = {
        "schemaVersion": 1,
        "openmmVersion": mm.__version__,
        "platform": args.platform,
        "device": platform.getName(),
        "precision": properties.get("Precision", "Reference-double" if args.platform == "Reference" else "platform-default"),
        "mode": args.mode,
        "atoms": system.getNumParticles(),
        "constraints": system.getNumConstraints(),
        "potentialEnergyKcalMol": float(energy),
        "timestepFs": protocol["timestepFs"],
        "positionRepresentation": (
            "gpu-f32-centered" if args.gpu_f32_centered_input else "checkpoint-f64"
        ),
        "checkpointSha256": hashlib.sha256(checkpoint_raw).hexdigest(),
        "checkpointStep": checkpoint["step"],
        "forcesKcalMolAngstrom": forces.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "forcesKcalMolAngstrom"}, indent=2))
    del context, integrator


if __name__ == "__main__":
    main()
