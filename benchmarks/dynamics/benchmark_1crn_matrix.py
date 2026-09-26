#!/usr/bin/env python3
"""Run sequential, provenance-rich GlySys/OpenMM 1CRN throughput comparisons.

Each GlySys and OpenMM process starts from the same saved step-zero state.
Engine order alternates within each paired repeat. This script preserves all
individual JSON results and logs; it never edits either protocol file.
"""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(bytes.fromhex(sha256(file_path)))
    return digest.hexdigest()


def git_provenance(repo: Path):
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    digest = hashlib.sha256()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=repo, check=True,
        capture_output=True,
    ).stdout
    digest.update(diff)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=repo,
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    for relative in sorted(untracked):
        candidate = repo / relative
        if candidate.is_file():
            digest.update(relative.encode())
            digest.update(bytes.fromhex(sha256(candidate)))
    return revision, digest.hexdigest()


def run_process(command, stdout_path: Path, stderr_path: Path):
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return {"exitCode": completed.returncode, "wallSeconds": elapsed,
            "command": [str(part) for part in command]}


def median(values):
    return statistics.median(values) if values else None


def measured_seconds(result, engine):
    if engine == "glysys":
        repeat = result["repeats"][0]
        return float(repeat["simulationSeconds"])
    return float(result["simulationSeconds"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--glysys-binary", required=True, type=Path)
    parser.add_argument("--openmm-python", required=True, type=Path)
    parser.add_argument("--explicit-input", required=True, type=Path)
    parser.add_argument("--implicit-input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=6000,
                        help="4000 equilibration + 2000 additional warmup steps")
    parser.add_argument("--device-index", default="0")
    parser.add_argument(
        "--explicit-gpu-tiled-nonbonded", action="store_true",
        help="use the qualified native fixed-row cooperative PBC kernel for explicit GPU legs",
    )
    args = parser.parse_args()
    if args.seconds <= 0 or args.repeats <= 0 or args.warmup_steps < 0:
        parser.error("seconds/repeats must be positive and warmup steps nonnegative")

    repo = args.repo.resolve()
    glysys_binary = args.glysys_binary.resolve()
    openmm_python = args.openmm_python.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / ("matrix-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir()
    revision, diff_hash = git_provenance(repo)
    helper = repo / "benchmarks" / "openmm_1crn_dynamics.py"
    protocols = {
        "explicit": repo / "benchmarks/dynamics/1crn-explicit-lf-middle-2fs-qualification.json",
        "implicit": repo / "benchmarks/dynamics/1crn-implicit-lf-middle-2fs-qualification.json",
    }
    inputs = {
        "explicit": args.explicit_input.resolve(),
        "implicit": args.implicit_input.resolve(),
    }

    manifest = {
        "schemaVersion": 1,
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "sourceRevision": revision,
        "dirtyDiffSha256": diff_hash,
        "glysysBinarySha256": sha256(glysys_binary),
        "openmmVersionPinned": "8.1.1 (helper enforces exact runtime version)",
        "host": {"platform": platform.platform(), "processor": platform.processor()},
        "settings": {
            "secondsPerWindow": args.seconds,
            "repeats": args.repeats,
            "warmupSteps": args.warmup_steps,
            "glysysGpuBackend": "Vulkan",
            "openmmGpuPlatform": "OpenCL mixed precision",
            "explicitGpuKernel": (
                "cooperative-64-lane/fixed-640"
                if args.explicit_gpu_tiled_nonbonded else "serial-per-atom/CSR"
            ),
            "cpuThreadCounts": [1, 6, 12],
            "acceptanceRatio": 0.8,
            "maximumIqrOverMedian": 0.10,
            "timingBoundary": "completed work, including final synchronization/readback",
        },
        "protocolSha256": {name: sha256(path) for name, path in protocols.items()},
        "inputTreeSha256": {name: tree_sha256(path) for name, path in inputs.items()},
        "preparedInputs": {name: str(path) for name, path in inputs.items()},
        "runs": [],
        "comparisons": [],
    }

    def invoke(engine, model, backend_name, threads, repeat, canonical, out_dir):
        stem = f"{engine}-{backend_name}-r{repeat:02d}"
        output_json = out_dir / f"{stem}.json"
        if engine == "glysys":
            command = [
                glysys_binary, "benchmark", "--input", inputs[model],
                "--protocol", protocols[model], "--backend",
                "vulkan" if backend_name == "gpu" else "cpu",
                "--threads", str(threads), "--warmup-steps",
                str(args.warmup_steps), "--seconds", str(args.seconds),
                "--repeats", "1", "--starting-checkpoint", canonical,
                "--json-output", output_json,
            ]
            if backend_name == "gpu":
                command[command.index("--threads") + 1] = "1"
                if model == "explicit" and args.explicit_gpu_tiled_nonbonded:
                    command.append("--explicit-gpu-tiled-nonbonded")
        else:
            platform_name = "OpenCL" if backend_name == "gpu" else "CPU"
            command = [
                openmm_python, helper, "--input", inputs[model],
                "--checkpoint", canonical, "--protocol", protocols[model],
                "--mode", model, "--platform", platform_name,
                "--benchmark-seconds", str(args.seconds), "--warmup-steps",
                str(args.warmup_steps), "--output", output_json,
            ]
            if backend_name == "gpu":
                command.extend(["--device-index", args.device_index])
            else:
                command.extend(["--threads", str(threads)])
        run_info = run_process(
            command, out_dir / f"{stem}.stdout.txt", out_dir / f"{stem}.stderr.txt"
        )
        run_info.update({"engine": engine, "model": model,
                         "backend": backend_name, "threads": threads,
                         "repeat": repeat, "jsonPath": str(output_json)})
        if output_json.exists():
            try:
                run_info["result"] = json.loads(output_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                run_info["resultParseError"] = str(error)
        manifest["runs"].append(run_info)
        return run_info

    failures = []
    for model in ("explicit", "implicit"):
        canonical_dir = run_dir / model / "canonical"
        canonical_dir.mkdir(parents=True)
        canonical_log = canonical_dir / "prepare.log.json"
        canonical_command = [
            glysys_binary, "run", "--input", inputs[model], "--protocol",
            protocols[model], "--backend", "cpu", "--threads", "1",
            "--steps", "1", "--output", canonical_dir / "run",
        ]
        canonical_info = run_process(
            canonical_command, canonical_dir / "stdout.txt", canonical_dir / "stderr.txt"
        )
        canonical_path = canonical_dir / "run/step-zero-reference-checkpoint.json"
        canonical_info.update({"model": model, "jsonPath": str(canonical_path)})
        manifest["runs"].append(canonical_info)
        if canonical_info["exitCode"] != 0 or not canonical_path.is_file():
            failures.append(f"{model}: failed to create canonical step-zero checkpoint")
            continue

        for backend_name, threads in (("cpu", 1), ("cpu", 6), ("cpu", 12), ("gpu", 1)):
            config_dir = run_dir / model / f"{backend_name}-{threads}threads"
            config_dir.mkdir(parents=True)
            glysys_runs = []
            openmm_runs = []
            for repeat in range(1, args.repeats + 1):
                order = ("glysys", "openmm") if (repeat + threads) % 2 else ("openmm", "glysys")
                pair = {}
                for engine in order:
                    pair[engine] = invoke(
                        engine, model, backend_name, threads, repeat,
                        canonical_path, config_dir,
                    )
                glysys_info = pair["glysys"]
                openmm_info = pair["openmm"]
                if glysys_info["exitCode"] != 0 or openmm_info["exitCode"] != 0:
                    failures.append(f"{model}/{backend_name}/{threads} threads/repeat {repeat}: process failure")
                    continue
                try:
                    glysys_result = glysys_info["result"]
                    openmm_result = openmm_info["result"]
                    glysys_record = glysys_result["repeats"][0]
                    glysys_rate = glysys_record["nsPerDay"]
                    openmm_rate = openmm_result["nsPerDay"]
                    upper_window = args.seconds + max(0.5, args.seconds * 0.25)
                    for engine, result in (("glysys", glysys_result), ("openmm", openmm_result)):
                        duration = measured_seconds(result, engine)
                        if not args.seconds <= duration <= upper_window:
                            failures.append(
                                f"{model}/{backend_name}/{threads} threads/repeat {repeat}: "
                                f"{engine} measured {duration:.3f}s outside the "
                                f"{args.seconds:.3f}–{upper_window:.3f}s window"
                            )
                    actual = glysys_record["backend"]
                    fallback = glysys_record.get("fallbackReason")
                    if (backend_name == "gpu" and actual != "GPU") or fallback:
                        failures.append(f"{model}/{backend_name}/{threads} threads/repeat {repeat}: backend fallback or mismatch")
                    glysys_runs.append(glysys_rate)
                    openmm_runs.append(openmm_rate)
                except (KeyError, IndexError, TypeError) as error:
                    failures.append(f"{model}/{backend_name}/{threads} threads/repeat {repeat}: malformed benchmark JSON: {error}")
            if len(glysys_runs) == args.repeats and len(openmm_runs) == args.repeats:
                ratios = [left / right for left, right in zip(glysys_runs, openmm_runs)]
                ratio_median = median(ratios)
                quartiles = statistics.quantiles(ratios, n=4, method="inclusive")
                spread = (quartiles[2] - quartiles[0]) / ratio_median
                comparison = {
                    "model": model, "backend": backend_name, "threads": threads,
                    "glysysMedianNsPerDay": median(glysys_runs),
                    "openmmMedianNsPerDay": median(openmm_runs),
                    "medianPairedRatio": ratio_median,
                    "ratioIqrOverMedian": spread,
                    "performanceGate": "pass" if ratio_median >= 0.8 and spread <= 0.10 else "fail-or-repeat-required",
                }
                manifest["comparisons"].append(comparison)

    manifest["failures"] = failures
    manifest["overallGate"] = (
        "pass" if not failures and len(manifest["comparisons"]) == 8
        and all(item["performanceGate"] == "pass" for item in manifest["comparisons"])
        else "not-passed"
    )
    manifest_path = run_dir / "matrix-results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(manifest_path), "overallGate": manifest["overallGate"],
                      "failures": failures, "comparisons": manifest["comparisons"]}, indent=2))
    raise SystemExit(0 if manifest["overallGate"] == "pass" else 1)


if __name__ == "__main__":
    main()
