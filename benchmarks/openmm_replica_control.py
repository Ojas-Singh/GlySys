#!/usr/bin/env python3
"""Run the replica-aggregation gate on two independent replica sets.

Uses `aggregate_1crn_replicas.py`'s 10 ps (and 20 ps) block statistics,
equivalence test and 0.3 block-correlation rule unchanged, but lets either side
be OpenMM. Comparing two OpenMM replica sets of the same protocol shows how
much the gate's verdict reflects sampling rather than engine differences; a
GlySys set can be compared against an OpenMM set from independent seeds.

usage: openmm_replica_control.py OUT.json A1 A2 A3 -- B1.json B2.json B3.json
  A entries are OpenMM result JSON files or GlySys run directories (whose
  per-atom energies use B's atom count); B entries are OpenMM result files.
"""
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "aggregate", Path(__file__).parent / "aggregate_1crn_replicas.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)

out = Path(sys.argv[1])
split = sys.argv.index("--")
sets = [sys.argv[2:split], sys.argv[split + 1:]]


def load(path):
    if Path(path).is_dir():  # GlySys run directory
        run = json.loads((Path(path) / "run.json").read_text())
        protocol = run["protocol"]
        atoms = json.loads(Path(sets[1][0]).read_text())["atoms"]
        samples = [json.loads(line) for line in (Path(path) / "observables.jsonl").read_text().splitlines()
                   if line.strip()]
        samples = [x for x in samples
                   if x["segment"] == "production" and x["step"] > protocol["equilibrationSteps"]]
        assert len(samples) == protocol["productionSteps"] // 100, path
        return ([x["temperatureK"] for x in samples],
                [x["potentialEnergyKcalMol"] / atoms for x in samples], protocol["seed"])
    d = json.loads(Path(path).read_text())
    start = d["equilibrationSteps"]
    samples = [s for s in d["samples"] if s["absoluteStep"] > start]
    assert len(samples) == d["productionSteps"] // 100, path
    return (
        [s["temperatureK"] for s in samples],
        [s["potentialEnergyKcalMol"] / d["atoms"] for s in samples],
        d["randomNumberSeed"],
    )


replicas = [[load(p) for p in paths] for paths in sets]
report = {"seeds": [[r[2] for r in rs] for rs in replicas], "blockDurationPs": {}}
for block_ps in (10.0, 20.0):
    block = round(block_ps / 0.2)
    results = {}
    for index, (label, margin) in enumerate((("temperatureK", 3.0),
                                             ("potentialEnergyKcalMolPerAtom", 0.005))):
        left = aggregate.block_stats_replicas([r[index] for r in replicas[0]], block, block_ps)
        right = aggregate.block_stats_replicas([r[index] for r in replicas[1]], block, block_ps)
        check = aggregate.equivalence(left, right, margin)
        if max(left["maxAbsoluteLagOneBlockCorrelation"],
               right["maxAbsoluteLagOneBlockCorrelation"]) > 0.3:
            check["status"] = "inconclusive"
            check["reason"] = f"residual {block_ps:g} ps block correlation exceeds 0.3"
        results[label] = {"setA": left, "setB": right, "equivalence": check}
        print(f"{block_ps:g} ps {label[:6]}: A {left['mean']:.5f} B {right['mean']:.5f} "
              f"diff {check['absoluteMeanDifference']:.5f} ciw {check['combined95PercentCiWidth']:.5f} "
              f"{check['status']} maxlag {left['maxAbsoluteLagOneBlockCorrelation']:.2f}/"
              f"{right['maxAbsoluteLagOneBlockCorrelation']:.2f}")
    report["blockDurationPs"][str(block_ps)] = results
out.write_text(json.dumps(report, indent=2))
