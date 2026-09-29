#!/usr/bin/env python3
"""Structure-repair benchmark: GlySys `fix` against OpenMM PDBFixer.

Each case is downloaded from the RCSB, repaired by both tools, and every
output is scored with the same metrics. Parameterization is checked with
OpenMM's Amber14 force field and with GlySys `prepare --no-water`.

PDBFixer runs inside a container that provides it (by default the Cookbook
compute image); GlySys runs from a local `glysysbuilder` binary.
"""
import argparse
import json
import math
import subprocess
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

PROTEIN = {
    "ALA": "N CA C O CB", "ARG": "N CA C O CB CG CD NE CZ NH1 NH2",
    "ASN": "N CA C O CB CG OD1 ND2", "ASP": "N CA C O CB CG OD1 OD2",
    "CYS": "N CA C O CB SG", "GLN": "N CA C O CB CG CD OE1 NE2",
    "GLU": "N CA C O CB CG CD OE1 OE2", "GLY": "N CA C O",
    "HIS": "N CA C O CB CG ND1 CD2 CE1 NE2", "ILE": "N CA C O CB CG1 CG2 CD1",
    "LEU": "N CA C O CB CG CD1 CD2", "LYS": "N CA C O CB CG CD CE NZ",
    "MET": "N CA C O CB CG SD CE", "PHE": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ",
    "PRO": "N CA C O CB CG CD", "SER": "N CA C O CB OG", "THR": "N CA C O CB OG1 CG2",
    "TRP": "N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "TYR": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH", "VAL": "N CA C O CB CG1 CG2",
}
PROTEIN = {name: set(atoms.split()) for name, atoms in PROTEIN.items()}
PROTEIN_ALIASES = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
                   "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}
SUGAR = "C1' C2' C3' C4' O4' C5' O5' O3'"
PURINE_A = "N9 C8 N7 C5 C6 N6 N1 C2 N3 C4"
PURINE_G = "N9 C8 N7 C5 C6 O6 N1 C2 N2 N3 C4"
NUCLEIC = {
    "DA": f"{SUGAR} {PURINE_A}", "DG": f"{SUGAR} {PURINE_G}",
    "DC": f"{SUGAR} N1 C2 O2 N3 C4 N4 C5 C6", "DT": f"{SUGAR} N1 C2 O2 N3 C4 O4 C5 C7 C6",
    "A": f"{SUGAR} O2' {PURINE_A}", "G": f"{SUGAR} O2' {PURINE_G}",
    "C": f"{SUGAR} O2' N1 C2 O2 N3 C4 N4 C5 C6", "U": f"{SUGAR} O2' N1 C2 O2 N3 C4 O4 C5 C6",
}
NUCLEIC = {name: set(atoms.split()) for name, atoms in NUCLEIC.items()}
SOLVENT_IONS = {"HOH", "WAT", "DOD", "SOL", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "FE2",
                "CU", "CU1", "CO", "NI", "CD", "HG", "IOD", "BR", "SO4", "PO4", "GOL", "EDO", "ACT"}
METALS = {"NA", "K", "MG", "CA", "ZN", "MN", "FE", "FE2", "CU", "CU1", "CO", "NI", "CD", "HG"}


def standard_template(name):
    """Heavy-atom template for a standard residue name, or None."""
    name = PROTEIN_ALIASES.get(name, name)
    if len(name) == 4 and name[0] in "NC" and name[1:] in PROTEIN:
        name = name[1:]
    return PROTEIN.get(name) or NUCLEIC.get(name)


def parse(pdb_text):
    """Atoms of the first model: (chain, resseq, icode, resname, name, element, xyz)."""
    atoms = []
    for line in pdb_text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        name = line[12:16].strip()
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if not element:
            element = name.lstrip("0123456789")[:1].upper()
        atoms.append((line[21], line[22:26].strip(), line[26].strip(), line[17:20].strip(), name, element, xyz))
    return atoms


def evaluate(pdb_text):
    atoms = parse(pdb_text)
    residues = defaultdict(list)
    for atom in atoms:
        residues[atom[:4]].append(atom)
    hydrogens = sum(1 for a in atoms if a[5] in ("H", "D"))
    standard = complete = 0
    other = Counter()
    solvent = 0
    for (_, _, _, resname), members in residues.items():
        template = standard_template(resname)
        if template is not None:
            standard += 1
            names = {a[4] for a in members if a[5] not in ("H", "D")}
            complete += template <= names
        elif resname in SOLVENT_IONS:
            solvent += 1
        else:
            other[resname] += 1
    return {
        "atoms": len(atoms), "heavy_atoms": len(atoms) - hydrogens, "hydrogens": hydrogens,
        "residues": len(residues), "standard_residues": standard, "complete_standard_residues": complete,
        "solvent_ion_residues": solvent, "other_residues": dict(sorted(other.items())),
        "overlaps": overlaps(atoms),
        "hydrogen_clashes": overlaps(atoms, cutoff=1.5, hydrogen_only=True),
        "heavy_clashes": heavy_clashes(atoms),
    }


def heavy_clashes(atoms, low=1.6, high=2.2):
    """Non-bonded heavy-atom contacts between residues in [low, high) Å.

    Covalent inter-residue bonds are shorter than `low`, except disulfides
    and metal coordination, which are excluded."""
    heavy = [a for a in atoms if a[5] not in ("H", "D")]
    grid = defaultdict(list)
    for index, atom in enumerate(heavy):
        grid[tuple(math.floor(c / high) for c in atom[6])].append(index)
    count = 0
    for index, atom in enumerate(heavy):
        cell = tuple(math.floor(c / high) for c in atom[6])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other in grid.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()):
                        b = heavy[other]
                        if other <= index or b[:4] == atom[:4]:
                            continue
                        if (atom[4] == "SG" and b[4] == "SG") or atom[5] in METALS or b[5] in METALS:
                            continue
                        if low <= math.dist(atom[6], b[6]) < high:
                            count += 1
    return count


def overlaps(atoms, cutoff=0.8, hydrogen_only=False):
    """Atom pairs in different residues closer than `cutoff` Å: placement failures.

    With `hydrogen_only`, count pairs involving a hydrogen (hydrogens are never
    bonded across residues, so any such contact under 1.5 Å is a clash)."""
    grid = defaultdict(list)
    for index, atom in enumerate(atoms):
        x, y, z = atom[6]
        grid[(math.floor(x), math.floor(y), math.floor(z))].append(index)
    count = 0
    for index, atom in enumerate(atoms):
        x, y, z = atom[6]
        cx, cy, cz = math.floor(x), math.floor(y), math.floor(z)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other in grid.get((cx + dx, cy + dy, cz + dz), ()):
                        if other <= index or atoms[other][:4] == atom[:4]:
                            continue
                        if hydrogen_only and "H" not in (atom[5], atoms[other][5]):
                            continue
                        ox, oy, oz = atoms[other][6]
                        if (x - ox) ** 2 + (y - oy) ** 2 + (z - oz) ** 2 < cutoff * cutoff:
                            count += 1
    return count


def run(command, stdin=None, timeout=900):
    started = time.time()
    try:
        completed = subprocess.run(command, input=stdin, capture_output=True, text=True, timeout=timeout)
        return completed.returncode, completed.stdout, completed.stderr, time.time() - started
    except subprocess.TimeoutExpired:
        return -1, "", f"timed out after {timeout}s", time.time() - started


def reference(container, mode, pdb):
    script = (HERE / "pdbfixer_reference.py").read_text()
    code, out, err, _ = run(["podman", "exec", "-i", container, "/opt/conda/envs/reglyco/bin/python", "-c", script],
                            stdin=json.dumps({"mode": mode, "pdb": pdb}))
    if code != 0 or not out.strip():
        return {"ok": False, "error": (err or out).strip()[-600:]}
    return json.loads(out)


def glysys_fix(binary, pdb):
    with tempfile.TemporaryDirectory() as tmp:
        source, output, report = Path(tmp, "in.pdb"), Path(tmp, "out.pdb"), Path(tmp, "report.json")
        source.write_text(pdb)
        code, _, err, seconds = run([binary, "fix", str(source), "--output", str(output), "--report", str(report)])
        if code != 0:
            return {"ok": False, "error": err.strip().splitlines()[-1][:600] if err.strip() else f"exit {code}",
                    "seconds": round(seconds, 3)}
        return {"ok": True, "pdb": output.read_text(), "seconds": round(seconds, 3),
                "report": json.loads(report.read_text()) if report.exists() else None}


def glysys_parameterize(binary, pdb):
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp, "in.pdb")
        source.write_text(pdb)
        code, _, err, _ = run([binary, "prepare", str(source), "--output", str(Path(tmp, "out")), "--no-water"])
        return {"ok": code == 0, **({} if code == 0 else {"error": err.strip().splitlines()[-1][:600] if err.strip() else f"exit {code}"})}


def fetch(pdb_id, cache):
    path = cache / f"{pdb_id}.pdb"
    if not path.exists():
        with urllib.request.urlopen(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=120) as response:
            path.write_bytes(response.read())
    return path.read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glysys", default=str(HERE.parents[1] / "target/release/glysysbuilder"))
    parser.add_argument("--container", default="glycoshape-live-compute")
    parser.add_argument("--cache", default=str(HERE / ".cache"))
    parser.add_argument("--output", default=str(HERE / "results/latest.json"))
    parser.add_argument("--only", nargs="*", help="run only these PDB IDs")
    parser.add_argument("--skip-reference", action="store_true", help="reuse reference results from --output")
    args = parser.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    cases = json.loads((HERE / "cases.json").read_text())["cases"]
    if args.only:
        cases = [case for case in cases if case["id"] in set(args.only)]
    previous = {}
    if args.skip_reference and Path(args.output).exists():
        previous = {row["id"]: row for row in json.loads(Path(args.output).read_text())["results"]}
    results = []
    for case in cases:
        pdb = fetch(case["id"], cache)
        row = {**case, "input": evaluate(pdb)}
        for tool in ("pdbfixer", "glysys"):
            if tool == "pdbfixer" and case["id"] in previous:
                row[tool] = previous[case["id"]][tool]
                saved = cache / "outputs" / tool / f"{case['id']}.pdb"
                if row[tool]["ok"] and saved.exists():
                    row[tool]["output"] = evaluate(saved.read_text())
                continue
            fixed = reference(args.container, "fix", pdb) if tool == "pdbfixer" else glysys_fix(args.glysys, pdb)
            entry = {"ok": fixed["ok"], "seconds": fixed.get("seconds")}
            if fixed["ok"]:
                keep = cache / "outputs" / tool
                keep.mkdir(parents=True, exist_ok=True)
                (keep / f"{case['id']}.pdb").write_text(fixed["pdb"])
                entry["output"] = evaluate(fixed["pdb"])
                entry["details"] = fixed.get("details")
                entry["openmm_amber14"] = reference(args.container, "parameterize", fixed["pdb"])
                entry["openmm_amber14"].pop("pdb", None)
                entry["glysys_prepare"] = glysys_parameterize(args.glysys, fixed["pdb"])
            else:
                entry["error"] = fixed.get("error")
            row[tool] = entry
        results.append(row)
        status = {tool: ("ok" if row[tool]["ok"] else "FAIL") for tool in ("pdbfixer", "glysys")}
        print(f"{case['id']:5} {case['category']:9} pdbfixer={status['pdbfixer']:4} glysys={status['glysys']:4}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"results": results}, indent=1))
    Path(args.output).with_suffix(".md").write_text(markdown(results))
    print(f"wrote {args.output} and {Path(args.output).with_suffix('.md')}")


def markdown(results):
    def cell(entry):
        if not entry["ok"]:
            return "fail"
        out = entry["output"]
        return (f"ok · {out['complete_standard_residues']}/{out['standard_residues']} complete · "
                f"H {out['hydrogens']} · clashes H {out.get('hydrogen_clashes', '?')} / heavy {out.get('heavy_clashes', '?')} · {entry.get('seconds')} s")

    def param(entry, key):
        return "–" if not entry["ok"] else ("yes" if entry[key]["ok"] else "no")

    lines = ["| PDB | Category | Input complete | PDBFixer | GlySys | OpenMM Amber14 (PDBFixer / GlySys) | GlySys prepare (PDBFixer / GlySys) |",
             "|---|---|---|---|---|---|---|"]
    for row in results:
        source = row["input"]
        lines.append(f"| {row['id']} | {row['category']} | {source['complete_standard_residues']}/{source['standard_residues']} | "
                     f"{cell(row['pdbfixer'])} | {cell(row['glysys'])} | "
                     f"{param(row['pdbfixer'], 'openmm_amber14')} / {param(row['glysys'], 'openmm_amber14')} | "
                     f"{param(row['pdbfixer'], 'glysys_prepare')} / {param(row['glysys'], 'glysys_prepare')} |")
    fixed = {tool: sum(row[tool]["ok"] for row in results) for tool in ("pdbfixer", "glysys")}
    lines += ["", f"Fixed: PDBFixer {fixed['pdbfixer']}/{len(results)}, GlySys {fixed['glysys']}/{len(results)}.", "",
              "## Failures", ""]
    for row in results:
        for tool in ("pdbfixer", "glysys"):
            if not row[tool]["ok"]:
                lines.append(f"- {row['id']} {tool}: {row[tool].get('error')}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
