"""Reference PDBFixer/OpenMM runner, executed inside the Cookbook compute image.

Reads one JSON request on stdin and writes one JSON result on stdout.
mode "fix": PDBFixer full repair (missing residues, nonstandard residues,
missing atoms, hydrogens at pH 7; heterogens and water kept).
mode "parameterize": build an OpenMM Amber14 system from the given PDB.
"""
import io
import json
import sys
import time
import traceback

request = json.load(sys.stdin)
started = time.time()
result = {"ok": False}
try:
    if request["mode"] == "fix":
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
        fixer = PDBFixer(pdbfile=io.StringIO(request["pdb"]))
        fixer.findMissingResidues()
        missing_residues = sum(len(v) for v in fixer.missingResidues.values())
        fixer.findNonstandardResidues()
        nonstandard = sorted({r.name for r, _ in fixer.nonstandardResidues})
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        missing_atoms = sum(len(v) for v in fixer.missingAtoms.values())
        missing_terminals = sum(len(v) for v in fixer.missingTerminals.values())
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        out = io.StringIO()
        PDBFile.writeFile(fixer.topology, fixer.positions, out, keepIds=True)
        result.update(ok=True, pdb=out.getvalue(), details={
            "missing_residues": missing_residues, "nonstandard_replaced": nonstandard,
            "missing_atoms": missing_atoms, "missing_terminals": missing_terminals})
    elif request["mode"] == "parameterize":
        from openmm.app import PDBFile, ForceField, NoCutoff
        pdb = PDBFile(io.StringIO(request["pdb"]))
        forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff)
        result.update(ok=True, details={"particles": system.getNumParticles()})
except Exception as error:  # the benchmark records every failure verbatim
    result["error"] = f"{type(error).__name__}: {error}"[:600]
result["seconds"] = round(time.time() - started, 3)
json.dump(result, sys.stdout)
