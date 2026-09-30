# Structure-repair benchmark: GlySys `fix` vs OpenMM PDBFixer

30 PDB entries ([`cases.json`](cases.json)) cover the repairs users need most:
clean crystal structures, alternate locations and microheterogeneity, NMR
ensembles, modified residues (MSE, SEP/TPO, PTR, PCA, KCX, CSO, CME, LLP, HYP,
D-amino acids), ligands and cofactors, metal sites, DNA/RNA with modified bases,
glycoproteins, antibody insertion codes, and the 58,000-atom GroEL assembly.

Both tools run their full repair: missing residues, modified-residue
replacement, missing heavy atoms and hydrogens at pH 7, keeping water and
heterogens. PDBFixer runs in the Cookbook compute image
(`pdbfixer_reference.py`); GlySys runs `glysysbuilder fix`. Every output is
scored by the same code in [`run_benchmark.py`](run_benchmark.py):

- **complete**: standard residues whose heavy atoms are all present;
- **H**: hydrogens in the output;
- **clashes H**: inter-residue contacts under 1.5 Å involving a hydrogen;
- **clashes heavy**: non-bonded inter-residue heavy-atom contacts between 1.6
  and 2.2 Å (disulfides and metal coordination excluded; many come from the
  deposited coordinates and are identical for both tools);
- **OpenMM Amber14** / **GlySys prepare**: whether the repaired file can be
  parameterized. Both fail on the same ligand-containing entries, which neither
  Amber14 nor ff14SB/GLYCAM covers.

```console
python3 run_benchmark.py --output results/latest.json            # both tools
python3 run_benchmark.py --output results/latest.json --skip-reference
```

`--skip-reference` reuses the PDBFixer results already in the output file (and
the PDBFixer PDBs kept in `.cache/outputs/pdbfixer`).

## Results (2026-09-29)

[`results/fixer-20260929.md`](results/fixer-20260929.md) has the per-entry
table; [`results/baseline-20260929.md`](results/baseline-20260929.md) is the
same benchmark before the fixer existed (GlySys fixed 7/30).

| | PDBFixer | GlySys `fix` |
|---|---|---|
| Entries repaired | 29/30 (GroEL timed out after 900 s) | 30/30 (GroEL in 10.5 s) |
| Hydrogen clashes, 29 common entries | 114 | 39 |
| Heavy-atom clashes, 29 common entries | 720 | 543 |
| Total time, 29 common entries | 557 s | 13 s |
| OpenMM Amber14 accepts the output | 13 entries | the same 13 entries |
| GlySys `prepare` accepts the output | 12 entries | the same 12 entries |

Hydrogen counts are identical for most entries. The differences are deliberate:

- GlySys keeps hydroxyproline (PDBFixer converts HYP to PRO in collagen, 1CAG);
- zinc-coordinating cysteines are thiolates (CYM) rather than thiols (1AAY);
- a 5′ nucleotide that keeps its phosphate gets no 5′ hydroxyl hydrogen (1EHZ);
- missing residues listed in `REMARK 465` are modelled even when SEQRES cannot
  be aligned to the residue numbering (antibody numbering in 1HZH, 1CA2, 5KZC);
- ligands whose definitions lack some deposited atoms still receive hydrogens
  on the complete parts of the molecule (lipids in 1C3W);
- histidine tautomers are chosen from hydrogen bonds and metal coordination.
