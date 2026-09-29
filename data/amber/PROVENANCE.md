# Amber force-field data provenance

These files are an unmodified subset of AmberTools 23.6 `dat/leap`, copied
from the conda-forge `ambertools-23.6` package. AmberTools states that the
force-field parameter files under `dat/leap` have been placed in the public
domain by their authors.

The subset supplies:

- Amber ff14SB amino-acid residue templates and parameters
- GLYCAM06j-1 carbohydrate and glycosylated-amino-acid templates/parameters
- TIP3P solvent geometry and parameters
- Joung-Chetham monovalent TIP3P ion parameters
- Li-Merz 12-6 divalent/trivalent/tetravalent TIP3P ion parameters
  (`frcmod.ions234lm_126_tip3p`), used for Mg2+, Ca2+, Zn2+ and other
  structural metal ions kept by `prepare`
- Amber OL15 DNA (`DNA.OL15.lib`) and OL3 RNA (`RNA.lib`) residue templates,
  used by the structure fixer for nucleic-acid atoms and hydrogens, and the
  OL15 DNA parameter modifications (`frcmod.DNA.OL15`) used by `prepare`

Files remain in their native Amber formats so their provenance is auditable
and the Rust parsers can be tested against the original source representation.

