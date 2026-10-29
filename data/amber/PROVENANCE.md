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


## GLYCAM-Web additions

`lib/GLYCAM_06k_gmml2_additions.lib` holds the entries of GLYCAM-Web's
library `GLYCAM_06k_gmml2.lib` (GLYCAM-Web/gmml, `dat/CurrentParams`, commit
`4e4184a4`, 2024-06-28, sha256 `ce2df67c…`) that GLYCAM06j-1 lacks, plus three
entries that correct GLYCAM06j-1: `3dR` (methyl hydrogens typed `Hc`), `4AE`
(06j-1 bonds H5, O5 and C6 to themselves) and `WdR` (06j-1 bonds O3 to C2).
The 1,474 other residues both files hold are identical. Entries are copied
unmodified by `scripts/data/extract_glycam_gmml2.py`, which checks the source
checksum. The GMML repository is licensed LGPL-3.0, unlike the AmberTools
files above; this matters for the commercial licence route.

GlySys never chooses a template that GLYCAM_06j.dat does not parameterize
completely (atom types, bonds, angles, torsions); 48 are excluded this way,
among them the L-IdoA ring conformers typed `CG`/`OS`/`OH`, the ammonium
(`N3`) and protonated-acid forms, and the GAG Delta-4,5 uronates.

Library coordinates that contradict the residue they name, found by reading
every template with crabWURCS: GLYCAM-Web's galactofuranoses (`xLU`, `xLD`)
have C5 inverted (L-altro), and `xYn` (named beta-D-GlcpN, charged unlike
`xYN`) is built alpha. These centres do not decide the residue and are
reported when an input follows them (`carbohydrate::LIBRARY_STEREO_ERRORS`).
