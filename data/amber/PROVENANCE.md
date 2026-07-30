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

Files remain in their native Amber formats so their provenance is auditable
and the Rust parsers can be tested against the original source representation.

