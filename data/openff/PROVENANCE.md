# Small-molecule force-field data provenance

## `openff_unconstrained-2.2.1.json`

OpenFF Sage 2.2.1 (`openff_unconstrained-2.2.1.offxml`) from
https://github.com/openforcefield/openff-forcefields (commit
e2b6c5c778db03c3c6b4436fb597f2e66db43b95), converted by
`scripts/data/convert_sage.py` (units kcal/mol, Å, degrees; SMIRKS, ids and
order unchanged; the source SHA-256 is recorded in the file).

Open Force Field Initiative, *Sage* force field. Licensed under the Creative
Commons Attribution 4.0 International licence (CC-BY-4.0),
https://creativecommons.org/licenses/by/4.0/. The data were reformatted, not
modified.

## `am1bcc-with-phosphorus.json`

Original AM1-BCC bond charge corrections (Jakalian, Jack and Bayly,
J. Comput. Chem. 2002, 23, 1623-1641) as tabulated in openff-recharge
`scripts/convert-am1-bcc/am1bcc.csv` (https://github.com/openforcefield/openff-recharge,
commit dad79f5a785b7013c0366d13efb1990f603ebd4a; MIT licence, Copyright (c)
2020 Simon Boothroyd), converted to SMIRKS by `scripts/data/convert_am1bcc.py`.
The conversion reproduces openff-recharge's `original-am1-bcc.json` exactly
and additionally enables its phosphorus atom types and delocalized P-O / S-O
corrections for ionized phosphates and sulfates.

## AM1 parameters (`src/smirnoff/am1.rs`)

AM1 parameters and the NDDO integral scheme follow MOPAC
(https://github.com/openmopac/mopac, commit
1d9d92b0283f197616f1e9e76d1ee09e2bc21e72), Copyright 2021 Virginia Polytechnic
Institute and State University, Apache License 2.0.
