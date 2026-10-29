"""Extract the GLYCAM-Web residues that GLYCAM06j-1 lacks.

GLYCAM-Web builds with GMML's ``dat/CurrentParams/GLYCAM_06k_gmml2.lib``. It
is a superset of ``GLYCAM_06j-1.prep`` (the AmberTools distribution): the
residues both files hold are identical except ``3dR``, whose methyl
hydrogens are typed ``Hc`` in GMML (``H1`` in 06j-1). This script copies,
unmodified, every GMML entry that GlySys does not already load from
06j-1/GAGS, plus ``3dR``, into ``data/amber/lib/GLYCAM_06k_gmml2_additions.lib``.

    python scripts/data/extract_glycam_gmml2.py [path/to/GLYCAM_06k_gmml2.lib]

Without a path the file is downloaded from the pinned GMML commit.
"""

from __future__ import annotations

import hashlib
import re
import sys
import urllib.request
from pathlib import Path

COMMIT = "4e4184a4"  # GLYCAM-Web/gmml, 2024-06-28, "Adding the correct lib from gmml2"
URL = f"https://raw.githubusercontent.com/GLYCAM-Web/gmml/{COMMIT}/dat/CurrentParams/GLYCAM_06k_gmml2.lib"
SHA256 = "ce2df67c1a189b9eb74e86ae7fc7fcf50d191a68b25d20bc2e0b8c5dd8d1749a"
ROOT = Path(__file__).resolve().parents[2]
# GLYCAM06j-1 entries GMML corrects: 3dR methyl H typed H1, 4AE three atoms bonded
# to themselves, WdR with O3 on C2.
REPLACED = {"3dR", "4AE", "WdR"}


def prep_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and len(fields[0]) == 3 and fields[1] == "INT":
            names.add(fields[0])
    return names


def main() -> None:
    if len(sys.argv) > 1:
        raw = Path(sys.argv[1]).read_bytes()
    else:
        raw = urllib.request.urlopen(URL, timeout=300).read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        sys.exit(f"unexpected source: sha256 {digest}, expected {SHA256}")
    text = raw.decode()
    have = prep_names(ROOT / "data/amber/prep/GLYCAM_06j-1.prep") | prep_names(
        ROOT / "data/amber/prep/GLYCAM_06j-1_GAGS.prep")
    index_block, rest = text.split("!entry.", 1)
    names = re.findall(r'^ "([^"]+)"$', index_block, re.M)
    keep = [n for n in names if n not in have or n in REPLACED]
    sections = re.split(r"(?m)^(?=!entry\.)", "!entry." + rest)
    out = ["!!index array str"] + [f' "{n}"' for n in keep]
    wanted = set(keep)
    for section in sections:
        match = re.match(r"!entry\.([^.]+)\.unit\.", section)
        if match and match.group(1) in wanted:
            out.append(section.rstrip("\n"))
    target = ROOT / "data/amber/lib/GLYCAM_06k_gmml2_additions.lib"
    target.write_text("\n".join(out) + "\n")
    print(f"source sha256 {digest}; {len(keep)} of {len(names)} entries -> {target}")


if __name__ == "__main__":
    main()
