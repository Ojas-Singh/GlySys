"""Rebuild openff-recharge's original AM1-BCC SMIRKS list, with phosphorus.

Run inside openff-recharge/scripts/convert-am1-bcc (MIT licence), whose
am1bcc.csv holds the original AM1-BCC bond charge corrections (Jakalian,
Jack and Bayly, J. Comput. Chem. 2002, 23, 1623):

    convert_am1bcc.py --with-p > data/openff/am1bcc-with-phosphorus.json

Without --with-p the output is identical to openff-recharge's shipped
original-am1-bcc.json.  --with-p enables the phosphorus atom types that
openff-recharge leaves commented out, and adds delocalized terminal P-O and
S-O corrections for ionized phosphates/sulfates in the same form as its
delocalized C-O corrections.
"""
import csv, json, sys

WITH_P = '--with-p' in sys.argv
atom_codes = {
    "11": "[#6X4:1]", "15": "[#6X1,#6X2:1]", "12": "[#6X3$(*=[#6]):1]",
    "13": "[#6X3$(*=[#7,#15]):1]", "14": "[#6X3$(*=[#8,#16]):1]",
    "17": "[#6a$(*~[#7aX2,#8aX2]):1]", "16": "[#6a:1]",
    "23": "[#7X3ar5,#7X3+1,#7X3+0$(*-[#6X3$(*=[#7X3+1])]),$([#7X3](-[#8X1-1])=[#8X1]),$([#7X3](=[#8X1])=[#8X1]):1]",
    "22": "[#7X2-1$(*-[#6X3$(*=[#8X1,#16X1])]),#7X3$(*-[#6X3$(*=[#8X1,#16X1])]):1]",
    "21": "[#7X4,#7X3,#7X2-1:1]", "25": "[#7X1,#7X2+1:1]", "24": "[#7X2+0,#7X2-1ar5:1]",
    "33": "[#8X1$(*=[#6r]@[#7r,#8r]):1]", "32": "[#8X1$(*=[#6X3]-[#8X2]):1]", "31": "[#8X1,#8X2:1]",
}
if WITH_P:
    atom_codes["42"] = "[#15X3$(*=[*]),#15X4$(*=[*]):1]"
    atom_codes["41"] = "[#15X2,#15X3:1]"
atom_codes.update({
    "53": "[#16X4:1]", "52": "[#16X3:1]", "51": "[#16X1,#16X2:1]", "61": "[#14X4:1]",
    "71": "[#9:1]", "72": "[#17:1]", "73": "[#35:1]", "74": "[#53:1]", "91": "[#1:1]",
})
bond_codes = {"01": "-", "02": "=", "03": "#", "06": "-", "07": ":", "08": ":", "09": "~"}
custom = {
    "110951": "[#6X4:1]-[#16X1-1:2]", "150951": "[#6X1,#6X2:1]-[#16X1-1:2]",
    "120951": "[#6X3$(*=[#6]):1]-[#16X1-1:2]", "130951": "[#6X3$(*=[#7,#15]):1]-[#16X1-1:2]",
    "140951": "[#6X3:1](~[#8X1,#16X1])(~[#16X1:2])", "170951": "[#6a$(*~[#7aX2,#8aX2]):1]-[#16X1-1:2]",
    "160951": "[#6a:1]-[#16X1-1:2]",
    "230931": "[$([#7X3](-[#8X1])=[#8X1]),$([#7X3](=[#8X1])=[#8X1]):1]~[#8X1:2]",
    "230631": "[#7X3+1:1]-[#8X1-1:2]", "220631": "[#7X3$(*-[#6X3$(*=[#8])]):1]-[#8X1-1:2]",
    "210631": "[#7X4+1:1]-[#8X1-1:2]", "250631": "[#7X1,#7X2+1:1]-[#8X1-1:2]",
    "240631": "[#7X2+0:1]-[#8X1-1:2]", "110931": "[#6X4:1]-[#8X1-1:2]",
    "150931": "[#6X1,#6X2:1]-[#8X1-1:2]", "120931": "[#6X3$(*=[#6]):1]-[#8X1-1:2]",
    "130931": "[#6X3$(*=[#7,#15]):1]-[#8X1-1:2]", "140931": "[#6X3:1](~[#8X1,#16X1])(~[#8X1:2])",
    "170931": "[#6a$(*~[#7aX2,#8aX2]):1]-[#8X1-1:2]", "160931": "[#6a:1]-[#8X1-1:2]",
}
if WITH_P:
    # Delocalized terminal S-O of ionized sulfates/sulfonates/sulfinates,
    # in the same form as openff-recharge's delocalized C-O corrections.
    custom["310953"] = "[#8X1:1]~[#16X4$(*-[#8X1-1]):2]"
    custom["310952"] = "[#8X1:1]~[#16X3$(*-[#8X1-1]):2]"
    # Delocalized terminal P-O of an ionized phosphate (analogous to C-O above):
    # every terminal O on a phosphorus that carries an O(-1).
    custom["310942"] = "[#8X1:1]~[#15X4$(*-[#8X1-1]):2]"
    custom["310941"] = "[#8X1:1]~[#15X3$(*-[#8X1-1]):2]"
overrides = {"110112": 0.0024, "120114": -0.0172}
all_codes = [*custom]
for a in atom_codes:
    for b in bond_codes:
        for c in atom_codes:
            code = f"{a}{b}{c}"
            if code not in all_codes:
                all_codes.append(code)
rows = sorted(csv.DictReader(open('am1bcc.csv')), key=lambda r: int(r['Index']))
params = {}
for row in rows:
    code = str(row['Code'])[0:6]
    if code not in all_codes:
        continue
    smirks = custom.get(code) or f"{atom_codes[code[0:2]]}{bond_codes[code[2:4]]}{atom_codes[code[4:6]].replace(':1', ':2')}"
    value = overrides.get(code, round(float(row['BCC']), 4))
    params[code] = {"smirks": smirks, "value": value, "provenance": {"code": code}}
ordered = [params[c] for c in all_codes if c in params]
seen = {}
unique = []
for p in ordered:
    if p['smirks'] in seen:
        assert abs(p['value'] - seen[p['smirks']]['value']) < 1e-9, p
        continue
    seen[p['smirks']] = p
    unique.append(p)
json.dump(unique, sys.stdout)
