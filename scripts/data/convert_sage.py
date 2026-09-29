#!/usr/bin/env python3
"""Convert an OpenFF SMIRNOFF .offxml file into GlySys's compact JSON.

Usage: convert_sage.py openff_unconstrained-2.2.1.offxml > data/openff/openff_unconstrained-2.2.1.json

Units become kcal/mol, Å and degrees; parameters keep their SMIRKS, ids and
order (the last matching parameter wins).
"""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET


def quantity(value, unit):
    number = float(value.split('*')[0].strip())
    text = value.replace(' ', '')
    if unit == 'A':
        return number * 10.0 if 'nanometer' in text else number
    if unit in ('kcal', 'kcalA2', 'kcalrad2'):
        assert 'kilocalorie' in text, value
    if unit == 'kcalA2':
        assert 'angstrom**-2' in text, value
    if unit == 'kcalrad2':
        assert 'radian**-2' in text, value
    if unit == 'deg':
        assert 'degree' in text, value
    return number


def main(path):
    raw = open(path, 'rb').read()
    root = ET.fromstring(raw)
    out = {'source': path.rsplit('/', 1)[-1], 'sha256': hashlib.sha256(raw).hexdigest(),
           'aromaticity_model': root.get('aromaticity_model')}
    for handler in root:
        tag = handler.tag
        if tag == 'Bonds':
            out['bonds'] = [{'smirks': b.get('smirks'), 'id': b.get('id'), 'k': quantity(b.get('k'), 'kcalA2'),
                             'length': quantity(b.get('length'), 'A')} for b in handler]
        elif tag == 'Angles':
            out['angles'] = [{'smirks': b.get('smirks'), 'id': b.get('id'), 'k': quantity(b.get('k'), 'kcalrad2'),
                              'angle': quantity(b.get('angle'), 'deg')} for b in handler]
        elif tag in ('ProperTorsions', 'ImproperTorsions'):
            key = 'propers' if tag == 'ProperTorsions' else 'impropers'
            items = []
            for t in handler:
                terms, n = [], 1
                while t.get(f'periodicity{n}') is not None:
                    idivf = t.get(f'idivf{n}')
                    terms.append({'periodicity': int(t.get(f'periodicity{n}')), 'phase': quantity(t.get(f'phase{n}'), 'deg'),
                                  'k': quantity(t.get(f'k{n}'), 'kcal'), 'idivf': float(idivf) if idivf else None})
                    n += 1
                assert not any(k.startswith('k_bondorder') for k in t.attrib), t.get('id')
                items.append({'smirks': t.get('smirks'), 'id': t.get('id'), 'terms': terms})
            out[key] = items
            out[key + '_default_idivf'] = handler.get('default_idivf')
        elif tag == 'vdW':
            out['vdw_scale14'] = float(handler.get('scale14'))
            out['vdw'] = []
            for a in handler:
                rmin_half = a.get('rmin_half')
                r = quantity(rmin_half, 'A') if rmin_half else quantity(a.get('sigma'), 'A') * 2 ** (1 / 6) / 2
                out['vdw'].append({'smirks': a.get('smirks'), 'id': a.get('id'),
                                   'epsilon': quantity(a.get('epsilon'), 'kcal'), 'rmin_half': r})
        elif tag == 'Electrostatics':
            out['electrostatics_scale14'] = float(handler.get('scale14'))
        elif tag == 'ToolkitAM1BCC':
            out['charges'] = 'AM1-BCC'
    json.dump(out, sys.stdout, separators=(',', ':'))


if __name__ == '__main__':
    main(sys.argv[1])
