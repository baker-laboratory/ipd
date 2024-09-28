import os
import re
import numpy as np

_matchers = None

def guess_sym_from_fnames(fnames):
    global _matchers
    if _matchers is None:
        _matchers = [re.compile(f'\\b({regex})\\b') for regex in 'tet oct icos c\\d+ d\\d+'.split()]
    hits = []
    for f in fnames:
        f = f.lower().replace('_', ' ')
        hits.append([])
        for matcher in _matchers:
            if match := matcher.search(f):
                hits[-1].append(match.group(1))
    allhits = [hit for hit in hits[0] if all(hit in h for h in hits)]
    if len(allhits) == 1:
        return allhits[0]
    return None

def guess_sym_from_directory(path, suffix=('.pdb', '.pdb.gz', '.cif', '.bcif')):
    assert os.path.isdir(path)
    if fnames := filter(lambda f: f.endswith(suffix) and f[0] != '_', os.listdir(path)):
        return guess_sym_from_fnames(fnames)
    else:
        return None

def guess_symmetry(xyz):
    'ca only, nchain x nres x p'
    match len(xyz):
        case 1:
            return 'C1'
        case 2:
            return 'C2'
        case 3:
            return 'C3'
        case 4:
            return 'C4'
        case 5:
            return 'C5'
        case 6:
            return 'C6'
        case 7:
            return 'C7'
        case 8:
            return 'C8'
        case 9:
            return 'C9'
        case 10:
            return 'C10'
        case 11:
            return 'C11'
        case 12:
            return 'tet'
        case 24:
            return 'oct'
        case 30:
            return 'icos'
        case 60:
            return 'icos'
        case _:
            return 'unknown'
