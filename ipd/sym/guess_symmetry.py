import os
import re

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
    # print(fnames[0], len(fnames), hits)
    if len(allhits) == 1:
        return allhits[0].upper()
    return None

def guess_sym_from_directory(path, suffix=('.pdb', '.pdb.gz', '.cif', '.bcif')):
    assert os.path.isdir(path)
    if fnames := filter(lambda f: f.endswith(suffix) and f[0] != '_', [os.path.join(path, f) for f in os.listdir(path)]):
        return guess_sym_from_fnames(list(fnames))
    else:
        return None

def guess_symmetry(xyz):
    'ca only, nchain x nres x p'
    if len(xyz) == 1: return 'C1'
    if len(xyz) == 2: return 'C2'
    if len(xyz) == 3: return 'C3'
    if len(xyz) == 4: return 'C4'
    if len(xyz) == 5: return 'C5'
    if len(xyz) == 6: return 'C6'
    if len(xyz) == 7: return 'C7'
    if len(xyz) == 8: return 'C8'
    if len(xyz) == 9: return 'C9'
    if len(xyz) == 10: return 'C10'
    if len(xyz) == 11: return 'C11'
    if len(xyz) == 12: return 'tet'
    if len(xyz) == 24: return 'oct'
    if len(xyz) == 30: return 'icos'
    if len(xyz) == 60: return 'icos'
    return 'unknown'
