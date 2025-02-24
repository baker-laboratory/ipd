import ipd

def build_from_components(atoms1: 'AtomArray', atoms2: 'AtomArray'):
    sinfo1, sinfo2 = map(ipd.sym.syminfo_from_atoms, (atoms1, atoms2))
    assert 0
