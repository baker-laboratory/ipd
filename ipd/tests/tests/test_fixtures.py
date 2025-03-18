import pytest

import ipd

def main():
    ipd.tests.maintest(namespace=globals())

def test_atoms():
    pytest.importorskip('biotite')
    atoms = ipd.tests.atoms('tiny.pdb')
    assert len(atoms) == 8

def test_pdbcontents():
    ...

def test_pdbfname():
    ...

def test_pdbfnames():
    ...

def test_three_PDBFiles():
    ...

def test_pdbfile():
    ...

def test_pdb1pgx():
    ...

def test_pdb1coi():
    ...

def test_pdb1qys():
    ...

def test_ncac():
    ...

def test_ncaco():
    ...

if __name__ == '__main__':
    main()
