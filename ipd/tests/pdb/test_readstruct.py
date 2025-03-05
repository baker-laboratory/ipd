import pytest

import ipd

bio = pytest.importorskip('biotite')

def main():
    ipd.tests.maintest(namespace=globals())

def helper_test_readatoms(fname):
    fname = ipd.dev.package_testdata_path(f'pdb/{fname}')
    atom = ipd.pdb.readatoms(fname)
    assert len(atom) == 2093
    assert 277 == sum(atom.atom_name == 'CA')
    chatom = ipd.pdb.readatoms(fname, caonly=True, chaindict=True)
    assert {k: len(v) for k, v in chatom.items()} == {'A': 267, 'B': 10}
    assert all(chatom.A.atom_name == 'CA')

def test_readatoms_bcif_gz():
    helper_test_readatoms('8u51.bcif.gz')

def test_readatoms_cif_gz():
    helper_test_readatoms('8u51.cif.gz')

def test_readatoms_cif():
    helper_test_readatoms('8u51.cif')

def test_readatoms_pdb():
    helper_test_readatoms('8u51.pdb.gz')

def test_readatoms_cif_assembly():
    ipd.pdb.readatoms(ipd.dev.package_testdata_path('pdb/5im6.bcif.gz'), assembly='largest')

def test_readatoms_1hv4():
    ipd.pdb.readatoms(ipd.dev.package_testdata_path('pdb/1hv4.bcif.gz'), assembly='largest')

def test_readatoms_1out():
    ipd.pdb.readatoms(ipd.dev.package_testdata_path('pdb/1out.bcif.gz'), assembly='largest')

def test_readatoms_1ql2():
    ipd.pdb.readatoms(ipd.dev.package_testdata_path('pdb/1ql2.bcif.gz'), assembly='largest')

if __name__ == '__main__':
    main()
