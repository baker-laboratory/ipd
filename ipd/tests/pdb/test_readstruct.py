import pytest

import ipd

pytest.importorskip('biotite')
TEST_PDBS = [
    '1hv4', '1hv4', '1ql2', '5im6', '1out', '3sne', '1dxh', '1n0e', '1wa3', '1a2n', '1n0e', '1bfr', '1g5q',
    '3woc', '7abl', '2tbv', '2btv'
]
ipd.pdb.download_test_pdbs(TEST_PDBS)

def main():
    ipd.tests.maintest(namespace=globals())

def helper_test_readatoms_types(atomslist):
    assert isinstance(atomslist, list)
    assert len(atomslist) > 0

def helper_test_readatoms_assembly_xforms(atomslist):
    asmx, pdbcode = ipd.get_metadata(atomslist[0])['assembly_xforms pdbcode']
    ic(pdbcode, asmx)
    # assert 0

ipd.tests.make_parametrized_tests(
    globals(),
    'helper_test_readatoms_',
    TEST_PDBS,
    ipd.pdb.readatoms,
    assembly='largest',
    strict=True,
)

def helper_read_8u51(fname):
    fname = ipd.dev.package_testdata_path(f'pdb/{fname}')
    atom = ipd.pdb.readatoms(fname)
    assert len(atom) == 2093
    assert 277 == sum(atom.atom_name == 'CA')
    chatom = ipd.pdb.readatoms(fname, caonly=True, chaindict=True)
    chainlens = {k: len(v) for k, v in chatom.items()}
    ic(chainlens)
    assert chainlens == {'A': 267, 'B': 10}
    assert all(chatom.A.atom_name == 'CA')

def test_readatoms_bcif_gz():
    helper_read_8u51('8u51.bcif.gz')

def test_readatoms_cif_gz():
    helper_read_8u51('8u51.cif.gz')

def test_readatoms_cif():
    helper_read_8u51('8u51.cif')

def test_readatoms_pdb():
    helper_read_8u51('8u51.pdb.gz')

if __name__ == '__main__':
    main()
