import pytest

import ipd

bio = pytest.importorskip('biotite')

def main():
    ipd.tests.maintest(namespace=globals())

@pytest.mark.skip
def test_build_from_components_abbas():
    atoms1 = ipd.pdb.readatoms(ipd.dev.package_testdata_path('pdb/L2_D1_C3_Apo.pdb'), bychain=True)
    atoms2 = ipd.pdb.readatoms(ipd.dev.package_testdata_path('pdb/L2_D1_C5.pdb'), bychain=True)
    atoms = ipd.sym.build_from_components_abbas(list(atoms1.values()), list(atoms2.values()), tol=1e-1)
    ipd.pdb.dumpatoms(atoms, 'lib/ipd/abbas.pdb')

if __name__ == '__main__':
    main()
