import pytest
import numpy as np

import ipd

bio = pytest.importorskip('biotite')

def main():
    ipd.tests.maintest(namespace=globals())

def test_build_from_components_abbas():
    atoms1 = ipd.atom.load(ipd.dev.package_testdata_path('pdb/L2_D1_C3_Apo.pdb'), chainlist=True)
    atoms2 = ipd.atom.load(ipd.dev.package_testdata_path('pdb/L2_D1_C5.pdb'), chainlist=True)
    atoms = ipd.sym.build_from_components_abbas(atoms1, atoms2, tol=1e-1)
    coms = np.stack(ipd.atom.chaincom(atoms))
    refcoms = np.stack([([13.49487, 8.46107, 53.79263]), ([7.62207, 35.9939, 43.83445]),
                        ([-14.70482, 17.68098, 51.24251]), ([14.22432, -12.10113, 53.72419]),
                        ([31.22279, -17.77857, 43.61546]), ([41.02541, -0.74452, 37.55242]),
                        ([30.03983, 15.46383, 43.83708])])
    assert np.allclose(coms, refcoms, atol=1e-2)
    # ipd.pdb.dumpatoms(atoms, '/tmp/abbas.pdb')

if __name__ == '__main__':
    main()
