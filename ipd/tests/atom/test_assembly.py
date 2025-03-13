import pytest
import numpy as np
import ipd

h = ipd.hnumpy

bs = pytest.importorskip('biotite.structure')

TEST_PDBS = [
    '3sne',
    '1dxh',
    '1n0e',
    '1wa3',
    '1a2n',
    '1n0e',
    '1bfr',
    '1g5q'  #, '3woc', '7abl', '2tbv'
]

config_test = ipd.Bunch(
    re_only=[],
    re_exclude=[],
)

def main():
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
        dryrun=False,
    )

# def test_assembly_simple():
# assembly = ipd.atom.assembly('1qys')

def helper_test_asu_selector(assembly):
    for i, j in assembly.symbodyids():
        asusel = ipd.atom.AsuSelector(bodyid=i, frameid=j)
        asu = asusel(assembly)
        assert asu._resbvh is assembly.bodies[i]._resbvh
        assert h.allclose(asu.pos, assembly.frames[i][j])

def helper_test_neighborhood_selector(assembly):
    for ibod, ifrm in assembly.symbodyids():
        asusel = ipd.atom.AsuSelector(bodyid=ibod, frameid=ifrm)
        hoodsel = ipd.atom.NeighborhoodSelector(min_contacts=10, contact_dist=7)
        hood = hoodsel(asusel, assembly)
        assert hood.bodies[0].atoms is assembly.bodies[ibod].atoms
        assert h.allclose(hood.bodies[0].pos, assembly.frames[ibod][ifrm])
        assert h.allclose(hood.frames[0][0], np.eye(4))
        assert h.allclose(assembly.body(ibod, ifrm))

def make_assembly_pdb_tests():
    for pdb in TEST_PDBS:

        class TestAssembly():

            def __init__(self):
                self.assembly = ipd.atom.create_assembly(pdb, min_chain_atoms=50)

            def test_asu_selector(self):
                helper_test_asu_selector(self.assembly)

            def test_neighborhood_selector(self):
                helper_test_neighborhood_selector(self.assembly)

        globals()[f'TestAssembly_{pdb.upper()}_'] = TestAssembly

make_assembly_pdb_tests()

if __name__ == '__main__':
    main()
