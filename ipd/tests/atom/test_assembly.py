import unittest
import pytest
import numpy as np
import ipd

h = ipd.hnumpy

bs = pytest.importorskip('biotite.structure')

TEST_PDBS = ['3sne', '1dxh', '1n0e', '1wa3', '1a2n', '1n0e', '1bfr', '1g5q']

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

def test_assembly_simple():
    assembly = ipd.atom.assembly_from_file('1qys')

def helper_test_assembly_iterate(assembly):
    for ibod, ifrm in assembly.symbodyids():
        body = assembly.body(ibod, ifrm)
        assert body.meta.bodyid == ibod
        assert body.meta.frameid == ifrm

    for ibod, ifrm, body, frame in assembly.enumerate_symbodies():
        assert body.meta.bodyid == ibod
        assert body.meta.frameid == ifrm

def helper_test_assembly_asu_selector(assembly):
    for ibody, iframe, borig, forig in assembly.enumerate_symbodies(order='random', n=10):
        asusel = ipd.atom.AsuSelector(bodyid=ibody, frameid=iframe)
        asu = asusel(assembly)
        assert asu.isclose(borig)

def helper_test_assembly_neighborhood_asu(assembly):
    for ibody, iframe, borig, forig in assembly.enumerate_symbodies(order='random', n=10):
        asusel = ipd.atom.AsuSelector(bodyid=ibody, frameid=iframe)
        hoodsel = ipd.atom.NeighborhoodSelector(min_contacts=10, contact_dist=7)
        hood = hoodsel(asusel, assembly)
        newasu = hood.body(0)
        for ibody2, newasu in enumerate(hood.bodies):
            origid = assembly._idmap[newasu]
            origfid = newasu.get_metadata().frameid
            origbody = assembly.body(origid, origfid)
            assert origid == newasu.get_metadata().bodyid
            assert origbody.atoms is newasu.atoms
            assert h.allclose(hood.frames[ibody][0], np.eye(4))
            assert newasu.isclose(origbody)

def helper_test_assembly_neighborhood_neighbors(assembly):
    for ibasu, ifasu, basu, fasu in assembly.enumerate_symbodies():
        # for ibasu, ifasu, basu, fasu in assembly.enumerate_symbodies(order='random', n=10):
        ic(ibasu, ifasu, basu.pos)
        asusel = ipd.atom.AsuSelector(bodyid=ibasu, frameid=ifasu)
        hoodsel = ipd.atom.NeighborhoodSelector(min_contacts=10, contact_dist=7)
        hood = hoodsel(asusel, assembly)
        assert hood.bodies[0].isclose(basu)
        for ibnew, ifnew, bnew, fnew in hood.enumerate_symbodies():
            borig = assembly.body

            ibmap = assembly._idmap[bnew]
            ifmap = 0
            # oldframe = assembly._framemap[bnew][orig_iframe_for_new]
            origbody = assembly.body(ibmap, ifmap)
            newbody = hood.body(ibnew, ifnew)
            ic(origbody.pos, newbody.pos)
            # assert newbody.isclose(origbody)
        # assert 0

def make_assembly_pdb_tests():
    for pdb in TEST_PDBS:

        class TestAssembly(unittest.TestCase):

            def setUp(self):
                self.assembly = ipd.atom.assembly_from_file(pdb, min_chain_atoms=50)

            for k, func in globals().items():
                if k.startswith('helper_test_assembly_'):
                    locals()[k[7:]] = lambda self, func=func: func(self.assembly)

        globals()[f'TestAssembly_{pdb.upper()}_'] = TestAssembly

make_assembly_pdb_tests()

if __name__ == '__main__':
    main()
