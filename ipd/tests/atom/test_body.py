import numpy as np
import pytest

bs = pytest.importorskip('biotite.structure')
wu = pytest.importorskip('willutil_cpp')

import ipd
import ipd.homog.hgeom as h

config_test = ipd.Bunch(
    re_only=[],
    re_exclude=[],
)
BODY_TEST_PDBS = ['1qys']
# BODY_TEST_PDBS = ['2tbv']
SYMBODY_TEST_PDBS = ['6u9d', '3sne', '1dxh', '1n0e', '1wa3', '1a2n', '1n0e', '1bfr', '1g5q']

def main():
    debug_body_load_speed()
    return
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
        # dryrun=True,
    )

def _celllist_nclash(cell_list, other, radius: float = 3) -> int:
    nclash = 0
    for pos in other[:]:
        nclash += len(cell_list.get_atoms(pos, radius=radius))
    return nclash

def debug_body_load_speed():
    symbody = ipd.atom.symbody_from_file('2tbv',
                                         assembly='largest',
                                         components='largest_assembly',
                                         maxsub=9999)
    ipd.icv(symbody.meta.assembly_xforms)
    symbody.asu.atoms = symbody.asu.atoms[symbody.asu.atoms.atom_name == 'CA']

    # ipd.showme(symbody.asu)
    ipd.dev.global_timer.report()

def helper_test_body_chash_v_celllist(body):
    results = ipd.Bunch(cell=[], bvh=[])
    with ipd.dev.Timer() as t:
        cell_list = bs.CellList(body.atoms, 3.0)
        t.checkpoint('cell')
        ipd.atom.Body(body.atoms)
        t.checkpoint('bvh')
        rg = int(body.rg)
        for i in range(rg, rg + 20):
            body2 = body.movedto(h.trans(i))
            t.checkpoint('setup')
            ncell = _celllist_nclash(cell_list, body2, 3.0)
            t.checkpoint('cell')
            nbvh = body.nclash(body2, 3.0)
            t.checkpoint('bvh')
            # print(ncell, nbvh)
            results.mapwise.append(ncell, nbvh)
            # assert abs(((ncell+1) / (nbvh+1)) - 1) < 0.1
            # ipd.icv(ncell, nbvh)
            if nbvh > 100: assert 1 - abs(ncell / nbvh) < 0.01
            else: assert nbvh == ncell
        ipd.icv(sum(t.cell) / sum(t.bvh))
        assert sum(t.cell) > 10 * sum(t.bvh)

def helper_test_body_positioned_atoms(body):
    assert h.allclose(h.xform(body.pos, body.atoms.coord), body.positioned_atoms.coord)

def helper_test_body_contacts(body):
    otherbody = body.clone()
    kissing = otherbody.slide_into_contact(body, radius=3, direction='random')
    # ipd.icv(kissing.pos[:3, 3] - otherbody.pos[:3, 3])
    # ipd.showme(kissing, body)
    contacts = kissing.contacts(body, radius=5)
    assert len(contacts.pairs) > 0
    assert len(contacts.breaks) == 1
    for i, j in contacts.pairs:
        assert 5 > h.norm(kissing[i] - body[j])

def helper_test_symbody_atoms(symbody):
    test = h.xform(symbody.pos, symbody.frames, symbody.asu.atoms)
    assert all(isinstance(t, bs.AtomArray) for t in test)
    symatoms = symbody.centered.atomsel(caonly=True)
    # assert h.allclose(symbody.rg, bs.gyration_radius(symatoms), atol=1) # slow
    assert len(symbody.asu.atomsel(caonly=True)) * len(symbody.frames) == len(symatoms)

def helper_test_symbody_selfclash(symbody):
    selfclash = symbody.hasclash(symbody)
    assert h.allclose(selfclash.astype(int), np.eye(len(symbody)))

def helper_test_symbody_selfnclash(symbody):
    assert symbody.nclash(symbody, radius=0.3).sum() == symbody.natoms

def helper_test_symbody_contct(symbody):
    rg = symbody.rg
    symbody2 = symbody.movedby([4 * rg, 0, 0])
    assert not np.any(symbody.hasclash(symbody2))
    assert h.allclose(symbody.pos, np.eye(4))
    # ipd.icv(symbody.pos)
    symbody2 = symbody.movedto([int(1.5 * rg), 0, 0])
    searchrange, stopisect = range(int(1.5 * rg), int(4 * rg)), False
    if not np.any(symbody.hasclash(symbody2)):
        searchrange, stopisect = range(int(4 * rg), 0, -1), True
    ipd.dev.global_timer.checkpoint(interject=True)
    for r in searchrange:
        symbody2 = symbody2.movedto([r, 0, 0])
        clash = symbody.hasclash(symbody2).sum()
        if stopisect and clash: break
        if not stopisect and not clash: break
    else:
        assert 0, 'clashes forever found'
    ipd.dev.global_timer.checkpoint('slide')

    contact4 = symbody.contacts(symbody2, radius=4)
    # if not contact4: ipd.showme(symbody, symbody2)
    assert len(contact4)
    for body1, body2, iatom1, iatom2 in contact4:
        assert np.all(h.dist(body1[iatom1], body2[iatom2]) < 4)

ipd.tests.make_parametrized_tests(
    globals(),
    'helper_test_body_',
    BODY_TEST_PDBS,
    ipd.atom.body_from_file,
)
ipd.tests.make_parametrized_tests(
    globals(),
    'helper_test_symbody_',
    SYMBODY_TEST_PDBS,
    ipd.atom.symbody_from_file,
    components='largest_assembly',
    strict=True,
)

if __name__ == '__main__':
    main()
