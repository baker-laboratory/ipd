import random
import numpy as np
import pytest

bs = pytest.importorskip('biotite.structure')
hg = pytest.importorskip('hgeom')

import ipd
import ipd.homog.hgeom as h

config_test = ipd.Bunch(
    re_only=[],
    re_exclude=[],
)
BODY_TEST_PDBS = ['1qys']
# BODY_TEST_PDBS = ['2tbv']
SYMBODY_TEST_PDBS = ['1dxh', '1wa3', '6u9d', '3sne', '1n0e', '1a2n', '1n0e', '1bfr', '1g5q']

def main():
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
    # body = ipd.atom.body_from_file('3sne', assembly='1')
    # ipd.showme(body)
    # return
    symbody = ipd.atom.symbody_from_file('3sne',
                                         assembly='largest',
                                         components='largest_assembly',
                                         maxsub=9999)
    ipd.showme(symbody)
    ipd.icv(symbody.meta.assembly_xforms)
    symbody.asu.atoms = symbody.asu.atoms[symbody.asu.atoms.atom_name == 'CA']

    # ipd.showme(symbody.asu)
    ipd.dev.global_timer.report()

def helper_test_body_centered(body):
    assert h.allclose(body.rg, body.centered.rg, atol=1)
    assert h.allclose(body.pos, np.eye(4))
    assert h.allclose(body.centered.com, [0, 0, 0])

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
    kissing = otherbody.slide_into_contact(body, radius=3)
    # ipd.icv(kissing.pos[:3, 3] - otherbody.pos[:3, 3])
    # ipd.showme(kissing, body)
    contacts = kissing.contacts(body, radius=5)
    assert len(contacts.pairs) > 0
    assert len(contacts.ranges) == 1
    for i, j in contacts.pairs:
        assert 5 > h.norm(kissing[i] - body[j])

def helper_test_symbody_slide(symbody):
    top7 = ipd.atom.body_from_file('1qys').centered
    symbody = symbody.centered
    for body1, body2 in ipd.it.combinations([symbody, top7], 2):
        # kissing, slidedir = body1.slide_into_contact_rand(body2, radius=2)
        kissing, slidedir = body1.slide_into_contact(body2, radius=2), h.vec([1, 0, 0])
        fail = np.any(body2.hasclash(kissing))
        body_clash = kissing.movedby(3 * slidedir)  # slide is a little goofy, move a lot
        fail |= not np.any(body_clash.hasclash(body2))
        fail |= not body2.contacts(kissing, radius=5).total_contacts
        # ic(np.any(body_clash.hasclash(body2)))
        # ic(body2.contacts(kissing, radius=5).total_contacts)
        assert not fail
        if fail:
            ipd.showme(body2, kissing)
            assert 0

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

def helper_test_symbody_contacts(symbody):
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
    for ibody1, ibody2, body1, body2, iatom1, iatom2 in contact4:
        assert np.all(h.dist(body1[iatom1], body2[iatom2]) < 4)

def helper_test_symbody_contact_scan(symbody):
    isub = random.randint(0, len(symbody.frames) - 1)
    asu = symbody.bodies[isub]
    contactlist = symbody.contacts(asu, exclude=isub, radius=5)
    contactmat = contactlist.contact_matrix_stack(symbody.asu.atoms.res_id)
    topk = contactmat.topk_fragment_contact_by_subset_summary(fragsize=21, k=13, stride=7)
    assert topk.index.keys() == topk.vals.keys()
    for subs, idx in topk.index.items():
        ic(subs, idx[:4], topk.vals[subs][:4])

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
