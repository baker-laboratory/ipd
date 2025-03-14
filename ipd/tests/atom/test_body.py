import pytest

bs = pytest.importorskip('biotite.structure')
wu = pytest.importorskip('willutil_cpp')

import ipd
import ipd.homog.hgeom as h

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
        # dryrun=True,
    )

atoms = ipd.atom.get('1qys')

def test_body_simple():
    body = ipd.atom.Body(atoms)

def _celllist_nclash(body, other, radius=3) -> bool:
    cell_list = bs.CellList(body.atoms, radius + 1)
    nclash = 0
    for pos in other[:]:
        nclash += len(cell_list.get_atoms(pos, radius=radius))
    return nclash

def test_body_chash_v_celllist():
    body = ipd.atom.Body(atoms)
    radius = 2.5
    results = ipd.Bunch(cell=[], bvh=[])
    with ipd.dev.Timer() as t:
        for i in range(20):
            body2 = body.movedby(h.trans(i))
            t.checkpoint('setup')
            ncell = _celllist_nclash(body, body2, radius)
            t.checkpoint('cell')
            nbvh = body.nclash(body2, radius)
            t.checkpoint('bvh')
            # print(ncell, nbvh)
            results.mapwise == []
            results.mapwise.append(ncell, nbvh)
            # assert abs(((ncell+1) / (nbvh+1)) - 1) < 0.1
            assert ncell == nbvh
        # print(sum(t.cell) / sum(t.bvh), flush=True)
        assert t.cell > 10 * t.bvh

def test_body_contacts():
    body = ipd.atom.Body(atoms)
    body2 = body.movedby(h.trans(25))
    contact = body.contacts(body2, radius=4)
    assert len(contact.breaks) == 1
    for i, j in contact.pairs:
        assert 4 > h.norm(body[i] - body2[j])

if __name__ == '__main__':
    main()
