import pytest

pytest.importorskip('biotite')
pytest.importorskip('willutil_cpp')

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

atoms = ipd.atom.testdata('1qys')

def test_body_simple():
    body = ipd.atom.Body(atoms)

def test_body_chash_v_celllist():
    body = ipd.atom.Body(atoms)
    radius = 2.5
    results = ipd.Bunch(cell=[], bvh=[])
    with ipd.dev.Timer() as t:
        for i in range(20):
            body2 = body.xformed(h.trans(i))
            t.checkpoint('setup')
            ncell = body._celllist_nclash(body2, radius)
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
    body2 = body.xformed(h.trans(25))
    contacts, ranges = body.contacts(body2, radius=4)
    assert len(ranges) == 1
    for i, j in contacts:
        assert 4 > h.norm(body[i] - body2[j])

if __name__ == '__main__':
    main()
