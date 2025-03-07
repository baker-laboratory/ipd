import pytest

pytest.importorskip('biotite')
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
    )

def test_body_simple():
    atoms = ipd.atom.testdata('1qys')
    body = ipd.atom.Body(atoms)

def test_body_chash_v_celllist():
    atoms = ipd.atom.testdata('1qys')
    body = ipd.atom.Body(atoms)
    radius = 2.5
    results = ipd.Bunch(cell=[], bvh=[])
    with ipd.dev.Timer() as t:
        for i in range(100):
            body2 = body.xformed(h.trans(i / 5))
            t.checkpoint('setup')
            ncell = body.nclash_celllist(body2, radius)
            t.checkpoint('cell')
            nbvh = body.nclash(body2, radius)
            t.checkpoint('bvh')
            results.mapwise == []
            results.mapwise.append(ncell, nbvh)
            assert abs(((ncell+1) / (nbvh+1)) - 1) < 0.1
        print(sum(t.cell) / sum(t.bvh), flush=True)
        assert t.cell > 10 * t.bvh

    # assert 0

if __name__ == '__main__':
    main()
