import random
import tempfile

import numpy as np
import pytest

import ipd

# ic.configureOutput(includeContext=True, contextAbsPath=False)

pytest.skip(allow_module_level=True)

def main():
    test_symelem()
    assert 0

    test_xtalrad_I213()
    assert 0

    test_xtal_cellframes_P_4_3_2_422()
    test_xtal_cryst1_P_4_3_2_422()

    test_xtal_cellframes_P_4_3_2_432()
    test_xtal_cryst1_P_4_3_2_432()
    assert 0

    test_xtal_performance()

    test_xtalrad_I213()

    _analize_xtal_asu_placement()
    assert 0

    test_xtal_cryst1_F_4_3_2(dump_pdbs=False)
    # test_xtal_cryst1_P_41_3_2(dump_pdbs=False)
    # test_xtal_cryst1_I_41_3_2(dump_pdbs=False)
    # test_xtal_cryst1_P_4_3_2(dump_pdbs=False)
    # test_xtal_cryst1_I_4_3_2(dump_pdbs=False)

    test_symelem_mobile()
    # assert 0

    # assert 0

    test_asucen()
    test_dump_pdb()

    if True:
        for xname in ipd.sym.all_xtal_names():
            ic(xname)  # type: ignore
            x = ipd.sym.Xtal(xname)

            # print(
            # f"def test_xtal_cellframes_{xname.replace(' ','_')}(*a, **kw):\n    helper_test_xtal_cellframes('{xname}', *a, **kw)"
            # )
            helper_test_xtal_cellframes(xname)
            if x.dimension == 3:
                helper_test_xtal_cryst1(xname)
                # if x.dimension == 3:
                # print(
                # f"def test_xtal_cryst1_{xname.replace(' ','_')}(*a, **kw):\n    helper_test_xtal_cryst1('{xname}', *a, **kw)"
                # )

    # csize = 50
    # x = ipd.sym.Xtal('I 4 3 2')
    # x.dump_pdb('xtal.pdb', cellsize=csize, cells=3, xtalrad=0.7)
    # test_xtal_cryst1_I_4_3_2(dump_pdbs=False)
    # assert 0

    helper_test_coords_to_asucen_0("I 21 3")  # type: ignore
    # assert 0

    test_hxtal_viz(
        xtalname="P 4 3 2 43",
        headless=False,
        # symelemscale=0.8,
        cells=1,
        # scaleptrad=1,
        showcube=False,
        showsymelems=True,
    )
    # test_xtal_cryst1_P_4_3_2_43()
    assert 0

    test_xtal_cryst1_P_4_3_2(dump_pdbs=False)

    # test_hxtal_viz(xtalname='I 41 3 2', headless=False, cells=1)
    # test_hxtal_viz(xtalname='P 4 3 2', headless=False, cells=1)
    # test_hxtal_viz(xtalname='F 4 3 2', headless=False, cells=1)
    # test_hxtal_viz(xtalname='I 4 3 2', headless=False, cells=1)
    # test_hxtal_viz(xtalname='P 41 3 2', headless=False, cells=1)

    assert 0

    # test_xtal_cryst1_P_41_3_2(dump_pdbs=False)
    # test_xtal_cryst1_P_4_3_2(dump_pdbs=False)
    # test_xtal_cryst1_F_4_3_2(dump_pdbs=False)
    # test_xtal_cryst1_I_4_3_2(dump_pdbs=False)
    # assert 0

    # assert 0

    if 1:
        test_hxtal_viz(
            xtalname="I 41 3 2",
            headless=False,
            cells=(-1, 0),
            symelemscale=0.8,
            fansize=0.08,
            # fansize=np.array([1.7, 1.2, 0.7]) / 3,
            # fancover=10,
            # symelemtwosided=True,
            showsymelems=True,
            # pointshift=(0.2, 0.2, 0.1),
            scaleptrad=1,
            showcube=False,
        )
        assert 0

    if 0:
        test_hxtal_viz(
            xtalname="I4132_322",
            headless=False,
            cells=(-1, 0),
            symelemscale=0.3,
            fansize=np.array([1.7, 1.2, 0.7]) / 3,
            fancover=10,
            symelemtwosided=True,
            showsymelems=True,
            # pointshift=(0.2, 0.2, 0.1),
            scaleptrad=1,
            showcube=False,
        )
        assert 0
        """Run /home/sheffler/pymol3/misc/G222.py;
        gyroid(10,r=11,cen=Vec(5,5,5)); set light, [ -0.3, -0.30, 0.8 ]"""
        assert 0, "aoisrtnoiarnsiot"

    noshow = True

    test_asucen(headless=noshow)

    test_xtal_L6m322(headless=noshow)
    test_xtal_L6_32(headless=noshow)
    # assert 0, 'stilltest viz'

    if not noshow:
        test_hxtal_viz(xtalname="I 21 3", headless=noshow)
        test_hxtal_viz(xtalname="P 2 3", headless=noshow)
        test_hxtal_viz(xtalname="P 21 3", headless=noshow)
        test_hxtal_viz(xtalname="I 41 3 2", headless=noshow)
        test_hxtal_viz(xtalname="I4132_322", headless=noshow)
        # assert 0

    test_xtal_cryst1_I_21_3(dump_pdbs=False)
    test_xtal_cryst1_P_2_3(dump_pdbs=False)
    test_xtal_cryst1_P_21_3(dump_pdbs=False)
    test_xtal_cryst1_I_41_3_2(dump_pdbs=False)
    test_xtal_cryst1_I4132_322(dump_pdbs=False)
    test_xtal_cellframes()
    test_symelem()

    # _test_hxtal_viz_gyroid(headless=False)
    ic("test_xtal.py DONE")  # type: ignore

@pytest.mark.fast
def test_xtal_performance():
    t = ipd.dev.Timer()

    ipd.sym.frames("L632", cells=None, cellsize=50)
    t.checkpoint("primaryframes_init")
    ipd.sym.frames("L632", cells=None, cellsize=50, timer=t)
    ipd.sym.frames("L632", cells=None, cellsize=50, timer=t)
    t.checkpoint("primaryframes")

    x = ipd.sym.xtal.xtal("L632")
    t.checkpoint("create_init")
    x = ipd.sym.xtal.xtal("L632")
    x = ipd.sym.xtal.xtal("L632")
    x = ipd.sym.xtal.xtal("L632")
    t.checkpoint("create")

    t.report()

@pytest.mark.fast
def test_xtalrad_I213():
    x = ipd.sym.Xtal("I213")
    # ipd.showme(x, scale=100)
    symcoord = x.symcoords([0, 0, 0], cellsize=10, cells=3, xtalrad=0.01, ontop=None)
    assert len(symcoord) == 0

    symcoord = x.symcoords([0, 0, 0], cellsize=10, cells=3, xtalrad=0.7, ontop=None)
    assert len(symcoord) == 33

    pt = [2, 2, 2]
    symcoord = x.symcoords(pt, cellsize=10, cells=3, xtalrad=0.7, center=pt, ontop=None)
    ic(symcoord.shape)  # type: ignore
    # ipd.showme(symcoord, kind='point')
    # assert 0

def _analize_xtal_asu_placement(sym="I_4_3_2_432", showme=False):
    x = ipd.sym.Xtal(sym)
    ic(x.asucen(method="closest_approach"))  # type: ignore
    ic(x.asucen(method="closest_to_cen"))  # type: ignore
    ic(x.asucen(method="closest_to_cen", use_olig_nbrs=True))  # type: ignore
    # ic(x.asucen(method='stored'))
    # assert 0
    ipd.sym.analyze_xtal_asu_placement(sym)

def helper_test_coords_to_asucen_0():
    xtal = ipd.sym.Xtal(sym)  # type: ignore
    cen0 = xtal.asucen()

    ic(cen0)  # type: ignore
    frames = xtal.frames()
    for i, f in enumerate(frames):
        cen1 = ipd.homog.hxform(f, cen0)
        deltalen = 1 / random.randint(1, 100)
        delta = ipd.homog.hrandunit() * deltalen
        cen1 += delta
        cen2 = xtal.coords_to_asucen(cen1).reshape(4)
        assert np.allclose(cen0, cen2, atol=deltalen + 0.0001)

@pytest.mark.fast
def test_dump_pdb():
    sym = "I213"
    xtal = ipd.sym.Xtal(sym)
    csize = 150
    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)
    asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
    xyz += ipd.homog.hvec(asucen)
    # xyz[:, 1] -= 2
    with tempfile.TemporaryDirectory() as td:
        xtal.dump_pdb(f"{td}/test.pdb", xyz, cellsize=csize, cells=(-1, 0), xtalrad=0.5, allowcellshift=True)

@pytest.mark.fast
def test_asucen(headless=True):
    return
    csize = 62.144
    xtal = ipd.sym.Xtal("P 21 3")
    asucen = xtal.asucen(cellsize=csize, xtalasumethod="closest_approach")
    cellpts = xtal.symcoords(asucen, cellsize=csize, flat=True, allowcellshift=True)
    frames = xtal.primary_frames(csize)
    ipd.showme(xtal, scale=csize, headless=headless)
    ipd.showme(asucen, sphere=4, headless=headless)
    ipd.showme(cellpts, sphere=4, headless=headless)

@pytest.mark.fast
def test_xtal_L6m322(headless=True):
    xtal = ipd.sym.Xtal("L6m322")

    # ic(xtal.symelems)
    # ic(xtal.genframes.shape)
    # ic(len(xtal.coverelems))
    # ic(len(xtal.coverelems[0]))
    # ic(len(xtal.coverelems[1]))
    ipd.showme(xtal.genframes, scale=3, headless=headless)
    # ipd.showme(xtal.unitframes, name='arstn', scale=3)
    ipd.showme(xtal, headless=headless, showgenframes=False, symelemscale=1, pointscale=0.8, vizfresh=True)

@pytest.mark.fast
def test_xtal_L6_32(headless=False):
    xtal = ipd.sym.Xtal("L6_32")
    # ic(xtal.symelems)vizfresh
    # ic(xtal.genframes.shape)
    # ic(len(xtal.coverelems))
    # ic(len(xtal.coverelems[0]))
    # ic(len(xtal.coverelems[1]))
    ipd.showme(xtal.genframes, scale=3, headless=headless)
    # ipd.showme(xtal.unitframes, name='arstn')
    # ipd.showme(xtal, headless=False, showgenframes=False, symelemscale=1, pointscale=0.8, vizfresh=True)

@pytest.mark.fast
def test_symelem_mobile():
    assert 0 == ipd.sym.xtal.SymElem(2, [1, 0, 0], [0, 0, 0]).mobile
    assert 0 == ipd.sym.xtal.SymElem(2, [1, 0, 0], [1, 0, 0]).mobile
    assert 0 == ipd.sym.xtal.SymElem(2, [1, 0, 0], [0, 0, 0], [0, 1, 0]).mobile
    assert 1 == ipd.sym.xtal.SymElem(2, [1, 0, 0], [0, 1, 0], [0, 1, 0]).mobile
    assert 1 == ipd.sym.xtal.SymElem(2, [1, 0, 0], [0, 10, 0]).mobile
    assert 0 == ipd.sym.xtal.SymElem(2, [1, 1, 0], [10, 10, 0]).mobile
    assert 1 == ipd.sym.xtal.SymElem(2, [1, 1, 0], [10, 10, 0.001]).mobile

@pytest.mark.fast
def test_symelem(headless=True):
    elem1 = ipd.sym.xtal.SymElem(2, [1, 0, 0], [0, 0, 0])
    elem2 = ipd.sym.xtal.SymElem(2, [1, 0, 0], [0, 10, 0])

    x = ipd.homog.hrandsmall()
    e2 = ipd.homog.hxform(x, elem1, strict=False)
    # ic(x)
    # ic(elem1.coords)
    # ic(elem2.coords)
    # ic(e2.coords)
    # ic(ipd.homog.hxform(x, elem1.coords))
    assert np.allclose(e2.axis, ipd.homog.hxform(x, elem1.axis))
    # assert np.allclose(e2.cen, ipd.homog.hxform(x, elem1.cen))

    # x = ipd.homog.hrandsmall()
    # e2 = ipd.homog.hxform(x, elem1)
    # assert np.allclose(e2.coords, ipd.homog.hxform(x, elem1.coords))
    # ipd.showme(elem1, headless=headless)
    # ipd.showme(ipd.homog.hxform(ipd.homog.hrot([0, 1, 0]s, 120, [0, 0, 1]), elem1), headless=headless)
    # ipd.showme([elem1], fancover=0.8)

@pytest.mark.fast
def test_xtal_cellframes():
    xtal = ipd.sym.Xtal("P 2 3")
    assert xtal.nsub == 12
    assert len(xtal.cellframes(cellsize=1, cells=1)) == xtal.nsub
    assert len(xtal.cellframes(cellsize=1, cells=None)) == 1
    assert len(xtal.cellframes(cellsize=1, cells=2)) == 8 * xtal.nsub
    assert len(xtal.cellframes(cellsize=1, cells=3)) == 27 * xtal.nsub
    assert len(xtal.cellframes(cellsize=1, cells=4)) == 64 * xtal.nsub
    assert len(xtal.cellframes(cellsize=1, cells=5)) == 125 * xtal.nsub

def helper_test_xtal_cellframes(xtalname):
    xtal = ipd.sym.Xtal(xtalname)
    frames = xtal.cellframes(cells=3, ontop=None)
    unitframes = xtal.cellframes(allowcellshift=True, ontop=None)
    assert len(unitframes) == xtal.nsub

def prune_bbox(coords, lb, ub):
    inboundslow = np.all(coords >= lb - 0.001, axis=-1)
    inboundshigh = np.all(coords <= ub + 0.001, axis=-1)
    inbounds = np.logical_and(inboundslow, inboundshigh)
    return inbounds

def helper_test_xtal_cryst1(xtalname, dump_pdbs=False):
    pymol = pytest.importorskip("pymol")
    xtal = ipd.sym.Xtal(xtalname)

    cellsize = 99.12345
    crd = cellsize * np.array([
        [0.28, 0.13, 0.13],
        [0.28, 0.16, 0.13],
        [0.28, 0.13, 0.15],
    ])

    if dump_pdbs:
        xtal.dump_pdb(f'test_{xtalname.replace(" ","_")}_1.pdb', crd, cellsize=cellsize, cells=1)
        xtal.dump_pdb(f'test_{xtalname.replace(" ","_")}_2.pdb', crd, cellsize=cellsize, cells=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        pymol.cmd.delete("all")
        fname = f"{tmpdir}/test.pdb"
        xtal.dump_pdb(fname, crd, cellsize=cellsize)
        pymol.cmd.load(fname)
        pymol.cmd.symexp("pref", "test", "all", 9e9)
        coords1 = pymol.cmd.get_coords()
        pymol.cmd.delete("all")
        coords2 = xtal.symcoords(crd, cellsize=cellsize, cells=(-2, 1), flat=True, ontop=None)
        assert len(coords1) == 27 * 3 * xtal.nsub
        assert len(coords2) == 64 * 3 * xtal.nsub

    if True:
        coords1 = coords1.round().astype("i")
        coords2 = coords2.round().astype("i")[..., :3]
        s1 = set([tuple(x) for x in coords1])
        s2 = set([tuple(x) for x in coords2])
        # ic(xtalname, len(s1), len(coords1))
        # ic(xtalname, len(s2), len(coords2))
        # ic(xtalname, len(s2 - s1), (64 - 27) * 3 * xtal.nsub)
        expected_ratio = (4**3 - 3**3) / 3**3
        assert len(s2 - s1) == (64-27) * 3 * xtal.nsub
        assert len(s1 - s2) == 0, f"canonical frames mismatch {xtalname}"
        assert len(s1.intersection(s2)) == len(coords1), f"canonical frames mismatch {xtalname}"

    lb = -105
    ub = 155
    coords1 = coords1[coords1[:, 0] < ub]
    coords1 = coords1[coords1[:, 0] > lb]
    coords1 = coords1[coords1[:, 1] < ub]
    coords1 = coords1[coords1[:, 1] > lb]
    coords1 = coords1[coords1[:, 2] < ub]
    coords1 = coords1[coords1[:, 2] > lb]
    coords2 = coords2[coords2[:, 0] < ub]
    coords2 = coords2[coords2[:, 0] > lb]
    coords2 = coords2[coords2[:, 1] < ub]
    coords2 = coords2[coords2[:, 1] > lb]
    coords2 = coords2[coords2[:, 2] < ub]
    coords2 = coords2[coords2[:, 2] > lb]

    if dump_pdbs:
        ipd.pdb.dump_pdb_from_points(f'test_{xtalname.replace(" ","_")}_pymol.pdb', coords1)
        ipd.pdb.dump_pdb_from_points(f'test_{xtalname.replace(" ","_")}_wxtal.pdb', coords2)

    # ic(xtalname)
    # ic(coords1.shape)
    # ic(coords2.shape)
    coords1 = coords1.round().astype("i")
    coords2 = coords2.round().astype("i")[..., :3]
    s1 = set([tuple(x) for x in coords1])
    s2 = set([tuple(x) for x in coords2])
    assert s1 == s2

@pytest.mark.fast
def test_hxtal_viz(headless=True, xtalname="P 2 3", symelemscale=0.7, cellsize=10, **kw):
    pymol = pytest.importorskip("pymol")
    xtal = ipd.sym.Xtal(xtalname)
    # ic(xtal.unitframes.shape)
    cen = xtal.asucen(cellsize=cellsize, xtalasumethod="closest_to_cen")
    # ic(cellsize, cen)
    # assert 0
    # ipd.showme(xtal.symelems, scale=cellsize)
    # ipd.showme(cen, vizsphereradius=1)

    ipd.showme(
        xtal,
        headless=headless,
        showgenframes=False,
        symelemscale=symelemscale,
        pointscale=0.8,
        scale=cellsize,
        showpoints=cen[None],
        pointradius=0.3,
        # vizfresh=True,
        **kw,
    )
    # sys.path.append('/home/sheffler/src/wills_pymol_crap')
    # pymol.cmd.do('@/home/sheffler/.pymolrc')
    # pymol.cmd.do('run /home/sheffler/pymol3/misc/G222.py; gyroid(10,r=8,c=Vec(5,5,5)); set light, [ -0.3, -0.30, 0.8 ]')
    # elem1 = ipd.sym.xtal.SymElem(2, [1, 0, 0], [0, 0.25, 0.0])
    # elem2 = ipd.sym.xtal.SymElem(3, [1, 1, -1], [0, 0, 0])
    # xtal = ipd.sym.Xtal([elem1, elem2])
    # for a, b, c in itertools.product(*[(0, 1)] * 3):
    #    # ipd.showme(xtal, cellshift=[a, b, c], showgenframes=a == b == c == 0)
    #    ipd.showme(xtal, cellshift=[a, b, c], headless=headless)

def _test_hxtal_viz_gyroid(headless=True):
    # elem1 = ipd.sym.xtal.SymElem(2, [1, 0, 0], [0, 0.25, 0.0])
    # elem2 = ipd.sym.xtal.SymElem(3, [1, 1, 1], [0, 0, 0])
    # xtal = ipd.sym.Xtal([elem1, elem2])

    # xtal = ipd.sym.Xtal('I 21 3')

    # ipd.showme(xtal, headless=headless, fanshift=[-0.03, 0.05], fansize=[0.15, 0.12])
    # ipd.showme(xtal, headless=headless, fanshift=[-0.03, 0.05], fansize=[0.15, 0.12])
    ipd.showme(xtal, headless=headless, showpoints=1)  # type: ignore
    # for a, b, c in itertools.product(*[(0, 1)] * 3):
    #   # ipd.showme(xtal, cellshift=[a, b, c], showgenframes=a == b == c == 0)
    #   ipd.showme(xtal, cellshift=[a, b, c], headless=headless, fanshift=[-0.03, 0.05],
    #             fansize=[0.15, 0.12])

@pytest.mark.fast
def test_xtal_cellframes_P_4_3_2_432(*a, **kw):
    helper_test_xtal_cellframes("P 4 3 2 432", *a, **kw)

@pytest.mark.fast
def test_xtal_cryst1_P_4_3_2_432(*a, **kw):
    helper_test_xtal_cryst1("P 4 3 2 432", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_P_4_3_2_422(*a, **kw):
    helper_test_xtal_cellframes("P 4 3 2 422", *a, **kw)

@pytest.mark.fast
def test_xtal_cryst1_P_4_3_2_422(*a, **kw):
    helper_test_xtal_cryst1("P 4 3 2 422", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_P_4_3_2(*a, **kw):
    helper_test_xtal_cellframes("P 4 3 2", *a, **kw)

@pytest.mark.fast
def test_xtal_cryst1_P_4_3_2(*a, **kw):
    helper_test_xtal_cryst1("P 4 3 2", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_P_4_3_2_443(*a, **kw):
    helper_test_xtal_cellframes("P 4 3 2 443", *a, **kw)

@pytest.mark.fast
def test_xtal_cryst1_P_4_3_2_443(*a, **kw):
    helper_test_xtal_cryst1("P 4 3 2 443", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_P_4_3_2_43(*a, **kw):
    helper_test_xtal_cellframes("P 4 3 2 43", *a, **kw)

@pytest.mark.fast
def test_xtal_cryst1_P_4_3_2_43(*a, **kw):
    helper_test_xtal_cryst1("P 4 3 2 43", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_F_4_3_2(*a, **kw):
    helper_test_xtal_cellframes("F 4 3 2", *a, **kw)

@pytest.mark.skip
def test_xtal_cryst1_F_4_3_2(*a, **kw):
    helper_test_xtal_cryst1("F 4 3 2", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_I_4_3_2(*a, **kw):
    helper_test_xtal_cellframes("I 4 3 2", *a, **kw)

@pytest.mark.fast
def test_xtal_cryst1_I_4_3_2(*a, **kw):
    helper_test_xtal_cryst1("I 4 3 2", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_P_2_3(*a, **kw):
    helper_test_xtal_cellframes("P 2 3", *a, **kw)

@pytest.mark.fast
def test_xtal_cryst1_P_2_3(*a, **kw):
    helper_test_xtal_cryst1("P 2 3", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_P_21_3(*a, **kw):
    helper_test_xtal_cellframes("P 21 3", *a, **kw)

@pytest.mark.fast
def test_xtal_cryst1_P_21_3(*a, **kw):
    helper_test_xtal_cryst1("P 21 3", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_I_21_3(*a, **kw):
    helper_test_xtal_cellframes("I 21 3", *a, **kw)

@pytest.mark.fast
def test_xtal_cryst1_I_21_3(*a, **kw):
    helper_test_xtal_cryst1("I 21 3", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_P_41_3_2(*a, **kw):
    helper_test_xtal_cellframes("P 41 3 2", *a, **kw)

@pytest.mark.fast
def test_xtal_cryst1_P_41_3_2(*a, **kw):
    helper_test_xtal_cryst1("P 41 3 2", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_I_41_3_2(*a, **kw):
    helper_test_xtal_cellframes("I 41 3 2", *a, **kw)

@pytest.mark.fast
def test_xtal_cryst1_I_41_3_2(*a, **kw):
    helper_test_xtal_cryst1("I 41 3 2", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_I4132_322(*a, **kw):
    helper_test_xtal_cellframes("I4132_322", *a, **kw)

@pytest.mark.fast
def test_xtal_cryst1_I4132_322(*a, **kw):
    helper_test_xtal_cryst1("I4132_322", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_L6_32(*a, **kw):
    helper_test_xtal_cellframes("L6_32", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_L6M_322(*a, **kw):
    helper_test_xtal_cellframes("L6M_322", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_L4_44(*a, **kw):
    helper_test_xtal_cellframes("L4_44", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_L4_42(*a, **kw):
    helper_test_xtal_cellframes("L4_42", *a, **kw)

@pytest.mark.fast
def test_xtal_cellframes_L3_33(*a, **kw):
    helper_test_xtal_cellframes("L3_33", *a, **kw)

if __name__ == "__main__":
    main()
