import numpy as np
import pytest
from icecream import ic
from opt_einsum import contract as einsum

import ipd

def main():
    # _test_spacegroup_symelems()

    test_spacegroup_frames_sanity_check()  # type: ignore
    test_spacegroup_lattice_transpose_bug()

    # test_subsym()
    test_spacegroup_frames_order()
    # test_spacegroup_frames_P3()
    test_lattice_cellgeom()

    test_spacegroup_frames_tounitcell("P1", [15, 25, 35, 75, 85, 95], 4)
    test_spacegroup_frames_tounitcell("P-1", [15, 25, 35, 75, 85, 95], 4)
    test_spacegroup_frames_tounitcell("P121", [15, 25, 35, 90, 85, 90], 4)
    test_spacegroup_frames_tounitcell("I213", [10], 2)
    test_spacegroup_frames_tounitcell("P212121", [10, 11, 12], 4)
    test_spacegroup_frames_tounitcell("P43", [10, 10, 12], 4)
    test_spacegroup_frames_tounitcell("P3", [11, 11, 13, 90, 90, 120], 2)

    test_spacegroup_frames_P1()

    test_lattice_vectors()
    test_two_iface()
    test_spacegroup_frames_P213()
    test_spacegroup_frames_P23()
    test_spacegroup_frames_I213()
    test_spacegroup_frames_P4132()
    test_spacegroup_frames_I4132()
    test_spacegroup_frames_P432()
    test_spacegroup_frames_F432()
    test_spacegroup_frames_I432()
    ic("PASS test_spacegroup")

@pytest.mark.fast
def test_spacegroup_identify_chiral():
    assert len(ipd.sym.xtal.sg_all_chiral) == 65

@pytest.mark.fast
@pytest.mark.parametrize('sym', ipd.sym.xtal.sg_all_chiral)
def test_spacegroup_frames_sanity_check(sym):
    f = ipd.sym.xtal.sgframes(sym, cells=3, cellgeom="nonsingular")
    if not ipd.homog.hvalid(f):
        assert not ipd.sym.xtal.sg_is_chiral(sym)

@pytest.mark.fast
def test_spacegroup_lattice_transpose_bug():
    sym = "P3"
    lat = ipd.sym.xtal.lattice_vectors(sym, cellgeom="nonsingular")
    linv = np.linalg.inv(lat)
    f = ipd.sym.xtal.sg_frames_dict[sym]
    f2 = einsum("ij,fjk,kl->fil", lat, f[:, :3, :3], linv)
    ax = ipd.homog.axis_of(ipd.homog.hconstruct(f2))
    assert np.allclose([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0]], ax)
    assert np.allclose([0, np.pi * 2 / 3, np.pi * 2 / 3], ipd.homog.angle_of(f2))

    p = ipd.homog.hrandpoint(10, [10, 0, 0])

    q = einsum("ij,fjk,kl,pl->fpi", lat, f[:, :3, :3], linv, p[:, :3])
    q = ipd.homog.hpoint(q)
    # q2 = einsum('ij,fjk,kl->fij', lat, f[:, :3, :3], linv)
    # q2 = einsum('fij')
    r = ipd.homog.hxformpts(ipd.homog.hrot([0, 0, 1], [0, 120, 240]), p)
    assert np.allclose(q, r)

@pytest.mark.fast
def test_spacegroup_frames_order():
    f1 = ipd.sym.xtal.sgframes("I213", cells=1, cellgeom="unit")
    f2 = ipd.sym.xtal.sgframes("I213", cells=2, cellgeom="unit")
    f3 = ipd.sym.xtal.sgframes("I213", cells=3, cellgeom="unit")
    f4 = ipd.sym.xtal.sgframes("I213", cells=4, cellgeom="unit")
    f5 = ipd.sym.xtal.sgframes("I213", cells=5, cellgeom="unit")
    assert np.allclose(f1, f2[:len(f1)])
    assert np.allclose(f2, f3[:len(f2)])
    assert np.allclose(f3, f4[:len(f3)])
    assert np.allclose(f4, f5[:len(f4)])

@pytest.mark.xfail
def test_spacegroup_frames_P3():
    f = ipd.sym.xtal.sgframes("P3")
    ic(f @ ipd.homog.htrans([0.1, 0.2, 0.3]))
    # ipd.showme(f @ ipd.homog.htrans([0.1, 0.2, 0.3]))
    latticevec = np.array([
        [1.00000000e00, 0.00000000e00, 0.00000000e00],
        [-5.00000000e-01, 8.66025404e-01, 0.00000000e00],
        [6.12323400e-17, 1.06057524e-16, 1.00000000e00],
    ])
    ic(latticevec)
    fname = ipd.dev.package_testdata_path("pdb/i213fittest.pdb")
    coords = ipd.pdb.readpdb(fname).subset(chain="A").ca()
    ipd.showme(coords)
    assert 0

def _test_spacegroup_symelems():
    # for se in ipd.sym.xtal.symelems('I213', psym=3):
    # print(se)
    # for se in ipd.sym.xtal.symelems('I213', psym=2):
    # print(se)
    for se in ipd.sym.xtal.symelems("P3", psym=3):
        print(se)
    ipd.showme(ipd.sym.xtal.sgframes("P3", cells=1) @ ipd.homog.htrans([0.1, 0.2, 0.3]))
    assert 0

@pytest.mark.fast
def test_lattice_cellgeom():
    for i in range(100):
        a, b, c = np.random.rand(3) * 100
        A, B, C = np.random.rand(3) * 60 + 60
        # if np.random.rand() < 0.5:
        # A, B, C = 180 - A, 180 - B, 180 - C
        g = [a, b, c, A, B, C]
        l = ipd.sym.xtal.lattice_vectors("TRICLINIC", g)
        # ic(l)
        assert not np.any(np.isnan(l))
        h = ipd.sym.xtal.cellgeom_from_lattice(l)

        # ic(g)
        # ic(h)
        assert np.allclose(g, h)

def helper_test_spacegroup_frames(sg):
    f1 = ipd.sym.xtal.sgframes(sg, [1, 1, 1])
    f2 = ipd.sym.frames(sg, cells=1, ontop=None, cellgeom="nonsingular")
    assert f1.shape == f2.shape
    f2[np.where(np.abs(f2) < 0.0001)] = 0
    f2[f2[:, 0, 3] > 0.999, 0, 3] = 0  # mod1
    f2[f2[:, 1, 3] > 0.999, 1, 3] = 0  # mod1
    f2[f2[:, 2, 3] > 0.999, 2, 3] = 0  # mod1
    for i, x in enumerate(f1):
        match = None
        for j, y in enumerate(f2):
            if np.allclose(x, y, atol=1e-4):
                assert match is None
                match = j
        assert match is not None
    # print(f'pass {sg}')

@pytest.mark.fast
def test_lattice_vectors():
    print(
        """
      15.737   15.887   25.156  85.93  86.19  69.86 P 1
      15.737000,   0.000000,   0.000000
       5.470136,  14.915575,   0.000000
       1.671566,   1.288705,  25.067297
   """,
        flush=True,
    )
    lvec = ipd.sym.xtal.spacegroup.lattice_vectors("P1", [15.737, 15.887, 25.156, 85.93, 86.19, 69.86])
    foo = np.array([[15.737, 0.0, 0.0], [5.47013594, 14.91557514, 0.0], [1.67156711, 1.28870451, 25.06729822]])
    assert np.allclose(lvec, foo.T)

@pytest.mark.fast
def test_spacegroup_frames_P1():
    f = ipd.sym.xtal.spacegroup.sgframes("P1", [15.737, 15.887, 25.156, 85.93, 86.19, 69.86], cells=3)
    ref = np.array([
        [0.0, 0.0, 0.0],
        [1.67156711, 1.28870451, 25.06729822],
        [5.47013594, 14.91557514, 0.0],
        [7.14170305, 16.20427966, 25.06729822],
        [15.737, 0.0, 0.0],
        [17.40856711, 1.28870451, 25.06729822],
        [21.20713594, 14.91557514, 0.0],
        [22.87870305, 16.20427966, 25.06729822],
        [-22.87870305, -16.20427966, -25.06729822],
        [-21.20713594, -14.91557514, 0.0],
        [-19.53556883, -13.62687063, 25.06729822],
        [-17.40856711, -1.28870451, -25.06729822],
        [-15.737, 0.0, 0.0],
        [-14.06543289, 1.28870451, 25.06729822],
        [-11.93843117, 13.62687063, -25.06729822],
        [-10.26686406, 14.91557514, 0.0],
        [-8.59529695, 16.20427966, 25.06729822],
        [-7.14170305, -16.20427966, -25.06729822],
        [-5.47013594, -14.91557514, 0.0],
        [-3.79856883, -13.62687063, 25.06729822],
        [-1.67156711, -1.28870451, -25.06729822],
        [3.79856883, 13.62687063, -25.06729822],
        [8.59529695, -16.20427966, -25.06729822],
        [10.26686406, -14.91557514, 0.0],
        [11.93843117, -13.62687063, 25.06729822],
        [14.06543289, -1.28870451, -25.06729822],
        [19.53556883, 13.62687063, -25.06729822],
    ])
    assert np.allclose(f[:, :3, 3], ref)
    # ipd.showme(f)

@pytest.mark.parametrize(
    "sgroup,cellgeom,ncell",
    [
        ("P1", [15, 25, 35, 75, 85, 95], 4),
        ("P-1", [15, 25, 35, 75, 85, 95], 4),
        ("P121", [15, 25, 35, 90, 85, 90], 4),
        ("I213", [10], 2),
        ("P212121", [10, 11, 12], 4),
        ("P43", [10, 10, 12], 4),
        ("P3", [11, 11, 13, 90, 90, 120], [1, 2, 1]),
    ],
)
def test_spacegroup_frames_tounitcell(sgroup, cellgeom, ncell):
    fcell = ipd.sym.xtal.spacegroup.sgframes(sgroup, cellgeom, cells=ncell)
    # ic(fcell.shape)
    funit = ipd.sym.xtal.spacegroup.sgframes(sgroup, cellgeom="unit", cells=ncell)
    # ic(funit.shape)
    lattice = ipd.sym.xtal.lattice_vectors(sgroup, cellgeom)
    ftest2 = ipd.sym.xtal.spacegroup.applylattice(lattice, funit)
    assert np.allclose(fcell, ftest2)

    ftest = ipd.sym.xtal.spacegroup.tounitcell(lattice, fcell)
    # ic(ftest.shape)
    # ic(funit)
    # ic(ftest)
    assert np.allclose(funit[:, :3, :3], ftest[:, :3, :3])
    assert np.allclose(funit[:, :3, 3], ftest[:, :3, 3])

@pytest.mark.fast
def test_two_iface():
    assert len(ipd.sym.xtal.spacegroup_data.two_iface_spacegroups) == 31
    for sg in ipd.sym.xtal.spacegroup_data.two_iface_spacegroups:
        if sg not in ipd.sym.xtal.spacegroup_data.sg_lattice:
            assert sg == "R3"

@pytest.mark.fast
def test_spacegroup_frames_I213():
    helper_test_spacegroup_frames("I213")

@pytest.mark.fast
def test_spacegroup_frames_P213():
    helper_test_spacegroup_frames("P213")

@pytest.mark.fast
def test_spacegroup_frames_P4132():
    helper_test_spacegroup_frames("P4132")

@pytest.mark.fast
def test_spacegroup_frames_I4132():
    helper_test_spacegroup_frames("I4132")

@pytest.mark.fast
def test_spacegroup_frames_P432():
    helper_test_spacegroup_frames("P432")

@pytest.mark.fast
def test_spacegroup_frames_F432():
    helper_test_spacegroup_frames("F432")

@pytest.mark.fast
def test_spacegroup_frames_I432():
    helper_test_spacegroup_frames("I432")

@pytest.mark.fast
def test_spacegroup_frames_P23():
    helper_test_spacegroup_frames("P23")

def ___test_spacegroup_frames_I4132():
    f1 = ipd.sym.xtal.sgframes("I4132", [1, 1, 1])
    f2 = ipd.sym.frames("I4132_322", cells=1, ontop=None)
    f2[np.where(np.abs(f2) < 0.0001)] = 0
    f2[f2[:, 0, 3] > 0.999, 0, 3] = 0
    f2[f2[:, 1, 3] > 0.999, 1, 3] = 0
    f2[f2[:, 2, 3] > 0.999, 2, 3] = 0

    # for x in f2:
    # print(list(x[:3, :3].T.reshape(9).round(6).astype('i')) + list(x[:3, 3].round(6)))
    # assert 0

    # f1[:, :3, 3] *= 10
    # f2[:, :3, 3] *= 10
    # ipd.showme(f1)
    # ipd.showme(f2)
    # assert 0
    for i, x in enumerate(f1[:24]):
        match = None
        for j, y in enumerate(f2):
            # if i == 25 and np.isclose(y[0, 1], -1):
            # if i == 25 and np.allclose(x[:3, :3], y[:3, :3]):
            # ic(i, j, x, y)
            if np.allclose(x, y, atol=1e-4):
                assert match is None
                match = j
        # if match is None:
        # ic(i)
        # for j, y in enumerate(f2):
        # if np.allclose(x[:3, :3], y[:3, :3]):
        # ic(x, y)
        assert match is not None
    # ic('pass')

if __name__ == "__main__":
    main()
