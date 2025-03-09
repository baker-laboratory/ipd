import numpy as np
import pytest

import ipd
from ipd.sym.asugrid import vispoints  # type: ignore

pytest.skip(allow_module_level=True)

def main():
    test_asugrid_case1()
    assert 0
    test_asugrid_I213_offposition2()

    assert 0
    test_asugrid_I213_offposition()
    test_asugrid_I213()

    test_asugrid_P432_432D2()
    test_asugrid_P432_432()
    test_asugrid_P432_422()

    test_asugrid_P213()

    test_asugrid_I4132()

    test_asugrid_L632()

@pytest.mark.fast
def test_asugrid_case1():
    # yapf: disable
    kw = {}
    newpos, newcell = ipd.sym.place_asu_grid(
      pos=np.array([13.02651048,8.16560037,5.64629888,1.]),
      cellsize=58.004592702631456,
      frames=np.array([[[1,0,0,0.],[0,1,0,0.],[0,0,1,0.],[0,0,0,1.]],[[0,-0,1,0.],[1,0,-0,0.],[-0,1,0,0.],[0,0,0,1.]],[[-0,1,0,0.],[0,-0,1,0.],[1,0,-0,0.],[0,0,0,1.]],[[-1,-0,0,1.],[0,-1,0,0.5],[0,0,1,0.],[0,0,0,1.]]]),
      framesavoid=np.array([[[1,0,0,-0.5],[-0,1,0,-0.5],[-0,-0,1,-0.5],[0,0,0,1.]],[[0,1,0,-0.5],[-0,-0,1,-0.5],[1,-0,0,-0.5],[0,0,0,1.]],[[0,-0,1,-0.5],[1,0,-0,-0.5],[-0,1,0,-0.5],[0,0,0,1.]],[[0,-0,-1,0.],[1,-0,0,-0.5],[-0,-1,0,0.5],[0,0,0,1.]],[[0,-1,0,-0.],[-0,0,1,-0.5],[-1,-0,-0,0.5],[0,0,0,1.]],[[0,0,1,-0.5],[-1,0,0,0.5],[-0,-1,0,0.],[0,0,0,1.]],[[1,-0,0,-0.5],[-0,-1,0,0.5],[0,-0,-1,0.],[0,0,0,1.]],[[0,-1,0,-0.],[0,-0,-1,0.5],[1,0,0,-0.],[0,0,0,1.]],[[0,0,1,-0.5],[-1,0,0,0.5],[-0,-1,0,1.],[0,0,0,1.]],[[0,1,0,-0.5],[-0,0,-1,0.5],[-1,0,0,1.],[0,0,0,1.]],[[0,0,-1,0.],[-1,0,-0,0.5],[0,1,0,-0.],[0,0,0,1.]],[[-1,-0,0,0.],[0,-1,0,0.5],[0,0,1,-0.],[0,0,0,1.]],[[1,-0,0,-0.5],[-0,-1,0,0.5],[0,-0,-1,1.],[0,0,0,1.]],[[-1,-0,-0,0.5],[0,-1,-0,-0.],[-0,-0,1,-0.5],[0,0,0,1.]],[[-0,-1,0,0.5],[-0,-0,-1,-0.],[1,-0,-0,-0.5],[0,0,0,1.]],[[0,-0,-1,0.5],[-1,0,-0,0.],[0,1,-0,-0.5],[0,0,0,1.]],[[1,0,-0,-0.],[0,-1,0,0.],[-0,-0,-1,0.5],[0,0,0,1.]],[[0,-0,-1,1.],[1,-0,0,-0.5],[-0,-1,0,0.5],[0,0,0,1.]],[[-0,-0,1,0.],[-1,0,-0,-0.],[-0,-1,-0,0.5],[0,0,0,1.]],[[0,-1,0,1.],[-0,0,1,-0.5],[-1,-0,-0,0.5],[0,0,0,1.]],[[-0,1,-0,-0.],[0,-0,-1,0.],[-1,-0,-0,0.5],[0,0,0,1.]],[[-1,-0,-0,1.],[-0,1,-0,-0.5],[0,-0,-1,0.5],[0,0,0,1.]],[[-1,-0,-0,0.5],[0,-1,-0,1.],[-0,-0,1,-0.5],[0,0,0,1.]],[[0,-1,0,0.5],[-0,0,1,-0.],[-1,-0,-0,0.],[0,0,0,1.]],[[-0,-0,-1,0.5],[1,-0,-0,0.],[-0,-1,0,0.],[0,0,0,1.]],[[-1,0,0,0.5],[0,1,-0,0.],[-0,-0,-1,0.],[0,0,0,1.]],[[-0,-1,0,0.5],[-0,-0,-1,1.],[1,-0,-0,-0.5],[0,0,0,1.]],[[0,-0,-1,0.5],[-1,0,-0,1.],[0,1,-0,-0.5],[0,0,0,1.]],[[1,0,-0,-0.],[0,-1,0,1.],[-0,-0,-1,0.5],[0,0,0,1.]],[[0,-1,0,0.5],[-0,0,1,-0.],[-1,-0,-0,1.],[0,0,0,1.]],[[0,-1,0,1.],[0,-0,-1,0.5],[1,0,0,-0.],[0,0,0,1.]],[[-0,-0,-1,0.5],[1,-0,-0,0.],[-0,-1,0,1.],[0,0,0,1.]],[[-1,0,0,0.5],[0,1,-0,0.],[-0,-0,-1,1.],[0,0,0,1.]],[[-0,-0,1,0.],[-1,0,-0,1.],[-0,-1,-0,0.5],[0,0,0,1.]],[[0,0,-1,1.],[-1,0,-0,0.5],[0,1,0,-0.],[0,0,0,1.]],[[-0,1,-0,-0.],[0,-0,-1,1.],[-1,-0,-0,0.5],[0,0,0,1.]]]),
      lbub=0.15,
      lbubcell=0.2,
      nsamp=25,
      nsampcell=40,
      distcontact=[14.707434199761364, 16.707434199761366],
      distavoid=22.707434199761366,
      distspread=9000000000.0,
      clusterdist=4,
      refpos=None,
      refcell=None,
    )
    # yapf: enable

    frames = ipd.hscaled(newcell[0], frames)  # type: ignore
    sympos = ipd.homog.hxformpts(frames, newpos[0])
    ic(sympos)  # type: ignore
    dist = ipd.homog.hnorm(sympos[:, None] - sympos[None])[0, 1:]
    ic(dist)  # type: ignore
    assert 0

@pytest.mark.fast
def test_asugrid_I213_offposition():
    sym = "I 21 3"
    x = ipd.sym.xtal.xtal(sym)
    frames = x.primary_frames()
    allframes = x.frames(cells=3, xtalrad=0.7)
    framesavoid = allframes[len(frames):]
    cellsize = 100
    pos = ipd.homog.hpoint([34.468862, 21.753182, 12.16508, 1])
    newpos, newcell = ipd.sym.place_asu_grid_multiscale(
        pos,
        cellsize,
        frames=frames,
        framesavoid=framesavoid,
        lbub=0.2,
        lbubcell=0.1,
        nsamp=20,
        nsampcell=7,
        distcontact=(0.15, 0.17),
        distavoid=0.25,
        distspread=9e9,
        clusterdist=0.05,
    )
    # ic(pos)
    # ic(newpos[0])
    # ipd.sym.asugrid.vispoints(newpos, newcell, frames, allframes)

@pytest.mark.fast
def test_asugrid_I213_offposition2():
    sym = "I 21 3"
    x = ipd.sym.xtal.xtal(sym)
    frames = x.primary_frames()
    allframes = x.frames(cells=3, xtalrad=0.7)
    framesavoid = allframes[len(frames):]

    pos = np.array([79.52798574, 28.70174261, 17.94670397, 1.0])
    cellsize = 107.25
    newpos, newcell = ipd.sym.place_asu_grid_multiscale(
        pos,
        cellsize,
        frames=frames,
        framesavoid=framesavoid,
        lbub=0.2,
        lbubcell=0.1,
        nsamp=20,
        nsampcell=7,
        distcontact=(18, 22),
        distavoid=25,
        distspread=9e9,
        clusterdist=5,
    )
    assert len(newpos)
    # ic(pos)
    ic(newpos[0], newcell[0])  # type: ignore
    ipd.sym.asugrid.vispoints(newpos, newcell, frames, allframes)

@pytest.mark.fast
def test_asugrid_I213():
    sym = "I 21 3"
    x = ipd.sym.xtal.xtal(sym)
    frames = x.primary_frames()
    allframes = x.frames(cells=3, xtalrad=0.7)
    framesavoid = allframes[len(frames):]
    cellsize = 100
    # pos = ipd.homog.hpoint([30, 20, 20])
    pos = x.asucen(cellsize=cellsize)
    newpos, newcell = ipd.sym.place_asu_grid(
        pos,
        cellsize,
        frames=frames,
        framesavoid=framesavoid,
        lbub=0.2,
        lbubcell=0.1,
        nsamp=20,
        nsampcell=5,
        distcontact=(0.2, 0.24),
        distavoid=0.35,
        # distspread=2,
        clusterdist=0.05,
    )
    print(repr(newpos))
    ref = np.array([
        [43.06842105, 34.64736842, 52.72105263, 1.0],
        [43.06842105, 32.54210526, 50.61578947, 1.0],
        [47.27894737, 36.75263158, 54.82631579, 1.0],
        [47.27894737, 34.64736842, 52.72105263, 1.0],
        [51.48947368, 38.85789474, 56.93157895, 1.0],
        [45.17368421, 32.54210526, 48.51052632, 1.0],
        [49.38421053, 36.75263158, 52.72105263, 1.0],
        [51.48947368, 36.75263158, 54.82631579, 1.0],
        [47.27894737, 32.54210526, 50.61578947, 1.0],
        [53.59473684, 38.85789474, 54.82631579, 1.0],
        [49.38421053, 34.64736842, 50.61578947, 1.0],
        [53.59473684, 36.75263158, 52.72105263, 1.0],
        [51.48947368, 36.75263158, 50.61578947, 1.0],
        [47.27894737, 32.54210526, 46.40526316, 1.0],
        [49.38421053, 32.54210526, 48.51052632, 1.0],
        [55.7, 38.85789474, 52.72105263, 1.0],
        [51.48947368, 34.64736842, 48.51052632, 1.0],
        [53.59473684, 36.75263158, 48.51052632, 1.0],
        [55.7, 38.85789474, 48.51052632, 1.0],
    ])
    vispoints(newpos, newcell, frames, allframes)
    assert np.allclose(ref, newpos, atol=0.01)

@pytest.mark.fast
def test_asugrid_P213():
    sym = "P 21 3"
    x = ipd.sym.xtal.xtal(sym)
    frames = x.primary_frames()
    allframes = x.frames(cells=3, xtalrad=0.6)
    framesavoid = allframes[len(frames):]
    cellsize = 100
    # pos = ipd.homog.hpoint([30, 20, 20])
    pos = x.asucen(cellsize=cellsize)
    newpos, newcell = ipd.sym.place_asu_grid(
        pos,
        cellsize,
        frames=frames,
        framesavoid=framesavoid,
        lbub=0.2,
        lbubcell=0.1,
        nsamp=20,
        nsampcell=5,
        distcontact=(0.1, 0.33),
        distavoid=0.42,
        distspread=3,
        clusterdist=0.09,
    )
    print(repr(newpos))
    ref = np.array([
        [37.63684211, 18.24210526, 42.63157895, 1.0],
        [41.84736842, 20.34736842, 44.73684211, 1.0],
        [46.05789474, 22.45263158, 44.73684211, 1.0],
        [41.84736842, 20.34736842, 40.52631579, 1.0],
        [37.63684211, 16.13684211, 38.42105263, 1.0],
        [46.05789474, 22.45263158, 40.52631579, 1.0],
        [41.84736842, 20.34736842, 36.31578947, 1.0],
        [50.26842105, 24.55789474, 42.63157895, 1.0],
        [33.42631579, 14.03157895, 38.42105263, 1.0],
        [37.63684211, 16.13684211, 34.21052632, 1.0],
        [46.05789474, 22.45263158, 36.31578947, 1.0],
        [50.26842105, 24.55789474, 38.42105263, 1.0],
        [41.84736842, 18.24210526, 32.10526316, 1.0],
        [54.47894737, 28.76842105, 42.63157895, 1.0],
        [33.42631579, 11.92631579, 34.21052632, 1.0],
        [43.95263158, 22.45263158, 32.10526316, 1.0],
        [54.47894737, 28.76842105, 38.42105263, 1.0],
        [37.63684211, 14.03157895, 30.0, 1.0],
        [50.26842105, 24.55789474, 34.21052632, 1.0],
        [52.37368421, 28.76842105, 34.21052632, 1.0],
        [48.16315789, 24.55789474, 30.0, 1.0],
        [50.26842105, 28.76842105, 30.0, 1.0],
        [56.58421053, 32.97894737, 34.21052632, 1.0],
        [54.47894737, 32.97894737, 30.0, 1.0],
    ])
    assert np.allclose(ref, newpos)
    return

@pytest.mark.fast
def test_asugrid_I4132():
    sym = "I4132_322"
    x = ipd.sym.xtal.xtal(sym)
    frames = x.primary_frames()
    allframes = x.frames(cells=3, xtalrad=0.7)
    framesavoid = allframes[len(frames):]
    cellsize = 100
    # pos = ipd.homog.hpoint([30, 20, 20])
    pos = x.asucen(cellsize=cellsize)
    newpos, newcell = ipd.sym.place_asu_grid(
        pos,
        cellsize,
        frames=frames,
        framesavoid=framesavoid,
        lbub=0.2,
        lbubcell=0.1,
        nsamp=30,
        nsampcell=1,
        distcontact=(0.0, 0.3),
        distavoid=0.31,
        distspread=8,
        clusterdist=0.01,
    )
    print(repr(newpos))
    ref = np.array([
        [-7.69576183, 3.52909483, 14.10201183, 1.0],
        [-7.69576183, 3.52909483, 12.72270148, 1.0],
        [-6.31645148, 3.52909483, 12.72270148, 1.0],
        [-6.31645148, 3.52909483, 11.34339114, 1.0],
        [-6.31645148, 2.14978448, 11.34339114, 1.0],
        [-4.93714114, 2.14978448, 12.72270148, 1.0],
        [-4.93714114, 2.14978448, 11.34339114, 1.0],
    ])
    assert np.allclose(newpos, ref)

@pytest.mark.fast
def test_asugrid_L632():
    sym = "L632"
    x = ipd.sym.xtal.xtal(sym)
    frames = x.primary_frames()
    allframes = x.frames(cells=3, xtalrad=0.7)
    framesavoid = allframes[len(frames):]
    cellsize = 100
    # pos = ipd.homog.hpoint([30, 20, 20])
    pos = x.asucen(cellsize=cellsize)
    newpos, newcell = ipd.sym.place_asu_grid(
        pos,
        cellsize,
        frames=frames,
        framesavoid=framesavoid,
        lbub=0.2,
        lbubcell=0.2,
        nsamp=20,
        nsampcell=10,
        distcontact=(0.0, 0.5),
        distavoid=0.6,
        distspread=0.005,
        clusterdist=0.12,
    )
    print(repr(newpos))
    ref = np.array([
        [21.49909241, 1.05263158, 1.05263158, 1.0],
        [21.49909241, -5.26315789, 1.05263158, 1.0],
        [21.49909241, 1.05263158, -5.26315789, 1.0],
        [21.49909241, 1.05263158, 7.36842105, 1.0],
        [21.49909241, 7.36842105, 1.05263158, 1.0],
        [21.49909241, -5.26315789, -5.26315789, 1.0],
        [21.49909241, -5.26315789, 7.36842105, 1.0],
        [21.49909241, 7.36842105, -5.26315789, 1.0],
        [21.49909241, 7.36842105, 7.36842105, 1.0],
        [21.49909241, -1.05263158, -11.57894737, 1.0],
        [21.49909241, 5.26315789, -11.57894737, 1.0],
        [21.49909241, -1.05263158, 13.68421053, 1.0],
        [21.49909241, -7.36842105, -11.57894737, 1.0],
        [21.49909241, 5.26315789, 13.68421053, 1.0],
        [23.60435556, 15.78947368, 1.05263158, 1.0],
        [23.60435556, -15.78947368, 1.05263158, 1.0],
        [21.49909241, -7.36842105, 13.68421053, 1.0],
        [23.60435556, -15.78947368, -5.26315789, 1.0],
        [23.60435556, 15.78947368, -5.26315789, 1.0],
        [23.60435556, -15.78947368, 7.36842105, 1.0],
        [23.60435556, 15.78947368, 7.36842105, 1.0],
        [21.49909241, 1.05263158, -17.89473684, 1.0],
        [23.60435556, 15.78947368, -11.57894737, 1.0],
        [23.60435556, -15.78947368, -11.57894737, 1.0],
        [21.49909241, -5.26315789, -17.89473684, 1.0],
        [21.49909241, 7.36842105, -17.89473684, 1.0],
        [23.60435556, -15.78947368, 13.68421053, 1.0],
        [23.60435556, 15.78947368, 13.68421053, 1.0],
        [21.49909241, 1.05263158, 20.0, 1.0],
        [21.49909241, -5.26315789, 20.0, 1.0],
        [21.49909241, 7.36842105, 20.0, 1.0],
        [23.60435556, -15.78947368, -17.89473684, 1.0],
        [23.60435556, 15.78947368, -17.89473684, 1.0],
        [23.60435556, -15.78947368, 20.0, 1.0],
        [23.60435556, 15.78947368, 20.0, 1.0],
    ])
    assert np.allclose(newpos, ref)

@pytest.mark.fast
def test_asugrid_P432_422():
    sym = "P 4 3 2 422"
    x = ipd.sym.xtal.xtal(sym)
    frames = x.primary_frames(contacting_only=True)
    allframes = x.frames(cells=3, xtalrad=0.9)
    framesavoid = allframes[len(x.primary_frames()):]
    cellsize = 100
    # pos = ipd.homog.hpoint([30, 20, 20])
    # pos = x.asucen(cellsize=cellsize)
    pos = np.array([0.16, 0.36, 0.0, 1])
    newpos, newcell = ipd.sym.place_asu_grid(
        pos,
        cellsize,
        frames=frames,
        framesavoid=framesavoid,
        lbub=0.2,
        lbubcell=0.2,
        nsamp=60,
        nsampcell=1,
        distcontact=(0.0, 0.35),
        distavoid=0.35,
        distspread=0.06,
        clusterdist=0.12,
    )
    print(repr(newpos))
    ref  # type: ignore
    assert np.allclose(
        newpos,
        np.array([
            [16.33898305, 35.66101695, 0.33898305, 1.0],
            [18.37288136, 35.66101695, -3.72881356, 1.0],
            [20.40677966, 37.69491525, 0.33898305, 1.0],
            [19.05084746, 35.66101695, 4.40677966, 1.0],
        ]),
    )

@pytest.mark.fast
def test_asugrid_P432_432():
    sym = "P 4 3 2 432"
    x = ipd.sym.xtal.xtal(sym)
    frames = x.primary_frames(contacting_only=True)
    allframes = x.frames(cells=3, xtalrad=0.9)
    framesavoid = allframes[len(x.primary_frames()):]
    cellsize = 100
    # pos = ipd.homog.hpoint([30, 20, 20])
    # pos = x.asucen(cellsize=cellsize)
    pos = np.array([0.2, 0.36, 0.1, 1])
    newpos, newcell = ipd.sym.place_asu_grid(
        pos,
        cellsize,
        frames=frames,
        framesavoid=framesavoid,
        lbub=0.15,
        lbubcell=0.2,
        nsamp=20,
        nsampcell=5,
        distcontact=(0.0, 0.35),
        distavoid=0.25,
        distspread=0.17,
        clusterdist=0.12,
    )
    print(repr(newpos))
    assert np.allclose(
        newpos,
        np.array([
            [17.63157895, 32.05263158, 9.21052632, 1.0],
            [20.78947368, 36.78947368, 9.21052632, 1.0],
            [14.47368421, 27.31578947, 9.21052632, 1.0],
            [16.05263158, 36.78947368, 10.78947368, 1.0],
            [14.47368421, 33.63157895, 6.05263158, 1.0],
            [12.89473684, 32.05263158, 10.78947368, 1.0],
            [9.73684211, 27.31578947, 7.63157895, 1.0],
            [20.78947368, 28.89473684, 12.36842105, 1.0],
            [11.31578947, 28.89473684, 2.89473684, 1.0],
            [8.15789474, 32.05263158, 9.21052632, 1.0],
            [5.0, 27.31578947, 7.63157895, 1.0],
        ]),
    )

@pytest.mark.fast
def test_asugrid_P432_432D2():
    sym = "P 4 3 2 432D2"
    x = ipd.sym.xtal.xtal(sym)
    frames = x.primary_frames(contacting_only=True)
    allframes = x.frames(cells=3, xtalrad=0.9)
    framesavoid = allframes[len(x.primary_frames()):]
    cellsize = 100
    # pos = ipd.homog.hpoint([30, 20, 20])
    # pos = x.asucen(cellsize=cellsize)
    pos = np.array([0.2, 0.36, 0.1, 1])
    newpos, newcell = ipd.sym.place_asu_grid(
        pos,
        cellsize,
        frames=frames,
        framesavoid=framesavoid,
        lbub=0.15,
        lbubcell=0.2,
        nsamp=40,
        nsampcell=1,
        distcontact=(0.0, 0.35),
        distavoid=0.3,
        distspread=0.15,
        clusterdist=0.07,
    )
    print(repr(newpos))
    assert np.allclose(
        newpos,
        np.array([
            [19.61538462, 35.61538462, 9.61538462, 1.0],
            [19.61538462, 37.15384615, 11.15384615, 1.0],
            [21.15384615, 37.15384615, 9.61538462, 1.0],
            [21.15384615, 35.61538462, 8.07692308, 1.0],
            [18.07692308, 37.15384615, 9.61538462, 1.0],
            [18.07692308, 35.61538462, 8.07692308, 1.0],
            [22.69230769, 35.61538462, 9.61538462, 1.0],
            [21.15384615, 38.69230769, 11.15384615, 1.0],
            [16.53846154, 35.61538462, 9.61538462, 1.0],
            [15.76923077, 34.84615385, 7.30769231, 1.0],
            [13.46153846, 34.84615385, 8.07692308, 1.0],
        ]),
    )
    return

if __name__ == "__main__":
    main()
