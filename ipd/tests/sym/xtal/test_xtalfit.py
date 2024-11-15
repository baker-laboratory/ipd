import numpy as np
import pytest
import ipd

pytest.skip(allow_module_level=True)

def main():
    test_fit_xtal_to_coords()
    test_xtalfit_I213()

@pytest.mark.fast
def test_xtalfit_I213():
    pytest.importorskip("willutil_cpp")
    sym = "I213_32"
    xtal = ipd.sym.xtal.xtal(sym)
    fname = ipd.dev.package_testdata_path("pdb/i213fittest.pdb")
    pdb = ipd.pdb.readpdb(fname)
    cell0 = float(pdb.cryst1[25:33])

    coords0 = ipd.homog.hpoint(pdb.ncac(splitchains=True))
    asym, cell = xtal.fit_coords(coords0)  # ca only
    # asym, cell = ipd.sym.fix_coords_to_xtal(sym, coords0)  # ca only

    ref = coords0.reshape(-1, 4)
    startsym = xtal.symcoords(coords0[0], cells=0, cellsize=cell0).reshape(-1, 4)
    fitsym = xtal.symcoords(asym, cells=0, cellsize=cell).reshape(-1, 4)
    rorig = ipd.homog.hrmsfit(ref, startsym)[0]
    rfit = ipd.homog.hrmsfit(ref, fitsym)[0]
    ic(rorig)  # type: ignore
    ic(rfit)  # type: ignore
    vals = list()
    for i in range(10):
        fitsym = xtal.symcoords(asym + ipd.homog.hrandvec(), cells=0, cellsize=cell + np.random.normal()).reshape(-1, 4)
        vals.append(ipd.homog.hrmsfit(ref, fitsym)[0] - rfit)
    vals = np.array(vals)
    ic(np.min(vals), np.mean(vals))  # type: ignore
    assert np.min(vals) > -0.2

def DISABLED_test_xtalfit_I213_bk():
    fname = ipd.dev.package_testdata_path("pdb/i213fittest.pdb")
    pdb = ipd.pdb.readpdb(fname)
    coords = ipd.homog.hpoint(pdb.ncac(splitchains=True))
    coords0 = coords.copy()
    ipd.pdb.dumppdb("start.pdb", coords)
    # ipd.showme(coords[0])
    # ipd.showme(coords[1])
    # ipd.showme(coords[2])
    # ipd.showme(coords[3])
    cacoords = coords[:, :, 1]
    xtal = ipd.sym.xtal.xtal("I213_32")
    ax3, cen3 = guessaxis(cacoords, [0, 1, 2])  # type: ignore
    ax2, cen2 = guessaxis(cacoords, [0, 3])  # type: ignore
    if ipd.homog.hdot([1, 1, 1], ax3) < 0:
        ax3 = -ax3
    if ipd.homog.hdot([0, 0, 1], ax2) < 0:
        ax2 = -ax2
    # ic(ax3, ax2)
    # ic(cen3, cen2)

    xalign = ipd.homog.halign2(ax3, ax2, [1, 1, 1], [0, 0, 1])
    ax2 = ipd.homog.hxform(xalign, ax2)
    ax3 = ipd.homog.hxform(xalign, ax3)
    cen2 = ipd.homog.hxform(xalign, cen2)
    cen3 = ipd.homog.hxform(xalign, cen3)
    coords = ipd.homog.hxform(xalign, coords)
    # ipd.pdb.dumppdb('xalign.pdb', coords)
    # ic(guessaxis(coords, 3))

    # ic(ax3, ax2)
    # ic(cen3, cen2)
    cen3 = ipd.homog.hcom(coords[:3, :, 1].mean(axis=0))
    cen2 = ipd.homog.hcom(coords[0, :, 1]) / 2 + ipd.homog.hcom(coords[3, :, 1]) / 2
    ic(cen3, cen2)  # type: ignore

    # assert 0

    def loss(x):
        # ((cen3[0] + x[0]) - (cen3[1] + x[1]))**2 + ((cen2[0] + x[0]) / 2 - (cen2[1] + x[1]))**2
        x3f = cen3[0] + x[0]
        y3f = cen3[1] + x[1]
        x2f = cen2[0] + x[0]
        y2f = cen2[1] + x[1]
        return (x3f - y3f)**2 + (0.75*x2f - 1.5*y2f)**2

    import scipy.optimize  # type: ignore
    opt = scipy.optimize.minimize(loss, [0, 0], method="Powell", tol=0.3)
    x = opt.x
    ic(opt.x)  # type: ignore
    z = (cen3[0] + x[0] + cen3[1] + x[1]) / 2
    xdelta = ipd.homog.htrans([x[0], x[1], z - cen3[2]])
    cen2 = ipd.homog.hxform(xdelta, cen2)
    cen3 = ipd.homog.hxform(xdelta, cen3)
    coords = ipd.homog.hxform(xdelta, coords)
    # ipd.pdb.dumppdb('xdelta.pdb', coords)

    ic(cen2, cen3)  # type: ignore
    cell = (cen2[0] / 2 + cen2[1]) / 2 * 4
    ic(cell)  # type: ignore
    # xtal.dump_pdb('orig.pdb', coords0[0], cellsize=100.942)
    # xtal.dump_pdb('test.pdb', coords[0], cellsize=cell)

    cell0 = float(pdb.cryst1[25:33])
    ref = coords0[:, :, 1].reshape(-1, 4)

    startsym = xtal.symcoords(coords0[0], cells=0, cellsize=cell0)[:, :, 1].reshape(-1, 4)
    fitsym = xtal.symcoords(coords[0], cells=0, cellsize=cell)[:, :, 1].reshape(-1, 4)
    rorig = ipd.homog.hrmsfit(ref, startsym)[0]
    rfit = ipd.homog.hrmsfit(ref, fitsym)[0]
    ic(rorig)  # type: ignore
    ic(rfit)  # type: ignore
    vals = list()
    for i in range(100):
        fitsym = xtal.symcoords(coords[0] + ipd.homog.hrandvec(), cells=0,
                                cellsize=cell + np.random.normal())[:, :, 1].reshape(-1, 4)
        vals.append(ipd.homog.hrmsfit(ref, fitsym)[0] - rfit)
    vals = np.array(vals)
    ic(np.min(vals), np.mean(vals))  # type: ignore
    # x = np.linspace(-cartbound, cartbound, 20)
    # y = np.linspace(-cartbound, cartbound, 20)
    # samp = np.meshgrid(x, y)
    # samp = np.stack(samp, axis=2).reshape(-1, 2)
    # x, y = samp.T
    # w = np.argmin(d)
    # ic(d[w], x[w], y[w])

@pytest.mark.fast
def test_fit_xtal_to_coords():
    pytest.importorskip("willutil_cpp")
    x = ipd.sym.Xtal("L632")

    np.random.seed(7)

    pts0 = np.array([
        [18.44458833, 0.02276394, -3.29067642, 1.0],
        [-5.08245051, 7.47204762, -2.46703027, 1.0],
        [-2.60027436, -6.70113306, 0.52336958, 1.0],
        [17.3808256, -0.92255618, -0.84246037, 1.0],
    ]).reshape(-1, 1, 4)
    pts = pts0.copy()

    cellsize, cartshift = x.fit_xtal_to_coords(pts)
    # ic(cellsize)
    # ic(cartshift)
    assert np.allclose(cellsize, 27.423643867)
    assert np.allclose(cartshift, np.array([-2.07587708, -4.49116458, -0.81612973, 0.0]))

    cellsize, cartshift = x.fit_xtal_to_coords(pts, noshift=True)
    # ic(cellsize)
    # ic(cartshift)
    assert np.allclose(cellsize, 35.828085361)
    assert np.allclose(cartshift, np.array([0, 0, 0, 0]))

    assert np.allclose(pts0, pts)

if __name__ == "__main__":
    main()
