import functools as ft

import numpy as np

import ipd
from ipd.sym.xtal.xtalcls import Xtal

def npscorefunc(xtal, scom, state):
    dis2 = 0
    for i, (s, com) in enumerate(zip(xtal.symelems, scom)):
        newcom = com + state.cartshift
        newcen = s.cen * state.cellsize
        dis2 = dis2 + ipd.homog.hpointlinedis(newcom, newcen, s.axis)**2
    err = np.sqrt(np.sum(dis2))
    return err

def torchscorefunc(xtal, scom, cellsize, cartshift, grad=True):
    import torch  # type: ignore
    dis2 = torch.tensor([0.0], requires_grad=grad)
    for i, (s, com) in enumerate(zip(xtal.symelems, scom)):
        com = torch.tensor(com[:3], requires_grad=grad)
        cen = torch.tensor(s.cen[:3], requires_grad=grad)
        axis = torch.tensor(s.axis[:3], requires_grad=grad)
        newcom = com + cartshift[:3]
        newcen = cen * cellsize
        dis2 = dis2 + ipd.homog.thgeom.thpoint_line_dist2(newcom, newcen, axis)
    err = torch.sqrt(torch.sum(dis2))
    return err

def fix_coords_to_xtal(sym, coords):
    if sym in ("I213", "I 21 3", "I213_32"):
        return xtalfit_I213(coords)
    else:
        raise ValueError(f"xtalfit: can't handle symmetry {sym}")

def guess_cx_axis(coords, nfold):
    if isinstance(nfold, int):
        idx = list(range(nfold))
    else:
        idx = nfold
        nfold = len(nfold)
    coords = coords.reshape(coords.shape[0], -1, coords.shape[-1])
    ncheck = nfold if nfold > 2 else 1
    # ic(coords.shape)
    fit = [ipd.hrmsfit(coords[idx[i]], coords[idx[(i+1) % len(idx)]]) for i in range(ncheck)]
    rms, fit, xform = zip(*fit)
    axis, ang, cen, hel = ipd.homog.axis_angle_cen_hel_of(np.stack(xform))
    # ic(hel)
    return axis.mean(0), cen.mean(0)

def xtalfit_I213(coords):
    coords = np.asarray(coords)
    if coords.ndim == 3:
        coords = coords[:, :, None]
    assert coords.ndim == 4
    assert len(coords) == 4
    coords0 = coords.copy()
    cacoords = coords[:, :, min(coords.shape[2] - 1, 1)]
    xtal = ipd.sym.xtal.xtal("I213_32")
    ax3, _ = guess_cx_axis(cacoords, [0, 1, 2])
    ax2, _ = guess_cx_axis(cacoords, [0, 3])
    # ic(ax3, ax2)
    if ipd.homog.hdot([1, 1, 1], ax3) < 0:
        ax3 = -ax3
    if ipd.homog.hdot([0, 0, 1], ax2) < 0:
        ax2 = -ax2
    xalign = ipd.homog.halign2(ax3, ax2, [1, 1, 1], [0, 0, 1])
    # ax2 = ipd.homog.hxform(xalign, ax2)
    # ax3 = ipd.homog.hxform(xalign, ax3)
    cacoords = ipd.homog.hxform(xalign, cacoords)

    cen3 = ipd.homog.hcom(cacoords[:3].mean(axis=0))
    cen2 = ipd.homog.hcom(cacoords[0]) / 2 + ipd.homog.hcom(cacoords[3]) / 2

    # ic(cen3, cen2)

    def loss(x):
        # ((cen3[0] + x[0]) - (cen3[1] + x[1]))**2 + ((cen2[0] + x[0]) / 2 - (cen2[1] + x[1]))**2
        x3f = cen3[0] + x[0]
        y3f = cen3[1] + x[1]
        x2f = cen2[0] + x[0]
        y2f = cen2[1] + x[1]
        return (x3f - y3f)**2 + (0.75*x2f - 1.5*y2f)**2

    import scipy.optimize  # type: ignore
    # method = 'Nelder-Mead'
    method = "COBYLA"
    # method = 'Powell'
    opt = scipy.optimize.minimize(loss, [0, 0], method=method, tol=0.1)
    x = opt.x
    z = (cen3[0] + x[0] + cen3[1] + x[1]) / 2
    xdelta = ipd.homog.htrans([x[0], x[1], z - cen3[2]])
    cen2 = ipd.homog.hxform(xdelta, cen2)
    cen3 = ipd.homog.hxform(xdelta, cen3)

    coords = ipd.homog.hxform(xdelta @ xalign, coords0)
    cell = (cen2[0] / 2 + cen2[1]) / 2 * 4

    return coords[0], cell

def fix_xtal_to_coords(xtal, coords, cellsize=None, domc=True, domin=False, noshift=False, mcsteps=1000, **kw):
    "OK... this is a pretty inefficient way..."
    coms = ipd.homog.hcom(coords)
    if isinstance(cellsize, np.ndarray):
        cellsize = cellsize[0]
    cellsize = cellsize if cellsize is not None else 100.0
    # ic(coms)

    scom = list()
    n = 0
    for s in xtal.symelems:
        scom.append(coms[0].copy())
        nops = len(s.operators)
        for i in range(nops - 1):
            n += 1
            # ic(len(scom), n)
            scom[-1] += coms[n]
        scom[-1] /= nops
    # ic(scom)
    elem0 = xtal.symelems[0]
    # ic(scom)

    if noshift:
        cartshift = ipd.homog.hvec([0, 0, 0])
    else:
        cartshift = -ipd.homog.hvec(ipd.homog.hpointlineclose(scom[0], elem0.cen, elem0.axis))

    assert not domin
    assert domc

    if domc:
        state = ipd.dev.Bunch(
            cellsize=cellsize,
            # cartshift=np.array([0., 0, 0, 0]),
            cartshift=cartshift,
        )
        step = 5
        mc = ipd.MonteCarlo(ft.partial(npscorefunc, xtal, scom), temperature=0.3)
        for i in range(mcsteps):
            # if i % 100 == 199:
            # state = mc.beststate
            prev = state.copy()  # type: ignore
            state.cellsize += step * np.random.randn()  # type: ignore
            if not noshift:
                state.cartshift += 0.02 * step * ipd.homog.hrandvec()  # type: ignore
            acccepted = mc.try_this(state)
            if not acccepted:
                state = prev
            else:
                # print(state.cellsize, ipd.homog.hnorm(state.cartshift), mc.best)
                pass
                # ic(mc.acceptfrac, step)
                # if mc.acceptfrac > 0.25: step *= 1.01
                # else: step *= 0.99

        # print(mc.best)
        # print(mc.beststate)
        # print(mc.acceptfrac)

        return mc.beststate.cellsize, mc.beststate.cartshift
        assert 0

        cellsize, cartshift = mc.beststate

        return cellsize, cartshift.astype(coords.dtype)

    if domin:
        import torch  # type: ignore
        # torch.autograd.set_detect_anomaly(True)

        # check
        v1 = npscorefunc(xtal, scom, ipd.dev.Bunch(cellsize=cellsize, cartshift=cartshift))
        v2 = torchscorefunc(xtal, scom, cellsize, cartshift, grad=False)
        assert np.allclose(v1, v2)

        cellsize = torch.tensor(cellsize, requires_grad=True)
        cartshift = torch.tensor(cartshift[:3], requires_grad=True)
        for i in range(10):
            err = torchscorefunc(xtal, scom, cellsize, cartshift)
            err.backward()
            cellgrad = cellsize.grad
            cartshiftgrad = cartshift.grad
            mul = 1
            cellsize = (cellsize - mul*cellgrad).clone().detach().requires_grad_(True)  # type: ignore
            cartshift = (cartshift - mul*cartshiftgrad).clone().detach().requires_grad_(True)  # type: ignore
            ic(err)  # , cellsize, cartshift, cartshiftgrad)  # type: ignore
        assert 0

        cellsize = 100
        cartshift = np.array([0.0, 0, 0, 0])
        besterr, beststate = 9e9, None
        step = 10.0
        lasterr, laststate = 9e9, None
        for i in range(1000):
            offaxis = list()
            for s, com in zip(xtal.symelems, scom):
                offaxis.append(ipd.homog.hpointlinedis(com, cellsize * s.cen, s.axis)**2)
            err = np.sqrt(np.sum(offaxis))
            if err < besterr:
                besterr = err
                beststate = cellsize, cartshift
            if err - lasterr < 0:
                lasterr = err
                laststate = cellsize, cartshift
            else:
                cellsize, cartshift = laststate  # type: ignore
            ic(cellsize, err, step)  # type: ignore
            step *= 0.99
        ic(besterr, beststate)  # type: ignore
    assert 0

def analyze_xtal_asu_placement(sym):
    import collections

    xtal = Xtal(sym)
    cen = xtal.asucen(xtalasumethod="closest_to_cen", use_olig_nbrs=True)
    ic(cen)  # type: ignore
    side = np.linspace(0, 1, 13, endpoint=False)
    samp = np.meshgrid(side, side, side)
    samp = np.stack(samp, axis=3).reshape(-1, 3)
    # ic(samp[:10])
    mindisxtal = collections.defaultdict(list)

    for i, pt in enumerate(samp):
        if i % 10 == 0:
            ic(i, len(samp))  # type: ignore
        ptsym = xtal.symcoords(pt, cells=3, ontop=None)
        # ipd.showme(ptsym * 5)
        # assert 0
        delta = np.sqrt(np.sum((ipd.homog.hpoint(pt) - ptsym)**2, axis=1))

        # ic(delta)
        # np.fill_diagonal(delta, 9e9)
        zero = np.abs(delta) < 0.000001
        if np.sum(zero) != 1:
            continue
        delta[zero] = 9e9
        mindis = np.round(np.min(delta), 3)
        if mindis > 0.001:
            mindisxtal[mindis].append(pt)
    ic(len(samp))  # type: ignore
    frames = xtal.cellframes(cells=3)

    ic(list(mindisxtal.keys()))  # type: ignore
    for k in reversed(sorted(mindisxtal.keys())):
        pts = mindisxtal[k]
        ptset = set()
        for i, pt in enumerate(pts):
            # ic(pt)
            ptset.add(tuple(np.round(xtal.coords_to_asucen(pt, frames=frames)[0], 3)))
        ic(k)  # type: ignore
        for pt in ptset:
            ic(pt, ipd.homog.hnorm(pt - cen), ipd.homog.hnorm(pt))  # type: ignore
    # assert 0

    keys = sorted(mindisxtal.keys())
    scale = 8
    for k in keys:
        v = mindisxtal[k]
        ic(k, len(v))  # type: ignore
        ptsym = xtal.symcoords(v[0] * 8, cellsize=8, cells=2, ontop=None)
        # ipd.showme(ptsym, name=f'{k}')
        ic(v[0])  # type: ignore
    for i, v in enumerate(mindisxtal[0.288]):
        ptsym = xtal.symcoords(v * 8, cellsize=8, cells=2, ontop=None)
        ipd.showme(ptsym, name=f"{i}")
    # assert 0
