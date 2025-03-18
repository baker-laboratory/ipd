import numpy as np

import ipd

def vispoints(pos, cell, frames, allframes):
    for i in range(len(pos)):
        # result0 = ipd.homog.hxform(ipd.hscaled(newcell[i], frames), pos[i], is_points=True)
        # ipd.showme(result0, sphere=25 / 2, kind='point'
        # result1 = ipd.homog.hxform(ipd.hscaled(newcell[i], framesavoid), pos[i])
        # ipd.showme(result1, sphere=3)
        f = np.concatenate([frames, allframes])
        colors = [(0, 1, 1)] + [(1, 0, 0)] * (len(frames) - 1) + [(1, 1, 1)] * (len(f))
        result = ipd.homog.hxform(ipd.hscaled(cell[i], f), pos[i])
        ipd.showme(result, sphere=9, col=colors)

def scaleunit(scale, val):
    if not isinstance(val, (int, float, np.ndarray)):
        return np.array([scaleunit(scale, v) for v in val])
    if np.all((-1 <= val) * (val <= 1)):
        return val * scale
    else:
        return val

def place_asu_grid_multiscale(
    pos,
    cellsize,
    *a,
    minpos=1,
    **kw,
):
    # print('place_asu_grid_multiscale', repr(pos), cellsize, flush=True)
    kw = ipd.dev.Bunch(kw)

    assert kw.lbub < 1
    assert kw.lbubcell < 1

    newpos, newcell = place_asu_grid(pos, cellsize, *a, **kw)
    if len(newpos) >= minpos:
        return newpos, newcell
    kw.distcontact = np.array(kw.distcontact)
    for i in range(5, 0, -1):
        if "refpos" not in kw:
            kw.refpos = pos.copy()
        if "refcell" not in kw:
            kw.refcell = cellsize
        # ic(i, repr(pos), cellsize)
        print("place_asu_grid_multiscale", i, flush=True)
        newpos, newcell = place_asu_grid(
            pos,
            cellsize,
            *a,
            **kw.sub(
                nsampcell=kw.nsampcell + (i-1),
                lbub=kw.lbub + (i-1) * 0.03,
                lbubcell=kw.lbubcell + (i-1) * 0.01,
                dnistcontact=(kw.distcontact[0], kw.distcontact[1] + (i-1)),
                distavoid=kw.distavoid - (i-1),
                distspread=kw.distspread + (i-1),
            ),
        )
        pos, cellsize = newpos[0], newcell[0]
        # ic(kw.refpos)
        # ic(newpos[1] - kw.refpos)
        # ic(newpos[:5])
        # ic(ipd.homog.hnorm(newpos - kw.refpos)[:5])

        # vispoints(newpos[:1], newcell[:1], kw.frames, kw.framesavoid)

    return newpos, newcell

def place_asu_grid(
        pos,
        cellsize,
        *,
        frames,
        framesavoid,
        lbub=(-10, 10),
        lbubcell=(-20, 20),
        nsamp=1,
        nsampcell=None,
        distcontact=(10, 15),
        distavoid=20,
        distspread=9e9,
        clusterdist=3,
        refpos=None,
        refcell=None,
        printme=False,
        **kw,
):
    if printme:
        print("   # yapf: disable")
        print("   kw =", repr(kw))
        print(f"""   newpos, newcell = ipd.sym.place_asu_grid(
      pos={ipd.dev.arraystr(pos)},
      cellsize={repr(cellsize)},
      frames={ipd.dev.arraystr(frames)},
      framesavoid={ipd.dev.arraystr(framesavoid)},
      lbub={repr(lbub)},
      lbubcell={repr(lbubcell)},
      nsamp={repr(nsamp)},
      nsampcell={repr(nsampcell)},
      distcontact={repr(distcontact)},
      distavoid={repr(distavoid)},
      distspread={repr(distspread)},
      clusterdist={repr(clusterdist)},
      refpos={repr(refpos)},
      refcell={repr(refcell)},
   )""")
        print("   # yapf: enable", flush=True)

    assert isinstance(cellsize, (int, float))
    nsampcell = nsampcell or nsamp
    if isinstance(lbub, (int, float)):
        lbub = (-lbub, lbub)
    if isinstance(lbubcell, (int, float)):
        lbubcell = (-lbubcell, lbubcell)
    cellsize0, frames0, framesavoid0 = cellsize, frames, framesavoid
    pos0 = scaleunit(cellsize0, pos)
    pos = scaleunit(cellsize0, pos)
    refpos = pos if refpos is None else refpos
    refpos = scaleunit(cellsize0, refpos)
    refcell = cellsize if refcell is None else refcell
    pos[3], pos0[3], refpos[3] = 1, 1, 1  # type: ignore
    lbub = scaleunit(cellsize0, lbub)
    lbubcell = scaleunit(cellsize0, lbubcell)
    distcontact = scaleunit(cellsize0, distcontact)
    distavoid, distspread, clusterdist = scaleunit(cellsize0, [distavoid, distspread, clusterdist])

    samp = np.linspace(*lbub, nsamp)  # type: ignore
    xyz = np.meshgrid(samp, samp, samp)
    delta = np.stack(xyz, axis=3).reshape(-1, 3)
    delta = ipd.homog.hvec(delta)
    posgrid = pos + delta
    posgrid = posgrid[np.all(posgrid > 0, axis=1)]
    # ipd.showme(posgrid)
    cellsizes = cellsize + np.linspace(*lbubcell, nsampcell)  # type: ignore
    if nsampcell < 2:
        cellsizes = np.array([cellsize])
    # ic(frames0.shape, framesavoid0.shape)
    allframes = np.concatenate([frames0, framesavoid0])
    frames = np.stack([ipd.hscaled(s, frames[1:]) for s in cellsizes])
    framesavoid = np.stack([ipd.hscaled(s, framesavoid) for s in cellsizes])

    contact = ipd.homog.hxformpts(frames, posgrid, outerprod=True)
    avoid = ipd.homog.hxformpts(framesavoid, posgrid, outerprod=True)
    dcontact = ipd.homog.hnorm(posgrid - contact)
    davoid = ipd.homog.hnorm(posgrid - avoid)
    dcontactmin = np.min(dcontact, axis=1)
    dcontactmax = np.max(dcontact, axis=1)
    davoidmin = np.min(davoid, axis=1)

    okavoid = davoidmin > distavoid
    okccontactmin = dcontactmin > distcontact[0]  # type: ignore
    okccontactmax = dcontactmax < distcontact[1]  # type: ignore
    okspread = dcontactmax - dcontactmin < distspread
    ic(np.sum(okavoid), np.sum(okccontactmin), np.sum(okccontactmax), np.sum(okspread))  # type: ignore
    ok = okavoid * okccontactmin * okccontactmax * okspread
    w = np.where(ok)
    goodcell = cellsizes[w[:][0]]
    goodpos = posgrid[w[:][1]]
    # cellpos = goodpos / goodcell[:, None]
    # cellpos0 = pos0 / cellsize0
    origdist = np.sqrt(ipd.homog.hnorm2(goodpos - refpos) + ((goodcell-refcell) * 1.1)**2)
    order = np.argsort(origdist)
    goodcell, goodpos = goodcell[order], goodpos[order]
    # ic(origdist[order])
    # ic(goodpos[0])
    # ic(goodcell[0])
    # ic(refpos)
    # assert 0
    if clusterdist > 0 and len(goodpos) > 1:
        coords = ipd.homog.hxformpts(frames0, goodpos, outerprod=True)
        coords = coords.swapaxes(0, 1).reshape(len(goodpos), -1)
        keep, clustid = ipd.homog.hcom.cluster.cookie_cutter(coords, float(clusterdist))
        goodpos = goodpos[keep]
        goodcell = goodcell[keep]

    # f = ipd.hscaled(goodcell[0], frames0)
    # p = ipd.homog.hxformpts(f, goodpos[0])
    # ic(ipd.homog.hnorm(p[0] - p[1]), ipd.homog.hnorm(p[0] - p[-1]))

    # if len(goodpos):
    # ic(refpos)
    # ic(goodpos[:5])
    # ic(goodpos[1] - refpos)
    # ic(ipd.homog.hnorm(goodpos - refpos)[:5])

    return goodpos, goodcell

def place_asu_sample_dof(
    sym,
    coords,
    cellsize,
    axis,
    contactdist,
    cartnsamp,
    angnsamp,
    cartrange,
    angrange,
    cellrange,
    cellnsamp,
):
    axis = ipd.homog.hnormalized(axis)
    angsamp = ipd.homog.hrot(axis, np.linspace(-angrange, angrange, angnsamp))
    cartsamp = ipd.homog.htrans(axis * np.linspace(-cartrange, cartrange, cartnsamp))
    cellsamp = np.linspace(-cellrange, cellrange, cellnsamp)
    cframes = np.stack([ipd.sym.frames(sym, cellsize=c) for c in cellsamp])
    ic(cframes.shape)  # type: ignore
