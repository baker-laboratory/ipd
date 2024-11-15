import numpy as np

import ipd
from ipd.sym.xtal.xtalcls import Xtal, interp_xtal_cell_list
from ipd.sym.xtal.xtalinfo import SymElem
from ipd.viz.pymol_viz import cgo_cube, cgo_cyl, cgo_fan, cgo_sphere, pymol_load

@pymol_load.register(SymElem)  # type: ignore
def pymol_viz_SymElem(
    toshow,
    name="SymElem",
    state=None,
    col="bycx",
    center=np.array([0, 0, 0, 1]),
    # scalefans=None,
    fansize=0.05,
    fanshift=0,
    fancover=1.0,
    make_cgo_only=False,
    cyc_ang_match_tol=0.1,
    axislen=0.2,
    axisrad=0.008,
    addtocgo=None,
    scale=1,
    cellshift=(0, 0, 0),
    fanrefpoint=[1, 2, 3, 1],
    symelemscale=1,
    symelemtwosided=False,
    shifttounit=False,
    symelemradiuscut=9e9,
    symelemcentercut=[0, 0, 0],
    **kw,
):
    import pymol  # type: ignore
    if state: state["seenit"][name] += 1

    v = pymol.cmd.get_view()

    axislen = axislen * scale * symelemscale
    axisrad = axisrad * scale * symelemscale
    fanthickness = 0.0 * scale * symelemscale
    fansize = fansize * scale * symelemscale
    fanshift = fanshift * scale * symelemscale

    cen = ipd.homog.hscaled(scale, toshow.cen)
    if shifttounit:
        if cen[0] < 0:
            cen[0] += scale
        if cen[1] < 0:
            cen[1] += scale
        if cen[2] < 0:
            cen[2] += scale
        if cen[0] > scale:
            cen[0] -= scale
        if cen[1] > scale:
            cen[1] -= scale
        if cen[2] > scale:
            cen[2] -= scale
    cen[0] += scale * cellshift[0]
    cen[1] += scale * cellshift[1]
    cen[2] += scale * cellshift[2]
    if ipd.homog.hnorm(cen - ipd.homog.hpoint(symelemcentercut)) > symelemradiuscut:
        return

    ang = toshow.angle
    if np.isclose(ang, np.pi * 4 / 5, atol=1e-4):
        ang /= 2
    if col == "bycx":
        if False:
            pass
        elif np.isclose(ang, np.pi * 2 / 2, atol=cyc_ang_match_tol):
            col = [1, 1, 0]
        elif np.isclose(ang, np.pi * 2 / 3, atol=cyc_ang_match_tol):
            col = [0, 1, 1]
        elif np.isclose(ang, np.pi * 2 / 4, atol=cyc_ang_match_tol):
            col = [1, 0, 1]
        elif np.isclose(ang, np.pi * 2 / 5, atol=cyc_ang_match_tol):
            col = [1, 0, 1]
        elif np.isclose(ang, np.pi * 2 / 6, atol=cyc_ang_match_tol):
            col = [1, 0, 1]
        else:
            col = [0.5, 0.5, 0.5]
    elif col == "random":
        col = np.random.rand(3) / 2 + 0.5
    if toshow.vizcol is not None:
        col = toshow.vizcol

    fanrefpoint = fanrefpoint.copy()
    fanrefpoint[0] += scale * cellshift[0]
    fanrefpoint[1] += scale * cellshift[1]
    fanrefpoint[2] += scale * cellshift[2]

    axis = toshow.axis
    c1 = cen + axis*axislen/2
    c2 = cen - axis*axislen/2

    mycgo = list()
    mycgo += cgo_cyl(c1, c2, axisrad, col=col)
    # ic(fansize, ang)

    arc = min(np.pi * 2, ang * fancover)
    # ic(axis)
    # ic(cen)
    # ic(fansize)
    # ic(fanthickness)
    # ic(fanrefpoint)
    # ic(fanshift)
    mycgo += cgo_fan(axis,
                     cen,
                     fansize,
                     arc=arc,
                     thickness=fanthickness,
                     col=col,
                     startpoint=fanrefpoint,
                     fanshift=fanshift)
    if symelemtwosided:
        col2 = (1, 1, 1)
        mycgo += cgo_fan(
            axis,
            cen,
            fansize,
            arc=arc,
            thickness=fanthickness,
            col=col2,
            startpoint=fanrefpoint,
            fanshift=fanshift - 0.01,
        )

    if addtocgo is None:
        pymol.cmd.load_cgo(mycgo, f'{name}_{state["seenit"][name]}')  # type: ignore
        pymol.cmd.set_view(v)
    else:
        addtocgo.extend(mycgo)
    if make_cgo_only:
        return mycgo
    return None

@pymol_load.register(Xtal)  # type: ignore
def pymol_viz_Xtal(
        toshow,
        name="xtal",
        state=None,
        scale=10,
        # neighbors=1,
        # cellshift=(0, 0, 0),
        cells=1,
        showsymelems=True,
        showgenframes=False,
        splitobjs=False,
        showpoints=None,
        fanshift=0,
        fansize=0.1,
        showcube=None,
        pointradius=1,
        addtocgo=None,
        pointcol=(0.5, 0.5, 0.5),
        **kw,
):
    import pymol  # type: ignore
    state["seenit"][name] += 1  # type: ignore
    if "cellsize" in kw:
        assert kw["cellsize"] == scale
    name = f'{name}_{state["seenit"][name]}'  # type: ignore
    # xcellshift = ipd.htrans(cellshift)
    allcgo = list() if addtocgo is None else addtocgo
    # for x in toshow.unitframes:
    # for s in toshow.symelems:
    # pymol_viz_SymElem(ipd.homog.hxform(x, s), scale=scale, **kw)
    if showsymelems:
        cgo = list()
        for cellshift in interp_xtal_cell_list(cells):
            xcellshift = ipd.htrans(cellshift)
            # for cell in range(cells**3):
            # cellshift = np.array((cell // cells**2, cell // cells % cells, cell % cells))
            # cellshift -= (cells - 1) // 2
            # continue
            for i, elems in enumerate(toshow.unitelems):
                size = fansize[i] if isinstance(fansize, (list, tuple, np.ndarray)) else fansize
                shift = fanshift[i] if isinstance(fanshift, (list, tuple, np.ndarray)) else fanshift
                for elem in elems:
                    fanrefpoint = get_fanrefpoint(toshow)
                    fanrefpoint = ipd.homog.hxform(elem.origin, fanrefpoint)
                    fanrefpoint = xcellshift @ fanrefpoint
                    fanrefpoint = ipd.hscaled(scale, fanrefpoint)
                    pymol_viz_SymElem(
                        elem,
                        state,
                        scale=scale,
                        addtocgo=cgo,
                        fanrefpoint=fanrefpoint,
                        fansize=size,
                        fanshift=fanshift,
                        cellshift=cellshift,
                        shifttounit=toshow.dimension == 3,
                        **kw,
                    )
        if splitobjs:
            pymol.cmd.load_cgo(cgo, f"{name}_symelem{i}")  # type: ignore
        allcgo += cgo
        xshift2 = xcellshift.copy()  # type: ignore
        xshift2[:3, 3] *= scale
        showcube = toshow.dimension == 3 if showcube is None else showcube
        if showcube:
            cgo = cgo_cube(ipd.homog.hxform(xshift2, [0, 0, 0]), ipd.homog.hxform(xshift2, [scale, scale, scale]), r=0.03)
            if splitobjs:
                pymol.cmd.load_cgo(cgo, f"{name}_cube")
        allcgo += cgo

    # for i, (elem, frame) in enumerate(toshow.unitelems[1]):

    if isinstance(showpoints, np.ndarray):
        frames = toshow.cellframes(cellsize=scale, cells=cells)
        px = ipd.homog.hxform(frames, showpoints, flat=True)
        cgo = list()
        for p in px:
            cgo += cgo_sphere(p, rad=pointradius, col=pointcol)
        # ic(px.shape)
        # assert 0
        pymol.cmd.load_cgo(cgo, f"{name}_pts{i}")  # type: ignore
    elif showpoints not in (None, False, 0):
        showpts = xtal_show_points(showpoints, **kw)
        frames = toshow.cellframes(cellsize=1, cells=cells)
        cgo = ipd.viz.cgo_frame_points(frames, scale, showpts, **kw)
        # cgo = list()
        # for i, frame in enumerate(frames):
        # for p, r, c in zip(*showpts):
        # cgo += cgo_sphere(scale * ipd.homog.hxform(frame, p), rad=scale * r, col=c)
        if splitobjs:
            pymol.cmd.load_cgo(cgo, f"{name}_pts{i}")  # type: ignore
        allcgo += cgo

    if showgenframes:
        col = (1, 1, 1)
        cgo = ipd.viz.cgo_frame_points(toshow.genframes, scale, showpts, **kw)  # type: ignore
        # cgo = list()
        # for i, frame in enumerate(toshow.genframes):
        # cgo += cgo_sphere(scale * ipd.homog.hxform(frame, showpts[0]), rad=scale * 0.05, col=col)
        # cgo += cgo_sphere(scale * ipd.homog.hxform(frame, showpts[1]), rad=scale * 0.03, col=col)
        # cgo += cgo_sphere(scale * ipd.homog.hxform(frame, showpts[2]), rad=scale * 0.02, col=col)
        if splitobjs:
            pymol.cmd.load_cgo(cgo, f"{name}_GENPTS{i}")  # type: ignore
        allcgo += cgo

    if not splitobjs:
        pymol.cmd.load_cgo(allcgo, f"{name}_all")

    return state

def xtal_show_points(which, pointscale=1, pointshift=(0, 0, 0), scaleptrad=1, **kw):
    s = pointscale * scaleptrad
    pointshift = np.asarray(pointshift)
    showpts = [
        np.empty(shape=(0, 3)),
        np.array([
            [0.28, 0.13, 0.13],
            [0.28, 0.13 + 0.06*s, 0.13],
            [0.28, 0.13, 0.13 + 0.05*s],
        ]),
        np.array([
            [0.18, 0.03, 0.03],
            [0.18, 0.03 + 0.06*s, 0.03],
            [0.18, 0.03, 0.03 + 0.05*s],
        ]),
        np.array([
            [0.18, 0.03, 0.03],
            [0.18, 0.03 + 0.06*s, 0.03],
            [0.18, 0.03, 0.03 + 0.05*s],
        ]),
        # C3(axis=[-1, -1, -1], cen=A([0, 0, 0]) / 8, label='C3_111_1m0_111_8', vizcol=(1, 0, 0)),
        # C2(axis=[1, 0, 0], cen=A([3, 0, 2]) / 8, label='D2_100_0m1_102_8', vizcol=(0, 1, 0)),
        # C2(axis=[1, -1, 0], cen=A([-2.7, 0.7, -1]) / 8, label='D3_111_1m0_mmm_8', vizcol=(0, 0, 1)),
        # yapf: disable
        np.array([
            [0.18, 0.03, 0.03],
            [0.18, 0.03 + 0.06*s, 0.03],
            [0.18, 0.03, 0.03 + 0.05*s],
            [-0.0, 0.2, 0.03],
            [-0.0, 0.2 + 0.06*s, 0.03],
            [-0.0, 0.2, 0.03 + 0.05*s],
            [0.15, -0.0, 0.13],
            [0.15, -0.0 + 0.06*s, 0.13],
            [0.15, -0.0, 0.03 + 0.15*s],
            [0.21, -0.21, -0.0],
            [0.21, -0.21 + 0.06*s, 0.0],
            [0.21, -0.21, 0],
            [0.0, -0.0, 0.2],
            [0.0, 0.0 + 0.06*s, 0.2],
            [0.0, 0.0, 0.2 + 0.05*s],
        ]),
        # yapf: enable
    ]
    # ic(ipd.homog.hxform(ipd.hrot([1, -1, 0], 90, np.array([-2.7, 0.7, -1]) / 8), [0, 0, 0.2]))
    # assert 0
    # showpts[3] = ipd.homog.hxform(ipd.htrans([0.2, 0.1, 0.1]), showpts[3])
    for p in showpts:
        p += pointshift

    radius = np.array([[0.05, 0.03, 0.02] * 10] * len(showpts))
    radius *= pointscale

    colors = np.array([[(1, 1, 1)] * 30] * len(showpts))
    # ic(len(colors[which]))
    return ipd.homog.hpoint(showpts[which]), radius[which], colors[which]

def get_fanrefpoint(xtal):
    pt = [0, 1, 0, 1]
    # yapf: disable
    if xtal.name == 'P 2 3' : pt= [0, 1, 0, 1]
    if xtal.name == 'I 21 3': pt= ipd.homog.hxform(ipd.hrot([0, 0, 1], -30), [0, 1, 0,1])
    # yapf: enable
    # ic(pt)
    return pt
