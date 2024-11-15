import numpy as np

import ipd
from ipd.sym.helix import Helix
from ipd.viz.pymol_viz import pymol_load

@pymol_load.register(Helix)  # type: ignore
def pymol_viz_Helix(
    helix,
    name="Helix",
    state=None,
    radius=20,
    spacing=10,
    phase=0,
    points=[],
    splitobjs=False,
    addtocgo=None,
    make_cgo_only=False,
    coils=2,
    **kw,
):
    import pymol  # type: ignore
    state["seenit"][name] += 1  # type: ignore
    v = pymol.cmd.get_view()
    allcgo = list()

    frames = helix.frames(radius, spacing, coils, **kw)
    s = 1
    showpts = np.array([
        [0.28, 0.13, 0.13],
        [0.28, 0.13 + 0.06*s, 0.13],
        [0.28, 0.13, 0.13 + 0.05*s],
    ])
    r = [1.0]
    c = [(1, 1, 1)]
    cgo = ipd.viz.cgo_frame_points(frames, scale=1, showpts=(showpts, r, c), **kw)

    if splitobjs:
        pymol.cmd.load_cgo(cgo, f"{name}_GENPTS{i}")  # type: ignore
    allcgo += cgo

    if addtocgo is None:
        pymol.cmd.load_cgo(allcgo, f'{name}_{state["seenit"][name]}')  # type: ignore
        pymol.cmd.set_view(v)
    else:
        addtocgo.extend(allcgo)
    if make_cgo_only:
        return allcgo

    return None
