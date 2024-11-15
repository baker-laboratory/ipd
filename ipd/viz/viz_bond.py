try:
    import pymol  # noqa  # type: ignore
except ImportError:
    pass

import numpy as np

import ipd

@ipd.viz.pymol_scene
def show_bonds(xyz, bonds, colors=None, **kw):
    cgo = list()
    if not colors: colors = [(0.3, 0.3, 0.3)] * len(xyz)
    assert xyz.ndim == 2
    # ic(np.nonzero(bonds).shape)`
    for i, j in np.nonzero(bonds):
        if i == j: continue
        cgo += ipd.viz.cgo_cyl(xyz[i], xyz[j], 0.1, colors[i], colors[j])

    return ipd.dev.Bunch(cgo=cgo)
