try:
    import pymol  # noqa
except ImportError:
    pass

import numpy as np
import willutil as wu

@wu.viz.pymol_frame
def show_bonds(xyz, bonds, colors=None, **kw):
    cgo = list()
    if not colors: colors = [(0.3, 0.3, 0.3)] * len(xyz)
    assert xyz.ndim == 2
    # ic(np.nonzero(bonds).shape)`
    for i, j in np.nonzero(bonds):
        if i == j: continue
        cgo += wu.viz.cgo_cyl(xyz[i], xyz[j], 0.1, colors[i], colors[j])

    return wu.Bunch(cgo=cgo)
