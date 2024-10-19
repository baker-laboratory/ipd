# try:
from willutil.viz.primitives import *
from willutil.viz.viz_helix import *
from willutil.viz.viz_xtal import *
from willutil.viz.viz_rigidbody import *
from willutil.viz.movietools import *
from willutil.viz.pymol_viz import *
from willutil.viz.pymol_cgo import *
from willutil.viz.viz_deco import *
from willutil.viz.viz_bond import *
from willutil.viz.pymol_viz import pymol_load
try:
    pass
except ImportError:
    printed_warning = False

    def showme(*a, **b):
        global printed_warning
        if not printed_warning:
            printed_warning = True
            print("!" * 80)
            print("WARNING willutil.viz.showme not available without pymol")
            print("!" * 80)

from willutil.viz.plot import *
