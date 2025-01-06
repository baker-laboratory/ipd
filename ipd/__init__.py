import contextlib
import os
from typing import TYPE_CHECKING

from ipd.lazy_import import lazyimport
from ipd.bunch import *
from ipd import observer as observer
from ipd.observer import hub as hub

if TYPE_CHECKING:
    from ipd import crud, cuda, dev, fit, h, homog, motif, pdb, qt, samp, protocol, sym, tests, tools, viz, voxel
else:
    crud = lazyimport('ipd.crud')
    cuda = lazyimport('ipd.dev.cuda')
    dev = lazyimport('ipd.dev')
    fit = lazyimport('ipd.fit')
    h = lazyimport('ipd.homog.thgeom')
    homog = lazyimport('ipd.homog')
    motif = lazyimport('ipd.motif')
    pdb = lazyimport('ipd.pdb')
    qt = lazyimport('ipd.dev.qt')
    samp = lazyimport('ipd.samp')
    protocol = lazyimport('ipd.protocol')
    sym = lazyimport('ipd.sym')
    tests = lazyimport('ipd.tests')
    tools = lazyimport('ipd.tools')
    viz = lazyimport('ipd.viz')
    voxel = lazyimport('ipd.voxel')

proj_dir = os.path.realpath(os.path.dirname(__file__))
STRUCTURE_FILE_SUFFIX = tuple('.pdb .pdb.gz .cif .bcif'.split())
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
projdir = os.path.realpath(os.path.dirname(__file__))

with contextlib.suppress(ImportError):
    from icecream import ic
    import builtins

    ic.configureOutput(includeContext=True)
    setattr(builtins, 'ic', ic)

def showme(*a, **kw):
    from ipd.viz import showme as viz_showme
    viz_showme(*a, **kw)

def __getattr__(name):
    if name == 'symmetrize':
        return sym.get_global_symmetry()
    if name == 'motif_applier':
        return motif.get_global_motif_manager()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

from ipd.project_config import install_ipd_pre_commit_hook

install_ipd_pre_commit_hook(projdir, '..')
