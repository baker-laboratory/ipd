import contextlib
import os
from typing import TYPE_CHECKING, Any

from ipd.bunch import Bunch as Bunch, bunchify as bunchify
from ipd import dev as dev
from ipd.lazy_import import lazyimport
from ipd.observer import hub as hub

if TYPE_CHECKING:
    from ipd import crud
    from ipd import cuda
    from ipd import fit
    from ipd import h
    from ipd import homog
    from ipd import motif
    from ipd import pdb
    from ipd import qt
    from ipd import samp
    from ipd import protocol
    from ipd import sym
    from ipd import tests
    from ipd import tools
    from ipd import viz
    from ipd import voxel
else:
    crud = lazyimport('ipd.crud')
    cuda = lazyimport('ipd.dev.cuda')
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

def __getattr__(name) -> Any:
    if name == 'symmetrize':
        symgr = sym.get_global_symmetry()
        assert sym is not None
        return symgr
    if name == 'motif_applier':
        mmgr = motif.get_global_motif_manager()
        assert mmgr is not None
        return mmgr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

from ipd.project_config import install_ipd_pre_commit_hook

install_ipd_pre_commit_hook(projdir, '..')
