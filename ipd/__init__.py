import contextlib
import os
import typing

from ipd.version import __version__ as __version__

from ipd.typehints import *  #noqa
from ipd.dev.error import panic as panic
from ipd.dev.meta import kwcheck as kwcheck, kwcall as kwcall, kwcurry as kwcurry
from ipd.dev.format import print_table as print_table, print as print
from ipd.bunch import Bunch as Bunch, bunchify as bunchify
from ipd.lazy_import import importornone as importornone, lazyimport as lazyimport
from ipd.lazy_import import LazyImportError as LazyImportError
from ipd.observer import hub as hub
from ipd.dev.tolerances import Tolerances as Tolerances
from ipd.dev.iterables import first as first

from ipd import dev as dev, typehints as typehints

if typing.TYPE_CHECKING:
    from ipd import atom
    from ipd import crud
    from ipd import cuda
    # from ipd import fit
    import ipd.homog.thgeom as h
    # import ipd.homog.thgeom as homog
    from ipd import motif
    from ipd import pdb
    from ipd import protocol
    from ipd import qt
    # from ipd import samp
    from ipd import sel
    from ipd import sym
    from ipd import tests
    from ipd import tools
    from ipd import viz
    # from ipd import voxel
else:
    atom = lazyimport('ipd.atom')
    crud = lazyimport('ipd.crud')
    cuda = lazyimport('ipd.dev.cuda')
    # fit = lazyimport('ipd.fit')
    h = lazyimport('ipd.homog.thgeom')
    homog = lazyimport('ipd.homog')
    hnumpy = lazyimport('ipd.homog.hgeom')
    htorch = lazyimport('ipd.homog.thgeom')
    motif = lazyimport('ipd.motif')
    pdb = lazyimport('ipd.pdb')
    protocol = lazyimport('ipd.protocol')
    qt = lazyimport('ipd.dev.qt')
    # samp = lazyimport('ipd.samp')
    sel = lazyimport('ipd.sel')
    sym = lazyimport('ipd.sym')
    tests = lazyimport('ipd.tests')
    tools = lazyimport('ipd.tools')
    viz = lazyimport('ipd.viz')
    # voxel = lazyimport('ipd.voxel')

proj_dir = os.path.realpath(os.path.dirname(__file__))
STRUCTURE_FILE_SUFFIX = tuple('.pdb .pdb.gz .cif .bcif'.split())
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
projdir = os.path.realpath(os.path.dirname(__file__))

with contextlib.suppress(ImportError):
    from icecream import ic
    import builtins

    setattr(builtins, 'panic', panic)
    ic.configureOutput(includeContext=True)
    setattr(builtins, 'ic', ic)

def showme(*a, **kw):
    from ipd.viz import showme as viz_showme
    viz_showme(*a, **kw)

def __getattr__(name) -> typing.Any:
    if name == 'symmetrize':
        symgr = sym.get_global_symmetry()
        assert sym is not None
        return symgr
    elif name == 'motif_applier':
        mmgr = motif.get_global_motif_manager()
        assert mmgr is not None
        return mmgr
    elif name.startswith('debug'):
        return getattr(hub, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

from ipd.project_config import install_ipd_pre_commit_hook

install_ipd_pre_commit_hook(projdir, '..')
