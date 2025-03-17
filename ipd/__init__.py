import contextlib
import os
import typing

import dataclasses as dc  # noqa
import functools as ft  # noqa
import itertools as it  # noqa
import numpy as np  # noqa
from pathlib import Path as Path
from copy import copy as copy, deepcopy as deepcopy

# from ipd._prelude import lazyimport, importornone, LazyImportError, cherry_pick_import, cherry_pick_imports
from ipd._prelude import *

_global_chrono = None

def __getattr__(attr):
    if attr == 'global_chrono':
        global _global_chrono
        _global_chrono = _global_chrono or Chrono()
        return _global_chrono
    if attr == 'symmetrize':
        symgr = sym.get_global_symmetry()
        assert sym is not None
        return symgr
    if attr == 'motif_applier':
        mmgr = motif.get_global_motif_manager()
        assert mmgr is not None
        return mmgr
    if attr.startswith('debug'):
        return getattr(hub, attr)
    raise AttributeError(f"module '{__name__}' has no attribute '{attr}'")

optional_imports = cherry_pick_import('ipd.dev.contexts.optional_imports')
capture_stdio = cherry_pick_import('ipd.dev.contexts.capture_stdio')
# ic from icecream, icm is 'mute' and icv is 'verbose', both hacked to bypass stdio redirection
ic, icm, icv = cherry_pick_imports('ipd.dev.debug', 'ic icm icv')

from ipd.dev.error import panic as panic
from ipd.dev.meta import kwcheck as kwcheck, kwcall as kwcall, kwcurry as kwcurry
from ipd.dev.metadata import get_metadata as get_metadata, set_metadata as set_metadata
from ipd.dev.functional import map as map, visit as visit
from ipd.dev.format import print_table as print_table, print as print
from ipd.bunch import Bunch as Bunch, bunchify as bunchify
from ipd.observer import hub as hub
from ipd.dev.tolerances import Tolerances as Tolerances
from ipd.dev.iterables import first as first
from ipd.dev.contexts import stdio as stdio

from ipd import dev as dev

if typing.TYPE_CHECKING:
    from ipd import crud
    # from ipd import fit
    import ipd.homog.thgeom as htorch
    import ipd.homog.hgeom as hnumpy
    from ipd import pdb
    from ipd import protocol
    # from ipd import samp
    from ipd import sel
    from ipd import sym
    from ipd import motif
    from ipd import atom
    # from ipd import voxel
    from ipd import tools
    from ipd import tests
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
    # voxel = lazyimport('ipd.voxel')
viz = lazyimport('ipd.viz')

proj_dir = os.path.realpath(os.path.dirname(__file__))
STRUCTURE_FILE_SUFFIX = tuple('.pdb .pdb.gz .cif .bcif'.split())
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
projdir = os.path.realpath(os.path.dirname(__file__))

with contextlib.suppress(ImportError):
    import builtins
    setattr(builtins, 'ic', ic)

def showme(*a, **kw):
    from ipd.viz import showme as viz_showme
    viz_showme(*a, **kw)

from ipd.project_config import install_ipd_pre_commit_hook

install_ipd_pre_commit_hook(projdir, '..')
