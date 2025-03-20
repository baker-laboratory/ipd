from time import perf_counter

_start, _timings = perf_counter(), dict()

def _checkpoint(name):
    global _start, _timings
    _timings[name] = [perf_counter() - _start]
    _start = perf_counter()

import contextlib
import os
import typing
import dataclasses as dc  # noqa
import functools as ft  # noqa
import itertools as it  # noqa
import numpy as np  # noqa
from pathlib import Path as Path
from copy import copy as copy, deepcopy as deepcopy
from typing import (TYPE_CHECKING as TYPE_CHECKING, Any as Any, Callable as Callable, cast as cast, Iterator as
                    Iterator, TypeVar as TypeVar, Union as Union, Iterable as Iterable, Mapping as Mapping,
                    MutableMapping as MutableMapping, Sequence as Sequence, MutableSequence as MutableSequence,
                    Optional as Optional, ParamSpec as ParamSpec)

_checkpoint('ipd basic imports')
STRUCTURE_FILE_SUFFIX = tuple('.pdb .pdb.gz .cif .bcif'.split())
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
projdir = os.path.realpath(os.path.dirname(__file__))
from ipd._prelude.version import __version__ as __version__
from ipd._prelude.wraps import wraps as wraps
from ipd._prelude.chrono import Chrono as Chrono, chrono as chrono, checkpoint as checkpoint
from ipd._prelude.import_util import (is_installed as is_installed, not_installed as not_installed,
                                      cherry_pick_import as cherry_pick_import, cherry_pick_imports as
                                      cherry_pick_imports)
from ipd._prelude.lazy_import import (lazyimport as lazyimport, lazyimports as lazyimports, maybeimport as
                                      maybeimport, maybeimports as maybeimports, LazyImportError as
                                      LazyImportError)
from ipd._prelude.structs import struct as struct, mutablestruct as mutablestruct, field as field
from ipd._prelude.typehints import (KW as KW, FieldSpec as FieldSpec, EnumerIter as EnumerIter, EnumerListIter
                                    as EnumerListIter, T as T, R as R, C as C, P as P, F as F, basic_typevars
                                    as basic_typevars, Frames44Meta as Frames44Meta, Frames44 as Frames44,
                                    FramesN44Meta as FramesN44Meta, FramesN44 as FramesN44, NDArray_MN2_int32
                                    as NDArray_MN2_int32, NDArray_N2_int32 as NDArray_N2_int32, isstr as isstr,
                                    isint as isint, islist as islist, isdict as isdict, isseq as isseq, ismap
                                    as ismap, isseqmut as isseqmut, ismapmut as ismapmut, isiter as isiter)

optional_imports = cherry_pick_import('ipd.dev.contexts.optional_imports')
capture_stdio = cherry_pick_import('ipd.dev.contexts.capture_stdio')
ic, icm, icv = cherry_pick_imports('ipd.dev.debug', 'ic icm icv')
timed = cherry_pick_import('ipd.dev.instrumentation.timer.timed')
_global_chrono = None
_checkpoint('ipd prelude imports')

def __getattr__(name):
    global _global_chrono
    if name == 'symmetrize':
        return sym.get_global_symmetry()
    elif name == 'motif_applier':
        return motif.get_global_motif_manager()
    elif name == 'global_chrono':
        _global_chrono = _global_chrono or Chrono(checkpoints=_timings)
        return _global_chrono
    elif name.startswith('debug'):
        return getattr(hub, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

_checkpoint('ipd globals')

from ipd.dev.error import panic as panic
from ipd.dev.meta import kwcheck as kwcheck, kwcall as kwcall, kwcurry as kwcurry
from ipd.dev.metadata import get_metadata as get_metadata, set_metadata as set_metadata
from ipd.dev.functional import map as map, visit as visit
from ipd.dev.format import print_table as print_table, print as print
from ipd.bunch import Bunch as Bunch, bunchify as bunchify
from ipd.observer import hub as hub
from ipd.dev.tolerances import Tolerances as Tolerances
from ipd.dev.iterables import first as first
from ipd.dev.contexts import stdio as stdio, catch_em_all as catch_em_all

_checkpoint('ipd from subpackage imports')
from ipd import dev as dev, homog as homog

if typing.TYPE_CHECKING:
    from ipd import crud
    import ipd.homog.thgeom as htorch
    import ipd.homog.hgeom as hnumpy
    from ipd import pdb
    from ipd import protocol
    from ipd import sel
    from ipd import sym
    from ipd import motif
    from ipd import atom
    from ipd import tools
    from ipd import tests
    # from ipd import fit
    # from ipd import samp
    # from ipd import voxel
else:
    atom = lazyimport('ipd.atom')
    crud = lazyimport('ipd.crud')
    cuda = lazyimport('ipd.dev.cuda')
    h = lazyimport('ipd.homog.thgeom')
    hnumpy = lazyimport('ipd.homog.hgeom')
    htorch = lazyimport('ipd.homog.thgeom')
    motif = lazyimport('ipd.motif')
    pdb = lazyimport('ipd.pdb')
    protocol = lazyimport('ipd.protocol')
    qt = lazyimport('ipd.dev.qt')
    sel = lazyimport('ipd.sel')
    sym = lazyimport('ipd.sym')
    tests = lazyimport('ipd.tests')
    tools = lazyimport('ipd.tools')
    # fit = lazyimport('ipd.fit')
    # samp = lazyimport('>ipd.samp')
    # voxel = lazyimport('ipd.voxel')
viz = lazyimport('ipd.viz')
_checkpoint('ipd subpackage imports')

with contextlib.suppress(ImportError):
    import builtins
    setattr(builtins, 'ic', ic)

def showme(*a, **kw):
    from ipd.viz import showme as viz_showme
    viz_showme(*a, **kw)

# from ipd.project_config import install_ipd_pre_commit_hook
# install_ipd_pre_commit_hook(projdir, '..')
# _checkpoint('ipd pre commit hook')

if _global_chrono: _global_chrono.checkpoints.update(_timings)
else: _global_chrono = Chrono(checkpoints=_timings)
dev.global_timer.checkpoints.update(_timings)
