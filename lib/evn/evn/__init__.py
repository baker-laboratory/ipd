# __version__ = '0.7.1'

from time import perf_counter
from pathlib import Path as Path

_start, _timings = perf_counter(), {}

pkgroot = Path(__file__).parent.absolute()
projroot = pkgroot.parent
evn_installed = False
projconf = projroot / 'pyproject.toml'
if not (projroot / 'pyproject.toml').exists:
    projroot = None
    evn_installed = True
show_trace = False
chronometer: 'evn._prelude.chrono.Chrono' = None  #type:ignore #noqa

def evn_init_checkpoint(name):
    global _start, _timings
    _timings[name] = [perf_counter() - _start]
    _start = perf_counter()

# import os
import typing as t  # noqa
import dataclasses as dc  # noqa
import functools as ft  # noqa
import itertools as it  # noqa
import contextlib as cl  # noqa
import os as os  # noqa
import sys as sys  # noqa
from copy import copy as copy, deepcopy as deepcopy
from multipledispatch import dispatch as dispatch
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
    Any as Any,
    Callable as Callable,
    cast as cast,
    Iterator as Iterator,
    IO as IO,
    TypeVar as TypeVar,
    Union as Union,
    Iterable as Iterable,
    Mapping as Mapping,
    MutableMapping as MutableMapping,
    Sequence as Sequence,
    MutableSequence as MutableSequence,
    Optional as Optional,
)
from ninja_import import ninja_import as ninja_import
from icecream import ic as ic

ic.configureOutput(includeContext=True)
import builtins

builtins.ic = ic  # make ic available globally

evn_init_checkpoint('INIT evn basic imports')
from evn._prelude.basic_types import (
    NA as NA,
    NoOp as NoOp,
    isstr as isstr,
    isint as isint,
    islist as islist,
    isdict as isdict,
    isseq as isseq,
    ismap as ismap,
    isseqmut as isseqmut,
    ismapmut as ismapmut,
    isiter as isiter,
    is_free_function as is_free_function,
    is_bound_method as is_bound_method,
    is_unbound_method as is_unbound_method,
    is_member_function as is_member_function,
    is_function as is_function,
    is_generator as is_generator,
)
from evn._prelude.make_decorator import make_decorator as make_decorator
from evn._prelude.import_util import (
    is_installed as is_installed,
    not_installed as not_installed,
)
from evn._prelude.lazy_import import (
    lazyimport as lazyimport,
    lazyimports as lazyimports,
    maybeimport as maybeimport,
    maybeimports as maybeimports,
    LazyImportError as LazyImportError,
)
from evn._prelude.lazy_dispatch import lazydispatch as lazydispatch
from evn._prelude.structs import (
    struct as struct,
    mutablestruct as mutablestruct,
    basestruct as basestruct,
    field as field,
    asdict as asdict,
)
from evn._prelude.typehints import (
    KW as KW,
    T as T,
    R as R,
    C as C,
    P as P,
    F as F,
    FieldSpec as FieldSpec,
    EnumerIter as EnumerIter,
    EnumerListIter as EnumerListIter,
    basic_typevars as basic_typevars,
    annotype as annotype,
)

from evn._prelude.chrono import (
    Chrono as Chrono,
    chrono as chrono,
    chrono_enter_scope as chrono_enter_scope,
    chrono_exit_scope as chrono_exit_scope,
    chrono_checkpoint as chrono_checkpoint
)
from evn.decofunc import (
    iterize_on_first_param as iterize_on_first_param,
    iterize_on_first_param_path as iterize_on_first_param_path,
    is_iterizeable as is_iterizeable,
    safe_lru_cache as safe_lru_cache,
)
from evn.decon import (
    item_wise_operations as item_wise_operations,
    subscriptable_for_attributes as subscriptable_for_attributes,
)
from evn.decon.iterables import (
    first as first,
    nth as nth,
    head as head,
    order as order,
    reorder as reorder,
    reorder_inplace as reorder_inplace,
    zipenum as zipenum,
    subsetenum as subsetenum,
    zipmaps as zipmaps,
    zipitems as zipitems,
    addreduce as addreduce,  # type: ignore
    orreduce as orreduce,  # type: ignore
    andreduce as andreduce,  # type: ignore
    mulreduce as mulreduce,  # type: ignore
)

# from evn.error import panic as panic
# from evn.meta import kwcheck as kwcheck, kwcall as kwcall, kwcurry as kwcurry
# from evn.metadata import get_metadata as get_metadata, set_metadata as set_metadata
# from evn.functional import map as map, visit as visit
# from evn.format import print_table as print_table, print as print
from evn.decon.bunch import Bunch as Bunch, bunchify as bunchify, unbunchify as unbunchify

# from evn.observer import hub as hub
# from evn.tolerances import Tolerances as Tolerances
# from evn.iterables import first as first
# from evn.contexts import force_stdio as force_stdio
from evn.meta import kwcall as kwcall, kwcheck as kwcheck
from evn.print import make_table as make_table
from evn.cli import CLI as CLI

installed = Bunch(_default=is_installed, _frozen=True)

from evn.dev.contexts import (
    capture_asserts as capture_asserts,
    capture_stdio as capture_stdio,
    catch_em_all as catch_em_all,
    cd as cd,
    cd_project_root as cd_project_root,
    force_stdio as force_stdio,
    just_stdout as just_stdout,
    nocontext as nocontext,
    redirect as redirect,
    set_class as set_class,
    trace_writes_to_stdout as trace_writes_to_stdout,
    np_printopts as np_printopts,
    np_compact as np_compact,
)
from evn._prelude.inspect import (
    inspect as inspect,
    show as show,
    diff as diff,
    summary as summary,
    trace as trace,
)

import evn.ident as ident

if TYPE_CHECKING:
    from evn import (
        config as config,
        cli as cli,
        dev as dev,
        decofunc as decofunc,
        decon as decon,
        doc as doc,
        format as format,
        meta as meta,
        testing as testing,
        tree as tree,
        tool as tool,
    )
else:
    config = lazyimport('evn.config')
    cli = lazyimport('evn.cli')
    dev = lazyimport('evn.dev')
    decofunc = lazyimport('evn.decofunc')
    decon = lazyimport('evn.decon')
    doc = lazyimport('evn.doc')
    format = lazyimport('evn.format')
    meta = lazyimport('evn.meta')
    testing = lazyimport('evn.testing')
    tree = lazyimport('evn.tree')
    tool = lazyimport('evn.tool')

# optional_imports = ninja_import('evn.contexts.optional_imports')
# capture_stdio = ninja_import('evn.contexts.capture_stdio')
# ic, icm, icv = ninja_imports('evn.debug', 'ic icm icv')
# timed = ninja_import('evn.instrumentation.timer.timed')
# item_wise_operations = ninja_import('evn.item_wise.item_wise_operations')
# subscriptable_for_attributes = ninja_import('evn.decorators.subscriptable_for_attributes')
# iterize_on_first_param = ninja_import('evn.decorators.iterize_on_first_param')

# _global_chrono = None

# evn_init_checkpoint('INIT evn prelude imports')

# evn_init_checkpoint('INIT evn from subpackage imports')
# from evn import dev as dev, homog as homog

# if typing.TYPE_CHECKING:
#     from evn import crud

#     import evn.homog.thgeom as htorch
#     import evn.homog.hgeom as hnumpy
#     from evn import pdb
#     from evn import protocol
#     from evn import sel
#     from evn import sym
#     from evn import ml
#     from evn import motif
#     from evn import atom
#     from evn import tools
#     from evn import tests
#     # from evn import fit
#     # from evn import samp
#     # from evn import voxel
# else:
#     atom = lazyimport('evn.atom')
#     crud = lazyimport('evn.crud')
#     cuda = lazyimport('evn.cuda')
#     h = lazyimport('evn.homog.thgeom')
#     hnumpy = lazyimport('evn.homog.hgeom')
#     htorch = lazyimport('evn.homog.thgeom')
#     ml = lazyimport('evn.ml')
#     motif = lazyimport('evn.motif')
#     pdb = lazyimport('evn.pdb')
#     protocol = lazyimport('evn.protocol')
#     qt = lazyimport('evn.qt')
#     sel = lazyimport('evn.sel')
#     sym = lazyimport('evn.sym')
#     tests = lazyimport('evn.tests')
#     tools = lazyimport('evn.tools')
#     # fit = lazyimport('evn.fit')
#     # samp = lazyimport('>evn.samp')
#     # voxel = lazyimport('evn.voxel')
# viz = lazyimport('evn.viz')
# evn_init_checkpoint('INIT evn subpackage imports')

# with contextlib.suppress(ImportError):
#     import builtins

#     setattr(builtins, 'ic', ic)

# def showme(*a, **kw):
#     if all(homog.viz.can_showme(a, **kw)):
#         return homog.viz.showme(a, **kw)
#     from evn.viz import showme as viz_showme

#     viz_showme(*a, **kw)

# # from evn.project_config import install_evn_pre_commit_hook
# # install_evn_pre_commit_hook(projdir, '..')
# # evn_init_checkpoint('INIT evn pre commit hook')

# if _global_chrono: _global_chrono.checkpoints.update(_timings)
# else: _global_chrono = Chrono(checkpoints=_timings)
# dev.global_chrono.checkpoints.update(_timings)

# caching_enabled = True
