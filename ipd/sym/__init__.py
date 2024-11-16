from ipd.dev import lazyimport
from ipd.sym.guess_symmetry import *
from ipd.sym.permutations import *
from ipd.sym.sym import *
from ipd.sym.sym_adapt import *
from ipd.sym.sym_check import *
from ipd.sym.sym_color import *
from ipd.sym.sym_factory import *
from ipd.sym.sym_fitting import *
from ipd.sym.sym_kind import *
from ipd.sym.sym_slice import *
from ipd.sym.sym_index import *
from ipd.sym.sym_options import *
from ipd.sym.sym_manager import *
from ipd.sym.ipd_sym_manager import *
from ipd.sym.sym_util import *
from ipd.sym.symframes import *

xtal = lazyimport('ipd.sym.xtal')
high_t = lazyimport('ipd.sym.high_t')
helix = lazyimport('ipd.sym.helix')
sym_tensor = lazyimport('ipd.sym.sym_tensor')

def set_global_symmetry(sym: SymmetryManager):
    global _global_symmetry
    _global_symmetry = sym

def get_global_symmetry() -> SymmetryManager:
    global _global_symmetry
    if _global_symmetry is None:
        raise RuntimeError('Global symmetry not set')
    return _global_symmetry

try:
    import torch  # noqa  # type: ignore
    _global_symmetry = create_sym_manager(symid='C1')
except ImportError:
    _global_symmetry = None
