from typing import TYPE_CHECKING, Optional
from ipd.sym.guess_symmetry import *
from ipd.sym.permutations import *
from ipd.sym.sym import *
from ipd.sym.sym_kind import *
from ipd.sym.sym_check import *
from ipd.sym.sym_color import *
from ipd.sym.sym_factory import *
from ipd.sym.sym_detect import *
from ipd.sym.sym_fitting import *
from ipd.sym.sym_slice import *
from ipd.sym.sym_index import *
from ipd.sym.sym_options import *
from ipd.sym.sym_manager import *
from ipd.sym.ipd_sym_manager import *
from ipd.sym.sym_util import *
from ipd.sym.symframes import *
from ipd.sym.sym_builder import *

if False and TYPE_CHECKING:
    from ipd.sym import xtal
    from ipd.sym import high_t
    from ipd.sym import helix
    from ipd.sym import sym_adapt
    # from ipd.sym import sym_tensor
else:
    from ipd.dev import lazyimport
    xtal = lazyimport('ipd.sym.xtal')
    high_t = lazyimport('ipd.sym.high_t')
    helix = lazyimport('ipd.sym.helix')
    sym_adapt = lazyimport('ipd.sym.sym_adapt')
    # sym_tensor = lazyimport('ipd.sym.sym_tensor')

# class IdentityFuncs:

#     def __call__(self, x):
#         return x

#     def __getattr__(self, name: str):
#         return lambda x: x

#     def __bool__(self):
#         return False

def set_global_symmetry(sym: 'SymmetryManager'):
    global _global_symmetry
    _global_symmetry = sym

def get_global_symmetry() -> Optional['SymmetryManager']:
    global _global_symmetry
    # if not _global_symmetry:
    # raise RuntimeError("SymmetryManager not set. Please create a sym mamager first.")
    return _global_symmetry

_global_symmetry = None
