from ipd.dev import LazyModule
from ipd.sym.sym_kind import *
from ipd.sym.sym_util import *
from ipd.sym.sym_color import *
from ipd.sym.sym_slice import *
from ipd.sym.sym_options import *
from ipd.sym.sym_adapt import *
from ipd.sym.sym_check import *
from ipd.sym.sym_factory import *
from ipd.sym.sym_fitting import *
from ipd.sym.sym_manager import *
from ipd.sym.ipd_sym_manager import *
from ipd.sym.guess_symmetry import *

sym_tensor = LazyModule('ipd.sym.sym_tensor')

def set_symmetry(sym):
    global symmetrize
    symmetrize = sym
