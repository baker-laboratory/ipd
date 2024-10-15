import os
from icecream import ic
from ipd import dev, tests
from ipd.observer import spy, DynamicParameters
import ipd.observer

cuda = dev.LazyModule('ipd.cuda')
fit = dev.LazyModule('ipd.fit')
samp = dev.LazyModule('ipd.samp')
sieve = dev.LazyModule('ipd.sieve')
sym = dev.LazyModule('ipd.sym')
voxel = dev.LazyModule('ipd.voxel')

ic.configureOutput(includeContext=True)
projdir = os.path.realpath(os.path.dirname(__file__))

def set_symmetry(sym):
    global symmetrize
    symmetrize = sym

from ipd.sym.sym_manager import create_sym_manager
set_symmetry(create_sym_manager())
