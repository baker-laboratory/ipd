import os

from ipd.dev import observer as observer
from ipd.dev.lazy_import import lazyimport
from ipd.dev.observer import hub as hub
from ipd.dev.state.bunch import Bunch as Bunch

ci = lazyimport('ipd.ci')
crud = lazyimport('ipd.crud')
cuda = lazyimport('ipd.dev.cuda')
dev = lazyimport('ipd.dev')
fit = lazyimport('ipd.fit')
h = lazyimport('ipd.homog.thgeom')
homog = lazyimport('ipd.homog')
pdb = lazyimport('ipd.pdb')
qt = lazyimport('ipd.dev.qt')
samp = lazyimport('ipd.samp')
sieve = lazyimport('ipd.sieve')
sym = lazyimport('ipd.sym')
tests = lazyimport('ipd.tests')
tools = lazyimport('ipd.tools')
viz = lazyimport('ipd.viz')
voxel = lazyimport('ipd.voxel')

proj_dir = os.path.realpath(os.path.dirname(__file__))
STRUCTURE_FILE_SUFFIX = tuple('.pdb .pdb.gz .cif .bcif'.split())
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
projdir = os.path.realpath(os.path.dirname(__file__))

from icecream import ic

ic.configureOutput(includeContext=True)

def showme(*a, **kw):
    from ipd.viz import showme as viz_showme
    viz_showme(*a, **kw)

def __getattr__(name):
    if name == 'symmetrize':
        return sym.get_global_symmetry()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

import builtins

builtins.ic = ic
