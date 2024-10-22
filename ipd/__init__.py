import os
from ipd.dev import lazyimport
import ipd.dev.observer
from ipd.dev.observer import hub as hub

ci = lazyimport('ipd.ci')
crud = lazyimport('ipd.crud')
cuda = lazyimport('ipd.dev.cuda')
dev = lazyimport('ipd.dev')
fit = lazyimport('ipd.fit')
h = lazyimport('ipd.homog.thgeom')
homog = lazyimport('ipd.homog')
pdb = lazyimport('ipd.pdb')
ppp = lazyimport('ipd.ppp')
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
        return ipd.sym.symmetrize
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

from box import Box
import builtins

builtins.ic = ic
builtins.Box = Box
