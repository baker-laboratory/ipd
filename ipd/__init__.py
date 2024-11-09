import os

from ipd import dev
from ipd.dev import observer as observer
from ipd.dev.lazy_import import lazyimport
from ipd.dev.observer import hub as hub
from ipd.dev.state.bunch import Bunch as Bunch

ci = lazyimport('ipd.ci')
crud = lazyimport('ipd.crud')
cuda = lazyimport('ipd.dev.cuda')
fit = lazyimport('ipd.fit')
h = lazyimport('ipd.homog.thgeom')
homog = lazyimport('ipd.homog')
motif = lazyimport('ipd.motif')
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
from collections import defaultdict, namedtuple
import builtins

ic.configureOutput(includeContext=True)
builtins.ic = ic
builtins.defaultdict = defaultdict
builtins.namedtuple = namedtuple

def showme(*a, **kw):
    from ipd.viz import showme as viz_showme
    viz_showme(*a, **kw)

def __getattr__(name):
    if name == 'symmetrize':
        return sym.get_global_symmetry()
    if name == 'motif_applier':
        return motif.get_global_motif_manager()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

dev.install_pre_commit_hook(projdir, '..')
