import os
from ipd.dev import lazyimport
import ipd.observer
from icecream import ic
from ipd.dev import Bunch, lazyimport
from ipd import dev, tests
from ipd.observer import hub, DynamicParameters
import ipd.observer

dev = lazyimport('ipd.dev')
crud = lazyimport('ipd.crud')
cuda = lazyimport('ipd.cuda')
fit = lazyimport('ipd.fit')
h = lazyimport('ipd.homog.thgeom')
homog = lazyimport('ipd.homog')
ppp = lazyimport('ipd.ppp')
qt = lazyimport('ipd.qt')
samp = lazyimport('ipd.samp')
sieve = lazyimport('ipd.sieve')
sym = lazyimport('ipd.sym')
voxel = lazyimport('ipd.voxel')

proj_dir = os.path.realpath(os.path.dirname(__file__))
STRUCTURE_FILE_SUFFIX = tuple('.pdb .pdb.gz .cif .bcif'.split())
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
ic.configureOutput(includeContext=True)
projdir = os.path.realpath(os.path.dirname(__file__))

def testpath(path):
    return os.path.join(proj_dir, 'tests', 'data', path)

def showme(*a, **kw):
    from ipd.viz import showme as viz_showme
    viz_showme(*a, **kw)
