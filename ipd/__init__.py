import os
from icecream import ic
from ipd.dev import Bunch, lazyimport
from ipd import dev, tests
from ipd.observer import hub, DynamicParameters
import ipd.observer

cuda = lazyimport('ipd.cuda')
fit = lazyimport('ipd.fit')
homog = lazyimport('ipd.homog')
samp = lazyimport('ipd.samp')
sieve = lazyimport('ipd.sieve')
sym = lazyimport('ipd.sym')
voxel = lazyimport('ipd.voxel')
h = lazyimport('ipd.homog.thgeom')

ic.configureOutput(includeContext=True)
projdir = os.path.realpath(os.path.dirname(__file__))

def showme(*a, **kw):
    from ipd.viz import showme as viz_showme
    viz_showme(*a, **kw)
