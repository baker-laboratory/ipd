import os
from ipd.dev import lazyimport
import ipd.observer

dev = lazyimport('ipd.dev')
cuda = lazyimport('ipd.cuda')
fit = lazyimport('ipd.fit')
ppp = lazyimport('ipd.ppp')
samp = lazyimport('ipd.samp')
sieve = lazyimport('ipd.sieve')
sym = lazyimport('ipd.sym')
voxel = lazyimport('ipd.voxel')

icecream = lazyimport('icecream', pip=True).ic.configureOutput(includeContext=True)

proj_dir = os.path.realpath(os.path.dirname(__file__))

from ipd.observer import spy, DynamicParameters
from ipd.dev import Bunch

def testpath(path):
    return os.path.join(proj_dir, 'tests', 'data', path)
