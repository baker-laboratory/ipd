import os
from icecream import ic
from ipd import dev

cuda = dev.LazyModule('ipd.cuda')
fit = dev.LazyModule('ipd.fit')
import ipd.observer
samp = dev.LazyModule('ipd.samp')
sieve = dev.LazyModule('ipd.sieve')
sym = dev.LazyModule('ipd.sym')
voxel = dev.LazyModule('ipd.voxel')

ic.configureOutput(includeContext=True)

proj_dir = os.path.realpath(os.path.dirname(__file__))

from ipd.observer import spy, DynamicParameters
