import os
from icecream import ic
from ipd import dev
from ipd.observer import spy

cuda = dev.LazyModule('ipd.cuda')
fit = dev.LazyModule('ipd.fit')
observer = dev.LazyModule('ipd.observer')
samp = dev.LazyModule('ipd.samp')
sieve = dev.LazyModule('ipd.sieve')
voxel = dev.LazyModule('ipd.voxel')

ic.configureOutput(includeContext=True)

proj_dir = os.path.realpath(os.path.dirname(__file__))
