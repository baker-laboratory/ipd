import numpy as np
import ipd
from ipd.lazy_import import lazyimport

th = lazyimport('torch')

def splitlastdim(x, i):
    return x[..., :i], x[..., i:]

def get_tensor_libraries_for(x):
    h, npth = (ipd.homog.hgeom, np) if isinstance(x, np.ndarray) else (ipd.h, th)
    return h, npth
