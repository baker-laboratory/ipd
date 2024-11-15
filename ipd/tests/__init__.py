import os
import random
import numpy as np
import ipd as ipd
from ipd.tests.maintest import *

th = ipd.lazyimport('torch')
sym = ipd.lazyimport('ipd.tests.sym')

def make_deterministic(seed=0):
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

def path(fname):
    return os.path.realpath(f'{ipd.projdir}/data/tests/{fname}')

def load(fname):
    return ipd.load(path(fname))

def force_pytest_skip(reason):
    import _pytest

    raise _pytest.outcomes.Skipped(reason)  # type: ignore

__all__ = ['sym', 'make_deterministic', 'path', 'load']
