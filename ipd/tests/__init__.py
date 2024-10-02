import os
import random
import willutil as wu
import numpy as np
import ipd

th = ipd.lazyimport('torch')

def make_deterministic(seed=0):
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

def path(fname):
    return os.path.realpath(f'{ipd.proj_dir}/tests/data/{fname}')

def load(fname):
    return wu.load(path(fname))

__all__ = ['sym', 'make_deterministic', 'path', 'load']
