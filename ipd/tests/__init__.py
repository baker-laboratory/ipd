import os
import random
import willutil as wu
import numpy as np
import torch
import ipd
from ipd.tests import sym

def make_deterministic(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def path(fname):
    return os.path.realpath(f'{ipd.projdir}/tests/data/{fname}')

def load(fname):
    return wu.load(path(fname))

__all__ = ['sym', 'make_deterministic', 'path', 'load']
