import torch as th  # type: ignore
if th.cuda.device_count() == 0:
    raise ImportError('cant import cudafunc with no cuda devices')

from ipd.dev.cuda.cudabuild import *
from ipd.dev.cuda.cudafunc import *
