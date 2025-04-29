import torch as th  # type: ignore
if th.cuda.device_count() == 0:
    raise ImportError('cant import cudafunc with no cuda devices')

from ipd.cuda.cudabuild import *
from ipd.cuda.cudafunc import *
