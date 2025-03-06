import inspect
from ipd.dev.funcs import iterize_on_first_param

@iterize_on_first_param(asdict=True)
def locals(name, idx=None):
    val = inspect.currentframe().f_back.f_back.f_locals[name]
    if idx is None: return val
    return val[idx]
