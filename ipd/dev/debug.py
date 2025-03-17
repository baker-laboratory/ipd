from contextlib import contextmanager

from icecream import ic as ic

import ipd

stdio = ipd.cherry_pick_import('ipd.dev.contexts.stdio')

ic.configureOutput(includeContext=True)

@contextmanager
def ic_config(*a, **kw):
    keys = 'prefix outputFunction argToStringFunction includeContext contextAbsPath'.split()
    prev_config = {k: getattr(ic, k) for k in keys}
    try:
        ic.configureOutput(*a, **kw)
        yield
    finally:
        ic.configureOutput(**prev_config)

def icq(*args, **kwargs):
    with stdio() and ic_config(includeContext=False):
        ic(*args, **kwargs)
