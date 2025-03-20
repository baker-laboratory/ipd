import types
from contextlib import contextmanager
from icecream import IceCreamDebugger, ic
import inspect
import os

import ipd

stdio = ipd.cherry_pick_import('ipd.dev.contexts.stdio')

def _getContextOneBack(self, callFrame, callNode=None):
    callFrame = callFrame.f_back
    frameInfo = inspect.getframeinfo(callFrame)
    lineNumber = frameInfo.lineno
    parentFunction = frameInfo.function
    filepath = (os.path.realpath if self.contextAbsPath else os.path.basename)(frameInfo.filename)
    return filepath, lineNumber, parentFunction

ic_one_frame_back = IceCreamDebugger()
ic_one_frame_back.configureOutput(includeContext=True)
newmember = types.MethodType(_getContextOneBack, ic_one_frame_back)

setattr(ic_one_frame_back, '_getContext', newmember)

@contextmanager
def ic_config(iclocal=ic, *a, **kw):
    keys = 'prefix outputFunction argToStringFunction includeContext contextAbsPath'.split()
    prev_config = {k: getattr(iclocal, k) for k in keys}
    try:
        iclocal.configureOutput(*a, **kw)
        yield
    finally:
        iclocal.configureOutput(**prev_config)

def icm(*args, **kwargs):
    with stdio(), ic_config(ic_one_frame_back, includeContext=False):
        ic_one_frame_back(*args, **kwargs)

def icv(*args, **kwargs):
    with stdio(), ic_config(ic_one_frame_back, includeContext=True):
        ic_one_frame_back(*args, **kwargs)

__all__ = ['ic', 'ic_config', 'icm', 'icv']
