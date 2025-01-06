import collections
import inspect
from typing import Optional
import types

def current_frame() -> types.FrameType:
    frame = inspect.currentframe()
    if frame is None: raise ValueError('frame is None')
    return frame

def frame_parent(frame: Optional[types.FrameType]) -> types.FrameType:
    if frame is None: raise ValueError('frame is None')
    frame = frame.f_back
    if frame is None: raise ValueError('frame is None')
    return frame

CallerInfo = collections.namedtuple('CallerInfo', 'filename lineno code')

def caller_info(excludefiles=None) -> CallerInfo:
    excludefiles = excludefiles or []
    excludefiles.append(__file__)
    frame: types.FrameType = current_frame()
    assert frame is not None
    if excludefiles:
        while frame.f_code.co_filename in excludefiles:
            frame = frame_parent(frame)
    lines, no = inspect.getsourcelines(frame)
    module = inspect.getmodule(frame)
    code = 'unknown source code'
    if module is not None: code = lines[frame.f_lineno - no - 1].strip()
    return CallerInfo(frame.f_code.co_filename, frame.f_lineno, code)
