import collections
from difflib import get_close_matches
import inspect
from typing import Optional
import types


def current_frame() -> types.FrameType:
    frame = inspect.currentframe()
    if frame is None:
        raise ValueError('frame is None')
    return frame


def frame_parent(frame: Optional[types.FrameType]) -> types.FrameType:
    if frame is None:
        raise ValueError('frame is None')
    frame = frame.f_back
    if frame is None:
        raise ValueError('frame is None')
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
    if module is not None:
        code = lines[frame.f_lineno - no - 1].strip()
    return CallerInfo(frame.f_code.co_filename, frame.f_lineno, code)


def find_close_argnames(word, string_list, n=3, cutoff=0.6):
    """Find close matches to a given word from a list of strings.

    Args:
        word (str): The word to find close matches for.
        string_list (list of str): A list of strings to search within.
        n (int): The maximum number of close matches to return.
        cutoff (float): The minimum similarity score (0-1) for a string to be considered a match.

    Returns:
        list: A list of close matches.

    Example:
        >>> find_close_argnames('apple', ['apple', 'ape', 'apply', 'banana'], n=2)
        ['apple', 'apply']
    """
    candidates = get_close_matches(word, string_list, n=n, cutoff=cutoff)
    candidates = filter(lambda s: abs(len(s) - len(word)) < len(word) // 5,
                        candidates)
    return list(candidates)
