import sys
import abc
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    cast as cast,
    Iterator,
    TypeVar,
    Union,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
    MutableSequence,
)
from typing_extensions import ParamSpec  # type: ignore noqa
import numpy as np

KW = dict[str, Any]
"""Type alias for keyword arguments represented as a dictionary with string keys and any type of value."""

FieldSpec = Union[str, list[str], tuple[str], Callable[..., str], tuple]
EnumerIter = Iterator[int]
EnumerListIter = Iterator[list[Any]]

T = TypeVar('T')
R = TypeVar('R')
C = TypeVar('C')
if sys.version_info.minor >= 10 or TYPE_CHECKING:
    P = ParamSpec('P')
    F = Callable[P, R]
else:
    P = TypeVar('P')
    P.args = list[Any]
    P.kwargs = KW
    F = Callable[[Any, ...], R]

def basic_typevars(which) -> list[Union[TypeVar, ParamSpec]]:
    result = [globals()[k] for k in which]
    return result

Vec = Union[np.ndarray, tuple, list]
Point = Union[np.ndarray, tuple, list]

class Frames44Meta(abc.ABCMeta):

    def __instancecheck__(cls, obj: Any) -> bool:
        return isinstance(obj, np.ndarray) and obj.shape[-2:] == (4, 4)

class Frames44(np.ndarray, metaclass=Frames44Meta):
    pass

class FramesN44Meta(abc.ABCMeta):

    def __instancecheck__(cls, obj: Any) -> bool:
        return isinstance(obj, np.ndarray) and len(obj.shape) == 3 and obj.shape[-2:] == (4, 4)

class FramesN44(np.ndarray, metaclass=FramesN44Meta):
    pass

class NDArray_MN2_int32(np.ndarray):
    pass

class NDArray_N2_int32(np.ndarray):
    pass

def isstr(s: Any) -> bool:
    return isinstance(s, str)

def isint(s: Any) -> bool:
    return isinstance(s, int)

def islist(s: Any) -> bool:
    return isinstance(s, list)

def isdict(s: Any) -> bool:
    return isinstance(s, dict)

def isseq(s: Any) -> bool:
    return isinstance(s, Sequence)

def ismap(s: Any) -> bool:
    return isinstance(s, Mapping)

def isseqmut(s: Any) -> bool:
    return isinstance(s, MutableSequence)

def ismapmut(s: Any) -> bool:
    return isinstance(s, MutableMapping)

def isiter(s: Any) -> bool:
    return isinstance(s, Iterable)
