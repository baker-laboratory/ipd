import abc
from typing import (Any, Callable, Iterator, TypeVar, Union, TYPE_CHECKING)
from typing_extensions import ParamSpec  # type: ignore noqa
import numpy as np

inchecker = TYPE_CHECKING

KW = dict[str, Any]
"""Type alias for keyword arguments represented as a dictionary with string keys and any type of value."""

FieldSpec = Union[str, list[str], tuple[str], Callable[..., str], tuple]
EnumerIter = Iterator[int]
EnumerListIter = Iterator[list[Any]]

def basic_typevars(which) -> list[Union[TypeVar, ParamSpec]]:
    """
    Generate a set of common type variables used for generic typing.

    This function creates a dictionary of type variables commonly used for generic typing:
    - `T`: A generic type variable.
    - `R`: A return type variable.
    - `C`: A type variable for class types.
    - `P`: A parameter specification (if available; falls back to `TypeVar` if `ParamSpec` is not available).
    - `F`: A callable type with specified parameters (`P`) and a return type (`R`).

    Args:
        which (Iterable[str]): A list of keys specifying which type variables to return.

    Returns:
        Generator: A generator yielding the requested type variables in the order specified in `which`.

    Examples:
        >>> list(basic_typevars(['T', 'R']))
        [TypeVar('T'), TypeVar('R')]
        >>> list(basic_typevars(['C']))
        [<class 'type'>]
    """
    typevars = dict(T=TypeVar('T'), R=TypeVar('R'), C=type[TypeVar('C')])
    try:
        typevars['P'] = ParamSpec('P')  # type: ignore
    except ImportError:
        typevars['P'] = TypeVar('P')  # type: ignore
        typevars['P'].args = list[Any]  # type: ignore
        typevars['P'].kwargs = KW  # type: ignore
    typevars['F'] = Callable[typevars['P'], typevars['R']]  # type: ignore
    return [typevars[k] for k in which]  # type: ignore

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
