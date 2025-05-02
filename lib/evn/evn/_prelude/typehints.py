import sys
from typing import    cast as cast

if sys.version_info.minor >= 10:
    import typing as t
    # from typing import t.ParamSpec
else:
    import typing_extensions as t
    # from typing_extensions import t.ParamSpec

KW = dict[str, t.Any]
IOBytes = t.IO[bytes]
IO = t.IO[str]
"""Type alias for keyword arguments represented as a dictionary with string keys and any type of value."""

FieldSpec = t.Union[str, list[str], tuple[str], t.Callable[..., str], tuple]
EnumerIter = t.Iterator[int]
EnumerListIter = t.Iterator[list[t.Any]]

T = t.TypeVar('T')
R = t.TypeVar('R')
C = t.TypeVar('C')
if sys.version_info.minor >= 10 or t.TYPE_CHECKING:
    P = t.ParamSpec('P')
    F = t.Callable[P, R]
else:
    P = t.TypeVar('P')
    P.args = list[t.Any]
    P.kwargs = KW
    F = t.Callable[[t.Any, ...], R]


def basic_typevars(which) -> list[t.Union[t.TypeVar, t.ParamSpec]]:
    result = [globals()[k] for k in which]
    return result

def annotype(typ:type, info) -> t.Annotated:
    return t.Annotated[typ, info]
