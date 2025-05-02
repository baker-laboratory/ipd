import inspect
import types
import functools
import typing as t

@functools.total_ordering
class Missing:
    __slots__ = ()

    def __repr__(self):
        return 'NA'

    def __eq__(self, other):
        return isinstance(other, Missing)

    def __lt__(self, other):
        return True

NA = Missing()

def NoOp(*a, **kw):
    None

def is_free_function(obj: t.Any) -> bool:
    return isinstance(obj, types.FunctionType)

def is_bound_method(obj: t.Any) -> bool:
    return inspect.ismethod(obj) and obj.__self__ is not None

def is_unbound_method(obj: t.Any) -> bool:
    return inspect.isfunction(obj) and hasattr(obj, '__qualname__') and '.' in obj.__qualname__

def is_member_function(obj: t.Any) -> bool:
    return is_bound_method(obj) or is_unbound_method(obj)

def is_function(obj: t.Any) -> bool:
    return is_member_function(obj) or is_free_function(obj)

def is_generator(obj: t.Any) -> bool:
    return isinstance(obj, types.GeneratorType)

def isstr(s: t.Any) -> bool:
    return isinstance(s, str)

def isint(s: t.Any) -> bool:
    return isinstance(s, int)

def islist(s: t.Any) -> bool:
    return isinstance(s, list)

def isdict(s: t.Any) -> bool:
    return isinstance(s, dict)

def isseq(s: t.Any) -> bool:
    return isinstance(s, t.Sequence)

def ismap(s: t.Any) -> bool:
    return isinstance(s, t.Mapping)

def isseqmut(s: t.Any) -> bool:
    return isinstance(s, t.MutableSequence)

def ismapmut(s: t.Any) -> bool:
    return isinstance(s, t.MutableMapping)

def isiter(s: t.Any) -> bool:
    return isinstance(s, t.Iterable)
