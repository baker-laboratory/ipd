import functools
import inspect
from pathlib import Path
from typing import TypeVar, Callable

T = TypeVar('T')

def call_with_args_from(argpool, func: Callable[..., T]) -> T:
    params = inspect.signature(func).parameters
    for p in params:
        if p not in argpool:
            raise ValueError(
                f'function: {func.__name__}{inspect.signature(func)} requred arg {p} not argpool: {list(argpool.keys())}')
    args = {p: argpool[p] for p in params}
    return func(**args)

class InfixOperator:

    def __init__(self, func, *a, **kw):
        self.func, self.kw, self.a = func, kw, a

    def __ror__(self, lhs, **kw):
        return InfixOperator(lambda rhs: self.func(lhs, rhs, *self.a, **self.kw))

    def __or__(self, rhs):
        return self.func(rhs)

    def __call__(self, *a, **kw):
        return self.func(*a, **kw)

def iterizeable(arg, basetype=None):
    if basetype and isinstance(arg, basetype): return False
    if hasattr(arg, '__iter__'): return True
    return False

def iterize_on_first_param(*metaargs, **metakw):
    """
    'vectorize' a function so it accepts iterables and returns lists
    """

    def deco(func):

        @functools.wraps(func)
        def wrapper(arg0, *args, **kw):
            if iterizeable(arg0, **metakw):
                return [func(a0, *args, **kw) for a0 in arg0]
            return func(arg0, *args, **kw)

        return wrapper

    if metaargs:  # hangle case with no call/args
        assert callable(metaargs[0])
        assert not metakw
        return deco(metaargs[0])
    return deco

iterize_on_first_param_path = iterize_on_first_param(basetype=(str, Path))
