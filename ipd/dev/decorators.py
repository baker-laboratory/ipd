import functools
from pathlib import Path

import ipd

def is_iterizeable(arg, basetype=None, splitstr=False):
    if isinstance(arg, str) and ' ' in arg: return True
    if basetype and isinstance(arg, basetype): return False
    if hasattr(arg, '__iter__'): return True
    return False

def iterize_on_first_param(func0=None, *, basetype=str, splitstr=True, asdict=False, asbunch=False):
    """Decorator that vectorizes a function over its first parameter.

    This decorator enables a function to handle both single values and iterables as its
    first parameter. When an iterable is passed, the function is applied to each item
    individually and returns a list of results. Otherwise, the function is called normally.

    Args:
        *metaargs: Optional positional arguments. If the first argument is callable,
            it is treated as the function to decorate (allowing for decorator use without
            parentheses).
        **metakw: Keyword arguments passed to the is_iterizeable() function for controlling
            iteration behavior. Common parameters include:
            - basetype: Type or tuple of types that should not be iterated over even if
              they have __iter__ method (e.g., strings, Path objects).

    Returns:
        callable: A decorated function that can handle both scalar and iterable inputs
        for its first parameter.

    Examples:
        Basic usage with default behavior:

        >>> @iterize_on_first_param
        ... def square(x):
        ...     return x * x
        ...
        >>> square(5)
        25
        >>> square([1, 2, 3])
        [1, 4, 9]

        With custom basetype parameter:

        >>> @iterize_on_first_param(basetype=str)
        ... def process(item):
        ...     return len(item)
        ...
        >>> process("hello")  # Treated as scalar despite being iterable
        5
        >>> process(["hello", "world"])
        [5, 5]

    Notes:
        - The decorator can be applied with or without parentheses.
        - The decorated function preserves its name, docstring, and other attributes.
        - For string and path-like objects, consider using iterize_on_first_param_path
          which is preconfigured with basetype=(str, Path).
    """

    def deco(func):

        @functools.wraps(func)
        def wrapper(arg0, *args, **kw):
            if is_iterizeable(arg0, basetype=basetype, splitstr=splitstr):
                if splitstr and isinstance(arg0, str) and ' ' in arg0:
                    arg0 = arg0.split()
                if asbunch:
                    return ipd.Bunch({a0: func(a0, *args, **kw) for a0 in arg0})
                if asdict:
                    return {a0: func(a0, *args, **kw) for a0 in arg0}
                else:
                    return [func(a0, *args, **kw) for a0 in arg0]
            return func(arg0, *args, **kw)

        return wrapper

    if func0:  # handle case with no call/args
        assert callable(func0)
        return deco(func0)
    return deco

iterize_on_first_param_path = iterize_on_first_param(basetype=(str, Path))

def preserve_random_state(func0=None, seed0=None):

    def deco(func):

        @functools.wraps(func)
        def wrapper(*args, **kw):
            with ipd.dev.temporary_random_seed(seed=kw.get('seed', seed0)):
                return func(*args, **kw)

        return wrapper

    if func0:  # handle case with no call/args
        assert callable(func0)
        assert seed0 is None
        return deco(func0)
    return deco
