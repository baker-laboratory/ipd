"""
"Vectorizing" python functions
=================================

Use :func:`iterize_on_first_param` to automatically vectorize a function over its first parameter.

Ues :func:`is_iterizeable` to determine if an object should be treated as iterable for vectorization purposes.

Example:

    >>> @iterize_on_first_param
    ... def square(x):
    ...     return x * x
    >>> square(4)
    16
    >>> square([1, 2, 3])
    [1, 4, 9]
"""

import contextlib
import functools
from pathlib import Path
import evn


def NoneFunc():
    """This function does nothing and is used as a default placeholder."""
    pass


def is_iterizeable(arg,
                   basetype: type = str,
                   splitstr: bool = True,
                   allowmap: bool = False) -> bool:
    """
    Determine if an object should be treated as iterable for vectorization purposes.

    This function checks several conditions:
      - Strings with spaces are considered iterable if `splitstr` is True.
      - Objects of the type specified by `basetype` are treated as scalars.
      - Mapping types are not considered iterable unless `allowmap` is True.

    :param arg: The object to test.
    :param basetype: A type (or tuple of types) that should be considered scalar. Defaults to str.
    :param splitstr: If True, strings containing spaces are considered iterable. Defaults to True.
    :param allowmap: If False, mapping types (e.g. dict) are not treated as iterable. Defaults to False.
    :return: True if the object is considered iterable, False otherwise.
    :rtype: bool

    Examples:
        >>> is_iterizeable([1, 2, 3])
        True
        >>> is_iterizeable('hello')
        False
        >>> is_iterizeable('hello world')
        True
        >>> is_iterizeable({'a': 1})
        False
        >>> is_iterizeable({'a': 1}, allowmap=True)
        True
    """
    if isinstance(basetype, str):
        if basetype == 'notlist':
            return isinstance(arg, list)
        elif arg.__class__.__name__ == basetype:
            basetype = type(arg)
        elif arg.__class__.__qualname__ == basetype:
            basetype = type(arg)
        else:
            basetype = type(None)
    if isinstance(arg, str) and ' ' in arg:
        return True
    if basetype and isinstance(arg, basetype):
        return False
    if not allowmap and isinstance(arg, evn.Mapping):
        return False
    if hasattr(arg, '__iter__'):
        return True
    return False


def iterize_on_first_param(
    func0: evn.F = NoneFunc,
    *,
    basetype: 'str|type|tuple[type,...]' = str,
    splitstr=True,
    asdict=False,
    asbunch=False,
    asnumpy=False,
    allowmap=False,
    nonempty=False,
) -> evn.F:
    """
    Decorator to vectorize a function over its first parameter.

    This decorator allows a function to seamlessly handle both scalar and iterable inputs for its first
    parameter. When the first argument is iterable (and not excluded by type), the function is applied
    to each element individually. The results are then combined and returned in a format determined by the
    decorator options.

    :param func0: The function to decorate. Can be omitted when using decorator syntax with arguments.
    :param basetype: Type(s) that should be treated as scalar, even if iterable. Defaults to str.
    :param splitstr: If True, strings containing spaces are split into lists before processing.
                     Defaults to True.
    :param asdict: If True, returns results as a dictionary with input values as keys. Defaults to False.
    :param asbunch: If True, returns results as a Bunch (a dict-like object with attribute access).
                    Defaults to False.
    :param asnumpy: If True, returns results as a numpy array. Defaults to False.
    :param allowmap: If True, allows mapping types (e.g. dict) to be processed iteratively. Defaults to False.
    :return: A decorated function that can handle both scalar and iterable inputs for its first parameter.
    :rtype: callable

    Examples:
        Basic usage:
        >>> @iterize_on_first_param
        ... def square(x):
        ...     return x * x
        >>> square(4)
        16
        >>> square([1, 2, 3])
        [1, 4, 9]

        Using asdict to return results as a dictionary:
        >>> @iterize_on_first_param(asdict=True, basetype=str)
        ... def double(x):
        ...     return x * 2
        >>> double(['a', 'b'])
        {'a': 'aa', 'b': 'bb'}

        **Basic usage with default behavior**:

        >>> @iterize_on_first_param
        ... def square(x):
        ...     return x * x
        >>> square(5)
        25
        >>> square([1, 2, 3])
        [1, 4, 9]

        **Using `basetype` to prevent iteration over strings**:

        >>> @iterize_on_first_param(basetype=str)
        ... def process(item):
        ...     return len(item)
        >>> process('hello')  # Treated as scalar despite being iterable
        5
        >>> process(['hello', 'world'])
        [5, 5]

        **Using `asdict` to return results as a dictionary**:

        >>> @iterize_on_first_param(asdict=True)
        ... def double(x):
        ...     return x * 2
        >>> double([1, 2, 3])
        {1: 2, 2: 4, 3: 6}

        **Using `asbunch` to return results as a Bunch**:

        >>> @iterize_on_first_param(asbunch=True)
        ... def triple(x):
        ...     return x * 3
        >>> result = triple(['a', 'b'])
        >>> result.a
        'aaa'
        >>> result.b
        'bbb'

        **Using `allowmap` to enable mapping support**:

        >>> @iterize_on_first_param(allowmap=True)
        ... def negate(x):
        ...     return -x
        >>> negate({'a': 1, 'b': 2})
        {'a': -1, 'b': -2}

    Notes:
        - The decorator can be applied with or without parentheses.
        - If `asdict` and `asbunch` are both `True`, `asbunch` takes precedence.
        - If `allowmap` is `True`, the decorator will apply the function to the values
          of the mapping and return a new mapping.
    """

    def deco(func: evn.F) -> evn.F:

        @functools.wraps(func)
        def wrapper(arg0, *args, **kw):
            if is_iterizeable(arg0,
                              basetype=basetype,
                              splitstr=splitstr,
                              allowmap=allowmap):
                if splitstr and isinstance(arg0, str) and ' ' in arg0:
                    arg0 = arg0.split()
                if allowmap and isinstance(arg0, evn.Mapping):
                    result = {k: func(v, *args, **kw) for k, v in arg0.items()}
                elif asdict or asbunch:
                    result = {a0: func(a0, *args, **kw) for a0 in arg0}
                else:
                    result = [func(a0, *args, **kw) for a0 in arg0]
                    with contextlib.suppress(TypeError, ValueError):
                        result = type(arg0)(result)
                if nonempty and evn.islist(result):
                    result = list(filter(len, result))
                if nonempty and evn.isdict(result):
                    {k: v for k, v in result.items() if len(v)}
                if asbunch and result and isinstance(evn.first(result.keys()),
                                                     str):
                    result = evn.Bunch(result)
                if asnumpy and evn.installed.numpy:
                    import numpy as np

                    result = np.array(result)
                return result
            return func(arg0, *args, **kw)

        return wrapper

    if func0 is not NoneFunc:  # handle case with no call/args
        assert callable(func0)
        return deco(func0)
    return deco


iterize_on_first_param_path = iterize_on_first_param(basetype=(str, Path))
