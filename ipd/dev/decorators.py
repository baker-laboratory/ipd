import contextlib
import functools
from pathlib import Path
from typing import Mapping, Any, Iterable

import numpy as np

import ipd

def generic_get_keys(obj, exclude: ipd.FieldSpec = ()):
    if hasattr(obj, 'keys') and callable(getattr(obj, 'keys')):
        return [k for k in obj.keys() if valid_element_name(k, exclude)]
    elif hasattr(obj, 'items'):
        return [k for k, v in obj.items() if valid_element_name(k, exclude)]
    elif isinstance(obj, list):
        return list(range(len(obj)))
    else:
        return [
            k for k in dir(obj) if valid_element_name_thorough(k, exclude) and not callable(getattr(obj, k))
        ]
    raise TypeError(f'dont know how to get elements from {obj}')

def generic_get_items(obj):
    if hasattr(obj, 'items'):
        return [(k, v) for k, v in obj.items() if valid_element_name(k)]
    elif hasattr(obj, 'keys') and callable(getattr(obj, 'keys')):
        return [(k, getattr(obj, k)) for k in obj.keys() if valid_element_name(k)]
    elif isinstance(obj, list):
        return list(enumerate(obj))
    else:
        return [(k, getattr(obj, k)) for k in dir(obj)
                if valid_element_name_thorough(k) and not callable(getattr(obj, k))]
    raise TypeError(f'dont know how to get elements from {obj}')

def valid_element_name(name, exclude=()):
    return not name[0] == '_' and not name[-1] == '_' and name not in exclude

def valid_element_name_thorough(name, exclude=()):
    return valid_element_name(name, exclude) and name not in _reserved_element_names

_reserved_element_names = set('mapwise npwise valwise dictwise'.split())

def get_fields(obj, fields: ipd.FieldSpec, exclude: ipd.FieldSpec = ()) -> tuple[Iterable, bool]:
    """return list of string fields and bool isplural"""
    if callable(fields): fields = fields(obj)
    if fields is None: return generic_get_keys(obj, exclude=exclude), True
    if ' ' in fields: return ipd.cast(str, fields).split(), True
    if isinstance(fields, str): return [fields], False
    return fields, True

def is_iterizeable(arg, basetype: type = str, splitstr: bool = True, allowmap: bool = False) -> bool:
    """Checks if an object can be treated as an iterable.

    This function determines if `arg` can be iterated over. It considers edge cases such as:
    - Strings with spaces (if `splitstr` is True) are considered iterable.
    - Instances of `basetype` are not treated as iterable.
    - Mappings (like dictionaries) are considered iterable unless `allowmap` is set to False.

    Args:
        arg (Any): The object to test for iterability.
        basetype (type, optional): A type that should not be considered iterable. If a string is passed,
            it will attempt to match it to the type's `__name__` or `__qualname__`. Defaults to `str`.
        splitstr (bool, optional): If True, strings containing spaces are considered iterable. Defaults to True.
        allowmap (bool, optional): If False, mappings (like dictionaries) are not treated as iterable. Defaults to False.

    Returns:
        bool: True if `arg` is iterable according to the specified rules, otherwise False.

    Example:
        is_iterizeable([1, 2, 3])  # True
        is_iterizeable('hello')  # False
        is_iterizeable('hello world')  # True (if `splitstr=True`)
        is_iterizeable({'a': 1})  # False (if `allowmap=False`)
        is_iterizeable({'a': 1}, allowmap=True)  # True
    """
    if isinstance(basetype, str):
        if basetype == 'notlist': return isinstance(arg, list)
        elif arg.__class__.__name__ == basetype: basetype = type(arg)
        elif arg.__class__.__qualname__ == basetype: basetype = type(arg)
        else: basetype = type(None)
    if isinstance(arg, str) and ' ' in arg: return True
    if basetype and isinstance(arg, basetype): return False
    if not allowmap and isinstance(arg, Mapping): return False
    if hasattr(arg, '__iter__'): return True
    return False

def NoneFunc():
    pass

def iterize_on_first_param(
    func0: ipd.F = NoneFunc,
    *,
    basetype: 'str|type|tuple[type,...]' = str,
    splitstr=True,
    asdict=False,
    asbunch=False,
    asnumpy=False,
    allowmap=False,
) -> ipd.F:
    """
    Decorator that vectorizes a function over its first parameter.

    This decorator allows a function to handle both single values and iterables as its
    first argument. If the first argument is iterable, the function is applied to each
    element individually, and the results are returned in an appropriate format (list,
    dictionary, or Bunch).

    If the first argument is not iterable (or is excluded by type), the function behaves
    normally.

    Args:
        basetype (type or tuple of types, optional):
            Type(s) that should be treated as scalar values, even if they are iterable.
            For example, `basetype=str` ensures strings are treated as single values.
            Defaults to `str`.

        splitstr (bool, optional):
            If `True`, strings with spaces are split into lists before processing.
            Defaults to `True`.

        asdict (bool, optional):
            If `True`, the results are returned as a dictionary with input values as keys.
            Defaults to `False`.

        asbunch (bool, optional):
            If `True`, the results are returned as a `Bunch` object (like a dict but
            with attribute-style access). If the keys are strings, they are used as
            attribute names. Defaults to `False`.

        allowmap (bool, optional):
            If `True`, allows mapping types (like dictionaries) to be processed as
            iterables, with the function applied to each value. Defaults to `False`.

    Returns:
        callable: A decorated function that can handle both scalar and iterable inputs
        for its first parameter.

    Examples:
        **Basic usage with default behavior**:

        >>> @iterize_on_first_param
        ... def square(x):
        ...     return x * x
        ...
        >>> square(5)
        25
        >>> square([1, 2, 3])
        [1, 4, 9]

        **Using `basetype` to prevent iteration over strings**:

        >>> @iterize_on_first_param(basetype=str)
        ... def process(item):
        ...     return len(item)
        ...
        >>> process("hello")  # Treated as scalar despite being iterable
        5
        >>> process(["hello", "world"])
        [5, 5]

        **Using `asdict` to return results as a dictionary**:

        >>> @iterize_on_first_param(asdict=True)
        ... def double(x):
        ...     return x * 2
        ...
        >>> double([1, 2, 3])
        {1: 2, 2: 4, 3: 6}

        **Using `asbunch` to return results as a Bunch**:

        >>> @iterize_on_first_param(asbunch=True)
        ... def triple(x):
        ...     return x * 3
        ...
        >>> result = triple(["a", "b"])
        >>> result.a
        'aaa'
        >>> result.b
        'bbb'

        **Using `allowmap` to enable mapping support**:

        >>> @iterize_on_first_param(allowmap=True)
        ... def negate(x):
        ...     return -x
        ...
        >>> negate({"a": 1, "b": 2})
        {'a': -1, 'b': -2}

    Notes:
        - The decorator can be applied with or without parentheses.
        - If `asdict` and `asbunch` are both `True`, `asbunch` takes precedence.
        - If `allowmap` is `True`, the decorator will apply the function to the values
          of the mapping and return a new mapping.
    """

    def deco(func: ipd.F) -> ipd.F:

        @ipd.wraps(func)
        def wrapper(arg0, *args, **kw):
            if is_iterizeable(arg0, basetype=basetype, splitstr=splitstr, allowmap=allowmap):
                if splitstr and isinstance(arg0, str) and ' ' in arg0:
                    arg0 = arg0.split()
                if allowmap and isinstance(arg0, Mapping):
                    result = {k: func(v, *args, **kw) for k, v in arg0.items()}
                elif asdict or asbunch:
                    result = {a0: func(a0, *args, **kw) for a0 in arg0}
                else:
                    result = [func(a0, *args, **kw) for a0 in arg0]
                    with contextlib.suppress(TypeError, ValueError):
                        resutn = type(arg0)(result)
                if asbunch and result and isinstance(ipd.first(result.keys()), str):
                    result = ipd.Bunch(result)
                if asnumpy:
                    result = np.array(result)
                return result
            return func(arg0, *args, **kw)

        return wrapper

    if func0 is not NoneFunc:  # handle case with no call/args
        assert callable(func0)
        return deco(func0)
    return deco

iterize_on_first_param_path = iterize_on_first_param(basetype=(str, Path))

import ipd.dev

def preserve_random_state(func0=None, seed0=None):
    """Decorator to preserve the random state during function execution.

    This decorator sets a temporary random seed during the execution of the decorated function.
    If a `seed` is passed as a keyword argument to the function, it will override the default seed.

    Args:
        func0 (callable, optional): The function to decorate. If provided, the decorator can be used without parentheses.
        seed0 (int, optional): The default random seed to use if not overridden by a `seed` keyword argument.

    Returns:
        callable: The decorated function.

    Example:
        >>> @preserve_random_state(seed0=42)
        >>> def my_function():
        >>>     # Function code here

    Raises:
        AssertionError: If `func0` is provided but is not callable or if `seed0` is not None when `func0` is used.
    """

    def deco(func):

        @ipd.wraps(func)
        def wrapper(*args, **kw):
            with ipd.dev.temporary_random_seed(seed=kw.get('seed', seed0)):
                return func(*args, **kw)

        return wrapper

    if func0:  # handle case with no call/args
        assert callable(func0)
        assert seed0 is None
        return deco(func0)
    return deco

def make_getitem_for_attributes(get=getattr, provide='value') -> 'Any':
    if provide not in ('value', 'item'):
        raise ValueError(f"provide must be 'value' or 'item', not {provide}")

    def getitem_for_attributes(self, field: ipd.FieldSpec, get=get) -> 'Any':
        """Enhanced `__getitem__` method to support attribute access with multiple keys.

    If the field is a string containing spaces, it will be split into a list of keys.
    If the field is a list of strings, it will return the corresponding attributes as a tuple.

    Args:
        field (list[str] | str): A single attribute name or a list of attribute names.

    Returns:
        Any: The attribute value(s) corresponding to the field(s).

    Example:
        >>> obj = MyClass()
        >>> value = obj['x']  # Single field
        >>> values = obj['x y z']  # Multiple keys as a string
        >>> values = obj[['x', 'y', 'z']]  # Multiple keys as a list
    """
        field, plural = get_fields(self, field)
        if provide == 'value':
            if plural: return tuple(get(self, k) for k in field)
            else: return get(self, field[0])
        if provide == 'item':
            if plural: return ipd.Bunch((k, get(self, k)) for k in field)
            return (field[0], get(self, field[0]))

    return getitem_for_attributes

def generic_enumerate(self, fields: ipd.FieldSpec, order=lambda x: x) -> ipd.EnumerIter:
    """Enhanced `enumerate` method to iterate over multiple attributes at once.

    This method allows enumeration over multiple attributes simultaneously, returning an index and the corresponding attribute values. If the fields is a string containing spaces, it will be split into a list of keys. If the fields is a list of strings, it will return the corresponding attributes as a tuple.

    Args:
        fields (list[str] | str): A single attribute name or a list of attribute names.

    Yields:
        tuple[int, ...]: A tuple containing the index and the attribute values.

    Example:
        >>> obj = MyClass()
        >>> for i, x, y in obj.enumerate(['x', 'y']):
        >>>     print(i, x, y)
    """
    fields = list(zip(*self[fields]))
    idx = range(len(fields[0]))
    for i, vals in zip(order(idx), order(fields)):
        yield i, *vals

def generic_groupby(
    self,
    groupby: ipd.FieldSpec,
    fields: ipd.FieldSpec = None,
    convert=None,
) -> ipd.EnumerListIter:
    exclude = None
    splat = isinstance(fields, str)
    if callable(groupby):
        groupby = groupby(self)
    else:
        groupby, plural = get_fields(self, groupby)
        exclude = groupby
        if not plural: groupby = groupby[0]
        groupby = self[groupby]
    fields, _ = get_fields(self, fields, exclude=exclude)
    vals = self[fields]
    groups = dict()
    for k, v in zip(groupby, zip(*vals)):
        groups.setdefault(k, []).append(v)
    for group, vals in groups.items():
        vals = zip(*vals)
        if convert: vals = map(convert, vals)
        if splat: yield group, *vals
        else: yield group, ipd.Bunch(zip(fields, vals))

def is_fuzzy_match(sub, string):
    """
    Checks if 'sub' is a subsequence of 'string'.

    Args:
      sub: The potential subsequence string.
      string: The main string.

    Returns:
      True if 'sub' is a subsequence of 'string', False otherwise.
    """
    if sub[:2] != string[:2]: return False
    i, j = 0, 0
    while i < len(sub) and j < len(string):
        if sub[i] == string[j]: i += 1
        j += 1
    return i == len(sub)

def getattr_fzf(obj, field):
    """Get attribute values using fuzzy matching.

    This function uses fuzzy matching to find attribute names that are similar to the given field.
    If multiple matches are found, the values of the first match are returned.

    Args:
        obj: The object to search for attributes.
        field: The field to search for.

    Returns:
        Any: The attribute value corresponding to the field.

    Example:
        obj = MyClass()
        value = obj.fzf('x')  # Finds 'x' or similar
    """
    fields = generic_get_keys(obj, exclude=())
    candidates = [f for f in fields if is_fuzzy_match(field, f)]
    if not candidates: raise AttributeError(f'no attribute found for {field}')
    if len(candidates) == 1: return getattr(obj, candidates[0])
    raise AttributeError(f'multiple attributes found for {field}: {candidates}')

def subscriptable_for_attributes(cls: type[ipd.C]) -> type[ipd.C]:
    """Class decorator to enable subscriptable attribute access and enumeration.

    This decorator adds support for `__getitem__` and `enumerate` methods to a class
    using `generic_getitem_for_attributes` and `generic_enumerate`.

    Args:
        cls (type): The class to modify.

    Returns:
        type: The modified class.

    Example:
        @subscriptable_for_attributes
        class MyClass:
            def __init__(self):
                self.x = 1
                self.y = 2

        obj = MyClass()
        print(obj['x'])  # 1
        print(obj['x y'])  # (1, 2)
        for i, x, y in obj.enumerate(['x', 'y']):
            print(i, x, y)

    Raises:
        TypeError: If the class already defines `__getitem__` or `enumerate`.
    """
    for member in 'enumerate fzf groupby'.split():
        if hasattr(cls, member):
            raise TypeError(f'class {cls.__name__} already has {member}')
    cls.__getitem__ = make_getitem_for_attributes(get=getattr)
    cls.fzf = make_getitem_for_attributes(get=getattr_fzf)
    cls.enumerate = generic_enumerate
    cls.groupby = generic_groupby
    cls.pick = make_getitem_for_attributes(provide='item')
    return cls

def safe_lru_cache(func=None, *, maxsize=128):
    if func is not None and callable(func):
        # Case when used as @safe_lru_cache without parentheses
        return safe_lru_cache(maxsize=maxsize)(func)

    def decorator(func):
        cache = functools.lru_cache(maxsize=maxsize)(func)

        @ipd.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                hash(args)
                frozenset(kwargs.items())
                return cache(*args, **kwargs)
            except TypeError:
                return func(*args, **kwargs)

        return wrapper

    return decorator
