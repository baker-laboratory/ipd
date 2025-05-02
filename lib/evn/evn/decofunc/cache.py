"""
**Safe Caching**: :func:`safe_lru_cache` provides an LRU cache that handles unhashable arguments gracefully.

"""

import functools


def safe_lru_cache(func=None, *, maxsize=128):
    """
    A safe LRU cache decorator that handles unhashable arguments gracefully.

    This decorator wraps a function with an LRU cache. If the arguments are hashable, the cached value
    is returned; if unhashable (raising a TypeError), the function is executed normally without caching.

    :param func: The function to decorate. If omitted, the decorator can be used with arguments.
    :param maxsize: The maximum size of the cache. Defaults to 128.
    :return: The decorated function.
    :rtype: callable

    Examples:
        Basic usage:
        >>> @safe_lru_cache(maxsize=32)
        ... def double(x):
        ...     return x * 2
        >>> double(2)
        4
        >>> double([1, 2, 3])  # Unhashable input; executes without caching.
        [1, 2, 3, 1, 2, 3]

        Using without arguments:
        >>> @safe_lru_cache
        ... def add(x, y):
        ...     return x + y
        >>> add(2, 3)
        5
    """
    if func is not None and callable(func):
        # Case when used as @safe_lru_cache without parentheses
        return safe_lru_cache(maxsize=maxsize)(func)

    def decorator(func):
        cache = functools.lru_cache(maxsize=maxsize)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                hash(args)
                frozenset(kwargs.items())
                return cache(*args, **kwargs)
            except TypeError:
                return func(*args, **kwargs)

        return wrapper

    return decorator
