"""
- **Random State Preservation**: Use :func:`preserve_random_state` to temporarily set a random seed
  during a function call.
"""

import functools
import evn


def preserve_random_state(func0=None, seed0=None):
    """Decorator to preserve the random state during function execution.

    This decorator sets a temporary random seed during the execution of the decorated function.
    If a `seed` is passed as a keyword argument to the function, it will override the default seed.

    Args:
        func0 (callable, optional): The function to decorate. If provided, the decorator can be used without parentheses.
        seed0 (int, optional): The default random seed to use if not overridden by a `seed` keyword argument.

    Returns:
        callable: The decorated function.

    Raises:
        AssertionError: If `func0` is provided but is not callable or if `seed0` is not None when `func0` is used.
    """

    def deco(func):

        @functools.wraps(func)
        def wrapper(*args, **kw):
            with evn.temporary_random_seed(seed=kw.get('seed', seed0)):
                return func(*args, **kw)

        return wrapper

    if func0:  # handle case with no call/args
        assert callable(func0)
        assert seed0 is None
        return deco(func0)
    return deco
