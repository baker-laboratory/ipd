"""
`make_decorator` is a utility for creating configurable decorators for functions, methods,
and classes. It wraps a user-defined function with consistent behavior, supports default
configuration, and allows per-use customization.

It simplifies writing decorators that need to handle:
- Plain functions
- Instance methods and class methods
- Entire classes (auto-wrapping all suitable methods)

Example:
    >>> @make_decorator(prefix='* ')
    ... def printer(func, *args, prefix='', **kwargs):
    ...     print(prefix + func.__name__)
    ...     return func(*args, **kwargs)

    >>> @printer
    ... def hello():
    ...     return 'hi'

    >>> hello()
    * hello
    'hi'

Basic usage:

    >>> @make_decorator
    ... def trace(func, *args, **kwargs):
    ...     print('Calling', func.__name__)
    ...     return func(*args, **kwargs)

    >>> @trace
    ... def greet():
    ...     return 'hello'

    >>> greet()
    Calling greet
    'hello'

Using default config:

    >>> @make_decorator(msg='start:')
    ... def tagged(func, *args, msg, **kwargs):
    ...     return msg + func(*args, **kwargs)

    >>> @tagged
    ... def foo():
    ...     return 'bar'

    >>> foo()
    'start:bar'

Overriding decorator config:

    >>> @tagged(msg='>>')
    ... def baz():
    ...     return 'boo'

    >>> baz()
    '>>boo'

Decorating a method:

    >>> @make_decorator(extra=10)
    ... def add_extra(func, *args, extra, **kwargs):
    ...     return func(*args, **kwargs) + extra

    >>> class Math:
    ...     @add_extra(extra=5)
    ...     def add(self, x, y):
    ...         return x + y

    >>> Math().add(1, 2)
    8

Decorating a class:

    >>> @add_extra(extra=2)
    ... class Ops:
    ...     def mul(self, x, y):
    ...         return x * y
    ...
    ...     def sub(self, x, y):
    ...         return x - y

    >>> o = Ops()
    >>> o.mul(3, 4)
    14
    >>> o.sub(5, 3)
    4
"""

import inspect
import functools
import wrapt
import typing as t

def make_decorator(userwrap: t.Callable | None = None, strict=True, **decokw):
    """
    A decorator factory that simplifies writing configurable function, method, and class decorators.

    `make_decorator` turns a user-supplied wrapper function into a fully-featured decorator.
    It can be used directly to decorate functions, methods, or entire classes, and it supports
    configurable arguments at decoration time.

    Parameters:
        userwrap (callable, optional): A function of the form
            `userwrap(func, args, kwargs, **config)` that defines the core behavior of the decorator.
            If None, `make_decorator` returns a partially-applied decorator factory.
        strict (bool): If True, raises an error when unknown decorator kwargs are passed.
        **decokw: Default keyword arguments for configuration.

    Returns:
        A decorator that can be applied to functions, methods, or classes.

    Usage:
        The wrapped `userwrap` receives:
            - `func`: the original function being decorated
            - `args`, `kwargs`: arguments with which the function is called
            - `**config`: all decorator configuration options passed at creation time

    Raises:
        TypeError: If `userwrap` is not callable or if strict mode is on and unexpected
                   keyword arguments are passed to the decorator.
    """

    if userwrap is None:
        return functools.partial(make_decorator, **decokw)

    if not callable(userwrap):
        raise TypeError(f'make_decorator first arg {type(userwrap)} is not callable')

    def decorator(userwrapped=None, *, strict=strict, decokwie=decokw, **decokw2) -> t.Callable:
        if userwrapped is None:
            if strict and not decokw2.keys() <= decokw.keys():
                raise TypeError(
                    f"Decorator {userwrap.__name__} doesn't accept args: {decokw2.keys() - decokw.keys()}")
            return functools.partial(decorator, **(decokw | decokw2))

        all_kwargs = decokw | decokw2

        if inspect.isclass(userwrapped):
            # Handle class wrapping directly, no wrapt.decorator
            cls = userwrapped
            for name, attr in vars(cls).items():
                if name.startswith('__'):
                    continue
                if isinstance(attr, staticmethod):
                    func = attr.__func__
                    wrapped_func = decorator(func, **all_kwargs)
                    setattr(cls, name, staticmethod(wrapped_func))
                elif isinstance(attr, classmethod):
                    func = attr.__func__
                    wrapped_func = decorator(func, **all_kwargs)
                    setattr(cls, name, classmethod(wrapped_func))
                elif callable(attr) and not inspect.isclass(attr):
                    wrapped_func = decorator(attr, **all_kwargs)
                    setattr(cls, name, wrapped_func)
            return cls

        # Otherwise, this is a normal function or method
        # the default args are for the benefit of the type checker; never used
        @wrapt.decorator()
        def wrapper(
            wrapped: t.Callable,
            instance=None,
            args=[],
            kwargs={},
        ) -> t.Callable:
            kwargs = {k: v for k, v in kwargs.items() if k not in all_kwargs}
            return userwrap(wrapped, *args, **kwargs, **all_kwargs)

        return wrapper(userwrapped)

    return decorator
