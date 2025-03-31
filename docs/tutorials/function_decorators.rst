Function Decorator Tutorial
===========================

This tutorial demonstrates how to use the decorators provided in the
``decorators`` module. It covers the following decorators:

- :func:`ipd.dev.decorators.iterize_on_first_param`
- :func:`ipd.dev.decorators.iterize_on_first_param_path`
- :func:`ipd.dev.decorators.preserve_random_state`
- :func:`ipd.dev.decorators.safe_lru_cache`

Each section includes examples written in a doctest-friendly format.

Introduction
------------

The ``decorators`` module provides a set of decorators to enhance the behavior of functions and classes. These decorators allow functions to be automatically vectorized, preserve random state, and cache function results safely. This tutorial covers how to use these decorators from a user's perspective.

Vectorizing Functions with ``ipd.dev.iterize_on_first_param``
--------------------------------------------------------------

The :func:`ipd.dev.decorators.iterize_on_first_param` decorator enables a function to handle both scalar
and iterable inputs for its first parameter. When provided with an iterable (that is not
excluded by the ``basetype`` parameter), the function is applied element-wise.

Basic usage:

.. code-block:: python

    >>> import ipd
    >>> @ipd.dev.iterize_on_first_param
    ... def square(x):
    ...     return x * x
    >>> square(5)
    25
    >>> square([1, 2, 3])
    [1, 4, 9]

Controlling scalar treatment with ``basetype``:

.. code-block:: python

    >>> @ipd.dev.iterize_on_first_param(basetype=str)
    ... def repeat(x):
    ...     return x * 2
    >>> repeat("hello")
    'hellohello'
    >>> repeat(["a", "b"])
    ['aa', 'bb']

Returning results as a dictionary:

.. code-block:: python

    >>> @ipd.dev.iterize_on_first_param(asdict=True, basetype=str)
    ... def shout(s):
    ...     return s.upper()
    >>> shout("foo")
    'FOO'
    >>> shout(["foo", "bar"])
    {'foo': 'FOO', 'bar': 'BAR'}

Using ``iterize_on_first_param_path``
---------------------------------------

The :func:`ipd.dev.decorators.iterize_on_first_param_path` decorator is a pre-configured variant of :func:`ipd.dev.decorators.iterize_on_first_param` that treats both strings and :class:`Path` objects as scalars.

.. code-block:: python

    >>> from pathlib import Path
    >>> @ipd.dev.iterize_on_first_param_path
    ... def process_path(p):
    ...     return f"Processed {p}"
    >>> process_path("file.txt")
    'Processed file.txt'
    >>> process_path(Path("file.txt"))
    'Processed file.txt'
    >>> process_path(["file1.txt", "file2.txt"])
    ['Processed file1.txt', 'Processed file2.txt']

Preserving Random State with ``preserve_random_state``
-------------------------------------------------------

The :func:`ipd.dev.decorators.preserve_random_state` decorator temporarily sets a random seed during the execution of a function. This is useful when you need reproducible random behavior.

.. code-block:: python

    >>> import random
    >>> @ipd.dev.preserve_random_state
    ... def random_value():
    ...     return random.randint(1, 100)
    >>> # The following call sets a fixed seed. Since the output depends on random,
    >>> # we skip the output check in doctest.
    >>> random_value(seed=42)  # doctest: +SKIP


Safe Caching with ``safe_lru_cache``
-------------------------------------

The :func:`safe_lru_cache` decorator caches function results with an LRU cache. If the
function arguments are unhashable, the cache is bypassed gracefully.

.. code-block:: python

    >>> @ipd.dev.safe_lru_cache(maxsize=32)
    ... def double(x):
    ...     print('double called')
    ...     return x * 2
    >>> double(4)
    double called
    8
    >>> double(4)  # Cached result is returned
    8
    >>> double([1, 2, 3])
    double called
    [1, 2, 3, 1, 2, 3]
    >>> double([1, 2, 3])  # Unhashable input: executed normally, not cached
    double called
    [1, 2, 3, 1, 2, 3]
