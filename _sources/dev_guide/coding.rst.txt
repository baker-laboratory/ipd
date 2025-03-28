Coding Guidelines
===================

This document describes the code style used in the IPD project.

Style
-------
Currently we use yapf. See settings in `pyproject.toml`.

Avoid comments in general, prefer docstrings. If your function is long enough to have multiple sections described my comments, break it up into smaller function with docstring descriptions instead. Comments should say "why" not "what." If you are thinkind about spending a line on a comment to say what code is doing, change the code so it is more self-descriptive instead. Longer variable names, etc.

Keep functions short and focused. If a function is more than 20 lines, consider breaking it up into smaller functions. This makes the code easier to read and test. Try to make your classes serve a single purpose, and try keep the number of methods in a class to a minimum.

Type Hints
-----------
Prefer to provide type hints, esp. for return types and important 'API' functions. Try to make your code pass pyright standard checks.  Not required, as much of IPD doesn't conform to this.

Classes
-------
Prefer to use dataclasses... they list all thier data members so nicedy. :py:func:`ipd.struct`, `:py:func:ipd.mutablestruct`, and :py:func:`ipd.field` are useful helpers for this, creating a dataclass with slots, without slots, and a dataclasses.field wrapper with the first arg being the default_factory. ``ipd.field(list)`` ranther than ``dataclasses.field(default_factory=list)``.

IPD also provides some pretty sweet decorators to enhance your classes with cool behavors:
seealso::

    `ipd.dev.decorators`
    `ipd.dev.elemwise`

Docstrings
------------

All modules, classes, and functions should have docstrings. Use google style, and annotations for sphinx. Include examples with ``>>>`` that can be checked with doctest:

.. code-block:: python

    def add(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: The first number.
            b: The second number.

        Returns:
            The sum of the two numbers.

        Examples:
            >>> add(1, 2)
            3
        """
        return a + b



