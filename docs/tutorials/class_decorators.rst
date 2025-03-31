.. _class_decorators:

=============================================
Class Decorators Tutorial
=============================================


.. contents:: Table of Contents
   :depth: 3

Element-wise Operations
========================

The :py:mod:`ipd.dev.element_wise` module provides tools to apply operations element-by-element across a structured container such as a ``dict`` or :py:mod:`ipd.preludu_.struct` (convenience wrapper around dataclass) or dataclass. Can be used on any class that holds a fairly homogeneous collection of elements in all non-private attributes. Attributes that end with _ will be ignored my element_wise.

Using the ``@element_wise_operations`` decorator, you can attach special descriptors (e.g., ``.mapwise``, ``.valwise``, ``.npwise``, ``.dictwise``) to a class. These descriptors provide access to an ``ElementWiseDispatcher``, which handles fancy element-wise behavior like:

- :ref:`Method forwarding <Method>`
- :ref:`Free function application <Free>`
- :ref:`Containment checks <Containment>`
- :ref:`Operator forwarding <Operator>`


Core Concepts
-------------

The ``ElementWise`` Descriptor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ElementWise`` descriptor is added to a class by the ``@element_wise_operations`` decorator. When accessed from an instance (e.g. ``obj.mapwise``), it returns an ``ElementWiseDispatcher`` configured with different accumulator type. The accululator type determines the return type of elemwise operations.

.. list-table::
   :header-rows: 1

   * - Descriptor (attribute) Name
     - Accumulator Type
     - return Type

   * - **mapwise**
     - ``BunchAccumulator``
     - ``ipd.Bunch``
   * - **valwise**
     - ``ListAccumulator``
     - ``list``
   * - **npwise**
     - ``NumpyAccumulator``
     - ``np.ndarray``
   * - **dictwise**
     - ``DictAccumulator``
     - ``dict``

You can configure which are added via the ``result_types`` argument:

.. doctest::

    >>> import ipd
    >>> @ipd.element_wise_operations(result_types="np val")
    ... @ipd.mutablestruct
    ... class MyStruct:
    ...     a: int
    ...     b: int

ElementWiseDispatcher
^^^^^^^^^^^^^^^^^^^^^^

The ``ElementWiseDispatcher`` forwards operations to each data attribute, field, or dictionary value in the decorated class, as defined by :py:func:`ipd.dev.decorators.generic_get_items`. It supports:

.. _Method:

Method forwarding
"""""""""""""""""""""""""""""""
Calling ``.mapwise.foo(...)`` on the instance will call ``obj.a.foo(...)``, ``obj.b.foo(...)``, etc.
This works with both method names (as strings) and callable objects:

.. doctest::

    >>> @ipd.element_wise_operations
    ... @ipd.mutablestruct
    ... class MyData:
    ...     a: list
    ...     b: list
    >>> d = MyData(a=[], b=[])
    >>> _ = d.mapwise.append(1, 2)
    >>> d.a
    [1]
    >>> d.b
    [2]

.. _Free:

Free function application
"""""""""""""""""""""""""""""""
Calling ``.mapwise(func, ...)`` applies the function to each element. This is implemented via ``__call__``:

.. doctest::

    >>> d = MyStruct(a=2, b=3)
    >>> d.valwise(lambda x: x**2)
    [4, 9]
    >>> d
    MyStruct(a=2, b=3)

.. _Containment:

Containment checks
"""""""""""""""""""""""""""""""
The dispatcher implements ``contains()`` and ``contained_by()`` methods:

.. doctest::

    >>> @ipd.element_wise_operations
    ... @ipd.mutablestruct
    ... class MyContainer:
    ...     a: list
    ...     b: list
    >>> c = MyContainer(a=[1, 2], b=[3])
    >>> c.mapwise.contains(2)
    Bunch(a=True, b=False)
    >>> c.mapwise.contained_by([[1, 2], [3]])
    Bunch(a=True, b=True)

Note: You must use these methods instead of Python’s ``in`` keyword.

.. _Operator:

Operator forwarding
"""""""""""""""""""""""""""""""
All standard unary and binary operators (e.g., ``+``, ``-``, ``*``, ``==``, ``-obj``, ``5 - obj``) are
forwarded element-wise:

.. doctest::

    >>> d = MyStruct(a=10, b=20)
    >>> d.npwise + 10
    array([20, 30])
    >>> d.valwise == 20
    [False, True]
    >>> -d.npwise
    array([-10, -20])
    >>> 5 - d.npwise
    array([ -5, -15])

Accumulator Behavior
--------------------
Each dispatcher uses an accumulator to collect the results:

- ``BunchAccumulator``: collects results in an ``ipd.Bunch``, retaining keys
- ``ListAccumulator``: collects results in insertion order into a Python list
- ``NumpyAccumulator``: collects results into a NumPy array
- ``DictAccumulator``: collects results into a plain ``dict``

Example
-------

.. doctest::

    >>> @ipd.element_wise_operations
    ... class MyMetrics(dict): pass
    >>> metrics = MyMetrics(a=1, b=2, c=3)
    >>> metrics.mapwise + 10
    Bunch(a=11, b=12, c=13)
    >>> metrics.valwise * 2
    [2, 4, 6]
    >>> metrics.npwise - 1
    array([0, 1, 2])

This design supports expressive and concise batch operations over named data
structures and is ideal for modeling multiple related quantities.

See Also
--------
- :py:func:`ipd.dev.element_wise.element_wise_operations`
- :py:class:`ipd.dev.element_wise.ElementWiseDispatcher`
- :py:class:`ipd.dev.element_wise.ListAccumulator`
- :py:class:`ipd.dev.element_wise.NumpyAccumulator`
- :py:class:`ipd.dev.element_wise.DictAccumulator`
- :py:class:`ipd.bunch.Bunch`


Enhanced Attribule Access
==============================

Introduction
------------
The :py:func:`ipd.dev.decorators.subscriptable_for_attributes` decorator is a class decorator provided by the py:mod:`ipd.dev.decorators` module. It augments a class by adding several convenience methods for attribute access:

- **Subscriptable Access**: Access attributes using square-bracket notation (``obj['attr']``).
- **Fuzzy Matching**: Use the ``fzf`` method to retrieve attributes based on partial or fuzzy names.
- **Enhanced Enumeration**: The ``enumerate`` method iterates over selected attributes with an index.
- **Grouping**: The ``groupby`` method groups attributes based on a key.
- **Pick Method**: The ``pick`` method retrieves key-value pairs for selected attributes.

This tutorial provides examples (written to pass doctest) showing how to use these features
from a user perspective.

Basic Usage
-----------
The most basic use case is to make a class subscriptable for attribute access. After decorating a class,
you can access its attributes using a string key. Multiple keys separated by spaces will return a tuple.

.. code-block:: python

    >>> @ipd.subscriptable_for_attributes
    ... class Person:
    ...     def __init__(self, name, age):
    ...         self.name = name
    ...         self.age = age
    >>> p = Person("Alice", 30)
    >>> p["name"]
    'Alice'
    >>> p["age"]
    30
    >>> p["name age"]
    ('Alice', 30)

Fuzzy Matching
--------------
The decorator also adds a ``fzf`` method to support fuzzy matching when retrieving attributes.
This method uses a fuzzy matching algorithm to locate attributes based on partial names.

.. code-block:: python

    >>> @ipd.subscriptable_for_attributes
    ... class City:
    ...     def __init__(self):
    ...         self.london = "London"
    ...         self.france = "Paris"
    ...         self.underpants = "Underpants"
    ...         self._hidden = "not accessible"
    >>> c = City()
    >>> c.fzf("lon")
    'London'
    >>> c.fzf("fr")
    'Paris'
    >>> c.fzf("underpants")
    'Underpants'
    >>> # Using multiple fuzzy keys to retrieve multiple attributes:
    >>> c.fzf("undpant loon frnc")
    ('Underpants', 'London', 'Paris')

Enhanced Enumeration
--------------------
The decorator provides an ``enumerate`` method that can be used to iterate over selected
attributes of a class instance. The method returns an index along with the corresponding
attribute values.

.. code-block:: python

    >>> @ipd.subscriptable_for_attributes
    ... class Data:
    ...     def __init__(self):
    ...         self.x = [10, 20, 30]
    ...         self.y = [1, 2, 3]
    >>> d = Data()
    >>> list(d.enumerate("x y"))
    [(0, 10, 1), (1, 20, 2), (2, 30, 3)]

Grouping Attributes
-------------------
Another added method is ``groupby``, which groups attribute values based on a key attribute.
This is useful when your class stores parallel lists and you want to group them by a certain criterion.

.. code-block:: python

    >>> @ipd.subscriptable_for_attributes
    ... class GroupData:
    ...     def __init__(self):
    ...         self.value = [1, 2, 3, 4]
    ...         self.label = ['A', 'A', 'B', 'B']
    >>> gd = GroupData()
    >>> # Group by the 'label' attribute, returning the 'value' for each group.
    >>> list(gd.groupby('label', 'value'))
    [('A', (1, 2)), ('B', (3, 4))]

Pick Method for Key-Value Pairs
--------------------------------
The ``pick`` method (another alias added by the decorator) allows retrieving a dictionary-like
object containing selected attributes as key-value pairs.

.. code-block:: python

    >>> @ipd.subscriptable_for_attributes
    ... class Attributes:
    ...     def __init__(self):
    ...         self.x = 100
    ...         self.y = 200
    >>> a = Attributes()
    >>> sorted(a.pick("x y").keys())
    ['x', 'y']

Conclusion
----------
The ``subscriptable_for_attributes`` decorator enhances classes by providing a flexible and
intuitive interface for attribute access. With support for subscriptable access, fuzzy matching,
enumeration, grouping, and key-value selection, it simplifies common tasks when working with class
attributes.

All examples provided above are doctest-friendly and can be automatically tested as part of your Sphinx
documentation build process.
