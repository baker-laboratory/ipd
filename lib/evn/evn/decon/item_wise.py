"""Element-wise operations for collections.

This module provides a decorator and supporting classes to enable element-wise
operations on custom collection classes. It allows applying methods or functions
to each element in a collection and collecting the results.
"""

from collections.abc import Mapping
import itertools
from functools import partial
import operator
from evn._prelude.lazy_import import lazyimport

np = lazyimport('numpy')

import evn

generic_get_items = evn.ninja_import(
    'evn.decon.attr_access','generic_get_items')


def get_available_result_types():
    return dict(
        map=BunchAccumulator,
        dict=DictAccumulator,
        val=ListAccumulator,
        np=NumpyAccumulator,
    )


class Missing:
    pass


def item_wise_operations(cls0: evn.Optional[type[evn.C]] = Missing,
                         result_types='map val') -> type[evn.C]:
    """Decorator that adds element-wise operation capabilities to a class.

    Adds up to four attributes to the decorated class:
    - dictwise: Returns results as a dict
    - mapwise: Returns results as a mapping (evn.Bunch)
    - valwise: Returns results as a list
    - npwise: Returns results as a numpy array

    Args:
        cls: The class to decorate
        result_types: selects which attributes to add

    Returns:
        The decorated class
    """
    if evn.is_installed('numpy'):
        result_types = f'np {result_types}'
    if cls0 is Missing:
        return partial(item_wise_operations, result_types=result_types)
    orig = result_types
    if isinstance(result_types, str):
        result_types = result_types.split() if ' ' in result_types else [
            result_types
        ]
    result_types = set(result_types)
    available_result_types = get_available_result_types()
    if not set(available_result_types) & result_types:
        raise TypeError(f'result_types {orig} is invalid')

    def decorate(cls: type[evn.C]) -> type[evn.C]:
        for rtype in result_types:
            setattr(cls, f'{rtype}wise',
                    ElementWise(available_result_types[rtype]))
        return cls

    return decorate(cls0)


class ElementWise:
    """Descriptor that creates and caches an ElementWiseDispatcher.

    When accessed from an instance, returns a dispatcher that applies
    operations element-wise and collects results using the specified
    accumulator.
    """

    def __init__(self, Accumulator):
        self.Accumulator = Accumulator

    def __get__(self, parent, _parenttype):
        # if parent is None: return None
        if not hasattr(parent, '_ewise_dispatcher'):
            parent.__dict__['_ewise_dispatcher'] = dict()
        if self.Accumulator not in parent._ewise_dispatcher:
            new = ElementWiseDispatcher(parent, self.Accumulator)
            parent._ewise_dispatcher[self.Accumulator] = new
        return parent._ewise_dispatcher[self.Accumulator]

    def __set__(self, parent, values):
        items = values.items() if isinstance(values, Mapping) else zip(
            parent.keys(), values)
        for k, v in items:
            parent[k] = v


class ElementWiseDispatcher:
    """Dispatcher that applies operations to each element in a collection.

    Dynamically creates methods that apply an operation to each element
    and collect the results using the specified accumulator.
    """

    def __init__(self, parent, Accumulator):
        """Initialize with a parent collection and an accumulator class.

        Args:
            parent: The collection to operate on
            Accumulator: Class that implements the accumulator interface
        """
        self._parent = parent
        self._Accumulator = Accumulator
        self._parent.__dict__['_ewise_method'] = dict()

    # create wrappers for binary operators and their 'r' right versions
    for name, op in vars(operator).items():
        if not (name.startswith('__')) and not name.startswith('i'):
            locals(
            )[f'__{name}__'] = lambda self, other, op=op: self.__getattr__(op)(
                other)
            locals(
            )[f'__r{name}__'] = lambda self, other, op=op: self.__getattr__(
                op)(other)

    def __getattr__(self, method):
        """Get or create a method that applies the operation element-wise.

        Args:
            method: Name of method to call on each element, or a callable

        Returns:
            Function that applies the operation and collects results
        """
        cachekey = (method, self._Accumulator)
        if cachekey not in self._parent._ewise_method:

            def apply_method(*args, **kw):
                """Apply method to each element with the given arguments.

                If no positional args are provided, applies the method to each element.
                If one arg is provided, applies it to each element.
                If multiple args are provided, they must match the number of elements.

                Args:
                    *args: Arguments to pass to the method
                    **kw: Keyword arguments to pass to the method

                Returns:
                    Accumulated results using the configured accumulator

                Raises:
                    ValueError: If the number of args doesn't match requirements
                """
                accum = self._Accumulator()
                items = generic_get_items(self._parent)
                if not args:
                    try:
                        for name, member in items:
                            if callable(method):
                                accum.add(name, method(member, **kw))
                            else:
                                accum.add(name, getattr(member, method)(**kw))
                        return accum.result()
                    except TypeError:
                        # kw forwarding to method failed, use kw ar elementwise args
                        args = [kw]
                        kw = {}
                if len(args) == 1:
                    arg = args[0]
                    itemkeys = set(item[0] for item in items)
                    if isinstance(arg, dict):
                        assert arg.keys(
                        ) == itemkeys, f'{itemkeys} != {arg.keys()}'
                        # elemwise args passed as single dict
                        args = [arg[k] for k, _ in items]
                    else:
                        args = itertools.repeat(arg)
                elif len(args) != len(items):
                    raise ValueError(
                        f'ElementWiseDispatcher arg must be len 1 or len(items) == {len(items)}'
                    )
                for arg, (name, member) in zip(args, items):
                    if callable(method):
                        accum.add(name, method(member, arg, **kw))
                    else:
                        accum.add(name, getattr(member, method)(arg, **kw))
                return accum.result()

            self._parent._ewise_method[cachekey] = apply_method

        return self._parent._ewise_method[cachekey]

    def __call__(self, func, *a, **kw):
        """call a function on each element of self._parent, forwarding any arguments"""
        assert callable(func)
        result = self.__getattr__(func)(*a, **kw)
        for k, v in generic_get_items(self._parent, all=True):
            if k[0] == '_' or k[-1] == '_':
                with evn.cl.suppress(AttributeError):
                    setattr(result, k, v)
        return result

    def contains(self, other):
        contains_check = getattr(other, 'contains', operator.contains)
        return self.__getattr__(contains_check)(other)

    def contained_by(self, other):
        if contained_by_check := getattr(other, 'contained_by', None):
            return self.__getattr__(contained_by_check)(other)
        contained = lambda s, o: operator.contains(o, s)
        return self.__getattr__(contained)(other)

    def __contains__(self, other):
        raise ValueError(
            'a in foo.*wise is invalid, use .contains or .contained_by')

    def __rsub__(self, other):
        """generic wrapper is reversed"""
        return generic_negate(self.__getattr__(operator.sub)(other))

    def __neg__(self):
        return self.__getattr__(operator.neg)()


class DictAccumulator:
    """Accumulator that collects results into an dict.

    Results are stored with their original keys from the parent collection.
    """

    def __init__(self):
        self.value = evn.Bunch()

    def add(self, name, value):
        self.value[name] = value

    def result(self):
        return self.value


class BunchAccumulator(DictAccumulator):
    """Accumulator that collects results into an evn.Bunch (dict-like object).

    Results are stored with their original keys from the parent collection.
    """

    def __init__(self):
        self.value = evn.Bunch()


class ListAccumulator:
    """Accumulator that collects results into a list.

    Results are stored in the order they are added.
    """

    def __init__(self):
        self.value = []

    def add(self, name, value):
        self.value.append(value)

    def result(self):
        return self.value


class NumpyAccumulator(ListAccumulator):
    """Accumulator that collects results into a numpy array.

    Results are stored in the order they are added.
    """

    def result(self):
        try:
            return np.array(self.value)
        except ValueError:
            return np.array(self.value, dtype=object)


def generic_negate(thing):
    if isinstance(thing, np.ndarray):
        return -thing
    if isinstance(thing, list):
        return [-x for x in thing]
    for k, v in thing.items():
        thing[k] = -v
    return thing
