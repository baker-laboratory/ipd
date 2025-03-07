"""
decorator for allowing element wise operations on any class
"""
from collections.abc import Mapping
import itertools
import operator

import numpy as np

import ipd

def element_wise_operations(cls):
    cls.mapwise = ElementWise(BunchAccumulator)
    cls.valwise = ElementWise(ListAccumulator)
    cls.npwise = ElementWise(NumpyAccumulator)
    return cls

class ElementWise:
    """descriptor creating/returning a cached ElementWiseDispatcher"""

    def __init__(self, Accumulator):
        self.Accumulator = Accumulator

    def __get__(self, parent, _parenttype):
        if not hasattr(parent, '_ewise_dispatcher'):
            parent.__dict__['_ewise_dispatcher'] = dict()
        if self.Accumulator not in parent._ewise_dispatcher:
            new = ElementWiseDispatcher(parent, self.Accumulator)
            parent._ewise_dispatcher[self.Accumulator] = new
        return parent._ewise_dispatcher[self.Accumulator]

    def __set__(self, parent, values):
        items = values.items() if isinstance(values, Mapping) else zip(parent.keys(), values)
        for k, v in items:
            parent[k] = v

class ElementWiseDispatcher:

    def __init__(self, parent, Accumulator):
        self._parent = parent
        self._Accumulator = Accumulator
        self._parent.__dict__['_ewise_method'] = dict()

    def __getattr__(self, method):
        cachekey = (method, self._Accumulator)
        if cachekey not in self._parent._ewise_method:

            def apply_method(*args, **kw):
                self.validate
                accum = self._Accumulator()
                if not args:
                    for name, member in self._parent.items():
                        if name[0] == '_': continue
                        if callable(method): accum.add(name, method(member, **kw))
                        else: accum.add(name, getattr(member, method)(**kw))
                    return accum.result()
                elif len(args) == 1:
                    args = itertools.repeat(args[0])
                elif len(args) != len(self._parent):
                    raise ValueError(
                        f'ElementWiseDispatcher arg must be len 1 or len(_parent) == {len(self._parent)}')
                for arg, (name, member) in zip(args, self._parent.items()):
                    if name[0] == '_': continue
                    if callable(method): accum.add(name, method(member, arg, **kw))
                    else: accum.add(name, getattr(member, method)(arg, **kw))
                return accum.result()

            self._parent._ewise_method[cachekey] = apply_method

        return self._parent._ewise_method[cachekey]

    # create wrappers for binary operators and their 'r' right versions
    for name, op in vars(operator).items():
        if not (name.startswith('__')) and not name.startswith('i'):
            locals()[f'__{name}__'] = lambda self, other, op=op: self.__getattr__(op)(other)
            locals()[f'__r{name}__'] = lambda self, other, op=op: self.__getattr__(op)(other)

    def contains(self, other):
        contains_check = getattr(other, 'contains', operator.contains)
        return self.__getattr__(contains_check)(other)

    def contained_by(self, other):
        if contained_by_check := getattr(other, 'contained_by', None):
            return self.__getattr__(contained_by_check)(other)
        contained = lambda s, o: operator.contains(o, s)
        return self.__getattr__(contained)(other)

    def __contains__(self, other):
        raise ValueError('a in foo.*wise is invalid, use .contains or .contained_by')

    def __rsub__(self, other):
        """generic wrapper is reversed"""
        return -self.__getattr__(operator.sub)(other)

    def __neg__(self):
        return self.__getattr__(operator.neg)()

class BunchAccumulator:

    def __init__(self):
        self.value = ipd.Bunch()

    def add(self, name, value):
        self.value[name] = value

    def result(self):
        return self.value

class ListAccumulator:

    def __init__(self):
        self.value = []

    def add(self, name, value):
        self.value.append(value)

    def result(self):
        return self.value

class NumpyAccumulator():

    def __init__(self):
        self.value = []

    def add(self, name, value):
        self.value.append(value)

    def result(self):
        return np.array(self.value)
