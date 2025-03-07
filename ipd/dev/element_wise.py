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

    def __get__(self, parent, parenttype):
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
                accum = self._Accumulator()
                if len(args) <= 1:
                    args = itertools.repeat(args)
                elif len(args) == len(self._parent):
                    args = [[arg] for arg in args]
                else:
                    raise ValueError(
                        f'ElementWiseDispatcher arg must be len 1 or len(_parent) == {len(self._parent)}')
                for arg, (name, val) in zip(args, self._parent.items()):
                    if name[0] == '_': continue
                    if callable(method): accum.add(name, method(val, *arg, **kw))
                    else: accum.add(name, getattr(val, method)(*arg, **kw))
                return accum.result()

            self._parent._ewise_method[cachekey] = apply_method

        return self._parent._ewise_method[cachekey]

    for name, op in vars(operator).items():
        if not (name.startswith('__')) and not name.startswith('i'):
            locals()[f'__{name}__'] = lambda self, other, op=op: self.__getattr__(op)(other)
            locals()[f'__r{name}__'] = lambda self, other, op=op: self.__getattr__(op)(other)

    def __rsub__(self, other):
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
