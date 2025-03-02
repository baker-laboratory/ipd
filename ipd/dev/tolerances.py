import attrs
from collections import ChainMap
import copy
import sys

import numpy as np

import ipd

class Tolerances:

    def __init__(self, parent=None, default=None, **kw):
        if isinstance(parent, Tolerances):
            self.kw = ChainMap(kw, parent.kw)
            self.checkers = parent.checkers
            self._default_tol = parent._default_tol
        else:
            if parent is not None:
                assert default is None, 'usage is Tolerances(parent, default, **kw) or Tolerances(default, **kw)'
                default = parent
            self.kw = kw
            self.checkers = {}
            self._default_tol = default or 1e-4

    def __getattr__(self, name):
        if name in self.__dict__: return self.__dict__[name]
        if not name.startswith('_') and 'checkers' in self.__dict__:
            if name not in self.__dict__['checkers']:
                threshold = float(self.kw.get(name, self._default_tol))
                self.checkers[name] = Checker(threshold)
            return self.checkers[name]
        raise AttributeError(f'Tolerances object has no attribute {name}')

    def reset(self):
        for c in self.checkers.values():
            c.n_checks = 0
            c.n_passes = 0
        return self

    def check_history(self):
        history = ipd.Bunch()
        for k, c in self.checkers.items():
            frac = round(c.n_passes / c.n_checks, 3) if c.n_checks > 0 else None
            history[k] = ipd.Bunch(frac=frac, tol=c.threshold, total=c.n_checks, passes=c.n_passes)
        return history

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        with ipd.dev.capture_stdio() as out:
            ipd.dev.print_table(self.check_history(), key='Tolerances object')
        return out.read()

@attrs.define
class Checker:
    threshold: float
    n_checks: int = 0
    n_passes: int = 0

    def _record(self, result):
        if isinstance(result, bool):
            self.n_checks += 1
            self.n_passes += result
        else:
            xr = sys.modules.get('xarray')
            if xr and isinstance(result, xr.Dataset):
                raise TypeError('Tolerances doesnt support whole xarray.Dataset comparisons')
            assert str(result.dtype)[-4:] == 'bool'
            self.n_checks += len(result)
            self.n_passes += np.sum(np.array(result))
        return result

    def __float__(self):
        return self.threshold

    def __gt__(self, val):
        return self._record(self.threshold > val)

    def __ge__(self, val):
        return self._record(self.threshold >= val)

    def __lt__(self, val):
        return self._record(self.threshold < val)

    def __le__(self, val):
        return self._record(self.threshold <= val)

    def __rgt__(self, val):
        return self._record(self.threshold < val)

    def __rge__(self, val):
        return self._record(self.threshold <= val)

    def __rlt__(self, val):
        return self._record(self.threshold > val)

    def __rle__(self, val):
        return self._record(self.threshold >= val)

    def __eq__(self, val):
        return self._record(self.threshold == val)

    gt, ge, lt, le = __gt__, __ge__, __lt__, __le__
