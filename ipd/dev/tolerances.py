from collections import ChainMap
import numpy as np

class Tolerances:

    def __init__(self, tol=1e-4, **kw):
        if isinstance(tol, Tolerances):
            self.kw = ChainMap(kw, tol.kw)
            self.checkers = tol.checkers
            self._default_tol = tol._default_tol
        else:
            self.kw = kw
            self.checkers = {}
            self._default_tol = tol

    def __getattr__(self, name):
        if name not in self.checkers:
            threshold = float(self.kw.get(name, self._default_tol))
            self.checkers[name] = _Checker(threshold)
        return self.checkers[name]

    def __float__(self):
        return self.tol

    def reset(self):
        for c in self.checkers.values():
            c.n_checks = 0
            c.n_passes = 0

    def check_history(self):
        history = {}
        for k, c in self.checkers.items():
            frac = round(c.n_passes / c.n_checks, 3) if c.n_checks > 0 else None
            history[k] = dict(total=c.n_checks, passes=c.n_passes, frac=frac, tol=c.threshold)
        return history

class _Checker:

    def __init__(self, threshold):
        self.threshold = threshold
        self.n_passes = 0
        self.n_checks = 0

    def _record(self, result):
        if isinstance(result, bool):
            self.n_checks += 1
            self.n_passes += result
        else:
            assert result.dtype == bool
            self.n_checks += len(result)
            self.n_passes += np.sum(result)
        return result

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
