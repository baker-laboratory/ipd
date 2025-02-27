from collections import ChainMap

class Tolerances:

    def __init__(self, tol=1e-4, **kw):
        if isinstance(tol, Tolerances):
            self.default = tol.default
            self.kw = ChainMap(kw, tol.kw)
        else:
            self.default = tol
            self.kw = kw

    def __getattr__(self, k):
        return float(self.kw.get(k, self.default))

    def __float__(self):
        return self.tol
