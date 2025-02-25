from collections import ChainMap

class Tolerances:

    def __init__(self, tol=1e-4, **kw):
        if isinstance(tol, Tolerances):
            self.tol = tol.tol
            self.kw = ChainMap(kw, tol.kw)
        else:
            self.tol = tol
            self.kw = kw

    def __getattr__(self, k):
        return float(self.kw.get(k, self.tol))

    def __float__(self):
        return self.tol
