import functools

_1d_modifiers = set(('add remove append push pop extend intert clear discard __iadd__ __isub__ '
                     '__imul__ __itruediv__ __imod__ __ifloordiv__ __ipow__ __imatmul__ __iand__ '
                     '__ior__ __ixor__ __irshift__ __ilshift__').split())

class Instrumented:
    def __init_subclass__(cls):
        if hasattr(cls, '__on_change__'):
            tomodify = {}
            for base in reversed(cls.__mro__):
                tomodify |= base.__dict__
            for fn in set(tomodify):
                if not fn.endswith('_update') and fn not in _1d_modifiers: continue
                func = tomodify[fn]

                @functools.wraps(func)
                def wrap(self, *a, __wrappedfunc__=func, **kw):
                    result = __wrappedfunc__(self, *a, **kw)
                    getattr(cls, '__on_change__')(self, result)
                    return result

                setattr(cls, fn, wrap)
