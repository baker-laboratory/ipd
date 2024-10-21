import builtins
import functools
from ipd.dev.types.type_aliases import KW

def change_exception(**kw:KW):
    excmap = {getattr(builtins, k): v for k, v in kw.items()}

    def deco(func):
        @functools.wraps(func)
        def wrap(*a, **kw):
            try:
                return func(*a, **kw)
            except tuple(excmap) as e:
                raise excmap[e.__class__](str(e)) from e

        return wrap

    return deco
