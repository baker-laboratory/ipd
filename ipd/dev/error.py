import builtins
import functools

_WARNINGS_ISSUED = set()

import ipd

def change_exception(**kw: dict[str, Exception]):
    """Decorator to change exceptions raised by a function."""
    excmap: dict[Exception, Exception] = {getattr(builtins, k): v for k, v in kw.items()}  # type: ignore

    def deco(func):

        @functools.wraps(func)
        def wrap(*a, **kw):
            try:
                return func(*a, **kw)
            except tuple(excmap) as e:  # type: ignore
                raise excmap[e.__class__](str(e)) from e  # type: ignore

        return wrap

    return deco

def WARNME(message, once=True, tag=None):
    tag = tag or message
    if once and tag not in _WARNINGS_ISSUED:
        import traceback

        print("-" * 80, flush=True)
        print(message, flush=True)
        traceback.print_stack()
        _WARNINGS_ISSUED.add(message)
        print("-" * 80)
        return True
    return False

class HaltAndCatchFire(Exception):
    pass

_panic_print_n_calls = 0

def panicprint(*a, allowcalls=1, **kw):
    global _panic_print_n_calls
    if allowcalls <= (_panic_print_n_calls := _panic_print_n_calls + 1):
        with ipd.dev.capture_stdio() as msg:
            print()
            print(f'{"panicprint":!^60}')
            print(*a, **kw)
            print('!' * 60)
            raise HaltAndCatchFire(msg.read())
    else:
        print(f'{f" panicprint {_panic_print_n_calls} ":!^60}')
        print(*a, **kw)
