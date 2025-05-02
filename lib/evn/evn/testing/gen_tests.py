import inspect
import copy
import typing as t
import evn

T = t.TypeVar('T')
R = t.TypeVar('R')


def generate_tests(
    args: list[T],
    prefix: str = 'helper_test_',
    convert: t.Callable[[T], R] = lambda x: x,
    namespace: t.MutableMapping = {},
    **kw,
):
    if not namespace:
        namespace = inspect.currentframe().f_back.f_globals

    for arg in args:
        testname = arg
        if not isinstance(testname, str):
            testname = arg[0].replace(' ', '_')
        assert isinstance(testname, str)

        @evn.chrono
        def run_convert(arg, kw=kw):
            return evn.kwcall(kw, convert, arg)

        processed: R = run_convert(arg)

        for k, func in list(namespace.items()):
            if k.startswith(prefix):
                name = k[prefix.find('test_'):]
                func = t.cast(t.Callable[[R], None], func)

                def testfunc(func=func, processed: R = processed, kw=kw):
                    return evn.kwcall(kw, func, *copy.copy(processed))

                testfunc.__name__ = testfunc.__qualname__ = f'{name}_{testname}'
                namespace[f'{name}_{testname.upper()}'] = testfunc
