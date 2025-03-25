import typing
import tempfile

import pytest

import ipd

T = typing.TypeVar('T')

class TestConfig(ipd.Bunch):

    def __init__(self, *a, **kw):
        super().__init__(self, *a, **kw)
        self.nofail = self.get('nofail', False)
        self.verbose = self.get('verbose', False)
        self.checkxfail = self.get('checkxfail', False)
        self.timed = self.get('timed', True)
        self.nocapture = self.get('nocapture', [])
        self.fixtures = self.get('fixtures', {})
        self.setup = self.get('setup', lambda: None)
        self.funcsetup = self.get('funcsetup', lambda: None)
        self.context = self.get('context', ipd.dev.nocontext)
        self.use_test_classes = self.get('use_test_classes', True)
        self.dryrun = self.get('dryrun', False)

    def detect_fixtures(self, namespace):
        if not ipd.ismap(namespace): namespace = vars(namespace)
        for name, obj in namespace.items():
            if callable(obj) and hasattr(obj, '_pytestfixturefunction'):
                assert name not in self.fixtures
                self.fixtures[name] = obj.__wrapped__()

class TestResult(ipd.Bunch):

    def __init__(self, *a, **kw):
        super().__init__(self, *a, **kw)
        for attr in 'passed failed errored xfailed skipexcn'.split():
            setattr(self, attr, [])

def _test_func_ok(name, obj):
    return name.startswith('test_') and callable(obj) and ipd.dev.no_pytest_skip(obj)

def _test_class_ok(name, obj):
    return name.startswith('Test') and isinstance(obj, type) and not hasattr(obj, '__unittest_skip__')

def maintest(namespace, config=ipd.Bunch(), **kw):
    orig = namespace
    if not ipd.ismap(namespace): namespace = vars(namespace)
    if '__file__' in namespace:
        print(f'maintest "{namespace["__file__"]}":', flush=True)
    else:
        print(f'maintest "{orig}":', flush=True)
    ipd.dev.onexit(ipd.dev.global_timer.report, timecut=0.01)
    config = TestConfig(**config, **kw)
    config.detect_fixtures(namespace)
    ipd.kwcall(config, ipd.dev.filter_namespace_funcs, namespace)
    timed = ipd.dev.timed if config.timed else lambda f: f
    test_suites, test_funcs = [], []
    for name, obj in namespace.items():
        if _test_class_ok(name, obj) and config.use_test_classes:
            test_suites.append((name, timed(obj)))
        elif _test_func_ok(name, obj):
            test_funcs.append((name, timed(obj)))
    ipd.dev.global_timer.checkpoint('maintest')
    result = TestResult()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = ipd.Path(tmpdir)
        ipd.kwcall(config.fixtures, config.setup)
        config.fixtures['tmpdir'] = str(tmpdir)
        config.fixtures['tmp_path'] = tmpdir

        for name, func in test_funcs:
            _maintest_run_maybe_parametrized_func(name, func, result, config, kw)

        for clsname, Suite in test_suites:
            suite = Suite()
            print(f'{f" Suite: {clsname} ":=^80}', flush=True)
            test_methods = ipd.dev.filter_namespace_funcs(vars(namespace[clsname]))
            test_methods = {k: v for k, v in test_methods.items() if _test_func_ok(k, v)}
            getattr(suite, 'setUp', lambda: None)()
            for name in test_methods:
                _maintest_run_maybe_parametrized_func(f'{clsname}.{name}', getattr(suite, name), result,
                                                      config, kw)
            getattr(suite, 'tearDown', lambda: None)

    if result.passed: print('PASSED   ', len(result.passed), 'tests')
    for label, tests in result.items():
        if label == 'passed' and not config.verbose and len(result.passed) > 7: continue
        for test in tests:
            print(f'{label.upper():9} {test}', flush=True)
    return result

def _maintest_run_maybe_parametrized_func(name, func, result, config, kw):
    names, values = ipd.dev.get_pytest_params(func) or ((), [()])
    for val in values:
        if len(names) == 1 and not isinstance(val, (list, tuple)): val = [val]
        paramkw = kw | dict(zip(names, val))
        _maintest_run_test_function(name, func, result, config, paramkw)

def _maintest_run_test_function(name, func, result, config, kw, check_xfail=True):
    error, testout = None, None
    nocapture = config.nocapture is True or name in config.nocapture
    context = ipd.dev.nocontext if nocapture else ipd.dev.capture_stdio
    with context() as testout:  # noqa
        try:
            ipd.kwcall(config.fixtures, config.funcsetup)
            if not config.dryrun:
                ipd.kwcall(config.fixtures | kw, func)
                result.passed.append(name)
        except pytest.skip.Exception:
            result.skipexcn.append(name)
        except AssertionError as e:
            if ipd.dev.has_pytest_mark(func, 'xfail'): result.xfailed.append(name)
            else: result.failed.append(name)
            error = e
        except Exception as e:  # noqa
            result.errored.append(name)
            error = e
    if any([
            name in result.failed,
            name in result.errored,
            config.checkxfail and name in result.xfailed,
    ]):
        print(f'{f" {func.__name__} ":-^80}', flush=True)
        if testout: print(testout.read(), flush=True, end='')
        if config.nofail and error: print(error)
        elif error: raise error

def maincrudtest(crud, namespace, fixtures=None, funcsetup=lambda: None, **kw):
    fixtures = fixtures or {}
    with crud() as crud:
        fixtures |= crud

        def newfuncsetup(backend):
            backend._clear_all_data_for_testing_only()
            ipd.kwcall(fixtures, funcsetup)

        return maintest(namespace, fixtures, funcsetup=newfuncsetup, **kw)

def make_parametrized_tests(namespace: ipd.MutableMapping,
                            prefix: str,
                            args: list[T],
                            convert: ipd.Callable[[T], ipd.Any] = lambda x: x,
                            **kw):
    for arg in args:

        @ipd.dev.timed(name=f'{prefix}setup')
        def run_convert(arg, kw=kw):
            return ipd.kwcall(kw, convert, arg)

        processed = run_convert(arg)

        for k, func in list(namespace.items()):
            if k.startswith(prefix):
                name = k[prefix.find('test_'):]

                def testfunc(arg=arg, func=func, processed=processed, kw=kw):
                    return ipd.kwcall(kw, func, processed)

                # c = ipd.dev.timed(lambda arg, kw=kw: ipd.kwcall(kw, convert, arg), name=f'{name}_setup')
                # testfunc = lambda func=func, arg=arg, c=c, kw=kw: ipd.kwcall(kw, func, c(arg))
                testfunc.__name__ = testfunc.__qualname__ = f'{name}_{arg}'
                namespace[f'{name}_{str(arg).upper()}'] = testfunc
