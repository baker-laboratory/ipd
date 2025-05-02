import sys
import time
import inspect
# from doctest import testmod
import typing
import tempfile
import pytest
import io
import evn

T = typing.TypeVar('T')

class TestConfig(evn.Bunch):

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
        self.context = self.get('context', evn.nocontext)
        self.use_test_classes = self.get('use_test_classes', True)
        self.dryrun = self.get('dryrun', False)

    def detect_fixtures(self, namespace):
        if not evn.ismap(namespace):
            namespace = vars(namespace)
        for name, obj in namespace.items():
            if callable(obj) and hasattr(obj, '_pytestfixturefunction'):
                assert name not in self.fixtures
                self.fixtures[name] = obj.__wrapped__()

@evn.struct
class TestResult:
    passed: list[str] = evn.field(list)
    failed: list[str] = evn.field(list)
    errored: list[str] = evn.field(list)
    xfailed: list[str] = evn.field(list)
    skipexcn: list[str] = evn.field(list)
    _runtime: dict[str, float] = evn.field(dict)

    def runtime(self, name: str) -> float:
        return self._runtime[name]

    def items(self) -> list[tuple[str, list[str]]]:
        return [
            ('passed', self.passed),
            ('failed', self.failed),
            ('errored', self.errored),
            ('xfailed', self.xfailed),
            ('skipexcn', self.skipexcn),
        ]

def quicktest(namespace, config=evn.Bunch(), **kw):
    t_start = time.perf_counter()
    orig = namespace
    if not evn.ismap(namespace):
        namespace = vars(namespace)
    if '__file__' in namespace:
        print(f'quicktest "{namespace["__file__"]}":', flush=True)
    else:
        print(f'quicktest "{orig}":', flush=True)
    # evn.onexit(evn.global_timer.report, timecut=0.01, spacer=1)
    config = TestConfig(**config, **kw)
    config.detect_fixtures(namespace)
    evn.kwcall(config, evn.meta.filter_namespace_funcs, namespace)
    # timed = evn.chrono if config.timed else lambda f: f
    # timed = lambda f:
    test_funcs, teardown = collect_tests(namespace, config)
    # evn.global_timer.checkpoint('quicktest')
    try:
        result = run_tests(test_funcs, config, kw)
    finally:
        for func in teardown:
            func()
    print_result(config, result, time.perf_counter() - t_start)
    return result

def print_result(config, result, t_total):
    if result.passed:
        print(f'PASSED {len(result.passed)} tests in {t_total:.3f} seconds')
    result.passed.sort(key=result.runtime, reverse=True)
    npassprinted = 0
    for label, tests in result.items():
        for test in tests:
            if label == 'passed' and not config.verbose and npassprinted > 9 and result._runtime[test] < 100:
                npassprinted += 1
                continue
            print(f'{label.upper():9} {result._runtime[test]*1000:7.3f} ms {test}', flush=True)

def test_func_ok(name, obj):
    return name.startswith('test_') and callable(obj) and evn.testing.no_pytest_skip(obj)

def test_class_ok(name, obj):
    return name.startswith('Test') and isinstance(obj, type) and not hasattr(obj, '__unittest_skip__')

def collect_tests(namespace, config):
    test_funcs, test_classes, teardown = [], [], []
    for name, obj in namespace.items():
        if test_class_ok(name, obj) and config.use_test_classes:
            suite = obj()
            test_classes.append(suite)
            # print(f'{f" obj: {name} ":=^80}', flush=True)
            test_methods = evn.meta.filter_namespace_funcs(vars(namespace[name]))
            test_methods = {
                f'{name}.{k}': getattr(suite, k)
                for k, v in test_methods.items() if test_func_ok(k, v)
            }
            # TODO: maybe call these lazilyt?
            getattr(suite, 'setUp', lambda: None)()
            # test_suites.append((name, obj))
            test_funcs.extend(test_methods.items())
            teardown.append(getattr(suite, 'tearDown', lambda: None))
        elif test_func_ok(name, obj):
            test_funcs.append((name, obj))
    testmodule = evn.Path(inspect.getfile(test_funcs[0][1])).stem
    for _, func in test_funcs:
        if evn.is_free_function(func):
            func.__module__ = func.__module__.replace('__main__', testmodule)
    for obj in test_classes:
        obj.__module__ = obj.__module__.replace('__main__', testmodule)
    return test_funcs, teardown

def run_tests(test_funcs, config, kw):
    result = TestResult()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = evn.Path(tmpdir)
        evn.kwcall(config.fixtures, config.setup)
        config.fixtures['tmpdir'] = str(tmpdir)
        config.fixtures['tmp_path'] = tmpdir
        for name, func in test_funcs:
            quicktest_run_maybe_parametrized_func(name, func, result, config, kw)
    return result

def quicktest_run_maybe_parametrized_func(name, func, result, config, kw):
    names, values = evn.testing.get_pytest_params(func) or ((), [()])
    for val in values:
        if len(names) == 1 and not isinstance(val, (list, tuple)):
            val = [val]
        paramkw = kw | dict(zip(names, val))
        quicktest_run_test_function(name, func, result, config, paramkw)

def quicktest_run_test_function(name, func, result, config, kw, check_xfail=True):
    error, testout = None, None
    nocapture = config.nocapture is True or name in config.nocapture
    context = evn.nocontext if nocapture else evn.capture_stdio
    with context() as testout:  # noqa
        try:
            evn.kwcall(config.fixtures, config.funcsetup)
            if not config.dryrun:
                kwthis = evn.kwcheck(config.fixtures | kw, func)
                t_start = time.perf_counter()
                func(**kwthis)
                result._runtime[name] = time.perf_counter() - t_start
                result.passed.append(name)
        except pytest.skip.Exception:
            result.skipexcn.append(name)
        except AssertionError as e:
            if evn.testing.has_pytest_mark(func, 'xfail'):
                result.xfailed.append(name)
            else:
                result.failed.append(name)
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

class CapSys:

    def __init__(self):
        self._stdout = None
        self._stderr = None
        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self):
        self._stdout = io.StringIO()
        self._stderr = io.StringIO()
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalize()

    def readouterr(self):
        self._stdout.seek(0)
        self._stderr.seek(0)
        return CapResult(self._stdout.read(), self._stderr.read())

    def _finalize(self):
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr

class CapResult:

    def __init__(self, out, err):
        self.out = out
        self.err = err

def maincrudtest(crud, namespace, fixtures=None, funcsetup=lambda: None, **kw):
    fixtures = fixtures or {}
    with crud() as crud:
        fixtures |= crud

        def newfuncsetup(backend):
            backend._clear_all_data_for_testing_only()
            evn.kwcall(fixtures, funcsetup)

        return quicktest(namespace, fixtures, funcsetup=newfuncsetup, **kw)
