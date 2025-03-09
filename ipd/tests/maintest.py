import tempfile

import ipd

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

class TestResult(ipd.Bunch):

    def __init__(self, *a, **kw):
        super().__init__(self, *a, **kw)
        self.passed, self.failed, self.errored, self.xfailed = [], [], [], []

def _test_func_ok(name, obj):
    return name.startswith('test_') and callable(obj) and ipd.dev.no_pytest_skip(obj)

def _test_class_ok(name, obj):
    return name.startswith('Test') and isinstance(obj, type) and not hasattr(obj, '__unittest_skip__')

def maintest(namespace, config=ipd.Bunch(), **kw):
    print(f'maintest "{namespace["__file__"]}":', flush=True)
    ipd.dev.onexit(ipd.dev.global_timer.report, timecut=0.1)
    config = TestConfig(**config, **kw)
    ipd.kwcall(ipd.dev.filter_namespace_funcs, config, namespace)
    timed = ipd.dev.timed if config.timed else lambda f: f
    test_suites, test_funcs = [], []
    for name, obj in namespace.items():
        if _test_class_ok(name, obj) and config.use_test_classes:
            test_suites.append((name, timed(obj)()))
        elif _test_func_ok(name, obj):
            test_funcs.append((name, timed(obj)))
    ipd.dev.global_timer.checkpoint('maintest')
    result = TestResult()
    with tempfile.TemporaryDirectory() as tmpdir:
        ipd.kwcall(config.setup, config.fixtures)
        config.fixtures['tmpdir'] = tmpdir

        for name, func in test_funcs:
            _maintest_run_test_function(name, func, result, config, kw)

        for clsname, suite in test_suites:
            print(f'{f" suite: {clsname} ":=^80}', flush=True)
            test_methods = ipd.dev.filter_namespace_funcs(vars(namespace[clsname]))
            test_methods = {k: v for k, v in test_methods.items() if _test_func_ok(k, v)}
            getattr(suite, 'setUp', lambda: None)()
            for name in test_methods:
                _maintest_run_test_function(name, func, result, config, kw)
            getattr(suite, 'tearDown', lambda: None)

    if result.passed: print('PASSED   ', len(result.passed), 'tests')
    for label, tests in result.items():
        if label == 'passed' and not config.verbose and len(result.passed) > 7: continue
        for test in tests:
            print(f'{label.upper():9} {test}', flush=True)
    return result

def _maintest_run_test_function(name, func, result, config, kw, check_xfail=True):
    error, testout = None, None
    context = ipd.dev.nocontext if name in config.nocapture else ipd.dev.capture_stdio
    with context() as testout:  # noqa
        try:
            ipd.kwcall(config.funcsetup, config.fixtures)
            if not config.dryrun:
                ipd.kwcall(func, config.fixtures)
                result.passed.append(name)
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

def maincrudtest(crud, namespace, fixtures=None, funcsetup=lambda: None):
    fixtures = fixtures or {}
    with crud() as crud:
        fixtures |= crud

        def newfuncsetup(backend):
            backend._clear_all_data_for_testing_only()
            ipd.kwcall(funcsetup, fixtures)

        return maintest(namespace, fixtures, funcsetup=newfuncsetup, **kw)
