import tempfile

import pydantic

import ipd

ipd.dev.onexit(ipd.dev.global_timer.report)

def maintest(
        namespace,
        fixtures=None,
        setup=lambda: None,
        funcsetup=lambda: None,
        nofail=False,
        config=ipd.Bunch(_strict=False),
        verbose=False,
        **kw,
):
    print(f'maintest {namespace["__file__"]}:', flush=True)
    ipd.kwcall(
        ipd.dev.filter_namespace_funcs,
        config,
        namespace,
    )
    test_classes, test_funcs = [], []
    for name, obj in namespace.items():
        if name.startswith('Test') and isinstance(obj, type) and not hasattr(obj, '__unittest_skip__'):
            test_classes.append((name, obj))
        elif name.startswith('test_') and callable(obj) and ipd.dev.no_pytest_skip(obj):
            test_funcs.append((name, obj))
    ipd.dev.global_timer.checkpoint('maintest')
    fixtures = fixtures or {}
    result = ipd.Bunch(passed=[], failed=[], errored=[], xfailed=[])
    with tempfile.TemporaryDirectory() as tmpdir:
        ipd.dev.call_with_args_from(fixtures, setup)
        fixtures['tmpdir'] = tmpdir

        for name, func in test_funcs:
            _maintest_run_test_function(name, func, result, nofail, fixtures, funcsetup, kw)

        for class_name, test_class in test_classes:
            print(f'{class_name:=^60}', flush=True)
            test_instance = test_class()
            setup_method = getattr(test_instance, 'setUp', lambda: None)
            teardown_method = getattr(test_instance, 'tearDown', lambda: None)
            test_methods = [
                method for method in dir(test_instance)
                if method.startswith('test_') and callable(getattr(test_instance, method))
            ]
            setup_method()
            for name in test_methods:
                full_name = f"{class_name}.{name}"
                print(f'{full_name:-^50}', flush=True)
                test_method = getattr(test_instance, name)
                try:
                    ipd.dev.call_with_args_from(fixtures, test_method, **kw)
                    passed.append(full_name)
                except AssertionError as error:
                    failed.append(full_name)
                    if nofail and error: print(error)
                    elif error: raise error

            teardown_method()

    if result.passed: print('PASSED   ', len(result.passed), 'tests')
    for label, tests in result.items():
        if label == 'passed' and not verbose and len(result.passed) > 7: continue
        for test in tests:
            print(f'{label.upper():9} {test}', flush=True)
    return result

def _maintest_run_test_function(name, func, result, nofail, fixtures, funcsetup, kw):
    error, testout = None, None
    with ipd.dev.capture_stdio() as testout:  # noqa
        # if True:
        try:
            ipd.dev.call_with_args_from(fixtures, funcsetup, **kw)
            try:
                ipd.dev.call_with_args_from(fixtures, func, **kw)
                result.passed.append(name)
            except (pydantic.ValidationError, TypeError, ValueError) as e:
                result.errored.append(name)
                error = e
        except AssertionError as e:
            if ipd.dev.has_pytest_mark(func, 'xfail'): result.xfailed.append(name)
            else: result.failed.append(name)
            error = e
    if name in result.failed or name in result.errored:
        print(f'{func.__name__:=^60}', flush=True)
        if testout: print(testout.read(), flush=True, end='')
        if nofail and error: print(error)
        elif error: raise error

def maincrudtest(crud, namespace, fixtures=None, funcsetup=lambda: None, **kw):
    fixtures = fixtures or {}
    with crud() as crud:
        fixtures |= crud

        def newfuncsetup(backend):
            backend._clear_all_data_for_testing_only()
            ipd.dev.call_with_args_from(fixtures, funcsetup)

        return maintest(namespace, fixtures, funcsetup=newfuncsetup, **kw)
