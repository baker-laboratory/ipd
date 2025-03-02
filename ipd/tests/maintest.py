import tempfile
import traceback

import pydantic

import ipd

def maintest(
        namespace,
        fixtures=None,
        setup=lambda: None,
        funcsetup=lambda: None,
        nofail=False,
        config=ipd.Bunch(_strict=False),
        **kw,
):
    if config:
        from ipd.dev import filter_namespace_funcs
        filter_namespace_funcs(namespace, **ipd.kwcheck(config, filter_namespace_funcs))

    print(f'maintest {namespace["__file__"]}:', flush=True)
    test_classes, test_funcs = [], []
    for name, obj in namespace.items():
        if name.startswith('Test') and isinstance(obj, type) and not hasattr(obj, '__unittest_skip__'):
            test_classes.append((name, obj))
        elif name.startswith('test_') and callable(obj) and ipd.dev.no_pytest_skip(obj):
            test_funcs.append((name, obj))
    fixtures, passed, failed = fixtures or {}, [], []
    with tempfile.TemporaryDirectory() as tmpdir:
        ipd.dev.call_with_args_from(fixtures, setup)
        fixtures['tmpdir'] = tmpdir
        for name, func in test_funcs:
            exc, testout = None, None
            with ipd.dev.capture_stdio() as testout:  # noqa
                # if True:
                try:
                    ipd.dev.call_with_args_from(fixtures, funcsetup, **kw)
                    try:
                        ipd.dev.call_with_args_from(fixtures, func, **kw)
                        passed.append(name)
                    except pydantic.ValidationError as e:
                        print(e, flush=True)
                        print(e.errors(), flush=True)
                        print(traceback.format_exc(), flush=True)
                        failed.append(name)
                except AssertionError as e:
                    failed.append(name)
                    exc = e
            if name in failed:
                if ipd.dev.has_pytest_mark(func, 'xfail'):
                    print('xfail', name)
                    continue
                print(f'{func.__name__:=^60}', flush=True)
                if testout: print(testout.read(), flush=True, end='')
                if nofail and exc: print(exc)
                elif exc: raise exc

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
                except AssertionError as exc:
                    failed.append(full_name)
                    if nofail and exc: print(exc)
                    elif exc: raise exc

            teardown_method()

    if passed: print('maintest passed', len(passed), 'tests')
    for f in failed:
        print(f'FAIL {f}', flush=True)
    ipd.dev.global_timer.report()
    return passed, failed

def maincrudtest(crud, namespace, fixtures=None, funcsetup=lambda: None, **kw):
    fixtures = fixtures or {}
    with crud() as crud:
        fixtures |= crud

        def newfuncsetup(backend):
            backend._clear_all_data_for_testing_only()
            ipd.dev.call_with_args_from(fixtures, funcsetup)

        return maintest(namespace, fixtures, funcsetup=newfuncsetup, **kw)
