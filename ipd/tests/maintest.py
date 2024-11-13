import tempfile
import traceback

import pydantic

import ipd

def maintest(namespace, fixtures=None, setup=lambda: None, funcsetup=lambda: None, just=None):
    just = just or []
    fixtures, passed, failed = fixtures or {}, [], []
    with tempfile.TemporaryDirectory() as tmpdir:
        ipd.dev.call_with_args_from(fixtures, setup)
        fixtures['tmpdir'] = tmpdir
        for name, func in [(n, f) for n, f in namespace.items() if n.startswith('test_')]:
            if just and name not in just: continue
            print('=' * 20, func.__name__, '=' * 20)
            ipd.dev.call_with_args_from(fixtures, funcsetup)
            try:
                ipd.dev.call_with_args_from(fixtures, func)
                passed.append(name)
            except pydantic.ValidationError as e:
                print(e)
                print(e.errors())
                print(traceback.format_exc())
                failed.append(name)

    print(f'maincrudtest {namespace["__file__"]}:')
    for p in passed:
        print(f'    PASS {p}')
    for f in failed:
        print(f'    FAIL {f}')
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
