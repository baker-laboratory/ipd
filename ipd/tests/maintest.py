import tempfile
import traceback

import pydantic

import ipd

def maintest(
    namespace,
    fixtures=None,
    setup=lambda: None,
    funcsetup=lambda: None,
    just=None,
    exclude=None,
    nofail=False,
):
    print(f'maintest {namespace["__file__"]}:', flush=True)
    just, exclude = just or [], exclude or []
    fixtures, passed, failed = fixtures or {}, [], []
    with tempfile.TemporaryDirectory() as tmpdir:
        ipd.dev.call_with_args_from(fixtures, setup)
        fixtures['tmpdir'] = tmpdir
        for name, func in [(n, f) for n, f in namespace.items() if n[:5] == 'test_' and callable(f)]:
            try:
                if just and name not in just: continue
                if name in exclude: continue
                print(f'{func.__name__:=^60}', flush=True)
                ipd.dev.call_with_args_from(fixtures, funcsetup)
                try:
                    ipd.dev.call_with_args_from(fixtures, func, timed=True)
                    passed.append(name)
                except pydantic.ValidationError as e:
                    print(e, flush=True)
                    print(e.errors(), flush=True)
                    print(traceback.format_exc(), flush=True)
                    failed.append(name)
            except AssertionError:
                if nofail: continue
                raise
    # for p in passed:
    #     print(f'    PASS {p}', flush=True)
    # for f in failed:
    #     print(f'    FAIL {f}', flush=True)
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
