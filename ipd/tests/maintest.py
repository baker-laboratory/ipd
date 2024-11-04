import inspect
import traceback

import pydantic

import ipd

def maintest(crud, namespace):
    passed, failed = [], []
    with crud() as (backend, server, client, testclient):
        for name, fn in [(n, f) for n, f in namespace.items() if n.startswith('test_')]:
            backend._clear_all_data_for_testing_only()
            print('=' * 20, fn.__name__, '=' * 20)
            try:
                args = {p: locals()[p] for p in inspect.signature(fn).parameters}
                fn(**args)
                passed.append(name)
            except pydantic.ValidationError as e:
                print(e)
                print(e.errors())
                print(traceback.format_exc())
                failed.append(name)

    print(f'maintest {namespace["__file__"]}:')
    for p in passed:
        print(f'    PASS {p}')
    for f in failed:
        print(f'    FAIL {f}')
    ipd.dev.global_timer.report()
    return passed, failed
