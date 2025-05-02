import types
import builtins
import pytest
import evn
import evn.testing as et

def main():
    # test_TestConfig_defaults()
    # test_TestConfig_detect_fixtures()
    # test_TestResult_runtime_and_items()
    # test__test_func_ok()
    # test__test_class_ok()
    # test_collect_and_run_tests()
    # test_main_and_print_result()
    # test_dryrun_mode()
    # test_nocapture_mode()
    # test_xfail_detection()
    # test_skip_detection()
    # test_setUp_tearDown_called()
    # test_parametrized_test_handling()

    evn.testing.quicktest(globals())

def test_TestConfig_defaults():
    cfg = et.TestConfig()
    assert cfg.nofail is False
    assert cfg.verbose is False
    assert cfg.checkxfail is False
    assert cfg.timed is True
    assert isinstance(cfg.fixtures, dict)
    assert callable(cfg.setup)
    assert callable(cfg.funcsetup)
    assert callable(cfg.context)
    assert cfg.use_test_classes is True
    assert cfg.dryrun is False

def test_TestConfig_detect_fixtures():

    def fake_fixture():
        pass

    fake_fixture._pytestfixturefunction = True
    fake_fixture.__wrapped__ = lambda: 'wrapped'
    ns = {'fake_fixture': fake_fixture}
    cfg = et.TestConfig()
    cfg.detect_fixtures(ns)
    assert 'fake_fixture' in cfg.fixtures
    assert cfg.fixtures['fake_fixture'] == 'wrapped'

def test_TestResult_runtime_and_items():
    r = et.TestResult(passed=['t1'],
                      failed=['t2'],
                      errored=[],
                      xfailed=[],
                      skipexcn=[],
                      _runtime={
                          't1': 1.23,
                          't2': 0.45
                      })
    assert r.runtime('t1') == 1.23
    items = dict(r.items())
    assert 'passed' in items and 'failed' in items

def test__test_func_ok():

    def test_abc():
        pass

    assert et.test_func_ok('test_abc', test_abc)

def test__test_class_ok():

    class TestThing:
        pass

    assert et.test_class_ok('TestThing', TestThing)

def test_collect_and_run_tests():
    state = {}

    def test_foo(tmpdir=None):
        assert tmpdir is not None
        state['ran'] = True

    ns = {'test_foo': test_foo}
    cfg = et.TestConfig()
    funcs, teardown = et.collect_tests(ns, cfg)
    assert len(funcs) == 1
    result = et.run_tests(funcs, cfg, {})
    assert 'test_foo' in result.passed
    assert state.get('ran') is True

def test_main_and_print_result():
    with et.CapSys() as capsys:
        ran = {}

        def test_foo():
            ran['yes'] = True

        ns = {'test_foo': test_foo, '__file__': 'dummy.py'}
        res = et.quicktest(ns, verbose=True, check_xfail=True)
        captured = capsys.readouterr()
        print(captured.out)
        assert 'PASSED' in captured.out
        assert 'test_foo' in captured.out
        assert ran['yes']

def test_dryrun_mode():
    flag = {'called': False}

    def test_func():
        flag['called'] = True

    cfg = et.TestConfig(dryrun=True)
    funcs, _ = et.collect_tests({'test_func': test_func}, cfg)
    result = et.run_tests(funcs, cfg, {})
    assert 'test_func' not in result.passed
    assert not flag['called']

def test_nocapture_mode():
    with et.CapSys() as capsys:

        def test_func():
            print("visible output")

        cfg = et.TestConfig(nocapture=['test_func'])
        funcs, _ = et.collect_tests({'test_func': test_func}, cfg)
        result = et.run_tests(funcs, cfg, {})
        out = capsys.readouterr().out
        assert "visible output" in out

def test_xfail_detection():

    @pytest.mark.xfail
    def test_func():
        assert False

    cfg = et.TestConfig(checkxfail=False)
    funcs, _ = et.collect_tests({'test_func': test_func}, cfg)
    result = et.run_tests(funcs, cfg, {})
    print(result)
    assert 'test_func' in result.xfailed

def test_skip_detection():

    def test_func():
        pytest.skip("skip")

    cfg = et.TestConfig()
    funcs, _ = et.collect_tests({'test_func': test_func}, cfg)
    result = et.run_tests(funcs, cfg, {})
    assert 'test_func' in result.skipexcn

def test_setUp_tearDown_called():
    trace = []

    class TestThing:

        def setUp(self):
            trace.append('setup')

        def tearDown(self):
            trace.append('teardown')

        def test_run(self):
            trace.append('run')

    ns = {'TestThing': TestThing}
    cfg = et.TestConfig()
    funcs, teardown = et.collect_tests(ns, cfg)
    result = et.run_tests(funcs, cfg, {})
    for fn in teardown:
        fn()
    assert trace == ['setup', 'run', 'teardown']
    assert 'TestThing.test_run' in result.passed

def test_parametrized_test_handling():

    @pytest.mark.parametrize('x', [(1, ), (5, ), (9, )])
    def test_func(x):
        assert x < 10

    cfg = et.TestConfig()
    funcs, _ = et.collect_tests({'test_func': test_func}, cfg)
    result = et.run_tests(funcs, cfg, {})
    assert result.passed.count('test_func') == 3

if __name__ == '__main__':
    main()
