from pprint import pprint
import statistics
import pytest
import time
import random
from evn._prelude.chrono import Chrono, chrono, TimerScope
# from evn.dynamic_float_array import DynamicFloatArray

import evn

# orig_name = __name__
# __name__ = 'test_chrono'

config_test = evn.Bunch(
    re_only=[
        #
    ],
    re_exclude=[
        #
    ],
)

def main():
    evn.testing.quicktest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
        chrono=False,
    )

def test_chronometer():
    assert evn.chronometer

class FuncNest:

    def __init__(self):
        self.runtime = {'method1': [], 'method2': [], 'recursive': [], 'generator': []}

    @chrono
    def method1(self):
        start = time.perf_counter()
        time.sleep(0.01)  # random.uniform(0.01, 0.03))
        # print(self)
        self.runtime['method1'].append(time.perf_counter() - start)
        self.method2()
        start = time.perf_counter()
        time.sleep(0.01)  # random.uniform(0.01, 0.03))
        self.runtime['method1'].append(time.perf_counter() - start)

    @chrono
    def method2(self):
        start = time.perf_counter()
        time.sleep(0.01)  # random.uniform(0.01, 0.03))
        self.runtime['method2'].append(time.perf_counter() - start)
        self.recursive(random.randint(1, 3))
        start = time.perf_counter()
        time.sleep(0.01)  # random.uniform(0.01, 0.03))
        self.runtime['method2'].append(time.perf_counter() - start)

    @chrono
    def recursive(self, depth):
        if not depth:
            return
        start = time.perf_counter()
        time.sleep(0.01)  # random.uniform(0.01, 0.03))
        self.runtime['recursive'].append(time.perf_counter() - start)
        self.recursive(depth - 1)
        start = time.perf_counter()
        time.sleep(0.01)  # random.uniform(0.01, 0.03))
        self.runtime['recursive'].append(time.perf_counter() - start)

    # @chrono
    # def generator(self):
    #     start = time.perf_counter()
    #     time.sleep(0.01)  # random.uniform(0.01, 0.03))
    #     self.runtime['generator'].append(time.perf_counter() - start)
    #     for i in range(5):
    #         start = time.perf_counter()
    #         time.sleep(0.01)  # random.uniform(0.01, 0.03))
    #         self.runtime['generator'].append(time.perf_counter() - start)
    #         yield i
    #         start = time.perf_counter()
    #         time.sleep(0.01)  # random.uniform(0.01, 0.03))
    #         self.runtime['generator'].append(time.perf_counter() - start)

FuncNest.__module__ = 'test_chrono'
FuncNest.method1.__module__ = 'test_chrono'
FuncNest.method2.__module__ = 'test_chrono'
FuncNest.recursive.__module__ = 'test_chrono'

# FuncNese.generator.__module__ = 'test_chrono'

# @pyt8est.mark.xfail
def test_chrono_nesting():
    instance = FuncNest()
    instance.method1()
    # assert list(instance.generator()) == [0, 1, 2, 3, 4]
    report = evn.chronometer.report_dict()
    pprint(report)
    for method in 'method1 method2 recursive'.split():
        try:
            recorded_time = sum(instance.runtime[method])
            print(report.keys())
            chrono_time = report[f'test_chrono.FuncNest.{method}']
            err = f'Mismatch in {method}, internal: {recorded_time} vs chrono: {chrono_time}'
            assert abs(recorded_time - chrono_time) < 0.005, err
        except KeyError:
            assert 0, f'missing key {method}'

    assert evn.chronometer.scopestack[-1].name == 'main'

def test_chrono_func():
    timer = Chrono()

    @chrono(chrono=timer)
    def foo():
        time.sleep(0.001)

    foo.__module__ = 'test_chrono'
    foo()
    assert 'test_chrono.test_chrono_func.foo' in timer.times
    assert len(timer.times['test_chrono.test_chrono_func.foo']) == 1
    print(timer.times['test_chrono.test_chrono_func.foo'])
    assert sum(timer.times['test_chrono.test_chrono_func.foo']) >= 0.001

def test_scope():
    with Chrono() as t:
        t.enter_scope('foo')
        t.enter_scope('bar')
        t.enter_scope('baz')
        t.exit_scope('baz')
        t.exit_scope('bar')
        t.exit_scope('foo')
    assert 'foo' in t.times
    assert 'bar' in t.times
    assert 'baz' in t.times

def allclose(a, b, atol):
    if isinstance(a, float): return abs(a - b) < atol
    return all(abs(a - b) <= atol for x, y in zip(a, b))

def test_chrono_checkpoint():
    with Chrono() as chrono:
        time.sleep(0.02)
        chrono.checkpoint('foo')
        time.sleep(0.06)
        chrono.checkpoint('bar')
        time.sleep(0.04)
        chrono.checkpoint('baz')

    times = chrono.report_dict()
    assert allclose(times['foo'], 0.02, atol=0.05)
    assert allclose(times['bar'], 0.06, atol=0.05)
    assert allclose(times['baz'], 0.04, atol=0.05)

    times = chrono.report_dict(order='longest')
    assert list(times.keys()) == ['total', 'bar', 'baz', 'foo', 'Chrono']

    times = chrono.report_dict(order='callorder')
    print(times.keys())
    assert list(times.keys()) == ['foo', 'bar', 'baz', 'Chrono', 'total']

    with pytest.raises(ValueError):
        chrono.report_dict(order='oarenstoiaen')

def chrono_deco_func():
    time.sleep(0.01)

chrono_deco_func.__module__ = 'test_chrono'
chrono_deco_func = chrono(chrono_deco_func)

def test_chrono_deco_func():
    evn.chronometer.clear()
    for _ in range(3):
        chrono_deco_func()

    times = evn.chronometer.find_times('test_chrono.chrono_deco_func')
    for t in times:
        assert 0.01 <= t < 0.012
    assert 'test_chrono.chrono_deco_func' in evn.chronometer.times

def chrono_deco_func2():
    time.sleep(0.005)
    chrono_deco_func()
    time.sleep(0.005)

chrono_deco_func2.__module__ = 'test_chrono'
chrono_deco_func2 = chrono(chrono_deco_func2)

def chrono_deco_func3():
    time.sleep(0.005)
    chrono_deco_func2()
    time.sleep(0.005)

chrono_deco_func3.__module__ = 'test_chrono'
chrono_deco_func3 = chrono(chrono_deco_func3)

def test_chrono_deco_func_nest():
    evn.chronometer.clear()
    N = 1
    for _ in range(N):
        chrono_deco_func3()
    times = evn.chronometer.find_times('test_chrono.chrono_deco_func')
    times2 = evn.chronometer.find_times('test_chrono.chrono_deco_func2')
    times3 = evn.chronometer.find_times('test_chrono.chrono_deco_func3')
    print(evn.chronometer.times.keys())
    assert N == len(times) == len(times2) == len(times3)
    for t, t2, t3 in zip(times, times2, times3):
        assert 0.01 <= t < 0.012
        assert 0.01 <= t2 < 0.012
        assert 0.01 <= t3 < 0.012
    assert 'test_chrono.chrono_deco_func' in evn.chronometer.times
    assert 'test_chrono.chrono_deco_func2' in evn.chronometer.times

def test_summary():
    with Chrono() as chrono:
        chrono.enter_scope('foo')
        time.sleep(0.01)
        chrono.exit_scope('foo')
        chrono.enter_scope('foo')
        time.sleep(0.03)
        chrono.exit_scope('foo')
        chrono.enter_scope('foo')
        time.sleep(0.02)
        chrono.exit_scope('foo')
    times = chrono.report_dict(summary=sum)
    assert allclose(times['foo'], 0.06, atol=0.02)

    times = chrono.report_dict(summary=statistics.mean)
    assert allclose(times['foo'], 0.02, atol=0.01)

    times = chrono.report_dict(summary=min)
    assert allclose(times['foo'], 0.01, atol=0.01)

    with pytest.raises(TypeError):
        chrono.report(summary='foo')

    with pytest.raises(TypeError):
        chrono.report(summary=1)

def test_chrono_stop_behavior():
    chrono = Chrono()
    chrono.enter_scope('foo')
    chrono.exit_scope('foo')
    chrono.stop()
    assert chrono.stopped
    with pytest.raises(AssertionError):
        chrono.enter_scope('bar')
    with pytest.raises(AssertionError):
        chrono.store_finished_scope(TimerScope('baz'))

def test_scope_mismatch():
    chrono = Chrono()
    chrono.enter_scope('foo')
    with pytest.raises(AssertionError, match='exiting scope: bar doesnt match: foo'):
        chrono.exit_scope('bar')

def test_scope_name_from_object():
    chrono = Chrono()

    class Dummy:
        pass

    name = chrono.scope_name(Dummy)
    assert isinstance(name, str)
    assert 'Dummy' in name

def test_report_string_return():
    with Chrono() as chrono:
        chrono.enter_scope('foo')
        time.sleep(0.01)
        chrono.exit_scope('foo')
    report = chrono.report(printme=False)
    assert isinstance(report, str)
    assert 'foo' in report

@pytest.mark.skip
def test_generator_deco():
    calls = []

    @chrono
    def gen():
        yield 1
        yield 2

    with pytest.raises(ValueError):
        for x in gen():
            calls.append(x)

    assert calls == [1, 2]
    print(evn.chronometer.times)

@pytest.mark.skip
def test_generator_with_exception():
    calls = []

    @chrono
    def gen():
        yield 1
        yield 2
        raise ValueError('boom')

    with pytest.raises(ValueError):
        for x in gen():
            calls.append(x)

    assert calls == [1, 2]
    print(evn.chronometer.times)

def test_nested_chrono_scopes():
    with Chrono() as outer:
        outer.enter_scope('outer')
        time.sleep(0.005)
        with Chrono() as inner:
            inner.enter_scope('inner')
            time.sleep(0.005)
            inner.exit_scope('inner')
        outer.exit_scope('outer')
    assert 'outer' in outer.times
    assert 'inner' in inner.times

def test_report_dict_bad_order():
    chrono = Chrono()
    with pytest.raises(ValueError):
        chrono.report_dict(order='invalid')

def test_chrono_context_manager():
    with Chrono('foo') as c:
        time.sleep(0.01)
    assert 'foo' in c.times
    assert 0.01 <= c.times['foo'][0] < 0.012

def test_scope_context_manager():
    c = Chrono()
    with c.scope('foo'):
        time.sleep(0.01)
    assert 'foo' in c.times
    assert 0.01 <= c.times['foo'][0] < 0.012

def test_nested_scope_context_manager():
    c = Chrono()
    with c.scope('foo'):
        time.sleep(0.005)
        with c.scope('bar'):
            time.sleep(0.005)
            with c.scope('baz'):
                time.sleep(0.01)
            time.sleep(0.005)
        time.sleep(0.005)
    for n in 'foo bar baz'.split():
        assert n in c.times
        assert 0.01 <= c.times[n][0] < 0.012

if __name__ == '__main__':
    main()
