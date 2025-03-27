import statistics
import pytest
import time
import random
from ipd._prelude.chrono import Chrono, chrono, chrono_class, checkpoint
# from ipd.dynamic_float_array import DynamicFloatArray

import ipd

config_test = ipd.Bunch(
    re_only=[
        #
    ],
    re_exclude=[
        #
    ],
)

def main():
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
        chrono=False,
    )

def measure_runtime(func):
    """Wrapper to track independent runtimes using time.perf_counter."""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        args[0].runtime[func.__name__].append(end - start)
        return result

    return wrapper

@chrono_class
class TestClass:

    def __init__(self):
        self.runtime = {"method1": [], "method2": [], "recursive": [], "generator": []}

    @measure_runtime
    @chrono
    def method1(self):
        time.sleep(random.uniform(0.01, 0.03))
        self.method2()

    @measure_runtime
    @chrono
    def method2(self):
        time.sleep(random.uniform(0.01, 0.03))
        self.recursive(random.randint(1, 3))

    @measure_runtime
    @chrono
    def recursive(self, depth):
        if depth > 0:
            time.sleep(random.uniform(0.01, 0.03))
            self.recursive(depth - 1)

    @measure_runtime
    @chrono
    def generator(self):
        for i in range(3):
            time.sleep(random.uniform(0.01, 0.03))
            yield i

@pytest.mark.xfail
def test_chrono_class():
    ipd.global_chrono = Chrono()
    instance = TestClass()

    instance.method1()
    instance.method2()
    list(instance.generator())

    report = ipd.global_chrono.report_dict()

    for method in instance.runtime:
        recorded_time = sum(instance.runtime[method])
        chrono_time = report.get(f"test_chrono_class.<locals>.TestClass.{method}", 0)
        assert abs(recorded_time -
                   chrono_time) < 0.01, f"Mismatch in {method}: {recorded_time} vs {chrono_time}"

def hypothesis_test_chrono_class():
    from hypothesis import given, strategies as st

    @given(
        st.lists(st.tuples(st.sampled_from(["method1", "method2", "recursive"]), st.integers(1, 3)),
                 min_size=5,
                 max_size=10))
    def run_test(call_sequence):
        ipd.global_chrono = Chrono(start=True, use_cython=random.choice([True, False]))
        instance = TestClass()

        for method, depth in call_sequence:
            if method == "recursive":
                instance.recursive(depth)
            else:
                getattr(instance, method)()

        ipd.global_chrono.stop()
        report = ipd.global_chrono.report_dict()
        for method in instance.runtime:
            recorded_time = sum(instance.runtime[method])
            chrono_time = report.get(f"test_chrono_class.<locals>.TestClass.{method}", 0)
            assert abs(recorded_time -
                       chrono_time) < 0.01, f"Mismatch in {method}: {recorded_time} vs {chrono_time}"

    run_test()

def test_chrono_func():

    @chrono
    def foo():
        time.sleep(0.001)

    foo()
    assert 'test_chrono_func.<locals>.foo' in ipd.global_chrono.checkpoints

def test_context():
    with Chrono() as t:
        t.checkpoint('foo')
        t.checkpoint('bar')
        t.checkpoint('baz')
    assert 'foo' in t.checkpoints
    assert 'bar' in t.checkpoints
    assert 'baz' in t.checkpoints

def allclose(a, b, atol):
    if isinstance(a, float):
        return abs(a - b) < atol
    for x, y in zip(a, b):
        if abs(a - b) > atol:
            return False
    return True

@pytest.mark.skip
def test_chrono():
    with Chrono() as chrono:
        time.sleep(0.02)
        chrono.checkpoint("foo")
        time.sleep(0.06)
        chrono.checkpoint("bar")
        time.sleep(0.04)
        chrono.checkpoint("baz")

    times = chrono.report_dict()
    assert allclose(times["foo"], 0.02, atol=0.05)
    assert allclose(times["bar"], 0.06, atol=0.05)
    assert allclose(times["baz"], 0.04, atol=0.05)

    times = chrono.report_dict(order="longest")
    assert list(times.keys()) == ["total", "bar", "baz", "foo"]

    times = chrono.report_dict(order="callorder")
    assert list(times.keys()) == ["foo", "bar", "baz", "total"]

    with pytest.raises(ValueError):
        chrono.report_dict(order="oarenstoiaen")

def aaaa(chrono=None):
    checkpoint(chrono=chrono, funcbegin=True)
    time.sleep(0.2)
    checkpoint(chrono=chrono)

##@chrono
def bbbb(**kw):
    time.sleep(0.2)

    t = Chrono()
    aaaa(t)
    areport = t.report(printme=False)

    t = Chrono()
    kw = ipd.Bunch(chrono=t)
    checkpoint('label', chrono=t)
    bbbb(**kw)
    breport = t.report(printme=False)

    print(areport)
    print(breport.replace("bbbb", "aaaa"))
    # print(breport.replace('bbbb', 'aaaa'))
    # assert areport.strip() == breport.replace('bbbb', 'aaaa').strip()

@pytest.mark.skip
def test_summary():
    with Chrono() as chrono:
        time.sleep(0.01)
        chrono.checkpoint("foo")
        time.sleep(0.03)
        chrono.checkpoint("foo")
        time.sleep(0.02)
        chrono.checkpoint("foo")
    times = chrono.report_dict(summary=sum)
    assert allclose(times["foo"], 0.06, atol=0.02)

    times = chrono.report_dict(summary=statistics.mean)
    assert allclose(times["foo"], 0.02, atol=0.01)

    times = chrono.report_dict(summary="mean")
    assert allclose(times["foo"], 0.02, atol=0.01)

    times = chrono.report_dict(summary="min")
    assert allclose(times["foo"], 0.01, atol=0.01)

    with pytest.raises(ValueError):
        chrono.report(summary="foo")

    with pytest.raises(ValueError):
        chrono.report(summary=1)

def test_chrono_interjection():
    with Chrono() as chrono:
        chrono.checkpoint("bar")
        chrono.checkpoint()
        chrono.checkpoint("baz")
        chrono.checkpoint("bar")
        chrono.checkpoint("foo")
    assert set(chrono.checkpoints) == {'foo', 'bar', 'baz'}
    assert len(chrono.checkpoints['foo']) == 1
    assert len(chrono.checkpoints['bar']) == 3
    assert len(chrono.checkpoints['baz']) == 1

def test_chrono_interjection_keyword():
    with Chrono() as chrono:
        chrono.checkpoint("foo")
        chrono.checkpoint(interject=True)
        chrono.checkpoint("baz")
        chrono.checkpoint("bar")
        chrono.checkpoint("foo")

    assert set(chrono.checkpoints) == {'foo', 'bar', 'baz'}
    assert len(chrono.checkpoints['foo']) == 2
    assert len(chrono.checkpoints['bar']) == 2
    assert len(chrono.checkpoints['baz']) == 1

if __name__ == '__main__':
    main()
