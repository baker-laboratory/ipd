import pytest
import time
from ipd.dev import Timer

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
        timed=False,
    )

def test_timed_func():

    @ipd.dev.timed
    def foo():
        time.sleep(0.001)

    foo()
    assert 'test_timer.py:test_timed_func.<locals>.foo$$$$' in ipd.dev.global_timer.checkpoints

def test_timed_class():

    @ipd.dev.timed
    class foo():

        def bar(self):
            time.sleep(0.001)

    foo().bar()
    ipd.icv(ipd.dev.global_timer.checkpoints)
    assert 'test_timer.py:test_timed_class.<locals>.foo.bar$$$$' in ipd.dev.global_timer.checkpoints

def test_context():
    context = ipd.dev.Timer()
    with context as t:
        t.checkpoint('foo')
        t.checkpoint('bar')
        t.checkpoint('baz')
    assert 'foo' in t.checkpoints
    assert 'bar' in t.checkpoints
    assert 'baz' in t.checkpoints

import statistics

import ipd

def allclose(a, b, atol):
    if isinstance(a, float):
        return abs(a - b) < atol
    for x, y in zip(a, b):
        if abs(a - b) > atol:
            return False
    return True

@pytest.mark.skip
def test_timer():
    with Timer() as timer:
        time.sleep(0.02)
        timer.checkpoint("foo")
        time.sleep(0.06)
        timer.checkpoint("bar")
        time.sleep(0.04)
        timer.checkpoint("baz")

    times = timer.report_dict()
    assert allclose(times["foo"], 0.02, atol=0.05)
    assert allclose(times["bar"], 0.06, atol=0.05)
    assert allclose(times["baz"], 0.04, atol=0.05)

    times = timer.report_dict(order="longest")
    assert list(times.keys()) == ["total", "bar", "baz", "foo"]

    times = timer.report_dict(order="callorder")
    assert list(times.keys()) == ["foo", "bar", "baz", "total"]

    with pytest.raises(ValueError):
        timer.report_dict(order="oarenstoiaen")

def aaaa(timer=None):
    ipd.dev.checkpoint(timer, funcbegin=True)
    time.sleep(0.2)
    ipd.dev.checkpoint(timer)

##@timed
def bbbb(**kw):
    time.sleep(0.2)

def test_auto():
    t = Timer()
    aaaa(t)
    areport = t.report(printme=False, scale=0)

    t = Timer()
    kw = ipd.dev.Bunch(timer=t)
    ipd.dev.checkpoint(t)
    bbbb(**kw)
    breport = t.report(printme=False, scale=0)

    print(areport)
    print(breport.replace("bbbb", "aaaa"))
    # print(breport.replace('bbbb', 'aaaa'))
    # assert areport.strip() == breport.replace('bbbb', 'aaaa').strip()

@pytest.mark.skip
def test_summary():
    with Timer() as timer:
        time.sleep(0.01)
        timer.checkpoint("foo")
        time.sleep(0.03)
        timer.checkpoint("foo")
        time.sleep(0.02)
        timer.checkpoint("foo")

    times = timer.report_dict(summary=sum)  # type: ignore
    assert allclose(times["foo"], 0.06, atol=0.02)

    times = timer.report_dict(summary=statistics.mean)  # type: ignore
    assert allclose(times["foo"], 0.02, atol=0.01)

    times = timer.report_dict(summary="mean")
    assert allclose(times["foo"], 0.02, atol=0.01)

    times = timer.report_dict(summary="min")
    assert allclose(times["foo"], 0.01, atol=0.01)

    with pytest.raises(ValueError):
        timer.report(summary="foo")

    with pytest.raises(ValueError):
        timer.report(summary=1)  # type: ignore

@pytest.mark.xfail
def test_timer_interjection():
    with Timer() as timer:
        timer.checkpoint("foo")
        timer.checkpoint()
        timer.checkpoint("bar")
        timer.checkpoint("bar")
        timer.checkpoint("foo")
    assert set(timer.checkpoints) == {'foo', 'bar', 'total'}
    assert len(timer.checkpoints['foo']) == 3
    assert len(timer.checkpoints['bar']) == 2

@pytest.mark.xfail
def test_timer_interjection_keyword():
    with Timer() as timer:
        timer.checkpoint("foo")
        timer.checkpoint(interject=True)
        timer.checkpoint("bar")
        timer.checkpoint("bar")
        timer.checkpoint("foo")
    assert set(timer.checkpoints) == {'foo', 'bar', 'total'}
    assert len(timer.checkpoints['foo']) == 3
    assert len(timer.checkpoints['bar']) == 2

if __name__ == '__main__':
    main()
