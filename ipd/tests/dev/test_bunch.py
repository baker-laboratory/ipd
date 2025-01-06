import _pickle
import os
import shutil
from argparse import Namespace

import pytest
import yaml

from ipd.bunch import *

def main():
    from tempfile import mkdtemp

    test_bunch_pickle(mkdtemp())
    test_bunch_init()
    test_bunch_sub()
    test_bunch_items()
    test_bunch_add()
    test_bunch_visit()
    test_bunch_strict()
    test_bunch_default()
    test_bunch_bugs()

    test_autosave(mkdtemp())
    test_autoreload(mkdtemp())

    print('test_bunch PASS')

def assert_saved_ok(b):
    with open(b._special['autosave']) as inp:
        b2 = make_autosave_hierarchy(yaml.load(inp, yaml.Loader))
    assert b == b2

@pytest.mark.fast
def test_autosave(tmpdir):  # sourcery skip: merge-list-append, merge-set-add
    fname = f'{tmpdir}/test.yaml'
    b = Bunch(a=1, b=[1, 2], c=[[[[0]]]], d={1, 3, 8}, e=([1], [2]))
    b = make_autosave_hierarchy(b, _autosave=fname)
    b.a = 7  # type: ignore
    assert_saved_ok(b)
    b.b.append(3)  # type: ignore
    assert_saved_ok(b)
    b.b[1] = 17  # type: ignore
    assert_saved_ok(b)
    b.c.append(100)  # type: ignore
    assert_saved_ok(b)
    b.c[0].append(200)  # type: ignore
    assert_saved_ok(b)
    b.c[0][0].append(300)  # type: ignore
    assert_saved_ok(b)
    b.c[0][0][0].append(400)  # type: ignore
    assert_saved_ok(b)
    b.c[0][0][0][0] = 7  # type: ignore
    assert_saved_ok(b)
    b.d.add(3)  # type: ignore
    assert_saved_ok(b)
    b.d.add(17)  # type: ignore
    assert_saved_ok(b)
    b.d.remove(1)  # type: ignore
    assert_saved_ok(b)
    b.d |= {101, 102, 10}  # type: ignore
    b.e[0].append(1000)  # type: ignore
    assert_saved_ok(b)
    b.e[1][0] = 2000  # type: ignore
    assert_saved_ok(b)
    b.f = 'f'  # type: ignore
    assert_saved_ok(b)
    delattr(b, 'f')
    assert_saved_ok(b)
    b.f = 'f2'  # type: ignore
    del b['f']  # type: ignore
    assert_saved_ok(b)
    b.g = []  # type: ignore
    b.g.append(283)  # type: ignore
    assert_saved_ok(b)
    b.h = set()  # type: ignore
    b.h.add('bar')  # type: ignore
    assert_saved_ok(b)
    b.i = [[[17]]]  # type: ignore
    b.i[0][0][0] = 18  # type: ignore
    assert_saved_ok(b)

def helper_test_autoreload(b, b2, tmpdir):
    fname = f'{tmpdir}/test.yaml'
    fname2 = f'{tmpdir}/test2.yaml'
    shutil.copyfile(fname, f'{fname2}.tmp')
    shutil.move(f'{fname2}.tmp', fname2)
    assert_saved_ok(b)
    assert b == b2
    assert set(os.listdir(tmpdir)) == {'test2.yaml', 'test.yaml'}

@pytest.mark.fast
def test_autoreload(tmpdir):
    fname = f'{tmpdir}/test.yaml'
    fname2 = f'{tmpdir}/test2.yaml'
    b = Bunch(a=1, b=[1, 2], c=[[[[0]]]], d={1, 3, 8}, e=([1], [2]))
    b = make_autosave_hierarchy(b, _autosave=fname)
    b2 = Bunch(_autoreload=fname2)
    b.a = 7  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.b.append(3)  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.b[1] = 17  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.c.append(100)  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.c[0].append(200)  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.c[0][0].append(300)  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.c[0][0][0].append(400)  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.c[0][0][0][0] = 7  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.d.add(3)  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.d.add(17)  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.d.remove(1)  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.d |= {101, 102, 10}  # type: ignore
    b.e[0].append(1000)  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.e[1][0] = 2000  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.f = 'f'  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    delattr(b, 'f')
    helper_test_autoreload(b, b2, tmpdir)
    b.f = 'f2'  # type: ignore
    del b['f']  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.g = []  # type: ignore
    b.g.append(283)  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.h = set()  # type: ignore
    b.h.add('bar')  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.i = [[[17]]]  # type: ignore
    b.i[0][0][0] = 18  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.bnch = Bunch()  # type: ignore
    b.bnch.c = 17  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    b.bnch._notify_changed('baz', 'biz')  # type: ignore
    helper_test_autoreload(b, b2, tmpdir)
    helper_test_autoreload(b, b2, tmpdir)

@pytest.mark.fast
def test_bunch_pickle(tmpdir):
    x = Bunch(dict(a=2, b="bee"))
    x.c = "see"
    with open(f"{tmpdir}/foo", "wb") as out:
        _pickle.dump(x, out)

    with open(f"{tmpdir}/foo", "rb") as inp:
        y = _pickle.load(inp)

    assert x == y
    assert y.a == 2
    assert y.b == "bee"
    assert y.c == "see"

@pytest.mark.fast
def test_bunch_init():
    b = Bunch(dict(a=2, b="bee"), _strict=False)  # type: ignore
    b2 = Bunch(b, _strict=False)  # type: ignore
    b3 = Bunch(c=3, d="dee", _strict=False, **b)  # type: ignore
    assert b.a == 2
    assert b.b == "bee"
    assert b.missing is None

    assert b.a == 2
    assert b.b == "bee"
    assert b.missing is None

    assert b3.a == 2
    assert b3.b == "bee"
    assert b3.missing is None
    assert b3.c == 3
    assert b3.d == "dee"

    foo = Namespace(a=1, b="c")
    b = Bunch(foo, _strict=False)  # type: ignore
    assert b.a == 1
    assert b.b == "c"
    assert b.missing is None

    b.missing = 7
    assert b.missing == 7
    b.missing = 8
    assert b.missing == 8

@pytest.mark.fast
def test_bunch_sub():
    b = Bunch(dict(a=2, b="bee"), _strict=False)  # type: ignore
    assert b.b == "bee"
    b2 = b.sub(b="bar")
    assert b2.b == "bar"  # type: ignore
    b3 = b.sub({"a": 4, "d": "dee"})
    assert b3.a == 4  # type: ignore
    assert b3.b == "bee"  # type: ignore
    assert b3.d == "dee"  # type: ignore
    assert b3.foobar is None  # type: ignore
    assert "a" in b
    b4 = b.sub(a=None)
    assert "a" not in b4
    assert "b" in b4

    b = Bunch(dict(a=2, b="bee"), _strict=False)  # type: ignore
    assert b.b == "bee"
    b2 = b.sub(b="bar", _onlynone=True)
    assert b2.b == "bee"  # type: ignore
    b3 = b.sub({"a": 4, "d": "dee"}, _onlynone=True)
    assert b3.a == 2  # type: ignore
    assert b3.b == "bee"  # type: ignore
    assert b3.d == "dee"  # type: ignore
    assert b3.foobar is None  # type: ignore
    assert "a" in b
    b4 = b.sub(a=None)
    assert "a" not in b4
    assert "b" in b4

@pytest.mark.fast
def test_bunch_items():
    b = Bunch(dict(item="item"))
    b.attr = "attr"
    assert len(list(b.items())) == 2
    assert list(b) == ["item", "attr"]
    assert list(b.keys()) == ["item", "attr"]
    assert list(b.values()) == ["item", "attr"]

@pytest.mark.fast
def test_bunch_add():
    b1 = Bunch(dict(a=2, b="bee", mergedint=4, mergedstr="b1"))
    b2 = Bunch(dict(a=2, c="see", mergedint=3, mergedstr="b2"))
    b1_plus_b2 = Bunch(a=4, b="bee", mergedint=7, mergedstr="b1b2", c="see")
    assert (b1 + b2) == b1_plus_b2

@pytest.mark.fast
def test_bunch_visit():
    count = 0

    def func(k, v, depth):
        # print('    ' * depth, k, type(v))
        nonlocal count
        count += 1
        if v == "b":
            return True
        return False

    b = Bunch(a="a", b="b", bnch=Bunch(foo="bar"))
    b.visit_remove_if(func)
    assert b == Bunch(a="a", bnch=Bunch(foo="bar"))
    assert count == 4

@pytest.mark.fast
def test_bunch_strict():
    b = Bunch(one=1, two=2, _strict=True)  # type: ignore
    assert len(b) == 2
    with pytest.raises(AttributeError):
        assert b.foo is None
    b.foo = 7
    assert b.foo == 7

    b2 = Bunch(one=1, two=2, _strict=False)  # type: ignore
    assert b2.foo is None

    with pytest.raises(ValueError) as e:
        b.clear = 8  # type: ignore
    assert str(e.value) == "clear is a reseved name for Bunch"
    with pytest.raises(ValueError) as e:
        b = Bunch(clear=True)
    assert str(e.value) == "clear is a reseved name for Bunch"

@pytest.mark.fast
def test_bunch_default():
    b = Bunch(foo="foo", _default=list, _strict=False)  # type: ignore
    assert b.foo == "foo"
    assert b.bar == list()
    assert b["foo"] == "foo"
    assert b["bar"] == list()

    b = Bunch(foo="foo", _default=list, _strict=False)  # type: ignore
    assert b.foo == "foo"
    assert b.bar == list()

    b = Bunch(foo="foo", _default=7, _strict=False)  # type: ignore
    assert b.foo == "foo"
    assert b.bar == 7

    b = Bunch(foo="foo", _default=list, _strict=True)  # type: ignore
    assert b.foo == "foo"
    with pytest.raises(AttributeError):
        b.bar

    b = Bunch(foo="foo", _strict=True)  # type: ignore
    assert b.foo == "foo"
    with pytest.raises(AttributeError):
        b.bar

@pytest.mark.fast
def test_bunch_bugs():
    with pytest.raises(ValueError) as e:
        showme_opts = Bunch(headless=0, spheres=0.0, showme=0, clear=True, weight=2)
    assert str(e.value) == "clear is a reseved name for Bunch"

if __name__ == "__main__":
    main()
