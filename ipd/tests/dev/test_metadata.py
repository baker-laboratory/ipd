import copy
import pytest
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
    )

def test_sync_metadata():

    objs = [type(f'Foo{i}', (), {})() for i in range(7)]
    for o, k, v in zip(objs, 'abcdefg', range(7)):
        ipd.dev.set_metadata(o, {k: v})
    ref = [dict(a=0), dict(b=1), dict(c=2), dict(d=3), dict(e=4), dict(f=5), dict(g=6)]
    assert list(map(ipd.dev.get_metadata, objs)) == ref
    ipd.dev.sync_metadata(*objs)
    ref2 = [
        ipd.Bunch(a=0, b=1, c=2, d=3, e=4, f=5, g=6),
        ipd.Bunch(b=1, a=0, c=2, d=3, e=4, f=5, g=6),
        ipd.Bunch(c=2, a=0, b=1, d=3, e=4, f=5, g=6),
        ipd.Bunch(d=3, a=0, b=1, c=2, e=4, f=5, g=6),
        ipd.Bunch(e=4, a=0, b=1, c=2, d=3, f=5, g=6),
        ipd.Bunch(f=5, a=0, b=1, c=2, d=3, e=4, g=6),
        ipd.Bunch(g=6, a=0, b=1, c=2, d=3, e=4, f=5)
    ]
    assert list(map(ipd.dev.get_metadata, objs)) == ref2

def test_metadata_decorator__init__():

    @ipd.dev.holds_metadata
    class Foo:

        def __init__(self, a, b):
            self.a, self.b = a, b

    obj = Foo(1, b=2)
    assert obj.get_metadata() == {}
    assert obj.a == 1 and obj.b == 2

    with pytest.raises(TypeError):
        obj = Foo(1, b=2, c=3)
    obj = Foo(1, b=2, _c=3)
    assert obj.a == 1 and obj.b == 2
    assert obj.get_metadata() == {'c': 3}

def test_metadata_decorator():

    @ipd.dev.holds_metadata
    class Foo:
        pass

    obj = Foo()
    obj.set_metadata({'a': 1, 'b': 2})
    assert obj.get_metadata() == {'a': 1, 'b': 2}
    obj.set_metadata({'c': 3})
    assert obj.get_metadata() == {'a': 1, 'b': 2, 'c': 3}

    obj2 = Foo()
    obj2.set_metadata({'x': 10})
    obj.sync_metadata(obj2)
    assert obj.get_metadata() == {'a': 1, 'b': 2, 'c': 3, 'x': 10}
    assert obj2.get_metadata() == {'a': 1, 'b': 2, 'c': 3, 'x': 10}

def test_metadata_copy():

    @ipd.dev.holds_metadata
    class Foo:
        pass

    a = Foo()
    a.set_metadata({'a': 1, 'b': 2})
    b = copy.copy(a)
    b.meta.c = 3
    assert a.get_metadata() == {'a': 1, 'b': 2}
    assert b.get_metadata() == {'a': 1, 'b': 2, 'c': 3}

def test_doctest_issue():

    class Example:
        pass

    obj = Example()
    ipd.dev.set_metadata(obj, {'key': 'value'})  # doctest:+SKIP
    assert isinstance(obj.__ipd_metadata__, ipd.Bunch)
    assert 'value' == ipd.dev.get_metadata(obj).key
    obj2 = Example()
    assert ipd.Bunch() == ipd.dev.get_metadata(obj2)

if __name__ == '__main__':
    main()
