import _pickle
import os
import shutil
from argparse import Namespace

import pytest
import yaml

import evn
from evn.decon.bunch import bunchfind, Bunch, make_autosave_hierarchy

config_test = Bunch(
    re_only=[
        # 'test_ewise_equal'
    ],
    re_exclude=[],
)


def main():
    evn.testing.quicktest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
    )


def assert_saved_ok(b):
    with open(b._config['autosave']) as inp:
        b2 = make_autosave_hierarchy(yaml.load(inp, yaml.Loader))
    assert b == b2


def test_bunch_split_nest():
    test = Bunch(a='a', _split=' ')
    test['foo bar baz'] = 8
    test['foo bar baz2'] = 7
    test['foo bar2 baz'] = 6
    with pytest.raises(TypeError):
        test['foo bar2 baz but'] = 5
    assert test.foo.bar.baz == 8
    assert test.foo.bar.baz2 == 7
    assert test.foo.bar2.baz == 6
    # assert test.foo.bar2.baz.but == 5


def test_bunch_split_space():
    test = Bunch(a='a', _split=' ')
    test['foo bar baz'] = 8
    test['foo bar2 baz'] = 6
    assert test.foo.bar.baz == 8
    assert test.foo.bar2.baz == 6


def test_bunch_split_space_get():
    test = Bunch(a='a', _split=' ')
    test['foo bar baz'] = 8
    test['foo bar2 baz'] = 6
    assert test.foo.bar.baz == 8
    assert test.foo.bar2.baz == 6
    assert test._get_split('foo bar baz') == 8
    assert test._get_split('foo bar2 baz') == 6


def test_bunch_split_dot():
    test = Bunch(a='a', _split='.')
    test['foo.bar.baz'] = 8
    test['foo.bar2.baz'] = 8
    assert test.foo.bar.baz == 8
    assert test.foo.bar2.baz == 8


def test_autosave(tmpdir):  # sourcery skip: merge-list-append, merge-set-add
    fname = f'{tmpdir}/test.yaml'
    b = Bunch(a=1, b=[1, 2], c=[[[[0]]]], d={1, 3, 8}, e=([1], [2]))
    b = make_autosave_hierarchy(b, _autosave=fname)
    b.a = 7
    assert_saved_ok(b)
    b.b.append(3)
    assert_saved_ok(b)
    b.b[1] = 17
    assert_saved_ok(b)
    b.c.append(100)
    assert_saved_ok(b)
    b.c[0].append(200)
    assert_saved_ok(b)
    b.c[0][0].append(300)
    assert_saved_ok(b)
    b.c[0][0][0].append(400)
    assert_saved_ok(b)
    b.c[0][0][0][0] = 7
    assert_saved_ok(b)
    b.d.add(3)
    assert_saved_ok(b)
    b.d.add(17)
    assert_saved_ok(b)
    b.d.remove(1)
    assert_saved_ok(b)
    b.d |= {101, 102, 10}
    b.e[0].append(1000)
    assert_saved_ok(b)
    b.e[1][0] = 2000
    assert_saved_ok(b)
    b.f = 'f'
    assert_saved_ok(b)
    delattr(b, 'f')
    assert_saved_ok(b)
    b.f = 'f2'
    del b['f']
    assert_saved_ok(b)
    b.g = []
    b.g.append(283)
    assert_saved_ok(b)
    b.h = set()
    b.h.add('bar')
    assert_saved_ok(b)
    b.i = [[[17]]]
    b.i[0][0][0] = 18
    assert_saved_ok(b)


def helper_test_autoreload(b, b2, tmpdir):
    fname = f'{tmpdir}/test.yaml'
    fname2 = f'{tmpdir}/test2.yaml'
    shutil.copyfile(fname, f'{fname2}.tmp')
    shutil.move(f'{fname2}.tmp', fname2)
    assert_saved_ok(b)
    assert b == b2
    assert set(os.listdir(tmpdir)) == {'test2.yaml', 'test.yaml'}


def test_autoreload(tmpdir):
    fname = f'{tmpdir}/test.yaml'
    fname2 = f'{tmpdir}/test2.yaml'
    b = Bunch(a=1, b=[1, 2], c=[[[[0]]]], d={1, 3, 8}, e=([1], [2]))
    b = make_autosave_hierarchy(b, _autosave=fname)
    b2 = Bunch(_autoreload=fname2)
    b.a = 7
    helper_test_autoreload(b, b2, tmpdir)
    b.b.append(3)
    helper_test_autoreload(b, b2, tmpdir)
    b.b[1] = 17
    helper_test_autoreload(b, b2, tmpdir)
    b.c.append(100)
    helper_test_autoreload(b, b2, tmpdir)
    b.c[0].append(200)
    helper_test_autoreload(b, b2, tmpdir)
    b.c[0][0].append(300)
    helper_test_autoreload(b, b2, tmpdir)
    b.c[0][0][0].append(400)
    helper_test_autoreload(b, b2, tmpdir)
    b.c[0][0][0][0] = 7
    helper_test_autoreload(b, b2, tmpdir)
    b.d.add(3)
    helper_test_autoreload(b, b2, tmpdir)
    b.d.add(17)
    helper_test_autoreload(b, b2, tmpdir)
    b.d.remove(1)
    helper_test_autoreload(b, b2, tmpdir)
    b.d |= {101, 102, 10}
    b.e[0].append(1000)
    helper_test_autoreload(b, b2, tmpdir)
    b.e[1][0] = 2000
    helper_test_autoreload(b, b2, tmpdir)
    b.f = 'f'
    helper_test_autoreload(b, b2, tmpdir)
    delattr(b, 'f')
    helper_test_autoreload(b, b2, tmpdir)
    b.f = 'f2'
    del b['f']
    helper_test_autoreload(b, b2, tmpdir)
    b.g = []
    b.g.append(283)
    helper_test_autoreload(b, b2, tmpdir)
    b.h = set()
    b.h.add('bar')
    helper_test_autoreload(b, b2, tmpdir)
    b.i = [[[17]]]
    b.i[0][0][0] = 18
    helper_test_autoreload(b, b2, tmpdir)
    b.bnch = Bunch()
    b.bnch.c = 17
    helper_test_autoreload(b, b2, tmpdir)
    b.bnch._notify_changed('baz', 'biz')
    helper_test_autoreload(b, b2, tmpdir)
    helper_test_autoreload(b, b2, tmpdir)


def test_bunch_pickle(tmpdir):
    x = Bunch(dict(a=2, b='bee'))
    x.c = 'see'
    with open(f'{tmpdir}/foo', 'wb') as out:
        _pickle.dump(x, out)

    with open(f'{tmpdir}/foo', 'rb') as inp:
        y = _pickle.load(inp)

    assert x == y
    assert y.a == 2
    assert y.b == 'bee'
    assert y.c == 'see'


def test_bunch_init():
    b = Bunch(dict(a=2, b='bee'), _strict=False)
    b2 = Bunch(b, _strict=False)
    b3 = Bunch(c=3, d='dee', _strict=False, **b)
    assert b.a == 2
    assert b.b == 'bee'
    assert b.missing is None

    assert b.a == 2
    assert b.b == 'bee'
    assert b.missing is None

    assert b3.a == 2
    assert b3.b == 'bee'
    assert b3.missing is None
    assert b3.c == 3
    assert b3.d == 'dee'

    foo = Namespace(a=1, b='c')
    b = Bunch(foo, _strict=False)
    assert b.a == 1
    assert b.b == 'c'
    assert b.missing is None

    b.missing = 7
    assert b.missing == 7
    b.missing = 8
    assert b.missing == 8


def test_bunch_sub():
    b = Bunch(dict(a=2, b='bee'), _strict=False)
    assert b.b == 'bee'
    b2 = b.sub(b='bar')
    assert b2.b == 'bar'
    b3 = b.sub({'a': 4, 'd': 'dee'})
    assert b3.a == 4
    assert b3.b == 'bee'
    assert b3.d == 'dee'
    assert b3.foobar is None
    assert 'a' in b
    b4 = b.sub(a=None)
    assert 'a' not in b4
    assert 'b' in b4

    b = Bunch(dict(a=2, b='bee'), _strict=False)
    assert b.b == 'bee'
    b2 = b.sub(b='bar', _onlynone=True)
    assert b2.b == 'bee'
    b3 = b.sub({'a': 4, 'd': 'dee'}, _onlynone=True)
    assert b3.a == 2
    assert b3.b == 'bee'
    assert b3.d == 'dee'
    assert b3.foobar is None
    assert 'a' in b
    b4 = b.sub(a=None)
    assert 'a' not in b4
    assert 'b' in b4


def test_bunch_items():
    b = Bunch(dict(item='item'))
    b.attr = 'attr'
    assert len(list(b.items())) == 2
    assert list(b) == ['item', 'attr']
    assert list(b.keys()) == ['item', 'attr']
    assert list(b.values()) == ['item', 'attr']


def test_bunch_add():
    b1 = Bunch(dict(a=2, b='bee', mergedint=4, mergedstr='b1'))
    b2 = Bunch(dict(a=2, c='see', mergedint=3, mergedstr='b2'))
    b1_plus_b2 = Bunch(a=4, b='bee', mergedint=7, mergedstr='b1b2', c='see')
    assert (b1 + b2) == b1_plus_b2


def test_bunch_visit():
    count = 0

    def func(k, v, depth):
        # print('    ' * depth, k, type(v))
        nonlocal count
        count += 1
        if v == 'b':
            return True
        return False

    b = Bunch(a='a', b='b', bnch=Bunch(foo='bar'))
    b.visit_remove_if(func)
    assert b == Bunch(a='a', bnch=Bunch(foo='bar'))
    assert count == 4


def test_bunch_strict():
    b = Bunch(one=1, two=2, _strict=True)
    assert len(b) == 2
    with pytest.raises(AttributeError):
        assert b.foo is None
    b.foo = 7
    assert b.foo == 7

    b2 = Bunch(one=1, two=2, _strict=False)
    assert b2.foo is None

    with pytest.raises(ValueError) as e:
        b.clear = 8
    assert str(e.value) == 'clear is a reseved name for Bunch'
    b = Bunch(clear=True)
    assert 'clear_' in b
    assert 'clear' not in b


def test_bunch_default():
    b = Bunch(foo='foo', _default=list, _strict=False)
    assert b.foo == 'foo'
    assert b.bar == []
    assert b['foo'] == 'foo'
    assert b['bar'] == []

    b = Bunch(foo='foo', _default=list, _strict=False)
    assert b.foo == 'foo'
    assert b.bar == []

    b = Bunch(foo='foo', _default=7, _strict=False)
    assert b.foo == 'foo'
    assert b.bar == 7

    b = Bunch(foo='foo', _default=list, _strict=True)
    assert b.foo == 'foo'
    # with pytest.raises(AttributeError):
    assert b.bar == []

    b = Bunch(foo='foo', _strict=True)
    assert b.foo == 'foo'
    with pytest.raises(AttributeError):
        b.bar


def test_bunch_bugs():
    # with pytest.raises(ValueError) as e:
    showme_opts = Bunch(headless=0,
                        spheres=0.0,
                        showme=0,
                        clear=True,
                        weight=2)
    assert 'clear_' in showme_opts
    assert 'clear' not in showme_opts
    # assert str(e.value) == "clear is a reseved name for Bunch"


def test_bunch_dict_reserved():
    b = Bunch(values='foo')
    assert b.values_ == 'foo'


def test_bunch_zip():
    zipped = evn.zipmaps(Bunch(a=1, b=2), Bunch(a='a', b='b'))
    assert isinstance(zipped, Bunch)
    assert zipped == Bunch(a=(1, 'a'), b=(2, 'b'))


def test_bunch_zip_missing():
    zipped = evn.zipmaps(Bunch(a=1, b=2), Bunch(a='a', b='b', c='c'))
    assert isinstance(zipped, Bunch)
    assert zipped == Bunch(c=(evn.NA, 'c'), a=(1, 'a'), b=(2, 'b'))


def test_bunch_zip_order():
    zipped = evn.zipmaps(Bunch(a=2, b=1),
                         Bunch(a='a', b='b', c='c'),
                         order='val')
    assert isinstance(zipped, Bunch)
    assert tuple(zipped.keys()) == ('c', 'b', 'a')


def test_search_basic_match():
    data = Bunch({'name': 'Alice', 'age': 30, 'location': 'New York'})
    result = bunchfind(data, 'name')
    assert result == {'name': 'Alice'}


def test_search_nested_match():
    data = {
        'person': {
            'name': 'Bob',
            'details': {
                'age': 25,
                'location_name': 'Los Angeles'
            }
        }
    }
    result = bunchfind(data, 'name')
    assert result == {
        'person.name': 'Bob',
        'person.details.location_name': 'Los Angeles'
    }


def test_search_nested_match_recursive():
    data = {
        'person': {
            'name': 'Bob',
            'details': {
                'age': 25,
                'location_name': 'Los Angeles'
            }
        }
    }
    data['self'] = data
    result = bunchfind(data, 'name')
    assert result == {
        'person.name': 'Bob',
        'person.details.location_name': 'Los Angeles'
    }


def test_search_no_match():
    data = Bunch({'name': 'Charlie', 'age': 40})
    result = bunchfind(data, 'location')
    assert result == {}


def test_search_partial_key_match():
    data = {
        'username': 'admin',
        'user_id': 1234,
        'details': {
            'profile_name': 'Admin'
        }
    }
    result = bunchfind(data, 'name')
    assert result == {'username': 'admin', 'details.profile_name': 'Admin'}


def test_search_empty_dict():
    data = Bunch()
    result = bunchfind(data, 'key')
    assert result == {}


def test_bunch_underscore():
    data = Bunch(_foo='bar', foo_='baz')
    assert data._foo == 'bar'
    assert data.foo_ == 'baz'


def test_bunch_merge():
    a = Bunch(b=Bunch(c=Bunch(d=1)))
    b = Bunch(b=Bunch(c=Bunch(b=2)))
    a._merge(b)
    assert a.b.c == Bunch(d=1, b=2)
    c = Bunch(b=Bunch(c=Bunch(b=8)))
    a._merge(c)
    assert a.b.c == Bunch(d=1, b=8)


def test_contains_flagmode():
    flags = Bunch(_default=lambda k: k.startswith('a'),
                  _frozen=True,
                  _flagmode=True)
    assert flags.a
    assert flags.abcdesf
    assert not flags.bnes
    assert 'africa' in flags
    assert 'bermuda' not in flags
    with pytest.raises(ValueError):
        flags.a = 7


def test_bunch_copy():
    b = Bunch(a=1, b=Bunch(c=Bunch(d='foo')), c=3)
    strb = str(b)
    b2 = b.copy()
    assert b == b2
    b2.b.c.d = 7
    assert b != b2
    assert str(b2) != strb


if __name__ == '__main__':
    main()
