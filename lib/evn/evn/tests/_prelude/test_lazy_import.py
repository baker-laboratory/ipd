import sys

import pytest

import evn
from evn._prelude.lazy_import import _LazyModule, lazyimport

testconfig = evn.Bunch(nocapture=['test_broken_package'], )

def main():
    evn.testing.quicktest(namespace=globals(), config=testconfig)

def test_broken_package():
    if 'doctest' not in sys.modules:
        borked = lazyimport('evn.tests._prelude.broken_py_file')
        with pytest.raises(ImportError):
            borked.foo

def test_maybeimport():
    re = evn.maybeimport('re')
    assert re is sys.modules['re']
    missing = evn.maybeimport('noufuomemioixecmeiorutnaufoinairesvoraisevmraoui')
    assert not missing
    missing = evn.maybeimports('noufuomem ioixecmeiorutnaufoina iresvoraisevmraoui')
    assert not any(missing)

def test_lazyimport_re():
    re = evn.lazyimport('re')
    assert isinstance(re, _LazyModule)
    assert 2 == len(re.findall('foo', 'foofoo'))
    assert isinstance(re, _LazyModule)

def test_lazyimport_this():
    this = evn.lazyimport('this')
    assert not this._lazymodule_is_loaded()
    with evn.capture_stdio() as poem:
        assert this.c == 97
    assert 'The Zen of Python, by Tim Peters' == evn.first(poem.readlines()).strip()
    assert this._lazymodule_is_loaded()

def helper_test_re_ft_it(re, ft, it):
    assert 2 == len(re.findall('foo', 'foofoo'))
    assert ft.partial(lambda x, y: x + y, 1)(2) == 3
    assert list(it.chain([0], [1], [2])) == [0, 1, 2]

def test_multi_lazyimport_args():
    helper_test_re_ft_it(*evn.lazyimports('re', 'functools', 'itertools'))

if __name__ == '__main__':
    main()
