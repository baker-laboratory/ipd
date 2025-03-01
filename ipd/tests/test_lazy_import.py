import sys

import ipd
from ipd.lazy_import import _LazyModule

def main():
    ipd.tests.maintest(namespace=globals())

def test_importornone():
    re = ipd.importornone('re')
    assert re is sys.modules['re']
    missing = ipd.importornone('noufuomemioixecmeiorutnaufoinairesvoraisevmraoui')
    assert missing is None
    missing = ipd.importornone('noufuomem ioixecmeiorutnaufoina iresvoraisevmraoui')
    assert missing == [None, None, None]

def test_lazyimport_re():
    re = ipd.lazyimport('re')
    assert isinstance(re, _LazyModule)
    assert 2 == len(re.findall('foo', 'foofoo'))
    assert isinstance(re, _LazyModule)

def test_lazyimport_this():
    this = ipd.lazyimport('this')
    assert not this.is_loaded()
    with ipd.dev.capture_stdio() as poem:
        assert this.c == 97
    assert 'The Zen of Python, by Tim Peters' == ipd.first(poem.readlines()).strip()
    assert this.is_loaded()

def helper_test_re_ft_it(re, ft, it):
    assert 2 == len(re.findall('foo', 'foofoo'))
    assert ft.partial(lambda x, y: x + y, 1)(2) == 3
    assert list(it.chain([0], [1], [2])) == [0, 1, 2]

def test_multi_lazyimport_list():
    helper_test_re_ft_it(*ipd.lazyimport(['re', 'functools', 'itertools']))

def test_multi_lazyimport_args():
    helper_test_re_ft_it(*ipd.lazyimport('re', 'functools', 'itertools'))

def test_multi_lazyimport_strsplit():
    helper_test_re_ft_it(*ipd.lazyimport('re functools itertools'))

if __name__ == '__main__':
    main()
