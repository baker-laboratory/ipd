import ipd

def main():
    ipd.tests.maintest(namespace=globals())

def count():
    i = 0
    while True:
        yield i
        i += 1

def test_iterables_nth():
    assert ipd.dev.nth(count(), 4) == 4

def test_iterables_first():
    assert ipd.dev.first(count()) == 0
    assert ipd.dev.first([]) is None

def test_iterables_head():
    assert ipd.dev.head(count(), 4) == [0, 1, 2, 3]
    assert ipd.dev.head([], 4) == []
    assert ipd.dev.head([1, 4], 4) == [1, 4]

def test_iterables_zipmap():
    zipped = ipd.dev.zipmaps(dict(a=1, b=2), dict(a='a', b='b'))
    assert isinstance(zipped, dict)
    assert zipped == dict(a=(1, 'a'), b=(2, 'b'))

def test_iterables_zipitems():
    zipped = ipd.dev.zipitems(dict(a=1, b=2), dict(a='a', b='b'))
    assert list(zipped) == [('a', 1, 'a'), ('b', 2, 'b')]

def test_iterables_zipmap_missing():
    zipped = ipd.dev.zipmaps(dict(a=1, b=2), dict(a='a', b='b', c='c'))
    assert isinstance(zipped, dict)
    assert zipped == dict(c=(ipd.dev.NA, 'c'), a=(1, 'a'), b=(2, 'b'))

def test_iterables_zipmap_missing_intersection():
    zipped = ipd.dev.zipmaps(dict(a=1, b=2), dict(a='a', b='b', c='c'), intersection=True)
    assert isinstance(zipped, dict)
    assert zipped == dict(a=(1, 'a'), b=(2, 'b'))

def test_iterables_zipmap_order():
    zipped = ipd.dev.zipmaps(dict(a=2, b=1), dict(a='a', b='b', c='c'), order='val')
    assert isinstance(zipped, dict)
    assert tuple(zipped.keys()) == ('c', 'b', 'a')

if __name__ == '__main__':
    main()
