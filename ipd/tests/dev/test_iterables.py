import ipd

def main():
    ipd.tests.maintest(namespace=globals())

def count():
    i = 0
    while True:
        yield i
        i += 1

def test_nth():
    assert ipd.dev.nth(count(), 4) == 4

def test_first():
    assert ipd.dev.first(count()) == 0
    assert ipd.dev.first([]) is None

def test_head():
    assert ipd.dev.head(count(), 4) == [0, 1, 2, 3]
    assert ipd.dev.head([], 4) == []
    assert ipd.dev.head([1, 4], 4) == [1, 4]

if __name__ == '__main__':
    main()
