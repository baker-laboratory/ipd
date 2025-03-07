import pytest
import numpy as np
import ipd

config_test = ipd.Bunch(
    re_only=[
        # 'test_ewise_equal'
    ],
    re_exclude=[],
)

def main():
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
    )

@ipd.dev.element_wise_operations
class EwiseDict(dict):
    pass

def test_mapwise():
    b = EwiseDict(zip('abcdefg', ([] for i in range(7))))
    ic(b)
    assert all(b.valwise == [])
    ic(b.mapwise == [])
    r = b.mapwise.append(1)
    assert all(b.valwise == [1])

def test_ewise_accum():
    b = EwiseDict(zip('abcdefg', (i for i in range(7))))
    assert isinstance(b.mapwise + 10, ipd.Bunch)
    assert isinstance(b.valwise + 10, list)
    assert isinstance(b.npwise + 10, np.ndarray)

def test_mapwise_multi():
    b = EwiseDict(zip('abcdefg', ([] for i in range(7))))
    assert b.mapwise == []
    with pytest.raises(ValueError):
        r = b.mapwise.append(1, 2)
    b.mapwise.append(*range(7))
    ic(b)
    assert list(b.values()) == [[i] for i in range(7)]

def test_mapwise_equal():
    b = EwiseDict(zip('abcdefg', ([] for i in range(7))))
    assert b.mapwise == []
    b.mapwise.append(*range(7))
    eq4 = b.mapwise == [4]
    assert list(eq4.values()) == [0, 0, 0, 0, 1, 0, 0]
    assert not any((b.mapwise == 3).values())

def test_mapwise_add():
    b = EwiseDict(zip('abcdefg', range(7)))
    assert (b.valwise == 4) == [0, 0, 0, 0, 1, 0, 0]
    assert not any(b.valwise == 'ss')
    c = b.mapwise + 7
    b.mapwise += 7
    assert b == c
    d = b.npwise - 4
    e = 4 - b.npwise
    assert np.all(d == -e)

def test_mapwise_contains():
    b = EwiseDict(zip('abcdefg', [[i] for i in range(7)]))
    with pytest.raises(ValueError):
        contains = b.valwise.__contains__(4)
    contains = b.valwise.contains(4)
    assert contains == [0, 0, 0, 0, 1, 0, 0]

def test_mapwise_contained_by():
    b = EwiseDict(zip('abcdefg', range(7)))
    contained = b.valwise.contained_by([1, 2, 3])
    assert contained == [0, 1, 1, 1, 0, 0, 0]

def test_mapwise_indexing():
    dat = np.arange(7 * 4).reshape(7, 4)
    b = EwiseDict(zip('abcdefg', dat))
    indexed = b.npwise[1]
    assert np.all(indexed == dat[:, 1])

def test_mapwise_slicing():
    dat = np.arange(7 * 4).reshape(7, 4)
    b = EwiseDict(zip('abcdefg', dat))
    indexed = b.npwise[1:3]
    ic(indexed)
    assert np.all(indexed == dat[:, 1:3])

if __name__ == '__main__':
    main()
