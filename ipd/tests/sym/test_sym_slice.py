import pytest

pytest.importorskip('torch')
import ipd
from ipd.lazy_import import lazyimport

th = lazyimport('torch')

import hypothesis

from ipd.tests.sym import symslices

@pytest.mark.fast
@hypothesis.settings(deadline=1000)
@hypothesis.given(symslices(100, 20, raw=True))
def test_sym_slices_fuzz(slices):
    s = ipd.sym.SymIndex(*slices)
    s.sanity_check()

@hypothesis.given(symslices(100, 20, bad=True))
@pytest.mark.fast
def test_sym_slices_fuzz_bad(slices):
    try:
        s = ipd.sym.SymIndex(*slices)
        raise NotImplementedError('should have failed')
    except (AssertionError):
        pass

@pytest.mark.fast
def test_sym_slice_errors():
    sym = ipd.tests.sym.create_test_sym_manager(symid='c3')
    with pytest.raises(AssertionError):
        ipd.sym.SymIndex(sym.nsub, [
            ipd.sym.SymSlice(th.Tensor([1, 1, 1]), False),
            ipd.sym.SymSlice(th.Tensor([1, 1, 1]), True),
            ipd.sym.SymSlice(th.Tensor([1, 1, 1]), False),
        ])
    for slice in [
        [1, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0],
    ]:
        ipd.sym.SymIndex(sym.nsub, [ipd.sym.SymSlice(th.Tensor(slice))])
    for slice in [
            th.Tensor([1, 1, 1, 0, 1]),
            th.Tensor([1, 0, 1, 1, 1]),
            th.Tensor([0, 1, 1, 1, 0, 1, 1, 1]),
            th.Tensor([1, 1, 1, 0, 1, 1, 1, 0]),
            th.Tensor([0, 1, 1, 1, 0, 1, 1, 1, 0]),
            th.Tensor([1]),
    ]:
        with pytest.raises(AssertionError):
            s = ipd.sym.SymIndex(sym.nsub, [ipd.sym.SymSlice(slice)])
            print(s)
    with pytest.raises(AssertionError):
        ipd.sym.SymIndex(sym.nsub, [
            ipd.sym.SymSlice([1, 1, 1, 0, 0]),
            ipd.sym.SymSlice([1, 1, 1, 0]),
        ])

    with pytest.raises(AssertionError):
        idx = ipd.sym.SymIndex(sym.nsub, [
            ipd.sym.SymSlice([1, 1, 1, 0, 0, 0, 0, 0, 0], fit=True),
            ipd.sym.SymSlice([1, 1, 1, 0, 0, 0, 0, 0, 0]),
            ipd.sym.SymSlice([1, 1, 1, 0, 0, 0, 0, 0, 0]),
        ])

@pytest.mark.fast
def test_beg_end():
    sym = ipd.tests.sym.create_test_sym_manager(symid='c1')
    s = ipd.sym.SymIndex(sym.nsub, [[1, 1, 0, 0], [0, 0, 1, 0]])
    assert s.slices[0].beg == 0
    assert s.slices[0].symend == 2
    assert s.slices[1].beg == 2
    assert s.slices[1].symend == 3

    s = ipd.sym.SymIndex(sym.nsub, [(10, 1, 7), (10, 7, 8)])
    assert all(s.slices[0].mask == th.tensor([0, 1, 1, 1, 1, 1, 1, 0, 0, 0]))
    assert all(s.slices[1].mask == th.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))

@pytest.mark.fast
def test_sym_slice_iter():
    sym = ipd.tests.sym.create_test_sym_manager(symid='c2')
    idx = ipd.sym.SymIndex(sym.nsub, [
        ipd.sym.SymSlice([1, 1, 0, 0, 0, 0]),
        ipd.sym.SymSlice([0, 0, 1, 1, 0, 0]),
        ipd.sym.SymSlice([0, 0, 0, 0, 1, 1]),
    ])
    assert len(idx) == 3
    for s in idx:
        assert isinstance(s.mask, th.Tensor)

@pytest.mark.fast
def test_fitmask():
    sym = ipd.tests.sym.create_test_sym_manager(symid='c2')
    idx = ipd.sym.SymIndex(sym.nsub, [
        ipd.sym.SymSlice([1, 1, 0, 0, 0, 0, 0, 0, 0], fit=True),
        ipd.sym.SymSlice([0, 0, 0, 1, 1, 0, 0, 0, 0]),
        ipd.sym.SymSlice([0, 0, 0, 0, 0, 1, 1, 0, 0]),
    ])
    assert th.all(idx.asufit == th.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert th.all(idx.asunotfit == th.tensor([0, 0, 1, 1, 0, 1, 0, 1, 1]))

@pytest.mark.fast
def test_asu_slice():
    sym = ipd.tests.sym.create_test_sym_manager(symid='c2')
    idx = ipd.sym.SymIndex(sym.nsub, [
        ipd.sym.SymSlice([1, 1, 0, 0, 0, 0, 0, 0, 0], fit=True),
        ipd.sym.SymSlice([0, 0, 0, 1, 1, 0, 0, 0, 0]),
        ipd.sym.SymSlice([0, 0, 0, 0, 0, 1, 1, 0, 0]),
    ])
    assert th.all(idx.asu == th.tensor([1, 0, 0, 1, 0, 1, 0, 0, 0]))
    assert th.all(idx.asym == th.tensor([1, 0, 1, 1, 0, 1, 0, 1, 1]))
    assert th.all(idx.unsym == th.tensor([0, 0, 1, 0, 0, 0, 0, 1, 1]))
    assert th.all(idx.asufit == th.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert th.all(idx.asunotfit == th.tensor([0, 0, 1, 1, 0, 1, 0, 1, 1]))
    assert th.all(idx.unsymasu == th.tensor([0, 1, 0, 0, 1, 1]))
    assert th.all(idx[0].asu == th.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert th.all(idx[1].asu == th.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0]))
    assert th.all(idx[2].asu == th.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0]))
    assert th.all(idx[0].toasu == th.tensor([1, 0, 0, 0, 0, 0]))
    assert th.all(idx[1].toasu == th.tensor([0, 0, 1, 0, 0, 0]))
    assert th.all(idx[2].toasu == th.tensor([0, 0, 0, 1, 0, 0]))

@pytest.mark.fast
def test_slice_sym_mask():
    sym = ipd.tests.sym.create_test_sym_manager(symid='c2')
    idx = ipd.sym.SymIndex(sym.nsub, [
        ipd.sym.SymSlice([1, 1, 0, 0, 0, 0, 0, 0, 0]),
        ipd.sym.SymSlice([0, 0, 0, 1, 1, 0, 0, 0, 0]),
        ipd.sym.SymSlice([0, 0, 0, 0, 0, 1, 1, 0, 0]),
    ])
    assert th.all(idx.sub[0] == th.tensor([1, 0, 0, 1, 0, 1, 0, 0, 0]))
    assert th.all(idx.sub[1] == th.tensor([0, 1, 0, 0, 1, 0, 1, 0, 0]))

@pytest.mark.fast
def test_slice_map_indices():
    sym = ipd.tests.sym.create_test_sym_manager(symid='c2')
    idx = ipd.sym.SymIndex(sym.nsub, [
        ipd.sym.SymSlice([1, 1, 0, 0, 0, 0, 0, 0, 0]),
        ipd.sym.SymSlice([0, 0, 0, 1, 1, 0, 0, 0, 0]),
        ipd.sym.SymSlice([0, 0, 0, 0, 0, 1, 1, 0, 0]),
    ])
    assert th.allclose(idx.idx_asym_to_sym[th.arange(6)], th.tensor([0, 2, 3, 5, 7, 8]))
    assert th.allclose(idx.idx_sym_to_asym[th.arange(9)], th.tensor([0, -1, 1, 2, -1, 3, -1, 4, 5]))

@pytest.mark.fast
def test_is_sym_subsequence():
    sym = ipd.tests.sym.create_test_sym_manager(symid='c2')
    idx = ipd.sym.SymIndex(
        sym.nsub,
        [
            #                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
            ipd.sym.SymSlice([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            ipd.sym.SymSlice([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
            ipd.sym.SymSlice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
        ])
    assert idx.is_sym_subsequence(th.arange(15))
    assert not idx.is_sym_subsequence([0, 3])
    assert idx.is_sym_subsequence([0, 2])

@pytest.mark.fast
def test_contiguous():
    sym = ipd.tests.sym.create_test_sym_manager(symid='c2')
    sidx = ipd.sym.SymIndex(
        sym.nsub,
        [
            #                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
            ipd.sym.SymSlice([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            ipd.sym.SymSlice([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
            ipd.sym.SymSlice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
        ])
    # ic(th.all(sidx.contiguous == th.tensor([0, 1, 5, 6, 7, 11, 2, 3, 8, 9, 10, 12])))
    t, t2 = th.arange(15), th.arange(15)
    contig = t[sidx.contiguous]
    t[sidx.contiguous] = contig
    assert th.all(t == t2)
    s = sidx.subnum[sidx.contiguous]
    assert all(i == j for i, j in zip(s, sorted(s)))

    idx = th.tensor([0, 2, 5, 8], dtype=int)
    idx2 = -th.ones(15, dtype=int)
    idx2[idx] = idx
    idx2 = idx2[sidx.contiguous]
    idx2 = idx2[idx2 >= 0]
    # ic(idx.subnum[idx2])
    # assert 0

@pytest.mark.fast
def test_chirals():
    sym = ipd.tests.sym.create_test_sym_manager(symid='c3')
    sym.idx = [(120, 0, 60), (120, 60, 120)]
    s = sym.idx
    chirals = th.tensor([[60.0000, 63.0000, 69.0000, 70.0000, -0.6155], [60.0000, 63.0000, 70.0000, 69.0000, 0.6155],
                         [60.0000, 69.0000, 70.0000, 63.0000, -0.6155], [61.0000, 64.0000, 71.0000, 72.0000, -0.6155],
                         [61.0000, 64.0000, 72.0000, 71.0000, 0.6155], [61.0000, 71.0000, 72.0000, 64.0000, -0.6155],
                         [80.0000, 83.0000, 89.0000, 90.0000, -0.6155], [80.0000, 83.0000, 90.0000, 89.0000, 0.6155],
                         [80.0000, 89.0000, 90.0000, 83.0000, -0.6155], [81.0000, 84.0000, 91.0000, 92.0000, -0.6155],
                         [81.0000, 84.0000, 92.0000, 91.0000, 0.6155], [81.0000, 91.0000, 92.0000, 84.0000, -0.6155],
                         [100.000, 103.000, 109.000, 110.000, -0.6155], [100.000, 103.000, 110.000, 109.000, 0.6155],
                         [100.000, 109.000, 110.000, 103.000, -0.6155], [101.000, 104.000, 111.000, 112.000, -0.6155],
                         [101.000, 104.000, 112.000, 111.000, 0.6155], [101.000, 111.000, 112.000, 104.000,
                                                                        -0.6155]]).to(sym.device)
    idx = chirals[:, 0].to(int)
    assert s.is_sym_subsequence(idx)  # type: ignore

@pytest.mark.fast
def test_nonprot():
    sym = ipd.tests.sym.create_test_sym_manager(symid='c3')
    sym.idx = [(120, 0, 60), (120, 60, 120)]
    assert th.all(sym.idx.nonprot.cpu() == th.repeat_interleave(th.arange(2), 60))  # type: ignore
    sym.idx = [33]
    assert th.all(sym.idx.nonprot.cpu() == th.repeat_interleave(th.arange(1), 33))  # type: ignore

@pytest.mark.fast
def test_slice2d():
    sym = ipd.tests.sym.create_test_sym_manager(symid='c3')
    sym.idx = [(12, 0, 6), (12, 6, 12)]
    d = th.arange(144).reshape(12, 12)
    assert th.all(
        sym.idx.slice2d(d, 'asu') == th.tensor([  # type: ignore
            [0, 1, 6, 7],
            [12, 13, 18, 19],
            [72, 73, 78, 79],
            [
                84,
                85,
                90,  # type: ignore
                91
            ]
        ]))  # type: ignore
    sym.idx.slice2d(d, 'asu', 1)  # type: ignore
    assert th.all(sym.idx.slice2d(d, 'asu') == 1)  # type: ignore
    sym.idx.slice2d(d, 'asu', th.arange(16).reshape(4, 4))  # type: ignore
    assert th.all(sym.idx.slice2d(d, 'asu') == th.arange(16).reshape(4, 4))  # type: ignore
    d = th.arange(144).reshape(12, 12)
    assert th.all(
        sym.idx.slice2d(d, th.tensor([0, 1, 6, 7])) == th.tensor([  # type: ignore
            [0, 1, 6, 7],
            [12, 13, 18, 19],
            [72, 73, 78, 79],  # type: ignore
            [84, 85, 90, 91]
        ]))

if __name__ == '__main__':
    test_slice2d()
    test_nonprot()
    test_chirals()
    test_contiguous()
    test_is_sym_subsequence()
    test_slice_map_indices()
    test_slice_sym_mask()
    test_asu_slice()
    test_fitmask()
    test_beg_end()
    test_sym_slice_errors()
    test_sym_slice_iter()
    test_sym_slices_fuzz()
    test_sym_slices_fuzz_bad()
