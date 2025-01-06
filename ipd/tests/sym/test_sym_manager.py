import pytest

pytest.importorskip('torch')
from ipd.lazy_import import lazyimport

th = lazyimport('torch')

import assertpy
import hypothesis
from icecream import ic

import ipd

# ic.configureOutput(includeContext=False, contextAbsPath=True)

pytestmark = pytest.mark.fast

@hypothesis.settings(deadline=2000, max_examples=10)
@hypothesis.given(ipd.tests.sym.sym_manager(L=50, maxslice=8))
def test_sym_manager_fuzz_basic_sym(sym):
    idx = sym.idx
    X = th.arange(idx.L, device=sym.device)
    Xsym = sym(X)
    assert th.all(X[idx.unsym] == Xsym[idx.unsym])
    for s in idx:
        assert th.all(Xsym[s.asu] == X[s.asu])
    sym.check(Xsym)

    m = th.arange(idx.L**2, device=sym.device).reshape(idx.L, idx.L)
    msym = sym(m)
    assert th.all(msym[idx.asym, idx.asym] == m[idx.asym, idx.asym])
    sym.check(msym)

@hypothesis.settings(deadline=2000, max_examples=10)
@hypothesis.given(ipd.tests.sym.sym_manager(L=50, maxslice=8))
def test_sym_manager_fuzz_fill_from_contiguous(sym):
    idx = sym.idx

    X = th.arange(idx.L, device=sym.device)
    thing = sym.sym_adapt(X)
    adapted, contig, Lasu = sym.to_contiguous(thing)
    assert th.all(sym.fill_from_contiguous(thing, adapted, contig) == X)
    contig[:] = -1
    test = sym.fill_from_contiguous(thing, adapted, contig)
    assert th.all(test[idx.unsym] == X[idx.unsym])
    smask = idx.sub.max(dim=0).values
    assert th.all(test[smask] == -1)

    m = th.arange(idx.L**2, device=sym.device).reshape(idx.L, idx.L)
    thing = sym.sym_adapt(m)
    adapted, contig, Lasu = sym.to_contiguous(thing)
    assert th.all(sym.fill_from_contiguous(thing, adapted, contig) == m)
    contig[:] = -1
    test = sym.fill_from_contiguous(thing, adapted, contig)
    assert th.all(test[idx.unsym, idx.unsym] == m[idx.unsym, idx.unsym])
    smask = idx.sub.max(dim=0).values
    assert th.all(test[smask, smask] == -1)

@pytest.mark.fast
def test_sym_manager_string_2slice():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=C3', '+sym.Lasu=1'])
    sym.idx = [(8, 0, 3), (8, 4, 7)]
    assertpy.assert_that(sym('abcdefgh')).is_equal_to('aaadeeeh')
    assertpy.assert_that(sym('adeh')).is_equal_to('aaadeeeh')
    sym.check('aaadeeeh')

@pytest.mark.fast
def test_sym_manager_1d_2slice():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=C3', '+sym.Lasu=1'])
    sym.idx = [(8, 0, 3), (8, 4, 7)]
    t = th.tensor
    assert th.all(sym(t([1, 2, 3, 4, 5, 6, 7, 8])) == t([1, 1, 1, 4, 5, 5, 5, 8]))
    assert th.all(sym(t([1, 4, 5, 8])) == t([1, 1, 1, 4, 5, 5, 5, 8]))

@pytest.mark.fast
def test_sym_manager_2d_2slice():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=C2', '+sym.Lasu=1'])
    n = 10
    sym.idx = [(n, 0, 4), (n, 5, 9)]
    m = th.arange(n * n).reshape(n, n).to(int)
    assert th.all(
        sym(m) == th.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                             [20, 21, 0, 1, 24, 25, 26, 5, 6, 29], [30, 31, 10, 11, 34, 35, 36, 15, 16, 39],
                             [40, 41, 42, 43, 44, 45, 46, 47, 48, 49], [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                             [60, 61, 62, 63, 64, 65, 66, 67, 68, 69], [70, 71, 50, 51, 74, 75, 76, 55, 56, 79],
                             [80, 81, 60, 61, 84, 85, 86, 65, 66, 89], [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]))

    n = 13
    sym.idx = [(n, 0, 4), (n, 5, 11)]
    m = th.arange(n * n).reshape(n, n).to(int).to(sym.device)
    asym = sym.idx.asym[None] * sym.idx.asym[:, None]  # type: ignore
    # ic(asym.to(int))
    asym = m[asym].reshape(8, 8)
    msym = sym(asym)
    # ic(msym)
    assert th.all(msym == th.tensor(
        [[0, 1, 0, 0, 4, 5, 6, 7, 0, 0, 0, 11, 12], [13, 14, 0, 0, 17, 18, 19, 20, 0, 0, 0, 24, 25],
         [0, 0, 0, 1, 0, 0, 0, 0, 5, 6, 7, 0, 0], [0, 0, 13, 14, 0, 0, 0, 0, 18, 19, 20, 0, 0],
         [52, 53, 0, 0, 56, 57, 58, 59, 0, 0, 0, 63, 64], [65, 66, 0, 0, 69, 70, 71, 72, 0, 0, 0, 76, 77],
         [78, 79, 0, 0, 82, 83, 84, 85, 0, 0, 0, 89, 90], [91, 92, 0, 0, 95, 96, 97, 98, 0, 0, 0, 102, 103],
         [0, 0, 65, 66, 0, 0, 0, 0, 70, 71, 72, 0, 0], [0, 0, 78, 79, 0, 0, 0, 0, 83, 84, 85, 0, 0],
         [0, 0, 91, 92, 0, 0, 0, 0, 96, 97, 98, 0, 0], [143, 144, 0, 0, 147, 148, 149, 150, 0, 0, 0, 154, 155],
         [156, 157, 0, 0, 160, 161, 162, 163, 0, 0, 0, 167, 168]]).to(sym.device))

@pytest.mark.fast
def test_sym_manager_string():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=C3', '+sym.Lasu=1'])
    sym.idx = [(4, 0, 3)]
    assert sym('abbb') == 'aaab'
    assert sym('ab') == 'aaab'

@pytest.mark.fast
def test_sym_manager_list():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=C3', '+sym.Lasu=1'])
    sym.idx = [(4, 0, 3)]
    assert sym(['abbb', 'bbcc', 'foo', 'bar']) == ['abbb', 'abbb', 'abbb', 'bar']

@pytest.mark.fast
def test_sym_manager_dict():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=c3', '+sym.Lasu=1'])
    sym.idx = [(4, 0, 3)]
    assert sym(dict(a='abcd')) == dict(a='aaad')

@pytest.mark.fast
def test_sym_manager_contiguous():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=C3', '+sym.Lasu=1'])
    sym.idx = [(4, 0, 3)]
    m = th.arange(16, device=sym.device).reshape(4, 4)
    thing = sym.sym_adapt(m)
    adapted, contig, Lasu = sym.to_contiguous(thing)
    assert th.all(sym.fill_from_contiguous(thing, adapted, contig) == m)

@pytest.mark.fast
def test_sym_asu_seq():
    sym = ipd.tests.sym.create_test_sym_manager([
        'sym.symid=C2',
        '+sym.Lasu=10',
    ])

    sym.idx = [20]
    assert sym.idx.Nasu == 10  # type: ignore
    assert sym.nsub == 2

    seq = th.arange(20)
    seq = sym(seq)
    assert all(seq[10:] == seq[:10])
    ic(seq.shape)
    ic(sym.masym.shape)
    asym = sym.asym(seq)
    asu = sym.asym(seq)
    assert all(asym == seq[:10])
    assert all(sym(asym) == seq)

    seq = th.cat([seq, th.arange(10)])
    sym.idx = ipd.sym.SymIndex(sym.nsub, [(30, 0, 20)])
    asym = sym.asym(seq)
    assert len(asym) == 20
    assert all(asym[:10] == seq[:10])
    assert all(asym[10:] == seq[20:])
    assert all(sym(asym) == seq)
    asu = sym.asu(seq)
    assert len(asu) == 10
    assert all(asu == seq[:10])

@pytest.mark.fast
def test_sym_asu_xyz():
    sym = ipd.tests.sym.create_test_sym_manager([
        'sym.symid=c3',
        '+sym.Lasu=10',
        'sym.asu_to_best_frame=false',
        'sym.fit=false',
    ])
    sym.idx = [30]
    assert sym.opt.Lasu == 10
    assert sym.nsub == 3

    xyz = th.randn((30, 3))
    # showme(xyz[:10], name='start')
    s = sym(xyz)
    assert s.shape == (30, 3)

    # showme(s[:10], name='final')
    assert th.allclose(xyz[:10], s[:10], atol=0.01)

    s = sym(xyz[:10])
    assert s.shape == (30, 3)

    xyz = th.randn((40, 3))
    sym.idx = ipd.sym.SymIndex(sym.nsub, [(40, 0, 30)])
    s = sym(xyz)
    assert s.shape == (40, 3)
    assert th.allclose(xyz[:10], s[:10], atol=0.001)
    assert th.allclose(xyz[30:], s[30:], atol=0.001)

    xyz = th.randn((1, 40, 1, 3))
    s = sym(xyz)
    # import ipd as ipd
    # ic(s.shape)
    # ipd.showme(xyz[0,:20,0])
    # ipd.showme(s[0,:20,0])
    # ipd.showme(s[0,20:,0])
    assert s.shape == (1, 40, 1, 3)
    assert th.allclose(xyz[:, :10], s[:, :10], atol=0.001)
    s2 = th.einsum('ij,rj->ri', sym.symmRs[1].cpu(), xyz[0, :10, 0])
    assert th.allclose(s2, s[0, 10:20, 0], atol=0.001)
    s3 = th.einsum('ij,rj->ri', sym.symmRs[2].cpu(), xyz[0, :10, 0])
    assert th.allclose(s3, s[0, 20:30, 0], atol=0.001)
    assert th.allclose(xyz[:, 30:], s[:, 30:], atol=0.001)

    sym.idx = ipd.sym.SymIndex(sym.nsub, [(39, 0, 30)])
    s = sym(xyz[0, :19, 0])
    assert s.shape == (39, 3)

@pytest.mark.fast
def test_sym_slices():
    sym = ipd.tests.sym.create_test_sym_manager([
        'sym.symid=c3',
        '+sym.Lasu=10',
        'sym.asu_to_best_frame=false',
    ])
    N = 30
    xyz = th.randn((N, 3), device=sym.device)
    sym.idx = [N]
    # symxyz = sym(xyz)
    # assert symxyz.shape == (N, 3)

    with pytest.raises(AssertionError):
        sym.idx = [(N, 0, 5), (N, 10, 16), (N, 20, 26)]

    sym.idx = [(N, 0, 6), (N, 10, 16), (N, 20, 26)]
    symxyz = sym(xyz)
    # ipd.showme(symxyz[~sym.idx.unsym] * 30, name='FOOF')
    assert th.allclose(xyz[sym.idx.asym], symxyz[sym.idx.asym])  # type: ignore
    assert ipd.sym.check_sym_asu(sym, xyz, symxyz)

@pytest.mark.fast
def test_sym_pair():
    sym = ipd.tests.sym.create_test_sym_manager([
        'sym.symid=c3',
        'sym.sympair_method=mean',
        'sym.symmsub_k=2',
        'sym.sympair_protein_only=False',
        'sym.sympair_enabled=True',
    ])
    sym.idx = [(30, 0, 30)]
    pair = th.randn((1, 30, 30, 10))
    # import torchshow
    # torchshow.show(pair.max(dim=-1).values)
    sympair = sym(pair)

    sym.assert_symmetry_correct(sympair)

@pytest.mark.fast
def test_create_test_sym_manager():
    assert ipd.sym.create_sym_manager().symid == 'C1'
    assert ipd.sym.create_sym_manager(symid='c3').symid == 'C3'

@pytest.mark.fast
def test_atom_on_axis():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=C3'])
    sym.idx = [(4, 0, 3)]
    assert sym.is_on_symaxis(th.tensor([[0, 0, 1.0], [1, 1, 1]]))[0]
    assert not sym.is_on_symaxis(th.tensor([[0, 0, 1.0], [1, 1, 1]]))[1]
    assert sym.is_on_symaxis(th.tensor([[0, 0, 2.0]]))[0]
    assert sym.is_on_symaxis(th.tensor([[0, 0, 3.0]]))[0]
    assert sym.is_on_symaxis(th.tensor([[0, 0, 4.0]]))[0]
    assert sym.is_on_symaxis(th.tensor([[0, 0, 5.0]]))[0]
    assert not len(sym.is_on_symaxis(th.tensor([[1, 0, 6.0]])))
    assert not len(sym.is_on_symaxis(th.tensor([[1, 0, 7.0]])))
    assert not len(sym.is_on_symaxis(th.tensor([[1, 0, 8.0]])))

if __name__ == '__main__':
    test_sym_asu_xyz()
    test_atom_on_axis()
    test_create_test_sym_manager()
    test_sym_asu_seq()
    test_sym_slices()
    test_sym_manager_fuzz_basic_sym()
    test_sym_manager_fuzz_fill_from_contiguous()
    test_sym_manager_contiguous()
    test_sym_manager_2d_2slice()
    test_sym_manager_1d_2slice()
    test_sym_manager_string_2slice()
    test_sym_manager_string()
    test_sym_manager_list()
    test_sym_manager_dict()
    test_sym_pair()
    print('DONE')
