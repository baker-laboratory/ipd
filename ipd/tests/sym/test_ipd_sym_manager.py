import pytest

pytest.importorskip('torch')
from ipd.lazy_import import lazyimport

th = lazyimport('torch')

import hypothesis
from icecream import ic

import ipd
from ipd import h

@hypothesis.settings(deadline=2000, max_examples=10)
@hypothesis.given(ipd.tests.sym.sym_manager(L=50, maxslice=8))
@pytest.mark.fast
def test_sym_manager_fuzz_xyz_sym(sym):
    sym.opt.asu_to_best_frame = False
    idx = sym.idx
    X = th.randn((idx.L, 3), device=sym.device)
    Xsym = sym(X)
    th.testing.assert_close(X[idx.unsym], Xsym[idx.unsym])
    sym.assert_symmetry_correct(Xsym)
    ipd.sym.check_sym_asu(sym, X, Xsym)
    sym.opt.asu_to_best_frame = True
    Xsym = sym(X)
    if sym.idx.Nasu > 2:
        ipd.sym.check_sym_asu(sym, X, Xsym)
    sym.assert_symmetry_correct(Xsym)
    if th.sum(sym.idx.asu) > 2:
        rms, _, xfit = h.rmsfit(Xsym[sym.idx.asym], X[sym.idx.asym])
        assert rms < 1e-3

@pytest.mark.fast
def test_sym_manager():
    sym = ipd.tests.sym.create_test_sym_manager([
        'sym.symid=C2',
        '+sym.Lasu=1',
        'sym.asu_to_best_frame=false',
    ])

    assert sym.idx.Nasu == 1
    ic(sym.nsub)
    assert sym.nsub == 2

    xyz = th.randn((2, 3))
    xyz2 = sym(xyz)
    assert ipd.sym.check_sym_asu(sym, xyz, xyz2)

@pytest.mark.fast
def test_sym_fit():
    sym = ipd.tests.sym.create_test_sym_manager([
        'sym.symid=C2',
        'sym.asu_to_best_frame=false',
    ])

    N = 7
    xyz = th.randn((N, 3))
    sym.idx = [ipd.sym.SymSlice((N, 0, 6), fit=True)]
    symxyz = sym(xyz, showme=0)
    assert ipd.sym.check_sym_asu(sym, xyz, symxyz)

@pytest.mark.fast
def test_sym_asu_align_icos_nounsym():
    sym = ipd.tests.sym.create_test_sym_manager([
        'sym.symid=I',
        'sym.asu_to_best_frame=true',
        'sym.max_nsub=6',
    ])

    N = 180
    xyz = th.randn((N, 3)) + 10
    n = sym.nsub
    sym.idx = [ipd.sym.SymSlice((N, 0, N))]
    symxyz = sym(xyz, showme=0)
    # ipd.showme(symxyz)
    assert ipd.sym.check_sym_asu(sym, xyz, symxyz, perm_ok=True)

@pytest.mark.fast
def test_sym_asu_align_icos_unsym():
    sym = ipd.tests.sym.create_test_sym_manager([
        'sym.symid=I',
        'sym.asu_to_best_frame=true',
        'sym.max_nsub=6',
    ])

    N = 200
    xyz = th.randn((N, 3)) + 10
    n = sym.nsub
    sym.idx = [ipd.sym.SymSlice((N, 0, 180))]
    # ic(sym.idx.asufit.to(int))
    # ic(sym.idx.asunotfit.to(int))
    symxyz = sym(xyz, showme=0)
    # ipd.showme(symxyz)
    assert ipd.sym.check_sym_asu(sym, xyz, symxyz, perm_ok=True)

@pytest.mark.fast
def test_sym_fit_icos_unsym():
    sym = ipd.tests.sym.create_test_sym_manager([
        'sym.symid=I',
        'sym.asu_to_best_frame=true',
        'sym.max_nsub=6',
        'sym.fit=True',
    ])

    N = 200
    xyz = th.randn((N, 3)) + 10
    n = sym.nsub
    sym.idx = [ipd.sym.SymSlice((N, 0, 180))]
    symxyz = sym(xyz, showme=0)
    # ipd.showme(symxyz)
    assert ipd.sym.check_sym_asu(sym, xyz, symxyz, perm_ok=True)

@pytest.mark.fast
def test_sym_fit_icos_unsym_multislice():
    sym = ipd.tests.sym.create_test_sym_manager([
        'sym.symid=I',
        'sym.asu_to_best_frame=true',
        'sym.max_nsub=3',
        'sym.fit=True',
        '+fittscale=1',
        '+fitwclash=1',
    ])

    N = 200
    xyz = th.randn((N, 3)) + 10
    n = sym.nsub
    sym.idx = [(N, 10, 40), (N, 50, 80), (N, 120, 150)]
    # ic(sym.idx.asufit.to(int))
    # ic(sym.idx.asunotfit.to(int))
    # ic(sym.idx.unsym.to(int))
    # ic(sym.idx.asu.to(int))
    symxyz = sym(xyz, showme=0)
    # ipd.showme(symxyz)
    assert ipd.sym.check_sym_asu(sym, xyz, symxyz, perm_ok=True)

if __name__ == '__main__':
    test_sym_fit_icos_unsym_multislice()
    test_sym_fit_icos_unsym()
    test_sym_asu_align_icos_unsym()
    test_sym_asu_align_icos_nounsym()
    test_sym_fit()
    test_sym_manager()

    print('DONE')
