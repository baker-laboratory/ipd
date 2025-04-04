import pytest

pytest.importorskip('torch')
from ipd import lazyimport

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch as th
else:
    th = lazyimport('torch')

import hypothesis

import ipd
import ipd.homog.thgeom as h

def main():
    test_sym_manager_fuzz_xyz_sym()
    test_sym_manager()
    test_sym_fit()
    test_sym_asu_align_icos_nounsym()
    test_sym_asu_align_icos_unsym()
    test_sym_fit_icos_unsym()
    test_sym_fit_icos_unsym_multislice()

@hypothesis.settings(deadline=2000, max_examples=10)
@hypothesis.given(ipd.tests.sym.sym_manager(L=50, maxslice=8))
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

def test_sym_manager():
    sym = ipd.tests.sym.create_test_sym_manager([
        'sym.symid=C2',
        '+sym.Lasu=1',
        'sym.asu_to_best_frame=false',
    ])

    assert sym.idx.Nasu == 1
    ipd.icv(sym.nsub)
    assert sym.nsub == 2

    xyz = th.randn((2, 3))
    xyz2 = sym(xyz)
    assert ipd.sym.check_sym_asu(sym, xyz, xyz2)

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
    # ipd.icv(sym.idx.asufit.to(int))
    # ipd.icv(sym.idx.asunotfit.to(int))
    symxyz = sym(xyz, showme=0)
    # ipd.showme(symxyz)
    assert ipd.sym.check_sym_asu(sym, xyz, symxyz, perm_ok=True)

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

def test_sym_fit_icos_unsym_multislice():
    sym = ipd.tests.sym.create_test_sym_manager([
        'sym.symid=I',
        'sym.asu_to_best_frame=true',
        'sym.max_nsub=3',
        'sym.fit=True',
        '+fit_tscale=1',
        '+fit_wclash=1',
    ])

    N = 200
    xyz = th.randn((N, 3)) + 10
    n = sym.nsub
    sym.idx = [(N, 10, 40), (N, 50, 80), (N, 120, 150)]
    # ipd.icv(sym.idx.asufit.to(int))
    # ipd.icv(sym.idx.asunotfit.to(int))
    # ipd.icv(sym.idx.unsym.to(int))
    # ipd.icv(sym.idx.asu.to(int))
    symxyz = sym(xyz, showme=0)
    # ipd.showme(symxyz)
    assert ipd.sym.check_sym_asu(sym, xyz, symxyz, perm_ok=True)

if __name__ == '__main__':
    main()
