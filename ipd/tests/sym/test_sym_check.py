import pytest

pytest.importorskip('torch')
import ipd
from ipd import lazyimport

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch as th
else:
    th = lazyimport('torch')

from ipd.sym.sym_adapt import DepRecatEd_symAdaptTensor

def test_symcheck_mapping():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=C2'])
    sym.idx = [2]
    sym.assert_symmetry_correct(dict(a=[1, 1]))

def test_symcheck_sequence():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=C2'])
    sym.idx = [2]
    sym.assert_symmetry_correct([[1, 1], [7, 7], [13, 13]])

def test_symcheck():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=C2'])
    sym.idx = [2]
    sym.assert_symmetry_correct(dict(a=[1, 1]))
    sym.assert_symmetry_correct([[1, 1], [7, 7], [13, 13]])
    sym.idx = [(20, 0, 4), (20, 10, 14)]
    SAT = DepRecatEd_symAdaptTensor
    # sym.assert_symmetry_correct(SAT(th.tensor([0, 1, 2, 3, 5, 10, 12]), sym, idx=[0, 1, 2, 3, 5, 10, 12], isidx=True,))
    # x = sym(SAT(th.tensor([0, 1, 5, 10]), sym, idx=[0, 1, 5, 10], isidx=slice(None)))
    # sym.assert_symmetry_correct(x)
    x = SAT(th.tensor([0, 1, 2, 3, 5, 10, 12]), sym, idx=[0, 1, 2, 3, 5, 10, 12], isidx=True).adapted
    # ipd.icv(x)
    sym.assert_symmetry_correct(x)
    with pytest.raises((AssertionError, ValueError)):
        sym.assert_symmetry_correct(
            SAT(th.tensor([0, 1, 2, 3, 5, 10, 12]), sym, idx=[0, 1, 2, 3, 5, 10, 13], isidx=True).adapted)
    with pytest.raises(AssertionError):
        sym.assert_symmetry_correct(
            SAT(th.tensor([0, 1, 2, 3, 5, 10]), sym, idx=[0, 1, 2, 3, 5, 10, 12], isidx=True).adapted)
    with pytest.raises(AssertionError):
        sym.assert_symmetry_correct(
            SAT(th.tensor([0, 1, 2, 3, 5, 10, 13]), sym, idx=[0, 1, 2, 3, 5, 10, 12], isidx=True).adapted)

if __name__ == '__main__':
    test_symcheck()
    print('DONE')
