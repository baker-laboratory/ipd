import pytest

pytest.importorskip('torch')
import ipd
from ipd.lazy_import import lazyimport

th = lazyimport('torch')

pytestmark = pytest.mark.fast

def main():
    # test_align_asu_c2()
    # test_align_asu_c7()
    test_align_asu_tet()
    test_align_asu_oct()
    test_align_asu_icos()

def make_test_points(npts, bound, cen=[0, 0, 0], ngen=None):
    ngen = ngen or npts * 10
    xyz = bound * th.randn((ngen, 3)).to(th.float32)
    xyz = xyz[th.topk(-ipd.h.norm(xyz), npts).indices]
    xyz = xyz[-bound < xyz[:, 0]]
    xyz = xyz[-bound < xyz[:, 1]]
    xyz = xyz[-bound < xyz[:, 2]]
    xyz = xyz[bound > xyz[:, 0]]
    xyz = xyz[bound > xyz[:, 1]]
    xyz = xyz[bound > xyz[:, 2]]
    return xyz[:npts] + th.tensor([cen])

def helper_test_align_asu(sym, Lasu=13):
    sym.idx = [Lasu * sym.nsub]
    xyz = make_test_points(Lasu, 10, cen=[1, 2, 3]).unsqueeze(1)
    xyz = sym(xyz)
    xyzs = [
        ipd.sym.asu_to_best_frame_if_necessary(sym,
                                               th.einsum('ij,raj->rai', R, xyz),
                                               Lasu,
                                               asu_to_best_frame_min_dist_to_origin=0,
                                               asu_to_best_frame=True) for R in sym.symmRs
    ]
    # ipd.showme(xyz.squeeze(1))
    # ipd.showme(xyzs[0][:Lasu].squeeze(1))
    for xyz in xyzs[1:]:
        # ipd.showme(xyz[:Lasu,0])
        assert th.allclose(xyzs[0][:Lasu], xyz[:Lasu], atol=1e-3)

@pytest.mark.fast
def test_align_asu_c2():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=C2'], max_nsub=5)
    helper_test_align_asu(sym)

@pytest.mark.fast
def test_align_asu_c7():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=C7'], max_nsub=5)
    helper_test_align_asu(sym)

@pytest.mark.fast
def test_align_asu_tet():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=T'], max_nsub=12)
    helper_test_align_asu(sym)

@pytest.mark.fast
def test_align_asu_oct():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=O'], max_nsub=5)
    helper_test_align_asu(sym)

@pytest.mark.fast
def test_align_asu_icos():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=I'], max_nsub=5)
    helper_test_align_asu(sym)

if __name__ == '__main__':
    main()
