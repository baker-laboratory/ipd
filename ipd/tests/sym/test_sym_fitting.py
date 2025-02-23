import pytest

pytest.importorskip('torch')
import ipd
from ipd.lazy_import import lazyimport
from ipd import h

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch as th
else:
    th = lazyimport('torch')

pytestmark = pytest.mark.fast

def main():
    # test_align_asu_c2()
    # test_align_asu_c7()
    # test_align_asu_tet()
    # test_align_asu_oct()
    # test_align_asu_icos()
    test_syminfo_from_frames_symid()
    test_symaxis_closest_to()
    test_symelems_from_frames()
    print('test_sym_fitting.py PASS')

@pytest.mark.fast
def test_syminfo_from_frames_symid():
    allsyms = ['T', 'O', 'I'] + ['C%i' % i for i in range(2, 13)] + ['D%i' % i for i in range(2, 13)]
    for symid in allsyms:
        frames = ipd.sym.frames(symid)
        sinfo = ipd.sym.syminfo_from_frames(frames)
        se = sinfo.symelem
        assert sinfo.symid == symid, f'{symid=}, {sinfo.symid=}'
        assert h.allclose(se.hel, 0, atol=1e-4)
        assert h.allclose(se.cen[:, :3], 0, atol=1e-4)
        ref = {k: v for k, v in ipd.sym.axes(symid).items() if isinstance(k, int)}
        for nf in ([] if symid in 'D2 '.split() else ref):
            assert h.allclose(ref[nf], se.axis[se.nfold == nf][0])

@pytest.mark.fast
def test_symaxis_closest_to():
    frames0 = ipd.sym.frames('oct', torch=True)
    testaxes0 = [[1, 0, 0], [0, -1, 0], [1, 1, 0], [1, 1, 0], [-1, 0, -1], [1, 1, 1], [-1, -1, 1], [1, 1, 1]]
    golden0 = [[0.0000, 0.0000, 1.0000, 0.0000], [0.0000, 0.0000, 1.0000, 0.0000], [0.7071, 0.0000, 0.7071, 0.0000],
               [0.7071, 0.0000, 0.7071, 0.0000], [0.7071, 0.0000, 0.7071, 0.0000], [0.5774, 0.5774, 0.5774, 0.0000],
               [0.5774, 0.5774, 0.5774, 0.0000], [0.5774, 0.5774, 0.5774, 0.0000]]
    closeaxes, _which = ipd.sym.symaxis_closest_to(frames0, testaxes0)
    assert h.allclose(closeaxes, golden0, atol=1e-4)

    randrot = h.rand(cart_sd=0).to(th.float64)
    frames = h.xform(randrot, frames0)
    testaxes = h.xform(randrot, testaxes0)
    golden = h.xform(randrot, golden0)
    closeaxes, _which = ipd.sym.symaxis_closest_to(frames, testaxes)
    assert h.allclose(closeaxes, golden, atol=1e-4)

    randtrans = h.randtrans(cart_sd=1).to(th.float64)
    frames = h.xform(randtrans, frames0)
    testaxes = h.xformvec(randtrans, testaxes0)
    golden = h.xform(randtrans, golden0)
    closeaxes, _which = ipd.sym.symaxis_closest_to(frames, testaxes)
    assert h.allclose(closeaxes, golden, atol=1e-4)

@pytest.mark.fast
def test_symelems_from_frames():
    frames0 = ipd.sym.frames('oct', torch=True)
    ref = ipd.sym.symelems_from_frames(frames0)
    # ic(set(ref.nfold.data))
    assert set(ref.nfold.data) == {2, 3, 4}
    pert = h.rand(cart_sd=0).to(th.float64)
    # print(repr(pert))
    pert = th.tensor([[-0.8299, 0.4733, 0.2954, 0.0000], [-0.5578, -0.6963, -0.4516, 0.0000],
                      [-0.0080, -0.5396, 0.8419, 0.0000], [0.0000, 0.0000, 0.0000, 1.0000]],
                     dtype=th.float64)
    frames = h.xform(pert, frames0)
    symelem = ipd.sym.symelems_from_frames(frames)
    for nf, se in symelem.groupby('nfold'):
        assert len(se.nfold) == 1
        refaxis = ref.axis[ref.nfold == nf].data
        assert h.allclose(se.axis, h.xform(pert, refaxis)) or h.allclose(-se.axis, h.xform(pert, refaxis))
        assert h.allclose(se.cen, h.xform(pert, ref.cen[ref.nfold == nf].data))

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
