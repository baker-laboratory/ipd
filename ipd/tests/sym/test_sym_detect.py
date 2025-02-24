import pytest
import numpy as np

import ipd
from ipd.homog import hgeom as h

def main():
    test_syminfo_from_frames_symid()
    test_symaxis_closest_to()
    test_symelems_from_frames()
    print('test_sym_detect.py PASS')

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
    frames0 = ipd.sym.frames('oct')
    testaxes0 = [[1, 0, 0], [0, -1, 0], [1, 1, 0], [1, 1, 0], [-1, 0, -1], [1, 1, 1], [-1, -1, 1], [1, 1, 1]]
    golden0 = [[0.0000, 0.0000, 1.0000, 0.0000], [0.0000, 0.0000, 1.0000, 0.0000], [0.7071, 0.0000, 0.7071, 0.0000],
               [0.7071, 0.0000, 0.7071, 0.0000], [0.7071, 0.0000, 0.7071, 0.0000], [0.5774, 0.5774, 0.5774, 0.0000],
               [0.5774, 0.5774, 0.5774, 0.0000], [0.5774, 0.5774, 0.5774, 0.0000]]
    closeaxes, _which = ipd.sym.symaxis_closest_to(frames0, testaxes0)
    assert h.allclose(closeaxes, golden0, atol=1e-4)

    randrot = h.rand(cart_sd=0, dtype=np.float64)
    frames = h.xform(randrot, frames0)
    testaxes = h.xform(randrot, testaxes0)
    golden = h.xform(randrot, golden0)
    closeaxes, _which = ipd.sym.symaxis_closest_to(frames, testaxes)
    assert h.allclose(closeaxes, golden, atol=1e-4)

    randtrans = h.randtrans(cart_sd=1, dtype=np.float64)
    frames = h.xform(randtrans, frames0)
    testaxes = h.xformvec(randtrans, testaxes0)
    golden = h.xform(randtrans, golden0)
    closeaxes, _which = ipd.sym.symaxis_closest_to(frames, testaxes)
    assert h.allclose(closeaxes, golden, atol=1e-4)

@pytest.mark.fast
def test_symelems_from_frames():
    frames0 = ipd.sym.frames('oct')
    ref = ipd.sym.symelems_from_frames(frames0)
    # ic(set(ref.nfold.data))
    assert set(ref.nfold.data) == {2, 3, 4}
    pert = h.rand(cart_sd=0, dtype=np.float64)
    # print(repr(pert))
    pert = np.array([[-0.8299, 0.4733, 0.2954, 0.0000], [-0.5578, -0.6963, -0.4516, 0.0000],
                     [-0.0080, -0.5396, 0.8419, 0.0000], [0.0000, 0.0000, 0.0000, 1.0000]],
                    dtype=np.float64)
    frames = h.xform(pert, frames0)
    symelem = ipd.sym.symelems_from_frames(frames)
    for nf, se in symelem.groupby('nfold'):
        assert len(se.nfold) == 1
        refaxis = ref.axis[ref.nfold == nf].data
        assert h.allclose(se.axis, h.xform(pert, refaxis)) or h.allclose(-se.axis, h.xform(pert, refaxis))
        assert h.allclose(se.cen, h.xform(pert, ref.cen[ref.nfold == nf].data))

if __name__ == '__main__':
    main()
