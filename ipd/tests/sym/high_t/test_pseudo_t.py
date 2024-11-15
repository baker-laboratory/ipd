import pytest
from icecream import ic

import ipd
from ipd import *

th = pytest.importorskip('torch')

@pytest.mark.fast
def test_t2():
    ipd.sym.high_t.pseudo_t_start(7)

@pytest.mark.fast
def test_t4():
    ipd.sym.high_t.pseudo_t_start(4)

@pytest.mark.skipif(not th.cuda.is_available(), reason="test needs CUDA")
def test_pseudo_t_dist_min():
    asym = ipd.sym.high_t.pseudo_t_start(2)
    loss, asym2 = ipd.sym.high_t.min_pseudo_t_dist2(asym)
    ic(asym2.shape)
    # ipd.showme(hscaled(0.1, asym))
    # ipd.showme(hscaled(0.1, asym2))
    # ipd.showme(hscaled(0.1, ipd.sym.make('I', asym)))

@pytest.mark.skipif(not th.cuda.is_available(), reason="test needs CUDA")
def test_pseudo_t_env_min():
    asym = ipd.sym.high_t.pseudo_t_start(2)
    loss, asym2 = ipd.sym.high_t.min_pseudo_t_symerror(asym)
    ic(asym2.shape)
    # ipd.showme(hscaled(0.1, asym))
    # ipd.showme(hscaled(0.1, asym2))
    # ipd.showme(hscaled(0.1, ipd.sym.make('I', asym)))

def debug_from_pdb():
    import numpy as np

    ficos = ipd.sym.frames("I")
    axes = ipd.sym.axes("I")

    # frames = np.load('ipd/data/pseudo_t/T2_4btg.npy')
    frames = np.load("ipd/data/pseudo_t/T2_3iz3.npy")
    # frames = np.load('ipd/data/pseudo_t/T2_7cbp_D.npy')  # 222 res
    # frames = np.load('ipd/data/pseudo_t/T2_7cbp_E.npy')  # 215 res
    # frames = np.load('ipd/data/pseudo_t/T3_7cbp_K.npy')  # 501 res
    # frames = np.load('ipd/data/pseudo_t/T3_7cbp_T.npy')  # 75 res
    # frames = np.load('ipd/data/pseudo_t/T3_2wbh_A129.npy')
    # frames = np.load('ipd/data/pseudo_t/T3_6rrs_A129.npy')
    # frames = np.load('ipd/data/pseudo_t/T3_2tbv.npy')
    # frames = np.load('ipd/data/pseudo_t/T4_1ohf_A510.npy')
    # frames = np.load('ipd/data/pseudo_t/T4_6rrt_A128.npy')
    # frames = np.load('ipd/data/pseudo_t/T4_1qgt_A142.npy')
    # frames = np.load('ipd/data/pseudo_t/T7_6o3h.npy')
    # frames = np.load('ipd/data/pseudo_t/T7_1ohg_A200.npy')
    # frames = np.load('ipd/data/pseudo_t/T9_8h89_J155.npy')
    # frames = np.load('ipd/data/pseudo_t/T13_2btv.npy')

    if 0:
        frames[:, :3, 3] -= frames[:, :3, 3].mean(0)
        dumppdb("test.pdb", hcart(frames) / 10)  # type: ignore
        a2 = (-13.205300, -7.494900, 20.899100)
        # a3 = (-4.202000,3.476000,15.283333)
        a5 = (-25.814400, 0.964600, 15.965600)
        # print(hangle(a2, a5))
        # print(hangle(axes[2], axes[5]))
        xaln = ipd.homog.halign2(a2, a5, axes[2], axes[5], strict=True)
        frames = hxform(xaln, frames)  # type: ignore
        # return
        # ipd.dev.save_package_data(frames, 'pseudo_t/T13_2btv.npy')

    # frames = ipd.sym.high_t.pseudo_t_start(2)
    asym = frames[0]
    ipd.showme(frames, weight=10, xyzlen=[10, 7, 4])
    # ipd.showme(ipd.homog.hxform(frames0, asym), weight=10, xyzlen=[10, 7, 4])
    ipd.showme(ipd.homog.hxform(ipd.sym.frames("I"), frames[0]), weight=10, xyzlen=[10, 7, 4])
    ipd.showme(ipd.homog.hxform(ipd.sym.frames("I"), frames[-1]), weight=10, xyzlen=[10, 7, 4])

def main():
    test_pseudo_t_env_min()
    # test_pseudo_t_start()
    # debug_from_pdb()
    # test_t2()

if __name__ == "__main__":
    main()
