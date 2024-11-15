from timeit import timeit

import numpy as np
import pytest
from icecream import ic

import ipd
from ipd import h

pytest.importorskip('torch')
pytest.importorskip('ipd.samp.samp_cuda')

def main():
    # fitsd()
    test_randxform_cen()
    test_welzl_sphere()
    test_randxform_big()
    test_randxform_small_angle()
    test_randxform_large_angle()
    test_randxform_angle()

    test_randxform_perf()
    test_quat_angle()

@pytest.mark.fast
def test_randxform_cen():
    cen = h.randpoint(2, device='cuda')
    x = ipd.samp.randxform(len(cen), cartmax=0, cen=cen)
    assert th.allclose(cen, h.xform(x, cen), atol=1e-3)  # type: ignore
    # assert 0

@pytest.mark.fast
def test_welzl_sphere():
    for npts in [1, 2, 3, 4, 5, 40, 100, 1000, 10_000]:
        xyz = th.randn((npts, 3))  # type: ignore
        cen, rad = ipd.samp.bounding_sphere(xyz)
        assert h.norm(xyz - cen).max() < rad + 0.001

@pytest.mark.fast
def test_randxform_big():
    ipd.samp.randxform(2**23)
    # x = ipd.samp.randxform(2**23 + 1) # fails?!? still!?!?

@pytest.mark.fast
def test_randxform_angle():
    # maxang = 0.1
    qunif = h.normQ(th.randn((10_000_000, 4), device='cuda'))  # type: ignore
    ang1 = 2 * th.arccos(qunif[:, 0])  #th.atan2(qunif[:, 1:].norm(dim=-1), qunif[:, 0])  # type: ignore
    ang1 = th.minimum(2 * th.pi - ang1, ang1)  # type: ignore
    # ang2 = h.angle(h.Qs2Rs(qunif))
    # assert th.allclose(ang1, ang2, atol=1e-3)
    # ang = ang[ang < maxang]
    quant1 = th.quantile(ang1, th.arange(0, 1.001, 0.12, device='cuda'))  # type: ignore
    xform = ipd.samp.randxform(1_000_000)
    assert th.allclose(xform[..., 3, :3], th.zeros(1, device='cuda'))  # type: ignore
    assert th.allclose(xform[..., 3, 3], th.ones(1, device='cuda'))  # type: ignore
    ang2 = ipd.h.angle(xform)
    quant2 = th.quantile(ang2, th.arange(0, 1.001, 0.12, device='cuda'))  # type: ignore
    print((quant1 * 1000).to(int).tolist())
    print((quant2 * 1000).to(int).tolist())
    assert th.allclose(quant2[1:], quant1[1:], atol=1e-2)  # type: ignore

"""
not unif
  3.500   0.000   0.062
  1.000   0.878   0.003
  0.900   0.900   0.003
  0.800   0.921   0.004
  0.700   0.939   0.006
  0.600   0.955   0.008
  0.500   0.969   0.013
  0.400   0.980   0.025
  0.300   0.989   0.060
  0.200   0.995   0.180
  0.100   0.999   1.694
  0.050   1.000  11.560
  """

@pytest.mark.fast
def test_randxform_large_angle():
    maxang = 1

    qunif = h.normQ(th.randn((10_000_000, 4), device='cuda'))  # type: ignore
    ang1 = 2 * th.arccos(qunif[:, 0])  #th.atan2(qunif[:, 1:].norm(dim=-1), qunif[:, 0])  # type: ignore
    ang1 = th.minimum(2 * th.pi - ang1, ang1)  # type: ignore
    ang1 = ang1[ang1 < maxang]
    quant1 = th.quantile(ang1, th.arange(0, 1.001, 0.12, device='cuda'))  # type: ignore
    quat_height = np.cos(np.clip(maxang, 0, th.pi) / 2)  # type: ignore
    quat = th.rand((1000000, 4), device='cuda')  # type: ignore
    quat[:, 0] = 0
    quat = quat[th.linalg.norm(quat, dim=1) <= 1]  # type: ignore
    scale = th.tan(th.acos(th.tensor(quat_height)))  # type: ignore
    quat *= scale
    quat[:, 0] = 1
    quat = h.normQ(quat)
    ang2 = 2 * th.arccos(quat[:, 0])  #th.atan2(qunif[:, 1:].norm(dim=-1), qunif[:, 0])  # type: ignore
    ang2 = th.minimum(2 * th.pi - ang2, ang2)  # type: ignore
    # ic(maxang, ang2.max())
    assert abs(ang2.max() - maxang) < 0.01
    quant2 = th.quantile(ang2, th.arange(0, 1.001, 0.12, device='cuda'))  # type: ignore
    print('randxlarge 1rad unif', (quant1 * 1000).to(int).tolist())
    print('randxlarge 1rad qhat', (quant2 * 1000).to(int).tolist())
    # assert 0

    for maxang in [3.14, 3.14, 3, 2.5, 2, 1.6, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
        with ipd.dev.Timer(verbose=False) as t:
            for i in range(10):
                xform = ipd.samp.randxform(5_000_000, orimax=maxang)
        assert th.allclose(xform[..., 3, :3], th.zeros(1, device='cuda'))  # type: ignore
        assert th.allclose(xform[..., 3, 3], th.ones(1, device='cuda'))  # type: ignore
        ang2 = ipd.h.angle(xform)
        # ic(maxang, ang2.max())
        assert abs(maxang - ang2.max()) < 0.01
        quat_height = np.cos(np.clip(maxang, 0, th.pi) / 2)  # type: ignore
        print(f'{maxang:7.3f} {quat_height:7.3f} {t.elapsed()*1000:7.3f}ms')

    for maxang in th.arange(0.1, 3.14, 0.1):  # type: ignore
        xform = ipd.samp.randxform(100_000, orimax=maxang)
        assert th.allclose(xform[..., 3, :3], th.zeros(1, device='cuda'))  # type: ignore
        assert th.allclose(xform[..., 3, 3], th.ones(1, device='cuda'))  # type: ignore
        ang2 = ipd.h.angle(xform)
        assert abs(maxang - ang2.max()) < 0.03
        # print(f'{maxang.item():7.3f} {ang2.max().item():7.3f} {ang2.std().item():7.3f}')

    return

    N = 1000_000
    from collections import defaultdict
    avg = defaultdict(list)
    std = defaultdict(list)
    for i, maxang in enumerate(th.arange(0.5, 3.15, 0.01)):
        xform = ipd.samp.randxform(N, orimax=maxang)
        assert th.allclose(xform[..., 3, :3], th.zeros(1, device='cuda'))
        assert th.allclose(xform[..., 3, 3], th.ones(1, device='cuda'))
        ang2 = ipd.h.angle(xform)
        avg[maxang.item()].append(float(ang2.mean()))
        std[maxang.item()].append(float(ang2.std()))
    from statistics import mean
    th.tensor(list(avg.keys()))
    avg = th.tensor([mean(l) for l in avg.values()])
    std = th.tensor([mean(l) for l in std.values()])
    print(maxang)
    print(avg)
    print(std)

@pytest.mark.fast
def test_randxform_small_angle():
    N = 100_000
    # from collections import defaultdict
    # qheight = dict()
    # avg = defaultdict(list)
    # std = defaultdict(list)
    for i, sd in enumerate(th.arange(0, 0.4, 0.025)):  # type: ignore
        # for i in range(10):
        # xform = ipd.samp.randxform_small_cuda(N, cartsd=0.1, orisd=sd)
        xform = ipd.samp.randxform(N, cartsd=0.1, orisd=sd)
        assert th.allclose(xform[..., 3, :3], th.zeros(1, device='cuda'))  # type: ignore
        assert th.allclose(xform[..., 3, 3], th.ones(1, device='cuda'))  # type: ignore
        ang2 = ipd.h.angle(xform)
        # avg[sd.item()].append(float(ang2.mean()))
        # std[sd.item()].append(float(ang2.std()))
        # continue
        th.quantile(ang2, th.arange(0, 1.001, 0.12, device='cuda'))  # type: ignore
        # xform = h.randsmall(N, rot_sd=sd, device='cuda', dtype=th.float32)
        # ang3 = ipd.h.angle(xform)
        # ic(ang3[:10])
        # quant3 = th.quantile(ang3, th.arange(0, 1.001, 0.12, device='cuda'))
        # print((quant2 * 1000).to(int).tolist())
        # print((quant3 * 1000).to(int).tolist())
        # print(quant2)
        # print(quant3)
        assert abs(sd - ang2.std()) < 0.05
        # print(f'{i:3} {sd:7.3f} {ang2.std():7.3f} {ang3.std():7.3f}')

    # from statistics import mean
    # qheight = th.tensor(list(avg.keys()))
    # avg = th.tensor([mean(l) for l in avg.values()])
    # std = th.tensor([mean(l) for l in std.values()])
    # print(qheight)
    # print(avg)
    # print(std)

    # assert th.allclose(quant2[1:], quant3[1:], atol=1e-2)
    # assert 0

@pytest.mark.fast
def test_quat_angle():
    maxang = 0.1
    q = th.randn((10, 4), device='cuda')  # type: ignore
    q[:, 0] = maxang
    # ic(th.sqrt(1 - q[:, 0]**2))
    q[:, 1:] *= th.sqrt(1 - q[:, 0]**2)[:, None] / th.linalg.norm(q[:, 1:], keepdim=True, dim=-1)  # type: ignore
    assert th.allclose(th.linalg.norm(q, keepdim=True, dim=-1), th.tensor(1.0))  # type: ignore
    ang = th.atan2(q[:, 1:].norm(dim=-1), q[:, 0])  # type: ignore
    print(th.quantile(ang, th.arange(0, 1.001, 0.2, device='cuda')))  # type: ignore

@pytest.mark.fast
def test_randxform_perf():

    N = 1_000_000
    for gentype in [
            'curandState',
            # 'curandStateScrambledSobol32_t',
            # 'curandStateSobol32_t',
            # 'curandStateMtgp32_t',
            # 'curandStateMRG32k3a_t',
            'curandStatePhilox4_32_10_t',
            'curandStateXORWOW_t',
            'ipd',
    ]:
        if gentype == 'ipd':

            def func():
                return ipd.h.rand(N, dtype=th.float32, device='cuda')  # type: ignore
        else:
            assert th.allclose(  # type: ignore
                ipd.samp.randxform(1, gentype=gentype, seed=0),  # type: ignore
                ipd.samp.randxform(1, gentype=gentype, seed=0))
            assert not th.allclose(  # type: ignore
                ipd.samp.randxform(1, gentype=gentype),
                ipd.samp.randxform(  # type: ignore
                    1, gentype=gentype))  # type: ignore

            def func():
                return ipd.samp.randxform(N, gentype=gentype, dtype=th.float32, device='cuda')  # type: ignore

        t = timeit(func, number=20)
        print(f'{gentype:>30} {t:7.3f}')

def fitsd():
    qh = th.tensor([  # type: ignore
        0.0000, 0.0100, 0.0200, 0.0300, 0.0400, 0.0500, 0.0600, 0.0700, 0.0800, 0.0900, 0.1000, 0.1100, 0.1200, 0.1300,
        0.1400, 0.1500, 0.1600, 0.1700, 0.1800, 0.1900, 0.2000, 0.2100, 0.2200, 0.2300, 0.2400, 0.2500, 0.2600, 0.2700,
        0.2800, 0.2900, 0.3000, 0.3100, 0.3200, 0.3300, 0.3400, 0.3500, 0.3600, 0.3700, 0.3800, 0.3900, 0.4000, 0.4100,
        0.4200, 0.4300, 0.4400, 0.4500, 0.4600, 0.4700, 0.4800, 0.4900, 0.5000, 0.5100, 0.5200, 0.5300, 0.5400, 0.5500,
        0.5600, 0.5700, 0.5800, 0.5900, 0.6000, 0.6100, 0.6200, 0.6300, 0.6400, 0.6500, 0.6600, 0.6700, 0.6800, 0.6900,
        0.7000, 0.7100, 0.7200, 0.7300, 0.7400, 0.7500, 0.7600, 0.7700, 0.7800, 0.7900, 0.8000, 0.8100, 0.8200, 0.8300,
        0.8400, 0.8500, 0.8600
    ])
    th.tensor([  # type: ignore
        0.0000, 0.0319, 0.0638, 0.0957, 0.1274, 0.1591, 0.1906, 0.2220, 0.2531, 0.2842, 0.3150, 0.3457, 0.3760, 0.4062,
        0.4359, 0.4654, 0.4948, 0.5235, 0.5522, 0.5803, 0.6083, 0.6359, 0.6629, 0.6898, 0.7165, 0.7427, 0.7684, 0.7940,
        0.8191, 0.8438, 0.8682, 0.8919, 0.9159, 0.9391, 0.9619, 0.9844, 1.0068, 1.0286, 1.0502, 1.0716, 1.0925, 1.1129,
        1.1329, 1.1530, 1.1727, 1.1921, 1.2111, 1.2299, 1.2482, 1.2664, 1.2838, 1.3018, 1.3191, 1.3362, 1.3527, 1.3695,
        1.3853, 1.4012, 1.4170, 1.4328, 1.4478, 1.4628, 1.4781, 1.4923, 1.5072, 1.5211, 1.5347, 1.5487, 1.5622, 1.5752,
        1.5884, 1.6011, 1.6141, 1.6264, 1.6388, 1.6509, 1.6632, 1.6751, 1.6867, 1.6981, 1.7095, 1.7206, 1.7318, 1.7426,
        1.7534, 1.7641, 1.7744
    ])
    std = th.tensor([  # type: ignore
        0.0000, 0.0135, 0.0269, 0.0403, 0.0536, 0.0668, 0.0798, 0.0927, 0.1054, 0.1179, 0.1302, 0.1423, 0.1541, 0.1656,
        0.1768, 0.1878, 0.1985, 0.2089, 0.2189, 0.2287, 0.2381, 0.2473, 0.2561, 0.2647, 0.2729, 0.2809, 0.2887, 0.2961,
        0.3031, 0.3100, 0.3166, 0.3229, 0.3289, 0.3348, 0.3406, 0.3457, 0.3508, 0.3557, 0.3603, 0.3647, 0.3691, 0.3729,
        0.3769, 0.3808, 0.3841, 0.3874, 0.3908, 0.3936, 0.3964, 0.3993, 0.4019, 0.4043, 0.4065, 0.4086, 0.4105, 0.4125,
        0.4145, 0.4161, 0.4178, 0.4193, 0.4207, 0.4219, 0.4229, 0.4244, 0.4253, 0.4262, 0.4270, 0.4277, 0.4287, 0.4294,
        0.4300, 0.4304, 0.4309, 0.4314, 0.4316, 0.4319, 0.4321, 0.4321, 0.4323, 0.4326, 0.4326, 0.4324, 0.4324, 0.4324,
        0.4321, 0.4321, 0.4321
    ])
    std2qh = np.polynomial.polynomial.Polynomial.fit(std, qh, 14)
    ic(std2qh)
    # ipd.viz.scatter(std, std2qh(std))

    assert 0

if __name__ == '__main__':
    main()
