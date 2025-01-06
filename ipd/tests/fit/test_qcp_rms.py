import itertools
import random
from timeit import timeit

import pytest

from ipd.lazy_import import lazyimport

th = lazyimport('torch')

import ipd

pytest.importorskip('ipd.fit.qcp_rms_cuda')
import numpy as np

from ipd import h
from ipd.fit.qcp_rms import _rms

def main():
    test_qcp_kernel_numba()
    test_rms()
    test_qcp_raw_cuda_xform()
    test_qcp_raw_cuda()
    test_qcp_vec()
    test_qcp_align(niter=10)
    test_qcp_align()
    # test_qcp_align_vec()
    # test_qcp_regions_junct_simple()
    # test_qcp_regions()
    # test_qcp_regions_junct()
    # test_qcp_regions_simple_1seg()
    # test_qcp_regions_simple_2seg()
    # test_qcp_regions_simple_Nseg()
    test_qcp(niter=10)
    test_qcp_align(niter=10)
    # perftest_qcp_regions()
    test_rms_perf()
    print('test_qcp PASS', flush=True)

@pytest.mark.fast
def test_qcp_kernel_numba():
    N1, N2, Natm = 19, 31, 17
    pts1 = th.randn((N1, Natm, 3), dtype=th.float32, device='cuda')
    pts2 = th.randn((N2, Natm, 3), dtype=th.float32, device='cuda')
    a, b, c1, c2, iprod, E0 = ipd.fit.calc_iprod_E0(pts1, pts2)
    rms2 = th.empty(N1 * N2, device='cuda')
    xfit1 = th.empty((N1 * N2, 4, 4), device='cuda')
    ipd.fit.numba_kernel_qcp_raw[len(iprod), 256](rms2, xfit1, iprod, E0, Natm, True)
    rms2 = rms2.reshape(N1, N2)
    xfit1 = xfit1.reshape(N1, N2, 4, 4)
    rms2, xfit2 = ipd.fit.rmsd(pts1.cpu(), pts2.cpu(), True)
    # ic(xfit1[...,:3,:3])
    # ic(xfit2[...,:3,:3])
    assert th.allclose(rms2.cpu(), rms2, atol=1e-3)
    assert th.allclose(xfit1[..., :3, :3].cpu(), xfit2[..., :3, :3], atol=1e-2)
    # assert 0

@pytest.mark.fast
def test_rms_perf():
    for dev in 'cuda cpu'.split():
        pts1 = th.randn((100, 50, 3), dtype=th.float32, device=dev)
        pts2 = th.randn((100, 50, 3), dtype=th.float32, device=dev)
        count = 100 if dev == 'cuda' else 1
        ipd.fit.rmsd(pts1, pts2)
        if dev == 'cuda':
            t = timeit(lambda: ipd.fit.rmsd(pts1, pts2, usenumba=True), number=count)
            print(f'numba noxform {t/count*1000:7.3f}ms')
        t = timeit(lambda: ipd.fit.rmsd(pts1, pts2), number=count)
        print(f'{dev:4}  noxform {t/count*1000:7.3f}ms')
        t = timeit(lambda: ipd.fit.rmsd(pts1, pts2, getfit=True), number=count)
        print(f'{dev:4} getfit {t/count*1000:7.3f}ms')

@pytest.mark.fast
def test_rms():
    for dev in 'cuda cpu'.split():
        pts1 = th.randn((13, 11, 3), dtype=th.float32, device=dev)
        pts2 = th.randn((7, 11, 3), dtype=th.float32, device=dev)
        rms1, xfit1 = ipd.fit.rmsd(pts1, pts2, getfit=True)
        rms2 = th.empty([13, 7])
        xfit2 = th.empty([13, 7, 4, 4])
        for i in range(len(pts1)):
            for j in range(len(pts2)):
                rms2[i, j], _, xfit2[i, j] = ipd.h.rmsfit(pts1[i], pts2[j])
        assert th.allclose(rms1.cpu(), rms2, atol=1e-3)
        assert th.allclose(xfit1.cpu(), xfit2, atol=3e-2)

def helper_test_qcp_raw_cuda(getfit):
    pts1 = th.randn((7, 10, 3), dtype=th.float32, device='cuda')
    pts2 = th.randn((7, 10, 3), dtype=th.float32, device='cuda')
    # ic(iprod.shape, E0.shape)
    # with ipd.dev.Timer():
    if True:
        pts1cen, pts2cen = pts1 - pts1.mean(1).unsqueeze(1), pts2 - pts2.mean(1).unsqueeze(1)
        iprod = th.matmul(pts1cen[:, None].transpose(-2, -1), pts2cen[None])
        E0 = (pts1cen[:, None].square().sum((-2, -1)) + pts2cen[None].square().sum((-2, -1))) / 2
        iprod = iprod.reshape(-1, 3, 3).contiguous()
        E0 = E0.reshape(-1).contiguous()
        if pts1.device.type == 'cpu':
            rms1, x1 = _rms.qcp_rmsd_raw_vec_f4(iprod, E0, th.ones_like(E0) * 10, getfit)
        else:
            rms1, x1 = _rms.qcp_rmsd_cuda(iprod, E0, th.ones_like(E0) * 10, getfit)
    if len(rms1) > 100_000: return
    if getfit:
        rms2, x2 = _rms.qcp_rms_vec_f4(pts1, pts2, getfit)
        x2 = x2.reshape(-1, 4, 4)
    else:
        rms2 = _rms.qcp_rms_vec_f4(pts1, pts2).reshape(-1)
    rms2 = rms2.reshape(-1)
    assert th.allclose(rms1.cpu(), rms2)
    if getfit:
        # ic(x1.shape, x2.shape)
        assert th.allclose(x1.cpu(), x2, atol=1e-3)  # type: ignore

@pytest.mark.fast
def test_qcp_raw_cuda():
    helper_test_qcp_raw_cuda(False)

@pytest.mark.fast
def test_qcp_raw_cuda_xform():
    helper_test_qcp_raw_cuda(True)

@pytest.mark.fast
def test_qcp_vec():
    pts1 = h.randpoint((13, 10), dtype=th.float32)[:, :, :3].contiguous()
    pts2 = h.randpoint((7, 10), dtype=th.float32)[:, :, :3].contiguous()
    rms1 = th.tensor([[_rms.qcp_rms_f4(p1, p2) for p2 in pts2] for p1 in pts1])
    rms2 = _rms.qcp_rms_vec_f4(pts1, pts2)
    # ic(rms1.shape, rms2.shape)
    assert th.allclose(rms1, rms2)

@pytest.mark.skip
@pytest.mark.fast
def test_qcp_align_vec(npts=(1000, 10)):
    pts1 = h.randpoint(npts)[:, :, :3].contiguous().to(th.float64)
    pts2 = h.randpoint(npts[1])[:, :3].contiguous().to(th.float64)
    # with ipd.dev.Timer():
    rms2, R2, T2 = zip(*[_rms.qcp_rms_align_f8(p1, pts2) for p1 in pts1])
    rms, R, T = np.stack(rms2), np.stack(R2), np.stack(T2)
    # with ipd.dev.Timer():
    rms2, R2, T2 = _rms.qcp_rms_align_vec_f8(pts1, pts2)
    assert np.allclose(rms, rms2)
    assert np.allclose(R, R2)
    assert np.allclose(T, T2)
    pts1 = pts1.to(th.float32)
    pts2 = pts2.to(th.float32)
    # with ipd.dev.Timer():
    rms3, R3, T3 = _rms.qcp_rms_align_vec_f4(pts1, pts2)

    assert np.allclose(rms, rms3, atol=1e-4)
    assert np.allclose(R, R3, atol=1e-4)
    assert np.allclose(T, T3, atol=1e-4)

@pytest.mark.skip
@pytest.mark.fast
def test_qcp_regions_simple_1seg():
    N = 100
    pts1 = h.randpoint(N + 10, th.float32)[:, :3].contiguous()
    pts2 = h.randpoint(N, th.float32)[:, :3].contiguous()
    # pts1[:100] = pts1[:100] * 0.01 + pts2 * 0.99
    offsets = np.arange(10).reshape(10, 1)
    # offsets = np.tile(offsets, (170_000, 1))
    # ic(offsets.shape)
    # with ipd.dev.Timer():
    rms = _rms.qcp_rms_regions_f4i4(pts1, pts2, [N], offsets)
    # ic(rms)
    for i in range(10):
        rmsref = _rms.qcp_rms_f8(pts1[i:N + i], pts2)
        assert rms.shape == (len(offsets), )
        assert np.allclose(rms[i], rmsref, atol=1e-4)

# @pytest.mark.skip
# @pytest.mark.fast
# def test_qcp_regions_simple_2seg():
#     N = 40
#     pts1 = h.randpoint(N, th.float32)[:, :3].contiguous()
#     pts2 = h.randpoint(N, th.float32)[:, :3].contiguous()
#     rmsref = _rms.qcp_rms_f8(pts1, pts2)
#     # pts1[:, :3] -= pts1[:, :3].mean(axis=0).reshape(1, 3)
#     # pts2[:, :3] -= pts2[:, :3].mean(axis=0).reshape(1, 3)
#     for i in range(5, 35):
#         [i, N - i]
#         assert rms.shape == len(offset)
#         assert np.allclose(rms, rmsref, atol=1e-4)

def _random_int_partition(n, r):
    p = list()
    while sum(p) < n:
        p.append(random.randint(0, r))
    p[-1] = n - sum(p[:-1])
    return p

def _random_offsets(n, l, sizes):
    return np.stack([np.random.randint(0, l - s + 1, n) for s in sizes], axis=-1)

@pytest.mark.skip
@pytest.mark.fast
def test_qcp_regions_simple_Nseg():
    N = 40
    pts1 = h.randpoint(N, dtype=th.float32)[:, :3].contiguous()
    pts2 = h.randpoint(N, dtype=th.float32)[:, :3].contiguous()
    rmsref = _rms.qcp_rms_f8(pts1, pts2)
    # pts1[:, :3] -= pts1[:, :3].mean(axis=0).reshape(1, 3)
    # pts2[:, :3] -= pts2[:, :3].mean(axis=0).reshape(1, 3)
    for i in range(100):
        sizes = _random_int_partition(N, 10)
        offsets = np.cumsum([0] + sizes[:-1]).reshape(1, len(sizes))
        rms = _rms.qcp_rms_regions_f4i4(pts1, pts2, sizes, offsets)
        assert rms.shape == (len(offsets), )
        assert np.allclose(rms, rmsref, atol=1e-4)

def compute_rms_offsets_brute(pts1, pts2, sizes, offsets, junct=0):
    rms = np.empty(len(offsets))

    offsets2 = np.cumsum([0] + list(sizes[:-1]))
    crd2 = list()
    for s, o in zip(sizes, offsets2):
        if junct == 0 or 2 * junct >= s:
            crd2.append(pts2[o:o + s, :3])
        else:
            crd2.append(pts2[o:o + junct, :3])
            crd2.append(pts2[o + s - junct:o + s, :3])
    crd2 = np.concatenate(crd2)
    for i, ofst in enumerate(offsets):
        crd1 = list()
        for s, o in zip(sizes, ofst):
            if junct == 0 or 2 * junct >= s:
                crd1.append(pts1[o:o + s, :3])
            else:
                crd1.append(pts1[o:o + junct, :3])
                crd1.append(pts1[o + s - junct:o + s, :3])
        crd1 = np.concatenate(crd1)

        rms[i] = _rms.qcp_rms_f8(crd1, crd2)
    return rms

def perftest_qcp_regions():
    t = ipd.dev.Timer()
    ncalc = 0
    for _ in range(30):
        pts1 = h.randpoint(200, dtype=th.float32)[:, :3].contiguous()
        pts2 = h.randpoint(50, dtype=th.float32)[:, :3].contiguous()
        sizes = _random_int_partition(len(pts2), len(pts2) - 5)
        offsets = _random_offsets(30_000, len(pts1), sizes)
        t.checkpoint('setup')
        rms = _rms.qcp_rms_regions_f4i4(pts1, pts2, sizes, offsets, junct=0)
        ncalc += len(rms)
        t.checkpoint('_rms.qcp_rms_regions_f4i4')
    t.report()
    rmspersec = ncalc / sum(t.checkpoints['_rms.qcp_rms_regions_f4i4'])
    print(f'rms ncalc: {ncalc:,}, rate: {rmspersec:7.3}', flush=True)

def helper_test_qcp_regions(noffset=1, junct=0, npts1=100, npts2=50):
    pts1 = h.randpoint(npts1, dtype=th.float32)[:, :3].contiguous()
    pts2 = h.randpoint(npts2, dtype=th.float32)[:, :3].contiguous()
    sizes = _random_int_partition(npts2, npts2 - 5)
    offsets = _random_offsets(noffset, len(pts1), sizes)
    rmsref = compute_rms_offsets_brute(pts1, pts2, sizes, offsets, junct=junct)
    rms = _rms.qcp_rms_regions_f4i4(pts1, pts2, sizes, offsets, junct=junct)
    assert np.allclose(rms, rmsref)

@pytest.mark.skip
@pytest.mark.fast
def test_qcp_regions():
    for _ in range(1000):
        helper_test_qcp_regions(noffset=10, junct=0)

@pytest.mark.skip
@pytest.mark.fast
def test_qcp_regions_junct_simple():
    pts1 = h.randpoint(9).astype(np.float32)[:, :3].contiguous()
    pts2 = h.randpoint(9).astype(np.float32)[:, :3].contiguous()

    # pts1 -= pts1[:4].mean(axis=0).reshape(-1, 3)
    # pts2 -= pts2[:4].mean(axis=0).reshape(-1, 3)

    sizes = [9]
    offsets = [[0]]
    rmsref = compute_rms_offsets_brute(pts1, pts2, sizes, offsets, junct=4)
    rms = _rms.qcp_rms_regions_f4i4(pts1, pts2, sizes, offsets, junct=4)
    # ic(rmsref)
    # ic(rms)
    assert np.allclose(rms, rmsref)

@pytest.mark.skip
@pytest.mark.fast
def test_qcp_regions_junct():
    for j, _ in itertools.product(range(2, 10), range(10)):
        helper_test_qcp_regions(noffset=10, junct=j)

@pytest.mark.fast
def test_qcp_align(niter=20, npts=50):
    for i in range(niter):
        pts1 = h.randpoint(npts, dtype=th.float64)[:, :3].contiguous()
        pts2 = h.randpoint(npts, dtype=th.float64)[:, :3].contiguous()
        assert pts1.dtype == th.float64
        rms, fit, x = ipd.hrmsfit(pts1, pts2)
        rms2, R, T = _rms.qcp_rms_align_f8(pts1, pts2)
        assert np.allclose(rms, rms2)
        assert np.allclose(x[:3, :3], R)
        assert np.allclose(x[:3, 3], T)
        rms2, R, T = _rms.qcp_rms_align_f4(pts1.to(th.float32), pts2.to(th.float32))
        assert np.allclose(rms, rms2, atol=1e-4)
        assert np.allclose(x[:3, :3], R, atol=1e-2)
        assert np.allclose(x[:3, 3], T, atol=1e-2)

@pytest.mark.fast
def test_qcp(niter=100, npts=50):
    for i in range(niter):
        pts1 = h.randpoint(npts, dtype=th.float64)[:, :3].contiguous()
        pts2 = h.randpoint(npts, dtype=th.float64)[:, :3].contiguous()
        rms, fit, x = ipd.hrmsfit(pts1, pts2)
        rms2 = _rms.qcp_rms_f8(pts1, pts2)
        assert np.allclose(rms, rms2)

if __name__ == '__main__':
    main()
