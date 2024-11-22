import random

import pytest

from ipd.lazy_import import lazyimport

th = lazyimport('torch')

import ipd

pytest.importorskip('ipd.fit.qcp_rms_cuda')
import numpy as np
from icecream import ic

from ipd import h
from ipd.fit.qcp_rms import _rms

def main():
    test_qcp_scan()
    test_qcp_bbhetero()
    test_qcp_scan_partition()
    test_qcp_scan_cuda()
    test_qcp_scan_cuda_ncac()
    test_qcp_scan_cuda_cyclic()
    test_qcp_scan_cuda_ncac_cyclic()
    perf_test_qcpscan_cuda()
    # assert 0
    test_qcp_scan_AB()
    print('test_qcp_scan PASS', flush=True)
    ipd.global_timer.report()

@ipd.timed
@pytest.mark.fast
def test_qcp_scan_partition():
    for i in range(100):
        bb = th.randn((60, 3), dtype=th.float32, device='cuda')
        tgt = th.randn((4, 3), dtype=th.float32, device='cuda')
        lbub = th.tensor([[0, 4]] * 4, dtype=th.int32, device='cuda')
        idx, rms = ipd.fit.scan_rms_seqpos(bb, tgt, lbub, chainbreak=2, nthread=1)
        assert idx.sum() == 6

@ipd.timed
@pytest.mark.fast
def test_qcp_bbhetero():
    nscan = 10
    bb0 = th.randn((60, 1, 3), dtype=th.float32, device='cuda')
    tgt = th.randn((3, 1, 3), dtype=th.float32, device='cuda')
    lbub = th.tensor([[0, nscan]] * 3, dtype=th.int32, device='cuda')

    bb = th.stack([bb0[th.randn(60) > 0][:nscan] for i in range(3)])
    idx, rms = ipd.fit.scan_rms_seqpos(bb, tgt, lbub, rmsout=True)
    bb = bb.reshape(3 * nscan, 1, 3)
    lbub[1] += nscan
    lbub[2] += 2 * nscan
    idx2, rms2 = ipd.fit.scan_rms_seqpos(bb, tgt, lbub, rmsout=True)
    # print(nscan, th.sum(rms > 9e8))
    # rms2[rms == 9e9] = 9e9
    assert th.allclose(rms, rms2, atol=1e-3)
    # assert 0

@ipd.timed
@pytest.mark.fast
def helper_test_qcp_scan_cuda(N, Ncyc, natom, i=0, ntgt=0, bbhetero=False):
    try:
        seed = hash((N, i, Ncyc, natom, random.random()))
        th.manual_seed(seed)
        bb = th.randn((60 * Ncyc, natom, 3), dtype=th.float32, device='cuda')
        tgt = th.randn((3 * Ncyc, natom, 3), dtype=th.float32, device='cuda')
        lbub = th.tensor([[0, N], [20, 20 + N], [40, 40 + N]], dtype=th.int32, device='cuda')
        idx, rms = ipd.fit.scan_rms_seqpos(bb, tgt, lbub, rmsout=True, cyclic=Ncyc)
        idx2, rms2, _ = ipd.fit.qcp_scan_ref(bb, tgt, lbub, rmsout=True, cyclic=Ncyc)
        rms = rms.reshape(rms2.shape)
        if not th.allclose(rms, rms2, atol=1e-2):
            ic(th.sum(~th.isclose(rms, rms2)))
        # ic(rms, rms2)
        assert th.abs(rms - rms2).mean() < 1e-4
        assert th.allclose(rms, rms2, atol=1e-2)
        if not th.allclose(idx, idx2):
            print('-' * 22, 'test_qcp_scan_cuda index mismatch', '-' * 22, flush=True)
            # ic(lbub)
            # ic(rms.shape, rms2.shape)
            idxrms = rms[tuple(idx - lbub[:, 0])].item()
            idx2rms = rms[tuple(idx2 - lbub[:, 0])].item()
            ic((i, N, Ncyc, natom), idx, idx2, idxrms - idx2rms)
            print('-' * 80, flush=True)
            assert th.allclose(rms2[tuple(idx - lbub[:, 0])], rms2[tuple(idx2 - lbub[:, 0])], atol=1e-3)
    except AssertionError as e:
        print('fail on', N, i, Ncyc, natom, seed)  # type: ignore
        raise e

@ipd.timed
@pytest.mark.fast
def test_qcp_scan_cuda():
    for N in range(1, 16):
        helper_test_qcp_scan_cuda(N, 1, 1)

@ipd.timed
@pytest.mark.fast
def test_qcp_scan_cuda_ncac():
    for N in range(1, 16):
        helper_test_qcp_scan_cuda(N, 1, 3)

@ipd.timed
@pytest.mark.fast
def test_qcp_scan_cuda_cyclic():
    for N in range(1, 16):
        helper_test_qcp_scan_cuda(N, 3, 1)

@ipd.timed
@pytest.mark.fast
def test_qcp_scan_cuda_ncac_cyclic():
    for N in range(1, 16):
        helper_test_qcp_scan_cuda(N, 3, 3)

def helper_test_qcpscan_perf(nscan, nres, natom, cyclic, nsamp):
    bb = th.randn((nscan * cyclic, natom, 3), dtype=th.float32, device='cuda')
    tgt = th.randn((4 * cyclic, natom, 3), dtype=th.float32, device='cuda')
    lbub = th.tensor([[0, nscan], [0, nscan], [0, nscan], [0, nscan]], dtype=th.int32, device='cuda')
    idx0 = th.tensor([1, 6, 2, 8], device='cuda')
    xrand = h.rand().to(bb.device).to(th.float32)
    for i, ii in enumerate(idx0):
        bb[ii] = h.xform(xrand, tgt[i])
    if nscan**len(lbub) < 1e8:
        with ipd.dev.Timer(verbose=False) as t:
            for isamp in range(nsamp):
                idx, rms, xfit = ipd.fit.qcp_scan_ref(bb, tgt, lbub, cyclic)
        assert all(idx == idx0)  # type: ignore
        assert abs(rms) < 0.001  # type: ignore
        assert th.allclose(xrand[:3, :3], xfit[:3, :3], atol=1e-4)  # type: ignore
        # ic(xrand[:3,3], xfit[:3,3])
        # ic(h.xform(xfit, tgt))
        # ic(h.xform(xrand, tgt))
        # ic(bb[idx])
        assert th.allclose(xrand[:3, 3], xfit[:3, 3], atol=1e-4)  # type: ignore
        rate = th.prod(lbub[:, 1] - lbub[:, 0]) / t.elapsed() / 1_000_000 * nsamp
        print(f'qcp_scan_ref torch        {nscan:4}s {nres:2}r {natom:3}a {cyclic:2}c rate {rate:8.3f}M')
    # lbub = th.tensor([[0, 50], [0, 50], [0, 50], [0, 50]], dtype=th.int32, device='cuda')
    # ic(lbub[:,1], bb.shape)
    for threads in [4]:  #range(1, 20):
        with ipd.dev.Timer(verbose=False) as t:
            for isamp in range(nsamp):
                idx, rms = ipd.fit.scan_rms_seqpos(bb, tgt, lbub, nthread=threads * 32)
        rate = th.prod(lbub[:, 1] - lbub[:, 0]) / t.elapsed() / 1_000_000 * nsamp
        print(
            f'scan_rms_seqpos cuda {threads*32:3}t {nscan:4}s {nres:2}r {natom:3}a {cyclic:2}c rate {rate:8.3f}M elapsed {t.elapsed()/nsamp:7.3f}'
        )

@ipd.timed
def perf_test_qcpscan_cuda():
    for natom in [1, 3, 10]:
        helper_test_qcpscan_perf(50, 4, natom, 1, 3)

@ipd.timed
@pytest.mark.fast
def test_qcp_scan_AB():
    for i in range(3, 7):
        pts1 = th.randn((30, 3), dtype=th.float32, device='cuda')
        pts2 = th.randn((i, 3), dtype=th.float32, device='cuda')
        idx, rms, xfit = ipd.fit.qcp_scan_AB(pts1, pts2, 10)
        assert i // 2 <= sum(10 <= idx) <= (i - i//2)

@ipd.timed
@pytest.mark.fast
def test_qcp_scan():
    helper_test_qcp_scan(ranges=[[0, 1], [10, 11], [20, 21]])
    helper_test_qcp_scan(ranges=[[0, 3], [10, 12], [20, 22]])
    helper_test_qcp_scan(ranges=[[0, 7], [30, 38], [50, 59]])
    # assert 0

def helper_test_qcp_scan(ranges):
    pts1 = th.randn((100, 5, 3), dtype=th.float32, device='cuda')
    pts2 = th.randn((len(ranges), 5, 3), dtype=th.float32, device='cuda')
    lbub = th.tensor(ranges, dtype=th.int32, device='cuda')
    _, scan_rms, _ = ipd.fit.qcp_scan_ref(pts1, pts2, lbub, rmsout=True)
    # return
    # scan_rms = _rms.qcp_scan_ref(pts1, pts2, lbub, False).cpu()
    # ic(scan_rms)
    pts1 = pts1.cpu()
    pts2 = pts2.cpu()
    scan_rms = scan_rms.cpu()
    pts2 -= pts2.mean((0, 1))
    lbub = lbub.cpu()
    sizes = lbub[:, 1] - lbub[:, 0]
    cp = th.cat([th.ones(1), th.cumprod(sizes, 0)]).to(int)
    state_size = int(cp[-1])
    myrms = th.zeros_like(scan_rms)
    for i in range(state_size):
        # for i in range(5, 6):
        # ic(i)
        cen = th.zeros(3)
        iprod = th.zeros((3, 3))
        E0 = 0
        E0 += pts2.square().sum()
        pts3 = th.zeros_like(pts2)
        for j in range(len(sizes)):
            idx = (i // cp[j]) % sizes[j]
            for k in range(pts1.shape[1]):
                cen += pts1[lbub[j, 0] + idx, k]
                pts3[j, k] = pts1[lbub[j, 0] + idx, k]
        cen /= len(sizes) * pts1.shape[1]
        index = list()
        for j in range(len(sizes)):
            idx = (i // cp[j]) % sizes[j]
            index.append(idx)
            p = pts1[lbub[j, 0] + idx] - cen
            iprod += th.matmul(p.T, pts2[j])
            E0 += (pts1[lbub[j, 0] + idx] - cen).square().sum()
        cen2 = pts3.mean((0, 1))
        pts3cen = pts3 - cen2
        iprod2 = th.matmul(pts3cen.reshape(-1, 3).transpose(0, 1), pts2.reshape(-1, 3))
        E02 = pts3cen.square().sum() + pts2.square().sum()
        E0 /= 2
        E02 /= 2
        # ic(cen, cen2)
        # ic(iprod)
        # ic(E0)
        rms1 = _rms.qcp_rms_f4(pts2.reshape(-1, 3), pts3.reshape(-1, 3))
        rms2, _ = _rms.qcp_rmsd_raw_vec_f4(iprod, E0, len(sizes) * pts1.shape[1] * th.ones_like(E0))
        rms3, _ = _rms.qcp_rmsd_raw_vec_f4(iprod2, E02, len(sizes) * pts1.shape[1] * th.ones_like(E0))
        # ic(rms1, rms2, rms3)
        assert np.allclose(rms3, rms2, atol=1e-3)
        assert np.allclose(rms1, rms2, atol=1e-3)
        assert np.allclose(rms3, rms1, atol=1e-3)
        # ic(rms1, scan_rms[i])
        # assert np.allclose(rms1, scan_rms[i], atol=1e-3)
        myrms[tuple(index)] = rms1
    assert th.allclose(myrms, scan_rms, atol=1e-3)
    for i in range(10):
        idx = th.tensor([random.randint(0, s - 1) for s in sizes])
        idxlb = idx + lbub[:, 0]
        rms = ipd.fit.rmsd(pts1[idxlb].reshape(-1, 3), pts2.reshape(-1, 3))
        assert abs(rms - scan_rms[tuple(idx)]) < 0.001

if __name__ == '__main__':
    main()
