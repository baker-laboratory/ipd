import itertools

from ipd.lazy_import import lazyimport

th = lazyimport('torch')

import numpy as np
import torch.utils.cpp_extension  # type: ignore
from icecream import ic

import ipd
from ipd import h

_rms = ipd.dev.lazyimport('ipd.fit.qcp_rms_cuda')

def scan_rms_seqpos(bb, tgt, ranges, cyclic=1, rmsout=False, maxdist=9e9, lasu=0, nthread=256, chainbreak=0):
    """"""
    return _rms.qcp_scan_cuda(bb, tgt, ranges, cyclic, chainbreak, lasu, nthread, rmsout)

def qcp_scan_AB(bb, tgt, L, **kw):
    """Scan bb to find best alignment of tgt, splitting matched points across
    L."""
    nreg = len(tgt)
    assert nreg > 1
    best = [None, 9e9, None, None]
    for a in itertools.product(*[[0, 1] for i in range(nreg)]):
        if sum(a) not in (nreg // 2, nreg - nreg//2): continue
        ranges = th.tensor([[L, 2 * L] if b else [0, L] for b in a], dtype=int, device=bb.device)
        # print(ranges)
        result = qcp_scan_ref(bb, tgt, ranges, **kw)
        if result[1] < best[1]:
            best = result
        break
    return best

def qcp_scan_ref(bb, tgt, ranges=None, cyclic=1, rmsout=False, maxdist=9e9):
    """Compute rmsds of tgt to bb, scanning ranges in bb use the full cuda
    version instead, it's 4-20x faster."""
    if bb.ndim == 2: bb = bb.unsqueeze(1)
    if tgt.ndim == 2: tgt = tgt.unsqueeze(1)
    Lasu = len(bb) // cyclic
    assert bb.shape[1:] == tgt.shape[1:]
    if ranges is None:
        ranges = th.zeros((len(tgt), 2), dtype=int, device=bb.device)
        ranges[:, 1] = len(bb)
    if len(tgt) != len(ranges) * cyclic:
        raise ValueError(f"len(tgt)={len(tgt)} != len(ranges)*cyclic={len(ranges)*cyclic}")
    if len(tgt) < 3:
        raise ValueError(f"len(tgt)={len(tgt)} need at least 3 points to fit")
    if ranges.shape[1] == 2:
        ranges0, ranges = ranges, th.zeros((len(ranges), len(bb)), dtype=bool, device=bb.device)
        for i, (lb, ub) in enumerate(ranges0):
            ranges[i][lb:ub] = True
    else:
        ranges0 = list()
    if maxdist < 9e9:
        ranges = th.stack([r & (maxdist > h.norm(rp[0] - bb[:, 0])) for r, rp in zip(ranges, tgt)])
        ic(ranges.to(int))
        assert 0
    tgtcen = tgt - tgt.mean((0, 1))
    sizes = ranges.sum(1)

    iprod, E0 = _qcp_scan_impl(bb, tgtcen, ranges, sizes, Lasu, cyclic)

    rms, _ = _rms.qcp_rmsd_cuda_fixlen(iprod, E0, float(len(ranges) * tgtcen.shape[1] * cyclic))
    rms = rms.reshape(tuple(sizes))

    _mark_overlapping_refpts(ranges0, sizes, rms)

    idx0 = th.tensor(np.unravel_index(th.argmin(rms).cpu().numpy(), rms.shape))
    minrms = rms[tuple(idx0)]
    idx = th.tensor([th.where(ranges[i])[0][j] for i, j in enumerate(idx0)], dtype=int, device=bb.device)

    xfit = _qcp_scan_checks(bb, tgt, idx0, idx, rms, cyclic, Lasu)

    return idx, rms if rmsout else minrms.item(), xfit

def _qcp_scan_impl(bb, tgt, ranges, sizes, Lasu=0, cyclic=1):
    cen = th.zeros(tuple(sizes) + (3, ), device=bb.device, dtype=bb.dtype)
    iprod = th.zeros(tuple(sizes) + (3, 3), device=bb.device, dtype=bb.dtype)
    E0 = th.zeros(tuple(sizes), device=bb.device, dtype=bb.dtype)
    E0 += tgt.square().sum()
    for i, mask in enumerate(ranges):
        shape = [1] * len(ranges) + [3]
        shape[i] = sizes[i]
        w = torch.where(mask)[0]
        for j in range(cyclic):
            # ic(bb[w+j*Lasu])
            cen += bb[w + j*Lasu].mean(1).reshape(shape)
    cen /= len(ranges) * cyclic
    # ic(cen)
    for i, mask in enumerate(ranges):
        shape = [1] * len(ranges) + list(tgt.shape[1:])
        shape[i] = sizes[i]
        w = torch.where(mask)[0]
        for j in range(cyclic):
            mcen = bb[w + j*Lasu].reshape(shape) - cen.unsqueeze(-2)
            # ic(mcen.shape, tgt[i, ..., None, :].shape)
            # ic((mcen[..., None] * tgt[i, ..., None, :]).shape)
            iprod += (mcen[..., None] * tgt[i + j * len(ranges), ..., None, :]).sum(-3)
            E0 += mcen.square().sum((-1, -2))

    E0 /= 2.0
    iprod = iprod.reshape(-1, 3, 3)
    E0 = E0.reshape(-1)
    return iprod, E0

def _mark_overlapping_refpts(ranges, sizes, rms):
    for i, iolap in enumerate(ranges[:, 0]):
        for j, jolap in enumerate(ranges[:, 0]):
            if i >= j: continue
            shift = jolap - iolap
            ir = th.arange(sizes[i], device=rms.device, dtype=int)
            jr = ir + shift
            ir = ir[jr < sizes[j]]
            jr = jr[jr < sizes[j]]
            idx = [slice(None)] * len(ranges)
            idx[i] = ir
            idx[j] = jr
            # ic(ranges)
            # ic(idx)
            rms[tuple(idx)] = 9e9

            # assert 0
def _qcp_scan_checks(bb, tgt, idx0, idx, rmsfull, cyclic, Lasu):
    idxasu = idx.clone()
    for i in range(1, cyclic):
        idx = th.cat([idx, idxasu + i*Lasu])
    minrms = rmsfull.min()
    assert rmsfull[tuple(idx0)] == minrms
    selpts = bb[idx].contiguous().cpu()
    bb = bb.reshape(-1, 3)
    selpts = selpts.reshape(-1, 3)
    tgt = tgt.reshape(-1, 3)
    rms, R, T = ipd.fit.qcp_rms_align(tgt.cpu().to(float), selpts.to(float))
    R, T = R.to(th.float32), T.to(th.float32)
    if abs(minrms - rms) > 0.001:
        ic(rms, minrms, idx)
        assert abs(minrms - rms) < 0.001
    hrms, _, xfit = h.rmsfit(tgt.cpu(), selpts)
    assert abs(rms - hrms) < 0.001
    if not th.allclose(R, xfit[:3, :3], atol=1e-2):
        ic(R, xfit[:3, :3])
        ic(R - xfit[:3, :3])
        ic(T, xfit[:3, 3])
        assert th.allclose(R, xfit[:3, :3], atol=1e-2)
    assert th.allclose(T, xfit[:3, 3], atol=1e-2)
    hnewxyz = h.xform(xfit, tgt.cpu())
    halnrms = th.sqrt((hnewxyz - selpts).square().sum() / len(tgt))
    # ic(halnrms, hrms)
    assert abs(halnrms - hrms) < 0.001
    newxyz = th.einsum('ij,rj->ri', R, tgt.cpu()) + T
    alnrms = th.sqrt((newxyz - selpts).square().sum() / len(tgt))
    assert abs(alnrms - rms) < 0.001  # whs todo figure this out
    return xfit.to(tgt.device)
