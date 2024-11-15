import concurrent.futures as cf
import functools

import numpy as np
from opt_einsum import contract as einsum

import ipd

def symframe_permutations(frames, **kw):
    func = functools.partial(_symperm1, frames=frames)
    with cf.ThreadPoolExecutor(max_workers=8) as exe:
        perm = exe.map(func, range(len(frames)))
    perm = np.stack(list(perm))
    return perm

def symframe_permutations_torch(frames, maxcols=None):
    import torch  # type: ignore
    frames = torch.tensor(frames, device="cuda").to(torch.float32)
    perm = list()
    for i, frame in enumerate(frames):
        # if i % 100 == 0:
        # ic(i, len(frames))
        local_frames = einsum("ij,fjk->fik", torch.linalg.inv(frame), frames)
        dist2 = torch.sum((local_frames[None] - frames[:, None])**2, axis=(2, 3))  # type: ignore
        idx = torch.argmin(dist2, axis=1)[:maxcols]  # type: ignore
        mindist = dist2[torch.arange(len(idx)), idx]
        missing = mindist > 1e-5
        idx[missing] = -1
        perm.append(idx)
    perm = torch.stack(perm).to(torch.int32)
    return perm.to("cpu").numpy().astype(np.int32)

def _symperm1(i, frames):
    # if i % 100 == 0:
    # ic(i, len(frames))
    local_frames = ipd.homog.hxform(ipd.homog.hinv(frames[i]), frames)
    dist = ipd.homog.hdiff(frames, local_frames, lever=3)
    idx = np.argmin(dist, axis=1)
    mindist = dist[np.arange(len(idx)), idx]
    missing = mindist > 1e-3
    idx[missing] = -1
    return idx

def permutations(sym, **kw):
    frames = ipd.sym.frames(sym, **kw)
    perm = symframe_permutations(frames, **kw)
    if ipd.sym.is_closed(sym):
        for idx in perm:
            assert len(set(idx)) == len(idx)
        assert np.all(perm >= 0)
    return perm
