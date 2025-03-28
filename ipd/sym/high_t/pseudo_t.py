import numpy as np

import ipd
from ipd.homog import *
from ipd.homog import thgeom as h

th = ipd.lazyimport('torch')

def create_pseudo_t(opt):
    frames = th.tensor(ipd.sym.frames('I'))
    losses, asyms = list(), list()
    for i in range(100):
        asym = make_asym(opt.high_t_number, 8 * np.sqrt(opt.high_t_number), frames)
        sym = h.xform(frames, asym).reshape(-1, 4, 4)
        loss, asym2 = min_pseudo_t_dist(asym, sym="I")
        losses.append(loss)
        asyms.append(asym2)
    losses = th.tensor(losses)
    # print(losses, min(losses))  # type: ignore
    asym = asyms[th.argmin(losses)]
    sym2 = h.xform(frames, asym).reshape(-1, 4, 4)  # type: ignore
    Ts, Rs, xforms = [], [], []
    for X in sym2:
        Ts.append(X[:3, 3])
        Rs.append(X[:3, :3])
    Ts = th.stack(Ts)
    Ts -= Ts.mean(dim=0)
    Ts /= Ts.norm(p=2, dim=1, keepdim=True)
    Ts *= opt.radius
    Ts = [T - Ts[0] for T in Ts]
    for i, R in enumerate(Rs):
        X = np.eye(4)
        X[:3, :3] = R[:3, :3]
        X[:3, 3] = Ts[i]
        xforms.append(th.tensor(X))
    return th.stack(xforms)

def min_pseudo_t_dist(asym, sym="I"):
    t = len(asym)
    frames = ipd.sym.frames(sym)
    frames0 = th.tensor(frames[:, :3, :3]).cuda().to(th.float32)
    asym0 = th.tensor(asym[:, :3, 3]).cuda().to(th.float32)
    cen = h.point(np.stack(list(ipd.sym.axes("I").values())).mean(0))[:3].cuda().to(th.float32)
    cen = cen * th.norm(asym0[0])

    def score():
        lbfgs.zero_grad()
        asym = th.einsum("aij,aj->ai", h.Q2R(Q), asym0)
        sym = th.einsum("sij,aj->sai", frames0, asym)
        d = th.norm(asym[None, None] - sym[:, :, None], dim=-1)
        d = th.where(d < 0.001, 9e9, d)
        # loss = -d.min()  # + dcen / 4
        # d = th.topk(-d, 5, dim=0).values
        loss = 0
        # for i in range(len(asym)):
        # for j in range(i, len(asym)):
        # denv = th.norm(d[:, i, :] - d[:, j, :])
        # loss += denv
        loss -= th.min(d)
        loss.backward()
        return loss

    Q = th.zeros((t, 3)).cuda().requires_grad_(True)
    lbfgs = th.optim.LBFGS([Q], history_size=10, max_iter=4, line_search_fn="strong_wolfe")

    for iter in range(4):
        loss = lbfgs.step(score)

    asym[:, :3, 3] = th.einsum("aij,aj->ai", h.Q2R(Q), asym0).cpu().detach()
    asym = to_canonical_frame(asym, frames)
    return loss, asym  # type: ignore

def make_asym(t, r, frames):
    asym = hrand(t)
    asym[:, :3, 3] *= r / hnorm(hcart3(asym))[:, None]
    return to_canonical_frame(asym, frames)

def extract_t_asu(frames, t, sym="I"):
    cen = hpoint(np.stack(list(ipd.sym.axes("I").values())).mean(0))
    # cen = hscaled(100, hpoint([0.17524365, 0.11894071, 0.92827442]))
    cen = hpoint([2, 3, 100])
    cens = hcart(frames)
    d = hnorm(cens - cen)
    candidates = np.argsort(d)
    accepted = np.array([candidates[0]])
    while len(accepted) < t:
        asusym = ipd.sym.makepts(sym, cens[accepted])
        asymcom = cens[accepted].mean(0)
        d = hnorm(cens - asymcom)
        candidates = np.argsort(d)
        for c in candidates:
            d2 = hnorm(cens[c] - asusym)
            if np.sum(d2 < 1) == 0:
                break
        accepted = np.concatenate([accepted, [c]])  # type: ignore
        if len(accepted) == t:
            break

    assert len(accepted) == t
    return frames[accepted]

def pseudo_t_start(t):
    dat = {
        2: "pseudo_t/T2_3iz3.npy",
        3: "pseudo_t/T3_2tbv.npy",
        4: "pseudo_t/T4_1ohf_A510.npy",
        7: "pseudo_t/T7_1ohg_A200.npy",
        9: "pseudo_t/T9_8h89_J155.npy",
        13: "pseudo_t/T13_2btv.npy",
    }
    frames = ipd.dev.load_package_data(dat[t])
    assert len(frames) == 60 * t
    ipd.icv(frames.shape)  # type: ignore
    asu = extract_t_asu(frames, t)
    return asu

def to_canonical_frame(asym, frames):
    # cen = hpoint(np.stack(list(ipd.sym.axes('I').values())).mean(0))
    cen = hpoint([1, 2, 100])
    d = np.square(hxformpts(frames, hcart(asym)) - cen).sum(2)
    closest = np.argmin(d, axis=0)
    asym = hxform(frames[closest], asym)
    # point z outward
    asym[:, :3, :3] = hori(halign(asym[:, :, 2], asym[:, :3, 3], doto=asym))
    return th.as_tensor(asym)

def min_dist_loss(asym0, frames0, lbfgs, indvar, **kw):
    lbfgs.zero_grad()
    asym = th.einsum("aij,aj->ai", h.Q2R(indvar[0]), asym0)
    sym = th.einsum("sij,aj->sai", frames0, asym)
    d = th.norm(asym[None, None] - sym[:, :, None], dim=-1)
    d = th.where(d < 0.001, 9e9, d)
    loss = -th.min(d)
    loss.backward()
    return loss

def min_pseudo_t_dist2(asym, sym="I"):
    t = len(asym)
    frames = ipd.sym.frames(sym)
    frames0 = th.tensor(frames[:, :3, :3]).cuda().to(th.float32)
    asym0 = th.tensor(asym[:, :3, 3]).cuda().to(th.float32)
    indvar = [th.zeros((t, 3)).cuda().requires_grad_(True)]
    loss = h.torch_min(min_dist_loss, **vars())
    asym[:, :3, 3] = th.einsum("aij,aj->ai", h.Q2R(indvar[0]), asym0).cpu().detach()
    asym = to_canonical_frame(asym, frames)
    return loss, asym.numpy()

def min_sym_environment_loss(asym0, frames0, lbfgs, indvar, **kw):
    lbfgs.zero_grad()
    asym = th.einsum("aij,aj->ai", h.Q2R(indvar[0]), asym0)
    sym = th.einsum("sij,aj->sai", frames0, asym)
    d = th.norm(asym[None, None] - sym[:, :, None], dim=-1)
    d = th.where(d < 0.001, 9e9, d)
    loss = -th.min(d)
    loss.backward()
    return loss

def min_pseudo_t_symerror(asym, sym="I"):
    t = len(asym)
    frames = ipd.sym.frames(sym)
    frames0 = th.tensor(frames[:, :3, :3]).cuda().to(th.float32)
    asym0 = th.tensor(asym[:, :3, 3]).cuda().to(th.float32)
    indvar = [th.zeros((t, 3)).cuda().requires_grad_(True)]
    loss = h.torch_min(min_sym_environment_loss, **vars())
    asym[:, :3, 3] = th.einsum("aij,aj->ai", h.Q2R(indvar[0]), asym0).cpu().detach()
    asym = to_canonical_frame(asym, frames)
    return loss, asym.numpy()
