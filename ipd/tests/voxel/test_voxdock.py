import numpy as np
import pytest
from icecream import ic

import ipd

pytest.importorskip('ipd.voxel.voxel_cuda')
th = pytest.importorskip('torch')

def main():
    test_voxdock_ab()
    test_voxdock_c3()
    test_voxdock_cage_T()
    test_voxdock_cage_I()
    test_voxdock_cage_O()

def make_test_points(npts, bound, ngen=None):
    ngen = ngen or npts * 2
    xyz = bound * th.randn((10 * ngen, 3)).to('cuda').to(th.float32)
    xyz = xyz[th.topk(-h.norm(xyz), npts).indices]  # type: ignore
    xyz = xyz[-bound < xyz[:, 0]]
    xyz = xyz[-bound < xyz[:, 1]]
    xyz = xyz[-bound < xyz[:, 2]]
    xyz = xyz[bound > xyz[:, 0]]
    xyz = xyz[bound > xyz[:, 1]]
    xyz = xyz[bound > xyz[:, 2]]
    return xyz[:npts]

@pytest.mark.fast
def test_voxdock_ab(timer=ipd.dev.Timer()):
    # th.manual_seed(0)
    # xform = ipd.h.rand(100, cart_sd=20, dtype=th.float32, device='cuda')
    # xform = ipd.samp.randxform(100, cart_sd=20, dtype=th.float32, device='cuda')

    xyz = make_test_points(100, 20)
    repulsive_only = th.ones(100, dtype=bool, device='cuda')
    repulsive_only[3:97] = False
    # rb = ipd.voxel.VoxRB(xyz[:128], resl=1, func=ipd.dev.cuda.ContactFunc())

    rb = ipd.voxel.VoxRB(xyz, resl=1, func=ipd.dev.cuda.ContactFunc(1000, -1, 4, 5, 9, 10), repulsive_only=repulsive_only)
    # trans_score = rb.score(rb, th.eye(4), ipd.h.trans(x=th.arange(40, 50, 0.1))).min()
    # ic(trans_score)
    timer.checkpoint('vox')
    # ipd.showme(rb, col=(1, 1, 1), name='ref', sphere=2)
    # rb._vizpos = ipd.h.trans([45,0,0])
    # ipd.showme(rb, col=(1, 1, 1), name='ref', sphere=2)
    # assert 0

    N = 100_000
    Ntop = 1000
    # xform = ipd.h.rand(N, cart_sd=20).to(th.float32).to('cuda')
    # xform = ipd.h.rand(N, cart_sd=20, dtype=th.float32, device='cuda')
    xform = ipd.samp.randxform(N, cartsd=20).to(th.float32).to('cuda')
    timer.checkpoint('rand xform')
    # for i in range(100):
    sc = rb.score(rb, th.eye(4), xform)
    timer.checkpoint('score')
    # timer.report()
    # assert 0
    maxcart, maxangle = 5, 0.6
    for _i in range(3):
        timer.checkpoint('misc')
        maxcart, maxangle = maxcart * 0.5, maxangle * 0.5
        # itop = ipd.samp.sort_inplace_topk(sc, Ntop)
        itop = sc.topk(Ntop, largest=False).indices
        # ic( len(set(itop) & set(itop2)))
        timer.checkpoint('topk')
        xform = th.tile(xform[itop], (N // Ntop, 1, 1))
        timer.checkpoint('xform tile')
        # delta = ipd.h.randsmall(N, cart_sd=10 / (i + 1), rot_sd=0.7 / (i + 0.1), dtype=th.float32, device='cuda')
        delta = ipd.samp.randxform(N, cartmax=maxcart, orimax=maxangle)
        timer.checkpoint('rand xform small')
        xform = th.matmul(xform, delta)
        timer.checkpoint('matmul')
        sc = rb.score(rb, th.eye(4), xform, repulsive_only=repulsive_only)
        timer.checkpoint('score')
    # print('iter', i, sc.min().item())
    timer.checkpoint('sc.min')
    assert sc.min() < -50
    # timer.stop()
    col = th.zeros(100, 3)
    col[:, 0] = 1
    col[~repulsive_only, 1] = 1
    ipd.showme(rb, col=col, sphere=2, name='ref')
    col[~repulsive_only, 2] = 1
    for i in sc.topk(31, largest=False).indices:
        rb._vizpos = xform[i]
        ipd.showme(rb, col=col, sphere=2)

@pytest.mark.fast
def test_voxdock_c3():
    xyz = make_test_points(100, 20)
    xyz -= xyz.mean(0)
    rg = th.sqrt(th.mean(h.norm(xyz)**2))  # type: ignore
    rb = ipd.voxel.VoxRB(xyz, resl=1, func=ipd.dev.cuda.ContactFunc(1000, -1, 4, 5, 9, 10))
    # ipd.showme(rb, col=(1, 1, 1), name='ref', sphere=2)
    N = 100_000
    Ntop = 1000
    xsym = th.tensor(ipd.sym.frames('C3'), device='cuda', dtype=th.float32)
    xform = ipd.samp.randxform(N, cartmean=[rg, 0, 0], cartsd=[20, 0, 0]).to(th.float32).to('cuda')
    sc = rb.score(rb, xsym[1] @ xform, xform)
    maxcart, maxangle = 5, 0.6
    for _i in range(3):
        maxcart, maxangle = maxcart * 0.5, maxangle * 0.5
        itop = sc.topk(Ntop, largest=False).indices
        xform = th.tile(xform[itop], (N // Ntop, 1, 1))
        delta = ipd.samp.randxform(N, cartmax=[maxcart, 0, 0], orimax=maxangle)
        xform = th.matmul(xform, delta)
        sc = rb.score(rb, xsym[1] @ xform, xform)
        # print('iter', _i, sc.min().item())
    assert sc.min() < -50
    # for i in sc.topk(10, largest=False).indices:
    # rb._vizpos = xsym @ xform[i]
    # ipd.showme(rb, sphere=2)

def asuvec_frames_minimal_z(sym):
    xsym = th.tensor(ipd.sym.frames(sym), device='cuda', dtype=th.float32)
    asuvec = h.normvec(  # type: ignore
        th.tensor(np.array(list(ipd.sym.axes(sym).values()))).mean(0),
        device='cuda',  # type: ignore
        dtype=th.float32)  # type: ignore
    iasu = th.argmax((xsym @ asuvec)[:, 2])
    asuvec = xsym[iasu] @ asuvec
    # xnbr = [th.eye(4, device='cuda')]
    # for nf in ipd.sym.axes(sym):
    #     ax = th.as_tensor(ipd.sym.axes(sym, nfold=nf, all=True)).cuda()
    #     ax = ax[th.argmax(th.abs(h.dot(asuvec, ax)))]
    #     # print(nf, ax[:3])
    #     xnbr.append(h.rot(ax, th.pi * 2 / nf, device='cuda'))
    # assert 0
    # xsymuniq = list()
    # ncopy = list()
    # for nf in ipd.sym.axes(sym):
    #     x = h.rot(ipd.sym.axes(sym, nfold=nf, all=True), th.pi * 2 / nf)
    #     ic(x.shape)
    #     xsymuniq.append(x)
    #     ncopy.append(th.ones(len(x)) * (2 if nf > 2 else 1))
    # xsymuniq = th.cat(xsymuniq).to('cuda')
    # ncopy = th.cat(ncopy).to('cuda')

    xsymuniq = xsym[1:]
    ncopy = th.ones(len(xsymuniq))

    return xsym, asuvec, xsymuniq, ncopy  #, th.stack(xnbr).to(th.float32)

def scoreme(rb, xform, xsym, ncopy, maxcart, Ntop):
    cen1, rad = rb.boundcen, rb.boundrad
    if Ntop: cen1 = h.xform(xform[:Ntop], rb.boundcen)  # type: ignore
    cen2 = h.xform(xsym, cen1)  # type: ignore
    inbounds = h.norm(cen1 - cen2) < 2*rad + rb.func.arg[-1] + maxcart  # type: ignore
    # if len(xsym) == 23:
    #     ipd.showme(h.xform(xform[0], rb.xyz))
    #     for i, x in enumerate(xsym):
    #         if inbounds[i]:
    #             ipd.showme(h.xform(x @ xform[0], rb.xyz))
    # ic(inbounds.shape)
    if Ntop: inbounds = inbounds.max(1).values
    return sum(ncopy[i] * rb.score(rb, xsym[i] @ xform, xform) for i in range(len(xsym)) if inbounds[i])

@pytest.mark.fast
def test_voxdock_cage_T():
    sym = 'T'
    xyz = make_test_points(100, 20, 200)
    x, sc = voxdock_cage(
        xyz,
        sym,
        N=100_000,
        Ntop=500,
        maxcart=5,
        maxori=0.3,
        showme=0,  # type: ignore
        resl=0.5,  # type: ignore
        niters=5)  # type: ignore
    ic(sc.min())  # type: ignore
    assert sc.min() < -100  # type: ignore

@pytest.mark.fast
def test_voxdock_cage_O():
    sym = 'O'
    xyz = make_test_points(100, 20, 200)
    x, sc = voxdock_cage(
        xyz,
        sym,
        N=100_000,
        Ntop=500,
        maxcart=5,
        maxori=0.3,
        showme=0,  # type: ignore
        resl=0.5,  # type: ignore
        niters=5)  # type: ignore
    ic(sc.min())  # type: ignore
    assert sc.min() < -150  # type: ignore

@pytest.mark.fast
def test_voxdock_cage_I():
    sym = 'I'
    xyz = make_test_points(100, 20, 200)
    x, sc = voxdock_cage(
        xyz,
        sym,
        N=100_000,
        Ntop=500,
        maxcart=5,
        maxori=0.3,
        showme=0,  # type: ignore
        resl=0.5,  # type: ignore
        niters=5)  # type: ignore
    ic(sc.min())  # type: ignore
    assert sc.min() < -200  # type: ignore

def voxdock_cage(xyz,
                 sym,
                 N=100_000,
                 Ntop=200,
                 initmaxcart=5,
                 maxcart=3,
                 maxori=0.3,
                 niters=5,
                 showme=False,
                 resl=1,
                 timer=None):
    xsym, asuvec, xsymuniq, ncopy = asuvec_frames_minimal_z(sym)
    # print(f'voxdock_cage {sym} {xsym.shape} {asuvec}')
    xyz = xyz - xyz.mean(0)
    rg = th.sqrt(th.mean(h.norm(xyz)**2)) + 1  # type: ignore
    cen = asuvec[:3] * rg * {60: 5, 24: 3, 12: 2}[len(xsym)]
    xyz = xyz + cen
    if timer: timer.checkpoint('startup')
    rb = ipd.voxel.VoxRB(xyz, resl=resl, func=ipd.dev.cuda.ContactFunc(1000, -1, 5, 6, 9, 10))
    if timer: timer.checkpoint('voxel')
    # cen = None
    # rb._vizpos = xsym
    # ipd.showme(rb, sphere=2)
    # assert 0
    xform = ipd.samp.randxform(N, cartmax=initmaxcart, cen=cen, dtype=th.float32, device='cuda')
    if timer: timer.checkpoint('randxform')
    sc = scoreme(rb, xform, xsymuniq, ncopy, initmaxcart * 2, Ntop=0)
    if timer: timer.checkpoint('score')
    for _iter in range(niters):
        maxcart, maxori = maxcart * 0.5, maxori * 0.5
        itop = sc.topk(Ntop, largest=False).indices  # type: ignore
        if timer: timer.checkpoint('topk')
        xform = th.tile(xform[itop], (N // Ntop, 1, 1))
        if timer: timer.checkpoint('tile')
        delta = ipd.samp.randxform(N, cartmax=maxcart, orimax=maxori, cen=cen)
        if timer: timer.checkpoint('randxform')
        xform = th.matmul(xform, delta)
        if timer: timer.checkpoint('matmul')
        sc = scoreme(rb, xform, xsymuniq, ncopy, maxcart, Ntop)
        if timer: timer.checkpoint('score')
        # print('iter', i, sc.min().item())
        # if timer: timer.checkpoint('sc.min')
    # print(topk.values)
    # return
    if showme:
        for i in sc.topk(10, largest=False).indices:  # type: ignore
            rb._vizpos = xsym @ xform[i]
            ipd.showme(rb, sphere=2)
    return xform, sc

if __name__ == '__main__':
    main()
