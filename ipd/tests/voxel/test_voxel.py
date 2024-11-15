import math
import random

import numpy as np
import pytest
from icecream import ic

import ipd
from ipd import h

pytest.importorskip('torch')
pytest.importorskip('gemmi')
pytest.importorskip('ipd.voxel.voxel_cuda')
th = lazimport('torch')  # type: ignore
from ipd.voxel.voxel_cuda import _voxel

def main():
    Voxel_score_converse()
    test_Voxel_score_boundscheck()
    test_numba_vox_create()
    test_Voxel_score_contact_perf()
    test_Voxel_score_clash_perf()
    test_create_voxel_grid_clash()
    test_create_voxel_grid_contact()
    # test_create_voxel_grid_contact_score()
    test_Voxel_score_outerfalse()
    test_Voxel_score_voxpos()
    test_Voxel_score()
    test_Voxel_score_symcheck()
    test_Voxel_class()
    test_Voxel_score_symcheck_perf()

    ipd.global_timer.report()
    print('test_voxel DONE')

@ipd.timed
def Voxel_score_converse():
    th.manual_seed(2)
    np.random.seed(2)
    xyz1 = make_test_points(100, 30, 200)
    xyz2 = make_test_points(200, 30, 200)
    vox1 = ipd.voxel.Voxel(xyz1, resl=0.5, func=ipd.dev.cuda.ContactFunc(10))
    vox2 = ipd.voxel.Voxel(xyz2, resl=0.5, func=ipd.dev.cuda.ContactFunc(10))
    x = ipd.samp.randxform(1000, cartmean=[30, 0, 0], cartsd=10)
    sc1 = vox1.score(xyz2, xyzpos=x)
    sc2 = vox2.score(xyz1, voxpos=x)
    ic(sc1[0], sc2[0])
    # ipd.viz.scatter(sc1.cpu(),sc2.cpu())
    # assert 0

@ipd.timed
@pytest.mark.fast
def test_Voxel_score_boundscheck():
    xyz = make_test_points(100, 30, 200)
    # xyz2 = make_test_points(100, 30, 200)
    vox = ipd.voxel.Voxel(xyz, func=ipd.dev.cuda.ContactFunc())
    x = ipd.samp.randxform(1000000, cartmean=[30, 0, 0], cartsd=10)
    sc = vox.score(xyz, xyzpos=x)
    sc2 = vox.score(xyz, xyzpos=x, boundscheck=True)
    ic(th.max(sc - sc2))
    assert th.allclose(sc, sc2, atol=1e-3)
    # print(th.quantile(sc, th.linspace(0,1,7,device='cuda')))
    print(sc.min(), th.sum(sc == 0) / len(sc))

@ipd.timed
@pytest.mark.fast
def test_numba_vox_create():
    xyz = th.rand(1000, 3, device='cuda') * 90 + 5
    rad, resl = th.tensor([3, 4], device='cuda', dtype=th.float32), 1
    vox = ipd.voxel.Voxel(xyz)

    with ipd.dev.Timer():
        for i in range(10):
            vox = ipd.voxel.Voxel(xyz)

    with ipd.dev.Timer():
        for i in range(10):
            lb = xyz.min(0).values - rad[-1] - resl
            ub = xyz.max(0).values + rad[-1] + resl
            grid = th.zeros(tuple(th.ceil((ub-lb) / resl).to(int)), dtype=th.float16, device='cuda')
            irad = int(math.ceil(rad[-1] / resl))
            block, thread = (len(xyz), 2*irad + 1, 2*irad + 1), (32, 2, 2)
            ipd.voxel.create_voxel_numba[block, thread](xyz.cuda(), lb.cuda(), rad.cuda(), irad, resl, grid.cuda())

    # ipd.showme(vox)
    vox.grid = grid  # type: ignore
    # ipd.showme(vox)
    assert th.allclose(vox.lb, lb)  # type: ignore
    assert th.allclose(vox.grid, grid, atol=1e-3)  # type: ignore

def make_test_points(npts, bound, ngen=None):
    ngen = ngen or npts
    xyz = bound * th.randn((10 * ngen, 3)).to('cuda').to(th.float32)
    xyz = xyz[th.topk(-h.norm(xyz), npts).indices]
    xyz = xyz[-bound < xyz[:, 0]]
    xyz = xyz[-bound < xyz[:, 1]]
    xyz = xyz[-bound < xyz[:, 2]]
    xyz = xyz[bound > xyz[:, 0]]
    xyz = xyz[bound > xyz[:, 1]]
    xyz = xyz[bound > xyz[:, 2]]
    xyz = xyz[:npts]
    return xyz

@ipd.timed
@pytest.mark.fast
def test_create_voxel_grid_clash():
    xyz = make_test_points(1000, 50)
    xyzorig = xyz.clone()
    rad = th.tensor([4, 5]).to('cuda')
    nthread = th.tensor([32, 2, 2])
    assert th.allclose(xyz, xyzorig)
    nsamp = 100
    mintime = 9e9
    ttot = ipd.dev.Timer(verbose=False, start=False)
    for i in range(nsamp + 1):
        with ipd.dev.Timer(verbose=False) as t:
            ttot.start()
            vox, _lb = _voxel.create_voxel_grid(
                xyz=xyz,
                resl=1.0,
                func='clash',
                funcarg=rad,
                nthread=nthread,
            )
            ttot.stop()
        assert vox.min() == 0
        assert vox.max() < 100
        assert th.allclose(xyz, xyzorig)
        mintime = min(t.elapsed(), mintime)
        if i == 0: inittime = ttot.elapsed()
    print(
        f'create clash   min {mintime*1000:7.3f}ms avg {(ttot.elapsed()-inittime)/nsamp*1000:7.3f}ms voxmean {vox.mean():7.3} shape {vox.shape}'  # type: ignore
    )
    assert mintime < 0.01

@ipd.timed
@pytest.mark.fast
def test_create_voxel_grid_contact():
    xyz = make_test_points(1000, 50)
    xyzorig = xyz.clone()
    th.tensor([4, 5]).to('cuda')
    nthread = th.tensor([32, 2, 2])
    assert th.allclose(xyz, xyzorig)
    nsamp = 100
    mintime = 9e9
    ttot = ipd.dev.Timer(verbose=False, start=False)
    func = ipd.dev.cuda.ContactFunc()
    for i in range(nsamp + 1):
        with ipd.dev.Timer(verbose=False) as t:
            ttot.start()
            vox, _lb = _voxel.create_voxel_grid(
                xyz=xyz,
                resl=1.0,
                func=func.label,
                funcarg=func.arg,
                nthread=nthread,
            )
            ttot.stop()
            assert vox.max() > 1000
            assert vox.min() < 0
        assert th.allclose(xyz, xyzorig)
        mintime = min(t.elapsed(), mintime)
        if i == 0: inittime = ttot.elapsed()
    print(
        f'create contact min {mintime*1000:7.3f}ms avg {(ttot.elapsed()-inittime)/nsamp*1000:7.3f}ms voxmean {vox.mean():7.3} shape {vox.shape}'  # type: ignore
    )
    assert mintime < 0.01

@ipd.timed
@pytest.mark.fast
def test_Voxel_score_outerfalse():
    voxpts = make_test_points(1000, 30)
    localxyz = make_test_points(200, 30)
    vox = ipd.voxel.Voxel(voxpts, func=ipd.dev.cuda.ClashFunc(3, 4))
    xyzpos = h.rand(10000, cart_sd=20).to(th.float32).to('cuda')
    voxpos = h.rand(10000, cart_sd=20).to(th.float32).to('cuda')
    pos2 = th.matmul(th.linalg.inv(voxpos), xyzpos).contiguous()
    pos3 = th.matmul(th.linalg.inv(xyzpos), voxpos).contiguous()
    sc1 = vox.score(localxyz, xyzpos, voxpos, outerprod=False)
    sc2 = vox.score(localxyz, xyzpos=pos2)
    sc3 = vox.score(localxyz, voxpos=pos3)
    # ic(th.sum(th.abs(sc1 - sc2) > 0.001), th.sum(th.abs(sc1 - sc3) > 0.001))
    assert th.sum(th.abs(sc2 - sc1) > 0.1) < 15
    assert th.sum(th.abs(sc2 - sc1) > 0.01) < 35
    assert th.sum(th.abs(sc2 - sc1) > 0.001) < 100
    assert th.sum(th.abs(sc3 - sc1) > 0.1) < 15
    assert th.sum(th.abs(sc3 - sc1) > 0.01) < 35
    assert th.sum(th.abs(sc3 - sc1) > 0.001) < 100

@ipd.timed
@pytest.mark.fast
def test_Voxel_score_voxpos():
    voxpts = make_test_points(1000, 30)
    localxyz = make_test_points(200, 30)
    vox = ipd.voxel.Voxel(voxpts, func=ipd.dev.cuda.ClashFunc(3, 4), resl=1)
    xyzpos = h.rand(10_000, cart_sd=20).to(th.float32).to('cuda')
    sc1 = vox.score(localxyz, xyzpos)
    sc2 = vox.score(localxyz, voxpos=th.linalg.inv(xyzpos).contiguous())
    assert th.sum(th.abs(sc2 - sc1) > 0.1) < 15
    assert th.sum(th.abs(sc2 - sc1) > 0.01) < 35
    assert th.sum(th.abs(sc2 - sc1) > 0.001) < 100
    # assert th.allclose(sc1, sc2, atol=1e-3)

@ipd.timed
@pytest.mark.fast
def test_Voxel_class():
    xyz = make_test_points(300, 30)
    ipd.voxel.Voxel(xyz, func=ipd.dev.cuda.ClashFunc(3, 4), resl=1)

@ipd.timed
@pytest.mark.fast
def test_Voxel_score_clash_perf():
    voxpts = make_test_points(1000, 30)
    localxyz = make_test_points(200, 30)
    vox = ipd.voxel.Voxel(voxpts, func=ipd.dev.cuda.ClashFunc(3, 4), resl=1)
    frame = h.rand(100_000, cart_sd=20).to(th.float32).to('cuda')
    nsamp = 10
    mintime = 9e9
    ttot = ipd.dev.Timer(verbose=False, start=False)
    for isamp in range(nsamp + 1):
        with ipd.dev.Timer(verbose=False) as t:
            if isamp: ttot.start()
            vox.score(localxyz, xyzpos=frame, nthread=th.tensor([1, 256, 1]))
            if isamp: ttot.stop()
        mintime = min(mintime, t.elapsed())
    print(f'score  clash   min {mintime*1000:7.2f}ms avg {ttot.elapsed()/nsamp*1000:7.2f}ms')
    assert mintime < 0.01

@ipd.timed
@pytest.mark.fast
def test_Voxel_score_contact_perf():
    voxpts = make_test_points(1000, 30)
    localxyz = make_test_points(200, 30)
    vox = ipd.voxel.Voxel(voxpts, func=ipd.dev.cuda.ContactFunc(), resl=1)
    frame = h.rand(100_000, cart_sd=20).to(th.float32).to('cuda')
    nsamp = 1
    mintime = 9e9
    ttot = ipd.dev.Timer(verbose=False, start=False)
    for isamp in range(nsamp + 1):
        with ipd.dev.Timer(verbose=False) as t:
            if isamp: ttot.start()
            vox.score(localxyz, xyzpos=frame, nthread=th.tensor([1, 256, 1]))
            if isamp: ttot.stop()
        mintime = min(mintime, t.elapsed())
    print(f'score  contact min {mintime*1000:7.2f}ms avg {ttot.elapsed()/nsamp*1000:7.2f}ms')
    assert mintime < 0.01
    # assert 0

@ipd.timed
@pytest.mark.fast
def test_Voxel_score_symcheck_perf():
    voxpts = make_test_points(1000, 30)
    localxyz = make_test_points(30, 30)
    vox = ipd.voxel.Voxel(voxpts, func=ipd.dev.cuda.ClashFunc(3, 4), resl=1)
    frame = h.rand(100_000, cart_sd=20).to(th.float32).to('cuda')
    symx = ipd.h.rot([0, 0, 1], 120).to(th.float32).to('cuda')
    nsamp = 10
    mintime = 9e9
    ttot = ipd.dev.Timer(verbose=False, start=False)
    for isamp in range(nsamp + 1):
        with ipd.dev.Timer(verbose=False) as t:
            if isamp: ttot.start()
            vox.score(localxyz, frame, symx=symx, symclashdist=3, nthread=th.tensor([1, 128, 1]))
            if isamp: ttot.stop()
        mintime = min(mintime, t.elapsed())
    print(f'symcheck min {mintime*1000:7.2f}ms avg {ttot.elapsed()/nsamp*1000:7.2f}ms')
    assert mintime < 0.1

# @pytest.mark.flaky(retries=2)
@ipd.timed
@pytest.mark.fast
def test_Voxel_score():
    nframe, nxyz = 100, 30
    for isamp in range(3):
        # th.manual_seed(isamp)
        # np.random.seed(isamp)
        voxpts = make_test_points(400, 30)
        localxyz = make_test_points(nxyz, 30)
        vox = ipd.voxel.Voxel(voxpts, func=ipd.dev.cuda.ClashFunc(3, 4), resl=1)
        # vox = ipd.voxel.Voxel(voxpts, func=ipd.dev.cuda.ContactFunc(10,-1,1,2,3,4), resl=1)
        voxpos = h.rand(1, cart_sd=20, dtype=th.float32, device='cuda')[0]
        # voxpos = th.eye(4, device='cuda')
        frame = h.rand(nframe, cart_sd=20, dtype=th.float32, device='cuda')
        assert frame.device.type == 'cuda'
        sc = vox.score(localxyz, frame, voxpos)
        xyz = h.xform(frame, localxyz, outerprod=True)
        xyzvox = h.xform(h.inv(voxpos), xyz, outerprod=False)
        # xyzvox2 = h.xform(h.xform(h.inv(voxpos), frame), localxyz, outerprod=True)
        # ic(xyzvox.shape)
        # ic(xyzvox2.shape)
        # assert th.allclose(xyzvox, xyzvox2, atol=1e-3) # sanity check in the midst of debugging...
        # ic(h.inv(voxpos))
        # ic(frame[0])
        # ic(h.xform(h.inv(voxpos), frame[0]))
        # ic(xyz)
        # ic(xyzvox)
        idx = ((xyzvox - vox.lb) / vox.resl).to(int).cpu()
        sc2 = th.zeros(len(frame), device='cuda')
        gridsize = th.tensor(vox.grid.shape, dtype=int)
        # idx[th.where((idx < 0).any(-1))[0]] = 0
        # idx[th.where((idx >= gridsize).any(-1))] = 0
        for i in range(len(frame)):
            for j in range(len(localxyz)):
                if any(idx[i, j] < 0): continue
                if any(idx[i, j] >= gridsize): continue
                sc2[i] += vox.grid[idx[i, j, 0], idx[i, j, 1], idx[i, j, 2]]
        if not th.allclose(sc2, sc, atol=1e-3):
            print("test_Voxel_score FAIL", isamp, th.sum(~th.isclose(sc, sc2, atol=1e-3)), th.sum(sc - sc2))
            # assert th.sum(~th.isclose(sc, sc2, atol=1e-3)) < 2
            assert th.allclose(sc2, sc, atol=1e-3)

@ipd.timed
@pytest.mark.fast
def test_Voxel_score_symcheck():
    nframe, nxyz = 100, 30
    for isamp in range(3):
        th.manual_seed(isamp)
        np.random.seed(isamp)
        symx = ipd.h.rot([0, 0, 1], random.choice([60, 72, 90, 120, 180])).to(th.float32).to('cuda')
        # ic(symx)
        symclashdist = float(th.rand(1) * 8)
        voxpts = make_test_points(400, 30)
        localxyz = make_test_points(nxyz, 30)
        vox = ipd.voxel.Voxel(voxpts, func=ipd.dev.cuda.ClashFunc(3, 4), resl=1)
        voxpos = h.rand(1, cart_sd=20).to(th.float32).to('cuda')[0]
        xyzpos = h.rand(nframe, cart_sd=20).to(th.float32).to('cuda')
        sc = vox.score(
            localxyz,
            xyzpos,
            voxpos[None],
            symx=symx,
            symclashdist=symclashdist,
        )
        xyz = h.xform(xyzpos, localxyz, outerprod=True)
        xyzvox = h.xform(h.inv(voxpos), xyz, outerprod=False)
        idx = ((xyzvox - vox.lb) / vox.resl).to(int).cpu()
        sc2 = th.zeros(len(xyzpos), device='cuda')
        gridsize = th.tensor(vox.grid.shape, dtype=int)
        for i in range(len(xyzpos)):
            for j in range(len(localxyz)):
                if any(idx[i, j] < 0): continue
                if any(idx[i, j] >= gridsize): continue
                sc2[i] += vox.grid[idx[i, j, 0], idx[i, j, 1], idx[i, j, 2]]
        if symclashdist > 0:
            symxyz = h.xform(symx, xyz)
            # ic(localxyz)
            # ic(xyz)
            # ic(symxyz)
            if nxyz == 1: symdist = (xyz - symxyz).norm()[None, None]
            else: symdist = (xyz[:, None] - symxyz[:, :, None]).norm(dim=-1).min(-1)[0]
            (xyz[:, None] - symxyz[:, :, None]).norm(dim=-1)
            # ic(symdistfull)
            sc = th.where(th.any(symdist < symclashdist, dim=1), 9e9, sc)
            sc2 = th.where(th.any(symdist < symclashdist, dim=1), 9e9, sc2)
        # ic(sc-sc2)
        if not th.allclose(sc2, sc, atol=1e-3):
            print("test_Voxel_score_symcheck FAIL", isamp)
            assert th.allclose(sc2, sc, atol=1e-3)

if __name__ == '__main__':
    main()
