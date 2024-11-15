import pytest

th = pytest.importorskip('torch')
import numpy as np
from icecream import ic

import ipd as ipd
from ipd import h
from ipd.homog import intersect_planes

# pytest.skip(allow_module_level=True)

def main():
    test_thxform_invalid44_bug()
    test_rand_dtype_device()
    test_randsmall_dtype_device()
    test_th_axis_angle_cen_hel_vec()

    test_th_vec()
    test_th_rog()
    # assert 0, 'DONE'
    test_torch_rmsfit()
    test_th_misc()
    test_axisangcenhel_roundtrip()
    test_th_intersect_planes()
    test_th_axis_angle_cen_hel()
    test_th_rot_single()
    test_th_rot_56789()
    test_th_axis_angle_hel()
    test_th_axis_angle()

    test_torch_grad()
    test_th_axis_angle_cen_rand()
    test_torch_rmsfit_grad()

    ic("test_thgeom.py DONE")

@pytest.mark.fast
def test_rand_dtype_device():
    th = th = pytest.importorskip("torch")
    if not th.cuda.is_available():
        ipd.tests.force_pytest_skip("CUDA not availble")
    test = ipd.h.rand(100_000, cart_sd=10, dtype=th.float32, device='cuda')
    assert test.dtype == th.float32
    assert test.device.type == 'cuda'

@pytest.mark.fast
def test_randsmall_dtype_device():
    th = th = pytest.importorskip("torch")
    if not th.cuda.is_available():
        ipd.tests.force_pytest_skip("CUDA not availble")
    test = ipd.h.randsmall(100_000, cart_sd=10, rot_sd=0.5, dtype=th.float32, device='cuda')
    assert test.dtype == th.float32
    assert test.device.type == 'cuda'

@pytest.mark.fast
def test_th_vec():
    th = th = pytest.importorskip("torch")
    v = h.randvec(10)
    # ic(v)
    v2 = h.vec(v)
    assert v is v2
    p = h.randpoint(10)
    v3 = h.vec(p)
    assert th.allclose(p[..., :3], v3[..., :3])
    assert th.allclose(v3[..., 3], th.tensor(0.0))

    v4 = h.vec(p[..., :3])
    assert th.allclose(p[..., :3], v4[..., :3])
    assert th.allclose(v4[..., 3], th.tensor(0.0))

@pytest.mark.fast
def test_th_rog():
    th = pytest.importorskip("torch")
    points = h.randpoint(10)
    rg = h.rog(points)
    rgx = h.rog(points, aboutaxis=[1, 0, 0])
    assert rg >= rgx
    points[..., 0] = 0
    rg = h.rog(points)
    assert np.allclose(rg, rgx)

@pytest.mark.fast
def test_axisangcenhel_roundtrip():
    th = pytest.importorskip("torch")

    axis0 = th.tensor([1.0, 0, 0, 0], requires_grad=True)
    axis = h.normalized(axis0)
    ang = th.tensor([2.0], requires_grad=True)
    cen = th.tensor([1, 2, 3, 1.0], requires_grad=True)
    hel = th.tensor([1.0], requires_grad=True)
    x = h.rot(axis, ang, cen, hel)
    axis2, ang2, cen2, hel2 = h.axis_angle_cen_hel(x)

    ang2.backward()
    assert np.allclose(axis0.grad.detach(), [0, 0, 0, 0])
    assert np.allclose(ang.grad.detach(), [1])
    assert np.allclose(cen.grad.detach(), [0, 0, 0, 0])
    assert np.allclose(hel.grad.detach(), [0])

    axis0 = th.tensor([1.0, 0, 0, 0], requires_grad=True)
    axis = h.normalized(axis0)
    ang = th.tensor([2.0], requires_grad=True)
    cen = th.tensor([1, 2, 3, 1.0], requires_grad=True)
    hel = th.tensor([1.0], requires_grad=True)
    x = h.rot(axis, ang, cen, hel)
    axis2, ang2, cen2, hel2 = h.axis_angle_cen_hel(x)
    axis2[0].backward()
    assert np.allclose(axis0.grad.detach(), [0, 0, 0, 0])
    assert np.allclose(ang.grad.detach(), [0])
    assert np.allclose(cen.grad.detach(), [0, 0, 0, 0])
    assert np.allclose(hel.grad.detach(), [0])

    axis0 = th.tensor([1.0, 0, 0, 0], requires_grad=True)
    axis = h.normalized(axis0)
    ang = th.tensor([2.0], requires_grad=True)
    cen = th.tensor([1, 2, 3, 1.0], requires_grad=True)
    hel = th.tensor([1.0], requires_grad=True)
    x = h.rot(axis, ang, cen, hel)
    axis2, ang2, cen2, hel2 = h.axis_angle_cen_hel(x)
    axis2[1].backward()
    assert np.allclose(axis0.grad.detach(), [0, 1, 0, 0])
    assert np.allclose(ang.grad.detach(), [0])
    assert np.allclose(cen.grad.detach(), [0, 0, 0, 0])
    assert np.allclose(hel.grad.detach(), [0])

    axis0 = th.tensor([1.0, 0, 0, 0], requires_grad=True)
    axis = h.normalized(axis0)
    ang = th.tensor([2.0], requires_grad=True)
    cen = th.tensor([1, 2, 3, 1.0], requires_grad=True)
    hel = th.tensor([1.0], requires_grad=True)
    x = h.rot(axis, ang, cen, hel)
    axis2, ang2, cen2, hel2 = h.axis_angle_cen_hel(x)
    axis2[2].backward()
    assert np.allclose(axis0.grad.detach(), [0, 0, 1, 0])
    assert np.allclose(ang.grad.detach(), [0])
    assert np.allclose(cen.grad.detach(), [0, 0, 0, 0])
    assert np.allclose(hel.grad.detach(), [0])

    axis0 = th.tensor([1.0, 0, 0, 0], requires_grad=True)
    axis = h.normalized(axis0)
    ang = th.tensor([2.0], requires_grad=True)
    cen = th.tensor([1, 2, 3, 1.0], requires_grad=True)
    hel = th.tensor([1.0], requires_grad=True)
    x = h.rot(axis, ang, cen, hel)
    axis2, ang2, cen2, hel2 = h.axis_angle_cen_hel(x)
    cen2[1].backward()
    # ic(axis0.grad)
    assert np.allclose(axis0.grad.detach(), [0, -1, 0, 0], atol=1e-4)
    assert np.allclose(ang.grad.detach(), [0], atol=1e-4)
    assert np.allclose(cen.grad.detach(), [0, 1, 0, 0], atol=1e-4)
    assert np.allclose(hel.grad.detach(), [0], atol=1e-4)

    axis0 = th.tensor([1.0, 0, 0, 0], requires_grad=True)
    axis = h.normalized(axis0)
    ang = th.tensor([2.0], requires_grad=True)
    cen = th.tensor([1, 2, 3, 1.0], requires_grad=True)
    hel = th.tensor([1.0], requires_grad=True)
    x = h.rot(axis, ang, cen, hel)
    axis2, ang2, cen2, hel2 = h.axis_angle_cen_hel(x)
    hel2.backward()
    assert np.allclose(axis0.grad.detach(), [0, 0, 0, 0], atol=1e-4)
    assert np.allclose(ang.grad.detach(), [0], atol=1e-4)
    assert np.allclose(cen.grad.detach(), [0, 0, 0, 0], atol=1e-4)
    assert np.allclose(hel.grad.detach(), [1], atol=1e-4)

    # assert 0

@pytest.mark.fast
def test_th_axis_angle_cen_hel_vec():
    xforms = ipd.homog.hrand(100)
    xgeom = ipd.homog.axis_angle_cen_hel_of(xforms)
    for i, (x, a, an, c, h) in enumerate(zip(xforms, *xgeom)):  # noqa
        a2, an2, c2, h2 = ipd.homog.axis_angle_cen_hel_of(x)
        assert np.allclose(a, a2)
        assert np.allclose(an, an2)
        assert np.allclose(c, c2)
        assert np.allclose(h, h2)

@pytest.mark.fast
def test_th_rot_56789():
    th = pytest.importorskip("torch")
    th.autograd.set_detect_anomaly(True)

    shape = (5, 6, 7, 8, 9)
    axis0 = ipd.homog.hnormalized(np.random.randn(*shape, 3))
    ang0 = np.random.random(shape) * (np.pi - 0.1) + 0.1
    cen0 = np.random.randn(*shape, 4) * 100.0
    cen0[..., 3] = 1.0
    axis0 = th.tensor(axis0, requires_grad=True)
    ang0 = th.tensor(ang0, requires_grad=True)
    cen0 = th.tensor(cen0, requires_grad=True)
    hel = th.randn(*shape)

    rot = h.rot3(axis0, ang0, shape=(4, 4))
    rot2 = ipd.homog.rot(axis0.detach(), ang0.detach(), shape=(4, 4))
    assert np.allclose(rot.detach(), rot2, atol=1e-3)

    rot = h.rot(axis0, ang0, cen0, hel=None)
    rot2 = ipd.homog.hrot(axis0.detach(), ang0.detach(), cen0.detach())
    assert np.allclose(rot.detach(), rot2, atol=1e-3)

    s = th.sum(rot)
    s.backward()
    assert axis0.grad is not None
    assert axis0.grad.shape == (5, 6, 7, 8, 9, 4)

@pytest.mark.fast
def test_th_rot_single():
    th = pytest.importorskip("torch")

    axis0 = ipd.homog.hnormalized(np.random.randn(3))
    ang0 = np.random.random() * (np.pi - 0.1) + 0.1
    cen0 = np.random.randn(4) * 100.0
    cen0[3] = 1.0
    axis0 = th.tensor(axis0, requires_grad=True)
    ang0 = th.tensor(ang0, requires_grad=True)
    cen0 = th.tensor(cen0, requires_grad=True)

    rot = h.rot3(axis0, ang0, shape=(4, 4))
    rot2 = ipd.homog.rot(axis0.detach(), ang0.detach(), shape=(4, 4))
    assert np.allclose(rot.detach(), rot2, atol=1e-3)

    rot = h.rot(axis0, ang0, cen0, hel=None)
    rot2 = ipd.homog.hrot(axis0.detach(), ang0.detach(), cen0.detach())
    assert np.allclose(rot.detach(), rot2, atol=1e-3)

    s = th.sum(rot)
    s.backward()
    assert axis0.grad is not None

@pytest.mark.fast
def test_th_axis_angle_cen_rand():
    th = pytest.importorskip("torch")
    th.autograd.set_detect_anomaly(True)

    shape = (5, 6, 7, 8, 9)
    if not th.cuda.is_available():
        shape = shape[3:]

    axis0 = ipd.homog.hnormalized(np.random.randn(*shape, 3))
    ang0 = np.random.random(shape) * (np.pi - 0.1) + 0.1
    cen0 = np.random.randn(*shape, 4) * 100.0
    cen0[..., 3] = 1.0
    axis0 = th.tensor(axis0, requires_grad=True)
    ang0 = th.tensor(ang0, requires_grad=True)
    cen0 = th.tensor(cen0, requires_grad=True)
    hel0 = th.randn(*shape)

    # ic(axis0.shape)
    # ic(ang0.shape)
    # ic(cen0.shape)
    # ic(hel0.shape)

    # rot = h.rot3(axis0, ang0, dim=4)
    rot = h.rot(axis0, ang0, cen0, hel0, dtype=th.float64)
    # ic(rot.shape)
    # assert 0

    axis, ang, cen = ipd.homog.axis_ang_cen_of(rot.detach().numpy())
    hel = hel0.detach().numpy()
    assert np.allclose(axis0.detach(), axis, rtol=1e-3)
    assert np.allclose(ang0.detach(), ang, rtol=1e-3)
    assert np.allclose(np.linalg.norm(axis, axis=-1), 1.0)
    cenhat = (rot.detach().numpy() @ cen[..., None]).squeeze()
    cenhel = cen + hel[..., None] * axis
    assert np.allclose(cenhel, cenhat, atol=1e-3)

    # ic(rot.shape)
    axis2, ang2, cen2, hel2 = h.axis_angle_cen_hel(rot, flipaxis=False)
    assert np.allclose(axis2.detach(), axis)
    assert np.allclose(ang2.detach(), ang)
    assert np.allclose(cen2.detach(), cen)
    assert np.allclose(hel2.detach(), hel0)

@pytest.mark.fast
def test_th_intersect_planes():
    th = pytest.importorskip("torch")
    th.autograd.set_detect_anomaly(True)

    p1 = th.tensor([0.0, 0, 0, 1], requires_grad=True)
    n1 = th.tensor([1.0, 0, 0, 0], requires_grad=True)
    p2 = th.tensor([0.0, 0, 0, 1], requires_grad=True)
    n2 = th.tensor([0.0, 1, 0, 0], requires_grad=True)
    isct, norm, status = h.intersect_planes(p1, n1, p2, n2)
    assert status == 0
    assert isct[2] == 0
    assert np.allclose(abs(norm[:3].detach()), (0, 0, 1))
    isct[0].backward()
    assert np.allclose(p1.grad.detach(), [1, 0, 0, 0])
    assert np.allclose(n1.grad.detach(), [0, 0, 0, 0])
    assert np.allclose(p2.grad.detach(), [0, 0, 0, 0])
    assert np.allclose(n2.grad.detach(), [0, 0, 0, 0])

    p1 = th.tensor([0.0, 0, 0, 1], requires_grad=True)
    n1 = th.tensor([1.0, 0, 0, 0], requires_grad=True)
    p2 = th.tensor([0.0, 0, 0, 1], requires_grad=True)
    n2 = th.tensor([0.0, 0, 1, 0], requires_grad=True)
    isct, norm, status = h.intersect_planes(p1, n1, p2, n2)
    assert status == 0
    assert isct[1] == 0
    assert np.allclose(abs(norm[:3].detach()), (0, 1, 0))

    p1 = th.tensor([0.0, 0, 0, 1], requires_grad=True)
    n1 = th.tensor([0.0, 1, 0, 0], requires_grad=True)
    p2 = th.tensor([0.0, 0, 0, 1], requires_grad=True)
    n2 = th.tensor([0.0, 0, 1, 0], requires_grad=True)
    isct, norm, status = h.intersect_planes(p1, n1, p2, n2)
    assert status == 0
    assert isct[0] == 0
    assert np.allclose(abs(norm[:3].detach()), (1, 0, 0))

    p1 = th.tensor([7.0, 0, 0, 1], requires_grad=True)
    n1 = th.tensor([1.0, 0, 0, 0], requires_grad=True)
    p2 = th.tensor([0.0, 9, 0, 1], requires_grad=True)
    n2 = th.tensor([0.0, 1, 0, 0], requires_grad=True)
    isct, norm, status = h.intersect_planes(p1, n1, p2, n2)
    assert status == 0
    assert np.allclose(isct[:3].detach(), [7, 9, 0])
    assert np.allclose(abs(norm.detach()), [0, 0, 1, 0])

    p1 = th.tensor([0.0, 0, 0, 1], requires_grad=True)
    n1 = th.tensor([1.0, 1, 0, 0], requires_grad=True)
    p2 = th.tensor([0.0, 0, 0, 1], requires_grad=True)
    n2 = th.tensor([0.0, 1, 1, 0], requires_grad=True)
    isct, norm, status = h.intersect_planes(p1, n1, p2, n2)
    assert status == 0
    assert np.allclose(abs(norm.detach()), ipd.homog.hnormalized([1, 1, 1, 0]))

    p1 = np.array([[0.39263901, 0.57934885, -0.7693232, 1.0], [-0.80966465, -0.18557869, 0.55677976, 0.0]]).T
    p2 = np.array([[0.14790894, -1.333329, 0.45396509, 1.0], [-0.92436319, -0.0221499, 0.38087016, 0.0]]).T
    isct2, sts = intersect_planes(p1, p2)
    isct2, norm2 = isct2.T
    p1 = th.tensor([0.39263901, 0.57934885, -0.7693232, 1.0])
    n1 = th.tensor([-0.80966465, -0.18557869, 0.55677976, 0.0])
    p2 = th.tensor([0.14790894, -1.333329, 0.45396509, 1.0])
    n2 = th.tensor([-0.92436319, -0.0221499, 0.38087016, 0.0])
    isct, norm, sts = h.intersect_planes(p1, n1, p2, n2)
    assert sts == 0
    assert th.all(h.ray_in_plane(p1, n1, isct, norm))
    assert th.all(h.ray_in_plane(p2, n2, isct, norm))
    assert np.allclose(isct.detach(), isct2)
    assert np.allclose(norm.detach(), norm2)

    p1 = th.tensor([2.0, 0, 0, 1], requires_grad=True)
    n1 = th.tensor([1.0, 0, 0, 0], requires_grad=True)
    p2 = th.tensor([0.0, 0, 0, 1], requires_grad=True)
    n2 = th.tensor([0.0, 0, 1, 0], requires_grad=True)
    isct, norm, status = h.intersect_planes(p1, n1, p2, n2)

    assert status == 0
    assert abs(h.dot(n1, norm)) < 0.0001
    assert abs(h.dot(n2, norm)) < 0.0001
    assert th.all(h.point_in_plane(p1, n1, isct))
    assert th.all(h.point_in_plane(p2, n2, isct))
    assert th.all(h.ray_in_plane(p1, n1, isct, norm))
    assert th.all(h.ray_in_plane(p2, n2, isct, norm))

@pytest.mark.fast
def test_th_axis_angle_cen_hel():
    th = pytest.importorskip("torch")
    th.autograd.set_detect_anomaly(True)

    axis0 = th.tensor(ipd.homog.hnormalized(np.array([1.0, 1, 1, 0])), requires_grad=True)
    ang0 = th.tensor(0.9398483, requires_grad=True)
    cen0 = th.tensor([1.0, 2, 3, 1], requires_grad=True)
    h0 = th.tensor(2.443, requires_grad=True)
    x = h.rot(axis0, ang0, cen0, h0)
    axis, ang, cen, hel = h.axis_angle_cen_hel(x)
    ax2, an2, cen2 = ipd.homog.axis_ang_cen_of(x.detach().numpy())
    assert np.allclose(ax2, axis.detach())
    assert np.allclose(an2, ang.detach())
    assert th.allclose(h.projperp(axis0, cen - cen0), h.point([0, 0, 0], dtype=th.float64))
    assert th.allclose(h.projperp(axis0, th.as_tensor(cen2) - cen0), h.point([0, 0, 0], dtype=th.float64))
    # assert np.allclose(cen2, cen.detach(), atol=1e-3)
    hel.backward()
    assert np.allclose(ang0.grad, 0, atol=1e-3)
    hg = h0.detach().numpy() * np.sqrt(3) / 3
    assert np.allclose(axis0.grad, [hg, hg, hg, 0])

@pytest.mark.fast
def test_torch_grad():
    th = pytest.importorskip("torch")
    x = th.tensor([2, 3, 4], dtype=th.float, requires_grad=True)
    s = th.sum(x)
    s.backward()
    assert np.allclose(x.grad.detach().numpy(), [1.0, 1.0, 1.0])

@pytest.mark.fast
def test_torch_quat():
    th = pytest.importorskip("torch")
    th.autograd.set_detect_anomaly(True)

    for v in (1.0, -1.0):
        q0 = th.tensor([v, 0.0, 0.0, 0.0], requires_grad=True)
        q = h.quat_to_upper_half(q0)
        assert np.allclose(q.detach(), [1, 0, 0, 0])
        s = th.sum(q)
        s.backward()
        assert q0.is_leaf
        assert np.allclose(q0.grad.detach(), [0, v, v, v])

@pytest.mark.fast
def test_torch_rmsfit(trials=10):
    th = pytest.importorskip("torch")
    th.autograd.set_detect_anomaly(True)

    for _ in range(trials):
        p = h.randpoint(10, std=10)
        q = h.randpoint(10, std=10)
        # ic(p)
        rms0 = h.rms(p, q)
        rms, qhat, xpqhat = h.rmsfit(p, q)
        assert rms0 > rms
        # ic(float(rms0), float(rms))
        assert np.allclose(h.rms(qhat, q), rms)
        for i in range(10):
            qhat2 = h.xform(h.randsmall(1, 0.01, 0.001), qhat)
            rms2 = h.rms(q, qhat2)

            if rms2 < rms - 0.001:
                print(float(rms), float(rms2))
            assert rms2 >= rms - 0.001

@pytest.mark.fast  # @pytest.mark.skip
def test_torch_rmsfit_grad():
    th = pytest.importorskip("torch")
    if not th.cuda.is_available():
        ipd.tests.force_pytest_skip("CUDA not availble")
    # th.autograd.set_detect_anomaly(True)
    # assert 0
    ntrials = 1
    npts = 50
    shift = 100
    nstep = 50
    for std in (0.01, 0.1, 1, 10, 100):
        for i in range(ntrials):
            xpq = h.rand_xform()
            points1 = h.randpoint(npts, std=10, dtype=th.float32)
            points2 = h.xform(xpq, points1) + h.randvec(npts, dtype=th.float32) * std
            points2[:, 0] += shift
            points1[:, 3] = 1
            points2[:, 3] = 1
            # ic(points1)
            # ic(points2)
            assert points2.shape == (npts, 4)

            for i in range(nstep):
                p = points1.clone().detach().requires_grad_(True)
                q = points2.clone().detach().requires_grad_(True)
                p2, q2 = h.point(p), h.point(q)
                assert p2.shape == (npts, 4)

                rms, qhat, xpqhat = h.rmsfit(p2, q2)
                rms2 = h.rms(qhat, q)
                assert th.allclose(qhat, h.xformpts(xpqhat, p), atol=0.0001)
                # ic(rms, rms2)
                assert th.allclose(rms, rms2, atol=0.0001)

                rms.backward()
                assert np.allclose(p.grad[:, 3].detach(), 0)
                assert np.allclose(q.grad[:, 3].detach(), 0)
                points1 = points1 - p.grad.detach().numpy() * 10 * float(rms)
                points2 = points2 - q.grad.detach().numpy() * 10 * float(rms)

                # if not i % 10:
                #     ic(std, i, float(rms))
                # ic(th.norm(p.grad), th.norm(q.grad))

            assert rms < 1e-3  # type: ignore

@pytest.mark.fast
def test_th_axis_angle():  # noqa
    th = pytest.importorskip("torch")
    th.autograd.set_detect_anomaly(True)

    axis0 = th.tensor([10.0, 10.0, 10.0, 0], requires_grad=True)
    ang0 = th.tensor(0.4, requires_grad=True)
    x = h.rot(axis0, ang0)
    assert x.shape == (4, 4)
    assert x[1, 3] == 0.0
    assert x[3, 1] == 0.0
    assert x[3, 3] == 1.0
    x[0, 0].backward()
    assert np.allclose(axis0.grad.detach(), [0.0035, -0.0018, -0.0018, 0], atol=0.002)
    assert np.allclose(ang0.grad.detach(), -0.2596, atol=0.001)

    axis0 = th.tensor(ipd.homog.hnormalized(np.array([1.0, 1, 1, 0])), requires_grad=True)
    ang0 = th.tensor(0.8483, requires_grad=True)
    x = h.rot(axis0, ang0)
    ax, an, hel = h.axis_angle_hel(x)
    assert np.allclose(an.detach(), ang0.detach())
    assert np.allclose(hel.detach(), 0)
    ax2, an2, h2 = ipd.homog.axis_angle_hel_of(x.detach())
    assert th.allclose(th.linalg.norm(ax, axis=-1), th.ones_like(ax))
    assert np.allclose(ax2, ax.detach())
    assert np.allclose(an2, an.detach())
    assert np.allclose(h2, hel.detach())
    an.backward()
    assert np.allclose(ang0.grad, 1)
    assert np.allclose(axis0.grad, [0, 0, 0, 0], atol=1e-3)

@pytest.mark.fast
def test_th_axis_angle_hel():
    th = pytest.importorskip("torch")
    th.autograd.set_detect_anomaly(True)

    axis0 = th.tensor(ipd.homog.hnormalized(np.array([1.0, 1, 1, 0])), requires_grad=True)
    ang0 = th.tensor(0.9398483, requires_grad=True)
    h0 = th.tensor(2.443, requires_grad=True)
    x = h.rot(axis0, ang0, hel=h0)
    ax, an, hel = h.axis_angle_hel(x)
    ax2, an2, h2 = ipd.homog.axis_angle_hel_of(x.detach())
    assert np.allclose(ax2, ax.detach())
    assert np.allclose(an2, an.detach())
    assert np.allclose(h2, hel.detach())
    hel.backward()
    assert np.allclose(ang0.grad, 0, atol=1e-3)
    hg = h0.detach().numpy() * np.sqrt(3) / 3
    assert np.allclose(axis0.grad, [hg, hg, hg, 0])

@pytest.mark.fast
def test_th_misc():
    th = pytest.importorskip("torch")

    r = th.randn(2, 4, 5, 3)
    p = h.point(r)
    assert np.allclose(p[..., :3], r)
    assert np.allclose(p[..., 3], 1)
    p = h.randpoint(11)
    assert p.shape == (11, 4)
    assert np.allclose(p[:, 3], 1)

#################################################

@pytest.mark.fast
def test_thxform_invalid44_bug():
    a_to_others = th.tensor([
        [
            [1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 1.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
        [
            [0.2406, -0.2430, 0.9397, -6.2558],
            [0.9178, -0.2582, -0.3017, -15.2322],
            [0.3159, 0.9350, 0.1609, 13.8953],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
        [
            [0.3498, 0.8144, 0.4629, -6.0973],
            [-0.2221, -0.4080, 0.8856, -13.5191],
            [0.9101, -0.4126, 0.0382, 13.4486],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
    ],
                            dtype=th.float64)
    stub = th.tensor([[-0.5669, 0.4278, -0.7040, 84.6459], [0.7742, 0.5688, -0.2777, 8.8629],
                      [-0.2817, 0.7024, 0.6536, 87.7865], [0.0000, 0.0000, 0.0000, 1.0000]],
                     dtype=th.float64)
    assert h.valid44(a_to_others, debug=True)
    assert not h.valid(stub, debug=True)
    assert not h.valid(h.xform(a_to_others, stub))
    assert not th.allclose(a_to_others @ stub, h.xform(a_to_others, stub))

if __name__ == "__main__":
    main()

# @pytest.mark.fast
# def test_th_axis_angle_cen_rand():
#    shape = (5, 6, 7, 8, 9)
#    axis0 = ipd.homog.hnormalized(np.random.randn(*shape, 3))
#    ang0 = np.random.random(shape) * (np.pi - 0.1) + 0.1
#    cen0 = np.random.randn(*shape, 4) * 100.0
#    cen0[..., 3] = 1.0
#    axis0 = th.tensor(axis0, requires_grad=True)
#    ang0 = th.tensor(ang0, requires_grad=True)
#    cen0 = th.tensor(cen0, requires_grad=True)
#    hel = th.randn(*shape)

#    ic(axis0.shape)
#    ic(ang0.shape)
#    ic(cen0.shape)
#    ic(hel.shape)

#    rot =rot(axis0, ang0, cen0, hel)
#    ic(rot.shape)
#    assert 0

#    axis, ang, cen = ipd.homog.axis_ang_cen_of(rot)

#    assert np.allclose(axis0, axis, rtol=1e-5)
#    assert np.allclose(ang0, ang, rtol=1e-5)
#    #  check rotation doesn't move cen
#    cenhat = (rot @ cen[..., None]).squeeze()
#    assert np.allclose(cen + hel, cenhat, rtol=1e-4, atol=1e-4)
#    assert np.allclose(np.linalg.norm(axis, axis=-1), 1.0)

# @pytest.mark.fast
# def test_th_intersect_planes():
# noqa
#    p1 = th.tensor([0., 0, 0, 1], requires_grad=True)
#    n1 = th.tensor([1., 0, 0, 0], requires_grad=True)
#    p2 = th.tensor([0., 0, 0, 1], requires_grad=True)
#    n2 = th.tensor([0., 1, 0, 0], requires_grad=True)
#    isct, norm, status =h.intersect_planes(p1, n1, p2, n2)
#    assert status == 0
#    assert isct[2] == 0
#    assert np.allclose(abs(norm[:3].detach()), (0, 0, 1))

#    p1 = th.tensor([0., 0, 0, 1], requires_grad=True)
#    n1 = th.tensor([1., 0, 0, 0], requires_grad=True)
#    p2 = th.tensor([0., 0, 0, 1], requires_grad=True)
#    n2 = th.tensor([0., 0, 1, 0], requires_grad=True)
#    isct, norm, status =h.intersect_planes(p1, n1, p2, n2)
#    assert status == 0
#    assert isct[1] == 0
#    assert np.allclose(abs(norm[:3].detach()), (0, 1, 0))

#    p1 = th.tensor([0., 0, 0, 1], requires_grad=True)
#    n1 = th.tensor([0., 1, 0, 0], requires_grad=True)
#    p2 = th.tensor([0., 0, 0, 1], requires_grad=True)
#    n2 = th.tensor([0., 0, 1, 0], requires_grad=True)
#    isct, norm, status =h.intersect_planes(p1, n1, p2, n2)
#    assert status == 0
#    assert isct[0] == 0
#    assert np.allclose(abs(norm[:3].detach()), (1, 0, 0))

#    p1 = th.tensor([7., 0, 0, 1], requires_grad=True)
#    n1 = th.tensor([1., 0, 0, 0], requires_grad=True)
#    p2 = th.tensor([0., 9, 0, 1], requires_grad=True)
#    n2 = th.tensor([0., 1, 0, 0], requires_grad=True)
#    isct, norm, status =h.intersect_planes(p1, n1, p2, n2)
#    assert status == 0
#    assert np.allclose(isct[:3].detach(), [7, 9, 0])
#    assert np.allclose(abs(norm.detach()), [0, 0, 1, 0])

#    p1 = th.tensor([0., 0, 0, 1], requires_grad=True)
#    n1 = th.tensor([1., 1, 0, 0], requires_grad=True)
#    p2 = th.tensor([0., 0, 0, 1], requires_grad=True)
#    n2 = th.tensor([0., 1, 1, 0], requires_grad=True)
#    isct, norm, status =h.intersect_planes(p1, n1, p2, n2)
#    assert status == 0
#    assert np.allclose(abs(norm.detach()), hnormalized([1, 1, 1, 0]))

#    p1 = np.array([[0.39263901, 0.57934885, -0.7693232, 1.],
#                   [-0.80966465, -0.18557869, 0.55677976, 0.]]).T
#    p2 = np.array([[0.14790894, -1.333329, 0.45396509, 1.],
#                   [-0.92436319, -0.0221499, 0.38087016, 0.]]).T
#    isct2, sts = h.intersect_planes(p1, p2)
#    isct2, norm2 = isct2.T
#    p1 = th.tensor([0.39263901, 0.57934885, -0.7693232, 1.])
#    n1 = th.tensor([-0.80966465, -0.18557869, 0.55677976, 0.])
#    p2 = th.tensor([0.14790894, -1.333329, 0.45396509, 1.])
#    n2 = th.tensor([-0.92436319, -0.0221499, 0.38087016, 0.])
#    isct, norm, sts =h.intersect_planes(p1, n1, p2, n2)
#    assert sts == 0
#    assert th.all(h.ray_in_plane(p1, n1, isct, norm))
#    assert th.all(h.ray_in_plane(p2, n2, isct, norm))
#    assert np.allclose(isct.detach(), isct2)
#    assert np.allclose(norm.detach(), norm2)

#    p1 = th.tensor([2., 0, 0, 1], requires_grad=True)
#    n1 = th.tensor([1., 0, 0, 0], requires_grad=True)
#    p2 = th.tensor([0., 0, 0, 1], requires_grad=True)
#    n2 = th.tensor([0., 0, 1, 0], requires_grad=True)
#    isct, norm, status =h.intersect_planes(p1, n1, p2, n2)

#    assert status == 0
#    assert abs(h.dot(n1, norm)) < 0.0001
#    assert abs(h.dot(n2, norm)) < 0.0001
#    assert th.all(h.point_in_plane(p1, n1, isct))
#    assert th.all(h.point_in_plane(p2, n2, isct))
#    assert th.all(h.ray_in_plane(p1, n1, isct, norm))
#    assert th.all(h.ray_in_plane(p2, n2, isct, norm))

# @pytest.mark.fast
# def test_th_axis_angle_cen_hel():
#    th = pytest.importorskip('th')
#    th.autograd.set_detect_anomaly(True)

#    axis0 = th.tensor(hnormalized(np.array([1., 1, 1, 0])), requires_grad=True)
#    ang0 = th.tensor(0.9398483, requires_grad=True)
#    cen0 = th.tensor([1., 2, 3, 1], requires_grad=True)
#    h0 = th.tensor(2.443, requires_grad=True)
#    x =rot(axis0, ang0, cen0, h0)
#    axis, ang, cen, hel =h.axis_angle_cen_hel(x)
#    ax2, an2, cen2 = ipd.homog.axis_ang_cen_of(x.detach().numpy())
#    assert np.allclose(ax2, axis.detach())
#    assert np.allclose(an2, ang.detach())

#    assert np.allclose(cen2, cen.detach())
#    hel.backward()
#    assert np.allclose(ang0.grad, 0, atol=1e-3)
#    hg = h0.detach().numpy() * np.sqrt(3) / 3
#    assert np.allclose(axis0.grad, [hg, hg, hg, 0])

#    assert 0

# @pytest.mark.fast
# def test_torch_grad():
#    x = th.tensor([2, 3, 4], dtype=th.float, requires_grad=True)
#    s = th.sum(x)
#    s.backward()
#    assert np.allclose(x.grad.detach().numpy(), [1., 1., 1.])

# @pytest.mark.fast
# def test_torch_quat():
#    th = pytest.importorskip('th')
#    th.autograd.set_detect_anomaly(True)

#    for v in (1., -1.):
#       q0 = th.tensor([v, 0., 0., 0.], requires_grad=True)
#       q =quat_to_upper_half(q0)
#       assert np.allclose(q.detach(), [1, 0, 0, 0])
#       s = th.sum(q)
#       s.backward()
#       assert q0.is_leaf
#       assert np.allclose(q0.grad.detach(), [0, v, v, v])

# @pytest.mark.fast
# def test_torch_rmsfit():
#    th = pytest.importorskip('th')
#    th.autograd.set_detect_anomaly(True)

#    ntrials = 100
#    n = 10
#    std = 1
#    shift = 100
#    for std in (0.001, 0.01, 0.1, 1, 10):
#       for i in range(ntrials):
#          # xform =rot([0, 0, 1], 120, degrees=True)
#          xform = th.tensor(h.rand_ h.xform(), requires_grad=True)
#          p =h.randpoint(10, std=10)
#          q = h.xform(xform, p)
#          delta =h.randvec(10, std=std)
#          q = q + delta
#          q = q + shift
#          q[:, 3] = 1
#          # ic(delta)
#          # ic(q - p)

#          # ic(p.dtype, q.dtype)

#          rms, qhat, xpq =h.rmsfit(p, q)
#          # ic(xpq)

#          assert np.allclose(qhat.detach().numpy(), h.xform(xpq, p).detach().numpy())
#          assert np.allclose(rms.detach().numpy(),rms(qhat, q).detach().numpy())
#          assert rms < std * 3
#          # print(rms)
#          rms.backward()
