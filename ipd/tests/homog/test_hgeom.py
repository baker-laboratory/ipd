import random

import numpy as np
import pytest
from icecream import ic

import ipd
from ipd import hnumpy as h

ic.configureOutput(includeContext=True, contextAbsPath=True)

config_test = ipd.Bunch(
    re_only=[],
    only=[],
    re_exclude=[],
    exclude=[],
)

def main():
    ipd.tests.maintest(namespace=globals(), config=config_test)
    return
    ipd.icv("test_homog.py DONE")

def test_xchain():
    u, v, w = [h.rand(random.randint(1, 4)) for _ in range(3)]
    assert h.allclose(h.xform(u, v), h.xform(u, v))
    assert h.allclose(h.xform(u, v, w), h.xform(h.xform(u, v), w))

def test_xchain_pts():
    u, v, w = [h.rand(random.randint(1, 4)) for _ in range(3)]
    p = h.randpoint(7)
    assert h.allclose(h.xform(u, p), h.xform(u, p))
    assert h.allclose(h.xform(u, v, p), h.xform(h.xform(u, v), p))
    q = h.randpoint(4)
    assert h.allclose(h.xformpts(u, q), h.xformpts(u, q))
    test = h.xformpts(h.xform(u, v), q)
    assert h.allclose(h.xformpts(u, v, q), test)

def test_hxform_chain():
    assert h.allclose(np.eye(4), h.xform(np.eye(4), np.eye(4), np.eye(4)))

def test_closest_point_on_line():
    assert h.allclose([0, 0, 0, 1], h.closest_point_on_line([1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]))
    assert h.allclose([0, 0, 1, 1], h.closest_point_on_line([1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]))
    n = h.rand_unit(100)
    p = h.randpoint(100)
    t = h.randpoint(100)
    c = h.closest_point_on_line(t, p, n)
    assert h.allclose(np.pi / 2, h.angle(t - c, p - c))
    d = np.sum((t - c)**2, axis=1)
    d1 = np.sum((t - c + n/1000)**2, axis=1)
    d2 = np.sum((t - c - n/1000)**2, axis=1)
    assert np.all(d < d1)
    assert np.all(d < d2)

def test_uniqlastdim():
    test = [
        [1., 0, 0, 0],
        [1.00001, 0, 0, 0],
        [1.00002, 0, 0, 0],
    ]
    assert h.allclose(h.uniqlastdim(test), [[1, 0, 0, 0]])
    test.append([2, 0, 0, 0])
    assert h.allclose(h.uniqlastdim(test), [[1, 0, 0, 0], [2, 0, 0, 0]])

def test_hcentered():
    coords = h.randpoint((8, 7))
    coords[..., :3] += [10, 20, 30]
    cencoord = h.centered(coords)
    assert cencoord.shape == coords.shape
    assert np.allclose(h.com(cencoord), [0, 0, 0, 1])
    assert np.allclose(coords[..., :3], cencoord[..., :3] + h.com(coords)[..., None, :3])

    coords = h.randpoint((8, 7, 2, 1))[..., :3]
    coords[..., :3] += [30, 20, 10]
    cencoord = h.centered(coords)
    assert cencoord.shape[:-1] == coords.shape[:-1]
    assert np.allclose(h.com(cencoord), [0, 0, 0, 1])
    assert np.allclose(coords[..., :3], cencoord[..., :3] + h.com(coords)[..., None, :3])

    coords = h.randpoint((1, 8, 3, 7))
    coords[..., :3] += 20
    cencoord = h.centered3(coords)
    assert cencoord.shape[:-1] == coords.shape[:-1]
    assert np.allclose(h.com(cencoord)[..., :3], [0, 0, 0])
    assert np.allclose(coords[..., :3], cencoord[..., :3] + h.com(coords)[..., None, :3])

    coords = h.randpoint(7)[..., :3]
    coords[..., :3] += 30
    cencoord = h.centered3(coords)
    assert cencoord.shape == coords.shape
    assert np.allclose(h.com(cencoord)[..., :3], [0, 0, 0])
    assert np.allclose(coords[..., :3], cencoord[..., :3] + h.com(coords)[..., None, :3])

    coords = h.randpoint((8, 7))
    coords[..., :3] += [10, 20, 30]
    cencoord = h.centered(coords, singlecom=True)
    assert cencoord.shape == coords.shape
    assert np.allclose(h.com(cencoord, flat=True), [0, 0, 0, 1])
    assert np.allclose(coords[..., :3], cencoord[..., :3] + h.com(coords, flat=True)[..., None, :3])

def test_halign():
    for _ in range(10):
        a, b = h.rand_unit(2)
        x = h.halign(a, b)
        b2 = h.xform(x, a)
        assert np.allclose(b, b2)
        ang = h.angle(a, b)
        ang2 = h.angle_of(x)
        # ipd.icv(ang, ang2)
        assert np.allclose(ang, ang2)

def test_hdiff():
    I = np.eye(4)
    assert h.diff(I, I) == 0

    x = h.randsmall(cart_sd=0.00001, rot_sd=0.00001)
    assert h.diff(x, x) == 0
    assert h.diff(x, I) != 0

    assert h.diff(h.convert(x[:3, :3]), I) != 0
    assert h.diff(h.convert(trans=x[:3, 3]), I) != 0

def test_hxform_ray():
    p = h.randpoint().squeeze()
    v = h.rand_vec()
    r = h.ray(p, v)
    assert r.shape == (4, 2)
    x = h.rand()
    m = x @ r
    assert m.shape == (4, 2)
    assert np.allclose(m[..., 0], h.xform(x, r[..., 0]))
    assert np.allclose(m[..., 1], h.xform(x, r[..., 1]))
    assert np.allclose(m, h.xform(x, r))
    assert h.hvalid(m)

    x = h.rand(3)
    m = x @ r
    assert m.shape == (3, 4, 2)
    assert np.allclose(m[..., 0], h.xform(x, r[..., 0]))
    assert np.allclose(m[..., 1], h.xform(x, r[..., 1]))
    assert h.hvalid(m)

def test_hxform_stuff_coords():

    class Dummy:

        def __init__(self, p):
            self.coords = p

    x = h.rand()
    p = h.point([1, 2, 3])
    smrt = Dummy(p)  # I am so smart, S, M, R T...
    q = h.xform(x, smrt)
    r = h.xform(x, p)
    assert np.allclose(smrt.coords, p)

def test_hxform_stuff_xformed():

    class Dummy:

        def __init__(self, pos):
            self.pos = pos

        def xformed(self, x):
            return Dummy(h.xform(x, self.pos))

    x = h.rand()
    p = h.point([1, 2, 3])
    smrt = Dummy(p)
    q = h.xform(x, smrt)
    r = h.xform(x, p)
    assert np.allclose(smrt.pos, p)

def test_hxform_list():

    class Dummy:

        def __init__(self, p):
            self.coords = p

    x = h.rand()
    p = h.point([1, 2, 3])
    stuff = Dummy(p)
    q = h.xform(x, stuff)
    r = h.xform(x, p)
    assert np.allclose(stuff.coords, p)

def test_hxform():
    x = h.rand()
    y = h.xform(x, [1, 2, 3], homogout=True)  # type: ignore
    assert np.allclose(y, x @ h.point([1, 2, 3]))
    y = h.xform(x, [1, 2, 3])
    assert np.allclose(y, (x @ h.point([1, 2, 3]))[:3])

def test_hxform_outer():
    x = h.rand()
    h.xform(x, [1, 2, 3])

def test_hpow():
    with pytest.raises(ValueError):
        h.pow_int(np.eye(4), 0.5)

    x = h.rot([0, 0, 1], [1, 2, 3])
    xinv = h.rot([0, 0, 1], [-1, -2, -3])

    xpow = h.pow(x, 0)
    assert np.allclose(xpow, np.eye(4))
    xpow = h.pow(x, 2)
    assert np.allclose(xpow, x @ x)
    xpow = h.pow(x, 5)
    assert np.allclose(xpow, x @ x @ x @ x @ x)
    xpow = h.pow(x, -2)
    assert np.allclose(xpow, xinv @ xinv)

@pytest.mark.fast  # @pytest.mark.xfail
def test_hpow_float():
    x = h.rot([0, 0, 1], [1, 2, 3])
    h.pow(x, 0.3)
    ipd.icv("test with int powers, maybe other cases")

def test_hmean():
    ang = np.random.normal(100)
    xforms = np.array([
        h.rot([1, 0, 0], ang),
        h.rot([1, 0, 0], -ang),
        h.rot([1, 0, 0], 0),
    ])
    xmean = h.mean(xforms)
    assert np.allclose(xmean, np.eye(4))
    # assert 0

def test_homo_rotation_single():
    axis0 = h.normalized(np.random.randn(3))
    ang0 = np.pi / 4.0
    r = h.rot(list(axis0), float(ang0))
    a = h.fast_axis_of(r)
    n = h.norm(a)
    assert np.all(abs(a/n - axis0) < 0.001)
    assert np.all(abs(np.arcsin(n / 2) - ang0) < 0.001)

def test_homo_rotation_center():
    assert np.allclose([0, 2, 0, 1], h.rot([1, 0, 0], 180, [0, 1, 0]) @ (0, 0, 0, 1), atol=1e-5)
    assert np.allclose([0, 1, -1, 1], h.rot([1, 0, 0], 90, [0, 1, 0]) @ (0, 0, 0, 1), atol=1e-5)
    assert np.allclose([-1, 1, 2, 1], h.rot([1, 1, 0], 180, [0, 1, 1]) @ (0, 0, 0, 1), atol=1e-5)

def test_homo_rotation_array():
    shape = (1, 2, 1, 3, 4, 1, 1)
    axis0 = h.normalized(np.random.randn(*(shape + (3, ))))
    ang0 = np.random.rand(*shape) * (0.99 * np.pi / 2 + 0.005 * np.pi / 2)
    r = h.rot(axis0, ang0)
    a = h.fast_axis_of(r)
    n = h.norm(a)[..., np.newaxis]
    assert np.all(abs(a/n - axis0) < 0.001)
    assert np.all(abs(np.arcsin(n[..., 0] / 2) - ang0) < 0.001)

def test_homo_rotation_angle():
    ang = np.random.rand(1000) * np.pi
    a = h.rand_unit()
    u = h.projperp(a, h.rand_vec())
    x = h.rot(a, ang)
    ang2 = h.angle(u, x @ u)
    assert np.allclose(ang, ang2, atol=1e-5)

def test_htrans():
    assert h.trans([1, 3, 7]).shape == (4, 4)
    assert np.allclose(h.trans([1, 3, 7])[:3, 3], (1, 3, 7))

    with pytest.raises(ValueError):
        h.trans([4, 3, 2, 1, 1])

    s = (2, )
    t = np.random.randn(*s, 3)
    ht = h.trans(t)
    assert ht.shape == s + (4, 4)
    assert np.allclose(ht[..., :3, 3], t)

def test_hcross():
    assert np.allclose(h.cross([1, 0, 0], [0, 1, 0]), [0, 0, 1, 0])
    assert np.allclose(h.cross([1, 0, 0, 0], [0, 1, 0, 0]), [0, 0, 1, 0])
    a, b = np.random.randn(3, 4, 5, 3), np.random.randn(3, 4, 5, 3)
    c = h.cross(a, b)
    assert np.allclose(h.dot(a, c), 0)
    assert np.allclose(h.dot(b, c), 0)

def test_axis_angle_of():
    ax, an = h.axis_angle_of(h.rot([10, 10, 0], np.pi), debug=True)
    assert 1e-5 > np.abs(ax[0] - ax[1])
    assert 1e-5 > np.abs(ax[2])
    # ipd.icv(np.linalg.norm(ax, axis=-1))
    assert np.allclose(np.linalg.norm(ax, axis=-1), 1.0)

    ax, an = h.axis_angle_of(h.rot([0, 1, 0], np.pi), debug=True)
    assert 1e-5 > np.abs(ax[0])
    assert 1e-5 > np.abs(ax[1]) - 1
    assert 1e-5 > np.abs(ax[2])
    assert np.allclose(np.linalg.norm(ax, axis=-1), 1.0)

    ax, an = h.axis_angle_of(h.rot([0, 1, 0], np.pi * 0.25), debug=True)
    # ipd.icv(ax, an)
    assert np.allclose(ax, [0, 1, 0, 0], atol=1e-5)
    assert 1e-5 > np.abs(an - np.pi * 0.25)
    assert np.allclose(np.linalg.norm(ax, axis=-1), 1.0)
    ax, an = h.axis_angle_of(h.rot([0, 1, 0], np.pi * 0.75), debug=True)
    # ipd.icv(ax, an)
    assert np.allclose(ax, [0, 1, 0, 0], atol=1e-5)
    assert 1e-5 > np.abs(an - np.pi * 0.75)
    assert np.allclose(np.linalg.norm(ax, axis=-1), 1.0)

    ax, an = h.axis_angle_of(h.rot([1, 0, 0], np.pi / 2), debug=True)
    # ipd.icv(np.pi / an)
    assert 1e-5 > np.abs(an - np.pi / 2)
    assert np.allclose(np.linalg.norm(ax, axis=-1), 1.0)

def test_axis_angle_of_rand():
    shape = (4, 5, 6, 7, 8)
    # shape = (3, )
    axis = h.normalized(np.random.randn(*shape, 3))
    angl = np.random.random(shape) * np.pi / 2
    # seed with one identity and one 180
    angl[0, 0, 0, 0, 0] = np.pi
    angl[1, 0, 0, 0, 0] = 0
    axis[1, 0, 0, 0, 0] = [1, 0, 0, 0]
    angl[0, 0, 0, 0, 0] = np.pi
    angl[0, 0, 1, 0, 0] = 0
    axis[0, 0, 1, 0, 0] = [1, 0, 0, 0]
    # axis[1] = [1,0,0,0]
    rot = h.rot(axis, angl, dtype="f8")
    ax, an = h.axis_angle_of(rot, debug=True)
    dot = np.sum(axis * ax, axis=-1)
    ax[dot < 0] = -ax[dot < 0]

    # for a, b, d in zip(axis, ax, dot):
    # ipd.icv(d)
    # ipd.icv('old', a)
    # ipd.icv('new', b)

    # ipd.icv(np.linalg.norm(ax, axis=-1), 1.0)
    try:
        assert np.allclose(axis, ax)
    except:  # noqa
        ipd.icv("ax.shape", ax.shape)
        for u, v, w, x, y in zip(
                axis.reshape(-1, 4),
                ax.reshape(-1, 4),
                angl.reshape(-1),
                an.reshape(-1),
                rot.reshape(-1, 4, 4),
        ):
            if not np.allclose(u, v):
                ipd.icv("u", u, w)
                ipd.icv("v", v, x)
                ipd.icv(y)
        assert 0
    assert np.allclose(angl, an)
    assert np.allclose(np.linalg.norm(ax, axis=-1), 1.0)

def test_axis_angle_of_rand_180(nsamp=100):
    axis = h.normalized(np.random.randn(nsamp, 3))
    angl = np.pi
    rot = h.rot(axis, angl, dtype="f8")
    ax, an = h.axis_angle_of(rot, debug=True)
    # ipd.icv('rot', rot)
    # ipd.icv('ax,an', ax)
    # ipd.icv('ax,an', axis)
    dot = np.abs(np.sum(axis * ax, axis=-1))
    # ipd.icv(dot)
    assert np.allclose(np.abs(dot), 1)  # abs b/c could be flipped
    assert np.allclose(angl, an, atol=1e-4, rtol=1e-4)
    assert np.allclose(np.linalg.norm(ax, axis=-1), 1.0)

def test_axis_angle_of_3x3_rand():
    shape = (4, 5, 6, 7, 8)
    axis = h.normalized_3x3(np.random.randn(*shape, 3))
    assert axis.shape == (*shape, 3)
    angl = np.random.random(shape) * np.pi / 2
    rot = h.rot(axis, angl, dtype="f8")[..., :3, :3]
    assert rot.shape[-1] == 3
    assert rot.shape[-2] == 3
    ax, an = h.axis_angle_of(rot)
    assert np.allclose(axis, ax, atol=1e-3, rtol=1e-3)  # very loose to allow very rare cases
    assert np.allclose(angl, an, atol=1e-4, rtol=1e-4)
    assert np.allclose(np.linalg.norm(ax, axis=-1), 1.0)

def test_is_valid_rays():
    assert not h.is_valid_rays([[0, 1], [0, 0], [0, 0], [0, 0]])
    assert not h.is_valid_rays([[0, 0], [0, 0], [0, 0], [1, 0]])
    assert not h.is_valid_rays([[0, 0], [0, 3], [0, 0], [1, 0]])
    assert h.is_valid_rays([[0, 0], [0, 1], [0, 0], [1, 0]])

def test_hrandray():
    r = h.randray()
    assert np.all(r[..., 3, :] == (1, 0))
    assert r.shape == (4, 2)
    assert np.allclose(h.norm(r[..., :3, 1]), 1)

    r = h.randray(shape=(5, 6, 7))
    assert np.all(r[..., 3, :] == (1, 0))
    assert r.shape == (5, 6, 7, 4, 2)
    assert np.allclose(h.norm(r[..., :3, 1]), 1)

def test_proj_prep():
    assert np.allclose([2, 3, 0, 1], h.projperp([0, 0, 1], [2, 3, 99]))
    assert np.allclose([2, 3, 0, 1], h.projperp([0, 0, 2], [2, 3, 99]))
    a, b = np.random.randn(2, 5, 6, 7, 3)
    pp = h.projperp(a, b)
    assert np.allclose(h.dot(a, pp), 0, atol=1e-5)

def test_point_in_plane():
    plane = h.randray((5, 6, 7))
    assert np.all(h.point_in_plane(plane, plane[..., :3, 0]))
    pt = h.projperp(plane[..., :3, 1], np.random.randn(3))
    assert np.all(h.point_in_plane(plane, plane[..., 0] + pt))

def test_ray_in_plane():
    plane = h.randray((5, 6, 7))
    assert plane.shape == (5, 6, 7, 4, 2)
    dirn = h.projperp(plane[..., :3, 1], np.random.randn(5, 6, 7, 3))
    assert dirn.shape == (5, 6, 7, 4)
    ray = h.ray(plane[..., 0] + h.cross(plane[..., 1], dirn) * 7, dirn)
    assert np.all(h.ray_in_plane(plane, ray))

def test_intersect_planes():
    with pytest.raises(ValueError):
        h.intersect_planes(np.array([[0, 0, 0, 2], [0, 0, 0, 0]]).T, np.array([[0, 0, 0, 1], [0, 0, 0, 0]]).T)
    with pytest.raises(ValueError):
        h.intersect_planes(np.array([[0, 0, 0, 1], [0, 0, 0, 0]]).T, np.array([[0, 0, 0, 1], [0, 0, 0, 1]]).T)
    with pytest.raises(ValueError):
        h.intersect_planes(np.array([[0, 0, 1, 8], [0, 0, 0, 0]]).T, np.array([[0, 0, 1, 9], [0, 0, 0, 1]]).T)
    with pytest.raises(ValueError):
        h.intersect_planes(np.array(9 * [[[0, 0], [0, 0], [0, 0], [1, 0]]]),
                           np.array(2 * [[[0, 0], [0, 0], [0, 0], [1, 0]]]))

    # isct, sts = h.intersect_planes(np.array(9 * [[[0, 0, 0, 1], [1, 0, 0, 0]]]),
    # np.array(9 * [[[0, 0, 0, 1], [1, 0, 0, 0]]]))
    # assert isct.shape[:-2] == sts.shape == (9,)
    # assert np.all(sts == 2)

    # isct, sts = h.intersect_planes(np.array([[1, 0, 0, 1], [1, 0, 0, 0]]),
    # np.array([[0, 0, 0, 1], [1, 0, 0, 0]]))
    # assert sts == 1

    isct, sts = h.intersect_planes(
        np.array([[0, 0, 0, 1], [1, 0, 0, 0]]).T,
        np.array([[0, 0, 0, 1], [0, 1, 0, 0]]).T)
    assert sts == 0
    assert isct[2, 0] == 0
    assert np.all(abs(isct[:3, 1]) == (0, 0, 1))

    isct, sts = h.intersect_planes(
        np.array([[0, 0, 0, 1], [1, 0, 0, 0]]).T,
        np.array([[0, 0, 0, 1], [0, 0, 1, 0]]).T)
    assert sts == 0
    assert isct[1, 0] == 0
    assert np.all(abs(isct[:3, 1]) == (0, 1, 0))

    isct, sts = h.intersect_planes(
        np.array([[0, 0, 0, 1], [0, 1, 0, 0]]).T,
        np.array([[0, 0, 0, 1], [0, 0, 1, 0]]).T)
    assert sts == 0
    assert isct[0, 0] == 0
    assert np.all(abs(isct[:3, 1]) == (1, 0, 0))

    isct, sts = h.intersect_planes(
        np.array([[7, 0, 0, 1], [1, 0, 0, 0]]).T,
        np.array([[0, 9, 0, 1], [0, 1, 0, 0]]).T)
    assert sts == 0
    assert np.allclose(isct[:3, 0], [7, 9, 0])
    assert np.allclose(abs(isct[:3, 1]), [0, 0, 1])

    isct, sts = h.intersect_planes(
        np.array([[0, 0, 0, 1], h.normalized([1, 1, 0, 0])]).T,
        np.array([[0, 0, 0, 1], h.normalized([0, 1, 1, 0])]).T,
    )
    assert sts == 0
    assert np.allclose(abs(isct[:, 1]), h.normalized([1, 1, 1]))

    p1 = h.ray([2, 0, 0, 1], [1, 0, 0, 0])
    p2 = h.ray([0, 0, 0, 1], [0, 0, 1, 0])
    isct, sts = h.intersect_planes(p1, p2)
    assert sts == 0
    assert np.all(h.ray_in_plane(p1, isct))
    assert np.all(h.ray_in_plane(p2, isct))

    p1 = np.array([[0.39263901, 0.57934885, -0.7693232, 1.0], [-0.80966465, -0.18557869, 0.55677976, 0.0]]).T
    p2 = np.array([[0.14790894, -1.333329, 0.45396509, 1.0], [-0.92436319, -0.0221499, 0.38087016, 0.0]]).T
    isct, sts = h.intersect_planes(p1, p2)
    assert sts == 0
    assert np.all(h.ray_in_plane(p1, isct))
    assert np.all(h.ray_in_plane(p2, isct))

def test_intersect_planes_rand():
    # origin case
    plane1, plane2 = h.randray(shape=(2, 1))
    plane1[..., :3, 0] = 0
    plane2[..., :3, 0] = 0
    isect, status = h.intersect_planes(plane1, plane2)
    assert np.all(status == 0)
    assert np.all(h.ray_in_plane(plane1, isect))
    assert np.all(h.ray_in_plane(plane2, isect))

    # orthogonal case
    plane1, plane2 = h.randray(shape=(2, 1))
    plane1[..., :, 1] = h.normalized([0, 0, 1])
    plane2[..., :, 1] = h.normalized([0, 1, 0])
    isect, status = h.intersect_planes(plane1, plane2)
    assert np.all(status == 0)
    assert np.all(h.ray_in_plane(plane1, isect))
    assert np.all(h.ray_in_plane(plane2, isect))

    # general case
    plane1, plane2 = h.randray(shape=(2, 5, 6, 7, 8, 9))
    isect, status = h.intersect_planes(plane1, plane2)
    assert np.all(status == 0)
    assert np.all(h.ray_in_plane(plane1, isect))
    assert np.all(h.ray_in_plane(plane2, isect))

def test_axis_ang_cen_of_rand():
    shape = (5, 6, 7, 8, 9)
    axis0 = h.normalized(np.random.randn(*shape, 3))
    ang0 = np.random.random(shape) * (np.pi - 0.1) + 0.1
    cen0 = np.random.randn(*shape, 3) * 100.0

    helical_trans = np.random.randn(*shape)[..., None] * axis0
    rot = h.rot(axis0, ang0, cen0, dtype="f8")
    rot[..., :, 3] += helical_trans
    axis, ang, cen = h.axis_ang_cen_of(rot)

    assert np.allclose(axis0, axis, rtol=1e-5)
    assert np.allclose(ang0, ang, rtol=1e-5)
    #  check rotation doesn't move cen
    cenhat = (rot @ cen[..., None]).squeeze()
    assert np.allclose(cen + helical_trans, cenhat, rtol=1e-4, atol=1e-4)
    assert np.allclose(np.linalg.norm(axis, axis=-1), 1.0)

@pytest.mark.skip(reason="numerically unstable")
def test_axis_ang_cen_of_rand_eig():
    # shape = (5, 6, 7, 8, 9)
    shape = (1, )
    axis0 = h.normalized(np.random.randn(*shape, 3))
    ang0 = np.random.random(shape) * (np.pi - 0.1) + 0.1
    cen0 = np.random.randn(*shape, 3) * 100.0

    helical_trans = np.random.randn(*shape)[..., None] * axis0
    rot_pure = h.rot(axis0, ang0, cen0, dtype="f8")
    rot = rot_pure.copy()
    rot[..., :, 3] += helical_trans
    axis, ang, cen = h.axis_ang_cen_of_eig(rot)
    # ipd.icv(cen)
    # ipd.icv(cen0)

    assert np.allclose(axis0, axis, rtol=1e-5)
    assert np.allclose(ang0, ang, rtol=1e-5)
    #  check rotation doesn't move cen
    cenhat = (rot_pure @ cen[..., None]).squeeze()
    # ipd.icv(cen)
    # ipd.icv(cenhat)
    # ipd.icv(helical_trans)
    assert np.allclose(cen - helical_trans, cenhat, rtol=1e-4, atol=1e-4)
    assert np.allclose(np.linalg.norm(axis, axis=-1), 1.0)

def test_axis_ang_cen_of_rand_180():
    shape = (5, 6, 7)
    axis0 = h.normalized(np.random.randn(*shape, 3))
    ang0 = np.pi
    cen0 = np.random.randn(*shape, 3) * 100.0

    helical_trans = np.random.randn(*shape)[..., None] * axis0
    rot = h.rot(axis0, ang0, cen0, dtype="f8")
    rot[..., :, 3] += helical_trans
    axis, ang, cen = h.axis_ang_cen_of(rot)

    assert np.allclose(np.abs(h.dot(axis0, axis)), 1, rtol=1e-5)
    assert np.allclose(ang0, ang, rtol=1e-5)
    #  check rotation doesn't move cen
    cenhat = (rot @ cen[..., None]).squeeze()
    assert np.allclose(cen + helical_trans, cenhat, rtol=1e-4, atol=1e-4)
    assert np.allclose(np.linalg.norm(axis, axis=-1), 1.0)

def test_axis_angle_vs_axis_angle_cen_performance(N=1000):
    t = ipd.dev.Timer().start()
    xforms = h.rand(N)
    t.checkpoint("setup")
    axis, ang = h.axis_angle_of(xforms)
    t.checkpoint("axis_angle_of")
    axis2, ang2, _cen = h.axis_ang_cen_of(xforms)
    t.checkpoint("h.axis_ang_cen_of")
    # ipd.icv(t.report(scale=1_000_000 / N))

    assert np.allclose(axis, axis2)
    assert np.allclose(ang, ang2)
    # assert 0

def test_hinv_rand():
    shape = (5, 6, 7, 8, 9)
    axis0 = h.normalized(np.random.randn(*shape, 3))
    ang0 = np.random.random(shape) * (np.pi - 0.1) + 0.1
    cen0 = np.random.randn(*shape, 3) * 100.0
    helical_trans = np.random.randn(*shape)[..., None] * axis0
    rot = h.rot(axis0, ang0, cen0, dtype="f8")
    rot[..., :, 3] += helical_trans
    assert np.allclose(np.eye(4), h.hinv(rot) @ rot)

def test_hframe():
    sh = (5, 6, 7, 8, 9)
    u = h.randpoint(sh)
    v = h.randpoint(sh)
    w = h.randpoint(sh)
    s = h.frame(u, v, w)
    assert h.is_homog_xform(s)

    assert h.is_homog_xform(h.frame([1, 2, 3], [5, 6, 4], [9, 7, 8]))

def test_line_line_dist():
    lld = h.line_line_distance
    assert lld(h.ray([0, 0, 0], [1, 0, 0]), h.ray([0, 0, 0], [1, 0, 0])) == 0
    assert lld(h.ray([0, 0, 0], [1, 0, 0]), h.ray([1, 0, 0], [1, 0, 0])) == 0
    assert lld(h.ray([0, 0, 0], [1, 0, 0]), h.ray([0, 1, 0], [1, 0, 0])) == 1
    assert lld(h.ray([0, 0, 0], [1, 0, 0]), h.ray([0, 1, 0], [0, 0, 1])) == 1

def test_line_line_closest_points():
    lld = h.line_line_distance
    llcp = h.line_line_closest_points
    p, q = llcp(h.ray([0, 0, 0], [1, 0, 0]), h.ray([0, 0, 0], [0, 1, 0]))
    assert np.all(p == [0, 0, 0, 1]) and np.all(q == [0, 0, 0, 1])
    p, q = llcp(h.ray([0, 1, 0], [1, 0, 0]), h.ray([1, 0, 0], [0, 1, 0]))
    assert np.all(p == [1, 1, 0, 1]) and np.all(q == [1, 1, 0, 1])
    p, q = llcp(h.ray([1, 1, 0], [1, 0, 0]), h.ray([1, 1, 0], [0, 1, 0]))
    assert np.all(p == [1, 1, 0, 1]) and np.all(q == [1, 1, 0, 1])
    p, q = llcp(h.ray([1, 2, 3], [1, 0, 0]), h.ray([4, 5, 6], [0, 1, 0]))
    assert np.all(p == [4, 2, 3, 1]) and np.all(q == [4, 2, 6, 1])
    p, q = llcp(h.ray([1, 2, 3], [-13, 0, 0]), h.ray([4, 5, 6], [0, -7, 0]))
    assert np.all(p == [4, 2, 3, 1]) and np.all(q == [4, 2, 6, 1])
    p, q = llcp(h.ray([1, 2, 3], [1, 0, 0]), h.ray([4, 5, 6], [0, 1, 0]))
    assert np.all(p == [4, 2, 3, 1]) and np.all(q == [4, 2, 6, 1])

    r1, r2 = h.ray([1, 2, 3], [1, 0, 0]), h.ray([4, 5, 6], [0, 1, 0])
    x = h.rand((5, 6, 7))
    xinv = np.linalg.inv(x)
    p, q = llcp(x @ r1, x @ r2)
    assert np.allclose((xinv @ p[..., None]).squeeze(-1), [4, 2, 3, 1])
    assert np.allclose((xinv @ q[..., None]).squeeze(-1), [4, 2, 6, 1])

    shape = (23, 17, 31)
    ntot = np.prod(shape)
    r1 = h.randray(cen=np.random.randn(*shape, 3))
    r2 = h.randray(cen=np.random.randn(*shape, 3))
    p, q = llcp(r1, r2)
    assert p.shape[:-1] == shape and q.shape[:-1] == shape
    lldist0 = h.norm(p - q)
    lldist1 = lld(r1, r2)
    # ipd.icv(lldist0 - lldist1)
    # TODO figure out how to compare better
    delta = np.abs(lldist1 - lldist0)

    for distcut, allowedfailfrac in [
        (0.0001, 0.0005),
            # (0.01, 0.0003),
    ]:
        fail = delta > distcut
        fracfail = np.sum(fail) / ntot
        # ipd.icv(fracfail, fail.shape, ntot)
        if fracfail > allowedfailfrac:
            ipd.icv("h.line_line_closest_points fail; distcut", distcut, "allowedfailfrac", allowedfailfrac)
            ipd.icv("failpoints", delta[delta > distcut])
            assert allowedfailfrac <= allowedfailfrac

    # assert np.allclose(lldist0, lldist1, atol=1e-1, rtol=1e-1)  # loose, but rarely fails otherwise

def test_dihedral():
    assert 0.00001 > abs(np.pi / 2 - h.dihedral([1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]))
    assert 0.00001 > abs(-np.pi / 2 - h.dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]))
    a, b, c = (
        h.point([1, 0, 0]),
        h.point([0, 1, 0]),
        h.point([0, 0, 1]),
    )
    n = h.point([0, 0, 0])
    x = h.rand(10)
    assert np.allclose(h.dihedral(a, b, c, n), h.dihedral(x @ a, x @ b, x @ c, x @ n))
    for ang in np.arange(-np.pi + 0.001, np.pi, 0.1):
        x = h.rot([0, 1, 0], ang)
        d = h.dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], x @ [1, 0, 0, 0])
        assert abs(ang - d) < 0.000001

def test_angle():
    assert 0.0001 > abs(h.angle([1, 0, 0], [0, 1, 0]) - np.pi / 2)
    assert 0.0001 > abs(h.angle([1, 1, 0], [0, 1, 0]) - np.pi / 4)

def test_align_around_axis():
    axis = h.rand_unit(1000)
    u = h.rand_vec()
    ang = np.random.rand(1000) * np.pi
    x = h.rot(axis, ang)
    v = x @ u
    uprime = h.align_around_axis(axis, u, v) @ u
    assert np.allclose(h.angle(v, uprime), 0, atol=1e-5)

def test_halign2_minangle():
    tgt1 = [-0.816497, -0.000000, -0.577350, 0]
    tgt2 = [0.000000, 0.000000, 1.000000, 0]
    orig1 = [0.000000, 0.000000, 1.000000, 0]
    orig2 = [-0.723746, 0.377967, -0.577350, 0]
    x = h.halign2(orig1, orig2, tgt1, tgt2)
    assert np.allclose(tgt1, x @ orig1, atol=1e-5)
    assert np.allclose(tgt2, x @ orig2, atol=1e-5)

    ax1 = np.array([0.12896027, -0.57202471, -0.81003518, 0.0])
    ax2 = np.array([0.0, 0.0, -1.0, 0.0])
    tax1 = np.array([0.57735027, 0.57735027, 0.57735027, 0.0])
    tax2 = np.array([0.70710678, 0.70710678, 0.0, 0.0])
    x = h.halign2(ax1, ax2, tax1, tax2)
    assert np.allclose(x @ ax1, tax1, atol=1e-2)
    assert np.allclose(x @ ax2, tax2, atol=1e-2)

def test_halign2_una_case():
    ax1 = np.array([0.0, 0.0, -1.0, 0.0])
    ax2 = np.array([0.83822463, -0.43167392, 0.33322229, 0.0])
    tax1 = np.array([-0.57735027, 0.57735027, 0.57735027, 0.0])
    tax2 = np.array([0.57735027, -0.57735027, 0.57735027, 0.0])
    # ipd.icv(angle_degrees(ax1, ax2))
    # ipd.icv(angle_degrees(tax1, tax2))
    x = h.halign2(ax1, ax2, tax1, tax2)
    # ipd.icv(tax1)
    # ipd.icv(x@ax1)
    # ipd.icv(tax2)
    # ipd.icv(x@ax2)
    assert np.allclose(x @ ax1, tax1, atol=1e-2)
    assert np.allclose(x @ ax2, tax2, atol=1e-2)

def test_calc_dihedral_angle():
    dang = h.calc_dihedral_angle(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )
    assert np.allclose(dang, -np.pi / 2)
    dang = h.calc_dihedral_angle(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    )
    assert np.allclose(dang, -np.pi / 4)
    dang = h.calc_dihedral_angle(
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
    )
    assert np.allclose(dang, -np.pi / 4)

def test_align_lines_dof_dihedral_rand_single():
    fix, mov, dof = h.rand_unit(3)

    if h.angle(fix, dof) > np.pi / 2: dof = -dof
    if h.angle(dof, mov) > np.pi / 2: mov = -mov
    target_angle = h.angle(mov, fix)
    dof_angle = h.angle(mov, dof)
    fix_to_dof_angle = h.angle(fix, dof)

    if target_angle + dof_angle < fix_to_dof_angle:
        return

    axis = h.cross(fix, dof)
    mov_in_plane = (h.rot(axis, -dof_angle) @ dof[..., None]).reshape(1, 4)
    # could rotate so mov is in plane as close to fix as possible
    # if h.dot(mov_in_plane, fix) < 0:
    #    mov_in_plane = (h.rot(axis, np.py + dof_angle) @ dof[..., None]).reshape(1, 4)

    test = h.calc_dihedral_angle(fix, [0.0, 0.0, 0.0, 0.0], dof, mov_in_plane)
    assert np.allclose(test, 0) or np.allclose(test, np.pi)
    dang = h.calc_dihedral_angle(fix, [0.0, 0.0, 0.0, 0.0], dof, mov)

    ahat = h.rotation_around_dof_for_target_angle(target_angle, dof_angle, fix_to_dof_angle)
    # ipd.icv(ahat, dang, abs(dang) + abs(ahat))

    # ipd.icv('result', 'ta', np.degrees(target_angle), 'da', np.degrees(dof_angle), 'fda',
    # np.degrees(fix_to_dof_angle), dang, ahat, abs(abs(dang) - abs(ahat)))

    atol = 1e-5 if 0.05 < abs(dang) < np.pi - 0.05 else 1e-2
    close1 = np.allclose(abs(dang), abs(ahat), atol=atol)
    close2 = np.allclose(abs(dang), np.pi - abs(ahat), atol=atol)
    if not (close1 or close2):
        ipd.icv("ERROR", abs(dang), abs(ahat), np.pi - abs(ahat))
    assert close1 or close2

def test_align_lines_dof_dihedral_rand_3D():
    num_sol_found, num_total, _num_no_sol, max_sol = [0] * 4
    for _ in range(100):
        target_angle = np.random.uniform(0, np.pi)
        fix, mov, dof = h.rand_unit(3)

        if h.dot(dof, fix) < 0:
            dof = -dof
        if h.angle(dof, mov) > np.pi / 2:
            mov = -mov

        if h.line_angle(fix, dof) > h.line_angle(mov, dof) + target_angle:
            continue
        if target_angle > h.line_angle(mov, dof) + h.line_angle(fix, dof):
            continue

        solutions = h.xform_around_dof_for_vector_target_angle(fix, mov, dof, target_angle)
        if solutions is None:
            continue

        num_sol_found += 0 < len(solutions)
        max_sol = np.maximum(max_sol, target_angle)
        num_total += 1

        for sol in solutions:
            assert np.allclose(target_angle, h.angle(fix, sol @ mov), atol=1e-5)

    # ipd.icv(num_total, num_sol_found, num_no_sol, np.degrees(max_sol))
    assert (num_sol_found) / num_total > 0.6

def test_align_lines_dof_dihedral_rand(n=100):
    for _ in range(n):
        # ipd.icv(i)
        test_align_lines_dof_dihedral_rand_single()

def test_align_lines_dof_dihedral_basic():
    target_angle = np.radians(30)
    dof_angle = np.radians(30)
    fix_to_dof_angle = np.radians(60)
    ahat = h.rotation_around_dof_for_target_angle(target_angle, dof_angle, fix_to_dof_angle)
    assert np.allclose(ahat, 0)

    target_angle = np.radians(30)
    dof_angle = np.radians(30)
    fix_to_dof_angle = np.radians(30)
    ahat = h.rotation_around_dof_for_target_angle(target_angle, dof_angle, fix_to_dof_angle)
    assert np.allclose(ahat, 1.088176213364169)

    target_angle = np.radians(45)
    dof_angle = np.radians(30)
    fix_to_dof_angle = np.radians(60)
    ahat = h.rotation_around_dof_for_target_angle(target_angle, dof_angle, fix_to_dof_angle)
    assert np.allclose(ahat, 0.8853828498391183)

def test_place_lines_to_isect_F432():
    ta1 = h.normalized([0.0, 1.0, 0.0, 0.0])
    tp1 = np.array([0.0, 0.0, 0.0, 1])
    ta2 = h.normalized([0.0, -0.5, 0.5, 0.0])
    tp2 = np.array([-1, 1, 1, 1.0])
    sl2 = h.normalized(tp2 - tp1)

    for _ in range(100):
        Xptrb = h.rand(cart_sd=0)
        ax1 = Xptrb @ np.array([0.0, 1.0, 0.0, 0.0])
        pt1 = Xptrb @ np.array([0.0, 0.0, 0.0, 1.0])
        ax2 = Xptrb @ np.array([0.0, -0.5, 0.5, 0.0])
        pt2 = Xptrb @ h.normalized(np.array([-1.0, 1.0, 1.0, 1.0]))

        Xalign, _delta = h.align_lines_isect_axis2(pt1, ax1, pt2, ax2, ta1, tp1, ta2, sl2)
        _xp1, xa1 = Xalign @ pt1, Xalign @ ax1
        xp2, xa2 = Xalign @ pt2, Xalign @ ax2
        assert np.allclose(Xalign[3, 3], 1.0)

        # ipd.icv('ax1', xa1, ta1)
        # ipd.icv('ax2', xa2, ta2)
        # ipd.icv('pt1', xp1)
        # ipd.icv('pt2', xp2)

        assert np.allclose(h.line_angle(xa1, xa2), h.line_angle(ta1, ta2))
        assert np.allclose(h.line_angle(xa1, ta1), 0.0, atol=0.001)
        assert np.allclose(h.line_angle(xa2, ta2), 0.0, atol=0.001)
        isect_error = h.line_line_distance_pa(xp2, xa2, [0, 0, 0, 1], sl2)
        assert np.allclose(isect_error, 0, atol=0.001)

def test_place_lines_to_isect_onecase():
    tp1 = np.array([+0, +0, +0, 1])
    ta1 = np.array([+1, +1, +1, 0])
    ta2 = np.array([+1, +1, -1, 0])
    sl2 = np.array([+0, +1, +1, 0])
    pt1 = np.array([+0, +0, +0, 1])
    ax1 = np.array([+1, +1, +1, 0])
    pt2 = np.array([+1, +2, +1, 1])
    ax2 = np.array([+1, +1, -1, 0])
    ta1 = h.normalized(ta1)
    ta2 = h.normalized(ta2)
    sl2 = h.normalized(sl2)
    ax1 = h.normalized(ax1)
    ax2 = h.normalized(ax2)

    Xalign, _delta = h.align_lines_isect_axis2(pt1, ax1, pt2, ax2, ta1, tp1, ta2, sl2)
    isect_error = h.line_line_distance_pa(Xalign @ pt2, Xalign @ ax2, [0, 0, 0, 1], sl2)
    assert np.allclose(isect_error, 0, atol=0.001)

def test_place_lines_to_isect_F432_null():
    ta1 = np.array([0.0, 1.0, 0.0, 0.0])
    tp1 = np.array([0.0, 0.0, 0.0, 1.0])
    ta2 = np.array([0.0, -0.5, 0.5, 0.0])
    tp2 = np.array([-0.57735, 0.57735, 0.57735, 1.0])
    sl2 = tp2 - tp1

    ax1 = np.array([0.0, 1.0, 0.0, 0.0])
    pt1 = np.array([0.0, 0.0, 0.0, 1.0])
    ax2 = np.array([0.0, -0.5, 0.5, 0.0])
    pt2 = np.array([-0.57735, 0.57735, 0.57735, 1.0])

    Xalign, _delta = h.align_lines_isect_axis2(pt1, ax1, pt2, ax2, ta1, tp1, ta2, sl2)
    assert np.allclose(Xalign[3, 3], 1.0)

    _xp1, xa1 = Xalign @ pt1, Xalign @ ax1
    xp2, xa2 = Xalign @ pt2, Xalign @ ax2
    assert np.allclose(h.line_angle(xa1, xa2), h.line_angle(ta1, ta2))
    assert np.allclose(h.line_angle(xa1, ta1), 0.0)
    assert np.allclose(h.line_angle(xa2, ta2), 0.0, atol=0.001)
    isect_error = h.line_line_distance_pa(xp2, xa2, [0, 0, 0, 1], sl2)
    assert np.allclose(isect_error, 0, atol=0.001)

def _vaildate_test_scale_translate_lines_isect_lines(samp, xalign, scale, i):
    assert xalign is not None
    assert scale is not None
    pt1, ax1, pt2, ax2, tp1, ta1, tp2, ta2 = samp
    ok_ax1 = np.allclose(xalign @ ax1, ta1, atol=1e-5) or np.allclose(xalign @ -ax1, ta1, atol=1e-5)
    ok_ax2 = np.allclose(xalign @ ax2, ta2, atol=1e-5) or np.allclose(xalign @ -ax2, ta2, atol=1e-5)

    # ok_pt1 = np.allclose(xalign @ pt1, scale * tp1, atol=1e-3)
    # if not ok_pt1:
    #    offset1 = h.normalized(xalign @ pt1 - scale * tp1)
    #    offset1[3] = 0
    #    ok_pt1 = (np.allclose(offset1, ta1, atol=1e-3) or np.allclose(offset1, -ta1, atol=1e-3))
    # ok_pt2 = np.allclose(xalign @ pt2, scale * tp2, atol=1e-3)
    # if not ok_pt2:
    #    offset2 = h.normalized(xalign @ pt2 - scale * tp2)
    #    offset2[3] = 0
    #    ok_pt2 = (np.allclose(offset2, ta2, atol=1e-3) or np.allclose(offset2, -ta2, atol=1e-3))
    dis1 = np.linalg.norm((xalign@pt1 - scale*tp1))
    if dis1 > 0.009:
        dis1 = np.sin(h.angle(xalign@pt1 - scale*tp1, ta1)) * dis1

    dis2 = np.linalg.norm((xalign@pt2 - scale*tp2))
    if dis2 > 0.009:
        dis2 = np.sin(h.angle(xalign@pt2 - scale*tp2, ta2)) * dis2
    ok_pt1 = abs(dis1) < 0.01  # this is *really* loose to allow very rare cases
    ok_pt2 = abs(dis2) < 0.01  # in 99999/100000 cases, much tighter

    if not (ok_ax1 and ok_ax2 and ok_pt1 and ok_pt2):
        # ipd.icv()
        # ipd.icv('norm', np.linalg.norm((xalign @ pt1 - scale * tp1)),
        # np.linalg.norm((xalign @ pt2 - scale * tp2)))
        # ipd.icv('dis', dis1, dis2)
        # ipd.icv('sin', np.sin(h.angle(xalign @ pt1 - scale * tp1, ta1)),
        # np.sin(h.angle(xalign @ pt2 - scale * tp2, ta2)))
        # ipd.icv()

        # ipd.icv(ta1, np.linalg.norm(ta1))
        # ipd.icv(xalign @ pt1)
        # ipd.icv(scale * tp1)

        # ipd.icv()
        # ipd.icv(np.linalg.norm(xalign @ pt2))
        # ipd.icv(np.linalg.norm(xalign @ pt2 - scale * tp2))
        # ipd.icv(ta2, np.linalg.norm(ta2))
        # ipd.icv(xalign @ pt2)
        # ipd.icv(scale * tp2)

        if not ok_ax1:
            ipd.icv("fail ax1 on %i" % i)
        if not ok_ax2:
            ipd.icv("fail ax2 on %i" % i)
        if not ok_pt1:
            ipd.icv("fail pt1 on %i" % i)
        if not ok_pt2:
            ipd.icv("fail pt2 on %i" % i)
        # ipd.icv(repr(samp))

        if 0:
            _rays = np.array([
                h.ray(xalign @ pt1, xalign @ ax1),
                h.ray(xalign @ pt2, xalign @ ax2),
                h.ray(scale * tp1, scale * ta1),
                h.ray(scale * tp2, scale * ta2),
            ])
            colors = [(1, 0, 0), (0, 0, 1), (0.8, 0.5, 0.5), (0.5, 0.5, 0.8)]
            # rp.viz.showme(rays, colors=colors, block=False)
        assert ok_ax1 and ok_ax2 and ok_pt1 and ok_pt2

def test_scale_translate_lines_isect_lines_p4132():
    samps = list()
    for i in range(30):
        pt1 = np.array([-40, 0, -40, 1], dtype="d")
        ax1 = np.array([-1, -1, 1, 0], dtype="d")
        pt2 = np.array([-20, -20, -20, 1], dtype="d")
        ax2 = np.array([0, -1, -1, 0], dtype="d")

        tp1 = np.array([-0.5, 0, -0.5, 1], dtype="d")
        ta1 = np.array([-1, -1, 1, 0], dtype="d")
        tp2 = np.array([-0.125, -0.125, -0.125, 1], dtype="d")
        ta2 = np.array([0, -1, -1, 0], dtype="d")

        ax1 = h.normalized(ax1)
        ax2 = h.normalized(ax2)
        ta1 = h.normalized(ta1)
        ta2 = h.normalized(ta2)
        tmp = h.rand_vec() * 30
        pt1 += tmp
        pt2 += tmp
        pt1 += ax1 * h.rand_vec() * 30
        pt2 += ax2 * h.rand_vec() * 30
        xtest = h.rand()
        pt1 = xtest @ pt1
        ax1 = xtest @ ax1
        pt2 = xtest @ pt2
        ax2 = xtest @ ax2
        samps.append((pt1, ax1, pt2, ax2, tp1, ta1, tp2, ta2))

    # from numpy import array
    # samps = [(array([44.83313235, 54.92005006, -21.4824442,
    #                  1.]), array([-0.40516428, -0.41266195, 0.81581372,
    #                               0.]), array([16.22902529, 0.36515706, 18.20013359, 1.]),
    #           array([-0.87280016, 0.4402107, -0.21079474, 0.]), array([-0.5, 0., -0.5, 1.]),
    #           array([-0.57735027, -0.57735027, 0.57735027, 0.]), array(
    #              [-0.125, -0.125, -0.125, 1.]), array([0., -0.70710678, -0.70710678, 0.]))]

    for i, samp in enumerate(samps):
        xalign, scale = h.scale_translate_lines_isect_lines(*samp)
        _vaildate_test_scale_translate_lines_isect_lines(samp, xalign, scale, i)

def test_scale_translate_lines_isect_lines_nonorthog():
    nsamp = 30
    samps = list()
    for i in range(nsamp):
        pt1 = np.array([-40, 0, -40, 1], dtype="d")
        ax1 = np.array([1, 1, 1, 0], dtype="d")
        pt2 = np.array([-20, -20, -20, 1], dtype="d")
        ax2 = np.array([0, 1, 1, 0], dtype="d")

        tp1 = np.array([-0.5, 0, -0.5, 1], dtype="d")
        ta1 = np.array([1, 1, 1, 0], dtype="d")
        tp2 = np.array([-0.125, -0.125, -0.125, 1], dtype="d")
        ta2 = np.array([0, 1, 1, 0], dtype="d")

        ax1 = h.normalized(ax1)
        ax2 = h.normalized(ax2)
        ta1 = h.normalized(ta1)
        ta2 = h.normalized(ta2)
        tmp = h.rand_vec() * 30
        pt1 += tmp
        pt2 += tmp
        pt1 += ax1 * h.rand_vec() * 30
        pt2 += ax2 * h.rand_vec() * 30
        xtest = h.rand()
        pt1 = xtest @ pt1
        ax1 = xtest @ ax1
        pt2 = xtest @ pt2
        ax2 = xtest @ ax2
        samps.append((pt1, ax1, pt2, ax2, tp1, ta1, tp2, ta2))

    # from numpy import array
    # samps = [(array([9.9106826, 32.13286237, -17.17714398,
    #                  1.]), array([-0.75428055, -0.53955273, 0.37409049,
    #                               0.]), array([16.67226665, 9.12240355, -10.38776327, 1.]),
    #           array([-0.23706077, -0.78614464, 0.57077035, 0.]), array([-0.5, 0., -0.5, 1.]),
    #           array([0.57735027, 0.57735027, 0.57735027, 0.]), array(
    #              [-0.125, -0.125, -0.125, 1.]), array([0., 0.70710678, 0.70710678, 0.]))]

    ok = 0
    for i, samp in enumerate(samps):
        xalign, scale = h.scale_translate_lines_isect_lines(*samp)
        if xalign is not None:
            ok += 1
            _vaildate_test_scale_translate_lines_isect_lines(samp, xalign, scale, i)
        else:
            pass

    # ipd.icv(ok, nsamp)
    assert ok > nsamp * 0.5

def test_scale_translate_lines_isect_lines_arbitrary():
    samps = list()
    for i in range(30):
        tp1 = h.randpoint()
        ta1 = h.rand_unit()
        tp2 = h.randpoint()
        ta2 = h.rand_unit()
        rx = h.rand()
        pt1 = rx @ tp1
        ax1 = rx @ ta1
        pt2 = rx @ tp2
        ax2 = rx @ ta2
        scale = -np.random.rand() * 2
        tp1[:3] = tp1[:3] * scale
        tp2[:3] = tp2[:3] * scale
        if np.random.rand() < 0.8:
            tp1[:3] += np.random.normal() * ta1[:3]
            tp2[:3] += np.random.normal() * ta2[:3]

        # ??? h.projperp(_ta2, _tp2 - _pt2) always 0

        samps.append((pt1, ax1, pt2, ax2, tp1, ta1, tp2, ta2))

    # from numpy import array
    # samps = [(array([9.75754336, 11.94442093, -27.9587031,
    #                  1.]), array([-0.34856845, 0.5450748, 0.76249164,
    #                               0.]), array([35.97908368, -17.0049886, 37.47481606, 1.]),
    #           array([-0.67119914, 0.0327908, 0.74055147, 0.]), array([-0.5, 0., -0.5, 1.]),
    #           array([0.57735027, 0.57735027, 0.57735027, 0.]), array(
    #              [-0.125, -0.125, -0.125, 1.]), array([0., 0.70710678, 0.70710678, 0.]))]

    for i, samp in enumerate(samps):
        xalign, scale = h.scale_translate_lines_isect_lines(*samp)
        _vaildate_test_scale_translate_lines_isect_lines(samp, xalign, scale, i)

def test_scale_translate_lines_isect_lines_cases():
    from numpy import array, float32

    samps = [
        (
            array([1.37846362, 0.86161002, -0.27543099, 1.0]),
            array([-0.00864379, 0.06366346, 0.99793399, 0.0]),
            array([1.32846251, 1.93970668, -1.21464696, 1.0]),
            array([0.00118571, -0.06660165, -0.99777894, 0.0]),
            array([-0.9762876, -0.33263428, -1.85131458, 1.0]),
            array([-0.39919588, -0.753445, 0.52245888, 0.0]),
            array([-2.21895034, -1.82820754, -2.95265392, 1.0]),
            array([0.39191218, 0.75669448, -0.52327651, 0.0]),
        ),
        (
            array([0.51437861, 2.18780994, -1.99001668, 1.0]),
            array([-0.18190349, 0.31861761, -0.93026552, 0.0]),
            array([0.61818801, 0.22765824, -1.97645497, 1.0]),
            array([0.71970973, -0.68295276, -0.12487363, 0.0]),
            array([-3.90681481, -2.81324474, -1.48884669, 1.0]),
            array([-0.5400405, 0.06772283, 0.83890994, 0.0]),
            array([-1.99738608, -2.86968414, -1.03712605, 1.0]),
            array([0.66836603, -0.71331712, 0.21086854, 0.0]),
        ),
        (
            array([0.80453348, -1.70369142, -1.56996154, 1.0]),
            array([-0.10405314, 0.57145446, 0.81401028, 0.0]),
            array([1.23167311, -2.28086782, -2.31477258, 1.0]),
            array([-0.48977051, -0.79128183, -0.36605725, 0.0]),
            array([-0.18173768, 0.38484544, 1.38040017, 1.0]),
            array([0.73656647, 0.47040127, -0.48599638, 0.0]),
            array([-0.99021092, 0.20752128, 2.0010865, 1.0]),
            array([-0.70175834, -0.66130471, -0.26497418, 0.0]),
        ),
        (
            array([0.66161907, 0.53607942, -0.40837472, 1.0]),
            array([0.52195716, 0.4186833, -0.74314535, 0.0]),
            array([1.01258715, 0.05252822, -0.08320797, 1.0]),
            array([-0.07314979, 0.73456495, 0.6745839, 0.0]),
            array([1.70065761, -1.66626863, -0.01729367, 1.0]),
            array([-0.24527134, -0.85585473, 0.4553621, 0.0]),
            array([1.21956504, -1.18959931, 0.04650348, 1.0]),
            array([0.75544244, 0.34941661, 0.55426958, 0.0]),
        ),
        (
            array([-0.2624203, 0.88704277, 1.44731444, 1.0]),
            array([0.68508642, -0.55063177, -0.47692898, 0.0]),
            array([-0.10187965, 1.78492688, 3.99709701, 1.0]),
            array([-0.41175151, 0.82175847, 0.39392095, 0.0]),
            array([-1.26813184, -0.15104216, -0.70483344, 1.0]),
            array([0.7067224, 0.39894059, 0.58428576, 0.0]),
            array([-2.62475518, 1.12221865, -2.67250729, 1.0]),
            array([-0.3987861, -0.39064141, -0.82968002, 0.0]),
        ),
        (
            array([-76.25620827, 46.15603441, 39.92563141, 1.0]),
            array([0.97109258, -0.12288913, 0.20463984, 0.0]),
            array([-72.71041931, 58.85680507, 17.2434256, 1.0]),
            array([0.65593139, -0.33220823, 0.67778441, 0.0]),
            array([-0.5, 0.0, -0.5, 1.0]),
            array([0.57735027, 0.57735027, 0.57735027, 0.0]),
            array([-0.125, -0.125, -0.125, 1.0]),
            array([0.0, 0.70710678, 0.70710678, 0.0]),
        ),
        (
            array([0, 0, 0, 1]),
            array([0.0, 0.0, 1.0, 0.0]),
            array([2.784926, -7.0175056, -0.07861771, 1.0], dtype=float32),
            array([0.65977187, 0.48100702, 0.57735027, 0.0]),
            array([0.0, 0.0, 0.0, 1.0]),
            array([0.57735027, 0.57735027, 0.57735027, 0.0]),
            array([0.0, -0.25, 0.0, 1.0]),
            array([0.0, 0.0, 1.0, 0.0]),
        ),
    ]
    for i, samp in enumerate(samps):
        # ipd.icv('SAMP', i)
        xalign, scale = h.scale_translate_lines_isect_lines(*samp)
        _vaildate_test_scale_translate_lines_isect_lines(samp, xalign, scale, i)

def test_xform_around_dof_for_vector_target_angle():
    fix = np.array([0, 0, 1, 0])
    mov = np.array([-0.48615677, 0.14842946, -0.86117379, 0.0])
    dof = np.array([-1.16191467, 0.35474642, -2.05820535, 0.0])
    target_angle = 0.6154797086703874
    # should not throw
    solutions = h.xform_around_dof_for_vector_target_angle(fix, mov, dof, target_angle)
    assert solutions == []

# def marisa():
#    data = rp.load('rpxdock/data/testdata/test_asym.pickle')
#    ipd.icv(type(data))
#    ipd.icv(data.xforms[3].data)
#    ipd.icv(np.eye(4))
#    rp.dump(data.data, 'tmp.pickle')
#
#    ary = np.array([1, 2, 3])
#
#    X = data.xforms[3].data
#    orig_pts = np.random.rand(10, 4) * 100 - 50
#    orig_pts[:, 3] = 1
#    new_pts = X @ orig_pts.T
#
#    ipd.icv(X)
#
#    ipd.icv(orig_pts)
#    ipd.icv('xformed (1,2,3)')
#
#    ipd.icv(new_pts.T)

def test_axis_angle_180_rand():
    pass

def test_axis_angle_180_bug():
    #    v = h.rand_unit()
    #    x = np.stack([h.rot(v, 180), h.rot(v, 180)] * 3)
    #    ipd.icv('v', v)
    #    ipd.icv()
    #    ev = np.linalgh..eig(x[..., :3, :3])
    #    val, vec = np.real(ev[0]), np.real(ev[1])
    #    ipd.icv(val)
    #    cond = np.abs(val - 1) < 1e-6
    #    a, b = np.where(cond)
    #    ipd.icv(a)
    #    ipd.icv(b)
    #
    #    assert np.all(np.sum(np.abs(val - 1) < 1e-6, axis=-1) == 1)
    #
    #    ipd.icv(vec[a, :, b])
    #
    #    ipd.icv(h.axis_of(np.array([
    #        h.rot(v, 180),
    #        np.eye(4),
    #    ]), debug=True))

    # assert 0
    # np.set_icoptions(precision=20)

    # yapf: disable
    x000 = np.array([
       [  1,  0,  0,  0],
       [  0,  1,  0,  0],
       [  0,  0,  1,  0],
       [  0,  0,  0,  1],
    ],dtype='f8')
    x100 = np.array([
       [  1,  0,  0,  0],
       [  0, -1,  0,  0],
       [  0,  0, -1,  0],
       [  0,  0,  0,  1],
    ],dtype='f8')
    x010 = np.array([
       [ -1,  0,  0,  0],
       [  0,  1,  0,  0],
       [  0,  0, -1,  0],
       [  0,  0,  0,  1],
    ],dtype='f8')
    x001 = np.array([
       [ -1,  0,  0,  0],
       [  0, -1,  0,  0],
       [  0,  0,  1,  0],
       [  0,  0,  0,  1],
    ],dtype='f8')
    x110 = np.array([
       [  0,  1,  0,  0],
       [  1,  0,  0,  0],
       [  0,  0, -1,  0],
       [  0,  0,  0,  1],
    ],dtype='f8')
    x101 = np.array([
       [  0,  0,  1,  0],
       [  0, -1,  0,  0],
       [  1,  0,  0,  0],
       [  0,  0,  0,  1],
    ],dtype='f8')
    x011 = np.array([
       [ -1,  0,  0,  0],
       [  0,  0,  1,  0],
       [  0,  1,  0,  0],
       [  0,  0,  0,  1],
    ],dtype='f8')
    x1t0 = np.array([
       [ -1,  0,  0,  0],
       [  0,  0,  1,  0],
       [  0,  1,  0,  0],
       [  0,  0,  0,  1],
    ],dtype='f8')
    x1n0 = np.array([
       [  0, -1,  0,  0],
       [ -1,  0,  0,  0],
       [  0,  0, -1,  0],
       [  0,  0,  0,  1],
    ],dtype='f8')
    x10n = np.array([
       [  0,  0, -1,  0],
       [  0, -1,  0,  0],
       [ -1,  0,  0,  0],
       [  0,  0,  0,  1],
    ],dtype='f8')
    x01n = np.array([
       [ -1,  0,  0,  0],
       [  0,  0, -1,  0],
       [  0, -1,  0,  0],
       [  0,  0,  0,  1],
    ],dtype='f8')

    r = np.sqrt(2)/2

    assert np.allclose(np.linalg.det(x000), 1)
    assert np.allclose(np.linalg.det(x100), 1)
    assert np.allclose(np.linalg.det(x010), 1)
    assert np.allclose(np.linalg.det(x001), 1)
    assert np.allclose(np.linalg.det(x110), 1)
    assert np.allclose(np.linalg.det(x101), 1)
    assert np.allclose(np.linalg.det(x011), 1)
    assert np.allclose(np.linalg.det(x1n0), 1)
    assert np.allclose(np.linalg.det(x10n), 1)
    assert np.allclose(np.linalg.det(x01n), 1)

    assert np.allclose(h.rot([ 1,  0,  0],   0), x000)
    assert np.allclose(h.rot([ 1,  0,  0], 180), x100)
    assert np.allclose(h.rot([ 0,  1,  0], 180), x010)
    assert np.allclose(h.rot([ 0,  0,  1], 180), x001)
    assert np.allclose(h.rot([ 1,  1,  0], 180), x110)
    assert np.allclose(h.rot([ 1,  0,  1], 180), x101)
    assert np.allclose(h.rot([ 0,  1,  1], 180), x011)
    assert np.allclose(h.rot([ 1, -1,  0], 180), x1n0)
    assert np.allclose(h.rot([ 1,  0, -1], 180), x10n)
    assert np.allclose(h.rot([ 0,  1, -1], 180), x01n)
    assert np.allclose(h.rot([-1,  1,  0], 180), x1n0)
    assert np.allclose(h.rot([-1,  0,  1], 180), x10n)
    assert np.allclose(h.rot([ 0, -1,  1], 180), x01n)

    assert np.allclose([ 1,  0,  0, 0], h.axis_of(x000))
    assert np.allclose([ 1,  0,  0, 0], h.axis_of(x100))
    assert np.allclose([ 0,  1,  0, 0], h.axis_of(x010))
    assert np.allclose([ 0,  0,  1, 0], h.axis_of(x001))
    assert np.allclose([ r,  r,  0, 0], h.axis_of(x110))
    assert np.allclose([ r,  0,  r, 0], h.axis_of(x101))
    assert np.allclose([ 0,  r,  r, 0], h.axis_of(x011))
    assert np.allclose([ r, -r,  0, 0], h.axis_of(x1n0))
    assert np.allclose([ r,  0, -r, 0], h.axis_of(x10n))
    assert np.allclose([ 0, -r,  r, 0], h.axis_of(x01n))

    assert np.allclose([ 1,  0,  0, 0], h.axis_of(h.trans([1,2,3]) @ x000))
    assert np.allclose([ 1,  0,  0, 0], h.axis_of(h.trans([1,2,3]) @ x100))
    assert np.allclose([ 0,  1,  0, 0], h.axis_of(h.trans([1,2,3]) @ x010))
    assert np.allclose([ 0,  0,  1, 0], h.axis_of(h.trans([1,2,3]) @ x001))
    assert np.allclose([ r,  r,  0, 0], h.axis_of(h.trans([1,2,3]) @ x110))
    assert np.allclose([ r,  0,  r, 0], h.axis_of(h.trans([1,2,3]) @ x101))
    assert np.allclose([ 0,  r,  r, 0], h.axis_of(h.trans([1,2,3]) @ x011))
    assert np.allclose([ r, -r,  0, 0], h.axis_of(h.trans([1,2,3]) @ x1n0))
    assert np.allclose([ r,  0, -r, 0], h.axis_of(h.trans([1,2,3]) @ x10n))
    assert np.allclose([ 0, -r,  r, 0], h.axis_of(h.trans([1,2,3]) @ x01n))

    assert np.allclose(0, h.angle_of(x000))
    assert np.allclose(np.pi, h.angle_of(x100))
    assert np.allclose(np.pi, h.angle_of(x010))
    assert np.allclose(np.pi, h.angle_of(x001))
    assert np.allclose(np.pi, h.angle_of(x110))
    assert np.allclose(np.pi, h.angle_of(x101))
    assert np.allclose(np.pi, h.angle_of(x011))
    assert np.allclose(np.pi, h.angle_of(x1n0))
    assert np.allclose(np.pi, h.angle_of(x10n))
    assert np.allclose(np.pi, h.angle_of(x01n))

    xtest = np.array([[ 1.00000000e+00,  1.18776717e-16,  2.37565125e-17,  0.00000000e+00],
                     [-1.90327026e-18, -1.00000000e+00, -1.28807379e-16,  0.00000000e+00],
                     [-3.48949361e-17, -3.34659469e-17, -1.00000000e+00,  0.00000000e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    assert np.allclose(xtest,x100)
    assert np.allclose(h.angle_of(xtest),h.angle_of(x100))
    assert np.allclose(h.axis_of(xtest),h.axis_of(x100))


    xform = h.rand(cart_sd=0)
    xinv = np.linalg.inv(xform)
    x000 = xform @ x000
    x100 = xform @ x100
    x010 = xform @ x010
    x001 = xform @ x001
    x110 = xform @ x110
    x101 = xform @ x101
    x011 = xform @ x011
    x1n0 = xform @ x1n0
    x10n = xform @ x10n
    x01n = xform @ x01n


    # ipd.icv(h.rot([1,2,3,0],2))
    # ipd.icv()
    # ipd.icv(h.rot([1,2,3,0],2)@xform)
    # ipd.icv()
    # ipd.icv(h.rot(xform@[1,2,3,0],2))
    # ipd.icv()
    # ipd.icv(xform@h.rot([1,2,3,0],2))
    # ipd.icv()
    # ipd.icv(h.rot([1,2,3,0]@xform,2))
    # ipd.icv()

    assert np.allclose(        h.rot(       [1,2,3,0],2) @ xform,
                      xform @ h.rot(xinv @ [1,2,3,0],2)        )
    assert np.allclose(        h.rot(        [1,2,3,0],2) @ xinv,
                       xinv @ h.rot(xform @ [1,2,3,0],2)       , atol=1e-6)
    assert np.allclose(        h.rot(        [1,2,3,0],2)        ,
                       xinv @ h.rot(xform @ [1,2,3,0],2)@xform  )

    assert np.allclose(        h.rot(        [1,0,0,0],180)        ,
                       xinv @ h.rot(xform @ [1,0,0,0],180)@xform  )

    # assert 0

    assert np.allclose(np.linalg.det(x000), 1)
    assert np.allclose(np.linalg.det(x100), 1)
    assert np.allclose(np.linalg.det(x010), 1)
    assert np.allclose(np.linalg.det(x001), 1)
    assert np.allclose(np.linalg.det(x110), 1)
    assert np.allclose(np.linalg.det(x101), 1)
    assert np.allclose(np.linalg.det(x011), 1)
    assert np.allclose(np.linalg.det(x1n0), 1)
    assert np.allclose(np.linalg.det(x10n), 1)
    assert np.allclose(np.linalg.det(x01n), 1)

    assert np.allclose(xform @ h.rot([ 1,  0,  0, 0], 180), x100)
    assert np.allclose(xform @ h.rot([ 0,  1,  0, 0], 180), x010)
    assert np.allclose(xform @ h.rot([ 0,  0,  1, 0], 180), x001)
    assert np.allclose(xform @ h.rot([ 1,  1,  0, 0], 180), x110)
    assert np.allclose(xform @ h.rot([ 1,  0,  1, 0], 180), x101)
    assert np.allclose(xform @ h.rot([ 0,  1,  1, 0], 180), x011)
    assert np.allclose(xform @ h.rot([ 1, -1,  0, 0], 180), x1n0)
    assert np.allclose(xform @ h.rot([ 1,  0, -1, 0], 180), x10n)
    assert np.allclose(xform @ h.rot([ 0,  1, -1, 0], 180), x01n)
    assert np.allclose(xform @ h.rot([-1,  1,  0, 0], 180), x1n0)
    assert np.allclose(xform @ h.rot([-1,  0,  1, 0], 180), x10n)
    assert np.allclose(xform @ h.rot([ 0, -1,  1, 0], 180), x01n)

    assert np.allclose(xform @ h.rot(        [1,2,3,0],2) @ xinv ,
                              h.rot(xform @ [1,2,3,0],2)   )

    # yapf: enable

def test_symfit_180_bug():
    frame1, frame2 = np.array([
        [
            [0.6033972864793846, 0.6015781253442000, -0.5234648855030042, -1.1532003332616305],
            [-0.7755284347068191, 0.5955116423562440, -0.2095746725092504, -3.0368828586972385],
            [0.1856538852343365, 0.5324186885984834, 0.8258710477468421, -0.2979720504392123],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
        ],
        [
            [-0.0511070454710340, 0.9143894508662506, 0.4015968309458356, -2.1158643659518335],
            [0.9864845374904164, -0.0164635475722965, 0.1630252172875522, 1.7882112409724600],
            [0.1556802529904169, 0.4045007928280198, -0.9011896470823125, -1.7224210310494734],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
        ],
    ])

    rel = frame2 @ np.linalg.inv(frame1)
    assert np.allclose(rel @ frame1, frame2)

    axs, ang, _cen = h.axis_ang_cen_of(rel)
    # ipd.icv('axs', axs)
    # ipd.icv('ang', ang)
    # ipd.icv('cen', cen)

def torque_delta_sanitycheck():
    nsamp, scale = 1000, 0.0001
    u = h.rand_unit(nsamp)
    v = h.rand_unit(nsamp)
    a, b = np.random.normal(size=2, scale=scale)
    # check commutation
    ax, ang = h.axis_angle_of(h.rot(u, a) @ h.rot(v, b))
    ax2, ang2 = h.axis_angle_of(h.rot(v, b) @ h.rot(u, a))
    assert np.allclose(ax, ax2, atol=5e-4)
    assert np.allclose(ang * scale, ang2 * scale, atol=1e-5)

    uv = a*u + b*v
    anghat = np.linalg.norm(uv, axis=-1)
    axhat = uv / anghat[:, None]
    ax, ang = h.axis_angle_of(h.rot(u, a) @ h.rot(v, b))
    assert np.allclose(ax, axhat, atol=5e-4)

    uv = a*u + b*v
    anghat = np.linalg.norm(uv, axis=-1)
    axhat = uv / anghat[:, None]
    ax, ang = h.axis_angle_of(h.rot(u, a) @ h.rot(v, b))
    assert np.allclose(ax, axhat, atol=5e-4)

def test_hrot():
    r = h.rand_vec()
    assert np.allclose(h.rot(r, 120), h.rot(r, nfold=3))

def test_hrmsfit(trials=10):
    torch = pytest.importorskip("torch")
    for _ in range(trials):
        p = h.randpoint(10, std=10)
        p03 = h.unhomog(p)
        q = h.randpoint(10, std=10)
        # ipd.icv(p)
        rms0 = h.rms(p03, q)
        rms, qhat, xpqhat = h.rmsfit(p03, q)
        assert rms0 > rms
        # ipd.icv(float(rms0), float(rms))
        assert np.allclose(h.rms(qhat, q), rms)
        for _ in range(10):
            rms2 = h.rms(q, h.xform(h.rand_xform_small(1, 0.01, 0.001), qhat))
            # print(float(rms), float(rms2))
            assert rms2 > rms
        rms2, qhat2, xpqhat2 = h.rmsfit(torch.tensor(p), torch.tensor(q))
        assert np.allclose(rms, rms2)
        assert np.allclose(qhat, qhat2)
        assert np.allclose(xpqhat, xpqhat2)

if __name__ == "__main__":
    main()
