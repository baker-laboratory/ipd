import sys

import numpy as np

import ipd

th = ipd.lazyimport("torch")
h = sys.modules[__name__]

def get_dtype_dev(example, dtype=None, device=None):
    if isinstance(example, list) and len(example) < 100:
        for e in example:
            if isinstance(e, th.Tensor):
                example = e
                break
    if device is None:
        if isinstance(example, th.Tensor): device = example.device  # type: ignore
        else: device = 'cpu'
    if dtype is None:
        if isinstance(example, th.Tensor): dtype = example.dtype  # type: ignore
        else: dtype = th.float32
    return dict(dtype=dtype, device=device)

def torch_min(func, iters=4, history_size=10, max_iter=4, line_search_fn="strong_wolfe", **kw):
    import functools

    lbfgs = th.optim.LBFGS(
        kw["indvar"],
        history_size=history_size,
        max_iter=max_iter,
        line_search_fn=line_search_fn,
    )
    closure = functools.partial(func, lbfgs=lbfgs, **kw)
    for iter in range(iters):
        loss = lbfgs.step(closure)
    return loss  # type: ignore

def construct(rot=None, trans=None, dtype=None, device=None):
    kw = get_dtype_dev([rot, trans], dtype, device)
    if rot is None:
        trans = th.as_tensor(trans, **kw)[..., :3]
        x = th.zeros((trans.shape[:-1] + (4, 4)), **kw)
        x[..., :3, :3] = th.eye(3, **kw)
    else:
        x = th.zeros((rot.shape[:-2] + (4, 4)), **kw)
        rot = th.as_tensor(rot, **kw)
        x[..., :3, :3] = rot[..., :3, :3]
    if trans is not None:
        x[..., :3, 3] = th.as_tensor(trans, **kw)[..., :3]
    x[..., 3, 3] = 1
    return x

def trans(trans=None, x=None, y=None, z=None, **kw):
    if trans is None:
        L = max([len(v) for v in (x, y, z) if v is not None])
        trans = th.zeros((L, 3), **kw)
        if x is not None: trans[:, 0] = x
        if y is not None: trans[:, 1] = y
        if z is not None: trans[:, 2] = z
    return construct(trans=trans, **kw)

def mean_along(vecs, along=None):
    vecs = vec(vecs)
    assert vecs.ndim == 2
    if not along:
        along = vecs[0]
    along = vec(along)
    sign = th.sign(dot(along, vecs))
    flipped = (vecs.T * sign).T
    tot = th.sum(flipped, axis=0)
    return normalized(tot)

def com_flat(points, closeto=None, closefrac=0.5):
    if closeto is not None:
        dist = norm(points - closeto)
        close = th.argsort(dist)[:closefrac * len(dist)]
        points = points[close]
    return th.mean(points, axis=-2)

def com(points, **kw):
    points = point(points)
    oshape = points.shape
    points = points.reshape(-1, oshape[-2], 4)
    com = com_flat(points)
    com = com.reshape(*oshape[:-2], 4)
    return com

def rog_flat(points):
    com = com_flat(points).reshape(-1, 1, 4)
    delta = th.linalg.norm(points - com, dim=2)
    rg = th.sqrt(th.mean(delta**2, dim=1))
    return rg

def rog(points, aboutaxis=None):
    points = point(points)
    oshape = points.shape
    points = points.reshape(-1, *oshape[-2:])
    if aboutaxis is not None:
        aboutaxis = vec(aboutaxis)
        points = projperp(aboutaxis, points)
    rog = rog_flat(points)
    rog = rog.reshape(oshape[:-2])
    return rog

def proj(u, v):
    u = vec(u)
    v = point(v)
    return dot(u, v)[..., None] / norm2(u)[..., None] * u

def projperp(u, v):
    u = vec(u)
    v = point(v)
    return v - proj(u, v)

def choose_axis_flip(axis, angle):
    if th.is_tensor(axis):
        flipaxis = th.matmul(axis, th.tensor([1, 2, 11, 0], dtype=axis.dtype)) < 0
        angle = th.where(flipaxis, -angle, angle)
        flipaxis = flipaxis.tile(4).reshape(4, *flipaxis.shape).swapdims(0, -1)
        axis = th.where(flipaxis, -axis, axis)
        return axis, angle
    else:
        oshape = axis.shape
        axis = axis.reshape(-1, 4)
        angle = angle.reshape(-1)
        flipaxis = np.dot(axis, [1, 2, 11, 0]) < 0
        axis[flipaxis] = -axis[flipaxis]
        angle[flipaxis] = -angle[flipaxis]
        return axis.reshape(oshape), angle.reshape(oshape[:-1])

def axis_angle_cen(xforms, ident_match_tol=1e-8, flipaxis=True):
    # ic(xforms.dtype)
    origshape = xforms.shape[:-2]
    xforms = xforms.reshape(-1, 4, 4)
    axis, angle = axis_angle(xforms)
    not_ident = th.abs(angle) > ident_match_tol
    cen = th.tile(
        th.tensor([0, 0, 0, 1]),
        angle.shape,
    ).reshape(*angle.shape, 4)

    # assert th.all(not_ident)
    # xforms1 = xforms[not_ident]
    # axis1 = axis[not_ident]
    #  sketchy magic points...
    p1, p2 = axis_ang_cen_magic_points_torch()
    p1 = p1.to(xforms.dtype)
    p2 = p2.to(xforms.dtype)
    tparallel = dot(axis, xforms[..., :, 3])[..., None] * axis
    q1 = xforms@p1 - tparallel
    q2 = xforms@p2 - tparallel
    n1 = normalized(q1 - p1).reshape(-1, 4)
    n2 = normalized(q2 - p2).reshape(-1, 4)
    c1 = (p1+q1) / 2.0
    c2 = (p2+q2) / 2.0

    isect, norm, status = intersect_planes(c1, n1, c2, n2)
    cen1 = isect[..., :]
    if len(cen) == len(cen1):
        cen = cen1
    else:
        cen = th.where(not_ident, cen1, cen)
    if flipaxis:
        axis, angle = choose_axis_flip(axis, angle)

    axis = axis.reshape(*origshape, 4)
    angle = angle.reshape(origshape)
    cen = cen.reshape(*origshape, 4)
    return axis, angle, cen

def rot(axis, angle, center=None, hel=None, squeeze=True, degrees=False, dtype=None, device=None):
    assert not degrees
    kw = get_dtype_dev(axis, dtype, device)
    if center is None:
        center = th.tensor([0, 0, 0, 1], **kw)
    angle = th.as_tensor(angle, **kw)
    axis = vec(axis, **kw)
    center = point(center, **kw)
    if hel is None:
        hel = th.tensor([0], **kw)
    if axis.ndim == 1:
        axis = axis[None]
    if angle.ndim == 0:
        angle = angle[None]
    if center.ndim == 1:
        center = center[None]
    if hel.ndim == 0:
        hel = hel[None]
    rot = rot3(axis, angle, shape=(4, 4), squeeze=False, **kw)
    shape = angle.shape
    if axis.ndim > 1 and not (axis.ndim == 2 and len(axis) == 1): shape = axis.shape[:-1]

    # assert 0
    x, y, z = center[..., 0], center[..., 1], center[..., 2]
    center = th.stack(
        [
            x - rot[..., 0, 0] * x - rot[..., 0, 1] * y - rot[..., 0, 2] * z,
            y - rot[..., 1, 0] * x - rot[..., 1, 1] * y - rot[..., 1, 2] * z,
            z - rot[..., 2, 0] * x - rot[..., 2, 1] * y - rot[..., 2, 2] * z,
            th.ones(*shape, **kw),
        ],
        axis=-1,
    )
    shift = axis * hel[..., None]
    center = center + shift
    r = th.cat([rot[..., :3], center[..., None]], axis=-1)
    if r.shape == (1, 4, 4):
        r = r.reshape(4, 4)
    return r

def randpoint(shape=(), mean=0, std=1, dtype=None, device=None):
    kw = get_dtype_dev([std, mean], dtype, device)
    if isinstance(shape, int):
        shape = (shape, )
    p = point(th.randn((shape + (3, )), **kw) * std + mean)
    return p

def randvec(shape=(), dtype=None, device=None):
    kw = get_dtype_dev(shape, dtype, device)
    if isinstance(shape, int):
        shape = (shape, )
    v = vec(th.randn(*(shape + (3, ))), **kw)
    if isinstance(shape, int):
        shape = (shape, )
    return v

def randunit(shape=(), device=None, dtype=None):
    kw = get_dtype_dev(shape, dtype, device)
    if isinstance(shape, int): shape = (shape, )
    v = normalized(th.randn(shape + (3, ), **kw))
    return v

def randsmall(shape=(), cart_sd=0.001, rot_sd=0.001, centers=None, device=None, dtype=None):
    kw = get_dtype_dev([cart_sd, rot_sd, centers], dtype, device)
    if isinstance(shape, int):
        shape = (shape, )
    axis = randunit(shape, **kw)
    ang = th.randn(shape, **kw) * rot_sd * np.pi
    if centers is None: centers = [0, 0, 0, 1]
    else: assert centers.shape[:-1] in ((), shape)
    x = rot(axis, ang, centers, degrees=False, **kw).squeeze()  # type: ignore
    trans = th.randn(x[..., :3, 3].shape, **kw) * cart_sd
    x[..., :3, 3] += trans
    return x

def rand_xform(shape=(), cart_cen=0, cart_sd=1, dtype=None, device=None):
    kw = get_dtype_dev([cart_cen, cart_sd], dtype, device)
    if isinstance(shape, int):
        shape = (shape, )
    t = ipd.dev.Timer()
    q = th.randn(shape + (4, ), **kw)
    q = normQ(q)
    # q = th.nn.functional.normalize(q)
    x = quat_to_xform(q)
    x[..., :3, 3] = th.randn(shape + (3, ), **kw) * cart_sd + cart_cen
    x[..., 3, 3] = 1
    return x

rand = rand_xform

def rot_to_quat(xform):
    raise NotImplementedError
    x = np.asarray(xform)
    t0, t1, t2 = x[..., 0, 0], x[..., 1, 1], x[..., 2, 2]
    tr = t0 + t1 + t2
    quat = np.empty(x.shape[:-2] + (4, ))

    case0 = tr > 0
    S0 = np.sqrt(tr[case0] + 1) * 2
    quat[case0, 0] = 0.25 * S0
    quat[case0, 1] = (x[case0, 2, 1] - x[case0, 1, 2]) / S0
    quat[case0, 2] = (x[case0, 0, 2] - x[case0, 2, 0]) / S0
    quat[case0, 3] = (x[case0, 1, 0] - x[case0, 0, 1]) / S0

    case1 = ~case0 * (t0 >= t1) * (t0 >= t2)
    S1 = np.sqrt(1.0 + x[case1, 0, 0] - x[case1, 1, 1] - x[case1, 2, 2]) * 2
    quat[case1, 0] = (x[case1, 2, 1] - x[case1, 1, 2]) / S1
    quat[case1, 1] = 0.25 * S1
    quat[case1, 2] = (x[case1, 0, 1] + x[case1, 1, 0]) / S1
    quat[case1, 3] = (x[case1, 0, 2] + x[case1, 2, 0]) / S1

    case2 = ~case0 * (t1 > t0) * (t1 >= t2)
    S2 = np.sqrt(1.0 + x[case2, 1, 1] - x[case2, 0, 0] - x[case2, 2, 2]) * 2
    quat[case2, 0] = (x[case2, 0, 2] - x[case2, 2, 0]) / S2
    quat[case2, 1] = (x[case2, 0, 1] + x[case2, 1, 0]) / S2
    quat[case2, 2] = 0.25 * S2
    quat[case2, 3] = (x[case2, 1, 2] + x[case2, 2, 1]) / S2

    case3 = ~case0 * (t2 > t0) * (t2 > t1)
    S3 = np.sqrt(1.0 + x[case3, 2, 2] - x[case3, 0, 0] - x[case3, 1, 1]) * 2
    quat[case3, 0] = (x[case3, 1, 0] - x[case3, 0, 1]) / S3
    quat[case3, 1] = (x[case3, 0, 2] + x[case3, 2, 0]) / S3
    quat[case3, 2] = (x[case3, 1, 2] + x[case3, 2, 1]) / S3
    quat[case3, 3] = 0.25 * S3

    assert np.sum(case0) + np.sum(case1) + np.sum(case2) + np.sum(case3) == np.prod(xform.shape[:-2])

    return quat_to_upper_half(quat)

xform_to_quat = rot_to_quat

def is_valid_quat_rot(quat):
    assert quat.shape[-1] == 4
    return np.isclose(1, th.linalg.norm(quat, axis=-1))

def quat_to_upper_half(quat):
    ineg0 = quat[..., 0] < 0
    ineg1 = (quat[..., 0] == 0) * (quat[..., 1] < 0)
    ineg2 = (quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] < 0)
    ineg3 = (quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] == 0) * (quat[..., 3] < 0)
    # ic(ineg0.shape)
    # ic(ineg1.shape)
    # ic(ineg2.shape)
    # ic(ineg3.shape)
    ineg = ineg0 + ineg1 + ineg2 + ineg3
    quat2 = th.where(ineg, -quat, quat)
    return normalized(quat2)

def cart(h):
    return h[..., :, 3]

def cart3(h):
    return h[..., :3, 3]

def ori(h):
    h = h.clone()
    h[..., :3, 3] = 0
    return h

def ori3(h):
    h[..., :3, :3]

def homog(rot, trans=None, **kw):
    if trans is None:
        trans = th.as_tensor([0, 0, 0, 0], device=rot.device, **kw)
    trans = th.as_tensor(trans)

    if rot.shape == (3, 3):
        rot = th.cat([rot, th.tensor([[0.0, 0.0, 0.0]], device=rot.device)], axis=0)
        rot = th.cat([rot, th.tensor([[0], [0], [0], [1]], device=rot.device)], axis=1)

    assert rot.shape[-2:] == (4, 4)
    assert trans.shape[-1:] == (4, )

    h = th.cat([rot[:, :3], trans[:, None]], axis=1)
    return h

def quat_to_xform(quat):
    r44 = Qs2Rs(quat, shape=(4, 4))
    r44[..., 3, 3] = 1
    return r44

def rot3(axis, angle, shape=(3, 3), squeeze=True, dtype=None, device=None):
    # axis = th.tensor(axis, dtype=dtype, requires_grad=requires_grad)
    # angle = angle * np.pi / 180.0 if degrees else angle
    # angle = th.tensor(angle, dtype=dtype, requires_grad=requires_grad)
    kw = get_dtype_dev(axis, dtype, device)

    if axis.ndim == 1:
        axis = axis[None]
    if isinstance(angle, (int, float)) or angle.ndim == 0:
        angle = th.as_tensor([angle])
    # if angle.ndim == 0
    if axis.shape and angle.shape and not is_broadcastable(axis.shape[:-1], angle.shape):
        raise ValueError(f"axis/angle not compatible: {axis.shape} {angle.shape}")
    if axis.ndim > 1 and not (axis.ndim == 2 and len(axis) == 1): zero = th.zeros(*axis.shape[:-1], **kw)
    else: zero = th.zeros(*angle.shape, **kw)

    axis = normalized(axis)
    a = th.cos(angle / 2.0)
    tmp = axis * -th.sin(angle / 2)[..., None]
    b, c, d = tmp[..., 0], tmp[..., 1], tmp[..., 2]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    # ic(axis.dtype)
    # ic(angle.dtype)
    # ic(a.dtype)
    if shape == (3, 3):
        rot = th.stack([
            th.stack([aa + bb - cc - dd, 2 * (bc+ad), 2 * (bd-ac)], axis=-1),
            th.stack([2 * (bc-ad), aa + cc - bb - dd, 2 * (cd+ab)], axis=-1),
            th.stack([2 * (bd+ac), 2 * (cd-ab), aa + dd - bb - cc], axis=-1),
        ],
                       axis=-2)
    elif shape == (4, 4):
        rot = th.stack([
            th.stack([aa + bb - cc - dd, 2 * (bc+ad), 2 * (bd-ac), zero], axis=-1),
            th.stack([2 * (bc-ad), aa + cc - bb - dd, 2 * (cd+ab), zero], axis=-1),
            th.stack([2 * (bd+ac), 2 * (cd-ab), aa + dd - bb - cc, zero], axis=-1),
            th.stack([zero, zero, zero, zero + 1], axis=-1),
        ],
                       axis=-2)
    else:
        raise ValueError(f"rot3 shape must be (3,3) or (4,4), not {shape}")
    # ic('foo')
    # ic(axis.shape)
    # ic(angle.shape)
    # ic(rot.shape)
    if squeeze and rot.shape == (1, 3, 3):
        rot = rot.reshape(3, 3)
    if squeeze and rot.shape == (1, 4, 4):
        rot = rot.reshape(4, 4)
    return rot

def rms(a, b):
    assert a.shape == b.shape
    return th.sqrt(th.sum(th.square(a - b)) / len(a))

def xform(xform, stuff, homogout="auto", **kw):
    kwdt = get_dtype_dev([xform, stuff], None, None)
    xform = th.as_tensor(xform, **kwdt)
    nothomog = stuff.shape[-1] == 3
    if stuff.shape[-1] == 3:
        stuff = point(stuff, **kwdt)
    result = _thxform_impl(xform, stuff, **kw)
    if homogout is False or homogout == "auto" and nothomog:
        result = result[..., :3]
    return result

_xform = xform

def xformpts(xform, stuff, **kw):
    return _xform(xform, stuff, is_points=True, **kw)

    # if result.shape[-1] == 4 and not ipd.homog.hvalid(result.cpu().detach().numpy(), **kw):
    #   # ic(result[:10])
    #   # is is a bad copout.. should make is check handle nans correctly
    #   if not stuff.shape[-2:] == (4, 1):
    #      raise ValueError(
    #         f'malformed homogeneous coords with shape {stuff.shape}, if points and shape is (...,4,4) try is_points=True'
    #      )

    return result

def rmsfit(mobile, target):
    """Use kabsch method to get rmsd fit."""
    assert mobile.shape == target.shape
    assert mobile.ndim > 1
    assert mobile.shape[-1] in (3, 4)
    if len(mobile) < 3:
        raise ValueError("need at least 3 points to fit")
    if mobile.dtype != target.dtype:
        mobile = mobile.to(target.dtype)
    mobile = point(mobile)
    target = point(target)
    mobile_cen = th.mean(mobile, axis=0)
    target_cen = th.mean(target, axis=0)
    mobile = mobile - mobile_cen
    target = target - target_cen
    # ic(mobile.shape)
    # ic(target.shape[-1] in (3, 4))
    covariance = mobile.T[:3] @ target[:, :3]
    V, S, W = th.linalg.svd(covariance)
    if 0 > th.det(V) * th.det(W):
        S = th.tensor([S[0], S[1], -S[2]], dtype=S.dtype, device=S.device)
        # S[-1] = -S[-1]
        # ic(S - S1)
        V = th.cat([V[:, :-1], -V[:, -1, None]], dim=1)
        # V[:, -1] = -V[:, -1]
        # ic(V - V1)
        # assert 0
    rot_m2t = homog(V @ W).T
    trans_m2t = target_cen - rot_m2t@mobile_cen
    xform_mobile_to_target = homog(rot_m2t, trans_m2t)

    mobile = mobile + mobile_cen
    target = target + target_cen
    mobile_fit_to_target = xform(xform_mobile_to_target, mobile)
    rms_ = rms(target, mobile_fit_to_target)

    return rms_, mobile_fit_to_target, xform_mobile_to_target

# def randunit(shape=(), cen=[0, 0, 0], std=1):
#     if isinstance(shape, int):
#         shape = (shape, )
#     v = normalized(th.randn(*(shape + (3, ))) * std)
#     return v

def point(point, dtype=None, device=None, **kw):
    kw = get_dtype_dev(point, dtype, device)
    point = th.as_tensor(point, **kw)
    shape = point.shape[:-1]
    points = th.cat([point[..., :3], th.ones(shape + (1, ), **kw)], axis=-1)
    if points.dtype not in (th.float32, th.float64):
        points = points.to(th.float32)
    return points

def vec(vec, dtype=None, device=None):
    kw = get_dtype_dev(vec, dtype, device)
    vec = th.as_tensor(vec, **kw)
    if vec.dtype not in (th.float32, th.float64):
        vec = vec.to(th.float32)
    if vec.shape[-1] == 4:
        if th.any(vec[..., 3] != 0):
            vec = th.cat([vec[..., :3], th.zeros(*vec.shape[:-1], 1, **kw)], dim=-1)
        return vec
    elif vec.shape[-1] == 3:
        r = th.zeros(vec.shape[:-1] + (4, ), **kw)
        r[..., :3] = vec
        return r
    else:
        raise ValueError("vec must len 3 or 4")

def normvec(inp, dtype=None, device=None):
    return normalized(vec(inp, dtype, device))

def normalized(a):
    kw = get_dtype_dev(a)
    return th.nn.functional.normalize(th.as_tensor(a, **kw), dim=-1)
    # a = th.as_tensor(a)
    # if (not a.shape and len(a) == 3) or (a.shape and a.shape[-1] == 3):
    #    a, tmp = th.zeros(a.shape[:-1] + (4, ), dtype=a.type), a
    #    a[..., :3] = tmp
    # a2 = a[:]
    # a2[..., 3] = 0
    # return a2 / norm(a2)[..., None]

def norm(a):
    a = th.as_tensor(a)
    return th.sqrt(th.sum(a[..., :3] * a[..., :3], axis=-1))

def norm2(a):
    a = th.as_tensor(a)
    return th.sum(a[..., :3] * a[..., :3], axis=-1)

def axis_angle_hel(xforms):
    axis, angle = axis_angle(xforms)
    hel = dot(axis, xforms[..., :, 3])
    return axis, angle, hel

def axis_angle_cen_hel(xforms, **kw):
    axis, angle, cen = axis_angle_cen(xforms, **kw)
    hel = dot(axis, xforms[..., :, 3])
    return axis, angle, cen, hel

def axis_angle(xforms):
    axis_ = axis(xforms)
    angl = angle(xforms)
    return axis_, angl

def axis(xforms):
    if xforms.shape[-2:] == (4, 4):
        return normalized(
            th.stack(
                (
                    xforms[..., 2, 1] - xforms[..., 1, 2],
                    xforms[..., 0, 2] - xforms[..., 2, 0],
                    xforms[..., 1, 0] - xforms[..., 0, 1],
                    th.zeros(xforms.shape[:-2], dtype=xforms.dtype, device=xforms.device),
                ),
                axis=-1,
            ))
    if xforms.shape[-2:] == (3, 3):
        return normalized(
            th.stack(
                (
                    xforms[..., 2, 1] - xforms[..., 1, 2],
                    xforms[..., 0, 2] - xforms[..., 2, 0],
                    xforms[..., 1, 0] - xforms[..., 0, 1],
                ),
                axis=-1,
            ))
    else:
        raise ValueError("wrong shape for xform/rotation matrix: " + str(xforms.shape))

def angle(xforms):
    tr = xforms[..., 0, 0] + xforms[..., 1, 1] + xforms[..., 2, 2]
    cos = (tr-1.0) / 2.0
    angl = th.arccos(th.clip(cos, -1, 1))
    return angl

def point_line_dist2(point, cen, norm):
    point, cen, norm = h.point(point), h.point(cen), h.normalized(norm)
    point = point - cen
    perp = h.projperp(norm, point)
    return h.norm2(perp)

def dot(a, b, outerprod=False):
    if outerprod:
        shape1 = a.shape[:-1]
        shape2 = b.shape[:-1]
        a = a.reshape((1, ) * len(shape2) + shape1 + (-1, ))
        b = b.reshape(shape2 + (1, ) * len(shape1) + (-1, ))
    return th.sum(a[..., :3] * b[..., :3], axis=-1)

def point_in_plane(point, normal, pt):
    inplane = th.abs(dot(normal[..., :3], pt[..., :3] - point[..., :3]))
    return inplane < 0.00001

def ray_in_plane(point, normal, p1, n1):
    inplane1 = point_in_plane(point, normal, p1)
    inplane2 = point_in_plane(point, normal, p1 + n1)
    return inplane1 and inplane2

def intersect_planes(p1, n1, p2, n2):
    """
    intersect_Planes: find e 3D intersection of two planes
       Input:  two planes represented (point, normal) as (p1,n1), (p2,n2)
       Output: L = e intersection line (when it exists)
       Return: rays shape=(...,4,2), status
               0 = intersection returned
               1 = disjoint (no intersection)
               2 = e two planes coincide
    """
    """Intersect two planes :param plane1: first plane represented by ray :type
    plane2: np.array shape=(..., 4, 2) :param plane1: second planes represented
    by rays :type plane2: np.array shape=(..., 4, 2) :return: line: np.array
    shape=(...,4,2), status: int (0 = intersection returned, 1 = no
    intersection, 2 = e two planes coincide)"""
    origshape = p1.shape
    # shape = origshape[:-1] or [1]
    assert p1.shape[-1] == 4
    assert p1.shape == n1.shape
    assert p1.shape == p2.shape
    assert p1.shape == n2.shape
    p1 = p1.reshape(-1, 4)
    n1 = n1.reshape(-1, 4)
    p2 = p2.reshape(-1, 4)
    n2 = n2.reshape(-1, 4)
    N = len(p1)

    u = th.linalg.cross(n1[..., :3], n2[..., :3])
    abs_u = th.abs(u)
    planes_parallel = th.sum(abs_u, axis=-1) < 0.000001
    p2_in_plane1 = point_in_plane(p1, n1, p2)
    status = th.zeros(N)
    status[planes_parallel] = 1
    status[planes_parallel * p2_in_plane1] = 2
    d1 = -dot(n1, p1)
    d2 = -dot(n2, p2)

    amax = th.argmax(abs_u, axis=-1)
    sel = amax == 0, amax == 1, amax == 2
    perm = th.cat([
        th.where(sel[0])[0],
        th.where(sel[1])[0],
        th.where(sel[2])[0],
    ])
    perminv = th.empty_like(perm)
    perminv[perm] = th.arange(len(perm))
    breaks = np.cumsum([0, sum(sel[0]), sum(sel[1]), sum(sel[2])])
    n1 = n1[perm]
    n2 = n2[perm]
    d1 = d1[perm]
    d2 = d2[perm]
    up = u[perm]

    zeros = th.zeros(N)
    ones = th.ones(N)
    l = []

    s = slice(breaks[0], breaks[1])
    y = (d2[s] * n1[s, 2] - d1[s] * n2[s, 2]) / up[s, 0]
    z = (d1[s] * n2[s, 1] - d2[s] * n1[s, 1]) / up[s, 0]
    l.append(th.stack([zeros[s], y, z, ones[s]], axis=-1))

    s = slice(breaks[1], breaks[2])
    z = (d2[s] * n1[s, 0] - d1[s] * n2[s, 0]) / up[s, 1]
    x = (d1[s] * n2[s, 2] - d2[s] * n1[s, 2]) / up[s, 1]
    l.append(th.stack([x, zeros[s], z, ones[s]], axis=-1))

    s = slice(breaks[2], breaks[3])
    x = (d2[s] * n1[s, 1] - d1[s] * n2[s, 1]) / up[s, 2]
    y = (d1[s] * n2[s, 0] - d2[s] * n1[s, 0]) / up[s, 2]
    l.append(th.stack([x, y, zeros[s], ones[s]], axis=-1))

    isect_pt = th.cat(l)
    isect_pt = isect_pt[perminv]
    isect_pt = isect_pt.reshape(origshape)

    isect_dirn = normalized(th.cat([u, th.zeros(N, 1)], axis=-1))
    isect_dirn = isect_dirn.reshape(origshape)

    return isect_pt, isect_dirn, status

def is_broadcastable(shape1, shape2):
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True

def axis_ang_cen_magic_points_torch():
    return th.from_numpy(ipd.homog.hgeom._axis_ang_cen_magic_points_numpy).float()

def diff(x, y, lever=10.0):
    shape1 = x.shape[:-2]
    shape2 = y.shape[:-2]
    a = x.reshape(shape1 + (1, ) * len(shape1) + (4, 4))
    b = y.reshape((1, ) * len(shape2) + shape2 + (4, 4))

    axyz = a[..., :3, :3] * lever + a[..., :3, 3, None]
    bxyz = b[..., :3, :3] * lever + b[..., :3, 3, None]

    diff = th.norm(axyz - bxyz, dim=-1)
    diff = th.mean(diff, dim=-1)

    return diff

def cross(u, v):
    return vec(th.linalg.cross(u[..., :3], v[..., :3]))

def frame(u, v, w, cen=None, primary='x', **kw):
    dd = get_dtype_dev([u, v, w, cen], **kw)
    assert u.shape == v.shape == w.shape
    if cen is None: cen = u
    assert cen.shape == u.shape
    u, v, w, cen = vec(u, **dd), vec(v, **dd), vec(w, **dd), point(cen, **dd)
    stubs = th.empty(u.shape[:-1] + (4, 4), **dd)

    if primary == 'z': order = [2, 1, 0]
    elif primary == 'y': order = [2, 0, 2]
    else: order = [0, 2, 1]
    stubs[..., :, order[0]] = normalized(u - v)
    stubs[..., :, order[1]] = normalized(cross(stubs[..., :, order[0]], w - v))
    stubs[..., :, order[2]] = cross(stubs[..., :, order[1]], stubs[..., :, order[0]])
    stubs[..., :, 3] = cen[..., :]
    assert valid44(stubs)
    return stubs

def Qs2Rs(Qs, shape=(3, 3)):
    Rs = th.zeros(Qs.shape[:-1] + shape, dtype=Qs.dtype, device=Qs.device)

    Rs[..., 0, 0] = (Qs[..., 0] * Qs[..., 0] + Qs[..., 1] * Qs[..., 1] - Qs[..., 2] * Qs[..., 2] - Qs[..., 3] * Qs[..., 3])
    Rs[..., 0, 1] = 2 * Qs[..., 1] * Qs[..., 2] - 2 * Qs[..., 0] * Qs[..., 3]
    Rs[..., 0, 2] = 2 * Qs[..., 1] * Qs[..., 3] + 2 * Qs[..., 0] * Qs[..., 2]
    Rs[..., 1, 0] = 2 * Qs[..., 1] * Qs[..., 2] + 2 * Qs[..., 0] * Qs[..., 3]
    Rs[..., 1, 1] = (Qs[..., 0] * Qs[..., 0] - Qs[..., 1] * Qs[..., 1] + Qs[..., 2] * Qs[..., 2] - Qs[..., 3] * Qs[..., 3])
    Rs[..., 1, 2] = 2 * Qs[..., 2] * Qs[..., 3] - 2 * Qs[..., 0] * Qs[..., 1]
    Rs[..., 2, 0] = 2 * Qs[..., 1] * Qs[..., 3] - 2 * Qs[..., 0] * Qs[..., 2]
    Rs[..., 2, 1] = 2 * Qs[..., 2] * Qs[..., 3] + 2 * Qs[..., 0] * Qs[..., 1]
    Rs[..., 2, 2] = (Qs[..., 0] * Qs[..., 0] - Qs[..., 1] * Qs[..., 1] - Qs[..., 2] * Qs[..., 2] + Qs[..., 3] * Qs[..., 3])

    return Rs

# ============================================================
def normQ(Q):
    """Normalize a quaternions."""
    return Q / th.linalg.norm(Q, keepdim=True, dim=-1)

def Q2R(Q):
    Qs = th.cat((th.ones((len(Q), 1), device=Q.device, dtype=Q.dtype), Q), dim=-1)
    Qs = normQ(Qs)
    return Qs2Rs(Qs[None, :]).squeeze(0)

def _thxform_impl(x, stuff, outerprod="auto", flat=False, is_points="auto", improper_ok=False):
    if is_points == "auto":
        is_points = not valid44(stuff, improper_ok=improper_ok)
        if is_points:
            if stuff.shape[-1] != 4 and stuff.shape[-2:] == (4, 1):
                raise ValueError(f"hxform cant understand shape {stuff.shape}")
    if not is_points:
        if outerprod == "auto":
            outerprod = x.shape[:-2] != stuff.shape[:-2]
        if outerprod:
            shape1 = x.shape[:-2]
            shape2 = stuff.shape[:-2]
            a = x.reshape(shape1 + (1, ) * len(shape2) + (4, 4))
            b = stuff.reshape((1, ) * len(shape1) + shape2 + (4, 4))
            result = a @ b
            if flat:
                result = result.reshape(-1, 4, 4)
        else:
            result = x @ stuff
        if flat:
            result = result.reshape(-1, 4, 4)
    else:
        if outerprod == "auto":
            outerprod = x.shape[:-2] != stuff.shape[:-1]

        if stuff.shape[-1] != 1:
            stuff = stuff[..., None]
        if outerprod:
            shape1 = x.shape[:-2]
            shape2 = stuff.shape[:-2]
            # ic(x.shape, stuff.shape, shape1, shape2)
            a = x.reshape(shape1 + (1, ) * len(shape2) + (4, 4))

            b = stuff.reshape((1, ) * len(shape1) + shape2 + (4, 1))
            result = a @ b
        else:
            # try to match first N dimensions, outer prod the rest
            shape1 = x.shape[:-2]
            shape2 = stuff.shape[:-2]
            sameshape = tuple()
            for i, (s1, s2) in enumerate(zip(shape1, shape2)):
                # ic(s1, s2)
                if s1 == s2:
                    shape1 = shape1[1:]
                    shape2 = shape2[1:]
                    sameshape = sameshape + (s1, )
                else:
                    break
            newshape1 = sameshape + shape1 + (1, ) * len(shape2) + (4, 4)
            newshape2 = sameshape + (1, ) * len(shape1) + shape2 + (4, 1)
            # ic(shape1, shape2, newshape1, newshape2)
            a = x.reshape(newshape1)
            b = stuff.reshape(newshape2)
            result = a @ b

        result = result.squeeze(axis=-1)

        if flat:
            result = result.reshape(-1, 4)

        # assert 0
    # ic('result', result.shape)
    return result

def valid(stuff, is_points=None, strict=False, **kw):
    if stuff.shape[-2:] == (4, 4) and not is_points:
        return valid44(stuff, **kw)
    if stuff.shape[-2:] == (4, 2) and not is_points:
        return is_valid_rays(stuff)  # type: ignore
    elif stuff.shape[-1] == 4 and strict:
        return th.allclose(stuff[..., 3], 0) or th.allclose(stuff[..., 3], 1)
    elif stuff.shape[-1] == 4:
        return th.all(th.logical_or(th.isclose(stuff[..., 3], 0), th.isclose(stuff[..., 3], 1)))
    elif stuff.shape[-1] == 3:
        return True
    return False

def valid_norm(x):
    normok = th.allclose(1, th.linalg.norm(x[..., :3, :3], axis=-1))
    normok &= th.allclose(1, th.linalg.norm(x[..., :3, :3], axis=-2))
    return th.all(normok)

def valid44(x, improper_ok=False, debug=False, **kw):
    if x.shape[-2:] != (4, 4):
        return False
    det = th.linalg.det(x[..., :3, :3])
    if improper_ok:
        det = th.abs(det)

    detok = th.allclose(det, th.tensor(1.0, dtype=x.dtype), atol=1e-4)
    is_one_33 = th.allclose(x[..., 3, 3], th.tensor(1.0, dtype=x.dtype))
    is_zero_3_012 = th.allclose(x[..., 3, :3], th.tensor(0.0, dtype=x.dtype))
    ok = is_zero_3_012 and is_one_33 and detok
    if debug and not ok: ic(improper_ok, det, detok, is_one_33, is_zero_3_012)  # type: ignore
    return ok

def inv(x):
    return th.linalg.inv(x)

def tocuda(x):
    return th.as_tensor(x, device='cuda')

def remove_diagonal_elements(a):
    assert a.shape[0] == a.shape[1]
    n = len(a)
    elemsize = th.prod(th.tensor(a.shape[2:]))
    return a.flatten()[elemsize:].view(n - 1, n + 1, -1)[:, :-1].reshape(n, n - 1, *a.shape[2:])
