import math

import ipd
from ipd.lazy_import import lazyimport

th = lazyimport('torch')

import numpy as np

from ipd import h

_sampling = ipd.dev.lazyimport('ipd.samp.sampling_cuda')

def sort_inplace_topk(data, k):
    data = data.to('cuda')
    if data.dtype == th.float32:
        return _sampling.sort_inplace_topk_float(data, k)
    else:
        raise ValueError('data must be float32')

def randxform(
    shape,
    cartmean=0,
    cartsd=None,
    cartmax=None,
    orisd=None,
    orimax=None,
    cen=None,
    dtype=th.float32,
    device='cuda',
    seed=-1,
    nthread=256,
    gentype='curandStatePhilox4_32_10_t',
    # gentype='curandState',
):
    if isinstance(shape, (int, float)): shape = (int(shape), )
    n = int(th.prod(th.tensor(shape), dtype=th.int64).item())
    if isinstance(cartmean, (float, int)): cartmean = [cartmean] * 3
    cartmean = th.as_tensor(cartmean, dtype=dtype, device='cuda')
    if seed < 0: seed = int(th.randint(0, 2**53, (1, ), dtype=th.float64))
    cart_uniform = True
    assert cartmax is None or cartsd is None
    if cartmax is None and cartsd is None:
        cartsd = 1
    elif cartmax is not None:
        cart_uniform = True
        cartsd = cartmax
    else:
        # cartsd = cartsd / np.sqrt(3)
        cart_uniform = False
    if isinstance(cartsd, (int, float)):
        cartsd = th.tensor([cartsd, cartsd, cartsd])
    cartsd = th.as_tensor(cartsd, dtype=dtype, device='cuda')
    assert orimax is None or orisd is None
    orinormal = False
    if orimax is not None:
        quat_height = np.cos(np.clip(orimax, 0, th.pi) / 2)
    elif orisd is None:
        quat_height = 0
    else:
        assert orimax is None
        if orisd > 0.4: raise ValueError('orisd must be less than 0.4')
        coeff = [
            1.7849621e-01, 2.7500242e-01, 2.1911124e-03, -1.7134663e+00, 5.3797978e-01, 1.5170078e+01, 3.3340999e-01,
            -5.5727810e+01, -1.0274599e+01, 9.9600327e+01, 2.8866692e+01, -8.5288895e+01, -3.1181292e+01, 2.8087727e+01,
            1.1939758e+01
        ]
        std2qh = np.polynomial.Polynomial(coeff, domain=[0., 0.4326], window=[-1., 1.])
        orimax = orisd
        orinormal = True
        quat_height = std2qh(orisd)

    x = _sampling.rand_xform(
        n=n,
        cart_mean=cartmean,
        cart_sd=cartsd,
        cart_uniform=cart_uniform,
        quat_height=quat_height,
        orinormal=orinormal,
        dtype=str(dtype),
        nthread=nthread,
        seed=seed,
        gentype=gentype,
    ).reshape(shape + (4, 4))

    if cen is not None:
        cen = cen.reshape((1, ) * (x.ndim - cen.ndim - 1) + cen.shape)
        x[..., :3, 3] += cen[:, :3] - (x[..., :3, :3] @ cen[..., :3, None])[..., 0]

    return x.to(device)

def quat_torus_xform(resl, maxtip=th.pi / 6, ringang=2 * th.pi, bcc=True, device='cuda'):
    """Samples orientations in a hypercone -- tilts plus rotation around cone
    axis recommend maxtip a multiple of resl."""
    n1 = int(math.ceil(ringang / resl))
    n2 = int(math.ceil(maxtip / resl))
    n3 = n2*2 + 1
    # ic(n1, n2, n3)
    o = th.as_tensor([0, resl / 2])[:, None] if bcc else th.as_tensor([0])
    wx = th.linspace(0, ringang, n1).repeat_interleave(n3**2)
    wx = (wx + o).reshape(-1)
    w = th.cos(wx / 2)
    x = th.sin(wx / 2)
    yz = (th.linspace(-maxtip, maxtip, n3) + o).reshape(-1)
    y = yz.repeat_interleave(n3).tile(n1)
    z = yz.tile(n1 * n3)
    # w, x, y, z = w.reshape(-1), x.reshape(-1), y.reshape(-1), z.reshape(-1)
    y, z = y / 3, z / 3  # why /3 rather than /2 ??
    idx = (y**2 + z**2 <= maxtip**2 / 4)
    # ic(oi, idx.sum() / len(idx))

    #
    # w, x, y, z = w[idx], x[idx], y[idx], z[idx]
    idx = y**2 + z**2 < maxtip**2 / 4
    q = h.normQ(th.stack([w.reshape(-1), y.reshape(-1), z.reshape(-1), x.reshape(-1)], dim=1))
    q = q[idx].to(device)
    # ic(q.shape, (2 * n2 + 1)**2 * n1, n2, n1)
    # d = h.norm(q[None] - q[:, None])
    # d.fill_diagonal_(9e9)
    # dclosest = d.min(1).values
    # print(th.quantile(dclosest, th.linspace(0, 1.0, 7)))
    x = h.quat_to_xform(q)
    return x

def bounding_sphere(xyz):
    cen = xyz.mean(0)
    rad = math.sqrt(h.norm2(xyz - cen).max())
    # import ipd.samp.sampling_cuda
    # cen, rad = ipd.samp.sampling_cuda.welzl_bounding_sphere(xyz.cpu().to(th.float32))
    # cen = cen.to(xyz.device).to(xyz.dtype)
    return cen, rad

# def randxform_small_cuda(
#     shape,
#     cartmean=0,
#     cartsd=0.1,
#     cart_uniform=False,
#     orisd=0.01,
#     dtype=th.float32,
#     device='cuda',
#     seed=-1,
#     nthread=256,
#     gentype='curandStatePhilox4_32_10_t',
# ):
#     if isinstance(shape, int): shape = (shape, )
#     n = int(th.prod(th.tensor(shape)).item())
#     if isinstance(cartmean, (float, int)): cartmean = [cartmean] * 3
#     cartmean = th.as_tensor(cartmean, dtype=dtype, device='cuda')
#     if seed < 0: seed = int(th.randint(0, 2**62, (1, ), dtype=th.int64))
#
#     if orisd > 0.4: raise ValueError('orisd must be less than 0.4')
#     coeff = [
#         1.7849621e-01, 2.7500242e-01, 2.1911124e-03, -1.7134663e+00, 5.3797978e-01, 1.5170078e+01,
#         3.3340999e-01, -5.5727810e+01, -1.0274599e+01, 9.9600327e+01, 2.8866692e+01, -8.5288895e+01,
#         -3.1181292e+01, 2.8087727e+01, 1.1939758e+01
#     ]
#     std2qh = np.polynomial.Polynomial(coeff, domain=[0., 0.4326], window=[-1., 1.], symbol='x')
#     print(
#         ipd.dev.Bunch(
#             n=n,
#             cartmean=cartmean,
#             cart_sd=cartsd,
#             # cart_uniform=cart_uniform,
#             quat_height=std2qh(orisd),
#             orinormal=True,
#             nthread=nthread,
#             seed=seed,
#             dtype=str(dtype),
#             gentype=gentype,
#             )
#         )
#     # return None
#     x = _sampling.rand_xform(
#         n=n,
#         cartmean=cartmean,
#         cart_sd=cartsd,
#         # cart_uniform=cart_uniform,
#         quat_height=std2qh(orisd),
#         orinormal=True,
#         nthread=nthread,
#         seed=seed,
#         dtype=str(dtype),
#         gentype=gentype)
#     return x.reshape(shape + (4, 4)).to(device)
