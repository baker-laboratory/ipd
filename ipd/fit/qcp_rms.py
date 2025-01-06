"""This module provides a PyTorch interface to the QCP RMSD algorithm.

The CUDA implementation is very fast, can compute > 100 million RMSDs
per second on a single GPU.
"""
from ipd.lazy_import import lazyimport

th = lazyimport('torch')

import math

from numba import cuda

import ipd

_rms = ipd.dev.lazyimport('ipd.fit.qcp_rms_cuda')

def rmsd(xyz1, xyz2, getfit=False, nthread=128, usenumba=False):
    """Compute RMSD of xyz1 to xyz2.

    Can  be AxNx3, BxNx3
    Args:
        getfit: if True, return xforms
    """
    if xyz1.shape[-1] not in (3, 4) or xyz1.ndim not in (2, 3):
        raise ValueError(f"Unsupported shape: {xyz1.shape}")
    if xyz2.shape[-1] not in (3, 4) or xyz2.ndim not in (2, 3):
        raise ValueError(f"Unsupported shape: {xyz2.shape}")
    pad1, pad2 = False, False
    if xyz1.ndim == 2:
        xyz1 = xyz1[None]
        pad1 = True
    if xyz2.ndim == 2:
        xyz2 = xyz2[None]
        pad2 = True
    assert xyz1.dtype == xyz2.dtype
    assert xyz1.device == xyz2.device
    if xyz1.shape[-1] == 4: xyz1 = xyz1[:, :, :3]
    if xyz2.shape[-1] == 4: xyz2 = xyz2[:, :, :3]
    a, b, c1, c2, iprod, E0 = calc_iprod_E0(xyz1, xyz2)
    # ic(iprod.shape)
    # ic(E0.shape)

    if xyz1.device.type == 'cuda':
        if usenumba:
            rms, xfit = numba_qcp_rmsd_cuda_fixlen(iprod, E0, xyz1.shape[1], getfit, nthread)
        else:
            rms, xfit = _rms.qcp_rmsd_cuda_fixlen(iprod, E0, xyz1.shape[1], getfit, nthread)
    elif xyz1.dtype == th.float32:
        rms, xfit = _rms.qcp_rmsd_raw_vec_f4(iprod, E0, th.ones_like(E0) * xyz1.shape[1], getfit)
    elif xyz1.dtype == th.float64:
        rms, xfit = _rms.qcp_rmsd_raw_vec_f8(iprod, E0, th.ones_like(E0) * xyz1.shape[1], getfit)
    else:
        raise ValueError(f'bad dtype {xyz1.dtype} or dev {xyz1.device}')
    rms = rms.reshape(len(a), len(b))
    if getfit:
        xfit = xfit.reshape(len(a), len(b), 4, 4)
        # ic(xfit[..., :3, :3].shape, c1[:, None, :, None].shape, c2[None, :, None, :].shape)
        # ic(th.matmul(xfit[..., :3, :3], -c1[:, None, :, None])[...,0].shape)
        # ic(th.matmul(xfit[..., :3, :3], -c1[:, None, :, None] + c2[None, :, :]).shape)
        xfit[..., :3, 3] = th.matmul(xfit[..., :3, :3], -c1[:, None, :, None])[..., 0] + c2[None, :, :]
        xfit[..., 3, 3] = 1

    if pad1: rms = rms[0]
    if pad2: rms = rms[..., 0]
    if getfit: return rms, xfit
    return rms

def calc_iprod_E0(xyz1, xyz2):
    a = xyz1.clone()
    b = xyz2.clone()
    if a.ndim != 3 or b.ndim != 3: raise ValueError("ndim must be 3")
    if a.shape[1] != b.shape[1]: raise ValueError("mismatched shape[1]")
    c1 = a.mean(1)
    c2 = b.mean(1)
    a -= c1.unsqueeze(1)
    b -= c2.unsqueeze(1)
    a2 = th.transpose(a.unsqueeze(1), -1, -2)
    b2 = b.unsqueeze(0)
    iprod = th.matmul(a.unsqueeze(1).transpose(-1, -2), b.unsqueeze(0)).view(-1, 3, 3)
    E0 = (a2.square().sum((2, 3)) + b2.square().sum((2, 3))).view(-1) / 2.0
    return a, b, c1, c2, iprod, E0

def numba_qcp_rmsd_cuda_fixlen(iprod, E0, npts, getfit, nthread):
    rms = th.empty_like(E0)
    xfit = th.empty((len(E0) if getfit else 0, 4, 4), dtype=E0.dtype, device=E0.device)
    numba_kernel_qcp_raw[len(iprod), nthread](rms, xfit, iprod, E0, npts, getfit)  # type: ignore
    return rms, xfit

@cuda.jit('f4( f4[:,:], f4[:,:], f4, f4, b1 )', cache=False, device=True)
def numba_device_calc_rms_rot(rot, iprod, E0, npts, calcrot=False):
    oldg = 0.0
    evecprec = 1e-3
    evalprec = 1e-6

    Sxx = iprod[0, 0]
    Sxy = iprod[0, 1]
    Sxz = iprod[0, 2]
    Syx = iprod[1, 0]
    Syy = iprod[1, 1]
    Syz = iprod[1, 2]
    Szx = iprod[2, 0]
    Szy = iprod[2, 1]
    Szz = iprod[2, 2]

    Sxx2 = Sxx * Sxx
    Syy2 = Syy * Syy
    Szz2 = Szz * Szz

    Sxy2 = Sxy * Sxy
    Syz2 = Syz * Syz
    Sxz2 = Sxz * Sxz

    Syx2 = Syx * Syx
    Szy2 = Szy * Szy
    Szx2 = Szx * Szx

    SyzSzymSyySzz2 = 2.0 * (Syz*Szy - Syy*Szz)
    Sxx2Syy2Szz2Syz2Szy2 = Syy2 + Szz2 - Sxx2 + Syz2 + Szy2

    C2 = -2.0 * (Sxx2+Syy2+Szz2+Sxy2+Syx2+Sxz2+Szx2+Syz2+Szy2)
    C1 = 8.0 * (Sxx*Syz*Szy + Syy*Szx*Sxz + Szz*Sxy*Syx - Sxx*Syy*Szz - Syz*Szx*Sxy - Szy*Syx*Sxz)

    SxzpSzx = Sxz + Szx
    SyzpSzy = Syz + Szy
    SxypSyx = Sxy + Syx
    SyzmSzy = Syz - Szy
    SxzmSzx = Sxz - Szx
    SxymSyx = Sxy - Syx
    SxxpSyy = Sxx + Syy
    SxxmSyy = Sxx - Syy
    Sxy2Sxz2Syx2Szx2 = Sxy2 + Sxz2 - Syx2 - Szx2

    # print()
    # print("numba", SxzpSzx, SyzpSzy, SxypSyx, SyzmSzy, SxzmSzx, SxymSyx, SxxpSyy, SxxmSyy)

    C0 = (Sxy2Sxz2Syx2Szx2*Sxy2Sxz2Syx2Szx2 + (Sxx2Syy2Szz2Syz2Szy2+SyzSzymSyySzz2) *
          (Sxx2Syy2Szz2Syz2Szy2-SyzSzymSyySzz2) + (-(SxzpSzx) * (SyzmSzy) + (SxymSyx) *
                                                   (SxxmSyy-Szz)) * (-(SxzmSzx) * (SyzpSzy) + (SxymSyx) * (SxxmSyy+Szz)) +
          (-(SxzpSzx) * (SyzpSzy) - (SxypSyx) * (SxxpSyy-Szz)) * (-(SxzmSzx) * (SyzmSzy) - (SxypSyx) * (SxxpSyy+Szz)) +
          (+(SxypSyx) * (SyzpSzy) + (SxzpSzx) * (SxxmSyy+Szz)) * (-(SxymSyx) * (SyzmSzy) + (SxzpSzx) * (SxxpSyy+Szz)) +
          (+(SxypSyx) * (SyzmSzy) + (SxzmSzx) * (SxxmSyy-Szz)) * (-(SxymSyx) * (SyzpSzy) + (SxzmSzx) * (SxxpSyy-Szz)))
    # print(C0, C1, C2, E0)
    # Newton-Raphson
    mxEigenV = E0
    for i in range(50):
        oldg = mxEigenV
        x2 = mxEigenV * mxEigenV
        b = (x2+C2) * mxEigenV
        a = b + C1
        delta = ((a*mxEigenV + C0) / (2.0*x2*mxEigenV + b + a))
        mxEigenV -= delta
        # printf("\n diff[%3d]: %16g %16g %16g", i, mxEigenV - oldg,
        # evalprec*mxEigenV, mxEigenV)
        if (math.fabs(mxEigenV - oldg) < math.fabs(evalprec * mxEigenV)): break

    # print(mxEigenV)

    # if (i == 50)
    # fprintf(stderr, "\nMore than %d iterations needed!\n", i)

    # the math.fabs() is to guard against extremely small, but *negative* numbers due
    # to floating point error
    rms = math.sqrt(math.fabs(2.0 * (E0-mxEigenV) / npts))

    # print(rms, npts)
    # printf("\n\n %16g %16g %16g \n", rms, E0, 2.0 * (E0 - mxEigenV)/npts)

    if not calcrot: return rms

    a11 = SxxpSyy + Szz - mxEigenV
    a12 = SyzmSzy
    a13 = -SxzmSzx
    a14 = SxymSyx
    a21 = SyzmSzy
    a22 = SxxmSyy - Szz - mxEigenV
    a23 = SxypSyx
    a24 = SxzpSzx
    a31 = a13
    a32 = a23
    a33 = Syy - Sxx - Szz - mxEigenV
    a34 = SyzpSzy
    a41 = a14
    a42 = a24
    a43 = a34
    a44 = Szz - SxxpSyy - mxEigenV
    a3344_4334 = a33*a44 - a43*a34
    a3244_4234 = a32*a44 - a42*a34
    a3243_4233 = a32*a43 - a42*a33
    a3143_4133 = a31*a43 - a41*a33
    a3144_4134 = a31*a44 - a41*a34
    a3142_4132 = a31*a42 - a41*a32
    q1 = a22*a3344_4334 - a23*a3244_4234 + a24*a3243_4233
    q2 = -a21 * a3344_4334 + a23*a3144_4134 - a24*a3143_4133
    q3 = a21*a3244_4234 - a22*a3144_4134 + a24*a3142_4132
    q4 = -a21 * a3243_4233 + a22*a3143_4133 - a23*a3142_4132

    qsqr = q1*q1 + q2*q2 + q3*q3 + q4*q4

    # The following code tries to calculate another column in the adjoint matrix
    #   when the norm of the current column is too small. Usually this block will
    #   never be activated.  To be absolutely safe this should be uncommented,
    #   but it is most likely unnecessary.

    if (qsqr < evecprec):
        q1 = a12*a3344_4334 - a13*a3244_4234 + a14*a3243_4233
        q2 = -a11 * a3344_4334 + a13*a3144_4134 - a14*a3143_4133
        q3 = a11*a3244_4234 - a12*a3144_4134 + a14*a3142_4132
        q4 = -a11 * a3243_4233 + a12*a3143_4133 - a13*a3142_4132
        qsqr = q1*q1 + q2*q2 + q3*q3 + q4*q4

        if (qsqr < evecprec):
            a1324_1423 = a13*a24 - a14*a23
            a1224_1422 = a12*a24 - a14*a22
            a1223_1322 = a12*a23 - a13*a22
            a1124_1421 = a11*a24 - a14*a21
            a1123_1321 = a11*a23 - a13*a21
            a1122_1221 = a11*a22 - a12*a21

            q1 = a42*a1324_1423 - a43*a1224_1422 + a44*a1223_1322
            q2 = -a41 * a1324_1423 + a43*a1124_1421 - a44*a1123_1321
            q3 = a41*a1224_1422 - a42*a1124_1421 + a44*a1122_1221
            q4 = -a41 * a1223_1322 + a42*a1123_1321 - a43*a1122_1221
            qsqr = q1*q1 + q2*q2 + q3*q3 + q4*q4

            if (qsqr < evecprec):
                q1 = a32*a1324_1423 - a33*a1224_1422 + a34*a1223_1322
                q2 = -a31 * a1324_1423 + a33*a1124_1421 - a34*a1123_1321
                q3 = a31*a1224_1422 - a32*a1124_1421 + a34*a1122_1221
                q4 = -a31 * a1223_1322 + a32*a1123_1321 - a33*a1122_1221
                qsqr = q1*q1 + q2*q2 + q3*q3 + q4*q4

                if (qsqr < evecprec):
                    # if qsqr is still too small, return the identity matrix.
                    rot[0, 0] = rot[1, 1] = rot[2, 2] = 1.0
                    rot[0, 1] = rot[0, 2] = rot[1, 0] = rot[1, 2] = rot[2, 0] = rot[2, 1] = 0.0
                    return rms

    normq = math.sqrt(qsqr)
    q1 /= normq
    q2 /= normq
    q3 /= normq
    q4 /= normq

    a2 = q1 * q1
    x2 = q2 * q2
    y2 = q3 * q3
    z2 = q4 * q4

    xy = q2 * q3
    az = q1 * q4
    zx = q4 * q2
    ay = q1 * q3
    yz = q3 * q4
    ax = q1 * q2

    rot[0, 0] = a2 + x2 - y2 - z2
    rot[1, 0] = 2 * (xy+az)
    rot[2, 0] = 2 * (zx-ay)
    rot[0, 1] = 2 * (xy-az)
    rot[1, 1] = a2 - x2 + y2 - z2
    rot[2, 1] = 2 * (yz+ax)
    rot[0, 2] = 2 * (zx+ay)
    rot[1, 2] = 2 * (yz-ax)
    rot[2, 2] = a2 - x2 - y2 + z2

    return rms

@cuda.jit('void(f4[:],f4[:,:,:],f4[:,:,:],f4[:],f4,b1)', cache=False, fastmath=True)
def numba_kernel_qcp_raw(rms, rot, iprod, E0, npts, getfit):
    i = cuda.grid(1)  # type: ignore
    if i < E0.shape[0]:
        rms[i] = numba_device_calc_rms_rot(rot[i], iprod[i], E0[i], npts, getfit)

# def qcp_rms(xyz1, xyz2):
#     if xyz1.shape[-1] == 4: xyz1 = xyz1[:, :3].contiguous()
#     if xyz2.shape[-1] == 4: xyz2 = xyz2[:, :3].contiguous()
#     if xyz1.dtype == th.float32:
#         return _rms.qcp_rms_f4(xyz1, xyz2)
#     elif xyz1.dtype == th.float64:
#         return _rms.qcp_rms_f8(xyz1, xyz2)
#     else:
#         raise ValueError(f"Unsupported dtype: {xyz1.dtype}")
#
# def qcp_rms_vec(xyz1, xyz2):
#     if xyz1.shape[-1] == 4: xyz1 = xyz1[:, :, :3].contiguous()
#     if xyz2.shape[-1] == 4: xyz2 = xyz2[:, :, :3].contiguous()
#     if xyz1.dtype == th.float32:
#         return _rms.qcp_rms_vec_f4(xyz1, xyz2)
#     elif xyz1.dtype == th.float64:
#         return _rms.qcp_rms_vec_f8(xyz1, xyz2)
#     else:
#         raise ValueError(f"Unsupported dtype: {xyz1.dtype}")

def qcp_rms_align(xyz1, xyz2):
    if xyz1.shape[-1] == 4: xyz1 = xyz1[:, :3].contiguous()
    if xyz2.shape[-1] == 4: xyz2 = xyz2[:, :3].contiguous()
    if xyz1.dtype == th.float32:
        return _rms.qcp_rms_align_f4(xyz1, xyz2)
    elif xyz1.dtype == th.float64:
        return _rms.qcp_rms_align_f8(xyz1, xyz2)
    else:
        raise ValueError(f"Unsupported dtype: {xyz1.dtype}")

# def qcp_rmsd_kernel(iprod, E0, npts):
#     if xyz1.dtype == th.float32:
#         return _rms.qcp_rmsd_raw_vec_f4(iprod, E0, len)
#     elif xyz1.dtype == th.float64:
#         return _rms.qcp_rmsd_raw_vec_f8(iprod, E0, len)
#     else:
#         raise ValueError(f"Unsupported dtype: {xyz1.dtype}")
