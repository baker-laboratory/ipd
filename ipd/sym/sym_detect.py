from dataclasses import dataclass

import numpy as np
import xarray as xr
import ipd

@dataclass
class SymInfo:
    symid: str
    frames: np.ndarray
    symcen: np.ndarray
    symelem: xr.Dataset

def syminfo_from_frames(frames: np.ndarray, tol=1e-4) -> SymInfo:
    h, npth = ipd.homog.get_tensor_libraries_for(frames)
    assert frames.ndim == 3 and frames.shape[1:] == (4, 4)
    if len(frames) == 1: return SymInfo('C1', frames, None, None)
    selms = symelems_from_frames(frames)
    nfolds = set(selms.nfold.data)
    isconcurrent, symcen = h.lines_concurrent_isect(selms.cen, selms.axis)
    # ic(selms.nfold)
    if len(nfolds) == 1 and h.allclose(selms.hel, 0) and isconcurrent:
        # nf;, axs, ang, cen, hel = next(iter(selms.keys())), *(x[None] for x in (allaxs, allang, allcen, allhel))
        nf = int(selms.nfold[0])
        if all(selms.nfold == 0):
            return SymInfo('HELIX', frames, symcen, selms)
        elif len(selms.nfold) == 3 and h.allclose(selms.ang, np.pi) and h.allclose(selms.hel, 0) and isconcurrent:
            return SymInfo('D2', frames, symcen, selms)
        else:
            return SymInfo(f'C{nf}', frames, symcen, selms)
    elif len(selms) == 1:
        raise NotImplementedError('')
    elif isconcurrent and h.allclose(selms.hel, 0):
        if 2 in nfolds and 3 in nfolds:
            ax2, ax3 = (selms.axis[selms.nfold == i] for i in (2, 3))
            testang = h.angle(ax2, ax3)
            if h.allclose(testang, npth.pi / 2): return SymInfo('D3', frames, symcen, selms)
            if h.allclose(testang, 0.955316621): return SymInfo('T', frames, symcen, selms)
            if h.allclose(testang, 0.615479714): return SymInfo('O', frames, symcen, selms)
            if h.allclose(testang, 0.364863837): return SymInfo('I', frames, symcen, selms)
            ic(testang)
            ic(selms)
            assert 0, f'unknown sym with nfold {nfolds} testang {testang}'
        elif len(nfolds) == 2 and 2 in nfolds:
            assert h.allclose(h.angle(*selms.axis[:2]), npth.pi / 2)
            return SymInfo(f'D{max(nfolds)}', frames, symcen, selms)
        ic(selms)
        assert 0, 'unknown sym'
    else:
        return SymInfo('Unknown', frames, symcen, allaxs, allang, list(selms.keys()), allcen, allhel)

def symaxis_closest_to(frames, axis, cen=..., closeto=...):
    h, npth = ipd.homog.get_tensor_libraries_for(frames)
    ddkw = h.get_dtype_dev([frames, axis])
    if closeto is ...: closeto = h.normalized([2, 1, 9, 0], **ddkw)
    axis = h.xform(h.inv(frames[0]), h.normalized(axis), **ddkw)
    if cen is ...:
        cen = npth.zeros(axis.shape, **ddkw)
        cen[..., 3] = 1
    assert h.allclose(axis[..., 3], 0)
    assert h.allclose(cen[..., 3], 1)
    cen = h.xform(h.inv(frames[0]), cen, **ddkw)
    frame0, frames = frames[0], h.xform(h.inv(frames[0]), frames)
    assert h.allclose(0, frames[:, :3, 3])
    assert axis.shape == cen.shape
    origshape, axis, cen = axis.shape, axis.reshape(-1, axis.shape[-1]), cen.reshape(-1, cen.shape[-1])
    symaxes = h.xform(frames, axis)
    symcen = h.xform(frames, cen)
    # ic(symaxes.shape, symcen.shape)
    # TODO: maybe do a line/point distace w/cen?
    which = npth.argmax(h.dot(symaxes, closeto), axis=0)
    bestaxis = symaxes[which, npth.arange(len(which))].reshape(origshape)
    bestcen = symcen[which, npth.arange(len(which))].reshape(origshape)
    return h.xform(frame0, bestaxis), h.xform(frame0, bestcen)

def isint2ormore(n, tol):
    return n > 1.9 and ((n % 1.0 < tol) or (n % 1.0 > 1 - tol))

def symelems_from_frames(frames, tol=1e-4, **kw):
    """
    compute a non-redundant set of simple symmetry elements from homog transforms
    """
    h, npth = ipd.homog.get_tensor_libraries_for(frames)
    rel = h.xform(h.inv(frames[0]), frames)
    axis, ang, cen, hel = h.axis_angle_cen_hel(rel[1:])
    axis[ang < 0] *= -1
    ang[ang < 0] *= -1
    axis, cen, ang, hel = h.lineuniq(axis, cen, ang, hel, frames=rel)
    ok = np.ones(len(axis), dtype=bool)
    result = list()
    for ax, cn, an, hl in zip(axis, cen, ang, hel):
        # ic(ax, float(an))
        for ax2, cn2, an2, hl2 in zip(axis, cen, ang, hel):
            cond = [
                an > an2 + tol and isint2ormore(an / an2, tol),  # is int mul of other
                np.abs(hl2) < tol or isint2ormore(hl / hl2, tol),
                npth.abs(h.dot(ax, ax2)) > 1 - tol,
                h.point_line_dist_pa(cn2, cn, ax) < tol,  # same line as other
            ]
            if all(cond): break
        else:  # if loop was not broken, is not a duplicate symelem
            ax, cn, nf = h.xform(frames[0], ax), h.xform(frames[0], cn), 2 * np.pi / an
            nf = h.toint(nf) if isint2ormore(nf, tol) else npth.zeros(an.shape, dtype=int)
            fields = dict(nfold=('index', nf),
                          axis=(['index', 'xyzw'], ax.reshape(-1, 4)),
                          ang=('index', an),
                          cen=(('index', 'xyzw'), cn.reshape(-1, 4)),
                          hel=('index', hl))
            result.append(xr.Dataset(fields))
    result = xr.concat(result, 'index')
    result = result.set_coords('nfold')

    return result
