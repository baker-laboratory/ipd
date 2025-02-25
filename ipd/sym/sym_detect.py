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

def syminfo_from_atomslist(atomslist: 'list[biotite.structure.AtomArray]', **kw) -> SymInfo:
    """
    get frames from list of AtomArrays via. sequence and rms alignment, then compute SymInfo
    """
    tol = ipd.dev.Tolerances(**kw)
    frames, rms = _atomlist_seqaln_rmsfit(atomslist, **kw)
    if np.any(rms > tol.rmstol): return None
    return syminfo_from_frames(frames, **kw)

def syminfo_from_frames(frames: np.ndarray, tol=1e-4, **kw) -> SymInfo:
    """
    infer sym elems and overall symmetry, return as SymInfo object
    """
    tol = ipd.dev.Tolerances(tol, **kw)
    h, npth = ipd.homog.get_tensor_libraries_for(frames)
    assert frames.ndim == 3 and frames.shape[1:] == (4, 4)
    if len(frames) == 1: return SymInfo('C1', frames, None, None)
    selms = symelems_from_frames(frames, tol=tol)
    nfolds = set(selms.nfold.data)
    isconcurrent, symcen = h.lines_concurrent_isect(selms.cen, selms.axis, tol=tol.isectol)
    helical = not h.allclose(selms.hel, 0, atol=tol.heltol)
    if len(nfolds) == 1 and not helical and isconcurrent:
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
    elif isconcurrent and not helical:
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
        return SymInfo('Unknown', frames, symcen, selms)

def symelems_from_frames(frames, tol=1e-4, **kw):
    """
    compute a non-redundant set of simple symmetry elements from homog transforms
    """
    h, npth = ipd.homog.get_tensor_libraries_for(frames)
    tol = ipd.dev.Tolerances(tol, **kw)
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
                an > an2 + tol.angtol and _isintgt1(an / an2, tol),  # is int mul of other
                np.abs(hl2) < tol.heltol or _isintgt1(hl / hl2, tol),
                npth.abs(h.dot(ax, ax2)) > 1 - tol.axistol,
                h.point_line_dist_pa(cn2, cn, ax) < tol.isectol,  # same line as other
            ]
            if all(cond): break
        else:  # if loop was not broken, is not a duplicate symelem
            ax, cn, nf = h.xform(frames[0], ax), h.xform(frames[0], cn), 2 * np.pi / an
            nf = h.toint(nf) if _isintgt1(nf, tol) else npth.zeros(an.shape, dtype=int)
            fields = dict(nfold=('index', nf),
                          axis=(['index', 'xyzw'], ax.reshape(-1, 4)),
                          ang=('index', an),
                          cen=(('index', 'xyzw'), cn.reshape(-1, 4)),
                          hel=('index', hl))
            result.append(xr.Dataset(fields))
    result = xr.concat(result, 'index')
    result = result.set_coords('nfold')
    return result

def _isintgt1(n, tol):
    return n > 1.9 and ((n % 1.0 < tol.nftol) or (n % 1.0 > 1 - tol.nftol))

def _align_atoms_seq(atoms1, atoms2):
    import biotite.structure as struc
    import biotite.sequence.align as align
    seq1 = struc.to_sequence(atoms1)[0][0]
    seq2 = struc.to_sequence(atoms2)[0][0]
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    alignment = align.align_optimal(seq1, seq2, matrix)
    return alignment[0]

def _atomlist_seqaln_rmsfit(atomslist, **kw):
    frames, rmsds = [np.eye(4)], [0]
    for i, atm in enumerate(atomslist[1:]):
        aln = _align_atoms_seq(atomslist[0], atm).trace
        xyz1 = atomslist[0].coord[aln[aln[:, 0] >= 0, 0]]
        xyz2 = atm.coord[aln[aln[:, 1] >= 0, 1]]
        rms, _, xfit = ipd.homog.hrmsfit(xyz1, xyz2)
        frames.append(xfit), rmsds.append(rms)
    return np.stack(frames), np.array(rms)

def depricated_symaxis_closest_to(frames, axis, cen=..., closeto=...):
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
