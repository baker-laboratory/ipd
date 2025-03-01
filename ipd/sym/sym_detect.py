from collections.abc import Iterable
from dataclasses import dataclass, field
import sys

import numpy as np
import xarray as xr
import ipd
import ipd.homog.hgeom as h

@dataclass
class SymInfo:
    symid: str
    frames: np.ndarray
    symcen: np.ndarray
    symelem: xr.Dataset
    helical: bool = None
    guess_symid: str = None
    axes_concurrent: bool = None
    axes_dists: np.ndarray = None
    provenance: ipd.Bunch = field(default_factory=ipd.Bunch)

    # debug: which tolerances caused rejection of sym
    tolerances: ipd.Tolerances = None
    tol_checks: dict = None

    # if constructed from coords
    rms: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    stub0: np.ndarray = None

    # if multichain asu
    t_number: int = 1
    asurms: np.ndarray = None
    seqmatch: np.ndarray = None
    asumatch: np.ndarray = None
    asuframes: np.ndarray = field(default_factory=lambda: np.eye(4)[None])
    allframes: np.ndarray = None

    # coords and multichain
    asustub: np.ndarray = None
    allstub: np.ndarray = None

    def __post_init__(self):
        if not self.guess_symid: self.guess_symid = self.symid
        if self.tolerances:
            self.tolerances = self.tolerances.copy()
            self.tol_checks = self.tolerances.check_history()
        self.allframes = self.frames

    @property
    def order(self):
        return len(self.frames)

    @property
    def pseudo_order(self):
        return len(self.frames) * len(self.asuframes)

    @property
    def multichain(self):
        return len(self.asuframes) > 1

    def __repr__(self):
        return syminfo_to_str(self)

def detect(thing, tol=None, order=None, **kw):
    tol = ipd.Tolerances(tol, **(symdetect_default_tolerances) | kw)
    if ipd.homog.is_tensor(thing) and ipd.homog.is_xform_stack(thing):
        return syminfo_from_frames(thing, tol=tol, **kw)
    if 'biotite' in sys.modules:
        from biotite.structure import AtomArray
        if order is not None and len(atoms) % order == 0 and isinstance(atoms, AtomArray):
            atoms = ipd.atom.split(atoms, order)
        elif order is None and isinstance(atoms, AtomArray):
            atoms = ipd.pdb.split(atoms, bychain=True)
        elif isinstance(atoms, Iterable) and all(isinstance(a, AtomArray) for a in atoms):
            return syminfo_from_atomslist(atoms, tol=tol, **kw)
    raise ValueError(f'cant detect symmetry on object {type(atoms)} order {order}')

symdetect_default_tolerances = dict(
    default=1e-1,
    angtol=1e-2,
    helical_shift=1,
    isect=1,
    dot_norm=0.04,
    misc_lineuniq=1,
    nfold=0.2,
    seqmatch=0.7,
    rms_fit=2,
)

def syminfo_from_atomslist(atomslist: 'list[biotite.structure.AtomArray]', tol=None, **kw) -> SymInfo:
    """
    get frames from list of AtomArrays via. sequence and rms alignment, then compute SymInfo
    """
    assert not ipd.atom.is_atoms(atomslist)
    if len(atomslist) == 1: return syminfo_from_frames(np.eye(4)[None])
    tol = ipd.Tolerances(tol, **(symdetect_default_tolerances | kw))
    frames, rms, match = ipd.atom.frames_by_seqaln_rmsfit(atomslist, **kw)
    syminfo = syminfo_from_frames(frames, tol=tol, **kw)
    syminfo.rms, syminfo.seqmatch = rms, match
    _add_multichain_info(syminfo, atomslist, frames, tol)
    _check_frames_with_asu_for_supersym(syminfo)
    return syminfo

def _add_multichain_info(si: SymInfo, atomslist, frames, tol):
    if len(np.unique(atomslist[0].chain_id)) > 1:
        asu = ipd.atom.split(atomslist[0])
        si.asuframes, si.asurms, si.asumatch = ipd.atom.frames_by_seqaln_rmsfit(asu, tol=tol)
        si.t_number = np.sum(si.asumatch > tol.seqmatch)
        si.stub0 = ipd.atom.stub(asu[0])
        si.asustub = h.xform(si.asuframes, si.stub0)
        si.allstub = h.xform(frames, si.asustub)
        si.allframes = h.xform(si.allstub, h.inv(si.stub0))
    else:
        si.asuframes, si.asurms, si.asumatch = np.eye(4)[None], np.zeros(1), np.ones(1)
        si.t_number = 1
        si.stub0 = ipd.atom.stub(atomslist[0])
        si.asustub = si.stub0[None]
        si.allstub = h.xform(frames, si.asustub)
        si.allframes = frames[:, None]
    assert si.allframes.shape == si.allstub.shape
    assert h.allclose(h.xform(si.allframes, si.stub0), si.allstub)

def _check_frames_with_asu_for_supersym(syminfo):
    if not syminfo.multichain or syminfo.axes_concurrent and syminfo.order > 30: return
    tol = syminfo.tolerances.copy().reset()
    jointframes = syminfo.allframes.reshape(-1, 4, 4)
    si = syminfo_from_frames(jointframes, tol=tol, lineuniq_method='mean')
    print(si)
    assert 0

def syminfo_from_frames(frames: np.ndarray, tol=None, **kw) -> SymInfo:
    """
    infer sym elems and overall symmetry, return as SymInfo object
    """
    tol = ipd.Tolerances(tol, **(symdetect_default_tolerances | kw))
    h, npth = ipd.homog.get_tensor_libraries_for(frames)
    assert ipd.homog.is_xform_stack(frames)
    if len(frames) == 1: return SymInfo('C1', frames, None, None)
    selms = symelems_from_frames(frames, tol=tol, **kw)
    nfolds = set(selms.nfold.data)
    axes_concurrent, symcen, axes_dists = h.lines_concurrent_isect(selms.cen, selms.axis, tol=tol.isect)
    helical = not all(selms.hel < tol.helical_shift)
    sym_info_args = dict(axes_concurrent=axes_concurrent, symcen=symcen, helical=helical, symelem=selms)
    sym_info_args |= dict(frames=frames, axes_dists=axes_dists, tolerances=tol)
    if len(nfolds) == 1 and not helical and axes_concurrent:
        if all(selms.nfold == 0):
            return SymInfo('HELIX', **sym_info_args)
        elif len(selms.nfold) == 3 and h.allclose(selms.ang, np.pi) and h.allclose(selms.hel, 0) and axes_concurrent:
            return SymInfo('D2', **sym_info_args)
        else:
            return SymInfo(f'C{int(selms.nfold[0])}', **sym_info_args)
    elif len(selms) == 1:
        raise NotImplementedError('')
    elif axes_concurrent and not helical:
        if 2 in nfolds and 3 in nfolds:
            ax2, ax3 = (selms.axis[selms.nfold == i] for i in (2, 3))
            testang = h.line_angle(ax2, ax3).min()
            if abs(testang - npth.pi / 2) < tol.cageang: return SymInfo('D3', **sym_info_args)
            if abs(testang - 0.955316621) < tol.cageang: return SymInfo('T', **sym_info_args)
            if abs(testang - 0.615479714) < tol.cageang: return SymInfo('O', **sym_info_args)
            if abs(testang - 0.364863837) < tol.cageang: return SymInfo('I', **sym_info_args)
            ic(testang)
            ic(selms)
            assert 0, f'unknown sym with nfold {nfolds} testang {testang}'
        elif len(nfolds) == 2 and 2 in nfolds:
            assert h.allclose(h.angle(*selms.axis[:2]), npth.pi / 2)
            return SymInfo(f'D{max(nfolds)}', **sym_info_args)
        ic(selms)
        assert 0, 'unknown sym'
    else:
        return SymInfo('Unknown', **sym_info_args)

def symelems_from_frames(frames, tol=None, **kw):
    """
    compute a non-redundant set of simple symmetry elements from homog transforms
    """
    h, npth = ipd.homog.get_tensor_libraries_for(frames)
    tol = ipd.dev.Tolerances(tol, **kw)
    rel = h.xform(h.inv(frames[0]), frames)
    axis, ang, cen, hel = h.axis_angle_cen_hel(rel[1:])
    axis[ang < 0] *= -1
    ang[ang < 0] *= -1
    axis, cen, ang, hel = h.symmetrically_unique_lines(axis, cen, ang, hel, frames=rel, tol=tol, **kw)
    ok = np.ones(len(axis), dtype=bool)
    result = list()
    for ax, cn, an, hl in zip(axis, cen, ang, hel):
        # ic(ax, float(an))
        for ax2, cn2, an2, hl2 in zip(axis, cen, ang, hel):
            cond = [
                not np.abs(an - an2) < tol.angle and _isintgt1(an / an2, tol),  # is int mul of other
                np.abs(hl2) < tol.helical_shift or _isintgt1(hl / hl2, tol),
                npth.abs(1 - h.dot(ax, ax2)) < tol.axistol,
                h.point_line_dist_pa(cn2, cn, ax) < tol.isect,  # same line as other
            ]
            if all(cond): break
        else:  # if loop was not broken, is not a duplicate symelem
            ax, cn, nf = h.xform(frames[0], ax), h.xform(frames[0], cn), 2 * np.pi / an
            nf = h.toint(nf) if _isintgt1(nf, tol) else npth.zeros(an.shape, dtype=int)
            fields = dict(nfold=('index', [nf]),
                          axis=(['index', 'xyzw'], ax.reshape(-1, 4)),
                          ang=('index', [an]),
                          cen=(('index', 'xyzw'), cn.reshape(-1, 4)),
                          hel=('index', [hl]))
            result.append(xr.Dataset(fields))
    result = xr.concat(result, 'index')
    result = result.set_coords('nfold')
    return result

def _isintgt1(n, tol):
    return n > 1.9 and min(n % 1.0, 1 - n%1) < tol.nfold

def depricated_symaxis_closest_to(frames, axis, cen=None, target=None):
    h, npth = ipd.homog.get_tensor_libraries_for(frames)
    ddkw = h.get_dtype_dev([frames, axis])
    if target is None: target = h.normalized([2, 1, 9, 0], **ddkw)
    axis = h.xform(h.inv(frames[0]), h.normalized(axis), **ddkw)
    if cen is None:
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
    # TODO: maybe do a line/point distace w/cen? h.line_line_diff_pa
    which = npth.argmax(h.dot(symaxes, target), axis=0)
    bestaxis = symaxes[which, npth.arange(len(which))].reshape(origshape)
    bestcen = symcen[which, npth.arange(len(which))].reshape(origshape)
    return h.xform(frame0, bestaxis), h.xform(frame0, bestcen)

def syminfo_to_str(self, verbose=True):
    npopt = np.get_printoptions()
    np.set_printoptions(precision=6, suppress=True)
    with ipd.dev.capture_stdio() as out:
        textmap = {'nfold': 'nf', '[': '', ']': '', '__REGEX__': False}
        tabes = []
        head = dict(symid=self.symid, guess=self.guess_symid, symcen=list(self.symcen.round(3))[:3])
        headtable = ipd.dev.make_table(dict(foo=head), key=False, textmap=textmap)
        setable = ipd.dev.make_table_dataset(self.symelem, textmap=textmap)
        asutable = ipd.dev.make_table(
            [[self.t_number, self.asurms, self.asumatch, self.asuframes.shape, self.allframes.shape]],
            header=['T', 'asu rms', 'asu seq match', 'asu frames', 'all frames'])
        geomtable = ipd.dev.make_table([[self.helical, self.axes_concurrent, self.axes_dists]],
                                       header=['helical', 'axes concurrent', 'axes dists'],
                                       textmap=textmap)
        checktable = ipd.dev.make_table(self.tol_checks, key='Geom Tests')
        tables = [[headtable], [geomtable], [asutable], [setable], [checktable]]
        if self.stub0 is not None:
            tables.append([rmstable])
            rmstable = ipd.dev.make_table([[self.seqmatch.min(), self.rms.max()]], header=['worst seq match', 'worst rms'])
        ipd.dev.print_table(tables, header=['SymInfo'])

    np.set_printoptions(npopt['precision'], suppress=npopt['suppress'])
    return out.read()
