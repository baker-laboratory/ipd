from collections.abc import Iterable
import sys
from typing import Union
import attrs
import numpy as np
import xarray as xr
import ipd
import ipd.homog.hgeom as h

def detect(
    thing: Union['np.ndarray', 'torch.Tensor', 'AtomArray', 'Iterable[AtomArray]'],
    tol: Union[ipd.Tolerances, float, None] = None,
    order: int = None,
    **kw,
):
    """
    detect symmetry from frames (N,4,4), Atomarray or Iterable[AtomArray]
    """
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

@attrs.define
class SymInfo:
    """
    Groups togethes various infomation about a symmetry, returned from ipd.sym.detect
    """
    symid: str
    frames: np.ndarray
    symcen: np.ndarray
    symelem: xr.Dataset

    guess_symid: str = None
    order: int = None
    t_number: int = 1
    pseudo_order: int = None
    is_helical: bool = None
    is_multichain: bool = None
    is_point: bool = None
    is_1d: bool = None
    is_2d: bool = None
    is_3d: bool = None
    is_cyclic: bool = None
    is_dihedral: bool = None
    provenance: ipd.Bunch = attrs.field(factory=ipd.Bunch)

    unique_nfold: list = None
    nfaxis: dict[int, np.ndarray] = None
    origin: np.ndarray = None
    toorigin: np.ndarray = None

    # debug: which tolerances caused rejection of sym
    axes_dists: np.ndarray = None
    tolerances: ipd.Tolerances = None
    tol_checks: dict = None

    # if constructed from coords
    rms: np.ndarray = attrs.field(factory=lambda: np.array([0.0]))
    stub0: np.ndarray = None

    # if is_multichain asu
    asurms: np.ndarray = None
    seqmatch: np.ndarray = None
    asumatch: np.ndarray = None
    asuframes: np.ndarray = attrs.field(factory=lambda: np.eye(4)[None])
    allframes: np.ndarray = None

    # coords and is_multichain
    asustub: np.ndarray = None
    allstub: np.ndarray = None

    def __attrs_post_init__(self):
        if not self.guess_symid: self.guess_symid = self.symid
        if self.tolerances:
            self.tolerances = self.tolerances.copy()
            self.tol_checks = self.tolerances.check_history()
        self.allframes = self.frames
        self.order = len(self.frames)
        self.pseudo_order = len(self.frames) * len(self.asuframes)
        self.is_multichain = len(self.asuframes) > 1
        self.unique_nfold = nf = []
        if self.order > 1:
            self.unique_nfold = nf = list(sorted(np.unique(self.nfold)))
        self.is_cyclic = len(nf) < 2 and not self.is_helical and not self.symid == 'D2'
        self.is_dihedral = len(nf) == 2 and 2 in nf and self.order // nf[1] == 2 or self.symid == 'D2'
        self.nfaxis = {int(nf): self.axis[self.nfold == nf] for nf in self.unique_nfold}
        if self.order > 1:
            self.symcen = self.symcen.reshape(4)
            self.origin, self.toorigin = syminfo_get_origin(self)

    def __getattr__(self, name):
        try:
            return self.symelem[name].data
        except KeyError:
            pass
        raise AttributeError(f'SymInfo has no attribute {name}')

    def __repr__(self):
        return syminfo_to_str(self)

symdetect_default_tolerances = dict(
    default=1e-1,
    angle=1e-2,
    helical_shift=1,
    isect=1,
    dot_norm=0.04,
    misc_lineuniq=1,
    nfold=0.2,
    seqmatch=0.7,
    rms_fit=2,
)
symdetect_ideal_tolerances = dict(
    default=1e-4,
    angle=1e-4,
    helical_shift=1e-4,
    isect=1e-4,
    dot_norm=1e-4,
    misc_lineuniq=1e-4,
    nfold=1e-4,
    seqmatch=0.99,
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
    if si.is_multichain:
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
    if not syminfo.is_multichain or syminfo.is_point and syminfo.order > 30: return
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
    se = symelems_from_frames(frames, tol=tol, **kw)
    nfolds = set(se.nfold.data)
    is_point, symcen, axes_dists = h.lines_concurrent_isect(se.cen, se.axis, tol=tol.isect)
    is_helical = not np.all(np.abs(se.hel) < tol.helical_shift)
    sym_info_args = dict(is_point=is_point, symcen=symcen, is_helical=is_helical, symelem=se)
    sym_info_args |= dict(frames=frames, axes_dists=axes_dists, tolerances=tol)
    if is_point: sym_info_args |= dict(is_1d=False, is_2d=False, is_3d=False)
    if len(nfolds) == 1 and not is_helical and is_point:
        if all(se.nfold == 0):
            return SymInfo('HELIX', **sym_info_args)
        elif all([
                is_point,
                len(se.nfold) >= 3,
                np.all(np.abs(se.ang - np.pi) < tol.angle),
                np.all(np.abs(se.hel) < tol.helical_shift)
        ]):
            return SymInfo('D2', **sym_info_args)
        else:
            return SymInfo(f'C{int(se.nfold[0])}', **sym_info_args)
    elif len(se) == 1:
        raise NotImplementedError('')
    elif is_point and not is_helical:
        if 2 in nfolds and 3 in nfolds:
            ax2, ax3 = (se.axis[se.nfold == i] for i in (2, 3))
            testang = h.line_angle(ax2, ax3).min()
            magicang = ipd.sym.magic_angle_DTOI
            if abs(testang - magicang.D) < tol.cageang: return SymInfo('D3', **sym_info_args)
            if abs(testang - magicang.T) < tol.cageang: return SymInfo('T', **sym_info_args)
            if abs(testang - magicang.O) < tol.cageang: return SymInfo('O', **sym_info_args)
            if abs(testang - magicang.I) < tol.cageang: return SymInfo('I', **sym_info_args)
            assert 0, f'unknown sym with nfold {nfolds} testang {testang}'
        elif len(nfolds) == 2 and 2 in nfolds:
            # ipd.dev.print_table(se)
            not2_vs_2ang = h.angle(se.axis[se.nfold == 2], se.axis[se.nfold != 2])
            assert np.all(np.abs(not2_vs_2ang - np.pi / 2) < tol.line_angle)
            return SymInfo(f'D{max(nfolds)}', **sym_info_args)
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

def syminfo_get_origin(si):
    """get the symmetry origin as a frame that aligns symaxes to canonical ant translates to symcen"""
    nf = si.unique_nfold
    if not si.is_point:
        return syminfo_get_origin_nonpoint(sel)
    elif si.is_cyclic:
        origin = h.align([0, 0, 1], si.axis[0]) @ h.trans(si.symcen)
    elif si.is_dihedral:
        ax2, axx = si.nfaxis[2][0], si.nfaxis[nf[1 % len(nf)]][0]
        if si.symid == 'D2': axx = si.nfaxis[2][-1]
        origin = h.align2([0, 0, 1], [1, 0, 0], ax2, axx) @ h.trans(si.symcen)
    else:
        # se = si.symelem.sel(index=si.nfold < 4)  # for DTIO, only need 2fold and 3fold
        # angbetweenaxes = h.line_angle(se.axis.data[None], se.axis.data[:, None])
        # magic = list(ipd.sym.magic_angle_DTOI.values())
        # ismagicangle = h.tensor_in(angbetweenaxes, magic, atol=si.tolerances.angle)
        # uniq = np.unique(ismagicangle)
        # uniq = uniq[uniq != -1]
        # magicang = uniq[0]
        # ic(magicang)
        # fitaxis = list(set([tuple(np.sort(x)) for x in np.where(ismagicangle >= 0)]))
        # ic(fitaxis)
        # bestpair = fitaxis[np.argmin(np.abs(magicang - angbetweenaxes[tuple(zip(*fitaxis))]))]
        assert 2 in nf and 3 in nf
        origax2, origax3 = ipd.sym.axes(si.symid)[2], ipd.sym.axes(si.symid)[3]
        ax2, ax3 = si.nfaxis[2][0], si.nfaxis[3][0]
        ic(si.axis)
        origin = h.align2(origax2, origax3, ax2, ax3) @ h.trans(si.symcen)
        assert h.valid44(origin)
    return origin, h.inv(origin)

def syminfo_to_str(si, verbose=True):
    """returns text tables with the data in this SymInfo"""
    npopt = np.get_printoptions()
    np.set_printoptions(precision=6, suppress=True)
    textmap = {'nfold': 'nf', '[': '', ']': '', '__REGEX__': False}
    targ = dict(justify='right', title_justify='right', caption_justify='right', textmap=textmap)
    if si.symid == 'C1': return 'SymInfo(C1)'
    with ipd.dev.capture_stdio() as out:
        symcen = '\n'.join(f'{x:7.3f}' for x in si.symcen.reshape(4)[:3])
        head = dict(symid=si.symid, guess=si.guess_symid, symcen=symcen, origin=si.origin)
        headtable = ipd.dev.make_table(dict(foo=head), key=False, strip=False, **targ)
        setable = ipd.dev.make_table_dataset(si.symelem, **targ)
        asutable = ipd.dev.make_table([[si.t_number, si.asurms, si.asumatch, si.asuframes.shape, si.allframes.shape]],
                                      header=['T', 'asu rms', 'asu seq match', 'asu frames', 'all frames'])
        geomtable = ipd.dev.make_table([[si.is_helical, si.is_point, si.axes_dists]],
                                       header=['is_helical', 'axes concurrent', 'axes dists'],
                                       **targ)
        checktable = ipd.dev.make_table(si.tol_checks, key='Geom Tests')
        tables = [[headtable], [geomtable], [asutable], [setable], [checktable]]
        if si.stub0 is not None:
            rmstable = ipd.dev.make_table([[si.seqmatch.min(), si.rms.max()]], header=['worst seq match', 'worst rms'])
            tables.append([rmstable])
        ipd.dev.print_table(tables, header=['SymInfo'])
    np.set_printoptions(npopt['precision'], suppress=npopt['suppress'])
    return out.read()
