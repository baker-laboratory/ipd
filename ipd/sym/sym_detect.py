from collections.abc import Iterable
import sys
from typing import Union
import attrs
import numpy as np
import ipd
import ipd.homog.hgeom as h

def detect(
    thing: Union['np.ndarray', 'torch.Tensor', 'AtomArray', 'Iterable[AtomArray]'],
    tol: Union[ipd.Tolerances, float, None] = None,
    order: int = None,
    **kw,
) -> 'SymInfo':
    """
    detect symmetry from frames (N,4,4), Atomarray or Iterable[AtomArray]
    """
    tol = ipd.Tolerances(tol, **(symdetect_default_tolerances) | kw)
    if ipd.homog.is_tensor(thing) and ipd.homog.is_xform_stack(thing):
        return syminfo_from_frames(thing, tol=tol, **kw)
    if 'biotite' in sys.modules:
        from biotite.structure import AtomArray
        atoms = thing
        if not isinstance(atoms, AtomArray) and len(atoms) == 1: atoms = atoms[0]
        if order is not None and len(atoms) % order == 0 and isinstance(atoms, AtomArray):
            atoms = ipd.atom.split(atoms, order)
        elif order is None and isinstance(atoms, AtomArray):
            atoms = ipd.atom.split(atoms, bychain=True)
        if isinstance(atoms, Iterable) and all(isinstance(a, AtomArray) for a in atoms):
            return syminfo_from_atomslist(atoms, tol=tol, **kw)
    raise ValueError(f'cant detect symmetry on object {type(atoms)} order {order}')

@ipd.dev.subscriptable_for_attributes
# @ipd.dev.element_wise_operations
@attrs.define
class SymInfo:
    """
    Groups togethes various infomation about a symmetry, returned from ipd.sym.detect
    """
    symid: str
    frames: np.ndarray
    symcen: np.ndarray
    symelem: ipd.Bunch

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
        _syminfo_post_init(self)

    def __getattr__(self, name):
        try:
            return self.symelem[name]
        except KeyError:
            pass
        raise AttributeError(f'SymInfo has no attribute {name}')

    def __repr__(self):
        return syminfo_to_str(self)

symdetect_default_tolerances = dict(
    default=1e-1,
    angle=3e-2,
    helical_shift=1,
    isect=2,
    dot_norm=0.04,
    misc_lineuniq=1,
    nfold=0.3,
    seqmatch=0.7,
    rms_fit=2,
    cageang=5e-2,
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
    cageang=1e-4,
)

def syminfo_from_atomslist(atomslist: 'list[biotite.structure.AtomArray]', **kw) -> SymInfo:
    """
    get frames from list of AtomArrays via. sequence and rms alignment, then compute SymInfo
    """
    assert not ipd.atom.is_atoms(atomslist)
    if len(atomslist) == 1: return syminfo_from_frames(np.eye(4)[None])
    tol = kw['tol'] = ipd.Tolerances(**(symdetect_default_tolerances | kw))
    components = ipd.atom.find_frames_by_seqaln_rmsfit(atomslist, maxsub=60, **kw)
    components.remove_small_chains()
    results = []
    for i, frames, match, rms in components.enumerate('frames seqmatch rmsd'):
        tol.reset()
        syminfo = syminfo_from_frames(frames, tol=tol)
        syminfo.rms, syminfo.seqmatch = rms, match
        _syminfo_add_atoms_info(syminfo, atomslist, frames, tol)
        results.append(syminfo)
    return check_sym_combinations(results)

def syminfo_from_frames(frames: np.ndarray, **kw) -> SymInfo:
    """
    infer sym elems and overall symmetry, return as SymInfo object
    """
    tol = kw['tol'] = ipd.Tolerances(**(symdetect_default_tolerances | kw))
    h, npth = ipd.homog.get_tensor_libraries_for(frames)
    assert ipd.homog.is_xform_stack(frames)
    if len(frames) == 1: return SymInfo('C1', frames, None, None)
    se = symelems_from_frames(frames, **kw)
    nfolds = set(int(nf) for nf in se.nfold)
    is_point, symcen, axes_dists = h.lines_concurrent_isect(se.cen, se.axis, tol=tol.isect)
    is_helical = not np.all(np.abs(se.hel) < tol.helical_shift)
    sym_info_args = dict(is_point=is_point, symcen=symcen, is_helical=is_helical, symelem=se)
    sym_info_args |= dict(frames=frames, axes_dists=axes_dists, tolerances=tol)
    if is_point: sym_info_args |= dict(is_1d=False, is_2d=False, is_3d=False)
    ic(is_point, is_helical, nfolds)
    ic(np.abs(se.ang - np.pi), tol.angle)
    ipd.print_table(se)
    if len(nfolds) == 1 and not is_helical and is_point:
        if all(se.nfold == 0):
            return SymInfo('HELIX', **sym_info_args)
        elif len(se.nfold) >= 3 and np.all(np.abs(se.ang - np.pi) < tol.angle):
            return SymInfo('D2', **sym_info_args)
        else:
            return SymInfo(f'C{int(se.nfold[0])}', **sym_info_args)
    elif len(se) == 1:
        raise NotImplementedError('')
    elif is_point and not is_helical:
        if 2 in nfolds and 3 in nfolds:
            ax2, ax3 = (se.axis[se.nfold == i] for i in (2, 3))
            testang = h.line_angle(ax2[None], ax3[:, None]).min()
            magicang = ipd.sym.magic_angle_DTOI
            ic(testang, magicang)
            if abs(testang - magicang.D) < tol.cageang: return SymInfo('D3', **sym_info_args)
            if abs(testang - magicang.T) < tol.cageang: return SymInfo('T', **sym_info_args)
            if abs(testang - magicang.O) < tol.cageang: return SymInfo('O', **sym_info_args)
            if abs(testang - magicang.I) < tol.cageang: return SymInfo('I', **sym_info_args)
            assert 0, f'unknown sym with nfold {nfolds} testang {testang} valid are: {magicang}'
        elif len(nfolds) == 2 and 2 in nfolds:
            ipd.dev.print_table(se)
            not2_vs_2ang = h.line_angle(se.axis[se.nfold == 2], se.axis[se.nfold != 2])
            assert np.all(np.abs(not2_vs_2ang - np.pi / 2) < tol.line_angle)
            return SymInfo(f'D{max(nfolds)}', **sym_info_args)
        assert 0, 'unknown sym'
    else:
        return SymInfo('Unknown', **sym_info_args)

def symelems_from_frames(frames, **kw):
    """
    compute a non-redundant set of simple symmetry elements from homog transforms
    """

    h, npth = ipd.homog.get_tensor_libraries_for(frames)
    tol = kw['tol'] = ipd.Tolerances(**(symdetect_default_tolerances | kw))
    rel = h.xformx(h.inv(frames[0]), frames)
    # if allbyall: rel = h.xformx(h.inv(frames), frames, outerprod=True, uppertri=1)
    axis, ang, cen, hel = h.axis_angle_cen_hel(rel[1:])
    axis, cen, ang, hel = h.unique_symaxes(axis, cen, ang, hel, frames=rel, debug=0, **kw)
    ok = np.ones(len(axis), dtype=bool)
    result = ipd.Bunch(nfold=[], axis=[], ang=[], cen=[], hel=[])
    for ax, cn, an, hl in zip(axis, cen, ang, hel):
        for ax2, cn2, an2, hl2 in zip(axis, cen, ang, hel):
            cond = [
                np.abs(an - an2) > tol.angle,
                _isintgt1(an / an2, tol),  # is int mul of other
                np.abs(hl2) < tol.helical_shift or _isintgt1(hl / hl2, tol),
                npth.abs(1 - np.abs(h.dot(ax, ax2))) < tol.axistol,
                h.point_line_dist_pa(cn2, cn, ax) < tol.isect,  # same line as other
            ]
            if all(cond): break
        else:  # if loop was not broken, is not a duplicate symelem
            ax, cn, nf = h.xform(frames[0], ax), h.xform(frames[0], cn), 2 * np.pi / an
            nf = h.toint(nf) if _isintgt1(nf, tol) else npth.zeros(an.shape, dtype=int)
            result.valwise.append(nfold=nf, axis=ax, ang=an, cen=cn, hel=hl)
        print('', flush=True)
    result = result.mapwise(np.array)
    assert result.nfold.ndim == 1 and result.axis.ndim == 2
    return result

def _isintgt1(nfold, tol):
    if 6 < nfold < 999: return True
    return nfold > 1.9 and min(nfold % 1.0, 1 - nfold%1) < tol.nfold

def syminfo_get_origin(sinfo):
    """get the symmetry origin as a frame that aligns symaxes to canonical ant translates to symcen"""
    nf = sinfo.unique_nfold
    if not sinfo.is_point:
        return syminfo_get_origin_nonpoint(sel)
    elif sinfo.is_cyclic:
        origin = h.align([0, 0, 1], sinfo.axis[0]) @ h.trans(sinfo.symcen)
    elif sinfo.is_dihedral:
        ax2, axx = sinfo.nfaxis[2][0], sinfo.nfaxis[nf[1 % len(nf)]][0]
        if sinfo.symid == 'D2': axx = sinfo.nfaxis[2][-1]
        origin = h.align2([0, 0, 1], [1, 0, 0], ax2, axx) @ h.trans(sinfo.symcen)
    else:
        assert 2 in nf and 3 in nf
        assert sinfo.symid in 'T I O D3'.split()
        origax2, origax3 = ipd.sym.axes(sinfo.symid)[2], ipd.sym.axes(sinfo.symid)[3]
        ax2, ax3 = sinfo.nfaxis[2][0], sinfo.nfaxis[3][0]
        origin = h.align2(origax2, origax3, ax2, ax3) @ h.trans(sinfo.symcen)
        assert h.valid44(origin)
    return origin, h.inv(origin)

def syminfo_to_str(sinfo, verbose=True):
    """returns text tables with the data in this SymInfo"""
    npopt = np.get_printoptions()
    np.set_printoptions(precision=6, suppress=True)
    textmap = {'nfold': 'nf', '__REGEX__': False}  # , '[': '', ']': '
    kw = dict(justify='right', title_justify='right', caption_justify='right', textmap=textmap)
    if sinfo.symid == 'C1': return 'SymInfo(C1)'
    with ipd.dev.capture_stdio() as out:
        symcen = '\n'.join(f'{x:7.3f}' for x in sinfo.symcen.reshape(4)[:3])
        head = dict(symid=sinfo.symid, guess=sinfo.guess_symid, symcen=symcen, origin=sinfo.origin)
        headtable = ipd.dev.make_table(dict(foo=head), key=False, strip=False, **kw)
        setable = ipd.dev.make_table_dataset(sinfo.symelem, **kw)
        asutable = ipd.dev.make_table(
            [[sinfo.t_number, sinfo.asurms, sinfo.asumatch, sinfo.asuframes.shape, sinfo.allframes.shape]],
            header=['T', 'asu rms', 'asu seq match', 'asu frames', 'all frames'])
        geomtable = ipd.dev.make_table([[sinfo.is_helical, sinfo.is_point, sinfo.axes_dists]],
                                       header=['is_helical', 'axes concurrent', 'axes dists'],
                                       **kw)
        checktable = ipd.dev.make_table(sinfo.tol_checks, key='Geom Tests')
        tables = [[headtable], [geomtable], [asutable], [setable], [checktable]]
        if sinfo.stub0 is not None:
            rmstable = ipd.dev.make_table([[sinfo.seqmatch.min(), sinfo.rms.max()]],
                                          header=['worst seq match', 'worst rms'])
            tables.append([rmstable])
        ipd.dev.print_table(tables, header=['SymInfo'])
    np.set_printoptions(npopt['precision'], suppress=npopt['suppress'])
    return out.read()

def _syminfo_post_init(sinfo: SymInfo) -> None:
    if not sinfo.guess_symid: sinfo.guess_symid = sinfo.symid
    if sinfo.tolerances:
        sinfo.tolerances = sinfo.tolerances.copy()
        sinfo.tol_checks = sinfo.tolerances.check_history()
    sinfo.allframes = sinfo.frames
    sinfo.order = len(sinfo.frames)
    sinfo.pseudo_order = len(sinfo.frames) * len(sinfo.asuframes)
    sinfo.is_multichain = len(sinfo.asuframes) > 1
    sinfo.unique_nfold = nf = []
    if sinfo.order > 1:
        sinfo.unique_nfold = nf = list(sorted(np.unique(sinfo.nfold)))
    sinfo.is_cyclic = len(nf) < 2 and not sinfo.is_helical and not sinfo.symid == 'D2'
    sinfo.is_dihedral = len(nf) == 2 and 2 in nf and sinfo.order // nf[1] == 2 or sinfo.symid == 'D2'
    sinfo.nfaxis = {int(nf): sinfo.axis[sinfo.nfold == nf] for nf in sinfo.unique_nfold}
    if sinfo.is_point:
        sinfo.symcen = sinfo.symcen.reshape(4)
        sinfo.origin, sinfo.toorigin = syminfo_get_origin(sinfo)
    else:
        sinfo.symcen = h.point([np.nan, np.nan, np.nan])
        sinfo.origin = sinfo.toorigin = np.eye(4)

def _syminfo_add_atoms_info(sinfo: SymInfo, atomslist, frames, tol) -> None:
    if sinfo.is_multichain:
        assert 0, 'sould now be handled my find_frames_by_seqaln_rmsfit'
        asu = ipd.atom.split(atomslist[0])
        sinfo.asuframes, sinfo.asurms, sinfo.asumatch = ipd.atom.frames_by_seqaln_rmsfit(asu, tol=tol)
        sinfo.t_number = np.sum(sinfo.asumatch > tol.seqmatch)
        sinfo.stub0 = ipd.atom.stub(asu[0])
        sinfo.asustub = h.xform(sinfo.asuframes, sinfo.stub0)
        sinfo.allstub = h.xform(frames, sinfo.asustub)
        sinfo.allframes = h.xform(sinfo.allstub, h.inv(sinfo.stub0))
    else:
        sinfo.asuframes, sinfo.asurms, sinfo.asumatch = np.eye(4)[None], np.zeros(1), np.ones(1)
        sinfo.t_number = 1
        sinfo.stub0 = ipd.atom.stub(atomslist[0])
        sinfo.asustub = sinfo.stub0[None]
        sinfo.allstub = h.xform(frames, sinfo.asustub)
        sinfo.allframes = frames[:, None]
    assert sinfo.allframes.shape == sinfo.allstub.shape
    assert h.allclose(h.xform(sinfo.allframes, sinfo.stub0), sinfo.allstub)

def check_sym_combinations(syminfos):
    if len(syminfos) == 1: return syminfos[0]
    assert 0, 'sym combinations not yet supported'
    tol = syminfos[0].tolerances.copy().reset()
    frames, stubs = list(), list()
    for sinfo in syminfos:
        # ipd.print_table(sinfo.symelem)
        frames.append(sinfo.frames)
        stubs.append(h.xform(sinfo.frames, sinfo.stub0))
        ic(sinfo.frames.shape)
    frames = np.concatenate(frames)
    frames = np.concatenate(stubs)
    ipd.showme(stubs, xyzscale=10, weight=10, spheres=4)
