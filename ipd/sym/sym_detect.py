from collections.abc import Iterable
import sys
import numpy as np
import ipd
import ipd.homog.hgeom as h

bs = ipd.lazyimport('biotite.structure')

def detect(
    thing,  #Union[ipd.Tensor, 'AtomArray', 'Iterable[bs.AtomArray]'],
    tol: ipd.Union[ipd.Tolerances, float, None] = None,
    order: int = 0,
    **kw,
) -> 'SymInfo':
    """
    Detect symmetry from frames (N, 4, 4), AtomArray, or an Iterable of AtomArray.

    Args:
        thing (Union[np.ndarray, torch.Tensor, AtomArray, Iterable[AtomArray]]):
            Input data representing the coordinate frames or atomic chains.
        tol (Union[ipd.Tolerances, float, None], optional):
            Tolerance for detecting symmetry. If None, defaults are used.
        order (int, optional):
            Expected order of symmetry. If None, order is inferred.
        **kw:
            Additional keyword arguments for tolerance settings.

    Returns:
        SymInfo: Object containing detected symmetry information.

    Raises:
        ValueError: If symmetry cannot be detected from the input data.

    Examples:
        >>> import numpy as np
        >>> from ipd.sym.sym_detect import detect
        >>> frames = np.eye(4)[None]  # Identity transformation
        >>> sym_info = detect(frames)
        >>> sym_info.symid
        'C1'
    """
    tol = ipd.Tolerances(tol, **(symdetect_default_tolerances) | kw)
    if ipd.homog.is_tensor(thing) and ipd.homog.is_xform_stack(thing):
        return syminfo_from_frames(thing, tol=tol, **kw)
    if 'biotite' in sys.modules:
        from biotite.structure import AtomArray
        atoms = thing
        if not isinstance(atoms, AtomArray) and len(atoms) == 1: atoms = atoms[0]
        if order and len(atoms) % order == 0 and isinstance(atoms, AtomArray):
            atoms = ipd.atom.split(atoms, order)
        elif not order and isinstance(atoms, AtomArray):
            atoms = ipd.atom.split(atoms, bychain=True)
        if isinstance(atoms, Iterable) and all(isinstance(a, AtomArray) for a in atoms):
            return syminfo_from_atomslist(atoms, tol=tol, **kw)
    raise ValueError(f'cant detect symmetry on object {type(thing)} order {order}')

@ipd.subscriptable_for_attributes
# @ipd.element_wise_operations
@ipd.struct
class SymInfo:
    """
    Contains information about detected symmetry, returned from `detect`.

    Attributes:
        symid (str): Symmetry identifier.
        frames (np.ndarray): Transformation frames.
        symcen (np.ndarray): Symmetry center coordinates.
        symelem (ipd.Bunch): Symmetry elements.
        guess_symid (str, optional): Guessed symmetry identifier.
        order (int, optional): Order of symmetry.
        t_number (int): Number of symmetry transformations.
        pseudo_order (int, optional): Pseudo-order of symmetry.
        has_translation (bool, optional): Whether the symmetry is helical.
        is_multichain (bool, optional): Whether symmetry spans multiple chains.
        is_point (bool, optional): Whether symmetry is point symmetry.
        is_1d (bool, optional): Whether symmetry is 1D.
        is_2d (bool, optional): Whether symmetry is 2D.
        is_3d (bool, optional): Whether symmetry is 3D.
        is_cyclic (bool, optional): Whether symmetry is cyclic.
        is_dihedral (bool, optional): Whether symmetry is dihedral.
        provenance (ipd.Bunch): Provenance information.

    Examples:
        >>> from ipd.sym.sym_detect import SymInfo
        >>> import numpy as np
        >>> frames = np.eye(4)[None]
        >>> sym_info = SymInfo('C1', frames, None, None)
        >>> sym_info.symid
        'C1'
    """
    symid: str
    frames: np.ndarray
    symcen: np.ndarray
    symelem: ipd.Bunch
    component: int = -1
    components: ipd.atom.Components = None
    asu_components: ipd.atom.Components = None

    guess_symid: str = None
    has_translation: bool = None
    is_point: bool = None
    is_1d: bool = None
    is_2d: bool = None
    is_3d: bool = None
    provenance: ipd.Bunch = ipd.field(ipd.Bunch)

    unique_nfold: list = None
    nfaxis: dict[int, np.ndarray] = None
    origin: np.ndarray = None
    toorigin: np.ndarray = None

    # debug: which tolerances caused rejection of sym
    axes_dists: np.ndarray = None
    tolerances: ipd.Tolerances = None
    tol_checks: dict = None

    # if constructed from coords
    rms: np.ndarray = ipd.field(lambda: np.array([0.0]))
    stub0: np.ndarray = None

    # if is_multichain asu
    asuframes: np.ndarray = ipd.field(lambda: np.eye(4)[None])
    allframes: np.ndarray = None

    # coords and is_multichain
    asustub: np.ndarray = None
    allstub: np.ndarray = None

    is_cyclic = property(lambda self: self.symid[0] == 'C')
    is_dihedral = property(lambda self: self.symid[0] == 'D')
    is_cage = property(lambda self: self.symid[0] in 'TIO')
    order = property(lambda self: len(self.frames))
    pseudo_order = property(lambda self: self.allframes.size // 16)
    t_number = property(lambda self: len(self.asuframes))
    is_multichain = property(lambda self: len(self.asuframes) > 1)

    def __post_init__(self):
        if not self.guess_symid: self.guess_symid = self.symid
        if self.tolerances:
            self.tolerances = self.tolerances.copy()
            self.tol_checks = self.tolerances.check_history()
        self.allframes = self.frames
        self.unique_nfold = nf = []
        if self.order > 1:
            self.unique_nfold = nf = list(sorted(np.unique(self.nfold)))
        # self.is_cyclic = len(nf) < 2 and not self.has_translation and not self.symid == 'D2'
        # self.is_dihedral = len(nf) == 2 and 2 in nf and self.order // nf[1] == 2 or self.symid == 'D2'
        self.nfaxis = {int(nf): self.axis[self.nfold == nf] for nf in self.unique_nfold}
        if self.is_point:
            self.symcen = self.symcen.reshape(4)
            self.origin, self.toorigin = syminfo_get_origin(self)
        else:
            self.symcen = h.point([np.nan, np.nan, np.nan])
            self.origin = self.toorigin = np.eye(4)

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
    seqmatch=0.5,
    matchsize=50,
    rms_fit=4,
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
    Generate symmetry information from a list of AtomArrays.

    Args:
        atomslist (list[biotite.structure.AtomArray]):
            List of AtomArray objects representing atomic chains.
        **kw:
            Additional keyword arguments for tolerance settings.

    Returns:
        SymInfo: Detected symmetry information.

    Examples:
        >>> from biotite.structure import AtomArray
        >>> from ipd.sym.sym_detect import syminfo_from_atomslist
        >>> atoms = ipd.atom.get('1a2n', chainlist=True)
        >>> sym_info = syminfo_from_atomslist(atoms)
    """
    assert not ipd.atom.is_atoms(atomslist)
    if len(atomslist) == 1: return syminfo_from_frames(np.eye(4)[None])
    tol = kw['tol'] = ipd.Tolerances(**(symdetect_default_tolerances | kw))
    components = ipd.atom.find_components_by_seqaln_rmsfit(atomslist, maxsub=60, **kw)
    # ipd.icv(len(components.atoms))
    # ipd.showme(components.atoms[0])
    # ipd.showme(components.atoms[1])
    # ipd.print_table(components.pick('seqmatch rmsd'))
    # for dat in components.intermediates_:
    # print(np.stack([dat.seqmatch, dat.rmsd]).T)
    # assert 0
    results = []
    for i, frames, match, rms in components.enumerate('frames seqmatch rmsd'):
        tol.reset()
        syminfo = syminfo_from_frames(frames, tol=tol)
        syminfo.component = i
        syminfo.components = components
        _syminfo_add_atoms_info(syminfo, atomslist, frames, tol)
        results.append(syminfo)
    return ipd.kwcall(kw, check_sym_combinations, results)

def syminfo_from_frames(frames: np.ndarray, **kw) -> SymInfo:
    """
    Infer symmetry elements and overall symmetry from transformation frames.

    Args:
        frames (np.ndarray):
            Transformation frames with shape (N, 4, 4).
        **kw:
            Additional keyword arguments for tolerance settings.

    Returns:
        SymInfo: Detected symmetry information.

    Examples:
        >>> import numpy as np
        >>> from ipd.sym.sym_detect import syminfo_from_frames
        >>> frames = np.eye(4)[None]
        >>> sym_info = syminfo_from_frames(frames)
        >>> sym_info.symid
        'C1'
    """
    tol = kw['tol'] = ipd.Tolerances(**(symdetect_default_tolerances | kw))
    h, npth = ipd.homog.get_tensor_libraries_for(frames)
    assert ipd.homog.is_xform_stack(frames)
    if len(frames) == 1: return SymInfo('C1', frames, None, None)
    se = symelems_from_frames(frames, **kw)
    nfolds = set(int(nf) for nf in se.nfold)
    is_point, symcen, axes_dists = h.lines_concurrent_isect(se.cen, se.axis, tol=tol.isect)
    has_translation = not np.all(np.abs(se.hel) < tol.helical_shift)
    sym_info_args = dict(is_point=is_point, symcen=symcen, has_translation=has_translation, symelem=se)
    sym_info_args |= dict(frames=frames, axes_dists=axes_dists, tolerances=tol)
    if is_point: sym_info_args |= dict(is_1d=False, is_2d=False, is_3d=False)
    if len(nfolds) == 1 and not has_translation and is_point:
        if len(se.nfold) >= 3 and np.all(np.abs(se.ang - np.pi) < tol.angle):
            return SymInfo('D2', **sym_info_args)
        else:
            return SymInfo(f'C{int(se.nfold[0])}', **sym_info_args)
    elif is_point and not has_translation:
        if 2 in nfolds and 3 in nfolds:
            ax2, ax3 = (se.axis[se.nfold == i] for i in (2, 3))
            testang = h.line_angle(ax2[None], ax3[:, None]).min()
            magicang = ipd.sym.magic_angle_DTOI
            if abs(testang - magicang.D) < tol.cageang: return SymInfo('D3', **sym_info_args)
            if abs(testang - magicang.T) < tol.cageang: return SymInfo('T', **sym_info_args)
            if abs(testang - magicang.O) < tol.cageang: return SymInfo('O', **sym_info_args)
            if abs(testang - magicang.I) < tol.cageang: return SymInfo('I', **sym_info_args)
            return SymInfo(f'Unknown sym with nfold {nfolds} testang {testang} valid are: {magicang}',
                           **sym_info_args)
        # other dihedrals D3+
        elif len(nfolds) == 2 and 2 in nfolds and len(frames) % 2 == 0:
            not2_vs_2ang = h.line_angle(se.axis[se.nfold == 2], se.axis[se.nfold != 2], outerprod=True)
            # ipd.dev.print_table(se, nohomog=True)
            if not np.all(np.abs(not2_vs_2ang - np.pi / 2) < tol.line_angle):
                return SymInfo(f'Unknown sym with nfold {nfolds}')
            othernf = max(nfolds)
            if 0 not in nfolds or _nfold_is_plasuible(len(frames) // 2, se.ang[se.nfold == 0], tol):
                if not _nfold_is_plasuible(othernf, se.axis[se.nfold == othernf], tol):
                    se.nfold[se.nfold == 0] = othernf
                return SymInfo(f'D{othernf}', **sym_info_args)
            # return SymInfo(f'Unknown dihedral-like sym, probably incomplete')
    elif not is_point and all(nf in (1, 2, 3, 4, 6) for nf in nfolds):
        return SymInfo('X', **sym_info_args)
    elif has_translation:
        return SymInfo('H', **sym_info_args)
    return SymInfo(f'Unknown Mysterious Sym nfolds {nfolds}', **sym_info_args)

def _nfold_is_plasuible(nfold, ang, tol):
    for i in range(1, nfold):
        target = 2 * np.pi / i
        if np.all(np.abs(ang - target) < tol.angle): return True
    return False

def symelems_from_frames(frames, **kw):
    """
    Compute a non-redundant set of symmetry elements from transformation frames.

    Args:
        frames (np.ndarray):
            Transformation frames with shape (N, 4, 4).
        **kw:
            Additional keyword arguments for tolerance settings.

    Returns:
        ipd.Bunch: Detected symmetry elements.

    Examples:
        >>> import numpy as np
        >>> from ipd.sym.sym_detect import symelems_from_frames
        >>> frames = ipd.sym.frames('c2')
        >>> elements = symelems_from_frames(frames)
        >>> elements
        Bunch(nfold=[2], axis=[[0. 0. 1. 0.]], ang=[3.14159], cen=[[0. 0. 0. 1.]], hel=[0.])
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
        return syminfo_get_origin_nonpoint(sinfo)
    elif sinfo.is_cyclic:
        origin = h.align([0, 0, 1], sinfo.axis[0]) @ h.trans(sinfo.symcen)
    elif sinfo.is_dihedral:
        ax2, axx = sinfo.nfaxis[2][0], sinfo.nfaxis[nf[1 % len(nf)]][0]
        if sinfo.symid == 'D2': axx = sinfo.nfaxis[2][-1]
        origin = h.align2([0, 0, 1], [1, 0, 0], ax2, axx) @ h.trans(sinfo.symcen)
    elif sinfo.is_cage:
        assert 2 in nf and 3 in nf
        assert sinfo.symid in 'T I O D3'.split()
        origax2, origax3 = ipd.sym.axes(sinfo.symid)[2], ipd.sym.axes(sinfo.symid)[3]
        ax2, ax3 = sinfo.nfaxis[2][0], sinfo.nfaxis[3][0]
        origin = h.align2(origax2, origax3, ax2, ax3) @ h.trans(sinfo.symcen)
    else:
        origin = h.align([0, 0, 1], sinfo.axis[0]) @ h.trans(sinfo.symcen)

    assert h.valid44(origin)
    return origin, h.inv(origin)

def syminfo_get_origin_nonpoint(sinfo):
    return np.nan * np.eye(4)

def syminfo_to_str(sinfo, verbose=True):
    """returns text tables with the data in this SymInfo"""
    textmap = {'nfold': 'C', '__REGEX__': False}  # , '[': '', ']': '
    # kw = dict(justify='right', title_justify='right', caption_justify='right', textmap=textmap)
    kw = dict(textmap=textmap, expand=True)
    if sinfo.symid == 'C1': return 'SymInfo(C1)'
    with ipd.dev.capture_stdio() as out:
        symcen = '\n'.join(f'{x:7.3f}' for x in sinfo.symcen.reshape(4)[:3])
        head = dict(symid=sinfo.symid, guess=sinfo.guess_symid, symcen=symcen, origin=sinfo.origin)
        headtable = ipd.dev.make_table(dict(foo=head), key=False, strip=False, **kw)
        setable = ipd.dev.make_table(sinfo.symelem, nohomog=True, **kw)
        asurms = sinfo.asu_components.rmsd if sinfo.asu_components else -1
        if asurms != -1: asurms = [float(r.max().round(3)) for r in asurms]
        asumatch = sinfo.asu_components.seqmatch if sinfo.asu_components else -1
        if asumatch != -1: asumatch = [float(sm.min().round(3)) for sm in asumatch]
        asutable = ipd.dev.make_table(
            [[sinfo.t_number, asurms, asumatch, sinfo.asuframes.shape, sinfo.allframes.shape]],
            header=['T', 'asu rms', 'asu seq match', 'asu frames', 'all frames'],
            **kw)
        geomdata = [[sinfo.has_translation, sinfo.is_point, sinfo.axes_dists]]
        geomhead = ['has_translation', 'axes concurrent', 'axes dists']
        geomtable = ipd.dev.make_table(geomdata, header=geomhead, **kw)
        checktable = ipd.dev.make_table(sinfo.tol_checks, key='Geom Tests')
        tables = [[headtable], [geomtable], [asutable], [setable], [checktable]]
        rms = sinfo.components.rmsd[sinfo.component].max() if sinfo.components else -1
        seqmatch = sinfo.components.seqmatch[sinfo.component].min() if sinfo.components else -1
        if sinfo.stub0 is not None:
            rmstable = ipd.dev.make_table([[seqmatch, rms]],
                                          header=['worst seq match', 'worst rms'])
            tables.append([rmstable])
        ipd.dev.print_table(tables, header=['SymInfo'])
    return out.read().rstrip()

def _syminfo_add_atoms_info(sinfo: SymInfo, atomslist, frames, tol) -> None:
    if sinfo.order >= 60:
        asu = ipd.atom.split(atomslist[0])
        asuchains = np.unique(atomslist[0].chain_id)
        sinfo.asu_components = ipd.atom.find_components_by_seqaln_rmsfit(asu, tol=tol)
        sinfo.asu_components.remove_small_chains(minres=20)
        assert len(sinfo.asu_components) == 1
    if sinfo.asu_components:
        sinfo.stub0 = ipd.atom.stub(asu[0])
        sinfo.asuframes = sinfo.asu_components.frames[0]
        sinfo.asustub = h.xform(sinfo.asuframes, sinfo.stub0)
        sinfo.allstub = h.xform(frames, sinfo.asustub)
        sinfo.allframes = h.xform(sinfo.allstub, h.inv(sinfo.stub0))
    else:
        sinfo.stub0 = ipd.atom.stub(atomslist[0])
        sinfo.asustub = sinfo.stub0[None]
        sinfo.allframes = frames[:, None]
        sinfo.allstub = h.xform(frames, sinfo.asustub).reshape(sinfo.allframes.shape)
    assert sinfo.allframes.shape == sinfo.allstub.shape
    assert h.allclose(h.xform(sinfo.allframes, sinfo.stub0), sinfo.allstub)

def check_sym_combinations(syminfos, incomplete=False):
    if len(syminfos) == 1: return syminfos[0]
    tol = syminfos[0].tolerances.copy().reset()

    symids = [si.symid for si in syminfos]
    if len(set(symids)) == 1:
        return syminfos[0]
    if not incomplete:
        return syminfos

    ipd.icv(symids)

    frames, stubs = list(), list()
    for sinfo in syminfos:
        # ipd.print_table(sinfo.symelem)
        frames.append(sinfo.frames)
        stubs.append(h.xform(sinfo.frames, sinfo.stub0))
        ipd.icv(sinfo.frames.shape)

    frames = np.concatenate(frames)
    frames = np.concatenate(stubs)
    # ipd.showme(stubs, xyzscale=10, weight=10, spheres=4)
    assert 0
