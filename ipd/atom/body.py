"""
Provides Body classes holding AtomArrays augmented with performance enhancements and extra structural information. It provides methods for spatial operations such as clash
and contact detection and supports transformation operations through homogeneous
transforms (using the ipd.homog.hgeom module).

Key features:
  - Construction of Body objects from PDB files using helper functions (e.g., body_from_file).
  - Efficient clash detection and contact checks between bodies and AtomArrays.
  - Application of 4x4 homogeneous transformations to manipulate the spatial arrangement.
  - Utilization of the highly efficient SphereBVH_double from hgeom, which can
    deliver hundreds of times speedup for large, lightly contacting structures (e.g., virus capsids).

Usage Examples:
    >>> from ipd import atom, hnumpy as h
    >>> # Create a Body from a PDB code using a helper function
    >>> b = atom.body_from_file('1byf').centered
    >>> # Apply a translation to the Body
    >>> T = h.trans([5, 0, 0])
    >>> b2 = h.xform(T, b)
    >>> # Check for a clash between the original and transformed bodies
    >>> b.hasclash(b2)
    True

Additional Examples:
    >>> # Apply a combined rotation and translation
    >>> T1 = h.trans([1, 2, 3])
    >>> T2 = h.rot([0, 0, 1], 45, [0, 0, 0])
    >>> b_transformed = h.xform(T1, T2, b)
    >>> contacts_new = b_transformed.contacts(b)
    >>> contacts_new.nuniq > 10
    True

    >>> # Create another Body with a different pdb code and check for clashes
    >>> b3 = atom.body_from_file('1ql2').centered
    >>> b3.hasclash(b)
    True

.. note::
    The SphereBVH_double bounding volume hierarchy used here is extremely efficient
    for large symmetrical structures and is far superior in performance compared to more
    traditional methods.

.. seealso::
    ipd.atom.components and ipd.atom.atom_utils for further atom-level operations.
"""

import copy
import numpy as np
import typing

import ipd
import ipd.homog.hgeom as h

if typing.TYPE_CHECKING:
    from biotite.structure import AtomArray
    import hgeom as hg

hg = ipd.maybeimport('hgeom')
bs = ipd.lazyimport('biotite.structure')

@ipd.ft.lru_cache
def body_from_file(
    fname: str,
    assembly='largest',
    min_chain_atoms=0,
    **kw,
) -> 'Body':
    ipd.dev.checkpoint()
    atoms = ipd.pdb.readatoms(fname, assembly=assembly, **kw)
    if isinstance(atoms, list): atoms = ipd.dev.addreduce(atoms)
    ipd.dev.checkpoint('read atoms')
    return Body(atoms)

@ipd.ft.lru_cache
def symbody_from_file(
    fname: str,
    assembly='largest',
    components='onlyone',
    min_chain_atoms=0,
    use_cif_xforms=False,
    **kw,
) -> 'SymBody':
    atomslist = ipd.pdb.readatoms(fname, chainlist=True, assembly=assembly, **kw)
    assert isinstance(atomslist, list)
    asmx = ipd.dev.get_metadata(atomslist[0]).assembly_xforms

    if False and use_cif_xforms and 'assembly_xforms' in ipd.dev.get_metadata(atomslist[0]):
        ipd.ic(asmx._chainasu)
        assert 0
    else:
        comp = ipd.atom.find_components_by_seqaln_rmsfit(atomslist, **kw)
        ipd.atom.process_components(comp, **kw)
        # ipd.icv(comp.frames[0][3], comp.frames[1][3])
        # ipd.showme(h.xform(comp.frames[0],h.trans([1,3,10])) , xyzscale=4, weight=4,name='baz')
        # ipd.showme(h.xform(comp.frames[1][0::2],h.trans([1,3,10])) , xyzscale=4, weight=4,name='bar')
        # ipd.showme(h.xform(comp.frames[1][1::2],h.trans([1,3,10])) , xyzscale=4, weight=4,name='foo')

    if components in ('merge', 'merge_unsafe'):
        assert len(set(map(len, comp.frames))) == 1
        if components == 'merge':
            assert all(h.allclose(f, g) for f, g in ipd.it.pairwise(comp.frames))
        return SymBody(Body(ipd.dev.addreduce(comp.atoms)), comp.frames[0])
    if components == 'onlyone':
        assert len(comp) == 1
        which = 0
    elif components == 'largest_monomer':
        which = np.argmax([len(atoms) for atoms in comp.atoms])
    elif components == 'largest_assembly':
        which = np.argmax([len(s) * len(a) for a, s in zip(*comp['atoms frames'])])
    elif components == 'most_symmetric':
        which = np.argmax([9e9 * len(s) + len(a) for a, s in zip(*comp['atoms frames'])])
    elif components == 'most_symmetric':
        which = np.argmax([9e9 * len(s) + len(a) for a, s in zip(*comp['atoms frames'])])
    else:
        raise ValueError(f'unknown component handling method "{components}"')
    asu = Body(comp.atoms[which])
    asu.meta.pdbcode = ipd.Path(fname).stem
    return SymBody(asu, comp.frames[which], _atomslist=atomslist, _assembly_xforms=asmx)

@ipd.dev.holds_metadata
@ipd.mutablestruct
class Body:
    atoms: 'AtomArray'
    pos: np.ndarray = ipd.field(lambda: np.eye(4))
    rescen: np.ndarray = None
    _atombvh: 'hg.SphereBVH_double' = None
    _resbvh: 'hg.SphereBVH_double' = None
    hydro: bool = False
    hetero: bool = False
    water: bool = False
    # seq: str = ''
    nres = property(lambda self: len(self.seq))
    natom = property(lambda self: len(self.atoms))
    positioned_atoms = property(lambda self: h.xform(self.pos, self.atoms))
    com = property(lambda self: h.xform(self.pos, bs.mass_center(self.atoms)))
    rg = property(lambda self: h.radius_of_gyration(self[:], self.com))
    centered = property(lambda self: self.movedby(-self.com))

    def __post_init__(self):
        if not self.hetero: self.atoms = self.atoms[~self.atoms.hetero]
        if not self.hydro: self.atoms = self.atoms[self.atoms.element != 'H']
        if not self.water: self.atoms = self.atoms[self.atoms.res_name != 'HOH']
        self.rescen = bs.apply_residue_wise(self.atoms, self.atoms.coord, np.mean, axis=0)
        self._atombvh = hg.SphereBVH_double(self.atoms.coord)
        self._resbvh = hg.SphereBVH_double(self.rescen)
        self.asu = self
        # self.seq = ipd.atom.atoms_to_seqstr(self.atoms)
        assert h.valid44(self.pos)

    def __eq__(self, other):
        return self.atoms is other.atoms

    def _get_pos_otherpos(self, other):
        kw = ipd.Bunch(pos=self.pos)
        if not isinstance(other, SymBody): kw.otherpos = other.pos
        else: kw.otherpos = h.xform(other.pos, other.frames, other.asu.pos)
        return kw

    def isclose(self, other):
        return self.atoms is other.atoms and np.allclose(self.pos, other.pos)

    def hasclash(self, other=None, radius: float = 2, **kw) -> bool:
        result = _bvh_binary_operation(hg.bvh_isect, self, other, radius=radius, **kw)
        return ipd.cast(bool, result)

    def nclash(self, other=None, radius: float = 2, **kw) -> int:
        result = _bvh_binary_operation(hg.bvh_count_pairs, self, other, radius=radius, **kw)
        return ipd.cast(int, result)

    def contacts(self, other=None, radius: float = 4, **kw) -> 'BodyContacts':
        return SymBody(self).contacts(other, radius, **kw)

    def slide_into_contact(self, other, along: ipd.Vec = (1, 0, 0), radius=3.0) -> 'Body':
        kwpos = self._get_pos_otherpos(other)
        delta = _bvh_binary_operation(hg.bvh_slide_vec, self, other, rad=radius, dirn=along, **kwpos)
        delta = np.min(delta)
        return self.movedby((delta - np.sign(delta) * radius) * np.array(along))

    def slide_into_contact_rand(self, *a, **kw) -> 'tuple[Body,np.ndarray]':
        along = h.rand_unit()[:3]
        return self.slide_into_contact(*a, along=along, **kw), along

    @property
    def coord(self):
        return h.xform(self.pos, self.atoms.coord)

    def movedby(self, xform):
        return self.movedto(xform, moveby=True)

    def movedto(self, xform, moveby=False):
        xform = np.asarray(xform)
        if xform.size in (3, 4): xform = h.trans(xform)
        new = self.clone()
        if moveby: xform = h.xformx(xform, self.pos)
        new.pos = xform
        return new

    def atomsel(self, **kw):
        return ipd.atom.select(self.positioned_atoms, **kw)

    def __getitem__(self, *slices):
        return h.xformpts(self.pos, self.atoms.coord[tuple(slices)])

    # def __getattr__(self, name):
    #     if name == 'atoms':
    #         raise AttributeError
    #     try:
    #         return getattr(self.atoms, name)
    #     except AttributeError:
    #         raise AttributeError(f'Body (nor AtomArray) has no attribute: {name}')

    def summary(self):
        nhet = np.sum(self.atoms.hetero)
        return f'Body(atom: {len(self.atoms)} res: {len(self._resbvh)} net: {nhet} pos: {self.pos[:3,3]})'

    __repr__ = summary

    def __str__(self):
        fields = {k: v for k, v in vars(self).items() if k[0] != '_'}
        fields['atoms'] = self.atoms.shape
        fields['rescen'] = self.rescen.shape
        table = ipd.dev.make_table(fields)
        with ipd.dev.capture_stdio() as printed:
            ipd.dev.print_table(table)
        return printed.read()

    def __hash__(self):
        """NOTE: this hash ignores body position!!"""
        return id(self.atoms)

    def clone(self):
        return copy.copy(self)

@ipd.dev.holds_metadata
@ipd.mutablestruct
class SymBody:
    asu: Body
    frames: np.ndarray = ipd.field(lambda: np.eye(4)[None])
    pos: np.ndarray = ipd.field(lambda: np.eye(4))
    bodies = property(lambda self: [self.asu.movedby(self.pos @ f) for f in self.frames])
    atoms = property(lambda self: ipd.atom.join(h.xform(self.pos, self.frames, self.asu.pos, self.asu.atoms)))
    com = property(lambda self: h.xform(self.pos, self.frames, self.asu.com).mean(0))
    centered = property(lambda self: self.movedby(-self.com))
    rg = property(lambda self: h.radius_of_gyration(self[:], self.com))
    natoms = property(lambda self: len(self) * len(self.asu.atoms))

    def __post_init__(self):
        assert h.valid44(self.frames)
        self._atombvh = self.asu._atombvh
        self._resbvh = self.asu._resbvh

    def __len__(self):
        return len(self.frames)

    def isclose(self, other):
        return self.asu.isclose(other.asu) and h.allclose(self.frames, other.frames)

    def _get_pos_otherpos(self, other, exclude=()):
        idx = [i for i in range(len(self.frames)) if i not in exclude]
        kw = ipd.Bunch(pos=h.xform(self.pos, self.frames[idx], self.asu.pos))
        if isinstance(other, SymBody):
            kw.otherpos = h.xform(other.pos, other.frames, other.asu.pos)
        else:
            kw.otherpos = other.pos
        return kw

    def hasclash(self, other: 'Body|SymBody|None' = None, radius: float = 2, **kw) -> bool:
        kw |= self._get_pos_otherpos(other)
        result = _bvh_binary_operation(hg.bvh_isect_vec, self, other, radius=radius, **kw)
        return ipd.cast(bool, result)

    def nclash(self, other=None, radius: float = 2, **kw) -> int:
        kw |= self._get_pos_otherpos(other)
        result = _bvh_binary_operation(hg.bvh_count_pairs_vec, self, other, radius=radius, **kw)
        return ipd.cast(int, result)

    def contacts(self, other=None, radius: float = 4, exclude: list[int] = (), **kw) -> 'BodyContacts':
        if not isinstance(exclude, ipd.Iterable): exclude = [exclude]
        kw |= self._get_pos_otherpos(other, exclude=exclude)
        result = _bvh_binary_operation(hg.bvh_collect_pairs_vec, self, other, radius=radius, **kw)
        p, r = ipd.cast(tuple[np.ndarray, np.ndarray], result)
        return BodyContacts(self, other or self, p, r, exclude)

    def slide_into_contact(self, other, along=(1, 0, 0), radius=3.0) -> 'SymBody':
        return ipd.cast(SymBody, Body.slide_into_contact(self, other, along, radius))  # type: ignore

    def slide_into_contact_rand(self, *a, **kw) -> 'tuple[SymBody,np.ndarray]':
        along = h.rand_unit()[:3]
        return self.slide_into_contact(*a, along=along, **kw), along

    @property
    def coord(self):
        return h.xform(self.pos, self.frames, self.atoms.coord)

    def movedby(self, xform):
        return self.movedto(xform, moveby=True)

    def movedto(self, xform, moveby=False):
        xform = np.asarray(xform)
        if xform.size in (3, 4): xform = h.trans(xform)
        new = self.clone()
        if moveby: xform = h.xformx(xform, self.pos)
        new.pos = xform
        return new

    def natomsel(self, **kw):
        return len(self.frames) * len(self.asu.atomsel(**kw))

    def atomsel(self, **kw):
        return ipd.atom.join(h.xform(self.pos, self.frames, self.asu.atomsel(**kw)))

    def __getitem__(self, *slices):
        first, *rest = slices
        frames = self.frames[first]
        if rest: return h.xformpts(frames, self.asu[rest])
        else: pass
        return h.xformpts(frames, self.asu[:])

    # def __getattr__(self, name):
    #     if name == 'asu': raise AttributeError
    #     try:
    #         return getattr(self.asu, name)
    #     except AttributeError:
    #         raise AttributeError(f'SymBody (nor Body nor AtomArray) has no attribute: {name}')

    def summary(self):
        n = len(self.frames)
        nhet = np.sum(self.asu.atoms.hetero) * n
        return f'SymBody(atom: {len(self.asu.atoms)*n} res: {len(self.asu._resbvh)*n} net: {nhet} pos: {self.pos[:3,3]} frams: {self.frames.shape})'

    __repr__ = summary

    def __str__(self):
        with ipd.dev.capture_stdio() as printed:
            print(f'SymBody:\n{self.asu}')
            print(f'Frames:\n{self.frames}')
        return printed.read()

    def __eq__(self, other):
        return self.asu == other.asu

    def __hash__(self):
        """NOTE: this hash ignores asu frames/position!!"""
        return id(self.asu)

    def clone(self):
        return copy.copy(self)

def _bvh_binary_operation(
    op,
    this,
    other=None,
    bvh=None,
    otherbvh=None,
    pos=None,
    otherpos=None,
    residue_wise=False,
    debug=False,
    **kw,
) -> 'bool|int|float|np.ndarray|tuple[np.ndarray, np.ndarray]':
    other = other or this
    bvh = bvh or this._resbvh if residue_wise else this._atombvh
    otherbvh = otherbvh or other._resbvh if residue_wise else other._atombvh
    pos = this.pos if pos is None else pos
    otherpos = other.pos if otherpos is None else otherpos
    npos, nother = 1, 1
    if op.__name__.endswith('_vec'):
        pos, otherpos = pos.reshape(-1, 4, 4), otherpos.reshape(-1, 4, 4)
        npos, nother = len(pos), len(otherpos)
        pos = np.repeat(pos, nother, axis=0)
        otherpos = np.tile(otherpos, (npos, 1, 1))
    # ipd.icv(op, pos.shape, otherpos.shape)
    extra = kw.values()
    if debug: ipd.icv(op, otherpos.shape)
    result = op(bvh, otherbvh, pos, otherpos, *extra)
    if op.__name__.endswith('_vec'):
        if isinstance(result, tuple):
            val, ranges = result
            result = val, ranges.reshape(npos, nother, *ranges.shape[1:])
        else:
            result = result.reshape(npos, nother)
    return result

@ipd.dc.dataclass
class BodyContacts:
    symbody1: SymBody
    body2: ipd.Union[Body, SymBody]
    pairs: ipd.NDArray_N2_int32
    ranges: ipd.NDArray_MN2_int32
    exclude: list[int]
    total_contacts = property(lambda self: len(self.pairs))
    max_contacts = property(lambda self: np.max(self.ranges[:, 1] - self.ranges[:, 0]))
    mean_contacts = property(lambda self: np.mean(self.ranges[:, 1] - self.ranges[:, 0]))
    min_contacts = property(lambda self: np.min(self.ranges[:, 1] - self.ranges[:, 0]))
    nuniq1 = property(lambda self: np.unique(self.pairs[:, 0]).size)
    nuniq2 = property(lambda self: np.unique(self.pairs[:, 1]).size)
    nuniq = property(lambda self: np.unique(self.pairs.flat).size)

    def __post_init__(self):
        self.body2 = self.body2 or self.symbody1

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        bodies2 = [(0, self.body2)] if isinstance(self.body2, Body) else enumerate(self.body2.bodies)
        isub1 = 0
        for i, sub1 in enumerate(self.symbody1.bodies):
            if i in self.exclude: continue
            for isub2, sub2 in bodies2:
                lb, ub = self.ranges[isub1, isub2]
                iatom1, iatom2 = self.pairs[lb:ub].T
                yield isub1, isub2, sub1, sub2, iatom1, iatom2
            isub1 += 1

    def contact_matrix_stack(self, tokens1=None, tokens2=None):
        if tokens1 is None: tokens1 = self.symbody1.asu.atoms.res_id
        if tokens2 is None: tokens2 = self.body2.asu.atoms.res_id
        tokens1, tokens2 = ipd.cast(np.ndarray, tokens1), ipd.cast(np.ndarray, tokens2)
        assert isinstance(self.body2, Body)
        mn1, mx1, mn2, mx2 = min(tokens1), max(tokens1), min(tokens2), max(tokens2)
        mats, subs = [], []
        for isub1, isub2, sub1, sub2, iatom1, iatom2 in self:
            if not len(iatom1): continue
            mat = np.zeros((mx1 - mn1 + 1, mx2 - mn2 + 1), dtype=np.int32)
            lev1, lev2 = tokens1[iatom1], tokens2[iatom2]
            np.add.at(mat, (lev1 - mn1, lev2 - mn2), 1)
            mats.append(mat.T)  # so asu is first
            subs.append(isub1)
        return ipd.homog.ContactMatrixStack(np.stack(mats), np.stack(subs))

    def __repr__(self):
        return f'SymContacts(ranges: {self.ranges.shape} pairs: {self.pairs.shape})'
