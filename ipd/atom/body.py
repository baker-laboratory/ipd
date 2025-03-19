"""
Module: ipd.atom.body
=====================

This module defines the Body class that represents a collection of atoms organized
as a coherent structure. It provides methods for spatial operations such as clash
and contact detection and supports transformation operations through homogeneous
transforms (using the ipd.homog.hgeom module).

Key features:
  - Construction of Body objects from PDB files using helper functions (e.g., body_from_file).
  - Efficient clash detection and contact checks between bodies and AtomArrays.
  - Application of 4x4 homogeneous transformations to manipulate the spatial arrangement.
  - Utilization of the highly efficient SphereBVH_double from willutil_cpp, which can
    deliver hundreds of times speedup for large, lightly contacting structures (e.g., virus capsids).

Usage Examples:
    >>> from ipd import atom, hgeom as h
    >>> # Create a Body from a PDB code using a helper function
    >>> b = atom.body_from_file("1byf")
    >>> # Apply a translation to the Body
    >>> T = h.trans([5, 0, 0])
    >>> b2 = b.apply_transform(T)
    >>> # Check for a clash between the original and transformed bodies
    >>> b.clash(b2) in [True, False]
    True

    >>> # Demonstrate contact checking between a Body and an AtomArray
    >>> aa = atom.load("1dxh")
    >>> contacts = b.contact_check(aa)
    >>> isinstance(contacts, list)
    True

Additional Examples:
    >>> # Apply a combined rotation and translation
    >>> T1 = h.trans([1, 2, 3])
    >>> T2 = h.rot([0, 0, 1], 45, [0, 0, 0])
    >>> b_transformed = b.apply_transform(h.xform(T1, T2))
    >>> contacts_new = b_transformed.contact_check(aa)
    >>> isinstance(contacts_new, list)
    True

    >>> # Create another Body with a different pdb code and check for clashes
    >>> b3 = atom.body_from_file("1ql2")
    >>> b3.clash(b)
    False

.. note::
    The SphereBVH_double bounding volume hierarchy used here is extremely efficient
    for large symmetrical structures and is far superior in performance compared to more
    traditional methods.

.. seealso::
    ipd.atom.components and ipd.atom.atom_utils for further atom-level operations.
"""

import copy
from dataclasses import dataclass, field
import numpy as np
import typing

import ipd
import ipd.homog.hgeom as h

if typing.TYPE_CHECKING:
    from biotite.structure import AtomArray
    import willutil_cpp as wu

wu = ipd.maybeimport('willutil_cpp')
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

    if use_cif_xforms and 'assembly_xforms' in ipd.dev.get_metadata(atomslist[0]):
        ipd.ic(asmx._chainasu)
        assert 0
    else:
        comp = ipd.atom.find_components_by_seqaln_rmsfit(atomslist, **kw)
        ipd.atom.process_components(comp, **kw)
        # ic(comp.frames[0][3], comp.frames[1][3])
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
    return SymBody(Body(comp.atoms[which]), comp.frames[which], _atomslist=atomslist, _assembly_xforms=asmx)

@ipd.dev.holds_metadata
@dataclass
class Body:
    atoms: 'AtomArray'
    pos: np.ndarray = field(default_factory=lambda: np.eye(4))
    rescen: np.ndarray = None
    _atombvh: 'wu.SphereBVH_double' = None
    _resbvh: 'wu.SphereBVH_double' = None
    hydro: bool = False
    hetero: bool = False
    water: bool = False
    # seq: str = ''
    nres = property(lambda self: len(self.seq))
    natom = property(lambda self: len(self.atoms))
    positioned_atoms = property(lambda self: h.xform(self.pos, self.atoms))
    com = property(lambda self: h.xform(self.pos, bs.mass_center(self.atoms)))
    rg = property(lambda self: h.radius_of_gyration(self[:], self.com))

    def __post_init__(self):
        if not self.hetero: self.atoms = self.atoms[~self.atoms.hetero]
        if not self.hydro: self.atoms = self.atoms[self.atoms.element != 'H']
        if not self.water: self.atoms = self.atoms[self.atoms.res_name != 'HOH']
        self.rescen = bs.apply_residue_wise(self.atoms, self.atoms.coord, np.mean, axis=0)
        self._atombvh = wu.SphereBVH_double(self.atoms.coord)
        self._resbvh = wu.SphereBVH_double(self.rescen)
        # self.seq = ipd.atom.atoms_to_seqstr(self.atoms)
        assert h.valid44(self.pos)

    def __eq__(self, other):
        return self.atoms is other.atoms

    def isclose(self, other):
        return self.atoms is other.atoms and np.allclose(self.pos, other.pos)

    def hasclash(self, other=None, radius: float = 2, **kw) -> bool:
        result = _bvh_binary_operation(wu.bvh_isect, self, other, radius=radius, **kw)
        return ipd.cast(bool, result)

    def nclash(self, other=None, radius: float = 2, **kw) -> int:
        result = _bvh_binary_operation(wu.bvh_count_pairs, self, other, radius=radius, **kw)
        return ipd.cast(int, result)

    def contacts(self, other=None, radius: float = 4, **kw) -> ipd.Bunch:
        result = _bvh_binary_operation(wu.bvh_collect_pairs_vec, self, other, radius=radius, **kw)
        p, b = ipd.cast(tuple[np.ndarray, np.ndarray], result)
        uniq0, uniq1 = np.unique(p[:, 0]), np.unique(p[:, 1])
        fields = 'pairs breaks pair0 pair1 uniq0 uniq1 nuniq0 nuniq'.split()
        vals = [p, b, p[:, 0], p[:, 1], uniq0, uniq1, len(uniq0), len(uniq1)]
        return ipd.Bunch(zip(fields, vals))

    def slide_into_contact(self, other, direction=(1, 0, 0), radius=3.0) -> 'Body':
        if direction == 'random': direction = h.rand_unit()[:3]
        delta = _bvh_binary_operation(wu.bvh_slide, self, other, rad=radius, dirn=direction)
        return self.movedby((delta - np.sign(delta) * radius) * np.array(direction))

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

    def __getattr__(self, name):
        if name == 'atoms':
            raise AttributeError
        try:
            return getattr(self.atoms, name)
        except AttributeError:
            raise AttributeError(f'Body (nor AtomArray) has no attribute: {name}')

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
@dataclass
class SymBody:
    asu: Body
    frames: np.ndarray
    pos: np.ndarray = field(default_factory=lambda: np.eye(4))
    bodies = property(lambda self: [self.asu.movedby(self.pos @ f) for f in self.frames])
    atoms = property(lambda self: ipd.atom.join(h.xform(self.pos, self.frames, self.asu.atoms)))
    com = property(lambda self: h.xform(self.pos, self.frames, self.asu.com).mean(0))
    centered = property(lambda self: self.movedby(h.trans(-self.com)))
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

    def _get_pos_otherpos(self, other):
        kw = ipd.Bunch(pos=h.xform(self.pos, self.frames, self.asu.pos))
        if isinstance(other, SymBody):
            kw.otherpos = h.xform(other.pos, other.frames, other.asu.pos)
        return kw

    def hasclash(self, other: 'Body|SymBody|None' = None, radius: float = 2, **kw) -> bool:
        kw |= self._get_pos_otherpos(other)
        result = _bvh_binary_operation(wu.bvh_isect_vec, self, other, radius=radius, **kw)
        return ipd.cast(bool, result)

    def nclash(self, other=None, radius: float = 2, **kw) -> int:
        kw |= self._get_pos_otherpos(other)
        result = _bvh_binary_operation(wu.bvh_count_pairs_vec, self, other, radius=radius, **kw)
        return ipd.cast(int, result)

    def contacts(self, other=None, radius: float = 4, **kw) -> 'SymBodyContacts':
        kw |= self._get_pos_otherpos(other)
        result = _bvh_binary_operation(wu.bvh_collect_pairs_vec, self, other, radius=radius, **kw)
        p, r = ipd.cast(tuple[np.ndarray, np.ndarray], result)
        return SymBodyContacts(self, other or self, p, r)

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
    **kw,
) -> 'bool|int|float|np.ndarray|tuple[np.ndarray, np.ndarray]':
    """abailable:
            bvh_collect_pairs bvh_collect_pairs_range_vec bvh_collect_pairs_vec
            bvh_count_pairs bvh_count_pairs_vec
            bvh_isect bvh_isect_range bvh_isect_vec
            bvh_min_dist bvh_min_dist_vec
            bvh_slide bvh_slide_vec
            bvh_print
        """
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
    # ic(op, pos.shape, otherpos.shape)
    extra = kw.values()
    result = op(bvh, otherbvh, pos, otherpos, *extra)
    if op.__name__.endswith('_vec'):
        if isinstance(result, tuple):
            val, ranges = result
            result = val, ranges.reshape(npos, nother, *ranges.shape[1:])
        else:
            result = result.reshape(npos, nother)
    return result

@ipd.dc.dataclass
class SymBodyContacts:
    symbody1: SymBody
    symbody2: SymBody
    values: ipd.NDArray_N2_int32
    ranges: ipd.NDArray_MN2_int32

    def __post_init__(self):
        self.symbody2 = self.symbody2 or self.symbody1
        assert len(self.symbody1) == self.ranges.shape[0]
        assert len(self.symbody2) == self.ranges.shape[1]

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        for isub1, sub1 in enumerate(self.symbody1.bodies):
            for isub2, sub2 in enumerate(self.symbody2.bodies):
                lb, ub = self.ranges[isub1, isub2]
                iatom1, iatom2 = self.values[lb:ub].T
                yield sub1, sub2, iatom1, iatom2
