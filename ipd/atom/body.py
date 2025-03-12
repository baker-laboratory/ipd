"""
represents coordinates as biotite AtomArray along with a bounding volume hierarchy for fast geom checks. should behave as a decorator around AtomArray and
"""
import copy

import numpy as np

import attrs
import ipd
import ipd.homog.hgeom as h

wu = ipd.lazyimport('willutil_cpp')
bs = ipd.lazyimport('biotite.structure')

@attrs.define
class Body:
    atoms: 'bs.AtomArray'
    pos: np.ndarray = np.eye(4)
    rescen: np.ndarray = None
    _atombvh: 'wu.SphereBVH_double' = None
    _resbvh: 'wu.SphereBVH_double' = None
    hydro: bool = False
    hetero: bool = False
    water: bool = False
    seq: str = None
    nres = property(lambda self: len(self.seq))
    natom = property(lambda self: len(self.atoms))

    def __attrs_post_init__(self):
        if not self.hetero: self.atoms = self.atoms[~self.atoms.hetero]
        if not self.hydro: self.atoms = self.atoms[self.atoms.element != 'H']
        if not self.water: self.atoms = self.atoms[self.atoms.res_name != 'HOH']
        self.rescen = bs.apply_residue_wise(self.atoms, self.atoms.coord, np.mean, axis=0)
        self._atombvh = wu.SphereBVH_double(self.atoms.coord)
        self._resbvh = wu.SphereBVH_double(self.rescen)
        self.seq = ipd.atom.atoms_to_seqstr(self.atoms)

    def _celllist_nclash(self, other, radius=3) -> bool:
        cell_list = bs.CellList(self.atoms, radius + 1)
        nclash = 0
        for pos in other[:]:
            nclash += len(cell_list.get_atoms(pos, radius=radius))
        return nclash

    def bvh_binary_operation(
        self,
        op,
        other=None,
        bvh=None,
        otherbvh=None,
        pos=None,
        otherpos=None,
        residue_wise=False,
        **kw,
    ):
        """abailable:
            bvh_collect_pairs bvh_collect_pairs_range_vec bvh_collect_pairs_vec
            bvh_count_pairs bvh_count_pairs_vec
            bvh_isect bvh_isect_range bvh_isect_vec
            bvh_min_dist bvh_min_dist_vec
            bvh_slide bvh_slide_vec
            bvh_print
        """
        other = other or self
        bvh = bvh or self._resbvh if residue_wise else self._atombvh
        otherbvh = otherbvh or other._resbvh if residue_wise else other._atombvh
        pos = self.pos if pos is None else pos
        otherpos = other.pos if otherpos is None else otherpos
        # ic(type(bvh), type(otherbvh), pos.dtype, otherpos.dtype, kw)
        extra = kw.values()
        return op(bvh, otherbvh, pos, otherpos, *extra)

    def hasclash(self, other=None, radius=2, **kw) -> bool:
        return self.bvh_binary_operation(wu.bvh_isect_vec, other, radius=radius, **kw)

    def nclash(self, other=None, radius=2, **kw) -> int:
        return self.bvh_binary_operation(wu.bvh_count_pairs, other, radius=radius, **kw)

    def contacts(self, other=None, radius=2, **kw) -> int:
        return self.bvh_binary_operation(wu.bvh_collect_pairs_vec, other, radius=radius, **kw)

    @property
    def coord(self):
        return self.atoms.coord

    def xformed(self, xform):
        new = copy.copy(self)
        new.pos = h.xform(xform, self.pos)
        return new

    def __getitem__(self, slice):
        return h.xform(self.pos, self.coord[slice])

    def __getattr__(self, name):
        try:
            return getattr(self.atoms, name)
        except AttributeError:
            raise AttributeError(f'Body (nor AtomArray) has no attribute: {name}')

    def summary(self):
        nhet = np.sum(self.atoms.hetero)
        return f'Body(atom: {len(self.atoms)} res: {len(self._resbvh)} net: {nhet})'

    def __repr__(self):
        fields = {key: getattr(self, key) for key in self.__slots__ if key[0] != '_'}
        fields['atoms'] = self.atoms.shape
        fields['rescen'] = self.rescen.shape
        table = ipd.dev.make_table(fields)
        with ipd.dev.capture_stdio() as printed:
            ipd.dev.print_table(table)
        return printed.read()
