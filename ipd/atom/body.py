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
    atombvh: 'wu.SphereBVH_double' = None
    resbvh: 'wu.SphereBVH_double' = None
    hydro: bool = False
    hetero: bool = False
    water: bool = False
    seq: str = None
    nres: int = None

    def __attrs_post_init__(self):
        if not self.hetero: self.atoms = self.atoms[~self.atoms.hetero]
        if not self.hydro: self.atoms = self.atoms[self.atoms.element != 'H']
        if not self.water: self.atoms = self.atoms[self.atoms.res_name != 'HOH']
        self.rescen = bs.apply_residue_wise(self.atoms, self.atoms.coord, np.mean, axis=0)
        self.atombvh = wu.SphereBVH_double(self.atoms.coord)
        self.resbvh = wu.SphereBVH_double(self.rescen)
        self.seq = ipd.atom.atoms_to_seq(self.atoms)
        self.nres = len(self.seq)

    def nclash_celllist(self, other, radius=3) -> bool:
        cell_list = bs.CellList(self.atoms, radius + 1)
        nclash = 0
        for pos in other[:]:
            nclash += len(cell_list.get_atoms(pos, radius=radius))
        return nclash

    def clashes(self, other, radius) -> bool:
        return wu.bvh_isect(self.atombvh, other.atombvh, self.pos, other.pos, radius)

    def nclash(self, other, radius=3) -> int:
        return wu.bvh_count_pairs(self.atombvh, other.atombvh, self.pos, other.pos, radius)

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
