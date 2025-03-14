"""
class to represent a potentially very large biological assembly efficiently. stores only the asymmetric coordinates and does calculations based on transformed asu coods. uses a sweet transformable bounding volume hierarchy to do itersection tests
"""
import random
import itertools as it
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import toolz

import ipd

h = ipd.hnumpy
bs = ipd.lazyimport('biotite.structure')

@ipd.dev.holds_metadata
@ipd.dev.subscriptable_for_attributes
@dataclass
class Assembly:
    bodies: list[ipd.atom.Body]
    frames: list[np.ndarray]
    _framemap: dict[ipd.atom.Body, np.ndarray] = field(default_factory=dict)
    _bodymap: dict[ipd.atom.Body, ipd.atom.Body] = field(default_factory=dict)
    _idmap: dict[ipd.atom.Body, int] = field(default_factory=dict)

    def __post_init__(self):
        self.bodies = list(self.bodies)
        self.frames = list(self.frames)
        self._bodymap = dict(zip(self.bodies, self.bodies))  # hash(body) ignores pos
        self._framemap = dict(zip(self.bodies, self.frames))
        self._idmap = dict(zip(self.bodies, range(len(self.bodies))))

    def body(self, bodyid=0, frameid=0):
        body = self.bodies[bodyid]
        frame = self.frames[bodyid][frameid]
        assert h.valid44(frame)
        new = body.movedby(frame)
        new.set_metadata(assembly=self, bodyid=bodyid, frameid=frameid)
        return new

    def enumerate_symbodies(self, **kw) -> Iterator[tuple[int, int, ipd.atom.Body, np.ndarray]]:
        ids = self.symbodyids(**kw)
        for i, j in ids:
            yield i, j, self.body(i, j), self.frames[i][j]

    def symbodyids(self, n=None, order='sorted'):
        allid = list(it.product(*map(lambda x: range(len(x)), (self.bodies, self.frames))))
        if order == 'random': random.shuffle(allid)
        return allid[:n]

    def __repr__(self):
        with ipd.dev.capture_stdio() as out:
            ipd.print_table(vars(self))
        return out.read()

def create_assembly(
    input: 'str|CifFile',
    assembly='largest',
    min_chain_atoms=0,
    **kw,
) -> Assembly:
    input = str(input)
    atomslist = ipd.pdb.readatoms(input, chainlist=True, assembly=assembly, **kw)
    assert isinstance(atomslist, list)
    components = ipd.atom.find_components_by_seqaln_rmsfit(atomslist, **kw)
    process_components(components, **kw)
    bodies = to_bodies(components.atoms, **kw)
    # print([b.summary() for b in bodies])
    return Assembly(bodies, components.frames, _atomslist=atomslist)

def to_bodies(atoms_or_bodies: 'bs.AtomArray', **kw):
    if all(map(ipd.atom.is_atomarray, atoms_or_bodies)):
        bodies = map(ipd.kwcurry(kw, ipd.atom.Body), atoms_or_bodies)
    return list(bodies)

def process_components(
    components: ipd.atom.Components,
    pickchain: str = 'largest',
    merge_chains: bool = True,
    min_chain_atoms: int = 0,
    **kw,
):
    for i, atoms, frames in components.enumerate('atoms frames', order=reversed):
        if len(atoms) < min_chain_atoms and i > 0:
            if components.frames[i - 1].shape == frames.shape:
                components.atoms[i - 1] += atoms
                components.atoms.pop(i)
                components.frames.pop(i)

@dataclass
class AsuSelector:
    bodyid: int = 9999
    frameid: int = 9999

    def __call__(self, assembly: Assembly) -> ipd.atom.Body:
        return assembly.body(self.bodyid, self.frameid)

def new_frame(asu, body, newasuframe, oldasuframe, oldsymframe):
    return oldsymframe

@toolz.curry
@dataclass
class NeighborhoodSelector:
    min_contacts: int = 10
    contact_dist: float = 7.0
    contact_byres: bool = False

    def __call__(self, asusel: AsuSelector, assembly: Assembly) -> Assembly:
        asu, nbrframe = asusel(assembly), {}
        nbrframe[asu] = [np.eye(4)]
        contactkw = dict(radius=self.contact_dist, residue_wise=self.contact_byres)
        asuskips = 0
        for ibod, ifrm, body, oldframe in assembly.enumerate_symbodies():
            if asu.isclose(body):
                assert (asuskips := asuskips + 1) == 1
                continue
            contacts = asu.contacts(body, **contactkw)
            # if not self.contact_check(contacts): continue
            if body not in nbrframe:
                m = body.get_metadata()
                # ic('newasu', ibod, ifrm, m.bodyid, m.frameid, body.pos[:3, 3])
                # ic(assembly.bodies[1].pos, assembly.frames[1][1])
                nbrframe[body] = [np.eye(4)]
            # ic(ibod, ifrm, body in nbrframe)
            # continue
            newasuframe = nbrframe[body][0]
            oldasuframe = oldframe
            oldsymframe = assembly.frames[ibod][ifrm]
            newsymframe = new_frame(asu, body, newasuframe, oldasuframe, oldsymframe)
            nbrframe[body].append(newsymframe)
        # ic(list(nbrframe.keys()))
        newframes = [np.stack(v) for v in nbrframe.values()]
        assert all(x.ndim == 3 for x in newframes)
        return Assembly(nbrframe.keys(), newframes, _source_assembly=assembly)

    def contact_check(self, contacts):
        return min(contacts.nuniq0, contacts.nuniq1) < self.min_contacts
