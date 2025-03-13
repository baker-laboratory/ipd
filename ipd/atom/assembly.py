"""
class to represent a potentially very large biological assembly efficiently. stores only the asymmetric coordinates and does calculations based on transformed asu coods. uses a sweet transformable bounding volume hierarchy to do itersection tests
"""

import attrs
import numpy as np
import toolz

import ipd

bs = ipd.lazyimport('biotite.structure')

@ipd.dev.holds_metadata
@ipd.dev.subscriptable_for_attributes
@attrs.define(slots=False, init=False)
class Assembly:
    bodies: list[ipd.atom.Body]
    frames: list[np.ndarray]
    __ipd_metadata__: ipd.Bunch = ipd.Bunch()

    def body(self, ibody=0, iframe=0):
        return self.bodies[ibody].movedby(self.frames[ibody][iframe])

    def symbodyids(self):
        for i in range(len(self.bodies)):
            for j in range(len(self.frames[i])):
                yield i, j

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
    return Assembly(bodies, components.frames, atomslist=atomslist)

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

@attrs.define
class AsuSelector:
    bodyid: int = 9999
    frameid: int = 9999

    def __attrs_post_init__(self):
        assert self.bodyid >= 0 and self.frameid >= 0, f'{self.bodyid=}, {self.frameid=}'

    def __call__(self, assembly: Assembly) -> ipd.atom.Body:
        return assembly.bodies[self.bodyid].movedby(assembly.frames[self.bodyid][self.frameid])

@toolz.curry
@attrs.define
class NeighborhoodSelector:
    min_contacts: int = 10
    contact_dist: float = 7.0
    contact_byres: bool = False

    def __attrs_post_init__(self):
        pass

    def __call__(self, asusel: AsuSelector, origassembly: Assembly) -> Assembly:
        asu = asusel(origassembly)
        bod = [asu]
        frames = [np.eye(4)[None]]
        return Assembly(bod, frames, origassembly)
