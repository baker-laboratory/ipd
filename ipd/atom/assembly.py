"""
class to represent a potentially very large biological assembly efficiently. stores only the asymmetric coordinates and does calculations based on transformed asu coods. uses a sweet transformable bounding volume hierarchy to do itersection tests
"""

import attrs
import numpy as np
import ipd

bs = ipd.lazyimport('biotite.structure')

def assembly(
    input: 'str|CifFile',
    assembly='largest',
    min_chain_atoms=0,
    **kw,
) -> 'Assembly':
    ic(input)
    atomslist = ipd.atom.load(input, chainlist=True, assembly=assembly, **kw)
    ic([len(a) for a in atomslist])
    assert isinstance(atomslist, list)
    components = ipd.atom.find_frames_by_seqaln_rmsfit(atomslist, **kw)
    for i, atoms, frames in components.enumerate('atoms frames', order=reversed):
        ic(i, len(atoms))
        ic(ipd.atom.chain_ranges(atoms))
        ic(frames.shape)
        # ic(ipd.sym.detect(frames).symelem)
        if len(atoms) < min_chain_atoms and i > 0:
            if True:
                components.atoms[i - 1] += atoms
                components.atoms.pop(i)
                components.frames.pop(i)

    bodies = to_bodies(components.atoms, **kw)
    print([b.summary() for b in bodies])
    return Assembly(bodies, components.frames)

def to_bodies(bodies, **kw):
    if all(map(ipd.atom.is_atomarray, bodies)):
        bodies = map(ipd.kwcurry(kw, ipd.atom.Body), bodies)
    return list(bodies)

@ipd.dev.subscriptable_for_attributes
@attrs.define(slots=False)
class Assembly:
    bodies: list[ipd.atom.Body] = attrs.field(converter=to_bodies)
    frames: list[np.ndarray]

    def __repr__(self):
        with ipd.dev.capture_stdio() as out:
            ipd.print_table(vars(self))
        return out.read()
