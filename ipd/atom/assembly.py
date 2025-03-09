"""
class to represent a potentially very large biological assembly efficiently. stores only the asymmetric coordinates and does calculations based on transformed asu coods. uses a sweet transformable bounding volume hierarchy to do itersection tests
"""

import attrs
import numpy as np
import ipd

bs = ipd.lazyimport('biotite.structure')

def assembly(input: str, **kw) -> 'Assembly':
    atomslist = ipd.atom.load(input, chainlist=True, assembly='largest')
    assert isinstance(atomslist, list)
    components = ipd.atom.find_frames_by_seqaln_rmsfit(atomslist)
    ic(components)
    return Assembly(components.atoms, components.frames)

@attrs.define
class Assembly:
    atoms: list[ipd.atom.Body]
    frames: list[np.ndarray]
