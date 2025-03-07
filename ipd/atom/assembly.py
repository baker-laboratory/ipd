"""
class to represent a potentially very large biological assembly efficiently. stores only the asymmetric coordinates and does calculations based on transformed asu coods. uses a sweet transformable bounding volume hierarchy to do itersection tests
"""

import attrs
import ipd

bs = ipd.lazyimport('biotite.structure')

@attrs.define
class Assembly:
    orig: 'bs.AtomArray'
