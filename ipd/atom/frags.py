import ipd
from ipd.atom.body import Body

@ipd.struct
class Frags:
    _bodies1: list[Body]
    _bodies2: list[Body]

@ipd.struct
class FragsOne2Many:
    body: Body
    mates: list[Body]
    cms: ipd.homog.ContactBlockMatrix

    def __post_init__(self):
        super([self.body], mates)
