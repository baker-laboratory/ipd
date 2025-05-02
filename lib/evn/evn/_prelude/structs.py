import sys
import dataclasses as dc

asdict = dc.asdict
# from typing import final
final = lambda x: x

if sys.version_info.minor > 9:
    struct = lambda cls: final(dc.dataclass(slots=True)(cls))
    basestruct = dc.dataclass(slots=True)
else:
    struct = lambda cls: final(dc.dataclass()(cls))
    basestruct = dc.dataclass()

mutablestruct = lambda cls: final(dc.dataclass()(cls))
basemutablestruct = dc.dataclass()

def field(dfac=dc.MISSING, *a, **kw):
    if dfac and 'default_factory' in kw:
        raise TypeError('default_factory specified twice (as arg0 dfac)')
    return dc.field(*a, default_factory=dfac, **kw)
