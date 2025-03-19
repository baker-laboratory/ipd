import dataclasses
import typing

struct = dataclasses.dataclass(frozen=True)
mutablestruct = dataclasses.dataclass()
finalstruct = typing.final(dataclasses.dataclass(frozen=True))
finalmutablestruct = typing.final(dataclasses.dataclass())

def field(dfac=None, *a, **kw):
    if dfac and 'default_factory' in kw:
        raise TypeError("default_factory specified twice (as arg0 dfac)")
    return dataclasses.field(*a, **kw)
