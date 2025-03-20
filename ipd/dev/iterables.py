from typing import Sequence, Any
import numpy as np
import typing
import ipd
from ipd.dev.strings import ascii_chars

T = typing.TypeVar('T')

def nth(thing: ipd.Iterable[T], n: int = 0) -> T:
    iterator = iter(thing)
    try:
        for i in range(n):
            next(iterator)
        return next(iterator)
    except StopIteration:
        return None

first = nth

def head(thing: ipd.Iterable[T], n=5, *, requireall=False, start=0) -> list[T]:
    iterator, result = iter(thing), []
    try:
        for i in range(start):
            next(iterator)
        for i in range(n):
            result.append(next(iterator))
    except StopIteration:
        if requireall: return None
    return result

def order(seq: Sequence[Any], key=None) -> list[int]:
    return [a[1] for a in sorted(((s, i) for i, s in enumerate(seq)), key=key)]

def reorder(seq: Sequence[T], order: Sequence[int]) -> Sequence[T]:
    return [seq[i] for i in order]

def reorder_inplace(seq: list[Any], order: Sequence[int]) -> None:
    result = reorder(seq, order)
    for i, v in enumerate(result):
        seq[i] = v

def reorderer(order: Sequence[int]) -> ipd.Callable[[T], T]:

    def reorder_func(*seqs: list[Any]):
        for seq in seqs:
            reorder_inplace(seq, order)

    return reorder_func

def zipmaps(*args: dict[str, T], order='key', intersection=False) -> dict[str, tuple[T, ...]]:
    if not args: raise ValueError('zipmaps requires at lest one argument')
    if intersection: keys = ipd.dev.andreduce(set(map(str, a.keys())) for a in args)
    else: keys = ipd.dev.orreduce(set(map(str, a.keys())) for a in args)
    if order == 'key': keys = sorted(keys)
    if order == 'val': keys = sorted(keys, key=lambda k: args[0].get(k, ipd.dev.NA))
    result = type(args[0])({k: tuple(a.get(k, ipd.dev.NA) for a in args) for k in keys})
    return result

def zipitems(*args, **kw):
    zipped = zipmaps(*args, **kw)
    for k, v in zipped.items():
        yield k, *v

@ipd.dc.dataclass
class UniqueIDs:
    alphabet: Sequence = ascii_chars
    idmap: dict = ipd.dc.field(default_factory=dict)
    offset: int = 0

    def __call__(self, ids: np.ndarray, reset=False):
        if reset:
            self.offset += len(self.idmap)
            self.idmap.clear()
        uniq = set(np.unique(ids))
        for cid in uniq - set(self.idmap):
            self.idmap[str(cid)] = self.alphabet[(len(self.idmap) - self.offset) % len(self.alphabet)]
        newids = ipd.copy(ids)
        for u in uniq:
            newids[ids == u] = self.idmap[u]
        # ipd.icv(self.idmap)
        return newids
