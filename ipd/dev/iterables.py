from typing import Sequence, Any
import ipd

def nth(thing, n=0):
    iterator = iter(thing)
    try:
        for i in range(n):
            next(iterator)
        return next(iterator)
    except StopIteration:
        return None

first = nth

def head(thing, n=5, *, requireall=False, start=0):
    iterator, result = iter(thing), []
    try:
        for i in range(start):
            next(iterator)
        for i in range(n):
            result.append(next(iterator))
    except StopIteration:
        if requireall: return None
    return result

def order(seq: Sequence[Any], key=None):
    return [a[1] for a in sorted(((s, i) for i, s in enumerate(seq)), key=key)]

def reorder(seq: Sequence[Any], order: Sequence[int]):
    return [seq[i] for i in order]

def reorder_inplace(seq: list[Any], order: Sequence[int]):
    result = reorder(seq, order)
    for i, v in enumerate(result):
        seq[i] = v

def reorderer(order: Sequence[int]):

    def reorder_func(*seqs: list[Any]):
        for seq in seqs:
            reorder_inplace(seq, order)

    return reorder_func

def zipmaps(*args, order='key', intersection=False):
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
