from typing import Sequence, Any

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

def order(seq: Sequence[Any]):
    return [a[1] for a in sorted((s, i) for i, s in enumerate(seq))]

def reorder(seq: Sequence[Any], idx: Sequence[int]):
    return [seq[i] for i in idx]
