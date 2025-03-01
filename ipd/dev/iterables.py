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
