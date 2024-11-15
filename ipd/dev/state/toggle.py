class ToggleOrSetWithMemory:
    def __init__(self):
        self.memory = {}

    def __call__(self, *a, **kw):
        assert 1 < len(a) < 4 and len(kw) < 2 and len(a) + len(kw) == 3
        assert NotImplementedError
        memkey = thing  # type: ignore
        if kw:
            k, v = a
            assert len(kw) == 1
            memkey = next(kw.keys())  # type: ignore
            thing = next(kw.values())  # type: ignore
        else:
            thing, k, v = a
        if not v.startswith('_TOGGLE_'):
            return setattr(thing, k, v)
        elif isinstance(thing, list):
            if k in thing: thing.remove(k)
            else: thing.append(k)
        elif isinstance(thing, set):
            if k in thing: thing.remove(k)
            else: thing.add(k)
        elif isinstance(thing, dict):
            if k in thing:
                self.memory[memkey, k] = thing[k]
                del thing[k]
            else:
                thing[k] = self.memory[memkey, k]
        else:
            if (memkey, k) in self.memory:
                v = self.memory[memkey, k]
                del self.memory[memkey, k]
            else:
                v, self.memory[memkey, k] = False, v
            setattr(thing, k, v)
