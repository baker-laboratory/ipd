class ToggleOrSetWithMemory:
    def __init__(self):
        self.memory = {}

    def __call__(self, *a, **kw):
        assert 1 < len(a) < 4 and len(kw) < 2 and len(a) + len(kw) == 3

        memkey = thing
        if kw:
            k, v = a
            assert len(kw) == 1
            memkey = next(kw.keys())
            thing = next(kw.values())
        else:
            thing, k, v = a
        if v.startswith('_TOGGLE_'):
            if isinstance(thing, list):
                if v in thing: thing.remove(k)
                else: thing.append(k)
            else:
                if memkey in self.memory: v = self.memory[memkey]
                else: v, self.memory[memkey] = False, v
        setattr(thing, k, v)
