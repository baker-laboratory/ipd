def enchanced_getitem(self, key):
    if ' ' in key: key = key.split()
    if not isinstance(key, str):
        return tuple(getattr(self, k) for k in key)
    return getattr(self, key)

def subscripable_for_attributes(cls):
    if hasattr(cls, '__getitem__'):
        raise TypeError(f'class {cls.__name__} alread has __getitem__')
    cls.__getitem__ = enchanced_getitem
    return cls
