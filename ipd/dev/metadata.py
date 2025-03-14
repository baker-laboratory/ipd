import copy
import ipd
from ipd.dev.decorators import iterize_on_first_param

@iterize_on_first_param(basetype='notlist')
def get_metadata(obj):
    """
    Retrieve metadata from an object.

    Args:
        obj (Any): The object from which to retrieve metadata.

    Returns:
        ipd.Bunch: The metadata stored in the object, or an empty `ipd.Bunch` if no metadata exists.
    """
    return vars(obj).get('__ipd_metadata__', ipd.Bunch())

def set_metadata(obj, data: 'dict|None' = None, **kw):
    """
    Set metadata on an object.

    If both `data` and keyword arguments are provided, they are merged.
    If `obj` is a list, metadata can be applied to each element in the list.
    If both `obj` and `data` are lists, they are zipped together, and metadata is applied pairwise.

    Args:
        obj (Any): The object or list of objects to which metadata should be added.
        data (Optional[dict]): A dictionary of metadata.
        **kw: Additional keyword arguments to include as metadata.

    Returns:
        Any: The modified object or list of objects.
    """
    if isinstance(obj, list):
        if isinstance(data, list):
            assert len(obj) == len(data)
            [set_metadata(o, d) for o, d in zip(obj, data)]
        else:
            [set_metadata(o, data) for o in obj]
        return
    data = data | kw if data else kw
    meta = obj.__dict__.setdefault('__ipd_metadata__', ipd.Bunch())
    meta.update(data)
    return obj

def sync_metadata(*objs):
    """
    Synchronize metadata across multiple objects.

    Merges metadata from all objects and applies the merged result to each object.

    Args:
        *objs (Any): A variable number of objects to synchronize metadata between.
    """
    data = ipd.dev.orreduce(get_metadata(obj) for obj in objs)
    for obj in objs:
        set_metadata(obj, data)
    return objs

def holds_metadata(cls):
    """
    Class decorator to enable metadata storage and retrieval.

    Adds `get_metadata`, `set_metadata`, and `sync_metadata` as class methods.
    Also modifies the class `__init__` method to accept metadata at initialization.

    Args:
        cls (type): The class to decorate.

    Returns:
        type: The modified class with metadata support.
    """

    def newinit(self, *a, **kw):
        initkw = ipd.kwcheck(kw, cls.__init_after_ipd_metadata__)
        metadata = {k: v for k, v in kw.items() if k not in initkw and k[0] == '_'}
        extra = {k for k in kw if k not in initkw and k not in metadata}
        if extra: raise TypeError(f"__init__() got an unexpected keyword argument(s): {', '.join(extra)}")
        metadata = {k[1:]: v for k, v in metadata.items()}

        self.set_metadata(metadata)
        cls.__init_after_ipd_metadata__(self, *a, **initkw)

    def newcopy(self):
        if self.__copy_after_ipd_metadata__:
            new = self.__copy_after_ipd_metadata__()
        else:
            new = ipd.dev.shallow_copy(self)
        new.__ipd_metadata__ = copy.copy(self.__ipd_metadata__)
        return new

    cls.__init_after_ipd_metadata__ = cls.__init__
    cls.__init__ = newinit
    cls.__copy_after_ipd_metadata__ = getattr(cls, '__copy__', None)
    cls.__copy__ = newcopy

    assert not any(hasattr(cls, name) for name in 'set_metadata get_metadata sync_metadata meta'.split())
    cls.set_metadata = set_metadata
    cls.get_metadata = get_metadata
    cls.sync_metadata = sync_metadata
    cls.meta = property(lambda self: get_metadata(self))

    return cls
