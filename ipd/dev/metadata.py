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

    Examples:
        >>> class Example:
        ...     pass
        >>> obj = Example()
        >>> set_metadata(obj, {'key': 'value'})
        >>> get_metadata(obj).key
        'value'
        >>> obj2 = Example()
        >>> get_metadata(obj2)  # No metadata, returns empty Bunch
        Bunch()
    """
    return vars(obj).get('__ipd_metadata__', ipd.Bunch())

def set_metadata(obj, data=None, **kw):
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

    Examples:
        >>> class Example:
        ...     pass
        >>> obj = Example()
        >>> set_metadata(obj, {'key': 'value'})
        >>> get_metadata(obj).key
        'value'

        >>> obj_list = [Example(), Example()]
        >>> set_metadata(obj_list, {'shared_key': 'shared_value'})
        >>> get_metadata(obj_list[0]).shared_key
        'shared_value'

        >>> obj_list = [Example(), Example()]
        >>> set_metadata(obj_list, [{'key1': 'value1'}, {'key2': 'value2'}])
        >>> get_metadata(obj_list[0]).key1
        'value1'
        >>> get_metadata(obj_list[1]).key2
        'value2'
    """
    if isinstance(obj, list) and isinstance(data, list):
        for a, d in zip(obj, data):
            set_metadata(a, d)
    elif isinstance(obj, list):
        for a in obj:
            set_metadata(a, data)
    else:
        if data: data.update(kw)
        data = data or kw
        assert not isinstance(data, list)
        if not hasattr(obj, '__ipd_metadata__'):
            setattr(obj, '__ipd_metadata__', ipd.Bunch())
        obj.__ipd_metadata__.update(data)

def sync_metadata(*objs):
    """
    Synchronize metadata across multiple objects.

    Merges metadata from all objects and applies the merged result to each object.

    Args:
        *objs (Any): A variable number of objects to synchronize metadata between.

    Examples:
        >>> class Example:
        ...     pass
        >>> obj1 = Example()
        >>> obj2 = Example()
        >>> set_metadata(obj1, {'key1': 'value1'})
        >>> set_metadata(obj2, {'key2': 'value2'})
        >>> sync_metadata(obj1, obj2)
        >>> get_metadata(obj1).key2
        'value2'
        >>> get_metadata(obj2).key1
        'value1'
    """
    data = ipd.dev.orreduce(get_metadata(obj) for obj in objs)
    for obj in objs:
        set_metadata(obj, data)

def holds_metadata(cls):
    """
    Class decorator to enable metadata storage and retrieval.

    Adds `get_metadata`, `set_metadata`, and `sync_metadata` as class methods.
    Also modifies the class `__init__` method to accept metadata at initialization.

    Args:
        cls (type): The class to decorate.

    Returns:
        type: The modified class with metadata support.

    Examples:
        >>> @holds_metadata
        ... class Example:
        ...     def __init__(self, value):
        ...         self.value = value
        >>> obj = Example(value=42, key='test_key')
        >>> get_metadata(obj).key
        'test_key'
        >>> obj2 = Example(value=99)
        >>> set_metadata(obj2, {'key2': 'value2'})
        >>> sync_metadata(obj, obj2)
        >>> get_metadata(obj).key2
        'value2'
        >>> get_metadata(obj2).key
        'test_key'
    """

    def newinit(self, *a, **kw):
        initkw = ipd.kwcheck(kw, cls.__init_after_ipd_metadata__)
        metadata = {k: v for k, v in kw.items() if k not in initkw}
        self.set_metadata(metadata)
        cls.__init_after_ipd_metadata__(self, *a, **initkw)

    cls.__init_after_ipd_metadata__ = cls.__init__
    cls.__init__ = newinit

    cls.set_metadata = set_metadata
    cls.get_metadata = get_metadata
    cls.sync_metadata = sync_metadata

    return cls
