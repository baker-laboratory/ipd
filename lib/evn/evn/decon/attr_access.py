"""
Decorators and Utilities for Enhanced Container Functionality
===============================================================

This module provides a collection of decorators and helper functions designed to extend the
behavior of functions and classes. Key features include:

- **Vectorization**: Use :func:`iterize_on_first_param` to automatically vectorize a function
  over its first parameter.
- **Enhanced Attribute Access**: The :func:`subscriptable_for_attributes` decorator adds support
  for subscriptable attribute access, fuzzy matching, enumeration, and grouping.
- **Utility Functions**: Other helpers (e.g. :func:`generic_get_keys`)
  provide common functionality for attribute and iterable handling.

Examples:
    Making a class subscriptable for attribute access::

        >>> @subscriptable_for_attributes
        ... class MyClass:
        ...     def __init__(self):
        ...         self.a = 1
        ...         self.b = 2
        >>> obj = MyClass()
        >>> obj['a']
        1
        >>> obj['a b']
        (1, 2)
"""

from typing import Any, Iterable
import evn


def NoneFunc():
    """This function does nothing and is used as a default placeholder."""
    pass


def subscriptable_for_attributes(cls: type[evn.C]) -> type[evn.C]:
    """Class decorator to enable subscriptable attribute access and enumeration.

    This decorator adds support for `__getitem__` and `enumerate` methods to a class
    using `generic_getitem_for_attributes` and `generic_enumerate`.

    Args:
        cls (type): The class to modify.

    Returns:
        type: The modified class.

    Example:
    >>> @subscriptable_for_attributes
    ... class MyClass:
    ...     def __init__(self):
    ...         self.x = 1
    ...         self.y = 2
    >>> obj = MyClass()
    >>> print(obj['x'])
    1
    >>> print(obj['x y'])  # (1, 2)
    (1, 2)
    >>> for i, x, y in obj.enumerate(['x', 'y']):
    ...     print(i, x, y)
    0 1 2

    Raises:
        TypeError: If the class already defines `__getitem__` or `enumerate`.
    """
    for member in 'enumerate fzf groupby'.split():
        if hasattr(cls, member):
            raise TypeError(f'class {cls.__name__} already has {member}')
    cls.__getitem__ = make_getitem_for_attributes(get=getattr)
    cls.fzf = make_getitem_for_attributes(get=getattr_fzf)
    cls.enumerate = generic_enumerate
    cls.groupby = generic_groupby
    cls.pick = make_getitem_for_attributes(provide='item')
    return cls


# helper functions


def generic_get_keys(obj, exclude: evn.FieldSpec = ()):
    """
    Retrieve keys or indices from an object.

    This function attempts to extract keys from an object. It checks for a ``keys()`` or ``items()``
    method, or if the object is a list returns its indices. Otherwise, it returns attribute names
    that pass the validity checks.

    :param obj: The object from which to extract keys.
    :param exclude: An iterable of keys to exclude. Defaults to an empty tuple.
    :return: A list of keys or indices.
    :rtype: list

    Example:
        >>> class A:
        ...     def __init__(self):
        ...         self.x = 1
        ...         self._y = 2
        >>> a = A()
        >>> generic_get_keys(a)
        ['x']
    """
    if hasattr(obj, 'keys') and callable(getattr(obj, 'keys')):
        return [k for k in obj.keys() if valid_element_name(k, exclude)]
    elif hasattr(obj, 'items'):
        return [k for k, v in obj.items() if valid_element_name(k, exclude)]
    elif isinstance(obj, list):
        return list(range(len(obj)))
    else:
        return [
            k for k in dir(obj) if valid_element_name_thorough(k, exclude)
            and not callable(getattr(obj, k))
        ]
    raise TypeError(f'dont know how to get elements from {obj}')


def generic_get_items(obj, all=False):
    """
    Retrieve key-value pairs from an object.

    This function returns a list of (key, value) pairs from the object. It supports objects that
    have an ``items()`` or ``keys()`` method, as well as lists (using indices) or attributes.

    :param obj: The object from which to extract items.
    :return: A list of (key, value) pairs.
    :rtype: list

    Example:
        >>> class A:
        ...     def __init__(self):
        ...         self.a = 1
        ...         self.b = 2
        >>> a = A()
        >>> generic_get_items(a)
        [('a', 1), ('b', 2)]
    """
    if hasattr(obj, 'items'):
        return [(k, v) for k, v in obj.items() if all or valid_element_name(k)]
    elif hasattr(obj, 'keys') and callable(getattr(obj, 'keys')):
        return [(k, getattr(obj, k)) for k in obj.keys()
                if all or valid_element_name(k)]
    elif isinstance(obj, list):
        return list(enumerate(obj))
    else:
        return [(k, getattr(obj, k)) for k in dir(obj)
                if (all or valid_element_name_thorough(k))
                and not callable(getattr(obj, k))]
    raise TypeError(f'dont know how to get elements from {obj}')


def valid_element_name(name, exclude=()):
    """
    Check if a name is valid based on naming conventions.

    A valid name must not start or end with an underscore and must not be in the excluded list.

    :param name: The name to check.
    :type name: str
    :param exclude: An iterable of names to exclude.
    :return: True if the name is valid, otherwise False.
    :rtype: bool

    Example:
        >>> valid_element_name('foo')
        True
        >>> valid_element_name('_bar')
        False
    """
    return not name[0] == '_' and not name[-1] == '_' and name not in exclude


def valid_element_name_thorough(name, exclude=()):
    """
    Thoroughly check if a name is valid by applying additional reserved name rules.

    In addition to the checks performed by :func:`valid_element_name`, this function also ensures
    that the name is not in a set of reserved element names.

    :param name: The name to check.
    :type name: str
    :param exclude: An iterable of names to exclude.
    :return: True if the name is valid, otherwise False.
    :rtype: bool

    Example:
        >>> valid_element_name_thorough('mapwise')
        False
    """
    return valid_element_name(name,
                              exclude) and name not in _reserved_element_names


_reserved_element_names = set('mapwise npwise valwise dictwise'.split())


def get_fields(
    obj, fields: evn.FieldSpec,
    exclude: evn.FieldSpec = ()) -> tuple[Iterable, bool]:
    """
    Determine and return the fields from an object.

    The function returns a tuple containing a list of field names and a boolean indicating whether
    multiple fields are expected.

    :param obj: The object from which to extract fields.
    :param fields: A field specification that may be a callable, a string, or an iterable.
    :param exclude: Fields to exclude from the result. Defaults to an empty tuple.
    :return: A tuple (fields, is_plural) where fields is a list of field names and is_plural is a bool.
    :rtype: tuple(list, bool)

    Example:
        >>> class A:
        ...     def __init__(self):
        ...         self.a = 1
        ...         self.b = 2
        >>> a = A()
        >>> get_fields(a, 'a')
        (['a'], False)
        >>> get_fields(a, 'a b')
        (['a', 'b'], True)
    """

    if callable(fields):
        fields = fields(obj)
    if fields is None:
        return generic_get_keys(obj, exclude=exclude), True
    if ' ' in fields:
        return evn.cast(str, fields).split(), True
    if isinstance(fields, str):
        return [fields], False
    return fields, True


def make_getitem_for_attributes(get=getattr, provide='value') -> 'Any':
    if provide not in ('value', 'item'):
        raise ValueError(f"provide must be 'value' or 'item', not {provide}")

    def getitem_for_attributes(self, fields: evn.FieldSpec, get=get) -> 'Any':
        """Enhanced `__getitem__` method to support attribute access with multiple keys.

        If the field is a string containing spaces, it will be split into a list of keys.
        If the field is a list of strings, it will return the corresponding attributes as a tuple.

        Args:
            field (list[str] | str): A single attribute name or a list of attribute names.

        Returns:
            Any: The attribute value(s) corresponding to the field(s).

        Example:
            >>> obj = MyClass()
            >>> value = obj['x']  # Single field
            >>> values = obj['x y z']  # Multiple keys as a string
            >>> values = obj[['x', 'y', 'z']]  # Multiple keys as a list
        """
        # try:
        field, plural = get_fields(self, fields)
        if provide == 'value':
            if plural:
                return tuple(get(self, k) for k in field)
            else:
                return get(self, field[0])
        if provide == 'item':
            if plural:
                return evn.Bunch((k, get(self, k)) for k in field)
            return (field[0], get(self, field[0]))

    # except AttributeError as e:
    #     # print(fields)
    #     if isinstance(self, evn.Bunch):
    #         if ' ' in self._conf('split'):
    #             print(fields)
    #             return self.__getitem__(fields)
    #     raise e from None

    return getitem_for_attributes


def generic_enumerate(self,
                      fields: evn.FieldSpec = None,
                      order=lambda x: x) -> evn.EnumerIter:
    """
    Enhanced enumerate method to iterate over multiple attributes simultaneously.

    This method retrieves the specified fields from the object and yields an enumeration of the field values.
    If the fields are provided as a string with spaces, they will be split into a list of field names.

    :param fields: A field specification (string or list of strings) indicating which attributes to enumerate.
                   If None, all valid attributes are enumerated.
    :param order: A function to order the enumeration indices and values. Defaults to identity.
    :return: An iterator yielding tuples containing the index and the corresponding attribute values.
    :rtype: iterator

    Example:
        >>> class A:
        ...     def __init__(self):
        ...         self.x = [1, 2]
        ...         self.y = [3, 4]
        ...
        ...     __getitem__ = make_getitem_for_attributes()
        ...     enumerate = generic_enumerate
        >>> a = A()
        >>> list(a.enumerate('x y'))
        [(0, 1, 3), (1, 2, 4)]
        >>> @evn.subscriptable_for_attributes
        ... class MyClass:
        ...     def __init__(self):
        ...         self.x = range(5)
        ...         self.y = range(5, 10)
        >>> obj = MyClass()
        >>> for i, x, y in obj.enumerate('x y'):
        ...     print(i, x, y)
        0 0 5
        1 1 6
        2 2 7
        3 3 8
        4 4 9
    """
    if fields is None:
        fields = generic_get_keys(self)
    vals = self[fields]
    try:
        fields = list(zip(*vals))
    except TypeError:
        fields = [vals]
    idx = range(len(fields))
    for i, vals in zip(order(idx), order(fields)):
        yield i, *vals


def generic_groupby(
    self,
    groupby: evn.FieldSpec,
    fields: evn.FieldSpec = None,
    convert=None,
) -> evn.EnumerListIter:
    """
    Group object attributes by a specified key.

    This method groups attributes based on the values obtained from the `groupby` field specification.
    Optionally, only a subset of fields may be selected and a conversion function applied to the grouped values.

    :param groupby: A field specification (or callable) to determine group keys.
    :param fields: A field specification indicating which fields to group. Defaults to None (all keys).
    :param convert: An optional function to convert the grouped values.
    :return: An iterator over grouped data. Each iteration yields a group key and the corresponding grouped values.
    :rtype: iterator

    Example:
        >>> class A:
        ...     def __init__(self):
        ...         self.a = [1, 2, 3, 4]
        ...         self.group = ['x', 'x', 'y', 'y']
        ...
        ...     __getitem__ = make_getitem_for_attributes()
        ...     groupby = generic_groupby
        >>> a = A()
        >>> list(a.groupby('group', 'a'))
        [('x', (1, 2)), ('y', (3, 4))]
    """
    exclude = None
    splat = isinstance(fields, str)
    if callable(groupby):
        groupby = groupby(self)
    else:
        groupby, plural = get_fields(self, groupby)
        exclude = groupby
        if not plural:
            groupby = groupby[0]
        groupby = self[groupby]
    fields, _ = get_fields(self, fields, exclude=exclude)
    vals = self[fields]
    groups = dict()
    for k, v in zip(groupby, zip(*vals)):
        groups.setdefault(k, []).append(v)
    for group, vals in groups.items():
        vals = zip(*vals)
        if convert:
            vals = map(convert, vals)
        if splat:
            yield group, *vals
        else:
            yield group, evn.Bunch(zip(fields, vals))


def is_fuzzy_match(sub, string):
    """
    Check if one string is a fuzzy subsequence of another.

    This function checks that the first two characters of `sub` match those of `string`
    and then verifies that all characters in `sub` appear in order in `string`.

    :param sub: The subsequence to check.
    :param string: The string to search within.
    :return: True if `sub` is a fuzzy match of `string`, False otherwise.
    :rtype: bool

    Example:
        >>> is_fuzzy_match('abc', 'ab2c3')
        True
        >>> is_fuzzy_match('acb', 'ab2c3')
        False
    """
    if sub[:2] != string[:2]:
        return False
    i, j = 0, 0
    while i < len(sub) and j < len(string):
        if sub[i] == string[j]:
            i += 1
        j += 1
    return i == len(sub)


def getattr_fzf(obj, field):
    """
    Retrieve an attribute from an object using fuzzy matching.

    This function uses fuzzy matching to find attribute names that are similar to the given field.
    If a single match is found, its value is returned. If multiple matches are found, an error is raised.

    :param obj: The object to search.
    :param field: The field name (or fuzzy substring) to search for.
    :return: The attribute value corresponding to the matched field.
    :rtype: Any
    :raises AttributeError: If no matching attribute is found or if multiple ambiguous matches exist.

    Example:
        >>> class A:
        ...     def __init__(self):
        ...         self.abc = 1
        ...         self.xyz = 2
        ...
        ...     fzf = make_getitem_for_attributes(get=getattr_fzf)
        >>> a = A()
        >>> a.fzf('ab')
        1
    """
    fields = generic_get_keys(obj, exclude=())
    candidates = [f for f in fields if is_fuzzy_match(field, f)]
    if not candidates:
        raise AttributeError(f'no attribute found for {field}')
    if len(candidates) == 1:
        return getattr(obj, candidates[0])
    raise AttributeError(
        f'multiple attributes found for {field}: {candidates}')
