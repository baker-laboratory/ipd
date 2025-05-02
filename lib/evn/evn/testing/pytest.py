def has_pytest_mark(obj, mark):
    """Checks if an object has a specific pytest mark.

    Args:
        obj (Any): The object to check.
        mark (str): The name of the pytest mark to check for.

    Returns:
        bool: True if the object has the specified mark, False otherwise.

    Example:
    >>> import pytest
    >>> @pytest.mark.ci
    ... def test_example():
    ...     pass

    >>> print(has_pytest_mark(test_example, 'ci'))
    True
    >>> print(has_pytest_mark(test_example, 'skip'))
    False
    """
    return mark in [m.name for m in getattr(obj, 'pytestmark', ())]


def no_pytest_skip(obj):
    """Checks if an object does not have the `skip` pytest mark.

    Args:
        obj (Any): The object to check.

    Returns:
        bool: True if the object does not have the `skip` mark, False otherwise.

    Example:
    >>> import pytest
    >>> @pytest.mark.skip
    ... def test_example():
    ...     pass
    >>> print(no_pytest_skip(test_example))
    False
    """
    return not has_pytest_mark(obj, 'skip')


def get_pytest_params(func):
    """
    Detect if a function is decorated with @pytest.mark.parametrize and return the arguments.

    Args:
        func (Callable): The function to inspect.

    Returns:
        tuple[str, list] | None: A tuple of (argnames, argvalues) if decorated, else None.
    """
    for mark in getattr(func, 'pytestmark', []):
        if mark.name == 'parametrize':
            names, vals = mark.args
            names = list(map(str.strip, names.split(',')))
            if len(names) > 1:
                for v in vals:
                    assert len(names) == len(vals)
            return names, vals
    return None
