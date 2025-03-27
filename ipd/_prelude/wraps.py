import functools

def wraps(func):
    """Decorator to preserve the metadata of the original function.

    This decorator is used to wrap a function and preserve its metadata, such as
    the function name, docstring, and module. It is typically used when creating
    decorators that modify the behavior of a function. Unlike ipd.wraps, this
    version removes the original __doc__ to avoid warnings in sphinx

    Args:
        func (callable): The function to wrap.

    Returns:
        callable: The wrapped function with preserved metadata.
    """
    newfunc = functools.wraps(func)
    # try:
    #     func.__doc__ = None
    # except AttributeError:
    #     pass
    return newfunc
