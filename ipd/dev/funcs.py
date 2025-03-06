import dis
import re
import functools
import inspect
import operator
from pathlib import Path
from typing import TypeVar, Callable

import ipd

T = TypeVar('T')

def addreduce(iterable):
    return functools.reduce(operator.add, iterable)

def call_with_args_from(
    argpool,
    func: Callable[..., T],
    timed: bool = False,
    dryrun=False,
    strict=False,
    **kw,
) -> T:
    params = func_params(func)
    required_params = func_params(func, required_only=True)
    if timed: func = ipd.dev.timed(func)
    for p in params:
        if p not in argpool and p in required_params:
            raise ValueError(
                f'function: {func.__name__}{inspect.signature(func)} requred arg {p} not argpool: {list(argpool.keys())}'
            )
    args = {p: argpool[p] for p in params if p in argpool}
    if dryrun: return None
    return func(**args)

class InfixOperator:

    def __init__(self, func, *a, **kw):
        self.func, self.kw, self.a = func, kw, a

    def __ror__(self, lhs, **kw):
        return InfixOperator(lambda rhs: self.func(lhs, rhs, *self.a, **self.kw))

    def __or__(self, rhs):
        return self.func(rhs)

    def __call__(self, *a, **kw):
        return self.func(*a, **kw)

def is_iterizeable(arg, basetype=None, splitstr=False):
    if isinstance(arg, str) and ' ' in arg: return True
    if basetype and isinstance(arg, basetype): return False
    if hasattr(arg, '__iter__'): return True
    return False

def iterize_on_first_param(func0=None, *, basetype=str, splitstr=True, asdict=False, asbunch=False):
    """Decorator that vectorizes a function over its first parameter.

    This decorator enables a function to handle both single values and iterables as its
    first parameter. When an iterable is passed, the function is applied to each item
    individually and returns a list of results. Otherwise, the function is called normally.

    Args:
        *metaargs: Optional positional arguments. If the first argument is callable,
            it is treated as the function to decorate (allowing for decorator use without
            parentheses).
        **metakw: Keyword arguments passed to the is_iterizeable() function for controlling
            iteration behavior. Common parameters include:
            - basetype: Type or tuple of types that should not be iterated over even if
              they have __iter__ method (e.g., strings, Path objects).

    Returns:
        callable: A decorated function that can handle both scalar and iterable inputs
        for its first parameter.

    Examples:
        Basic usage with default behavior:

        >>> @iterize_on_first_param
        ... def square(x):
        ...     return x * x
        ...
        >>> square(5)
        25
        >>> square([1, 2, 3])
        [1, 4, 9]

        With custom basetype parameter:

        >>> @iterize_on_first_param(basetype=str)
        ... def process(item):
        ...     return len(item)
        ...
        >>> process("hello")  # Treated as scalar despite being iterable
        5
        >>> process(["hello", "world"])
        [5, 5]

    Notes:
        - The decorator can be applied with or without parentheses.
        - The decorated function preserves its name, docstring, and other attributes.
        - For string and path-like objects, consider using iterize_on_first_param_path
          which is preconfigured with basetype=(str, Path).
    """

    def deco(func):

        @functools.wraps(func)
        def wrapper(arg0, *args, **kw):
            if is_iterizeable(arg0, basetype=basetype, splitstr=splitstr):
                if splitstr and isinstance(arg0, str) and ' ' in arg0:
                    arg0 = arg0.split()
                if asbunch:
                    return ipd.Bunch({a0: func(a0, *args, **kw) for a0 in arg0})
                if asdict:
                    return {a0: func(a0, *args, **kw) for a0 in arg0}
                else:
                    return [func(a0, *args, **kw) for a0 in arg0]
            return func(arg0, *args, **kw)

        return wrapper

    if func0:  # handle case with no call/args
        assert callable(func0)
        return deco(func0)
    return deco

iterize_on_first_param_path = iterize_on_first_param(basetype=(str, Path))

def kwcall(func, kw, *a, **kwargs):
    """Call a function with filtered keyword arguments.

    This function merges provided keyword arguments, filters them to match only those
    accepted by the target function using kwcheck, and then calls the function with
    these filtered arguments. **kwargs take precedence over kw args.

    Args:
        func (callable): The function to call.
        kw (dict): Primary dictionary of keyword arguments.
        *a: Positional arguments to pass to the function.
        **kwargs: Additional keyword arguments that will be merged with kw.

    Returns:
        Any: The return value from calling func.

    Examples:
        >>> def example_function(x, y, z=3):
        ...     return x + y + z
        ...
        >>> args = {'x': 1, 'y': 2, 'extra_arg': 'ignored'}
        >>> kwcall(example_function, args)
        6
        >>> kwcall(example_function, {'x': 1}, y=2, z=10)
        13

    Note:
        This function is useful for calling functions with dictionaries that may contain
        extraneous keys not accepted by the target function. It combines the functionality
        of dictionary merging and keyword argument filtering.

    See Also:
        kwcheck: The underlying function used to filter keyword arguments.
    """
    kw = kw | kwargs
    kw = kwcheck(kw, func)
    return func(*a, **kw)

def kwcheck(kw, func=None, checktypos=True):
    """
    Filter keyword arguments to match only those accepted by the target function.

    This function examines a dictionary of keyword arguments and returns a new
    dictionary containing only the keys that are valid parameter names for the
    specified function. When no function is explicitly provided, it automatically
    detects the function for which this kwcheck call is being used as an argument.

    Parameters
    ----------
    kw : dict
        Dictionary of keyword arguments to filter.
    func : callable, optional
        Target function whose signature will be used for filtering.
        If None, automatically detects the calling function.
    checktypos : bool, default=True
        Whether to check for possible typos in argument names.
        If True, raises TypeError for arguments that closely match valid parameters.

    Returns
    -------
    dict
        Filtered dictionary containing only valid keyword arguments for the target function.

    Raises
    ------
    ValueError
        If func is None and the automatic detection of the calling function fails.
    TypeError
        If checktypos is True and a likely typo is detected in the argument names.

    Examples
    --------
    >>> def my_function(a, b, c=3):
    ...     pass
    ...
    >>> kwargs = {'a': 1, 'b': 2, 'd': 4}
    >>> filtered_kwargs = kwcheck(kwargs, my_function)
    >>> filtered_kwargs
    {'a': 1, 'b': 2}

    >>> # Using it directly in a function call
    >>> my_function(**kwcheck(kwargs))

    Notes
    -----
    When used with checktypos=True, this function helps detect possible misspelled
    parameter names, improving developer experience by providing helpful error messages.
    """
    func = func or get_function_for_which_call_to_caller_is_argument()
    if not callable(func): raise TypeError('Couldn\'t get function for which kwcheck(kw) is an argument')
    params = func_params(func)
    newkw = {k: v for k, v in kw.items() if k in params}
    if checktypos:
        unused = kw.keys() - newkw.keys()
        unset = params - newkw.keys()
        for arg in unused:
            if typo := ipd.dev.find_close_strings(arg, unset):
                raise TypeError(f'{func.__name__} got unexpected arg {arg}, did you mean {typo}')
    return newkw

def get_function_for_which_call_to_caller_is_argument():
    """
    returns the function being called with an arg that is a call to enclosing function, if any

    finds TARGET_FUNCTION, if any
    def process_args(kw):
        uncle_func = get_function_for_which_call_to_caller_is_argument()
        assert uncle_func == TARGET_FUNCTION
        ...
        return kw

    TARGET_FUNCTION(arg1, arg2, arg3=7, **process_args(kw))
    """
    frame = inspect.currentframe().f_back.f_back  # grandparent
    frame_info = inspect.getframeinfo(frame)
    code = frame.f_code
    bytecode = dis.Bytecode(code)
    current_offset = frame.f_lasti
    for instr in bytecode:
        if instr.offset < current_offset and (instr.opname == 'LOAD_GLOBAL' or instr.opname == 'LOAD_NAME'):
            potential_func_name = instr.argval
            if potential_func_name != 'kwcheck' or True:
                func = frame.f_globals.get(potential_func_name) or frame.f_locals.get(potential_func_name)
                if func: return func

def filter_namespace_funcs(namespace, prefix='test_', only=(), re_only=(), exclude=(), re_exclude=()):
    if only or re_only:
        allfuncs = [k for k, v in namespace.items() if callable(v)]
        allfuncs = list(filter(lambda s: s.startswith(prefix), allfuncs))
        namespace_copy = namespace.copy()
        for func in allfuncs:
            del namespace[func]
        for func in only:
            namespace[func] = namespace_copy[func]
        for func_re in re_only:
            for func in allfuncs:
                if re.match(func_re, func):
                    namespace[func] = namespace_copy[func]
    for func in exclude:
        if func in namespace: del namespace[func]
    allfuncs = [k for k, v in namespace.items() if callable(v)]
    allfuncs = list(filter(lambda s: s.startswith(prefix), allfuncs))
    for func_re in re_exclude:
        for func in allfuncs:
            if re.match(func_re, func): del namespace[func]

def param_is_required(param):
    return param.default is param.empty and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)

@functools.lru_cache
def func_params(func, required_only=False):
    """
    Returns a list of names of the non-default parameters for a function.

    Args:
        func: The function to inspect

    Returns:
        list: Names of parameters that don't have default values

    Examples:
        >>> def example(a, b, c=3, d=4):
        ...     pass
        >>> get_non_default_params(example)
        ['a', 'b']
    """
    signature = inspect.signature(func)
    params = inspect.signature(func).parameters
    if required_only:
        params = {k: param for k, param in params.items() if param_is_required(param)}
    return params

def has_pytest_mark(obj, mark):
    return mark in [m.name for m in getattr(obj, 'pytestmark', ())]

def no_pytest_skip(obj):
    return not has_pytest_mark(obj, 'skip')
