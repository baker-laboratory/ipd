"""
Utility functions for working with Python callables, frames, and arguments.

This module provides a collection of utility functions to simplify common Python tasks related to:
- Inspecting and manipulating function arguments.
- Reducing iterables using operators.
- Handling local variables in the call stack.
- Filtering namespaces.
- Working with pytest marks.

Functions:
----------
- `picklocals` – Access local variables from the caller's caller frame.
- `opreduce` – Reduce an iterable using a specified operator.
- `kwcall` – Call a function with filtered keyword arguments.
- `kwcheck` – Filter keyword arguments to match accepted function parameters.
- `filter_namespace_funcs` – Filter functions in a namespace by patterns.
- `param_is_required` – Check if a function parameter is required.
- `func_params` – Retrieve parameters of a function.
- `has_pytest_mark` – Check if an object has a specific pytest mark.
- `no_pytest_skip` – Check if an object does not have the `skip` pytest mark.

Examples:
---------
Example for `picklocals`:
    >>> def example():
    ...     x = [10, 20, 30]
    ...     print(evn.meta.picklocals('x'))  # [10, 20, 30]
    ...     print(evn.meta.picklocals('x', 1))  # 20
    >>> example()
    [10, 20, 30]
    20

Example for `opreduce`:
    >>> from operator import add, mul
    >>> opreduce(add, [1, 2, 3, 4])
    10
    >>> opreduce('mul', [1, 2, 3, 4])
    24

Example for `kwcall`:
    >>> def example_function(x, y, z=3):
    ...     return x + y + z
    >>> args = {'x': 1, 'y': 2, 'extra_arg': 'ignored'}
    >>> kwcall(args, example_function)
    6
    >>> kwcall({'x': 1}, example_function, y=2, z=10)
    13

Example for `kwcheck`:
    >>> def my_function(a, b, c=3):
    ...     pass
    >>> kwargs = {'a': 1, 'b': 2, 'd': 4}
    >>> filtered_kwargs = kwcheck(kwargs, my_function)
    >>> filtered_kwargs
    {'a': 1, 'b': 2}

Example for `filter_namespace_funcs`:
    >>> def test_func1():
    ...     pass
    >>> def test_func2():
    ...     pass
    >>> def helper_func():
    ...     pass
    >>> ns = dict(test_func1=test_func1, test_func2=test_func2, helper_func=helper_func)
    >>> filtered_ns = filter_namespace_funcs(ns, prefix='test_', only=('test_func1',))
    >>> print(filtered_ns.keys())
    dict_keys(['helper_func', 'test_func1'])

See Also:
---------
- `inspect` – Python’s standard library for introspecting live objects.
- `functools` – Higher-order functions and operations on callable objects.
- `operator` – Standard library for functional-style operators.
- `pytest.mark` – Markers for controlling pytest test execution.

"""

import dis
import re
import functools
import inspect
import operator

import evn

T, P, R, F = evn.basic_typevars('TPRF')


def instanceof(obj_or_types, types=None):
    """wrapper so isinstane can be called with kwargs"""
    if types:
        return isinstance(obj_or_types, types)
    return lambda obj: isinstance(obj, obj_or_types)


def picklocals(name, idx=None, asdict=False):
    """Accesses a local variable from the caller's caller frame.

    This function retrieves the value of a local variable from the frame two levels up in the call stack.
    If `idx` is provided, it will index into the value (if it's indexable).

    Args:
        name (str): The name of the local variable to retrieve.
        idx (int, optional): If provided, returns `val[idx]` instead of `val`. Defaults to None.

    Returns:
        Any: The value of the local variable or the indexed value if `idx` is provided.

    Example:
        >>> def example():
        ...     x = [10, 20, 30]
        ...     print(evn.meta.picklocals('x'))  # [10, 20, 30]
        ...     print(evn.meta.picklocals('x', 1))  # 20
        >>> example()
        [10, 20, 30]
        20

    """
    single = False
    if isinstance(name, str):
        name, single = name.split(), True
        single &= len(name) == 1

    result = {}
    for n in name:
        # if sys.version_info.minor < 12:
            # val = inspect.currentframe().f_back.f_locals[n]  # type: ignore
        # else:
        val = inspect.currentframe().f_back.f_locals[n]  # type: ignore
        result[n] = val if idx is None else val[idx]
    if asdict: return result
    result = list(result.values())
    return result[0] if single else result


def opreduce(op, iterable):
    """Reduces an iterable using a specified operator or function.

    This function applies a binary operator or callable across the elements of an iterable, reducing it to a single value.
    If `op` is a string, it will look up the corresponding operator in the `operator` module.

    Args:
        op (str | callable): A callable or the name of an operator from the `operator` module (e.g., 'add', 'mul').
        iterable (iterable): The iterable to reduce.

    Returns:
        Any: The reduced value.

    Raises:
        AttributeError: If `op` is a string but not a valid operator in the `operator` module.
        TypeError: If `op` is not a valid callable or operator.

    Example:
        >>> from operator import add, mul
        >>> print(opreduce(add, [1, 2, 3, 4]))  # 10
        10
        >>> print(opreduce('mul', [1, 2, 3, 4]))  # 24
        24
        >>> print(opreduce(lambda x, y: x * y, [1, 2, 3, 4]))  # 24
        24
    """
    if isinstance(op, str):
        op = getattr(operator, op)
    return functools.reduce(op, iterable)


for op in 'add mul matmul or_ and_'.split():
    opname = op.strip('_')
    globals()[f'{opname}reduce'] = functools.partial(opreduce,
                                                     getattr(operator, op))


def kwcall(kw: evn.KW, func: F, *a: P.args, **kwargs: P.kwargs) -> R:
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
        >>> args = {'x': 1, 'y': 2, 'extra_arg': 'ignored'}
        >>> kwcall(args, example_function)
        6
        >>> kwcall({'x': 1}, example_function, y=2, z=10)
        13

    Note:
        This function is useful for calling functions with dictionaries that may contain
        extraneous keys not accepted by the target function. It combines the functionality
        of dictionary merging and keyword argument filtering.

    See Also:
        kwcheck: The underlying function used to filter keyword arguments.
    """
    return func(*a, **kwcheck(kw | kwargs, func))


def kwcheck(kw: evn.KW, func=None, checktypos=True) -> evn.KW:
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
    if not callable(func):
        raise TypeError(
            "Couldn't get function for which kwcheck(kw) is an argument")
    params = func_params(func)
    takeskwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD
                      for param in params.values())
    if takeskwargs:
        return kw
    newkw = {k: v for k, v in kw.items() if k in params}
    if checktypos:
        unused = kw.keys() - newkw.keys()
        unset = params - newkw.keys()
        for arg in unused:
            if typo := evn.meta.find_close_argnames(arg, unset, cutoff=0.8):
                raise TypeError(
                    f'{func.__name__} got unexpected arg {arg}, did you mean {typo}'
                )
    return newkw


def get_function_for_which_call_to_caller_is_argument():
    """
    returns the function being called with an arg that is a call to enclosing function, if any

    >>> def FIND_THIS_FUNCTION(*a, **kw): ...
    >>> def CALLED_TO_PRODUCE_ARGUMENT(**kw):
    ...     uncle_func = get_function_for_which_call_to_caller_is_argument()
    ...     print('detected caller:', uncle_func.__name__)
    ...     assert uncle_func == FIND_THIS_FUNCTION
    ...     ...
    ...     return kw
    >>> FIND_THIS_FUNCTION(1, 2, CALLED_TO_PRODUCE_ARGUMENT(), 3)
    detected caller: FIND_THIS_FUNCTION
    """
    frame = inspect.currentframe().f_back.f_back  # grandparent
    # frame_info = inspect.getframeinfo(frame)
    code = frame.f_code
    bytecode = dis.Bytecode(code)
    current_offset = frame.f_lasti
    for instr in bytecode:
        if instr.offset < current_offset and (instr.opname == 'LOAD_GLOBAL'
                                              or instr.opname == 'LOAD_NAME'):
            potential_func_name = instr.argval
            if potential_func_name != 'kwcheck' or True:
                func = frame.f_globals.get(
                    potential_func_name) or frame.f_locals.get(
                        potential_func_name)
                if func:
                    return func


def filter_namespace_funcs(namespace,
                           prefix='test_',
                           only=(),
                           re_only=(),
                           exclude=(),
                           re_exclude=()):
    """Filters functions in a namespace based on specified inclusion and exclusion rules.

    This function filters out functions from the given `namespace` based on:
    - A `prefix` that functions must start with.
    - Explicit names (`only`) or regex patterns (`re_only`) to include.
    - Explicit names (`exclude`) or regex patterns (`re_exclude`) to exclude.

    Args:
        namespace (dict): The namespace (usually `globals()` or `locals()`) to filter.
        prefix (str, optional): Only keep functions that start with this prefix. Defaults to `'test_'`.
        only (tuple, optional): Names of functions to explicitly keep. Defaults to ().
        re_only (tuple, optional): Regex patterns to match function names to keep. Defaults to ().
        exclude (tuple, optional): Names of functions to explicitly remove. Defaults to ().
        re_exclude (tuple, optional): Regex patterns to match function names to remove. Defaults to ().

    Returns:
        dict: The filtered namespace with functions matching the specified criteria.

    Example:
        def test_func1(): pass
        def test_func2(): pass
        def helper_func(): pass

        ns = globals()
        filtered_ns = filter_namespace_funcs(ns, only=('test_func1',))
        print(filtered_ns)  # {'test_func1': <function test_func1 at ...>}
    """
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
        if func in namespace:
            del namespace[func]
    allfuncs = [k for k, v in namespace.items() if callable(v)]
    allfuncs = list(filter(lambda s: s.startswith(prefix), allfuncs))
    for func_re in re_exclude:
        for func in allfuncs:
            if re.match(func_re, func):
                del namespace[func]
    return namespace


def param_is_required(param):
    """Checks if a function parameter is required.

    A parameter is considered required if:
    - It has no default value.
    - It is not a `*args` or `**kwargs` type parameter.

    Args:
        param (inspect.Parameter): The parameter to check.

    Returns:
        bool: True if the parameter is required, False otherwise.

    Example:
    >>> def my_func(x, y=2, *args, **kwargs):
    ...     pass
    >>> params = inspect.signature(my_func).parameters
    >>> for name, param in params.items():
    ...     print(name, param_is_required(param))
    x True
    y False
    args False
    kwargs False

    """
    return param.default is param.empty and param.kind not in (
        param.VAR_POSITIONAL, param.VAR_KEYWORD)


@functools.lru_cache
def func_params(func, required_only=False):
    """Gets the parameters of a function.

    Uses `inspect.signature` to retrieve function parameters.
    Can optionally return only required parameters.

    Args:
        func (callable): The function to inspect.
        required_only (bool, optional): If True, returns only required parameters. Defaults to False.

    Returns:
        dict: A dictionary mapping parameter names to `inspect.Parameter` objects.

    Example:
    >>> def my_func(a, b, c=1):
    ...     pass
    >>> print(dict(func_params(my_func)))
    {'a': <Parameter "a">, 'b': <Parameter "b">, 'c': <Parameter "c=1">}
    >>> print(func_params(my_func, required_only=True))
    {'a': <Parameter "a">, 'b': <Parameter "b">}
    """
    signature = inspect.signature(func)
    params = signature.parameters
    if required_only:
        params = {
            k: param
            for k, param in params.items() if param_is_required(param)
        }
    return params


def list_classes(data):
    seenit = set()

    def visitor(x):
        seenit.add(x.__class__)

    visit(data, visitor)
    return seenit


def change_class(data, clsmap) -> None:

    def visitor(x):
        if x.__class__ in clsmap:
            x.__class__ = clsmap[x.__class__]

    visit(data, visitor)


def visit(data, func) -> None:
    if isinstance(data, dict):
        visit(list(data.keys()), func)
        visit(list(data.values()), func)
    elif isinstance(data, list):
        for x in data:
            visit(x, func)
    else:
        func(data)
