import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional, Set, TypeVar, get_type_hints, Union, ParamSpec

# Type variables for generic typing
T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')

# Type alias for keyword arguments dictionary
KW = Dict[str, Any]


def kwcheck(kw: KW,
            func: Optional[Callable] = None,
            checktypos: bool = True) -> KW:
    """Filter keyword arguments to match those accepted by a function.

    Args:
        kw: Dictionary of keyword arguments to filter
        func: The function whose signature defines accepted parameters
        checktypos: If True, raises TypeError for arguments that look like typos

    Returns:
        Filtered dictionary containing only accepted keyword arguments
    """
    if func is None:
        return kw

    # Get the signature of the function
    sig = inspect.signature(func)
    params = sig.parameters

    # Check if the function accepts arbitrary keyword arguments
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                         for p in params.values())

    if accepts_kwargs:
        # If the function accepts **kwargs, keep all arguments
        return kw

    # Get the names of all keyword parameters
    valid_params = {
        name
        for name, param in params.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.KEYWORD_ONLY)
    }

    # Filter the keyword arguments
    filtered_kw = {k: v for k, v in kw.items() if k in valid_params}

    # Check for possible typos if enabled
    if checktypos:
        invalid_keys = set(kw.keys()) - valid_params
        if invalid_keys:
            # Check for possible typos (keys that look similar to valid params)
            for invalid_key in invalid_keys:
                close_matches = [
                    valid for valid in valid_params
                    if _similar_strings(invalid_key, valid)
                ]
                if close_matches:
                    suggestions = ', '.join(f"'{match}'"
                                            for match in close_matches)
                    raise TypeError(
                        f"'{invalid_key}' is not a valid parameter for {func.__name__}. "
                        f'Did you mean {suggestions}?')

    return filtered_kw


def _similar_strings(a: str, b: str, threshold: float = 0.8) -> bool:
    """Check if two strings are similar using a simple metric.

    This is a basic implementation that could be replaced with more
    sophisticated algorithms like Levenshtein distance.
    """
    # Simple implementation - just check if one string is contained in the other
    # or if they share many characters
    if a in b or b in a:
        return True

    # Count common characters
    common = set(a) & set(b)
    # Calculate similarity as ratio of common chars to average string length
    similarity = len(common) / ((len(a) + len(b)) / 2)

    return similarity >= threshold


def kwcall(kw: KW, func: Callable[P, R], *args: Any, **kwargs: Any) -> R:
    """Call a function with filtered keyword arguments.

    Args:
        kw: Primary dictionary of keyword arguments
        func: The function to call
        *args: Positional arguments to pass to the function
        **kwargs: Additional keyword arguments that will be merged with kw

    Returns:
        The return value from calling func
    """
    # Merge kwargs with kw, with kwargs taking precedence
    merged_kwargs = {**kw, **kwargs}
    # Filter to only accepted keyword arguments
    filtered_kwargs = kwcheck(merged_kwargs, func)
    # Call the function
    return func(*args, **filtered_kwargs)


def kwargs_tunnel(
        *,
        allowed_keys: Optional[Set[str]] = None,
        type_check: bool = True,
        debug: bool = False) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that enables kwargs tunneling for a function.

    This decorator makes a function automatically forward unused kwargs to any
    callable parameters that accept **kwargs.

    Args:
        allowed_keys: If provided, only these keys will be tunneled through
        type_check: If True, validate type hints when tunneling kwargs
        debug: If True, log information about tunneled kwargs

    Returns:
        A decorator function
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Get function signature
        sig = inspect.signature(func)
        params = sig.parameters

        # Get parameter types if type checking is enabled
        param_types = get_type_hints(func) if type_check else {}

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            # Make a copy of kwargs to avoid modifying the original
            remaining_kwargs = kwargs.copy()
            used_kwargs: Dict[str, Any] = {}

            # Filter kwargs for the wrapped function itself
            for name, param in params.items():
                if (param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY)
                        and name in remaining_kwargs):
                    # Check type if enabled
                    if type_check and name in param_types:
                        value = remaining_kwargs[name]
                        expected_type = param_types[name]
                        if not _is_type_compatible(value, expected_type):
                            raise TypeError(
                                f"Argument '{name}' has type {type(value).__name__}, "
                                f'but {func.__name__} expects {expected_type}')

                    # Move the kwarg to used_kwargs
                    used_kwargs[name] = remaining_kwargs.pop(name)
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    # The function itself accepts **kwargs
                    # Add all remaining kwargs if allowed
                    if allowed_keys is not None:
                        # Only pass through allowed keys
                        for key in list(remaining_kwargs.keys()):
                            if key in allowed_keys:
                                used_kwargs[key] = remaining_kwargs.pop(key)
                    else:
                        # Pass all remaining kwargs
                        used_kwargs.update(remaining_kwargs)
                        remaining_kwargs.clear()

            # Call the function with filtered kwargs
            result = func(*args, **used_kwargs)

            # Handle tunneling to callable results
            if remaining_kwargs and callable(result):
                # Check if the result accepts **kwargs
                try:
                    result_sig = inspect.signature(result)
                    result_accepts_kwargs = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in result_sig.parameters.values())

                    if result_accepts_kwargs:
                        # If we have kwargs to tunnel and result can accept them
                        if debug:
                            print(
                                f'Tunneling {len(remaining_kwargs)} kwargs to {result.__name__}'
                            )

                        # Filter and tunnel the kwargs to the callable result
                        tunneled_kwargs = kwcheck(remaining_kwargs, result)
                        return result(**tunneled_kwargs)
                except (ValueError, TypeError):
                    # If we can't inspect the result signature, don't tunnel
                    pass

            return result

        # Add an attribute to indicate this function supports tunneling
        wrapper._supports_kwargs_tunnel = True

        return wrapper

    return decorator


def _is_type_compatible(value: Any, expected_type: Any) -> bool:
    """Check if a value is compatible with an expected type."""
    # Handle Union types
    if hasattr(expected_type,
               '__origin__') and expected_type.__origin__ is Union:
        return any(
            _is_type_compatible(value, arg) for arg in expected_type.__args__)

    # Handle Optional types (Union[Type, None])
    if (hasattr(expected_type, '__origin__')
            and expected_type.__origin__ is Union
            and type(None) in expected_type.__args__):
        if value is None:
            return True
        other_types = [
            t for t in expected_type.__args__ if t is not type(None)
        ]
        return any(_is_type_compatible(value, t) for t in other_types)

    # Basic isinstance check
    try:
        return isinstance(value, expected_type)
    except TypeError:
        # Some types like List[int] will cause TypeError with isinstance
        # In these cases, we just return True to avoid false positives
        return True


# Example usage
def example():

    @kwargs_tunnel(type_check=True)
    def create_user(name: str, age: int, **extra):
        print(f'Creating user: {name}, {age}')
        if extra:
            print(f'Extra info: {extra}')

        # Return a function that will receive tunneled kwargs
        return lambda **kwargs: print(f'User details: {kwargs}')

    # Call with tunneled kwargs
    create_user(name='Alice', age=30, email='alice@example.com', role='admin')
