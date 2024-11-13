import inspect
from typing import TypeVar, Callable

T = TypeVar('T')

def call_with_args_from(argpool, func: Callable[..., T]) -> T:
    params = inspect.signature(func).parameters
    for p in params:
        if p not in argpool:
            raise ValueError(
                f'function: {func.__name__}{inspect.signature(func)} requred arg {p} not argpool: {list(argpool.keys())}')
    args = {p: argpool[p] for p in params}
    return func(**args)
