import re
import contextlib
import sys
from functools import wraps
from importlib import import_module
from typing import Callable, Union, Optional

GLOBAL_DISPATCHERS: dict[str, 'LazyDispatcher'] = {}

class NoType:
    pass

def NoPred(pred) -> bool:
    return False

def _qualify(func: Callable, scope: Optional[str]) -> str:
    mod = func.__module__
    name = func.__name__
    qname = func.__qualname__
    if scope == 'local':
        return f'{mod}.{qname}'
    elif scope == 'global':
        return name
    elif scope == 'project':
        parts = mod.split('.')
        root = parts[0] if parts else mod
        return f'{root}.{name}'
    elif scope == 'subpackage':
        parts = mod.split('.')
        subpkg = '.'.join(parts[:-1])
        return f'{subpkg}.{name}'
    else:
        return f'{mod}.{qname}'  # default to local

class LazyDispatcher:

    def __init__(self, func: Callable, scope: Optional[str] = None):
        self._base_func = func
        self._registry: dict[Union[str, type], Callable] = {}
        self._predicate_registry: dict[Callable[[type], bool], Callable] = {}
        self._resolved_types = {}
        self._key = _qualify(func, scope)
        wraps(func)(self)

        # GLOBAL_DISPATCHERS[self._key] = self

    def _register(
        self,
        func: Callable,
        typ: Union[str, type] = NoType,
        predicate: Callable[[type], bool] = NoPred,
    ):
        if typ is object: self._base_func = func
        elif predicate is NoPred: self._registry[typ] = func
        else: self._predicate_registry[predicate] = func
        return self

    def register(self, typ: Union[str, type] = NoType, predicate: Callable[[type], bool] = NoPred):

        def decorator(func):
            result = self._register(func, typ, predicate)
            return result

        return decorator

    def _resolve_lazy_types(self):
        for typ in list(self._registry):
            if isinstance(typ, str) and typ not in self._resolved_types:
                if not is_valid_qualname(typ):
                    raise ValueError(f'Invalid type name: {typ}')
                modname, _, typename = typ.rpartition('.')
                # if not evn.installed[modname]:
                # raise TypeError(f"Module {modname} is not installed.")
                if mod := sys.modules.get(modname):
                    with contextlib.suppress(AttributeError):
                        self._resolved_types[typ] = getattr(mod, typename)
                        self._registry[self._resolved_types[typ]] = self._registry[typ]

    def __call__(self, obj, *args, debug=False, **kwargs):
        self._resolve_lazy_types()

        # TODO: make predicate work with obj, eg. floats > 7

        if (obj_type := type(obj)) in self._registry:
            if debug: print('in registery')
            return self._registry[obj_type](obj, *args, **kwargs)

        for pred, func in self._predicate_registry.items():
            if pred(obj):
                if debug: print('select by predicates', pred, obj)
                self._registry[type(obj)] = func
                return func(obj, *args, **kwargs)

        for key, func in self._registry.items():
            if debug: print('unfound key', key, obj)
            if key := self.check_type(key):
                if isinstance(obj, key):
                    self._registry[type(obj)] = func
                    return func(obj, *args, **kwargs)

        return self._base_func(obj, *args, **kwargs)

    def check_type(self, key):
        if isinstance(key, type): return key
        if isinstance(key, str):
            modname, _, typename = key.rpartition('.')
            try:
                import_module(modname)
                self._resolve_lazy_types()
                return self._resolved_types[key]
            except ImportError:
                return None
        # raise TypeError(f'Key {key} is not a type or str of type')

def lazydispatch(
    arg=None,
    *,
    predicate: Callable[[type], bool] = NoPred,
    scope: Optional[str] = None,
) -> LazyDispatcher:
    if not isinstance(arg, type) and callable(arg) and predicate == NoPred and scope is None:
        # Case: used as @lazydispatch without arguments
        return LazyDispatcher(arg)

    # Case: used as @lazydispatch("type.path", scope=...)
    def wrapper(func):
        key = _qualify(func, scope)
        if key not in GLOBAL_DISPATCHERS:
            GLOBAL_DISPATCHERS[key] = LazyDispatcher(func, scope)
        dispatcher = GLOBAL_DISPATCHERS[key]
        return dispatcher._register(func, arg, predicate=predicate)

    return wrapper

_QUALNAME_RE = re.compile(r'^[a-zA-Z_][\w\.]*\.[A-Z_a-z]\w*$')

def is_valid_qualname(s: str) -> bool:
    """
    Check if a string looks like a valid qualified name for a type.

    A valid qualname is expected to have:
      - one or more dot-separated components
      - all parts must be valid identifiers
      - the final part (the type name) must start with a letter or underscore

    Examples:
        >>> is_valid_qualname('torch.Tensor')
        True
        >>> is_valid_qualname('numpy.ndarray')
        True
        >>> is_valid_qualname('builtins.int')
        True
        >>> is_valid_qualname('not.valid.')
        False
        >>> is_valid_qualname('1bad.name')
        False
        >>> is_valid_qualname('no_dot')
        False
    """
    return bool(_QUALNAME_RE.match(s))
