"""
click_type_handler.py

Defines the `ClickTypeHandler` interface and the `ClickTypeHandlers` registry, which together
allow parameter types in the CLI framework to be inferred from Python type hints and optional
metadata.

Each handler subclass (e.g. BasicIntHandler, BasicBoolHandler) provides conversion logic
for a specific base type, and supports preprocessing, postprocessing, and failure reporting
via Click's `BadParameter`.

Key Classes:
- ClickTypeHandler: abstract base class for param type handlers
- ClickTypeHandlers: a prioritized set of handlers for resolving annotated Python types
- MetadataPolicy: enum to control whether metadata is required, forbidden, or optional

Features:
- Type resolution via `typehint_to_click_paramtype(basetype, metadata)`
- Automatic fallback if multiple handlers match
- Metadata-aware filtering and priority-based resolution
- Memoization via `get_cached_paramtype()` (for performance)

Example (doctestable):

>>> from evn.cli.click_type_handler import ClickTypeHandlers, ClickTypeHandler
>>> class DummyHandler(ClickTypeHandler):
...     __supported_types__ = {int: 'optional_metadata'}
...
...     def convert(self, value, param, ctx):
...         return int(value)

>>> handlers = ClickTypeHandlers()
>>> handlers.add(DummyHandler())
>>> param_type = handlers.typehint_to_click_paramtype(int, metadata=None)
>>> assert param_type.convert('42', None, None) == 42

See Also:
- evn.cli.basic_click_type_handlers
- evn.cli.auto_click_decorator
- test_click_type_handler.py
"""

import inspect
import uuid
import contextlib
import enum
from functools import lru_cache
import click

class MetadataPolicy(str, enum.Enum):
    FORBID = 'no_metadata'
    OPTIONAL = 'optional_metadata'
    REQUIRED = 'require_metadata'

class HandlerNotFoundError(RuntimeError):
    pass

class ClickTypeHandlers(set):
    """
    A container for Click type handlers.

    This class is used to manage a list of type handlers.
    It can be used to retrieve the appropriate handler for a given type.
    """

    # @classmethod
    # def __new__(cls, val=(), *a, **kw):
    #     if isinstance(cls, ClickTypeHandlers):
    #         return val
    #     return super().__new__(val)

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        for arg in args:
            if isinstance(arg, ClickTypeHandlers):
                self.update(arg)
            elif issubclass(arg, ClickTypeHandler):
                self.add(arg)
            else:
                raise TypeError(
                    f"Expected ClickTypeHandler cls or ClickTypeHandlers instance, got {type(arg)}\n{arg}")

    def ordered_handlers(self, basetype, metadata):
        [h for h in self if h.metadata_policy(basetype)]
        return list(sorted(self, key=lambda x: x.priority(), reverse=True))

    def typehint_to_click_paramtype(self, basetype, metadata) -> click.ParamType:
        """Given a basetype and optional metadata, return the Click ParamType to use."""
        handlers = self.ordered_handlers(basetype, metadata)
        if metadata:
            for handler_class in handlers:
                # ic(handler_class.metadata_policy)
                if handler_class.metadata_policy(basetype) == MetadataPolicy.REQUIRED:
                    with contextlib.suppress(HandlerNotFoundError):
                        return get_cached_paramtype(handler_class, basetype, metadata)
        for handler_class in handlers:
            if handler_class.metadata_policy(basetype) == MetadataPolicy.OPTIONAL:
                with contextlib.suppress(HandlerNotFoundError):
                    return get_cached_paramtype(handler_class, basetype)
        if not metadata:
            for handler_class in handlers:
                if handler_class.metadata_policy(basetype) == MetadataPolicy.FORBID:
                    with contextlib.suppress(HandlerNotFoundError):
                        return get_cached_paramtype(handler_class, basetype)
        if not metadata and basetype in (int, float, str, bool, uuid.UUID):
            return basetype
        if not metadata and basetype == inspect._empty:
            # Special case for empty annotations (e.g., no type hint).
            return click.ParamType()
        raise HandlerNotFoundError(
            f'No suitable Click ParamType found for basetype {basetype} with metadata {metadata} using handlers: {handlers}'
        )

    def __repr__(self):
        return f'ClickTypeHandlers({[h.__name__ for h in self]})'

class ClickTypeHandler(click.ParamType):
    """
    Base class for handling conversion of type hints to Click ParamTypes.

    Subclasses should define:
      - __supported_types__: a dict mapping types (e.g., int, list[int]) to booleans.
        The boolean is True if the handler requires metadata for that type.
      - __priority_bonus__: an integer bonus (default 0) that subclasses can override.
      - METADATA_BONUS: a fixed bonus (default 10) applied if metadata is used.

    This class provides default no-op implementations for preprocess_value and postprocess_value.
    It also defines a method 'typehint_to_click_paramtype' (the conversion function)
    and a priority computation function.
    """

    # Dictionary of types this handler applies to.
    # Example: {int: False, float: False, list: True}
    __supported_types__: dict[type, MetadataPolicy] = {}
    __priority_bonus__ = 0
    METADATA_BONUS = 10

    def __init__(self):
        super().__init__()

    @classmethod
    def metadata_policy(cls, basetype):
        return cls.__supported_types__.get(basetype)

    @classmethod
    def typehint_to_click_paramtype(cls, basetype, metadata):
        """
        Given a type hint (basetype) and optional metadata, return the Click ParamType to use.
        Default behavior is to return cls if this handler handles the type; otherwise, raises HandlerNotFoundError.
        """
        if not cls.handles_type(basetype, metadata):
            raise HandlerNotFoundError(
                f'{cls.__class__.__name__} does not handle type {basetype} with metadata {metadata}')
        return cls()

    @classmethod
    def handles_type(cls, basetype, metadata=None):
        """
        Check whether this handler applies to the given basetype.
        Iterates over __supported_types__; if a key matches basetype, then if the boolean flag is True,
        metadata must be provided (and non-empty) for a positive result.
        """
        for typ, metapol in cls.__supported_types__.items():
            if issubclass(basetype, typ):
                if metapol == MetadataPolicy.REQUIRED:
                    return bool(metadata)
                if metapol == MetadataPolicy.FORBID:
                    return not metadata
                return True
        return False

    def preprocess_value(self, raw: str):
        """
        Preprocess the raw input value before Click's conversion.
        Default implementation is a aano-op.
        """
        return raw

    def postprocess_value(self, value):
        """
        Postprocess the value after Click's conversion.
        Default implementation is a no-op.
        """
        return value

    def convert(self, value, param, ctx):
        """
        Override click.ParamType.convert to incorporate pre- and post-processing.
        By default, this implementation simply returns the preprocessed value.
        Subclasses should override this method if further conversion is needed.
        """
        try:
            preprocessed = self.preprocess_value(value)
            # Default conversion: return the preprocessed value unchanged.
            converted = preprocessed
            postprocessed = self.postprocess_value(converted)
            return postprocessed
        except Exception as e:
            self.fail(f'Conversion failed for value {value}: {e}', param, ctx)

    @classmethod
    def priority(cls):
        return cls.__priority_bonus__

    # @classmethod
    # def type_specificity(cls, basetype):
    # """
    # Compute a basic measure of specificity for the basetype.
    # # For generic types (with __args__), count the number of arguments that are not typing.Any.
    # """
    # if hasattr(basetype, '__args__') and basetype.__args__:
    # specificity = sum(1 for arg in basetype.__args__ if arg is not typing.Any)
    # return specificity
    # return 0

    # def compute_priority(self, basetype, metadata, mro_rank: int):
    #     """
    #    Compute the overall priority for this handler.
    #    Lower numeric values are better.
    #    Priority is computed as:
    #        mro_rank + __priority_bonus__ + (METADATA_BONUS if metadata is required and provided#) + type_specificity
    #    """
    #     specificity = self.type_specificity(basetype)
    #     bonus = self.__priority_bonus__
    #     if self.handles_type(basetype, metadata):
    #         for typ, metadata_policy in self.__supported_types__.items():
    #             if basetype == typ and metadata_policy:
    #                 bonus += self.METADATA_BONUS
    #                 break
    #     return mro_rank + bonus + specificity

# Caching function: cache the computed Click ParamType based on handler class, basetype, and metadata.
@lru_cache(maxsize=None)
def get_cached_paramtype(handler_class, basetype, metadata=None):
    """
    Retrieve (or compute and cache) the Click ParamType for the given handler class, basetype, and metadata.
    """
    return handler_class.typehint_to_click_paramtype(basetype, metadata)
