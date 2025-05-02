"""
auto_click_decorator.py

This module provides `auto_click_decorate_command()`, which inspects a function's signature
and applies Click decorators for each parameter based on type annotations and defaults. The decorators applied will be either click.option or click.argument, depending on whether the parameter is required or optional.

It does **not** apply @click.command — this must be added separately by the caller.

Features:
- Required arguments become `click.argument(...)`
- Optional arguments become `click.option(...)`
- Boolean values become flags (with `is_flag=True`)
- `typing.Annotated` metadata is supported
- Type resolution is handled via `ClickTypeHandlers`

Raises:
- RuntimeError if passed a function already decorated with `@click.command`
- ValueError if the parameter type cannot be resolved

Example (doctestable):

>>> import click
>>> from evn.cli.auto_click_decorator import auto_click_decorate_command
>>> from evn.cli.click_type_handler import ClickTypeHandlers
>>> from click.testing import CliRunner

>>> def greet(name: str = 'world'):
...     print(f'Hello {name}!')

>>> click_params_greet = auto_click_decorate_command(greet, ClickTypeHandlers())
>>> click_command_greet = click.command(click_params_greet)
>>> result = CliRunner().invoke(click_command_greet, ['--name', 'Alice'])
>>> assert 'Hello Alice!' in result.output

See Also:
- click.argument
- click.option
- evn.cli.click_type_handler.ClickTypeHandlers
- test_auto_click_decorator.py
"""

import contextlib
import inspect
import click
import typing
from functools import wraps

from evn.doc.docstring import extract_param_help
from evn.cli.click_type_handler import ClickTypeHandlers, HandlerNotFoundError


def make_hashable(stuff):
    return tuple(
        make_hashable(item) if isinstance(item, list) else item
        for item in stuff)


def _extract_annotation(annotation):
    """
    If the annotation is a typing.Annotated, split it into base type and metadata.
    Otherwise, return the annotation and None.
    """
    if typing.get_origin(annotation) is typing.Annotated:
        if args := typing.get_args(annotation):
            return args[0], make_hashable(args[1:])
    return annotation, None


def _generate_click_decorator(name, param,
                              type_handlers: list[ClickTypeHandlers],
                              help: str):
    """
    Given a parameter (an inspect.Parameter object) and a list of type handlers,
    generate a Click decorator (click.argument for required parameters, click.option for optional ones)
    for that parameter.
    """
    basetype, metadata = _extract_annotation(param.annotation)
    # Determine the Click ParamType via our type handlers.
    param_type = None
    for handlers in type_handlers:
        with contextlib.suppress(HandlerNotFoundError):
            param_type = handlers.typehint_to_click_paramtype(
                basetype, metadata)
            break
    has_default = param.default != inspect.Parameter.empty
    # For booleans, always use option with is_flag=True.
    if basetype is bool and not metadata:
        deco = click.option(f"--{name.replace('_', '-')}",
                            is_flag=True,
                            default=param.default)
    elif not has_default and param_type:
        deco = click.argument(name, type=param_type)
    elif not has_default:
        deco = click.argument(name)
    else:
        option_names = [f"--{name.replace('_', '-')}"]
        kwargs = {'default': param.default, 'show_default': True}
        if param_type:
            kwargs['type'] = param_type
        if basetype is bool:
            kwargs['is_flag'] = True
        deco = click.option(*option_names, help=help, **kwargs)
    # ic(name, type(deco))
    deco
    return deco


def auto_click_decorate_command(fn, type_handlers: list[ClickTypeHandlers]):
    """
    Auto-decorate a function with Click parameter decorators based on its signature.

    This function inspects the signature of fn and automatically generates decorators
    (click.argument for required parameters, click.option for parameters with defaults)
    for every public parameter (ignoring those whose names start with '_') that is not already
    manually decorated.

    Additionally, for parameters whose names start with "_" (internal parameters) that lack a default,
    a wrapper is added to supply None when they are missing from the CLI input.

    Raises a RuntimeError if fn is already a full Click command (i.e. if isinstance(fn, click.Command)),
    ensuring that our framework controls command creation.
    """
    if isinstance(fn, click.Command):
        raise RuntimeError(
            'Function is already a full Click command; manual @click.command decorators are not allowed.'
        )
    sig = inspect.signature(fn)
    # Collect names of parameters that already have manual Click decoration.
    manual_params = set()
    if hasattr(fn, '__click_params__'):
        for p in fn.__click_params__:
            if hasattr(p, 'name') and p.name:
                manual_params.add(p.name)

    decorators = []
    params = list(sig.parameters.items())
    if params and params[0][0] in ('self', 'cls'):
        params = params[1:]  # Skip "self"

    arghelp = extract_param_help(fn.__doc__ or '')
    for name, param in params:
        # Check if the parameter has a docstring help description.
        if name.startswith('_'):
            continue  # We'll handle internals separately.
        if name in manual_params:
            continue
        decorator = _generate_click_decorator(name,
                                              param,
                                              type_handlers,
                                              help=arghelp.get(name))
        decorators.append(decorator)

    # Apply the parameter decorators.
    for decorator in reversed(decorators):
        fn = decorator(fn)

    # Wrap the function to supply None for any internal parameter missing a default.
    orig_sig = inspect.signature(fn)
    internal_params = [
        name for name, param in orig_sig.parameters.items()
        if name.startswith('_') and param.default == inspect.Parameter.empty
    ]
    if internal_params:
        original_fn = fn  # capture the function before wrapping to avoid recursion

        @wraps(original_fn)
        def wrapper(*args, **kwargs):
            for name in internal_params:
                if name not in kwargs:
                    kwargs[name] = None
            return original_fn(*args, **kwargs)

        fn = wrapper

    return fn
