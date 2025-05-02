"""
cli_metaclass.py

This module defines the `CliMeta` metaclass, which automatically turns classes into Click-based CLI command groups.
It forms the backbone of the EVeN CLI system, enabling command registration via inheritance and minimal boilerplate.

The key behaviors of this metaclass include:

- Creating a `click.Group` for each CLI class.
- Registering the group with a parent group (if inherited).
- Registering each public method as a Click command via `auto_click_decorate_command`.
- Injecting a structured logging method using `CliLogger`.
- Composing type handlers from `ClickTypeHandlers`.

Each CLI class using `CliMeta` gains the following attributes:
- `__group__`: a `click.Group` containing its registered commands.
- `__parent__`: a reference to its parent CLI class (if any).
- `__type_handlers__`: merged parameter conversion handlers.
- `_log`: a callable logger that delegates to `CliLogger`.

Usage Example:

    >>> class ProjectCLI(CLI):
    ...     def create(self, name: str):
    ...         print(f'Creating project {name}')
    >>> from click.testing import CliRunner
    >>> cli = ProjectCLI.__group__
    >>> runner = CliRunner()
    >>> result = runner.invoke(cli, ['create', 'demo'])
    >>> assert 'Creating project demo' in result.output

Dependencies:
- click
- evn.cli.auto_click_decorator
- evn.cli.cli_logger
- evn.cli.cli_registry
- evn.cli.click_type_handler

See Also:
- test_cli_metaclass.py — for validated examples and coverage.
"""

import click
import typing
import functools
from evn.cli.auto_click_decorator import auto_click_decorate_command
from evn.cli.cli_registry import CliRegistry
from evn.cli.cli_logger import CliLogger
from evn.cli.click_type_handler import ClickTypeHandler, ClickTypeHandlers

def cls_to_groupname(cls):
    """
    Converts a class name to a Click group name.
    This is typically the same as the class name but can be customized if needed.

    Args:
        cls (type): The class to convert.

    Returns:
        str: The name to be used for the Click group.
    """
    if cls.__name__ == 'CLI':
        return '<exe>'
    return cls.__name__.lower().removeprefix('cli').removesuffix('cli')


class AliasedGroup(click.Group):

    def get_command(self, ctx, cmd_name):
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        matches = [
            x for x in self.list_commands(ctx) if x.startswith(cmd_name)
        ]
        if not matches:
            return None
        elif len(matches) == 1:
            return click.Group.get_command(self, ctx, matches[0])
        ctx.fail(f"Too many matches: {', '.join(sorted(matches))}")

    def resolve_command(self, ctx, args):
        # always return the full command name
        _, cmd, args = super().resolve_command(ctx, args)
        return cmd.name, cmd, args


class CliMeta(type):
    # there are NOT ACTUALLY USED just to make the type checker happy
    __group__: click.Group
    __parent__: 'CLI | None'
    __type_handlers__: 'ClickTypeHandlers | list[ClickTypeHandler]'
    __all_type_handlers__: list[ClickTypeHandlers]
    _config: classmethod
    _log: typing.Callable[['dict|str'], None]

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)

        @classmethod
        def log(cls, *a, **kw):
            CliLogger.log(cls, *a, **kw)

        cls._log, cls.__log__ = log, []
        cls.__all_type_handlers__ = []
        if hasattr(cls, '__type_handlers__'):
            cls.__type_handlers__ = ClickTypeHandlers(cls.__type_handlers__)
        else:
            cls.__type_handlers__ = []
        for base in cls.__mro__[:-1]:
            handlers = ClickTypeHandlers(getattr(base, '__type_handlers__', []))
            if handlers: cls.__all_type_handlers__.append(handlers)

        if callback := cls.__dict__.get('_callback', None):
            callback = cls_to_instance_method(cls)(callback)
            decorated = auto_click_decorate_command(callback,
                                                    cls.__all_type_handlers__)
            # print('<<< instantiating partilly created class, may be sketchy >>>')
            cls.__group__ = click.command(cls=AliasedGroup,
                                          name=cls_to_groupname(cls),
                                          help=callback.__doc__)(decorated)
        else:
            cls.__group__ = AliasedGroup(name=cls_to_groupname(cls),
                                         help=cls.__doc__)
        CliLogger.log(cls,
                      f'Registered group: {cls.__group__.name}',
                      event='group_registered')

        cls.__parent__ = None
        for base in cls.__bases__:
            if base is object:
                break  # skip the base object class
            assert hasattr(base, '__group__')
            assert not cls.__parent__
            cls.__parent__ = base
            base.__group__.add_command(cls.__group__, name=cls.__group__.name)

        for name in cls.__dict__:
            if cls.__name__ == 'CLI':
                break
            if name.startswith('_'):
                continue
            method = getattr(cls, name)
            if not callable(method):
                continue
            decorated = auto_click_decorate_command(method,
                                                    cls.__all_type_handlers__)
            cls.__group__.command(help=method.__doc__)(
                cls_to_instance_method(cls)(decorated))

        CliRegistry.register(cls)

        return cls

    def __call__(cls, *a, **kw):
        if '_instance' in cls.__dict__:
            return cls.__dict__['_instance']
        instance = super().__call__(*a, **kw)
        cls._instance = instance
        CliLogger.log(instance, 'Instance created', event='instance_created')
        return instance


class CLI(metaclass=CliMeta):
    __parent__ = None

    @classmethod
    def get_full_path(cls):
        if parent := cls.__parent__:
            return f'{parent.get_full_path()} {cls.__name__}'
        return cls.__name__

    @classmethod
    def get_command_path(cls):
        path = cls_to_groupname(cls)
        if parent := cls.__parent__:
            path = f'{parent.get_command_path()} {path}'
        return path

    @classmethod
    def _testroot(cls, cmd):
        return click.testing.CliRunner().invoke(cls._root().__group__, cmd)

    @classmethod
    def _test(cls, cmd):
        return click.testing.CliRunner().invoke(cls.__group__, cmd)

    @classmethod
    def _run(cls):
        cls._root().__group__()

    @classmethod
    def _run_this(cls):
        cls.__group__()

    @classmethod
    def _root(cls):
        while cls.__parent__:
            cls = cls.__parent__
        return cls

    @classmethod
    def _walk_click(cls, visitor=lambda *a: None):
        return walk_click_group(cls.__group__, visitor)


def walk_click_group(group: click.Group = CLI.__group__,
                     visitor=lambda *a: None):
    """
    Recursively prints all command names in a Click group as a comma-separated list.

    Args:
        group (click.Group): The root Click group to inspect.
        prefix (str): Internal use for recursive path tracking.
    """
    commands = []

    def walk(g: click.Group, path: str = ''):
        path = path or group.name
        visitor(g, path)
        for name, cmd in g.commands.items():
            full_path = f'{path} {name}' if path else name
            if isinstance(cmd, click.Group):
                walk(cmd, full_path)
            else:
                commands.append(full_path)

    walk(group)
    return commands


def cls_to_instance_method(cls):
    """Assumes cls is a singleton, or stateless, or else user knows what they is doing..."""

    def deco(func):

        @functools.wraps(func)
        def wrap(*a, **kw):
            instance = cls()
            # print(cls, func)
            return func(instance, *a, **kw)

        return wrap

    return deco
