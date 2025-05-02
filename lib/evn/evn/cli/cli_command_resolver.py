"""
cli_command_resolver.py

This module provides utilities for inspecting and resolving Click commands within
the EVeN CLI metaclass system. It enables discovery of all registered commands,
resolution by dotted path, and traversal of CLI hierarchies.

Functions:
- walk_commands(root): Recursively yield all (path, command) pairs from a CLI root.
- find_command(root, path): Resolve a command from a dotted path string.
- get_all_cli_paths(): List all command paths from all registered CLI classes.

Example:

>>> from evn.cli import CLI
>>> class Top(CLI):
...     def greet(self):
...         pass
>>> from evn.cli.cli_command_resolver import walk_commands
>>> commands = list(walk_commands(Top))
>>> assert any(p == '<exe> top greet' for p, _ in commands)

See Also:
- test_cli_command_resolver.py
"""

from typing import Iterator
from click import Command, Group
from evn.cli.cli_registry import CliRegistry
from evn.cli.cli_metaclass import CLI


def walk_commands(root: type[CLI],
                  seenit=None) -> Iterator[tuple[str, Command]]:
    if seenit is None:
        seenit = set()
    if root in seenit:
        return
    seenit.add(root)
    group = getattr(root, '__group__', None)
    assert group is not None and isinstance(group, Group)
    base_path = f'{root.get_command_path()}'
    for name, cmd in group.commands.items():
        # Check if this command is itself a group associated with a registered CLI class
        if isinstance(cmd, Group):
            for cls in CliRegistry.all_cli_classes():
                if getattr(cls, '__group__', None) is cmd:
                    yield from walk_commands(cls, seenit)
                    break
        else:
            full_path = f'{base_path} {name}'
            yield full_path, cmd


def find_command(root: type[CLI], path: str) -> Command:
    parts = path.split('.')
    current = root.__group__
    for part in parts:
        if not isinstance(current, Group) or part not in current.commands:
            raise KeyError(f'Command path not found: {path}')
        current = current.commands[part]
    return current


def get_all_cli_paths() -> list[str]:
    paths = []
    seenit = set()
    for cls in CliRegistry.all_cli_classes():
        paths.extend([p for p, _ in walk_commands(cls, seenit)])
    return paths
