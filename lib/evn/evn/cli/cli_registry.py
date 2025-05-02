"""
cli_registry.py

This module provides `CliRegistry`, a global registry of all CLI classes that use
the `CliMeta` system.

The registry tracks CLI class registration, provides discovery tools, and includes
reset and diagnostic methods useful for testing and CLI orchestration.

Features:
- Deduplicated class tracking
- Reset of singleton/log/config state
- Root CLI group discovery
- Registry summary printing

See Also:
- cli_metaclass.py
- cli_command_resolver.py
- test_cli_registry.py
"""

from typing import Type, List, Dict
import click
import evn


class CliRegistry:
    """
    Global registry to track all CLI classes that use CliMeta.
    Used for diagnostics, testing, and reset functionality.
    """

    _cli_classes: List[Type] = []

    @classmethod
    def register(cls, cli_class: Type) -> None:
        if cli_class not in cls._cli_classes:
            cls._cli_classes.append(cli_class)
        else:
            raise ValueError(
                f'CLI class {cli_class.__name__} is already registered.')

    @classmethod
    def all_cli_classes(cls) -> List[Type]:
        return list(cls._cli_classes)

    @classmethod
    def get_root_commands(cls) -> Dict[str, click.Group]:
        roots = {
            c.__group__.name: c.__group__
            for c in cls._cli_classes if getattr(c, '__parent__') == evn.CLI
        }
        return roots

    @classmethod
    def reset(cls) -> None:
        for c in cls._cli_classes:
            if hasattr(c, '_instance'):
                del c._instance
            if hasattr(c, '__log__'):
                c.__log__.clear()

    @classmethod
    def print_summary(cls) -> None:
        print('\n📦 CLI Registry Summary:')
        for c in cls._cli_classes:
            print(f'- {c.__name__}')
            if hasattr(c, '__group__'):
                print(f'  └── group: {c.__group__.name}')
            if hasattr(c, '__parent__') and c.__parent__:
                print(f'  └── parent: {c.__parent__.__name__}')
            if hasattr(c, '__type_handlers__'):
                print(
                    f'  └── handlers: {[h.__class__.__name__ for h in c.__type_handlers__]}'
                )
            print()
