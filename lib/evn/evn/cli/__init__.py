"""
evn.cli

The `evn.cli` package provides a declarative framework for building composable
command-line interfaces using Python classes and Click.

Key Features:
- Class-based CLI declaration via the `CliMeta` metaclass
- Automatic command registration using method signatures and type annotations
- Support for custom and built-in Click param types via type handlers
- Centralized logging and CLI registry for diagnostics and testing
- Tools for resolving and traversing CLI hierarchies

Modules:
- `cli_metaclass`: Auto-registers CLI classes and commands
- `auto_click_decorator`: Generates Click decorators from method signatures
- `click_type_handler`: Maps type hints to Click param types
- `basic_click_type_handlers`: Default type handlers (e.g. str, int, Path)
- `cli_logger`: Thread-safe structured logger for CLI execution
- `cli_registry`: Global registry of CLI classes
- `cli_command_resolver`: Introspects and resolves CLI command trees

Usage Example:

>>> from evn.cli import CLI
>>> class Hello(CLI):
...     def greet(self, name: str = 'world'):
...         print(f'Hello, {name}')
>>> cli = Hello.__group__
>>> from click.testing import CliRunner
>>> result = CliRunner().invoke(cli, ['greet', '--name', 'Alice'])
>>> result.output
'Hello, Alice\\n'

See Also:
- Click documentation: https://click.palletsprojects.com/
- test_* modules for validation and examples
"""

from evn.cli.auto_click_decorator import *
from evn.cli.basic_click_type_handlers import *
from evn.cli.cli_command_resolver import *
from evn.cli.cli_config import *
from evn.cli.cli_logger import *
from evn.cli.cli_metaclass import *
from evn.cli.cli_registry import *
from evn.cli.click_type_handler import *
from evn.cli.click_util import *
