# evn/tests/cli/test_cli_metaclass.py
import pytest
import click
from click.testing import CliRunner
from evn.cli.cli_metaclass import CLI
from evn.cli.auto_click_decorator import auto_click_decorate_command
from evn.cli.cli_logger import CliLogger
from evn.cli.click_type_handler import ClickTypeHandlers

runner = CliRunner()


# Define a dummy parent CLI tool.
class CLIParent(CLI):
    __type_handlers__ = ClickTypeHandlers(
    )  # For testing, no extra handlers needed.

    def _callback(self, debug: bool = False):
        if debug:
            click.echo('parent debug is on')
        self._parent_debug = debug

    def greet(self, name: str):
        "Command that greets a person."
        click.echo(f'Hello, {name}!')


# Define a dummy child CLI tool.
class CLIChild(CLIParent):
    __type_handlers__ = ClickTypeHandlers()  # Inherit parent's handlers.

    def _callback(self, debug: bool = False):
        if debug:
            click.echo('child debug is on')
        self._child_debug = debug

    def farewell(self, name: str):
        "Command that says goodbye."
        click.echo(f'Goodbye, {name}!')


def test_root_is_CLI():
    assert CLIChild.get_command_path() == '<exe> parent child'
    assert CLIChild._root().__name__ == 'CLI'


def test_singleton_behavior():
    parent1 = CLIParent()
    parent2 = CLIParent()
    assert parent1 is parent2
    child1 = CLIChild()
    child2 = CLIChild()
    assert child1 is child2


def test_parent_command_registration():
    runner = CliRunner()
    # CLIParent's group should include the 'greet' command.
    result = runner.invoke(CLIParent.__group__, ['greet', 'Alice'])
    assert result.exit_code == 0
    assert 'Hello, Alice!' in result.output


def test_child_command_registration():
    runner = CliRunner()
    # CLIChild's group is attached as a subcommand of CLIParent's group.
    # Invoke as: parent group 'child' then command 'farewell'.
    result = runner.invoke(CLIParent.__group__, ['child', 'farewell', 'Bob'])
    assert result.exit_code == 0
    assert 'Goodbye, Bob!' in result.output


def test_child_callback():
    runner = CliRunner()
    result = runner.invoke(CLIParent.__group__,
                           ['child', '--debug', 'farewell', 'Bob'])
    # ic(result.output)
    assert result.exit_code == 0
    assert 'Goodbye, Bob!' in result.output
    assert 'child debug is on' in result.output
    assert 'parent debug is on' not in result.output


def test_parent_callback():
    runner = CliRunner()
    result = runner.invoke(CLIParent.__group__,
                           ['--debug', 'child', 'farewell', 'Bob'])
    assert result.exit_code == 0
    assert 'Goodbye, Bob!' in result.output
    assert 'child debug is on' not in result.output
    assert 'parent debug is on' in result.output


def test_parent_and_child_callback():
    runner = CliRunner()
    result = runner.invoke(CLIParent.__group__,
                           ['--debug', 'child', '--debug', 'farewell', 'Bob'])
    assert result.exit_code == 0
    assert 'Goodbye, Bob!' in result.output
    assert 'child debug is on' in result.output
    assert 'parent debug is on' in result.output


def test_instance_logging():
    parent_instance = CLIParent()
    logs = CliLogger.get_log(CLIParent)
    found = any('Instance created' in log.get('message', '') for log in logs)
    assert found


def test_get_full_path():
    parent_instance = CLIParent()
    child_instance = CLIChild()
    assert parent_instance.get_full_path() == 'CLI CLIParent'
    assert child_instance.get_full_path() == 'CLI CLIParent CLIChild'


def test_greet_command():
    runner = CliRunner()
    result = runner.invoke(CLIParent.__group__, ['greet', 'Alice'])


def test_greet_command_exists():
    assert 'greet' in CLIParent.__group__.commands


class CLIDebugTest(CLI):

    def greet(self, name: str, default=7):
        "Basic test for argument passing."
        click.echo(f'Hello, {name}!')


def test_debug_sanity_command_runs():
    runner = CliRunner()
    result = runner.invoke(CLIDebugTest.__group__, ['greet', 'TestUser'])
    assert result.exit_code == 0
    assert 'Hello, TestUser!' in result.output


def test_click_metadata_capture():
    decorated = auto_click_decorate_command(CLIDebugTest.greet,
                                            ClickTypeHandlers)
    assert [['--default'], ['name']
            ] == [p.opts for p in getattr(decorated, '__click_params__', [])]
    assert {} == getattr(decorated, '__click_attrs__', {})


def test_empty_cli_group_creates_successfully():

    class EmptyCLI(CLI):
        pass

    assert isinstance(EmptyCLI.__group__, click.Group)
    assert len(EmptyCLI.__group__.commands) == 0


def test_command_override():

    class BaseCLI(CLI):

        def greet(self):
            click.echo('base')

    class SubCLI(BaseCLI):

        def greet(self, name: str = 'world'):
            click.echo('sub')

    result = SubCLI._test(['greet'])
    assert 'base' not in result.output
    assert 'sub' in result.output
    result = SubCLI._testroot(['base greet'])
    assert 'base' in result.output
    assert 'sub' not in result.output
    result = SubCLI._testroot(['sub', 'greet'])
    assert 'base' not in result.output
    assert 'sub' in result.output


def test_command_test_noarg():

    class TestCLI(CLI):

        def greet(self):
            click.echo('Hello')

    result = TestCLI._test(['greet'])
    assert result.exit_code == 0
    assert 'Hello' in result.output


def test_default_option():

    class CliGreet(CLI):

        def greet(self, name: str = 'world'):
            click.echo(f'Hello, {name}')

    result = CliGreet._test(['greet'])
    assert 'Hello, world' in result.output
    result = CliGreet._test(['greet', '--name', 'Alice'])
    assert 'Hello, Alice' in result.output
    assert result.exit_code == 0


if __name__ == '__main__':
    pytest.main([__file__])
9
