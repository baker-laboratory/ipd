# test_cli_command_resolver.py
import pytest
from click import Command
from evn.cli.cli_command_resolver import walk_commands, find_command, get_all_cli_paths
from evn.cli.cli_metaclass import CLI

pytestmark = pytest.mark.usefixtures('capfd')


# CLI hierarchy using inheritance
class CLITestTop(CLI):

    def greet(self, name: str):
        return f'Hi {name}'


class CLITestSub(CLITestTop):

    def hello(self, name: str):
        return f'Hello {name}'


def test_walk_commands_finds_all():
    paths = [p for p, _ in walk_commands(CLITestTop)]
    # print("\nwalked command paths:", paths)
    assert '<exe> testtop greet' in paths
    assert '<exe> testtop testsub hello' in paths


def test_find_command_by_path():
    cmd = find_command(CLITestTop, 'testsub')
    assert isinstance(cmd, Command)
    sub_cmd = find_command(CLITestTop, 'testsub.hello')
    assert isinstance(sub_cmd, Command)


def test_get_all_cli_paths_contains_expected():
    paths = get_all_cli_paths()
    # print("all CLI paths:", paths)
    assert any('greet' in p for p in paths)
    assert any('testsub hello' in p for p in paths)


def test_get_all_cli_paths_unique():
    paths = get_all_cli_paths()
    for k in paths:
        assert paths.count(k) == 1, f'Duplicate path found: {k}'


def test_find_command_invalid_path():
    with pytest.raises(KeyError):
        find_command(CLITestTop, 'nonexistent.command')
