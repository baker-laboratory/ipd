import click
from evn.cli.cli_registry import CliRegistry
from evn.cli.cli_metaclass import CLI


def setup_module():
    CliRegistry.reset()


class CLIExample(CLI):

    def hi(self, name: str):
        click.echo(f'Hi {name}!')


def test_register_and_reset():
    assert CLIExample in CliRegistry.all_cli_classes()
    CLIExample()  # instantiate singleton
    assert hasattr(CLIExample, '_instance')
    assert hasattr(CLIExample, '__log__')
    CLIExample._log('Test log')
    assert len(CLIExample.__log__) > 0

    CliRegistry.reset()
    assert not hasattr(CLIExample, '_instance')
    assert CLIExample.__log__ == []


def test_print_summary(capsys):
    CliRegistry.print_summary()
    captured = capsys.readouterr()
    assert 'CLIExample' in captured.out
    assert 'group' in captured.out


def test_get_root_commands():
    roots = CliRegistry.get_root_commands()
    group_name = CLIExample.__group__.name
    assert group_name in roots, f"Expected group '{group_name}' in {roots.keys()}"


def test_registry_unique():
    clis = CliRegistry.all_cli_classes()
    assert len(clis) == len(set(clis))
