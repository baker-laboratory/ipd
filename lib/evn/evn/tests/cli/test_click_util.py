import click
from evn.cli.click_util import *


def test_extract_command_info():

    def greet(count, name):
        """Greet someone a specified number of times."""
        for _ in range(count):
            click.echo(f'Hello, {name}!')

    arg = click.argument('name')(greet)
    opt = click.option('--count', default=1, help='Number of greetings.')(arg)
    cmd = click.command()(opt)

    info = extract_command_info(cmd)
    assert info.function == greet
    assert info['parameters'] == [
        {
            'count': False,
            'default': 1,
            'envvar': None,
            'flag_value': False,
            'help': 'Number of greetings.',
            'hidden': False,
            'is_flag': False,
            'multiple': False,
            'name': 'count',
            'nargs': 1,
            'opts': ['--count'],
            'param_type_name': 'option',
            'prompt': None,
            'required': False,
            'secondary_opts': [],
            'type': {
                'name': 'integer',
                'param_type': 'Int'
            },
        },
        {
            'default': None,
            'envvar': None,
            'multiple': False,
            'name': 'name',
            'nargs': 1,
            'opts': ['name'],
            'param_type_name': 'argument',
            'required': True,
            'secondary_opts': [],
            'type': {
                'name': 'text',
                'param_type': 'String'
            },
        },
    ]
