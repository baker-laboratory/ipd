import click
from evn import bunchify


def extract_command_info(cmd: click.Command):
    """
    Extracts the underlying function (callback) and details of all parameters
    (arguments and options) from a click.Command object.

    Returns:
        A dictionary with:
         - 'function': the callback function wrapped by the command.
         - 'parameters': a list of dictionaries for each parameter containing:
             - name: The parameter's name.
             - type: A string representation of the parameter's type.
             - help: Help text (if available, mostly for options).
             - default: The default value.
             - opts: For options, a list of option flags (e.g., ['--count']); absent for arguments.
    """
    # Extract the callback function
    command_func = cmd.callback

    # Prepare list to hold parameter details
    params_info = []
    for param in cmd.params:
        params_info.append(param.to_info_dict())
        continue
        info = {
            'name': param.name,
            'type': str(param.type),
            'help': getattr(param, 'help',
                            None),  # Only options typically have help text
            'default': param.default,
            'required': param.required,
        }
        # If it's an option, also record its flag names
        if isinstance(param, click.Option):
            info['opts'] = param.opts
        elif isinstance(param, click.Argument):
            print(dir(param))
            info['nargs'] = param.nargs
            info['opts'] = param.opts
            info['attrs'] = param.attrs
        else:
            raise TypeError(f'Unsupported parameter type: {type(param)}')
        params_info.append(info)

    return bunchify({'function': command_func, 'parameters': params_info})
