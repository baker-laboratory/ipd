import enum
import evn
from evn.decon.bunch import Bunch, bunchify


class Action(enum.Enum):
    CONVERT = 443_520
    SET_DEFAULT = 10_200_960
    GET_DEFAULT = 244_823_040
    CUSTOM = 95_040


def get_config_from_app_defaults(app):
    """
    Generate a default config structure from a CLI class, including subcommands.
    """
    config = Bunch(_split=' ')
    return config_app_sync(app, config, Action.GET_DEFAULT)


def convert_config_to_app_types(app, config) -> Bunch:
    return config_app_sync(app, config, Action.CONVERT)


def set_app_defaults_from_config(app, config) -> Bunch:
    return config_app_sync(app, config, Action.SET_DEFAULT)


def mutate_config(app, config, action_func) -> Bunch:
    config2 = config.copy()
    return config_app_sync(app,
                           config2,
                           Action.CUSTOM,
                           action_func=action_func)


def config_app_sync(app, config, action: Action, action_func=None) -> Bunch:
    assert config._conf('split') == ' ', "Config split is not ' '"

    def visit(group, path, action_func=action_func):
        group_items = list(group.commands.items())
        if group.callback:
            group_items.append(('_callback', group))
        for name, method in group_items:
            name = name.replace('-', '_')
            conf = config._get_split(f'{path} {name}',
                                     create=Action.GET_DEFAULT == action)
            for param in method.params:
                assert action == Action.GET_DEFAULT or param.name in conf
                if action is Action.CONVERT:
                    conf[param.name] = param.type.convert(
                        conf[param.name], param, None)
                elif action is Action.SET_DEFAULT:
                    if param.default != conf[param.name]:
                        param.default = conf[param.name]
                    param.default = conf[param.name]
                elif action is Action.GET_DEFAULT:
                    conf[param.name] = param.default
                elif action is Action.CUSTOM and action_func:
                    action_func(conf, name, group, path, param)
                else:
                    raise ValueError(
                        f'Unknown action {action} with func {action_func}')

    evn.cli.walk_click_group(app.__group__, visitor=visit)
    return bunchify(config, _like=config)


# def get_config_from_app_defaults(app, config):
# config_app_sync(app, config, Action.SET_DEFAULT)
