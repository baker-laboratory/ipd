import socket
import tomli
from pathlib import Path
import os
import evn

XDG_CONFIG_HOME = Path(
    os.environ.get('XDG_CONFIG_HOME',
                   Path.home() / '.config'))

CONFIG_PATHS = dict(
    defaults=lambda: {
    },  # built-in fallback (can point to static defaults later)
    user_dev_config=lambda: load_toml_layer(XDG_CONFIG_HOME / 'dev' /
                                            'dev_config.toml'),
    application_defaults=lambda: {},
    pyproject_toml=lambda: load_pyproject_toml(Path('pyproject.toml')),
    user_local_dev_config=lambda: load_toml_layer(
        XDG_CONFIG_HOME / 'dev' / 'local' / f'{socket.gethostname()}.toml'),
    user_project_config=lambda: load_toml_layer(
        Path('local/{getpass.getuser()}/userconfig.toml')),
    environment_vars=lambda: load_env_layer('EVN_'),
)


def get_config_layers(app=None):
    layers = {k: v() for k, v in CONFIG_PATHS.items()}
    layers['application_defaults'] = load_app_layer(app)
    return layers


def get_config(app=None, layers=None):
    """
    Return merged configuration from all layers using ChainMap.
    """
    appname = app.__group__.name if app else ''
    layers = layers or get_config_layers(app)
    config = evn.Bunch({appname: evn.Bunch(_split=' ')}, _split=' ')
    for name, layer in layers.items():
        config[appname]._merge(layer.get(appname, {}), layer=name)
    return config


def load_toml_layer(path: Path) -> dict:
    if not path.exists():
        return evn.Bunch()
    with open(path, 'rb') as f:
        return evn.bunchify(tomli.load(f))


def load_pyproject_toml(path: Path) -> dict:
    if not path.exists():
        return evn.Bunch()
    with open(path, 'rb') as f:
        data = tomli.load(f)
    return evn.bunchify(data.get('tool', {}).get('evn', {}))


def load_env_layer(prefix: str) -> dict:
    result: dict[str, evn.Any] = {}
    for key, val in os.environ.items():
        if key.startswith(prefix):
            parts = key[len(prefix):].lower().split('__')
            ref = result
            for part in parts[:-1]:
                ref = ref.setdefault(part, {})
            ref[parts[-1]] = val
    # print('ENV', result)
    return evn.bunchify(result)


def load_app_layer(app):
    if not app:
        return {}
    return evn.cli.get_config_from_app_defaults(app)
