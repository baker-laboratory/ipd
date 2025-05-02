from pathlib import Path
import evn
from evn.testing import TestApp as App


def main():
    # test_set_app_defaults_from_config()
    # return
    evn.quicktest(
        namespace=globals(),
        verbose=1,
        check_xfail=False,
        use_test_classes=True,
        # re_only=['test_generic_get_items'],
        re_exclude=[],
    )


def test_config_from_app():
    config = evn.config.get_config(App)
    # evn.show(config, format='forest', max_width=111)
    confstr = str(evn.unbunchify(config))
    assert '_static_func' not in confstr
    assert '_private_func' not in confstr
    assert '_private_method' not in confstr
    assert 'name-with-under-scores' not in confstr
    assert 'name_with_under_scores' in confstr
    # config.testapp.doccheck.doctest.fail-loopverbose =


def test_convert_config_to_app_types():
    config = evn.config.get_config(App)
    assert config._conf('split') == ' '
    assert config.testapp._conf('split') == ' '
    assert config.testapp._conf('split') == ' '
    assert config.testapp.doccheck._conf('split') == ' '
    # evn.config.print_config(config)
    config = evn.cli.convert_config_to_app_types(App, config)


def test_set_app_defaults_from_config():
    config = evn.config.get_config(App)
    config.testapp.version.verbose = True
    evn.cli.set_app_defaults_from_config(App, config)
    config2 = evn.cli.get_config_from_app_defaults(App)
    assert config == config2


def test_set_app_callback_defaults_from_config():
    config = evn.config.get_config(App)
    evn.cli.set_app_defaults_from_config(App, config)
    config2 = evn.cli.get_config_from_app_defaults(App)
    if config != config2:
        print('fail test_set_app_defaults_from_config:')
        evn.diff(config, config2, out=print)
        assert config == config2
    config.testapp._callback.foo = 'bar'


def test_big_change():

    def mutate(cfg, prm, group, path, param):
        new = param.default
        if isinstance(new, str):
            new = f'"{new}foo"'
        elif isinstance(new, bool):
            new = not new
        elif isinstance(new, int):
            new = new + 1
        elif isinstance(new, float):
            new = new * 2
        elif isinstance(new, Path):
            new = new.parent
        # print(f'config.{'.'.join(path.split())}.{name}{param.name} = {new}')
        cfg[param.name] = new

    config = evn.config.get_config(App)
    mutated = evn.cli.mutate_config(App, config, action_func=mutate)
    # evn.diff(config, mutated, out=print)
    assert config != mutated
    print('\nconfig', config.testapp.buildtools.clean.all.verbose)
    print('mutate', mutated.testapp.buildtools.clean.all.verbose)
    evn.cli.set_app_defaults_from_config(App, mutated)
    print('config', config.testapp.buildtools.clean.all.verbose)
    print('mutate', mutated.testapp.buildtools.clean.all.verbose)
    assert config != mutated
    config2 = evn.cli.get_config_from_app_defaults(App)
    evn.cli.set_app_defaults_from_config(App, config)  # restore
    assert config != mutated
    assert config2 == mutated
    assert config != config2
    # evn.diff(mutated, config, out=print)
    # evn.diff(mutated, config2, out=print)
    if mutated != config2:
        print('fail test_big_change:')
        evn.diff(mutated, config2, out=print)
        assert mutated == config2


if __name__ == '__main__':
    main()
