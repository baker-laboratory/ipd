import tempfile
import evn
from evn.config import get_config, load_env_layer, load_toml_layer
from evn.testing import TestApp


def test_env_layer_parsing(monkeypatch):
    monkeypatch.setenv('EVN_TESTAPP__FORMAT__TAB_WIDTH', '4')
    monkeypatch.setenv('EVN_TESTAPP__DEV__TEST__FAIL_FAST', 'true')
    env = load_env_layer('EVN_')
    assert env.testapp.format.tab_width == '4'
    assert env.testapp.dev.test.fail_fast == 'true'


def test_toml_layer_loading():
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.toml',
                                     delete=False) as f:
        f.write("""
        [testapp.format]
        tab_width = 2

        [testapp.dev.test]
        fail_fast = true
        """)
        f.flush()
        fpath = evn.Path(f.name)

    loaded = load_toml_layer(fpath)
    assert loaded.testapp.format.tab_width == 2
    assert loaded.testapp.dev.test.fail_fast is True

    fpath.unlink()  # cleanup


def test_get_config_from_app_defaults():
    config = evn.cli.get_config_from_app_defaults(TestApp)
    # print()
    # print_config(config.testapp.dev.format)
    # print(config.testapp.dev.format.format.keys())
    assert isinstance(config, dict)
    assert config.testapp.dev.format.stream.tab_width == 4
    assert not config.testapp.dev.test.file.fail_fast
    assert not config.testapp.dev.doc.build.open_browser
    assert config.testapp.qa.review.coverage.min_coverage == 75
    assert config.testapp.qa.review.changes.summary
    assert config.testapp.run.dispatch.file.path is None


def test_config_includes_all_layers(monkeypatch):
    monkeypatch.setenv('EVN_TESTAPP__PROJ__TAGS__REBUILD', 'true')
    config = get_config(TestApp)
    # print_config(config)
    assert config.testapp.proj.tags.rebuild == 'true'


def test_config_includes_callbacks():
    config = get_config(TestApp)
    assert config.testapp._callback.foo == 'bar'
    assert config.testapp.doccheck._callback.docsdir == 'docs'


def test_config_is_corrent_bunch_type():
    config = get_config(TestApp)
    assert config.__dict__['_config'][
        'split'], 'Config should have split attribute'
