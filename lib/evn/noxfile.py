import sys
import tomli
import json5 as json
import glob
import nox
import os
from packaging.tags import sys_tags

nox.options.sessions = ['test_matrix']
# nox.options.sessions = ['test_matrix', 'build']
sesh = dict(python=['3.10', '3.11', '3.12', '3.13'], venv_backend='uv')


@nox.session(**sesh)
def test_matrix(session):
    nprocs = min(8, os.cpu_count() or 1)
    session.install('packaging')
    if session.posargs and (session.python) != session.posargs[0]:
        session.skip(f"Skipping {session.python} because it's not in posargs {session.posargs}")
    # session.install(*'.[dev]'.split())
    # session.run('doit test')
    with open('pyproject.toml', 'rb') as f:
        conf = tomli.load(f)
        deps = conf['project']['dependencies']
        deps += conf['project']['optional-dependencies']['dev']
    session.install(*deps)
    print(deps)
    whl = select_wheel(session)
    print(f'Installing {whl}')
    session.install(whl)
    session.run(*'mkdir -p tmp; cd tmp'.split())
    session.run(*'pytest --doctest-modules --ignore evn/tests/_prelude/test_chrono.py --ignore env/tests/tool/test_filter_python_output.py --ignore evn/format --ignore evn/tests/format --pyargs evn'.split())


def get_supported_tags_session(session):
    result = session.run(
        'python',
        '-c',
        (
            'from packaging.tags import sys_tags; import json5 as json;'
            'print(json.dumps([str(tag) for tag in sys_tags()]))'
        ),
        silent=True,
    )
    result = json.loads(result)
    return result


def get_supported_tags(session=None):
    if session:
        return get_supported_tags_session(session)
    return {(tag.interpreter, tag.abi, tag.platform) for tag in sys_tags()}


def parse_wheel_tags(filename):
    parts = filename.split('-')
    if len(parts) < 5:
        return None
    tag = '-'.join([parts[2], parts[3], parts[4].split('.')[0]])
    return tag


def select_wheel(session):
    supported = get_supported_tags(session)
    wheels = glob.glob('wheelhouse/*.whl')
    picks = []
    for wheel in wheels:
        tags = parse_wheel_tags(wheel)
        if tags and tags in supported:
            picks.append(wheel)
    assert picks
    return picks[0]
