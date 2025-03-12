import nox

@nox.session(venv_backend='uv')
@nox.parametrize(
    'python,extra',
    [
        (python, extra)
        for python in '3.9 3.10 3.11 3.12 3.13'.split()
        # for python in ['3.9']
        # for extra in [''] + 'ml biotite crud all'.split()
        for extra in ['all', '']
        if not (python=='3.9' and extra=='all')
    ],
)
def test_matrix(session, extra):
    'Run pytest tests with arguments.'
    session.install(f'.[dev]')
    session.install(f'.[{extra}]')
    session.run('pytest', '-n8')
