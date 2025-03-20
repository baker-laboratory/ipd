import os
import nox

nox.options.sessions = ["test_matrix"]

test_matrix_args = [
    'python,extra',
    [
        (python, extra) for python in '3.9 3.10 3.11 3.12 3.13'.split()
        for extra in ['all', ''] if not (python == '3.9' and extra == 'all')
    ]
]

@nox.session(venv_backend='uv')
@nox.parametrize(*test_matrix_args)
def test_matrix(session, python, extra):
    """Run tests with different Python versions & extras."""
    # Allow filtering by passing arguments like: nox -s test_matrix -- 3.11 all
    if session.posargs and (str(python) != session.posargs[0] or len(session.posargs)>1 and extra != session.posargs[1]):
        session.skip(f"Skipping {python}/{extra} because it's not in posargs {session.posargs}")

    session.install('.[dev]')
    session.install(f'.[{extra}]')

    args = ['pytest', f'-n{min(8, os.cpu_count() or 1)}']
    if python >= '3.12' and extra == 'all':
        args.append('--doctest-modules')

    session.run(*args)
