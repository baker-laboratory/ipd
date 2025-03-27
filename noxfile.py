import contextlib
import os
import nox

nox.options.sessions = ["test_matrix"]

@nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"], venv_backend='uv')
@nox.parametrize('extra', ['all', ''])
def test_matrix(session, extra):
    if session.python == '3.9' and extra == 'all':
        session.skip("Skipping 3.9 with all extras")
    # Allow filtering by passing arguments like: nox -s test_matrix -- 3.11 all
    nproc, args = 8, session.posargs
    with contextlib.suppress(ValueError, IndexError):
        nproc, args = int(args[0]), args[1:]
    if args and (session.python != args[0] or len(args) > 1 and extra != args[1]):
        session.skip(f"Skipping {session.python}/{extra}, not in posargs {args}")
    session.install(f'.[dev,{extra}]' if extra else '.[dev]')
    args = ['pytest']
    if nproc > 0: args.append(f'-n{min(nproc, os.cpu_count() or 1)}')
    if session.python >= '3.12' and extra == 'all':
        args.append('--doctest-modules')
    args.extend([
        '--ignore', 'ipd/tests/homog/test_hgeom_library.py', '--ignore',
        'ipd/tests/dev/code/test_format_code.py', '--ignore', 'ipd/cuda', '--ignore', 'ipd/tests/cuda'
    ])
    session.run(*args)
