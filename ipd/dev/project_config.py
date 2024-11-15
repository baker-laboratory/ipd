import os
import shutil
import subprocess
from os.path import abspath, exists, isdir, join
from pathlib import Path

import ipd

class ProjecConfigtError(RuntimeError):
    pass

def substitute_project_vars(*args, path: str = '.') -> list[str]:
    args = list(args)
    pyproj = pyproject(path)
    for i in range(len(args)):
        args[i] = args[i].replace('[gitroot]', f'{git_root(path)}/')
        args[i] = args[i].replace('[projname]', pyproj.project.name)
    return args

def git_root(path: str = '.') -> str:
    with ipd.dev.cd(path):
        result = ipd.dev.run('git rev-parse --show-toplevel')
        if isinstance(result, int):
            raise ProjecConfigtError('not a git repository')
        return result

def pyproject_file(path: str = '.') -> str:
    if fname := join(git_root(path), 'pyproject.toml'): return fname
    raise ProjecConfigtError('no pyproject.toml in project')

def pyproject(path: str = '.') -> 'ipd.Bunch':
    import tomllib
    with open(pyproject_file(path), 'rb') as inp:
        return ipd.dev.bunchify(tomllib.load(inp))

def git_status(header=None, footer=None, printit=False):
    srcdir = ipd.proj_dir
    with ipd.dev.cd(srcdir):
        s = ''
        if header: s += f'{f" {header} ":@^80}\n'
        try:
            subprocess.check_output(['git', 'config', '--global', '--add', 'safe.directory', srcdir])
            s += subprocess.check_output(['git', 'log', '-n1']).decode()
            s += subprocess.check_output(['git', 'status']).decode()
        except subprocess.CalledProcessError as e:
            s += 'error fetching git status'
            s += str(e)
        if footer: s += f'\n{f" {footer} ":@^80}'
        if printit: print(s)
        return s

def install_ipd_pre_commit_hook(projdir, path=None):
    if path: projdir = join(projdir, path)
    projdir = abspath(projdir)
    if isdir(join(projdir, '.git')):
        hookdir = join(projdir, '.git/hooks')
    else:
        gitdir = Path(join(projdir, '.git')).read_text()
        assert gitdir.startswith('gitdir: ')
        hookdir = abspath(join(projdir, gitdir[7:].strip(), 'hooks'))
    frm, to = abspath(f'{ipd.projdir}/../git_pre_commit.sh'), f'{hookdir}/pre-commit'
    if exists(to): return
    if os.path.islink(to): os.remove(to)  # remove broken symlink
    os.makedirs(hookdir, exist_ok=True)
    assert os.path.exists(frm)
    print(f'symlinking {frm} {to}')
    os.symlink(frm, to)
    ensure_pre_commit_packages()

def ensure_pre_commit_packages():
    for pkg in 'yapf ruff pyright'.split():
        if not shutil.which('yapf'):
            print(f'installing missing package: {pkg}')
            install_package('yapf')

def install_package(pkg):
    try:
        ipd.dev.run(f'pip install {pkg}', echo=True)
    except RuntimeError:
        ipd.dev.run(f'pip install --user {pkg}', echo=True)
