import contextlib
import collections
import os
import shutil
import subprocess
from os.path import abspath, exists, isdir, join
from pathlib import Path

import ipd

class ProjecConfigtError(RuntimeError):
    pass

RunOnChangedFilesResult = collections.namedtuple('RunOnChangedFilesResult', 'exitcode, files_modified')

def run_on_changed_files(
    cmd_template: str,
    path: str = '[projname]',
    dryrun: bool = False,
    excludefile: str = '[gitroot].[cmd]_exclude',
    hashfile: str = '[gitroot].[cmd]_hash',
    conffile: str = '[gitroot]/pyproject.toml',
):
    """Run a command on changed files in a git repo.

    Args:
        cmd_template (str): The command to run in fstring syntax, with {conffile} and {changed_files}.
        path (str): The directory to run the command in.
        dryrun (bool): Whether to print the command without running it.
        excludefile (str): The file containing a list of files to exclude.
        hashfile (str): The file containing the previous md5sum of the files.
        conffile (str): The configuration file.
    """
    cmdname = cmd_template.split()[0]
    path, excludefile, hashfile, conffile = substitute_project_vars(path, excludefile, hashfile, conffile, cmd=cmdname)
    with ipd.dev.cd(os.path.dirname(excludefile)):
        prevhash = ipd.dev.run(fr'find {path} -name \*.py -exec md5sum {{}} \;')
        files = set(prevhash.strip().split(os.linesep))
        exclude = ipd.dev.set_from_file(excludefile)
        prev = ipd.dev.set_from_file(hashfile)
        exitcode, files_modified = 0, False
        if changed_files := {x.split()[1] for x in (files - prev)} - exclude:
            cmd = ipd.dev.eval_fstring(cmd_template, vars())
            print(cmd)
            if not dryrun:
                exitcode = int(ipd.dev.run(cmd, capture=False, errok=True))
                if not exitcode:
                    os.system(f'find {path} -name \\*.py -exec md5sum {{}} \\; > {hashfile}')
                    if prevhash != Path(hashfile).read_text():
                        files_modified = True
    return RunOnChangedFilesResult(exitcode, files_modified)

def substitute_project_vars(*args, path: str = '.', **kw) -> list[str]:
    args = list(args)
    pyproj = pyproject(path)
    for i in range(len(args)):
        args[i] = args[i].replace('[gitroot]', f'{git_root(path)}/')
        args[i] = args[i].replace('[projname]', pyproj.project.name)
        for k, v in kw.items():
            args[i] = args[i].replace(f'[{k}]', v)
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
        return ipd.bunchify(tomllib.load(inp))

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
    with contextlib.suppress(Exception):
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
    for pkg in 'yapf ruff pyright validate-pyproject'.split():
        if not shutil.which('yapf'):
            print(f'installing missing package: {pkg}')
            install_package(pkg)

def install_package(pkg):
    try:
        ipd.dev.run(f'pip install {pkg}', echo=True)
    except RuntimeError:
        ipd.dev.run(f'pip install --user {pkg}', echo=True)
