import contextlib
import collections
import os
import shutil
import subprocess
from os.path import abspath, exists, isdir, join
from pathlib import Path
import tomli

import ipd

class ProjecConfigtError(RuntimeError):
    pass

RunOnChangedFilesResult = collections.namedtuple('RunOnChangedFilesResult', 'exitcode, files_modified')

def project_local_config(key: str = '', path: str = '.', conffile: str = '[gitroot]/local.toml') -> dict:
    """Get the local configuration for a project.

    Args:
        path (str): The directory to look for the configuration file in.
        conffile (str): The configuration file to look for.

    Returns:
        dict: The local configuration for the project.
    """
    conffile = substitute_project_vars(conffile, path=path)[0]
    if not exists(conffile): return {}
    with open(conffile, 'rb') as inp:
        conf = tomli.load(inp)
    if key: return conf.get(key, {})
    return conf

@contextlib.contextmanager
def OnlyChangedFiles(label: str,
                     path: str = '[projname]',
                     excludefile: str = '[gitroot].[cmd]_exclude',
                     hashfile: str = '[gitroot].[cmd]_hash',
                     conffile: str = '[gitroot]/pyproject.toml',
                     **kw_project_vars):
    """Track and operate on Python files that have changed since the last run.

    This context manager identifies Python files that have changed by comparing their current
    MD5 hashes against previously stored hashes. It's useful for performing operations only
    on files that have been modified, such as running linters or tests selectively.

    On context entry, yields a Bunch object with the following attributes:
        - changed_files: Set of file paths that have changed since the last run
        - prevhash: Set of current MD5 hash strings for all Python files
        - files_changed: Initially empty set, populated on exit with files processed
        - path: Resolved path to search for Python files
        - excludefile: Resolved path to file listing excluded files
        - hashfile: Resolved path to file storing previous MD5 hashes
        - conffile: Resolved path to configuration file

    On context exit, updates the hash file with current hashes.

    Args:
        label (str): Command to execute on changed files, can include {conffile} and
                     {changed_files} placeholders for string formatting
        path (str): Directory to scan for Python files, with support for project variables
        excludefile (str): Path to file containing a list of files to exclude from processing,
                          with support for project variables
        hashfile (str): Path to file that stores MD5 hashes from previous runs,
                       with support for project variables
        conffile (str): Path to project configuration file, with support for project variables
        **kw_project_vars: Additional key-value pairs for variable substitution in paths

    Example:
        ```
        with OnlyChangedFiles('pylint --rcfile={conffile} {changed_files}') as env:
            if env.changed_files:
                run_command_on_files(env.changed_files)
        ```

    Notes:
        - Path strings can include variables like [projname], [gitroot], or [cmd] which
          will be substituted with actual values
        - The context updates the hash file on exit to track the processed files
    """
    changed_files, prevhash, files_changed = set(), set(), set()
    try:
        _ = substitute_project_vars(path, excludefile, hashfile, conffile, **kw_project_vars)
        path, excludefile, hashfile, conffile = _
        with ipd.dev.cd(os.path.dirname(excludefile)):
            hashes = ipd.dev.run(fr'find {path} -name \*.py -exec md5sum {{}} \;')
            prevhash |= {l.strip() for l in hashes.split(os.linesep)}
            exclude = ipd.dev.set_from_file(excludefile)
            prev = ipd.dev.set_from_file(hashfile)
            changed_files |= {x.split()[1] for x in (prevhash - prev)} - exclude
            yield ipd.Bunch(changed_files=changed_files,
                            prevhash=prevhash,
                            files_changed=files_changed,
                            path=path,
                            excludefile=excludefile,
                            hashfile=hashfile,
                            conffile=conffile)
    finally:
        files_changed.clear()
        files_changed |= update_hashes(hashfile, changed_files, prevhash)

def update_hashes(hashfile: str, changed_files: set[str], prevhash) -> set[str]:
    """Update the hash file with the new md5sum of the files.

    Args:
        hashfile (str): The file containing the previous md5sum of the files.
        changed_files (set[str]): The set of changed files.
    """
    if not changed_files: return set()
    orfiles = '|'.join(changed_files)
    cmd = f"""grep -Ev '{orfiles}' {hashfile} > {hashfile}.tmp && mv {hashfile}.tmp {hashfile}
    find {" ".join(changed_files)} -name \\*.py -exec md5sum {{}} \\; >> {hashfile}"""
    exitcode = int(ipd.dev.run(cmd, capture=False, errok=True))
    if exitcode: raise RuntimeError(f'failed to update hash file {hashfile}')
    return changed_files

def run_on_changed_files(
    cmd_template: str,
    path: str = '[projname]',
    dryrun: bool = False,
    excludefile: str = '[gitroot].[cmd]_exclude',
    hashfile: str = '[gitroot].[cmd]_hash',
    conffile: str = '[gitroot]/pyproject.toml',
):
    """Run code only on changed files in a git repo, updates hash file.

    Args:
        cmd_template (str): The command to run in fstring syntax, with {conffile} and {changed_files}.
        path (str): The directory to run the command in.
        dryrun (bool): Whether to print the command without running it.
        excludefile (str): The file containing a list of files to exclude.
        hashfile (str): The file containing the previous md5sum of the files.
        conffile (str): The configuration file.
    """
    cmdname = cmd_template.split()[0]
    exitcode = 0
    with OnlyChangedFiles(cmdname, path, cmd=cmdname) as info:
        cmd = ipd.dev.eval_fstring(cmd_template, info)
        print(cmd)
        if not dryrun:
            exitcode = int(ipd.dev.run(cmd, capture=False, errok=True))
            if not exitcode:
                os.system(f'find {path} -name \\*.py -exec md5sum {{}} \\; > {hashfile}')
    return RunOnChangedFilesResult(exitcode, info.changed_files)

def substitute_project_vars(*args, path: str = '.', **kw) -> list[str]:
    args = list(args)
    pyproj = tomlfile(path)
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

def tomlfile(path: str = '.') -> 'ipd.Bunch':
    with open(pyproject_file(path), 'rb') as inp:
        return ipd.bunchify(tomli.load(inp))

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

def install_ipd_pre_commit_hook(projdir, path=None, config=None):
    with contextlib.suppress(Exception):
        if not config or not config['ipd_pre_commit']: return
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

def project_files():
    root = git_root()
    files = ipd.dev.run(f'git ls-files {root}|grep -v Eigen')
    files = [f.strip() for f in files.split(os.linesep) if f.strip()]
    return files
