import glob
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Annotated
from typer import Argument

import git
import submitit

import ipd

class CITool(ipd.tools.IPDTool):
    def __init__(self, secretfile: str = '~/.secrets'):
        secrets: list[str] = Path(secretfile).expanduser().read_text().splitlines()
        self.secrets = ipd.Bunch({s.split('=')[0].replace('export ', ''): s.split('=')[1] for s in secrets})
        self.repos: dict[str, str] = {
            'cifutils': f'https://{self.secrets.GITLAB_SHEFFLER}@git.ipd.uw.edu/ai/cifutils.git',
            'datahub': f'https://{self.secrets.GITLAB_SHEFFLER}@git.ipd.uw.edu/ai/datahub.git',
            'frame-flow': f'https://{self.secrets.GITHUB_SHEFFLER}@github.com/baker-laboratory/frame-flow.git',
            'fused_mpnn': f'https://{self.secrets.GITHUB_SHEFFLER}@github.com/baker-laboratory/fused_mpnn.git',
            'RF2-allatom': f'https://{self.secrets.GITLAB_SHEFFLER}@git.ipd.uw.edu/jue/RF2-allatom.git',
            'rf_diffusion': f'https://{self.secrets.GITHUB_SHEFFLER}@github.com/baker-laboratory/rf_diffusion.git',
            'ipd': f'https://{self.secrets.GITHUB_SHEFFLER}@github.com/baker-laboratory/ipd.git',
        }

    def update_library(self, libs: Annotated[list[str] | None, Argument()] = None, path: Path = Path('~/bare_repos')):
        # sourcery skip: default-mutable-arg
        path = path.expanduser()
        assert os.path.isdir(path)
        for repo, url in self.repos.items():
            if libs and repo not in libs: continue
            repo_dir = f'{path}/{repo}.git'
            if os.path.isdir(repo_dir):
                print(f'Directory {repo_dir} exists... remove all heads and fetching latest changes...')
                os.system(f'rm -rf {repo_dir}/refs/heads/*')
                ipd.dev.run(f'git --git-dir={repo_dir} fetch origin "*:*" -f', echo=True)
            else:
                print(f'Directory {repo_dir} does not exist. Cloning repository...')
                ipd.dev.run(f'cd {path} && git clone --bare {url}', echo=True)

def init_submodules(repo: git.Repo, repolib: str = '~/bare_repos', recursive: bool = False):
    repolib = os.path.expanduser(repolib)
    with ipd.dev.cd(repo.git.rev_parse('--show-toplevel')):
        for sub in repo.submodules:
            if os.path.exists(sub.path): shutil.rmtree(sub.path)
            subrepo = f'{repolib}/{os.path.basename(sub.url)}'
            print('setup submodule', sub.path, subrepo, sub.hexsha)
            subrepo = git.Repo.clone_from(subrepo, sub.path)
            subrepo.git.checkout(sub.hexsha)
            if recursive:
                init_submodules(subrepo, repolib, True)

class RepoTool(CITool):
    def setup_submodules(self, path: str = '.', repolib: str = '~/bare_repos', recursive: bool = False):
        """Setup submodules in a git repository from a bare repo library."""
        repo = git.Repo(path, search_parent_directories=True)
        repodir = repo.git.rev_parse('--show-toplevel')
        with ipd.dev.cd(path):
            init_submodules(repo, repolib, recursive)

class Future:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

def get_re(pattern, text) -> int:
    result = re.findall(pattern, text)
    assert len(result) < 2
    if not result: return 0
    return int(result[0])

def parse_pytest(fname) -> ipd.Bunch[int]:
    result = ipd.Bunch()
    if not os.path.exists(fname):
        print(f'missing {fname} in {os.getcwd()}')
        return result
    result.fname = fname
    content = Path(fname).read_text()
    print('*' * 80)
    print(content)
    print('*' * 80, flush=True)
    os.system(f'grep "collecting ..." {fname}')
    os.system(f'grep "===" {fname} | grep -v FAILURES')
    os.system(f'grep "FAILED" {fname}')
    print('*' * 80)
    # print(content)
    result.ncollect = get_re(r'collected (\d+) ', content)
    result.deselected = get_re(r'collected .* / (\d+) deselected', content)
    result.skipped = get_re(r'collected .* / (\d+) skipped', content)
    result.selected = get_re(r'collected .* / (\d+) selected', content)
    result.collected = get_re(r'===== .*?(\d+) tests collected .* =====', content)
    result.passed = get_re(r'=.*? (\d+) passed.* =', content)
    result.errors = get_re(r'=.*? (\d+) error.* =', content)
    result.failed = get_re(r'=.*? (\d+) failed.* =', content)
    result.xfailed = get_re(r'=.*? (\d+) xfailed.* =', content)
    result.xpassed = get_re(r'=.*? (\d+) xpassed.* =', content)
    return result

def run_pytest(
    env: str,
    exe: str,
    log: str,
    mark: str = '',
    sel: str = '',
    parallel: int = 1,
    mem: str = '16G',
    timeout: int = 60,
    executor=None,
    dryrun: bool = False,
    tee: bool = False,
    gpu: str = '',
    flags: str = '',
    testdir: str = '.',
    cmdonly: bool = False,
):
    dry = '--collect-only' if dryrun else ''
    stee = '2>&1 | tee' if tee else '>'
    sel = f'-k "{sel}"' if sel else ''
    par = '' if parallel == 1 else f'-n {parallel}'
    if cmdonly:
        cmd = f'cd TESTDIR && {env} PYTHONPATH=. EXE {mark} {sel} {dry} {par} {flags} {stee} {log}'
        return ipd.dev.strip_duplicate_spaces(cmd)
    cmd = f'cd {testdir} && {env} PYTHONPATH=. {exe} {mark} {sel} {dry} {par} {flags} {stee} {log}'
    cmd = ipd.dev.strip_duplicate_spaces(cmd)
    print(f'running: {cmd}')
    if os.path.exists(log): os.remove(log)
    if executor:
        executor.update_parameters(timeout_min=timeout, slurm_mem=mem, cpus_per_task=parallel)
        return cmd, executor.submit(ipd.dev.run, cmd, capture=False), log
    else:
        return cmd, Future(ipd.dev.run(cmd, errok=True, capture=False)), log

class TestsTool(CITool):
    def ruff(self, project):
        ipd.dev.run(f'ruff check {project} 2>&1 | tee ruff_ipd_ci_test_run.log', echo=True, capture=False)

    def pytest(
        self,
        slurm: bool = False,
        gpu: str = '',
        exe: str = sys.executable,
        threads: int = 1,
        log: str = 'pytest_ipd_ci_test_run.log',
        mark: str = '',
        parallel: int = 1,
        timeout: int = 60,
        verbose: bool = True,
        which: str = '',
        dryrun: bool = False,
        tee: bool = False,
        mem: list[str] = ['16G'],
        flags: str = '',
        testdir: str = '.',
        cmdonly: bool = False,
    ):  # sourcery skip: merge-list-appends-into-extend
        """Run pytest with the given parameters.

        Args:
            slurm: bool: whether to run on slurm
            gpu: str: gpu to use
            exe: str: python executable
            threads: int: number of threads
            log: Path: log file
            mark: str: pytest marks
            parallel: int: number of parallel jobs
            timeout: int: slurm timeout in minutes
            verbose: bool: verbose output
            which: str: which tests to run (-k)
            dryrun: bool: dryrun (--collect-only)
            tee: bool: tee outputn to stdout and log file
            mem: list[str]: slurm memory requirements
            flags: str: extra flags to pytest
            testdir: str: test directory
            cmdonly: bool: print command only
        Returns:
            list of tuples (cmd, job, log)
        """
        # os.makedirs(os.path.dirname(log), exist_ok=True)
        if mark: mark = f'-m "{mark}"'
        if not str(exe).endswith('pytest'): exe = f'{exe} -mpytest'
        if verbose: exe += ' -v'
        flags = f'{flags} --benchmark-disable --disable-warnings --cov --junitxml=junit.xml -o junit_family=legacy --durations=10'
        env = f'OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}'
        sel = ' or '.join(which.split()) if which else ''
        jobs = []
        executor = submitit.AutoExecutor(folder='slurm_logs_%j') if slurm else None
        kw: dict[str, Any] = dict(exe=exe,
                                  env=env,
                                  mark=mark,
                                  dryrun=dryrun,
                                  executor=executor,
                                  tee=tee,
                                  gpu=gpu,
                                  flags=flags,
                                  testdir=testdir,
                                  cmdonly=cmdonly)
        if not slurm:
            jobs.append(run_pytest(sel=sel, parallel=parallel, log=log, **kw))  # type: ignore
        else:
            if gpu and executor is not None:
                executor.update_parameters(slurm_partition='gpu', slurm_gres=f'gpu:{gpu}:1')
            if parallel == 1:
                jobs.append(run_pytest(sel=sel, parallel=1, log=log, mem=mem[0], **kw))  # type: ignore
            else:
                nosel = ' and '.join([f'not {t}' for t in which.split()])
                jobs.append(run_pytest(sel=nosel, parallel=parallel, mem=mem[1 % len(mem)], log=f'{log}.par.log', **kw))
                kw['exe'] = None  # run the nonparallel tests on head node... they are quick
                kw['flags'] = kw['flags'].replace('junit.xml', 'junit2.xml')
                jobs.append(run_pytest(sel=sel, parallel=1, mem=mem[0], log=f'{log}.nopar.log', **kw))
        if not cmdonly:
            return [(cmd, job.result(), parse_pytest(log)) for cmd, job, log in jobs]
        print(os.linesep.join(jobs))
        return jobs

    def check(self, path: str = '.'):
        fail = False
        for log in glob.glob('ruff*.log'):
            with open(log) as inp:
                lines = inp.readlines()
                if 'All checks passed!' not in lines[-1]:
                    print('RUFF FAILED')
                    print(str.join('', lines))
                    fail = True
        pytestlogs = glob.glob('pytest*.log')
        for log in pytestlogs:
            print()
            print(f'pytest log: {log}')
            result = parse_pytest(log)
            print(dict(result))
            print()
            fail |= result.errors
            fail |= result.failed
        sys.exit(fail)
