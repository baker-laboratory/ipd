import glob
import os
import re
import shutil
import sys
from pathlib import Path

import git
from rich import print
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
            'rf_diffusion':
            f'https://{self.secrets.GITHUB_SHEFFLER}@github.com/baker-laboratory/rf_diffusion.git',
            'ipd': f'https://{self.secrets.GITHUB_SHEFFLER}@github.com/baker-laboratory/ipd.git',
        }

    def update_library(self, path: Path = Path('~/bare_repos')):
        path = path.expanduser()
        assert os.path.isdir(path)
        for repo, url in self.repos.items():
            repo_dir = f'{path}/{repo}.git'
            if os.path.isdir(repo_dir):
                print(f'Directory {repo_dir} exists. Fetching latest changes...')
                try:
                    ipd.dev.run(f'git --git-dir={repo_dir} fetch origin "*:*" -f', echo=True)
                except RuntimeError:
                    ipd.dev.run(f'git --git-dir={repo_dir} fetch --all -f', echo=True)
            else:
                print(f'Directory {repo_dir} does not exist. Cloning repository...')
                ipd.dev.run(f'cd {path} && git clone --bare {url}', echo=True)

def init_submodules(repo: git.Repo, repolib: str = '~/bare_repos'):
    repolib = os.path.expanduser(repolib)
    with ipd.dev.cd(repo.git.rev_parse('--show-toplevel')):
        for sub in repo.submodules:
            if os.path.exists(sub.path): shutil.rmtree(sub.path)
            subrepo = f'{repolib}/{os.path.basename(sub.url)}'
            print('setup submodule', sub.path, subrepo, sub.hexsha)
            subrepo = git.Repo.clone_from(subrepo, sub.path)
            subrepo.git.checkout(sub.hexsha)
            init_submodules(subrepo, repolib)

class RepoTool(CITool):
    def setup_submodules(self, path: Path = '.', repolib: str = '~/bare_repos'):
        repo = git.Repo(path, search_parent_directories=True)
        repodir = repo.git.rev_parse('--show-toplevel')
        with ipd.dev.cd(path):
            init_submodules(repo, repolib)

class Future:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

def run_pytest(
    env,
    exe,
    log,
    mark='',
    sel='',
    parallel=1,
    mem='16G',
    timeout=60,
    executor=None,
    dryrun=False,
    tee=False,
    gpu='',
    flags='',
):
    dry = '--collect-only' if dryrun else ''
    tee = '2>&1 | tee' if tee else '>'
    sel = f'-k "{sel}"' if sel else ''
    par = '' if parallel == 1 else f'-n {parallel}'
    cmd = f'{env} PYTHONPATH=. {exe} {mark} {sel} {dry} {par} --benchmark-disable {tee} {log}'
    while '  ' in cmd:
        cmd = cmd.replace('  ', ' ')
    if executor:
        executor.update_parameters(timeout_min=timeout, slurm_mem=mem, cpus_per_task=parallel)
        return cmd, executor.submit(ipd.dev.run, cmd), log
    else:
        return cmd, Future(ipd.dev.run(cmd)), log

def get_re(pattern, text):
    result = re.findall(pattern, text)
    assert len(result) < 2
    if not result: return 0
    return int(result[0])

def parse_pytest(fname):
    if not os.path.exists(fname):
        print(f'missing {fname} in {os.getcwd()}')
        return None
    result = ipd.Bunch()
    result.fname = fname
    content = Path(fname).read_text()
    # collecting ... collected 230 items / 2 deselected / 22 skipped / 228 selected
    # =============== 228/230 tests collected (2 deselected) in 0.81s ================
    os.system(f'grep "collecting ..." {fname}')
    os.system(f'grep "===" {fname} | grep -v FAILURES')
    os.system(f'grep "FAILED" {fname}')
    # print(content)
    result.ncollect = get_re(r'collecting ... collected (\d+) ', content)
    result.deselected = get_re(r'collecting ... collected .* / (\d+) deselected', content)
    result.skipped = get_re(r'collecting ... collected .* / (\d+) skipped', content)
    result.selected = get_re(r'collecting ... collected .* / (\d+) selected', content)
    result.collected = get_re(r'===== .*?(\d+) tests collected .* =====', content)
    result.passed = get_re(r'=.*? (\d+) passed.* =', content)
    result.errors = get_re(r'=.*? (\d+) errors.* =', content)
    result.failed = get_re(r'=.*? (\d+) failed.* =', content)
    result.xfailed = get_re(r'=.*? (\d+) xfailed.* =', content)
    result.xpassed = get_re(r'=.*? (\d+) xpassed.* =', content)
    return result

class TestsTool(CITool):
    def run(self, project):
        TestsTool.ruff(project)
        TestsTool.pytest()
        TestsTool.check()

    def ruff(self, project):
        ipd.dev.run(f'ruff check {project} 2>&1 | tee ruff_ipd_ci_test_run.log', echo=True)

    def pytest(self,
               slurm: bool = False,
               gpu: str = '',
               exe: str = sys.executable,
               threads: int = 1,
               log: Path = Path('pytest_ipd_ci_test_run.log'),
               mark: str = '',
               parallel: int = 1,
               timeout: int = 60,
               verbose: bool = True,
               which: str = '',
               dryrun: bool = False,
               tee: bool = False,
               mem: list[str] = ['16G'],
               flags: str = ''):
        # os.makedirs(os.path.dirname(log), exist_ok=True)
        if mark: mark = f'-m "{mark}"'
        if not str(exe).endswith('pytest'): exe = f'{exe} -mpytest'
        if verbose: exe += ' -v'
        par = '' if parallel == 1 else f'-n {parallel}'
        env = f'OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}'
        sel = ' or '.join(which.split()) if which else ''
        nosel = ' and '.join([f'not {t}' for t in which.split()])
        jobs = []
        executor = submitit.AutoExecutor(folder='slurm_logs_%j') if slurm else None
        kw = dict(env=env, mark=mark, dryrun=dryrun, executor=executor, tee=tee, gpu=gpu, flags=flags)
        if not slurm:
            jobs.append(run_pytest(exe=exe, sel=sel, parallel=parallel, log=log, **kw))
        else:
            if gpu:
                executor.update_parameters(slurm_partition='gpu', slurm_gres=f'gpu:{gpu}:1')
            if parallel == 1:
                jobs.append(run_pytest(exe=exe, sel=sel, parallel=1, log=log, mem=mem[0], **kw))
            else:
                jobs.append(
                    run_pytest(exe=exe, sel=sel, parallel=1, mem=mem[0], log=f'{log}.noparallel.log', **kw))
                jobs.append(
                    run_pytest(exe=exe,
                               sel=nosel,
                               parallel=par,
                               mem=mem[1 % len(mem)],
                               log=f'{log}.parallel.log',
                               **kw))
        result = [(cmd, job.result(), parse_pytest(log)) for cmd, job, log in jobs]
        for cmd, job, log in jobs:
            os.system'cat {log}')
        return result

    def check(self, path: Path = '.'):
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
        assert not fail
