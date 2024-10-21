import pathlib
import sys
import shutil
import glob
import git
import os
import ipd
from pathlib import Path

class CITool(ipd.tools.IPDTool):
    def __init__(self: 'CITool', secrets: str = '~/.secrets'):
        secrets = Path(secrets).expanduser().read_text().splitlines()
        self.secrets = Box({s.split('=')[0].replace('export ', ''): s.split('=')[1] for s in secrets})
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

    def update_library(self, path: Path = '~/bare_repos'):
        path = os.path.expanduser(path)
        assert os.path.isdir(path)
        for repo, url in self.repos.items():
            repo_dir = f'{path}/{repo}.git'
            if os.path.isdir(repo_dir):
                print(f'Directory {repo_dir} exists. Fetching latest changes...')
                ipd.dev.bash(f'git --git-dir={repo_dir} fetch')
            else:
                print(f'Directory {repo_dir} does not exist. Cloning repository...')
                ipd.dev.bash(f'cd {path} && git clone --bare {url}')

def get_repo(path):
    repo = git.Repo(path, search_parent_directories=True)
    repodir = repo.git.rev_parse("--show-toplevel")
    return repo, repodir

def init_submodules(repo: git.Repo, repolib: str = '~/bare_repos'):
    repolib = os.path.expanduser(repolib)
    for sub in repo.submodules:
        if os.path.exists(sub.path): shutil.rmtree(sub.path)
        subrepo = f'{repolib}/{os.path.basename(sub.url)}'
        print('setup submodule', sub.path, subrepo, sub.hexsha)
        subrepo = git.Repo.clone_from(subrepo, sub.path)
        subrepo.git.checkout(sub.hexsha)
        init_submodules(subrepo, repolib)

class RepoTool(CITool):
    def setup_submodules(self, path: Path = '.', repolib: str = '~/bare_repos'):
        repo, path = get_repo(path)
        with ipd.dev.cd(path):
            init_submodules(repo, repolib)

class TestsTool(CITool):
    def run(self):
        TestsTool.ruff()
        TestsTool.pytest()

    def ruff(self):
        ipd.dev.bash('ruff check 2>&1 | tee ruff_ipd_ci_test_run.log')

    def pytest(
        self,
        slurm: bool = False,
        gpu: bool = False,
        exe: Path = sys.executable,
        threads: int = 1,
        log: Path = 'pytest_ipd_ci_test_run.log',
        mark: str = '',
        parallel: int = 1,
        timeout: int = 60,
        verbose: bool = True
    ):
        # os.makedirs(os.path.dirname(log), exist_ok=True)
        if mark: mark = f'-m "{mark}"'
        if not str(exe).endswith('pytest'): exe = f'{exe} -mpytest'
        if verbose: exe += ' -v'
        par = '' if parallel == 1 else f'-n {parallel}'
        threads = f'OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}'
        if not slurm:
            cmd = f'{threads} PYTHONPATH=. {exe} {mark} {par} 2>&1 | tee {log}.log'
            ipd.dev.bash(cmd, echo=True)
        else:
            #  srun --cpus-per-task=4 --mem=32G ../ci/run_pytest.sh parallel 2>&1 | tee pytest_parallel.log
            #  srun --cpus-per-task=1 --mem=16G ../ci/run_pytest.sh notparallel 2>&1 | tee pytest_single.log
            import submitit
            executor = submitit.AutoExecutor(folder='slurm_logs_%j')
            if gpu: executor.update_parameters(slurm_partition='gpu', gres=f'gpu:{gpu}:1')
            if parallel == 1:
                executor.update_parameters(timeout_min=timeout, slurm_mem='16G', cpus_per_task=1)
                cmd = f'{threads} PYTHONPATH=. {exe} {mark} 2>&1 | tee {log}.log'
                print('SLURM run:', cmd, flush=True)
                job = executor.submit(ipd.dev.bash, cmd)
                job.result()
            else:
                executor.update_parameters(timeout_min=timeout, slurm_mem='32G', cpus_per_task=parallel)
                cmd = f'{threads} PYTHONPATH=. {exe} {mark} -m "not noparallel" {par} 2>&1 | tee {log}.parallel.log'
                print('SLURM run:', cmd, flush=True)
                parallel_job = executor.submit(ipd.dev.bash, cmd)
                executor.update_parameters(timeout_min=timeout, slurm_mem='16G', cpus_per_task=1)
                cmd = f'{threads} PYTHONPATH=. {exe} {mark} -m noparallel 2>&1 | tee {log}.noparallel.log'
                print('SLURM run:', cmd, flush=True)
                nonparallel_job = executor.submit(ipd.dev.bash, cmd)
                parallel_job.result()
                nonparallel_job.result()

    def check(self, path: Path = '.'):
        fail = False
        for f in glob.glob('pytest*.log'):
            with open(f) as inp:
                lines = inp.readlines()
                fail |= 'failed' in lines[-1]
                for line in lines:
                    fail |= 'ERROR' in line
                    fail |= 'FAILED' in line
                    fail |= 'FATAL' in line
                    fail |= 'Error while loading ' in line
            if fail:
                print('PYTEST FAILED:', f)
                print(str.join('', lines))
        for f in glob.glob('ruff*.log'):
            with open(f) as inp:
                lines = inp.readlines()
                if 'All checks passed!' not in lines[-1]:
                    print('RUFF FAILED')
                    print(str.join('', lines))
                    fail = True
        assert not fail
