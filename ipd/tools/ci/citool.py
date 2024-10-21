import sys
import shutil
import glob
import git
import os
import ipd
from pydantic import DirectoryPath
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
    def setup_submodules(self, path: DirectoryPath = '.', repolib: str = '~/bare_repos'):
        repo, path = get_repo(path)
        with ipd.dev.cd(path):
            init_submodules(repo, repolib)

    def update_library(self, path: DirectoryPath = '~/bare_repos'):
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

class TestsTool(CITool):
    def run(self):
        TestsTool.ruff()
        TestsTool.pytest()

    def ruff(self):
        ipd.dev.bash('ruff check 2>&1 | tee ruff_ipd_ci_test_run.log')

    def pytest(self):
        ipd.dev.bash(f'{sys.executable} -m pytest 2>&1 | tee pytest_ipd_ci_test_run.log')

    def check(self, path: DirectoryPath = '.'):
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
