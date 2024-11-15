import sys
from pathlib import Path
import os
from typing_extensions import Annotated

import git
import typer

import ipd
from ipd.dev.cli.clibase import CliBase

class IPDTool(CliBase):
    def __init__(self, secretfile: str = '~/.secrets'):
        self.secrets = ipd.Bunch()
        self._add_secrets(secretfile)
        super().__init__()

    def _add_secrets(self, secretfile=''):
        if not os.path.exists(os.path.expanduser(secretfile)): return
        secrets: list[str] = Path(os.path.expanduser(secretfile)).read_text().splitlines()
        self.secrets |= ipd.Bunch({s.split('=')[0].replace('export ', ''): s.split('=')[1] for s in secrets})

    def _fill_secrets(self, stuff: list[str] | str):
        if isinstance(stuff, list): return [self._fill_secrets(_) for _ in stuff]
        for k, v in self.secrets.items():
            stuff = stuff.replace(f'${{{k}}}', v)
            stuff = stuff.replace(f'{k}', v)
        return stuff

    def update(self):
        repo = git.Repo(f'{ipd.projdir}/..')
        if repo.is_dirty():
            print('ipd repo dirty, not updating...')
            return
        old = repo.head.commit
        repo.remotes.origin.pull()
        if repo.head.commit != old:
            print(f'ipd: new head {repo.head.commit} old head {old}')
            ipd.dev.bash(f'{sys.executable} -mpip install -e {ipd.projdir}/..')

    def clone(self,
              url: str,
              branch: Annotated[str, typer.Argument()] = 'main',
              path: Annotated[str | None, typer.Argument()] = None,
              secrets: str = '',
              norecurse: bool = False):
        path = path or branch
        if path == 'main': path = os.path.basename(url).replace('.git', '')
        if 'git.ipd.uw.edu' in url and '@' not in url:
            url = url.replace('git.ipd.uw.edu', 'GITLAB_USER:GITLAB_TOKEN@git.ipd.uw.edu')
        elif 'github.com' in url and '@' not in url:
            url = url.replace('github.com', 'GITHUB_USER:GITHUB_TOKEN@github.com')
        url = self._fill_secrets(url)  # type: ignore
        # git.Repo.clone_from(url, path, branch=branch)
        os.system(f'git clone {url} {path} -b {branch} --recursive')
