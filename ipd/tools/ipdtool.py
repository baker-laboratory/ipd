import sys
from pathlib import Path
import os

import git
from box import Box

import ipd
from ipd.dev.cli.clibase import CliBase

class IPDTool(CliBase):
    def __init__(self, secretfile: str = '~/.secrets'):
        self.secrets = Box()
        self._add_secrets(secretfile)
        super().__init__()

    def _add_secrets(self, secretfile=''):
        if not secretfile: return
        secrets: list[str] = Path(os.path.expanduser(secretfile)).read_text().splitlines()
        self.secrets |= Box({s.split('=')[0].replace('export ', ''): s.split('=')[1] for s in secrets})

    def _fill_secrets(self, stuff: list[str] | str):
        if isinstance(stuff, list): return [self._fill_secrets(_) for _ in stuff]
        for k, v in self.secrets.items():
            stuff = stuff.replace(f'${{{k}}}', v)
            stuff = stuff.replace(f'{k}', v)
        return stuff

    def update(self):
        repo = git.Repo(f'{ipd.projdir}/..')
        assert not repo.is_dirty()
        old = repo.head.commit
        repo.remotes.origin.pull()
        if repo.head.commit != old:
            ipd.dev.bash(f'{sys.executable} -mpip install -e {ipd.projdir}/..')
