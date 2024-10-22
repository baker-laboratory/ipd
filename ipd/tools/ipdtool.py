import sys

import git

import ipd
from ipd.dev.cli.clibase import CliBase

class IPDTool(CliBase):
    def update(self):
        repo = git.Repo(f'{ipd.projdir}/..')
        assert not repo.is_dirty()
        old = repo.head.commit
        repo.remotes.origin.pull()
        if repo.head.commit != old:
            ipd.dev.bash(f'{sys.executable} -mpip install -e {ipd.projdir}/..')
