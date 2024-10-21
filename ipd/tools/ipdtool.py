import sys
import git
import ipd

from ipd.dev.cli.clibase import CliBase

class IPDTool(CliBase):
    def update(self):
        repo = git.Repo(f'{ipd.projdir}/..')
        repo.remotes.origin.pull()
        ipd.dev.bash(f'{sys.executable} -mpip install -e --upgrade {ipd.projdir}/..')

