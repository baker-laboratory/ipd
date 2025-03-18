import hashlib
from pathlib import Path
from typing import Annotated

import typer

class RunGroup():
    name = "RunGroup"

    def __init__(self, active_group=0, ngroups=1):
        self.active_group = active_group
        self.ngroups = ngroups

    def __contains__(self, identifier):
        this_group = hashlib.sha1(str(identifier).encode()).hexdigest()
        this_group = int(f'0x{this_group}', base=16) % self.ngroups
        return this_group == self.active_group

def parse_rungroup(s: str):
    if not s: return RunGroup()
    return RunGroup(*map(int, s.split(',')))

RunGroupArg = Annotated[RunGroup, typer.Option(parser=parse_rungroup, default_factory=str)]

def enumerate_inputs(fnames, pattern='*', rungroup=()):
    for fileordir in map(Path, fnames):
        inputs = [fileordir]
        if fileordir.is_dir():
            inputs = [path for path in fileordir.glob(pattern)]
        for i, path in enumerate(inputs):
            if path in rungroup:
                yield i, path
