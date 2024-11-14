import os
from pathlib import Path

import ipd

def set_from_file(fname):
    if os.path.exists(fname):
        return set(Path(fname).read_text().strip().split(os.linesep))
    return set()

def yapf_fast(codedir: str, dryrun: bool, excludefile: str, hashfile: str, conffile: str):
    """Run yapf on changed files in a git repo."""
    codedir, excludefile, hashfile, conffile = ipd.dev.substitute_project_vars(codedir, excludefile, hashfile, conffile)
    with ipd.dev.cd(os.path.dirname(excludefile)):
        cmd = fr'find {codedir} -name \*.py -exec md5sum {{}} \;'
        prevhash = ipd.dev.run(cmd)
        files = set(prevhash.strip().split(os.linesep))
        exclude = set_from_file(excludefile)
        prev = set_from_file(hashfile)
        diff = {x.split()[1] for x in (files - prev)} - exclude
        if diff:
            yapfcmd = f'yapf -ip --style {conffile} -m {" ".join(diff)}'
            print(yapfcmd)
            if not dryrun:
                os.system(yapfcmd)
                os.system(f'find {codedir} -name \\*.py -exec md5sum {{}} \\; > {hashfile}')
                if prevhash != Path(hashfile).read_text():
                    exit(-1)
