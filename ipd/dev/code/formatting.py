import os
import pathlib
import subprocess

import ipd

def yapf_fast(codedir: str, dryrun: bool, excludefile: str, hashfile: str, conffile: str):
    codedir, excludefile, hashfile, conffile = ipd.dev.substitute_project_vars(codedir, excludefile, hashfile, conffile)
    with ipd.dev.cd(os.path.dirname(excludefile)):
        cmd = f'find {codedir} -name *.py -exec md5sum {{}} ;'
        files = set(subprocess.check_output(cmd.split()).decode().strip().split(os.linesep))
        exclude = set()
        if os.path.exists(excludefile):
            exclude = set(pathlib.Path(excludefile).read_text().strip().split())
        prev = set()
        if os.path.exists(hashfile):
            prev = set(pathlib.Path(hashfile).read_text().strip().split(os.linesep))
        diff = {x.split()[1] for x in (files - prev)} - exclude
        yapfcmd = f'yapf -ip --style {conffile} -m {" ".join(diff)}'
        if diff:
            print(yapfcmd)
            if not dryrun:
                os.system(yapfcmd)
                os.system(f'find {codedir} -name \\*.py -exec md5sum {{}} \\; > .yapf_hash')
                exit(-1)
