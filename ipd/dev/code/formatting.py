import os
import pathlib
import subprocess

import ipd

def yapf_fast(codedir: str, dryrun: bool, excludefile: str, hashfile: str):
    excludefile, hashfile = ipd.dev.substitute_project_vars(excludefile, hashfile)
    cmd = f'find {codedir} -name *.py -exec md5sum {{}} ;'
    files = set(subprocess.check_output(cmd.split()).decode().strip().split(os.linesep))
    exclude = set()
    if os.path.exists('.yapf_exclude'):
        exclude = set(pathlib.Path('.yapf_exclude').read_text().strip().split())
    prev = set()
    if os.path.exists('.yapf_hash'):
        prev = set(pathlib.Path('.yapf_hash').read_text().strip().split(os.linesep))
    diff = {x.split()[1] for x in (files - prev)} - exclude
    yapfcmd = f'yapf -ip --style {codedir}/../pyproject.toml -m {" ".join(diff)}'
    if diff:
        print(yapfcmd)
        if not dryrun:
            os.system(yapfcmd)
            os.system(f'find {codedir} -name \\*.py -exec md5sum {{}} \\; > .yapf_hash')
            exit(-1)

if __name__ == '__main__':
    main()
