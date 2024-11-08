from pathlib import Path
import os
from os.path import join, isdir, abspath, exists
import sys

def install_pre_commit_hook(projdir, path=None, precommit='.git-pre-commit'):
    if path: projdir = join(projdir, path)
    projdir = abspath(projdir)
    if not exists(join(projdir, precommit)): return
    if isdir(join(projdir, '.git')):
        hookdir = join(projdir, '.git/hooks')
    else:
        gitdir = Path(join(projdir, '.git')).read_text()
        assert gitdir.startswith('gitdir: ')
        hookdir = abspath(join(projdir, gitdir[7:].strip(), 'hooks'))
    frm, to = f'{projdir}/{precommit}', f'{hookdir}/pre-commit'
    if os.path.exists(to): return
    print(f'symlinking {frm} {to}')
    os.symlink(frm, to)

if __name__ == '__main__':
    install_pre_commit_hook(sys.argv[1])
