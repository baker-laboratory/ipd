from pathlib import Path
import os
from os.path import join, isdir, abspath, exists
import sys
import ipd

def install_ipd_pre_commit_hook(projdir, path=None):
    if path: projdir = join(projdir, path)
    projdir = abspath(projdir)
    if isdir(join(projdir, '.git')):
        hookdir = join(projdir, '.git/hooks')
    else:
        gitdir = Path(join(projdir, '.git')).read_text()
        assert gitdir.startswith('gitdir: ')
        hookdir = abspath(join(projdir, gitdir[7:].strip(), 'hooks'))
    frm, to = abspath(f'{ipd.projdir}/../git_pre_commit.sh'), f'{hookdir}/pre-commit'
    if exists(to): return
    os.makedirs(os.path.basename(to), exist_ok=True)
    assert os.path.exists(frm)
    print(f'symlinking {frm} {to}')
    os.symlink(frm, to)

if __name__ == '__main__':
    install_pre_commit_hook(sys.argv[1])
