import subprocess

import ipd

def git_status(header=None, footer=None, printit=False):
    srcdir = ipd.proj_dir
    with ipd.dev.cd(srcdir):
        s = ''
        if header: s += f'{f" {header} ":@^80}\n'
        try:
            subprocess.check_output(['git', 'config', '--global', '--add', 'safe.directory', srcdir])
            s += subprocess.check_output(['git', 'log', '-n1']).decode()
            s += subprocess.check_output(['git', 'status']).decode()
        except subprocess.CalledProcessError as e:
            s += 'error fetching git status'
            s += str(e)
        if footer: s += f'\n{f" {footer} ":@^80}'
        if printit: print(s)
        return s
