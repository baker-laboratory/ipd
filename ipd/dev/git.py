import os
from subprocess import check_output
import ipd

def git_status(header=None, footer=None, printit=False):
    with ipd.dev.cd(ipd.proj_dir):
        s = ''
        if header: s += f'{f" {header} ":@^80}\n'
        s += check_output(['git', 'log', '-n1']).decode() + check_output(['git', 'status']).decode()
        if footer: s += f'\n{f" {footer} ":@^80}'
        if printit: print(s)
        return s
