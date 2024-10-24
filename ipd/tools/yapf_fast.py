import os
import pathlib
import subprocess

def main():
    cmd = 'find . -name *.py -exec md5sum {} ;'
    files = subprocess.check_output(cmd.split()).decode().split(os.linesep)
    files = {f for f in files if f}
    exclude = set()
    if os.path.exists('.yapf_exclude'):
        exclude = pathlib.Path('.yapf_exclude').read_text().split()
        exclude = {f for f in exclude if f}
    prev = set()
    if os.path.exists('.yapf_hash'):
        prev = pathlib.Path('.yapf_hash').read_text().split(os.linesep)
        prev = {f for f in prev if f}
    diff = {x.split()[1] for x in (files - prev)} - exclude
    if diff:
        os.system(f'yapf -ip {" ".join(diff)}')
        os.system('find . -name \\*.py -exec md5sum {} \\; > .yapf_hash')

if __name__ == '__main__':
    main()
