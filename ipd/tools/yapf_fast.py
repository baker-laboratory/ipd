import argparse
import os
import pathlib
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("codedir", type=str, default='.')
    parser.add_argument("--dry-run", action='store_true', default=False)
    args = parser.parse_args()

    cmd = f'find {args.codedir} -name *.py -exec md5sum {{}} ;'
    files = set(subprocess.check_output(cmd.split()).decode().strip().split(os.linesep))
    exclude = set()
    if os.path.exists('.yapf_exclude'):
        exclude = set(pathlib.Path('.yapf_exclude').read_text().strip().split())
    prev = set()
    if os.path.exists('.yapf_hash'):
        prev = set(pathlib.Path('.yapf_hash').read_text().strip().split(os.linesep))
    diff = {x.split()[1] for x in (files - prev)} - exclude
    ppath = f'{os.path.dirname(__file__)}/../../lib'
    yapfcmd = f'PYTHONPATH={ppath} python -m yapf -ip --style {args.codedir}/../pyproject.toml -m {" ".join(diff)}'
    if diff:
        print(yapfcmd)
        if not args.dry_run:
            os.system(yapfcmd)
            os.system(f'find {args.codedir} -name \\*.py -exec md5sum {{}} \\; > .yapf_hash')
            exit(-1)

if __name__ == '__main__':
    main()
