import os
import sys
import argparse
import subprocess

os.chdir(subprocess.check_output('git rev-parse --show-toplevel', shell=True, text=True).strip())
sys.path.append('.')

import ipd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("codedir", type=str, default='[gitroot]/[projname]')
    parser.add_argument("--dryrun", action='store_true', default=False)
    parser.add_argument("--excludefile", default='[gitroot]/.yapf_exclude')
    parser.add_argument("--hashfile", default='[gitroot]/.yapf_hash')
    args = parser.parse_args()
    ipd.dev.yapf_fast(args.codedir, args.dryrun, args.excludefile, args.hashfile)

if __name__ == '__main__':
    main()
