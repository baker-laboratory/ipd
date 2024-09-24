import sys
import argparse
import subprocess

parser = argparse.ArgumentParser(
    prog='Prettier Protein Project Service',
    description='Runs backend sharing service for Prettier Protein Project PyMol Plugin',
)
parser.add_argument('--port', type=int, default=12345)
parser.add_argument('--dburl', type=str, default='postgresql://sheffler@localhost:5432/ppp')
parser.add_argument('--datadir', type=str, default='/projects/ml/prettier_protein_project/server_data')
parser.add_argument('--update_libs', type=str, default=None)
parser.add_argument('--loglevel', type=str, default='info')

def update_libs(libdir):
    if not libdir: return
    print(f'UPDATING LIBRARIES IN {libdir}')
    subprocess.check_output(f'cd {libdir}/ipd && git pull'.split(), shell=True)
    subprocess.check_output(f'cd {libdir}/willutil && git pull'.split(), shell=True)
    subprocess.check_output(f'cd {libdir}/wills_pymol_crap && git pull'.split(), shell=True)

def main():
    args = parser.parse_args()
    update_libs(args.update_libs)
    import ipd
    print(f'STARTING SERVER localhost:{args.port} database: {args.dburl} datadir: {args.datadir}')
    ipd.ppp.server.run(args.port, args.dburl, args.datadir, args.loglevel)

if __name__ == '__main__':
    main()
