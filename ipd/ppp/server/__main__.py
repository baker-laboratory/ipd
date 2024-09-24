import os
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser(
    prog='Prettier Protein Project Service',
    description='Runs backend sharing service for Prettier Protein Project PyMol Plugin',
)
parser.add_argument('--port', type=int, default=12345)
parser.add_argument('--dburl', type=str, default='postgresql://sheffler@localhost:5432/ppp')
parser.add_argument('--datadir', type=str, default=os.path.abspath('./data'))
parser.add_argument('--loglevel', type=str, default='info')

def main():
    args = parser.parse_args()
    import ipd
    print(f'STARTING SERVER localhost:{args.port} database: {args.dburl} datadir: {args.datadir}')
    ipd.ppp.server.run(args.port, args.dburl, args.datadir, args.loglevel)

if __name__ == '__main__':
    main()
