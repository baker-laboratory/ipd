import argparse
import os
import time

import ipd

parser = argparse.ArgumentParser(
    prog='Prettier Protein Project Service',
    description='Runs backend sharing service for Prettier Protein Project PyMol Plugin',
)
parser.add_argument('--port', type=int, default=12345)
parser.add_argument('--workers', type=int, default=9)
parser.add_argument('--dburl', type=str, default=None)  #'postgresql://sheffler@127.0.0.1:5432/ppp')
parser.add_argument('--datadir', type=str, default=os.path.abspath('./ppp_server_data'))
parser.add_argument('--loglevel', type=str, default='info')
parser.add_argument('--stress_test_polls', action='store_true', default=False)

def main():
    args = ipd.dev.Bunch(parser.parse_args())
    print(f'STARTING SERVER 127.0.0.1:{args.port} database: {args.dburl} datadir: {args.datadir}')
    ipd.ppp.server.run(**args)
    while True:
        time.sleep(1)

if __name__ == '__main__':
    main()
