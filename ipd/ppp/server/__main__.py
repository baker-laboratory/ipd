import os
import sys
import ipd

def main():
    port = int(sys.argv[1])
    datadir = os.path.realpath(sys.argv[2])
    ipd.ppp.server.run(port, datadir)

if __name__ == '__main__':
    main()
