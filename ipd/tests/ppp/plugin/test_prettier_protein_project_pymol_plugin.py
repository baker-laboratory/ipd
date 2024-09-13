import pytest
import os
import time

def main():
    test_ppppp()
    print('test_ppppp DONE', flush=True)

def test_ppppp():
    # os.environ['QT_QPA_PLATFORM'] = 'xcb'
    pymol = pytest.importorskip('pymol')
    pymol.pymol_argv = ['pymol', '-q']
    pymol.finish_launching()
    # from ipd.ppp.plugin.ppppp import run_ppppp_gui
    # ui = run_ppppp_gui()
    # while time.sleep(1): pass
    # assert 0
    # from ipd.pymol import ppppp

if __name__ == '__main__':
    main()
