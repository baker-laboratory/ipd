import pytest
import os

def main():
    test_ppppp()

def test_ppppp():
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    pymol = pytest.importorskip('pymol')
    pymol.pymol_argv = ['pymol']
    pymol.finish_launching()
    from ipd.ppp.plugin.ppppp import run_ppppp_gui
    run_ppppp_gui()
    # from ipd.pymol import ppppp

if __name__ == '__main__':
    main()
