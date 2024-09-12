import pytest

def main():
    test_ppppp()

def test_ppppp():
    pymol = pytest.importorskip('pymol')
    pymol.finish_launching()
    # from ipd.pymol import ppppp

if __name__ == '__main__':
    main()
