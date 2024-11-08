import pytest

th = pytest.importorskip('torch')
import ipd

def main():
    test_get_high_t_frames_from_file()

def test_get_high_t_frames_from_file():
    ipd.sym.high_t.get_high_t_frames_from_file(ipd.tests.path('pdb/icos_t3_1js9.pdb'))

if __name__ == '__main__':
    main()
