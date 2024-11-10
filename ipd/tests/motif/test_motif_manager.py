import ipd
import pytest

th = pytest.importorskip('torch')

def test_motif_manager():
    motif = ipd.motif.NullMotifManager()
    xyz = th.randn((10, 3))
    assert th.allclose(motif(xyz), xyz)

if __name__ == '__main__':
    test_motif_manager()
