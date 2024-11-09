import ipd
import torch

def test_motif_manager():
    motif = ipd.motif.NullMotifManager()
    xyz = torch.randn((10, 3))
    assert torch.allclose(motif(xyz), xyz)

if __name__ == '__main__':
    test_motif_manager()
