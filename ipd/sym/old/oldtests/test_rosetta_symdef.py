import numpy as np
import pytest

import ipd

def main():
    test_rosetta_symdef()
    test_rosetta_symdata()
    test_rosetta_symdata_mod()

@pytest.mark.fast
def test_rosetta_symdef():
    pytest.importorskip("pyrosetta")
    s = ipd.sym.get_rosetta_symfile_contents("p4m_d4_c2")
    assert s.count("xyz") == 163

@pytest.mark.fast
def test_rosetta_symdata():
    pytest.importorskip("pyrosetta")
    d = ipd.sym.get_rosetta_symdata("p4m_d4_c2")
    assert d.get_num_virtual() == 163
    origs = set()
    for k, v in d.get_virtual_coordinates().items():
        o = tuple(v.get_origin())
        assert np.allclose(o[0] % 1.0, 0.0)
        assert np.allclose(o[1] % 1.0, 0.0)
        assert np.allclose(o[2] % 1.0, 0.0)
        origs.add(o)
    assert len(origs) == 21

@pytest.mark.fast
def test_rosetta_symdata_mod():
    pytest.importorskip("pyrosetta")
    scale = 137.34028439
    d, _symfilestr = ipd.sym.get_rosetta_symdata_modified("p4m_d4_c2", scale_positions=scale)
    origs = set()
    for k, v in d.get_virtual_coordinates().items():
        o = tuple(v.get_origin())
        assert np.allclose(o[0] % scale, 0.0, atol=1e-4) or np.allclose(o[0] % scale - scale, 0.0, atol=1e-4)
        assert np.allclose(o[1] % scale, 0.0, atol=1e-4) or np.allclose(o[1] % scale - scale, 0.0, atol=1e-4)
        assert np.allclose(o[2] % scale, 0.0, atol=1e-4) or np.allclose(o[2] % scale - scale, 0.0, atol=1e-4)
        origs.add(o)
    assert len(origs) == 21

if __name__ == "__main__":
    main()
