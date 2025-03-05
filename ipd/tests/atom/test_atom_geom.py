import pytest
import numpy as np
import ipd

pytest.importorskip('biotite')

# import ipd.atom.atom_geom

def main():
    ipd.tests.maintest(namespace=globals())

def test_seqaln_rmsfit():
    atoms = ipd.pdb.readatoms(ipd.dev.package_testcif_path('1dxh'), assembly='largest', het=False)
    # for i, a in enumerate(atoms):
    # ipd.pdb.dump(a, f'lib/ipd/test{i}.pdb')
    frames, rms, matches = ipd.atom.find_frames_by_seqaln_rmsfit(atoms)['frames rmsd seqmatch']
    assert np.allclose(rms, 0, atol=1e-3)
    assert np.allclose(matches, 1)
    assert len(frames) == len(rms) == len(matches) == 1
    assert matches[0].shape == (12, )

def test_seqaln_rmsfit_multicomp_substruct():
    atoms = ipd.atom.load(ipd.dev.package_testdata_path('pdb/chelsea_tube_1.pdb.gz'))
    found = ipd.atom.find_frames_by_seqaln_rmsfit(atoms)
    assert len(found.frames[0]) == 3
    assert len(found.frames[1]) == 6
    assert np.allclose(1, found.seqmatch[0])
    assert np.allclose(1, found.seqmatch[1])

def test_stub():
    atoms = ipd.tests.top7
    stub = ipd.atom.stub(atoms)
    assert ipd.homog.hvalid44(stub)

if __name__ == '__main__':
    main()
