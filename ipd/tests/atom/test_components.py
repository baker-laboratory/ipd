import pytest
import numpy as np
import ipd

h = ipd.hnumpy

pytest.importorskip('biotite')

COMPONENTS_TEST_PDBS = [
    '6u9d',
]

def main():
    ipd.tests.maintest(namespace=globals())

def test_components_detection_6u9d():
    atoms = ipd.atom.load('6u9d', assembly='1')[0]
    atoms = ipd.atom.centered(atoms, primary_only=True)
    comp = ipd.atom.find_components_by_seqaln_rmsfit(atoms)
    assert len(comp.frames[0]) == 8
    assert len(comp.frames[1]) == 8
    # ipd.showme(comp.frames[0], 'frames_comp0', xyzlen=[11,11,11], weight=15)
    # ipd.showme(comp.frames[1], 'frames_comp1', xyzlen=[1e1,11,11], weight=15)
    # ipd.showme(atoms, name='original', xyzlen=[11,11,11], weight=15)
    # assert h.allclose(comp.frames[0], comp.frames[1])

def test_seqaln_rmsfit_1dxh():
    atoms = ipd.atom.get('1dxh', assembly='largest', het=False, chainlist=True)
    # for i, a in enumerate(atoms):
    # ipd.pdb.dump(a, f'lib/ipd/test{i}.pdb')
    findframes = ipd.atom.find_components_by_seqaln_rmsfit(atoms)
    atoms, frames, rms, matches = findframes['atoms frames rmsd seqmatch']
    assert np.allclose(rms, 0, atol=1e-3)
    assert np.allclose(matches, 1)
    assert len(frames) == len(rms) == len(matches) == 1
    assert matches[0].shape == (12, )

def test_seqaln_rmsfit_1g5q():
    atoms = ipd.atom.get('1g5q', assembly='largest', het=False, chainlist=True)
    # ipd.icv(len(atoms), len(atoms[0]))
    # ipd.icv(ipd.atom.chain_ranges(atoms))

    # for i, a in enumerate(atoms):
    # ipd.pdb.dump(a, f'lib/ipd/test{i}.pdb')
    found = ipd.atom.find_components_by_seqaln_rmsfit(atoms)
    found.remove_small_chains()
    # print(found)
    atoms, frames, rms, matches = found['atoms frames rmsd seqmatch']
    assert np.all(rms[0] < 1)
    assert np.allclose(matches, 1)
    ipd.print_table(h.xinfo(found.frames[1]), nohomog=1)
    assert len(frames) == len(rms) == len(matches) == 2
    assert matches[0].shape == (12, )

def test_seqaln_rmsfit_multicomp_substruct():
    atoms = ipd.atom.load(ipd.dev.package_testdata_path('pdb/chelsea_tube_1.pdb.gz'))
    found = ipd.atom.find_components_by_seqaln_rmsfit(atoms)
    assert len(found.frames[0]) == 6
    assert len(found.frames[1]) == 3
    assert np.allclose(1, found.seqmatch[0])
    assert np.allclose(1, found.seqmatch[1])

def test_stub():
    atoms = ipd.atom.load('1qys')
    stub = ipd.atom.stub(atoms)
    assert ipd.homog.hvalid44(stub)

if __name__ == '__main__':
    main()
