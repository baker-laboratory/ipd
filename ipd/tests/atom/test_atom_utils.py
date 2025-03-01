import pytest
import numpy as np
import ipd

pytest.importorskip('biotite')

def main():
    ipd.tests.maintest(namespace=globals())

def _small_atoms_2chain():
    import biotite.structure as struc
    atoms1 = ipd.tests.fixtures.small_atoms
    atoms2 = atoms1.copy()
    atoms1.chain_id[:] = 'A'
    atoms2.chain_id[:] = 'B'
    atoms = struc.concatenate([atoms1, atoms2])
    return atoms, atoms1, atoms2

def test_split():
    atoms, atoms1, atoms2 = _small_atoms_2chain()
    atoms3, atoms4 = ipd.atom.split(atoms, 2)
    assert np.all(atoms3.chain_id == 'A')
    assert np.all(atoms4.chain_id == 'B')
    assert np.all(atoms1.res_name == atoms4.res_name)

def test_chain_dict():
    atoms, atoms1, atoms2 = _small_atoms_2chain()
    chains = ipd.atom.chain_dict(atoms)
    assert np.all(chains.A.chain_id == 'A')
    assert np.all(chains.B.chain_id == 'B')
    assert np.all(chains.A.res_name == chains.B.res_name)

def test_seqaln_rmsfit():
    atoms = ipd.pdb.readatoms(ipd.dev.package_testcif_path('1dxh'), biounit='largest', het=False)
    # for i, a in enumerate(atoms):
    # ipd.pdb.dump(a, f'lib/ipd/test{i}.pdb')
    frames, rms, matches = ipd.atom.frames_by_seqaln_rmsfit(atoms)
    assert np.allclose(rms, 0, atol=1e-3)
    assert np.allclose(matches, 1)
    assert matches.shape == (12, )

def test_stub():
    atoms = ipd.tests.top7
    stub = ipd.atom.stub(atoms)
    assert ipd.homog.hvalid44(stub)

if __name__ == '__main__':
    main()
