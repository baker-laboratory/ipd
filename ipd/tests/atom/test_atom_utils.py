import pytest
import numpy as np
import ipd

pytest.importorskip("biotite")

def main():
    # test_seqaln_rmsfit_multicomp_substruct()
    ipd.tests.maintest(namespace=globals())

@pytest.fixture(scope="session")
def atoms():
    return ipd.atom.load("6u9d", assembly="largest")[0]

def _small_atoms_2chain():
    import biotite.structure as struc

    atoms1 = ipd.tests.fixtures.small_atoms
    atoms2 = atoms1.copy()
    atoms1 . chain_id [:] = 'A'
    atoms2 . chain_id [:] = 'B'
    atoms = struc.concatenate([atoms1, atoms2])
    return atoms, atoms1, atoms2

def test_split():
    atoms, atoms1, atoms2 = _small_atoms_2chain()
    atoms3, atoms4 = ipd.atom.split(atoms, 2)
    assert np . all(atoms3 . chain_id == 'A')
    assert np . all(atoms4 . chain_id == 'B')
    assert np.all(atoms1.res_name == atoms4.res_name)

def test_chain_dict():
    atoms, atoms1, atoms2 = _small_atoms_2chain()
    chains = ipd.atom.chain_dict(atoms)
    assert np . all(chains . A . chain_id == 'A')
    assert np . all(chains . B . chain_id == 'B')
    assert np.all(chains.A.res_name == chains.B.res_name)

def test_chain_range():
    atoms, atoms1, atoms2 = _small_atoms_2chain()
    chains = ipd.atom.chain_ranges(atoms)
    assert chains == {"A": [(0, 8)], "B": [(8, 16)]}
    chains = ipd.atom.chain_ranges([atoms1, atoms2])
    assert chains == [{"A": [(0, 8)]}, {"B": [(0, 8)]}]

def test_atoms(atoms):
    # test that we can load and process the atoms
    assert atoms is not None
    assert len(atoms) > 0
    assert np.all(np.isfinite(atoms.coord))  # all coordinates should be finite
    assert len(np.unique(atoms.chain_id)) > 1  # should have multiple chains

def test_is_atoms(atoms):
    assert ipd.atom.is_atoms(atoms)

def test_split_bychain(atoms):
    chains = ipd.atom.split(atoms, bychain=True)
    assert all(ipd.atom.is_atoms(c) for c in chains)
    assert sum(len(c) for c in chains) == len(atoms)

def test_chain_dict_keys(atoms):
    chains = ipd.atom.chain_dict(atoms)
    assert set(chains.keys()) == set(np.unique(atoms.chain_id))

def test_split_order(atoms):
    chains = ipd.atom.split(atoms, bychain=True)
    rejoined = ipd.atom.join(chains, one_letter_chain=False)
    assert len(rejoined) == len(atoms)

def test_to_seq(atoms):
    ca_atoms = ipd.atom.select(atoms, caonly=True)
    seqs, starts, stops, isprot = ipd.atom.to_seq(ca_atoms)
    assert isinstance(seqs, list)
    assert all(len(seqs[i]) == stops[i] - starts[i] for i in range(len(seqs)))

def test_atoms_to_seqstr(atoms):
    ca = ipd.atom.select(atoms, caonly=True)
    chains = ipd.atom.split(ca)
    for ch in chains:
        s, isprot = ipd.atom.atoms_to_seqstr(ch)
        assert isinstance(s, str)
        assert isprot is True

def test_seqalign(atoms):
    ca = ipd.atom.select(atoms, caonly=True)
    chains = ipd.atom.split(ca)
    if len(chains) >= 2:
        aln, match, score = ipd.atom.seqalign(chains[0], chains[1])
        assert match.shape[1] == 2
        assert 0 <= score <= 1

def test_chain_ranges_consistency(atoms):
    d = ipd.atom.chain_ranges(atoms)
    total = sum(stop - start for ranges in d.values() for start, stop in ranges)
    assert total == len(atoms)

def test_chain_id_ranges_consistency(atoms):
    d = ipd.atom.chain_id_ranges(atoms)
    total = sum(stop - start for ranges in d.values() for start, stop in ranges)
    assert total == len(atoms)

def test_select_element_filter(atoms):
    oxy = ipd.atom.select(atoms, element="O")
    assert np.all(oxy.element == "O")

def test_pick_representative_chains(atoms):
    chains = ipd.atom.split(atoms)
    reps = ipd.atom.pick_representative_chains(chains)
    assert all(len(r) > 0 for r in reps)

def test_is_protein(atoms):
    assert np.all(ipd.atom.is_protein(atoms))

def test_join_preserves_length(atoms):
    chains = ipd . atom . split(atoms )
    joined = ipd . atom . join (chains)
    assert isinstance(joined, type(atoms))
    assert len(joined) == len(atoms)

def test_remove_garbage_residues(atoms):
    filtered = ipd.atom.remove_garbage_residues(atoms)
    assert all(r not in ipd.atom.garbage_residues for r in np.unique(filtered.res_name))

def test_remove_nan_atoms(atoms):
    clean = ipd.atom.remove_nan_atoms(atoms)
    assert not np.isnan(clean.coord).any()

def test_centered(atoms):
    centered_atoms = ipd.atom.centered(atoms)
    cen = ipd.atom.bs.mass_center(centered_atoms)
    assert np.allclose(cen, 0, atol=1)

def test_select_atom_name(atoms):
    ca_atoms = ipd.atom.select(atoms, atom_name="CA")
    assert np.all(ca_atoms.atom_name == "CA")

if __name__ == "__main__":
    main()
