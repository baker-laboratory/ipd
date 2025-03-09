import numpy as np
import ipd

bs = ipd.lazyimport('biotite.structure')

load = ipd.pdb.readatoms
dump = ipd.pdb.dump

def testdata(pdbcode, **kw):
    return load(ipd.dev.package_testcif_path(pdbcode), **kw)

def is_atomarray(atoms):
    from biotite.structure import AtomArray
    return isinstance(atoms, AtomArray)

def is_atomarraystack(atoms):
    from biotite.structure import AtomArrayStack
    return isinstance(atoms, AtomArrayStack)

def is_atoms(atoms):
    return is_atomarray(atoms) or is_atomarraystack(atoms)

def split(atoms, order=None, bychain=None, nasu=None, minlen=0):
    if nasu is not None:
        assert not order and len(atoms) % nasu == 0
        order = len(atoms) // nasu
    if not order and bychain is None:
        bychain = True
    if ipd.atom.is_atomarray(atoms):
        if order and not bychain:
            assert len(atoms) % order == 0, f'bad order for leno{len(atoms)} {order}'
            nasu = len(atoms) // order
            return [atoms[i * nasu:(i+1) * nasu] for i in range(order)]
        if bychain and not order:
            split = list(chain_dict(atoms).values())
            while len(split[0]) < minlen and len(split) > 1:
                split[0] += split.pop(1)
            i = 1
            while i < len(split):
                if len(split[i]) < minlen: split[i - 1] += split.pop(i)
                else: i += 1
            return split
    raise TypeError(f'bad split args {type(atoms)=} {order=} {bychain=}')

def split_chains(atoms, minlen=0):
    return split(atoms, bychain=True, minlen=minlen)

def chain_dict(atoms):
    """Group an AtomArray by chain_id and return a dictionary."""
    chain_ids = np.unique(atoms.chain_id)  # Get unique chain IDs
    chain_groups = {chain: atoms[atoms.chain_id == chain] for chain in chain_ids}
    return ipd.Bunch(chain_groups)

def atoms_to_seq(atoms):
    return ipd.dev.addreduce(bs.to_sequence(atoms)[0])  # oddly slow

def atoms_to_seqstr(atoms):
    idx = bs.get_residue_starts(atoms)
    return ''.join(bs.info.one_letter_code(x) for x in atoms.res_name[idx])

def seqalign(atoms1, atoms2):
    import biotite.sequence.align as align
    seq1 = atoms_to_seq(atoms1)
    seq2 = atoms_to_seq(atoms2)
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    aln = align.align_optimal(seq1, seq2, matrix)[0]
    s1, s2 = aln.sequences
    match = aln.trace[(aln.trace[:, 0] >= 0) & (aln.trace[:, 1] >= 0)]
    matchfrac = 2 * len(match) / (len(s1) + len(s2))
    return aln, match, matchfrac

@ipd.dev.iterize_on_first_param(basetype='AtomArray')
def chain_ranges(atoms):
    assert is_atomarray(atoms)
    result = {}
    starts = list(sorted(bs.get_chain_starts(atoms)))
    starts.append(len(atoms) + 1)
    for i, start in enumerate(starts[:-1]):
        c = atoms.chain_id[start]
        stop = starts[i + 1]
        result.setdefault(str(c), []).append((int(start), int(stop)))
    return result

def select(
    atoms,
    chainlist=False,
    caonly=False,
    chaindict=False,
    het=True,
    chains=None,
    **kw,
) -> 'Atoms':
    if isinstance(atoms, bs.AtomArrayStack):
        assert len(atoms) == 1
        atoms = atoms[0]
    if caonly: atoms = atoms[atoms.atom_name == 'CA']
    if not het: atoms = atoms[~atoms.hetero]
    if chains is not None:
        atoms = atoms[np.isin(atoms.chain_id, chains)]
    if chaindict: atoms = ipd.atom.chain_dict(atoms)
    if chainlist: atoms = ipd.atom.split(atoms)
    return atoms
