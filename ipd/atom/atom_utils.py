import os
import numpy as np
import ipd

bs = ipd.lazyimport('biotite.structure')

load = ipd.pdb.readatoms
dump = ipd.pdb.dump

def get(pdbcode, path='', **kw):
    if path: fname = os.path.join(path, f'{pdbcode}.bcif.gz')
    else: fname = ipd.dev.package_testcif_path(pdbcode)
    assert len(pdbcode) == 4, f'bad pdbcode {pdbcode}'
    return load(fname, path=path, **kw)

def is_atomarray(atoms):
    from biotite.structure import AtomArray
    return isinstance(atoms, AtomArray)

def is_atomarraystack(atoms):
    from biotite.structure import AtomArrayStack
    return isinstance(atoms, AtomArrayStack)

def is_atoms(atoms):
    return is_atomarray(atoms) or is_atomarraystack(atoms)

def split(atoms, order=None, bychain=None, nasu=None, min_chain_atoms=0):
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
            while len(split[0]) < min_chain_atoms and len(split) > 1:
                merge_chains(split[0], split.pop(1))
            i = 1
            while i < len(split):
                if len(split[i]) < min_chain_atoms:
                    merge_chains(split[i - 1], split.pop(i))
                else:
                    i += 1
            return split
    raise TypeError(f'bad split args {type(atoms)=} {order=} {bychain=}')

def merge_chains(atoms1, atoms2):
    assert len(np.unique(atoms1.chain_id)) == 1
    if len(atoms1): atoms2.chain_id[:] = atoms1.chain_id[0]
    atoms1 += atoms2

def split_chains(atoms, **kw):
    return split(atoms, bychain=True, **kw)

def chain_dict(atoms):
    """Group an AtomArray by chain_id and return a dictionary."""
    chain_ids = np.unique(atoms.chain_id)  # Get unique chain IDs
    chain_groups = {chain: atoms[atoms.chain_id == chain] for chain in chain_ids}
    return ipd.Bunch(chain_groups)

@ipd.dev.iterize_on_first_param(basetype='AtomArray')
def to_seq(atoms, concat=True) -> 'biotite.sequence.Sequence':
    try:
        biotite_seq = bs.to_sequence(atoms, allow_hetero=True)  # oddly slow
    except (IndexError, bs.BadStructureError):
        return None
    if concat: return ipd.dev.addreduce(biotite_seq[0])
    return biotite_seq[0]

def atoms_to_seqstr(atoms):
    idx = bs.get_residue_starts(atoms)
    return ''.join(bs.info.one_letter_code(x) for x in atoms.res_name[idx])

def seqalign(atoms1, atoms2):
    import biotite.sequence.align as align
    isprot1, isprot2 = is_protein([atoms1, atoms2])
    assert isprot1 == isprot2
    seq1, seq2 = to_seq([atoms1, atoms2])
    if not seq1 or not seq2: return None, np.zeros((0, 2)), 0
    if isprot1: matrix = align.SubstitutionMatrix.std_protein_matrix()
    else: matrix = align.SubstitutionMatrix.std_nucleic_matrix()
    # matrix = align.SubstitutionMatrix(matrix.get_alphabet1(), matrix.get_alphabet2(), 'IDENTITY')
    aln = align.align_optimal(
        seq1,
        seq2,
        matrix,
        gap_penalty=-1000,
        terminal_penalty=True,
        local=False,
        max_number=1,
    )
    assert len(aln) == 1 and (aln := aln[0])
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
    bbonly=False,
    chaindict=False,
    het=True,
    chains=None,
    **kw,
) -> 'Atoms':
    if isinstance(atoms, bs.AtomArrayStack):
        assert len(atoms) == 1
        atoms = atoms[0]
    meta = ipd.dev.get_metadata(atoms)
    if caonly: atoms = atoms[atoms.atom_name == 'CA']
    elif bbonly: atoms = atoms[atoms.atom_nameisin(('CA', 'N', 'C', 'O'))]
    if not het: atoms = atoms[~atoms.hetero]
    if chains is not None:
        atoms = atoms[np.isin(atoms.chain_id, chains)]
    if chaindict: atoms = ipd.atom.chain_dict(atoms)
    if chainlist: atoms = ipd.atom.split(atoms)
    ipd.dev.set_metadata(atoms, meta)
    return atoms

def pick_representative_chains(atomslist):
    chains = []
    for atoms in atomslist:
        crange = chain_ranges(atoms)
        assert len(crange) == 1
        crange = list(crange.values())[0][0]
        chains.append(atoms[crange[0]:crange[1]])
    return chains

@ipd.dev.iterize_on_first_param(basetype='AtomArray', asnumpy=True)
def is_protein(atoms, strict_protein_or_nucleic=False) -> bool:
    """
    Check if an atomic structure is a protein.

    This function checks if an atomic structure is a protein by comparing the
    fraction of CA atoms to the total number of atoms.

    Args:
        atoms (bs.AtomArray):
            Atomic structure object containing atom name data.

    Returns:
        bool:
            True if the structure is a protein, False otherwise.

    Example:
        >>> import ipd
        >>> atoms = ipd.atom.get('1qys')
        >>> print(is_protein(atoms))
        True
    """
    if not len(atoms): return np.nan
    if strict_protein_or_nucleic: raise NotImplementedError()
    frac = np.sum(atoms.atom_name == 'CA') / bs.get_residue_count(atoms)
    return frac > 0.9

def join(atomslist, one_letter_chain=True):
    formt = f'0{len(str(len(atomslist)))}'
    for i, atoms in enumerate(atomslist):
        atoms.chain_id = f'S{i:{formt}}' + atoms.chain_id
    if one_letter_chain:
        unique_ids = ipd.dev.UniqueIDs()
        for i, atoms in enumerate(atomslist):
            atoms.chain_id = unique_ids(atoms.chain_id)
    atoms = ipd.dev.addreduce(atomslist)

    assert len(np.unique(atoms.chain_id)) == len(atomslist) * len(np.unique(atomslist[0].chain_id))
    assert isinstance(atoms, bs.AtomArray)
    return atoms
