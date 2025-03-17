import os
import typing
import numpy as np
import ipd

if typing.TYPE_CHECKING:
    from biotite.structure import AtomArray
    from biotite.sequence import Sequence

bs = ipd.lazyimport('biotite.structure')

from ipd.pdb.readstruct import readatoms as load, dump as dump

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

def split(atoms, order=None, bychain=None, nasu=None, min_chain_atoms=0, **kw) -> list['AtomArray']:
    if nasu is not None:
        assert not order and len(atoms) % nasu == 0, f'bad nasu for leno{len(atoms)} {nasu}'
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
    assert len(np.unique(atoms1.chain_id)) == 1, f'bad chain_id {atoms1.chain_id}'
    if len(atoms1): atoms2.chain_id[:] = atoms1.chain_id[0]
    atoms1 += atoms2

def split_chains(atoms, **kw):
    return split(atoms, bychain=True, **kw)

def chain_dict(atoms):
    """Group an AtomArray by chain_id and return a dictionary."""
    chain_ids = np.unique(atoms.chain_id)  # Get unique chain IDs
    chain_groups = {chain: atoms[atoms.chain_id == chain] for chain in chain_ids}
    return ipd.Bunch(chain_groups)

@ipd.dev.timed
@ipd.dev.iterize_on_first_param(basetype='AtomArray')
def to_seq(atoms, pick_longest=False) -> 'Sequence':
    import biotite.sequence
    # atoms = atoms[~atoms.hetero & np.isin(atoms.atom_name, np.array(['CA', 'P'], dtype='>U5'))]
    assert np.all(np.isin(atoms.atom_name, np.array(['CA', 'P'],
                                                    dtype='>U5'))), f'bad atom names {atoms.atom_name}'
    seqs, isprot = [], []
    for catoms in split_chains(atoms):
        seq, _isprot = atoms_to_seqstr(catoms)
        isprot.append(_isprot)
        if isprot: seqs.append(biotite.sequence.ProteinSequence(seq))
        else: seqs.append(biotite.sequence.NucleotideSequence(seq))
    lens = np.array(list(map(len, seqs)))
    starts, stops = np.cumsum([0] + lens), np.cumsum(lens + [0])
    if pick_longest:
        i = np.argmax(lens)
        return seqs[i], starts[i], stops[i], isprot[i]
    return seqs, starts, stops, isprot
    # try:
    #     biotite_seq = bs.to_sequence(atoms, allow_hetero=True)  # oddly slow
    #     if concat: return ipd.dev.addreduce(biotite_seq[0])
    #     return biotite_seq[0]
    # except (IndexError, bs.BadStructureError):
    #     return None

@ipd.dev.timed
def atoms_to_seqstr(atoms):
    # atoms = atoms[~atoms.hetero & np.isin(atoms.atom_name, np.array(['CA', 'P'], dtype='>U5'))]
    if len(atoms) == 0: return None
    # protein = np.isin(atoms.res_name, np_amino_acid)
    # nucleic = np.isin(atoms.res_name, np_nucleotide)
    # ic(atoms[~protein])
    # assert all(nucleic) or all(protein)
    # ic(atoms.atom_name[0], atoms.atom_name[0]=='CA')
    if atoms.atom_name[0] == 'CA':
        return ''.join(amino_acid_321[x] for x in atoms.res_name), True
    return ''.join(nucleotide_321[x] for x in atoms.res_name), False

@ipd.dev.timed
def seqalign(atoms1, atoms2):
    assert all(np.isin(atoms1.atom_name, np.array(['CA', 'P'],
                                                  dtype='>U5'))), f'bad atom names {atoms1.atom_name}'
    assert all(np.isin(atoms2.atom_name, np.array(['CA', 'P'],
                                                  dtype='>U5'))), f'bad atom names {atoms2.atom_name}'
    import biotite.sequence.align as align
    seq1, start1, stop1, isprot1 = to_seq(atoms1, pick_longest=True)
    seq2, start2, stop2, isprot2 = to_seq(atoms2, pick_longest=True)
    assert isprot1 == isprot2, f'bad seqalign {isprot1=} {isprot2=}'
    if not seq1 or not seq2: return None, np.zeros((0, 2)), 0
    if isprot1: matrix = align.SubstitutionMatrix.std_protein_matrix()
    else: matrix = align.SubstitutionMatrix.std_nucleic_matrix()
    # matrix = align.SubstitutionMatrix(matrix.get_alphabet1(), matrix.get_alphabet2(), 'IDENTITY')
    aln = align.align_optimal(
        seq1,
        seq2,
        matrix,
        gap_penalty=-100,
        terminal_penalty=True,
        local=True,
        max_number=1,
    )
    assert len(aln) == 1 and (aln := aln[0]), f'bad seqalign {len(aln)=} {aln=}'
    s1, s2 = aln.sequences
    match = aln.trace[(aln.trace[:, 0] >= 0) & (aln.trace[:, 1] >= 0)]
    matchfrac = 2 * len(match) / (len(s1) + len(s2))
    return aln, match, matchfrac

@ipd.dev.iterize_on_first_param(basetype='AtomArray')
def chain_ranges(atoms) -> dict[str, list[tuple[int, int]]]:
    assert is_atomarray(atoms)
    breaks = list(map(int, sorted(bs.get_chain_starts(atoms))))
    breaks.append(len(atoms))
    return chain_ranges_from_breaks(atoms, breaks)

def chain_ranges_from_breaks(atoms, breaks) -> dict[str, list[tuple[int, int]]]:
    result = {}
    for i, start in enumerate(breaks[:-1]):
        c = atoms.chain_id[start]
        stop = breaks[i + 1]
        assert stop <= len(atoms), f'bad stop {stop} {len(atoms)}'
        result.setdefault(str(c), []).append((start, stop))
    return result

@ipd.dev.iterize_on_first_param(basetype='AtomArray')
def chain_id_ranges(atoms) -> dict[str, list[tuple[int, int]]]:
    assert is_atomarray(atoms), f'bad atoms {type(atoms)=} {len(atoms)=}'
    breaks, breaks0 = [0], list(map(int, sorted(bs.get_chain_starts(atoms))))
    for i, b in enumerate(breaks[1:], start=1):
        if atoms.chain_id[b] != atoms.chain_id[breaks[i - 1]]:
            breaks.append(b)
    breaks.append(len(atoms))
    return chain_ranges_from_breaks(atoms, breaks)

def select(
    atoms: 'AtomsArray',
    chainlist=False,
    caonly=False,
    bbonly=False,
    chaindict=False,
    het=True,
    chains=None,
    **kw,
) -> 'AtomArray':
    if isinstance(atoms, bs.AtomArrayStack):
        assert len(atoms) == 1, f'bad select {len(atoms)=} {atoms=}'
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

    assert len(np.unique(atoms.chain_id)) == len(atomslist) * len(
        np.unique(atomslist[0].chain_id)
    ), f'bad join {len(np.unique(atoms.chain_id))} {len(atomslist) * len(np.unique(atomslist[0].chain_id))}'
    assert isinstance(atoms, bs.AtomArray), f'bad join {type(atoms)=} {len(atoms)=}'
    return atoms

amino_acid_321 = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLU': 'E',
    'GLN': 'Q',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V',
    'SEC': 'U',
    'PYL': 'O'  # Selenocysteine and Pyrrolysine (rare)
}
nucleotide_321 = {
    # RNA nucleotides
    'ADE': 'A',
    'CYT': 'C',
    'GUA': 'G',
    'URA': 'U',
    # DNA nucleotides
    'DA': 'A',
    'DC': 'C',
    'DG': 'G',
    'DT': 'T'
}
np_amino_acid = np.array(list(amino_acid_321.keys()), dtype='>U5')
np_nucleotide = np.array(list(nucleotide_321.keys()), dtype='>U5')
