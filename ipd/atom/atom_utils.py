import numpy as np
import ipd

def is_atomarray(atoms):
    from biotite.structure import AtomArray
    return isinstance(atoms, AtomArray)

def split(atoms, order=None, bychain=None):
    if not order and bychain is None: bychain = True
    if ipd.atom.is_atomarray(atoms):
        if order and not bychain:
            assert len(atoms) % order == 0, f'bad order for leno{len(atoms)} {order}'
            nasu = len(atoms) // order
            return [atoms[i * nasu:(i+1) * nasu] for i in range(order)]
        if bychain and not order:
            return list(chain_dict(atoms).values())

    raise TypeError(f'bad split args {type(atoms)=} {order=} {bychain=}')

def chain_dict(atoms):
    """Group an AtomArray by chain_id and return a dictionary."""
    chain_ids = np.unique(atoms.chain_id)  # Get unique chain IDs
    chain_groups = {chain: atoms[atoms.chain_id == chain] for chain in chain_ids}
    return ipd.Bunch(chain_groups)

def atoms_to_seq(atoms):
    import biotite.structure as struc
    return ipd.dev.addreduce(struc.to_sequence(atoms)[0])

def frames_by_seqaln_rmsfit(atomslist, **kw):
    frames, rmsds, matches = [np.eye(4)], [0], [[len(atoms_to_seq(atomslist[0]))] * 3]
    for i, atm in enumerate(atomslist[1:]):
        aln = seqalign(atomslist[0], atm)
        s1, s2 = aln.sequences
        match = aln.trace[(aln.trace[:, 0] >= 0) & (aln.trace[:, 1] >= 0)]
        matches.append([len(match), len(s1), len(s2)])
        xyz1 = atomslist[0].coord[match[:, 0]]
        xyz2 = atm.coord[match[:, 1]]
        rms, _, xfit = ipd.homog.hrmsfit(xyz1, xyz2)
        frames.append(xfit), rmsds.append(rms)
    return np.stack(frames), np.array(rms), np.array(matches)

def seqalign(atoms1, atoms2):
    import biotite.sequence.align as align
    seq1 = atoms_to_seq(atoms1)
    seq2 = atoms_to_seq(atoms2)
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    alignments = align.align_optimal(seq1, seq2, matrix)
    return alignments[0]
