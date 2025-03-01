import numpy as np
import ipd

bs = ipd.lazyimport('biotite.structure')

load = ipd.pdb.readatoms
dump = ipd.pdb.dump

def is_atomarray(atoms):
    from biotite.structure import AtomArray
    return isinstance(atoms, AtomArray)

def is_atomarraystack(atoms):
    from biotite.structure import AtomArrayStack
    return isinstance(atoms, AtomArrayStack)

def is_atoms(atoms):
    return is_atomarray(atoms) or is_atomarraystack(atoms)

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

def frames_by_seqaln_rmsfit(atomslist, tol=0.7, **kw):
    tol = ipd.Tolerances(tol)
    frames, rmsds, matches = [np.eye(4)], [0], [1]
    ca = [a[a.atom_name == 'CA'] for a in atomslist]
    for i, ca_i_ in enumerate(ca[1:]):
        _, match, matchfrac = seqalign(ca[0], ca_i_)
        xyz1 = ca[0].coord[match[:, 0]]
        xyz2 = ca_i_.coord[match[:, 1]]
        rms, _, xfit = ipd.homog.hrmsfit(xyz1, xyz2)
        frames.append(xfit), rmsds.append(rms), matches.append(matchfrac)
    frames, rmsds, matches = np.stack(frames), np.array(rmsds), np.array(matches)
    ok = (rmsds < tol.rms_fit) & (matches > tol.seq_match)
    return frames[ok], rmsds[ok], matches[ok]

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

def stub(atoms):
    cen = bs.mass_center(atoms)
    _, sigma, components = np.linalg.svd(atoms.coord[atoms.atom_name == 'CA'] - cen)
    return ipd.homog.hframe(*components.T, cen)
