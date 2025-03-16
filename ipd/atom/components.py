"""
Module for frame searching and alignment of atomic structures.

This module provides tools to calculate frames from atomic coordinates and
perform sequence alignment and RMS fitting on atomic structures. It includes
a `Components` class to store and manipulate search results.

Examples:
    >>> atoms = ipd.atom.get('1dxh', assembly='largest', het=False, chainlist=True)
    >>> frameset = ipd.atom.find_components_by_seqaln_rmsfit(atoms)
    >>> print(frameset)
    Components:
      atoms: [2669]
      frames: [(12, 4, 4)]
      seqmatch: [array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])]
      rmsd: [array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]
      idx: [array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])]
      source_: <class 'list'>
    >>> atoms, frames, rms, matches = frameset['atoms frames rmsd seqmatch']


Dependencies:
    - attrs
    - numpy
    - biotite
"""

import attrs
import numpy as np
import ipd

bs = ipd.lazyimport('biotite.structure')

def find_components_by_seqaln_rmsfit(
    atomslist,
    tol=0.7,
    finalresult=None,
    idx=None,
    maxsub=60,
    **kw,
):
    """
    Find frames by sequence alignment and RMS fitting.

    This function aligns pairs of atomic structures based on their sequence
    similarity and calculates the RMSD (Root Mean Square Deviation) between
    them. It stores the aligned frames and matching sequences.

    Args:
        atomslist (list[bs.AtomArray]):
            List of atomic structures to align.
        tol (float, optional):
            Tolerance for RMSD and sequence match. Defaults to 0.7.
        finalresult (Components, optional):
            Existing finalresult object to append to. Defaults to None.
        idx (list[int], optional):
            Indices of atoms to align. Defaults to None.
        maxsub (int, optional):
            Maximum number of subunits to align. Defaults to 60.
        **kw:
            Additional keyword arguments for alignment.

    Returns:
        Components:
            Result object containing aligned frames and statistics.
    """
    tol = ipd.Tolerances(tol)
    if not finalresult:
        if isinstance(atomslist, bs.AtomArray):
            atomslist = ipd.kwcall(kw, ipd.atom.split_chains, atomslist)
        if len(atomslist) < maxsub:
            atomslist = [ipd.kwcall(kw, ipd.atom.split_chains, sub) for sub in atomslist]
            atomslist = ipd.dev.addreduce(atomslist)
        atomslist = [a for a in atomslist if len(a)]
        idx = np.arange(len(atomslist))
        finalresult = Components(source_=atomslist, tolerances_=tol)
        # atomslist = ipd.atom.pick_representative_chains(atomslist)
        # atomslist = [atoms[np.isin(atoms.atom_name,['CA', 'P'])] for atoms in atomslist]
    results = ipd.Bunch(frames=[np.eye(4)], rmsd=[0], seqmatch=[1], idx=[0])

    # for i, atoms_i in enumerate(atomslist):
    # no good, fails on various inputsi
    # a, x, m1, m2 = bs.superimpose_homologs(atomslist[0], atoms_i)

    ca = [a[a.atom_name == 'CA'] for a in atomslist]
    aligned_on_protein = accumulate_seqalign_rmsfit(ca, results.mapwise.append)
    if not aligned_on_protein:
        phos = [a[a.atom_name == 'P'] for a in atomslist]
        aligned_on_nucleic = accumulate_seqalign_rmsfit(phos, results.mapwise.append)
    assert np.all(results.npwise(len) == len(atomslist))
    finalresult.add_intermediate_result(results)
    results = results.mapwise(np.array)

    ok = (results.rmsd < tol.rms_fit) & (results.seqmatch > tol.seqmatch)
    # ic(results.rmsd,results.seqmatch)
    finalresult.add(atoms=atomslist[0], **results.mapwise[ok])
    if all(ok): return finalresult
    unfound = [a for i, a in enumerate(atomslist) if not ok[i]]
    # ic(len(atomslist), len(unfound), idx, ok, kw.keys())
    return find_components_by_seqaln_rmsfit(unfound, finalresult=finalresult, idx=idx[~ok], tol=tol, **kw)

listfield = attrs.field(factory=list)

@ipd.dev.subscriptable_for_attributes
@ipd.dev.element_wise_operations
@attrs.define(slots=False)
class Components:
    """
    Result container for frame searching and alignment.

    This class stores the aligned frames, RMSD values, and sequence matches
    resulting from the frame search operation.

    Attributes:
        atoms (list[bs.AtomArray]):
            List of aligned atomic structures.
        frames (list[np.ndarray]):
            List of 4x4 transformation matrices.
        seqmatch (list[float]):
            List of sequence match fractions.
        rmsd (list[float]):
            List of RMSD values.
        idx (list[list[int]]):
            List of aligned indices.
        source_ (list[bs.AtomArray]):
            Source atomic structures.
        tolerances_ (ipd.Tolerances):
            Tolerance parameters for alignment.
    """

    atoms: list['bs.AtomArray'] = listfield
    frames: list[np.ndarray] = listfield
    seqmatch: list[float] = listfield
    rmsd: list[float] = listfield
    idx: list[list[int]] = listfield
    source_: list['bs.AtomArray'] = listfield
    intermediates_: list[dict] = listfield
    tolerances_: ipd.Tolerances = None

    def add(self, **atom_frame_match_rms_idx):
        """
        Add aligned frames and statistics to the result.

        Args:
            **atom_frame_match_rms_idx:
                Dictionary of frames, RMSD values, sequence matches, and indices.
        """
        self.mapwise.append(atom_frame_match_rms_idx)
        self.idx = [np.array(i, dtype=int) for i in self.idx]
        order = list(reversed(ipd.dev.order(map(len, self.atoms))))
        if len(order) > 1:
            reorder = ipd.dev.reorderer(order)
            reorder(self.atoms, self.frames, self.seqmatch, self.rmsd, self.idx, self.source_)

    def add_intermediate_result(self, intermediate_result):
        """
        Add intermediate results to the final result.

        Args:
            intermediate_result (Components):
                Intermediate result object to append to the final result.
        """
        self.intermediates_.append(intermediate_result)

    def __repr__(self):
        """
        Return a string representation of the object.
        """
        with ipd.dev.capture_stdio() as printed:
            with ipd.dev.np_compact(4):
                print('Components:')
                print('  atoms:', [len(atom) for atom in self.atoms])
                print('  frames:', [f.shape for f in self.frames])
                print('  seqmatch:', self.seqmatch)
                print('  rmsd:', self.rmsd)
                print('  idx:', self.idx)
                print('  source_:', type(self.source_))
                # print(self.tolerances_)
        return printed.read().rstrip()

    def remove_small_chains(self, minatom=40, minres=3):
        """
        Remove small chains from the result.

        Args:
            minatom (int): Minimum number of atoms in a chain. Defaults to 40.
            minres (int): Minimum number of residues in a chain. Defaults to 3.
        """
        toremove = [
            i for i, a in enumerate(self.atoms) if len(a) < minatom or len(np.unique(a.res_id)) < minres
        ]
        for i in reversed(toremove):
            del self.atoms[i]
            del self.frames[i]
            del self.seqmatch[i]
            del self.rmsd[i]
            del self.idx[i]
            del self.source_[i]

    def __len__(self):
        return len(self.atoms)

    def __getitem__(self, key: object) -> object:
        'just to make type checker happy'

def stub(atoms):
    """
    Compute the frame based on the mass center and SVD decomposition of CA atoms.

    This function computes the transformation frame of an atomic structure
    by aligning its CA atom coordinates to the center of mass using Singular
    Value Decomposition (SVD).

    Args:
        atoms (bs.AtomArray):
            Atomic structure object containing coordinate and atom name data.

    Returns:
        np.ndarray:
            4x4 homogeneous transformation matrix.

    Example:
        >>> import ipd
        >>> atoms = ipd.atom.get('1qys')
        >>> frame = stub(atoms)
        >>> print(frame)
        [[ 0.91638034  0.33895946 -0.21296376  4.20984915]
         [ 0.06904252  0.39019742  0.91813894  5.34146775]
         [ 0.39430978 -0.85606802  0.33416662 11.52761865]
         [ 0.          0.          0.          1.        ]]
    """
    cen = bs.mass_center(atoms)
    _, sigma, Components = np.linalg.svd(atoms.coord[atoms.atom_name == 'CA'] - cen)
    return ipd.homog.hframe(*Components.T, cen)

def accumulate_seqalign_rmsfit(bb, accumulator, min_align_points=3):
    if len(bb) < 2 or len(bb[0]) < min_align_points:
        return False
    is_protein = ipd.atom.is_protein(bb)
    for i, bb_i_ in enumerate(bb[1:], start=1):
        if is_protein[0] == is_protein[i]:  # dont match protein with nucleic
            _, match, matchfrac = ipd.atom.seqalign(bb[0], bb_i_)
        elif len(bb_i_) == len(bb[0]):
            match, matchfrac = np.ones((len(bb[0]), 2), dtype=bool), 1
        else:
            match = None
        if match is None or len(match) < min_align_points:
            accumulator(np.nan / np.zeros((4, 4)), 9e9, 0, i)
        else:
            xyz1 = bb[0].coord[match[:, 0]]
            xyz2 = bb_i_.coord[match[:, 1]]
            rms, _, xfit = ipd.homog.hrmsfit(xyz1, xyz2)
            accumulator(xfit, rms, matchfrac, i)
    return True

def process_components(
    components: Components,
    pickchain: str = 'largest',
    merge_chains: bool = True,
    min_chain_atoms: int = 0,
    **kw,
):
    for i, atoms, frames in components.enumerate('atoms frames', order=reversed):
        if len(atoms) < min_chain_atoms and i > 0:
            if components.frames[i - 1].shape == frames.shape:
                components.atoms[i - 1] += atoms
                components.atoms.pop(i)
                components.frames.pop(i)
