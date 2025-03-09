"""
Module for frame searching and alignment of atomic structures.

This module provides tools to calculate frames from atomic coordinates and
perform sequence alignment and RMS fitting on atomic structures. It includes
a `FrameSearchResult` class to store and manipulate search results.

Examples:
    >>> atoms = ipd.atom.testdata('1dxh', assembly='largest', het=False, chainlist=True)
    >>> frameset = ipd.atom.find_frames_by_seqaln_rmsfit(atoms)
    >>> print(frameset)
    FrameSearchResult:
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
        >>> atoms = ipd.atom.testdata('1qys')
        >>> frame = stub(atoms)
        >>> print(frame)
        [[ 0.91638034  0.33895946 -0.21296376  4.20984915]
         [ 0.06904252  0.39019742  0.91813894  5.34146775]
         [ 0.39430978 -0.85606802  0.33416662 11.52761865]
         [ 0.          0.          0.          1.        ]]
    """
    cen = bs.mass_center(atoms)
    _, sigma, components = np.linalg.svd(atoms.coord[atoms.atom_name == 'CA'] - cen)
    return ipd.homog.hframe(*components.T, cen)

def find_frames_by_seqaln_rmsfit(atomslist, tol=0.7, result=None, idx=None, maxsub=60, **kw):
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
        result (FrameSearchResult, optional):
            Existing result object to append to. Defaults to None.
        idx (list[int], optional):
            Indices of atoms to align. Defaults to None.
        maxsub (int, optional):
            Maximum number of subunits to align. Defaults to 60.
        **kw:
            Additional keyword arguments for alignment.

    Returns:
        FrameSearchResult:
            Result object containing aligned frames and statistics.
    """
    if isinstance(atomslist, bs.AtomArray):
        atomslist = ipd.atom.split_chains(atomslist)
    tol = kw['tol'] = ipd.Tolerances(tol)
    if result is None: result = FrameSearchResult(source_=atomslist, tolerances_=tol)
    if len(atomslist) < maxsub:
        atomslist = [ipd.atom.split_chains(casub, minlen=20) for casub in atomslist]
        atomslist = ipd.dev.addreduce(atomslist)
    ca = [a[a.atom_name == 'CA'] for a in atomslist]
    if idx is None: idx = np.arange(len(ca))
    val = ipd.Bunch(frames=[np.eye(4)], rmsd=[0], seqmatch=[1], idx=[0])
    for i, ca_i_ in enumerate(ca[1:], start=1):
        _, match, matchfrac = ipd.atom.seqalign(ca[0], ca_i_)
        xyz1 = ca[0].coord[match[:, 0]]
        xyz2 = ca_i_.coord[match[:, 1]]
        rms, _, xfit = ipd.homog.hrmsfit(xyz1, xyz2)
        val.mapwise.append(xfit, rms, matchfrac, i)
    assert len(val.frames) == len(ca)
    assert len(ca) == len(atomslist)
    val = val.mapwise(np.array)
    ok = (val.rmsd < tol.rms_fit) & (val.seqmatch > tol.seq_match)
    result.add(atoms=atomslist[0], **val.mapwise[ok])
    if all(ok): return result
    unfound = [a for i, a in enumerate(ca) if not ok[i]]
    return find_frames_by_seqaln_rmsfit(unfound, result=result, idx=idx[ok], **kw)

listfield = attrs.field(factory=list)

@ipd.dev.subscriptable_for_attributes
@ipd.dev.element_wise_operations
@attrs.define(slots=False)
class FrameSearchResult:
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
    source_: list['bs.AtomArray'] = None
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

    def __repr__(self):
        """
        Return a string representation of the object.
        """
        with ipd.dev.capture_stdio() as printed:
            with ipd.dev.np_compact(4):
                print('FrameSearchResult:')
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
