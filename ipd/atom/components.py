"""
This module provides core functionality for aligning and processing atomic structures. It defines building blocks used in higher-level constructs such as `Body` and `SymBody`. The module includes routines for aligning atomic structures via sequence alignment and RMSD fitting, storing alignment results in a dedicated container, and processing intermediate alignment components.

:ref:`Determining Biological Architecture <processing_assemblies>`

Key Features:
    - Recursive alignment of atomic structures using sequence similarity and RMSD fitting.
    - Management and storage of alignment results in a structured `Components` container.
    - Utility functions for computing transformation frames based on SVD of C-alpha atoms.
    - Tools for refining and merging alignment results.

Usage Example:
    >>> import ipd
    >>> atoms = ipd.atom.load("1hv4", assembly='largest')
    >>> components = ipd.atom.find_components_by_seqaln_rmsfit(atoms)
    >>> print(components)
    Components:
      atoms: [1084]
      frames: [(4, 4, 4)]
      seqmatch: [array([1.    , 0.6272, 1.    , 0.6272])]
      rmsd: [array([0.    , 1.0535, 0.0203, 1.0551])]
      idx: [array([0, 1, 2, 3])]
      source_: <class 'list'>

Dependencies:
    - numpy
    - biotite (accessed via lazy import)
"""

import numpy as np
import ipd

bs = ipd.lazyimport('biotite.structure')

@ipd.dev.timed
def find_components_by_seqaln_rmsfit(
    atomslist,
    tol=None,
    finalresult=None,
    idx=None,
    maxsub=60,
    minatoms=10,
    prune_res=True,
    protein_only=True,
    **kw,
):
    """
    Align atomic structures using sequence alignment and RMSD fitting.

    This function recursively aligns a list of atomic structures (or chains) based on their
    sequence similarity. It computes transformation frames using RMSD (Root Mean Square Deviation)
    fitting and checks the quality of the alignment using defined tolerance thresholds.
    The aligned frames, along with statistics like RMSD and sequence match fractions, are accumulated
    into a `Components` object.

    Parameters:
        atomslist (list[bs.AtomArray] or bs.AtomArray):
            A list of atomic structures to be aligned or a single structure that will be split into chains.
        tol (float, optional):
            Tolerance value for assessing RMSD and sequence match quality. Defaults to 0.7.
        finalresult (Components, optional):
            An existing Components instance to which new alignment results will be appended.
            If None, a new Components instance is created.
        idx (list[int], optional):
            List of indices corresponding to the atomic structures to align. Defaults to None.
        maxsub (int, optional):
            Maximum number of subunits (chains) to align. Defaults to 60.
        **kw:
            Additional keyword arguments passed to alignment helper functions.

    Returns:
        Components:
            A Components object containing:
                - Aligned atomic structures.
                - 4x4 transformation matrices representing the alignment frames.
                - RMSD values and sequence match fractions.
                - Original source structures and intermediate alignment results.

    Notes:
        The function initially attempts to align the structures based on protein C-alpha atoms.
        If that fails, it falls back to aligning nucleic acid phosphate atoms.
        The process is performed recursively until all components meet the tolerance criteria.
    """
    kw = ipd.sym.symdetect_default_tolerances | kw
    tol = ipd.Tolerances(tol, **kw)
    if isinstance(atomslist, bs.AtomArray):
        atomslist = ipd.kwcall(kw, ipd.atom.split_chains, atomslist)
    if len(atomslist) < maxsub:
        atomslist = [ipd.kwcall(kw, ipd.atom.split_chains, sub) for sub in atomslist]
        atomslist = ipd.dev.addreduce(atomslist)
    atomslist = [a for a in atomslist if len(a)]
    if prune_res: atomslist = ipd.atom.remove_garbage_residues(atomslist)
    if protein_only: atomslist = ipd.atom.remove_nonprotein(atomslist)
    idx = np.arange(len(atomslist))
    finalresult = Components(source_=atomslist, tolerances_=tol)
    return find_components_by_seqaln_rmsfit_recurse(atomslist, tol, finalresult, idx)

@ipd.dev.timed
def find_components_by_seqaln_rmsfit_recurse(atomslist, tol, finalresult, idx):
    alignment = ipd.Bunch(frames=[np.eye(4)], rmsd=[0], seqmatch=[1], idx=[0], match=[True ])
    ca = [a[(a.atom_name == 'CA') & ~a.hetero] for a in atomslist]
    aligned_on_protein = accumulate_seqalign_rmsfit(ca, alignment.mapwise.append)
    if not aligned_on_protein:
        phos = [a[a.atom_name == 'P'] for a in atomslist]
        aligned_on_nucleic = accumulate_seqalign_rmsfit(phos, alignment.mapwise.append)
        if not aligned_on_nucleic:
            return finalresult

    assert aligned_on_protein or aligned_on_nucleic
    assert np.all(alignment.npwise(len) == len(atomslist))
    finalresult.add_intermediate_result(alignment)
    alignment.idx = idx
    alignment = alignment.mapwise(ipd.homog.np_array)

    ok = (alignment.rmsd < tol.rms_fit) & (alignment.seqmatch > tol.seqmatch)
    finalresult.add(atoms=atomslist[0], **alignment.mapwise[ok])
    if all(ok): return finalresult
    unfound = [a for i, a in enumerate(atomslist) if not ok[i]]
    return find_components_by_seqaln_rmsfit_recurse(unfound, tol, finalresult, idx[~ok])

@ipd.subscriptable_for_attributes
@ipd.element_wise_operations
@ipd.mutablestruct
class Components:
    """
    Container for Alignment Components and Results.

    The `Components` class stores the results from aligning atomic structures.
    It holds aligned atomic structures along with associated transformation frames,
    RMSD values, sequence match fractions, and index mappings. In addition, it
    retains the original source structures and any intermediate alignment results.

    Attributes:
        atoms (list[bs.AtomArray]):
            List of aligned atomic structures.
        frames (list[np.ndarray]):
            List of 4x4 transformation matrices representing the alignment frames.
        seqmatch (list[float]):
            List of sequence match fractions for each alignment.
        rmsd (list[float]):
            List of RMSD (Root Mean Square Deviation) values for each alignment.
        idx (list[list[int]]):
            List of index arrays mapping alignment positions.
        source_ (list[bs.AtomArray]):
            Original source atomic structures used in the alignment.
        intermediates_ (list[dict]):
            Intermediate alignment results accumulated during the recursive alignment process.
        tolerances_ (ipd.Tolerances):
            Tolerance parameters used during the alignment process.
    """

    atoms: list['bs.AtomArray'] = ipd.field(list)
    frames: list[np.ndarray] = ipd.field(list)
    seqmatch: list[float] = ipd.field(list)
    rmsd: list[float] = ipd.field(list)
    idx: list[list[int]] = ipd.field(list)
    match: list = ipd.field(list)
    source_: list['bs.AtomArray'] = ipd.field(list)
    intermediates_: list[dict] = ipd.field(list)
    tolerances_: ipd.Tolerances = None

    def add(self, **atom_frame_match_rms_idx):
        """
        Append new alignment results to the Components container.

        This method adds a new set of alignment data—including the transformation frame,
        RMSD value, sequence match fraction, and indices—to the stored results. It also
        reorders the components based on the length of the atomic structures to maintain consistency.

        Parameters:
            **atom_frame_match_rms_idx:
                A dictionary containing keys such as 'atoms', 'frames', 'rmsd', 'seqmatch', and 'idx'
                with corresponding values for the new alignment result.
        """
        self.mapwise.append(atom_frame_match_rms_idx)
        self.idx = [np.array(i, dtype=int) for i in self.idx]
        order = list(reversed(ipd.dev.order(map(len, self.atoms))))
        if len(order) > 1:
            reorder = ipd.dev.reorderer(order)
            reorder(self.atoms, self.frames, self.seqmatch, self.rmsd, self.idx, self.source_)

    def add_intermediate_result(self, intermediate_result):
        """
        Append an intermediate alignment result to the Components container.

        Parameters:
            intermediate_result (Components):
                A Components object representing an intermediate state of the alignment process.
        """
        self.intermediates_.append(intermediate_result)

    def enumerate_intermediates(self):
        """
        Generator that yields intermediate alignment results.

        Yields:
            tuple:
                A tuple in the format (intermediate_index, atoms, frames, rmsd, seqmatch, idx)
                for each intermediate result.
        """
        for i, val in enumerate(self.intermediates_):
            for tup in val.enumerate():
                yield i, *tup

    def print_intermediates(self):
        """
        Print all intermediate alignment results to the console.
        """
        n = len(self.intermediates_[0].frames)
        table: dict[str, ipd.Any] = dict(
            idx=range(n),
            nres=[bs.get_residue_count(atoms) for atoms in self.source_],
        )
        for i, val in enumerate(self.intermediates_):
            frm, rmsd, seqm, idx, match = val.values()
            seq, rms = np.array([''] * n, dtype=object), np.array([''] * n, dtype=object)
            # print(list(map(str, (np.array(seqm) * 100).astype(int))))
            seq[idx] = list(map(str, (np.array(seqm) * 100).astype(int)))
            rms[idx] = list(map(str, np.array(rmsd).round(1)))
            table[f'{i}sm'] = seq
            table[f'{i}rms'] = rms
            # if i > 4: break
        table = {k: list(v) for k, v in table.items()}
        ipd.print_table(table, justify='right', expand=False, collapse_padding=True, show_lines=n < 16)

    def __repr__(self):
        """
        Return a string representation of the Components container.

        The representation includes a summary of the number of atoms per structure,
        the shapes of transformation frames, sequence match fractions, RMSD values, and index arrays.
        It also indicates the type of the stored source atomic structures.

        Returns:
            str:
                A formatted string summarizing the alignment components.
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
        return printed.read().rstrip()

    def remove_small_chains(self, minatom=40, minres=3):
        """
        Remove chains that do not meet minimum size criteria.

        Chains with fewer than `minatom` atoms or fewer than `minres` unique residues are removed
        from the alignment results. This method modifies the Components container in place.

        Parameters:
            minatom (int, optional):
                Minimum number of atoms required for a chain to be retained. Defaults to 40.
            minres (int, optional):
                Minimum number of unique residues required for a chain to be retained. Defaults to 3.
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

def stub(atoms):
    """
    Compute the transformation frame for an atomic structure using SVD on C-alpha atoms.

    This function calculates the center of mass of the atomic structure and performs a Singular Value Decomposition (SVD)
    on the coordinates of the C-alpha atoms. A 4x4 homogeneous transformation matrix is then constructed to represent
    the alignment frame of the structure.

    Parameters:
        atoms (bs.AtomArray):
            An atomic structure containing coordinate data and atom type information.

    Returns:
        np.ndarray:
            A 4x4 homogeneous transformation matrix representing the computed alignment frame.

    Example:
        >>> import ipd
        >>> atoms = ipd.atom.get('1qys')
        >>> frame = stub(atoms)
        >>> print(frame)
        [[ 0.91638  0.33896 -0.21296  4.20985]
         [ 0.06904  0.3902   0.91814  5.34147]
         [ 0.39431 -0.85607  0.33417 11.52762]
         [ 0.       0.       0.       1.     ]]
    """
    cen = bs.mass_center(atoms)
    _, sigma, Components = np.linalg.svd(atoms.coord[atoms.atom_name == 'CA'] - cen)
    return ipd.homog.hframe(*Components.T, cen)

@ipd.dev.timed
def accumulate_seqalign_rmsfit(bb, accumulator, min_align_points=3):
    """
    Perform sequence alignment and RMSD fitting on a list of backbone atoms.

    This helper function attempts to align the first set of backbone atoms in `bb` with each subsequent set.
    It computes the alignment using a sequence alignment algorithm and performs RMSD fitting between the matched atoms.
    The resulting transformation frame, RMSD value, and sequence match fraction are passed to the provided `accumulator`
    callback function.

    Parameters:
        bb (list[bs.AtomArray]):
            List of atomic structure subsets (e.g., C-alpha or phosphate atoms) used for alignment.
        accumulator (callable):
            A function that accepts alignment results (transformation frame, RMSD, sequence match fraction, and index)
            and accumulates them.
        min_align_points (int, optional):
            Minimum number of alignment points required for a valid alignment. Defaults to 3.

    Returns:
        bool:
            True if alignment was successfully performed for at least one pair of structures; otherwise, False.

    Notes:
        - When aligning structures of different types (e.g., protein vs. nucleic acid), a fallback is applied
          if the structures are of equal length.
        - If the alignment does not yield enough matching points, a default high RMSD and zero sequence match are recorded.
    """
    if len(bb) < 2 or len(bb[0]) < min_align_points:
        return False
    is_protein = ipd.atom.is_protein(bb)
    for i, bb_i_ in enumerate(bb[1:], start=1):
        if is_protein[0] == is_protein[i]:
            _, match, matchfrac = ipd.atom.seqalign(bb[0], bb_i_)
        elif len(bb_i_) == len(bb[0]):
            match, matchfrac = np.ones((len(bb[0]), 2), dtype=bool), 1
        else:
            match = None
        if match is None or len(match) < min_align_points:
            accumulator(np.nan / np.zeros((4, 4)), 9e9, 0, i, None)
        else:
            xyz1 = bb[0].coord[match[:, 0]]
            xyz2 = bb_i_.coord[match[:, 1]]
            ipd.dev.global_timer.checkpoint()
            rms, _, xfit = ipd.homog.hrmsfit(xyz1, xyz2)
            ipd.dev.global_timer.checkpoint('hrmsfit')
            accumulator(xfit, rms, matchfrac, i, match)
    return True

def merge_small_components(
    components: Components,
    pickchain: str = 'largest',
    merge_chains: bool = True,
    min_chain_atoms: int = 0,
    **kw,
):
    """
    Process and refine alignment components by optionally merging small chains.

    This function iterates over the alignment results stored in a `Components` container and processes each chain.
    Depending on the provided parameters, it may merge chains that do not meet the minimum atom threshold,
    based on chain selection strategies and the shape compatibility of transformation frames.

    Parameters:
        components (Components):
            A Components object containing the alignment results.
        pickchain (str, optional):
            Strategy for selecting chains. Defaults to 'largest'.
        merge_chains (bool, optional):
            If True, chains that do not meet the `min_chain_atoms` threshold may be merged with adjacent chains.
            Defaults to True.
        min_chain_atoms (int, optional):
            Minimum number of atoms required for a chain to be processed individually.
            Chains with fewer atoms may be merged. Defaults to 0.
        **kw:
            Additional keyword arguments for customizing the processing behavior.

    Returns:
        None:
            The function modifies the `components` object in place.
    """
    for i, atoms, frames in components.enumerate('atoms frames', order=reversed):
        if len(atoms) < min_chain_atoms and i > 0:
            if components.frames[i - 1].shape == frames.shape:
                components.atoms[i - 1] += atoms
                components.atoms.pop(i)
                components.frames.pop(i)
