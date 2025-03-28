.. _processing_assemblies:

===================================
Determining Biological Architecture
===================================

.. contents:: Table of Contents
   :depth: 3

The IPD library provides tools to automatically determine the biological organization
of a molecular assembly. This is done by detecting repeated chains based on both
sequence identity and structural alignment. The centerpiece of this process is the function:

**``find_components_by_seqaln_rmsfit()``**

This function analyzes a list of chains or subunits to determine how many *unique components* exist based on biological similarity, and it determines the homogeneous transforms that relate each similar subunit to a chosen reference. This allows for handling pseudo symmetrical configurations to some degree, depending on sequence matching and rmsd tolerance. In many cases, you can adjust these until you get the correct unique components. For example, a T4 icosahedral cage may initially result in multiple components depending on differences in the asymmetric unit or other factors. By carefully setting tolerances, you shold be able to get a system with one unique component and 60*4 = 240 frames.

Overview
--------

- Uses **sequence alignment** (via Biotite) to find chains with matching sequences
- Uses **least-squares RMSD fitting** to determine the transformation that maps one chain onto another
- Operates recursively: once a matching group is found, unmatched chains are re-evaluated
- Returns a `Components` object containing atoms, frames, RMSDs, sequence matches, and metadata

API: ``find_components_by_seqaln_rmsfit``
-----------------------------------------

.. autofunction:: ipd.atom.find_components_by_seqaln_rmsfit

Recursive Matching Process
--------------------------

1. Input: a list of chains or chain-like AtomArrays
2. Select a representative chain as reference
3. For each other chain:
   - Align sequences
   - If a sufficient sequence match is found, compute RMSD alignment
   - Store the transformation frame, RMSD value, and sequence match score
4. Accept matches where:
   - RMSD < `tol.rms_fit`
   - sequence match > `tol.seqmatch`
5. Remaining unmatched chains are recursively processed as a new group

Sequence Matching
-----------------

Biological similarity is first established by aligning the amino acid or nucleotide sequences
of each chain against the reference chain. The function uses Biotite's
:func:`~biotite.sequence.align.align_optimal` function with a standard substitution matrix.

RMS Fitting
-----------

If sequence alignment is successful, 3D coordinates for the matching residues are extracted,
and a rigid transformation is computed using least-squares fitting (via `hrmsfit`) to minimize RMSD.
Only matches below the specified RMSD threshold are accepted as symmetric subunits.

The result is a frame (4×4 transform matrix) that maps the reference structure to the matched structure.

Example Usage
-------------

.. doctest::

    >>> import ipd, numpy as np
    >>> # Load the largest biological assembly of a PDB file
    >>> chains = ipd.atom.get("1g5q", assembly="largest", het=False, chainlist=True)
    >>> comp = ipd.atom.find_components_by_seqaln_rmsfit(chains)
    >>> isinstance(comp.frames[0], np.ndarray)
    True
    >>> comp.seqmatch[0] > 0.95
    array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True])
    >>> comp.rmsd[0] < 1.0
    array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True])

    >>> # The returned Components object
    >>> print(comp)  # doctest: +ELLIPSIS
    Components:
      atoms: [...]
      frames: [...]
      seqmatch: [...]
      rmsd: [...]
      idx: [...]
      source_: <class 'list'>

Working with the Components Object
----------------------------------

The result from :py:func:`ipd.atom.components.find_components_by_seqaln_rmsfit` is a :py:class:`ipd.atom.components.Components` object. Each field is a list indexed by component:

- ``atoms``: the aligned atom arrays (reference and matched)
- ``frames``: 4x4 homogeneous transforms from reference to target
- ``seqmatch``: sequence match fractions (0 to 1)
- ``rmsd``: root mean square deviations between matched coordinates
- ``idx``: matched indices used during alignment

You can remove short or partial chains with:

.. doctest::

    >>> comp.remove_small_chains(minatom=50)

Stubs and Frame Estimation
---------------------------

If you only have a single chain and wish to extract a reference frame,
you can use the ``stub()`` function to compute a local frame from its CA atoms:

.. autofunction:: ipd.atom.stub

.. doctest::

    >>> atoms = ipd.atom.get("1qys")
    >>> frame = ipd.atom.stub(atoms)
    >>> frame.shape == (4, 4)
    True

Related Utilities
-----------------

The alignment process uses:

- ``atom.seqalign()`` for sequence matching
- ``hrmsfit()`` from `ipd.homog` for structural fitting

If Biotite or your PDB files are unavailable, make sure you’ve downloaded the required files using:

.. code-block:: python

    ipd.pdb.download_test_pdbs(["1g5q", "1dxh", "1qys"])

Conclusion
----------

This system allows you to automatically extract structural symmetry relationships
within molecular assemblies and to build compact representations of those symmetries
for further use in modeling, visualization, or symmetry-aware computations.

``find_components_by_seqaln_rmsfit`` powers the construction of the `SymBody` class
and other higher-level structures in IPD.
