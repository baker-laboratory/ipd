.. _working_with_atoms:

====================================
Working with .cif files and atoms
====================================

This document explains how to read, manipulate, and analyze atomic structures
using functions from the ``ipd.pdb.readstruct`` and ``ipd.atom.atom_utils`` modules.
These tools primarily operate on :class:`~biotite.structure.AtomArray` objects
from the Biotite library.

.. seealso::
   `Biotite AtomArray Documentation <https://www.biotite-python.org/apidoc/biotite.structure.AtomArray.html>`_

Overview
--------

The IPD library provides utilities for:

- Loading atomic structures from PDB, CIF, or BCIF files
- Filtering atoms based on attributes (e.g., element type, chain ID)
- Splitting and merging atom arrays
- Performing sequence extraction and alignment
- Applying transformations
- Exporting structures to disk

Reading Structures
------------------

.. doctest::

    >>> from ipd import pdb
    >>> atoms = pdb.readatoms("1a2n")  # Load from PDB ID or filename
    >>> atoms.coord.shape[1]
    3

Reading with Transformations (CIF assembly)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doctest::

    >>> atoms_list = pdb.readatoms("1dxh", assembly="largest")
    >>> len(atoms_list) > 1
    True

Filtering and Selecting Atoms
-----------------------------

.. doctest::

    >>> from ipd import atom
    >>> oxy = atom.select(atoms, element='O')
    >>> all(oxy.element == 'O')
    True

Split by Chain
~~~~~~~~~~~~~~

.. doctest::

    >>> chains = atom.chain_dict(atoms)
    >>> isinstance(chains, dict)
    True
    >>> any(len(chain) > 0 for chain in chains.values())
    True

Joining Atom Arrays
~~~~~~~~~~~~~~~~~~~

.. doctest::

    >>> joined = atom.join(list(chains.values()))
    >>> joined.coord.shape[1]
    3

Sequence Extraction
-------------------

.. doctest::

    >>> from ipd import atom
    >>> ca_atoms = atom.select(atoms, caonly=True)
    >>> seqs, *_ = atom.to_seq(ca_atoms)
    >>> len(seqs) >= 1
    True

Atom Type Checks
----------------

.. doctest::

    >>> atom.is_atomarray(atoms)
    True
    >>> atom.is_atoms(atoms)
    True

Structure Classification
------------------------

.. doctest::

    >>> atom.is_protein(ca_atoms)
    np.True_

Splitting Atom Arrays
---------------------

.. doctest::

    >>> splits = atom.split(atoms, bychain=True)
    >>> isinstance(splits, list)
    True

Chain Range Mapping
-------------------

.. doctest::

    >>> cr = atom.chain_ranges(atoms)
    >>> isinstance(cr, dict)
    True
    >>> all(isinstance(rng, list) for rng in cr.values())
    True

Dumping Structures
------------------

.. doctest::

    >>> from tempfile import NamedTemporaryFile
    >>> with NamedTemporaryFile(suffix=".pdb") as tmp:
    ...     pdb.dump(atoms, tmp.name)  # Save AtomArray to PDB
    ...     atoms2 = pdb.readatoms(tmp.name)
    >>> atoms2.coord.shape == atoms.coord.shape
    True

Sequence Alignment
------------------

.. doctest::

    >>> aln, match, score = atom.seqalign(ca_atoms, ca_atoms)
    >>> match.shape[1] == 2
    True
    >>> score > 0.95
    True

Exporting CIF/BCIF
------------------

.. doctest::

    >>> from tempfile import NamedTemporaryFile
    >>> with NamedTemporaryFile(suffix=".cif") as tmp:
    ...     pdb.dumpatoms(atoms, tmp.name)
    ...     atoms2 = pdb.readatoms(tmp.name)
    >>> len(atoms2) == len(atoms)
    True

Notes
-----

- For real PDB/CIF/BCIF data, Biotite must be installed and able to access the internet or your test data.
- The IPD module adds rich metadata and assembly parsing features over the base Biotite readers.

