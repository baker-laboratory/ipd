========================
Body and SymBody Objects
========================

Overview
--------
The ``ipd.atom.body`` module defines the `Body` and `SymBody` classes for representing and manipulating structured collections of atoms. These classes support:

- Spatial transformation using 4×4 homogeneous matrices
- Efficient clash detection and contact analysis
- Subunit assembly through `SymBody`
- Utilities like `.movedby()`, `.hasclash()`, `.contacts()`, and slicing

Body and SymBody objects are lazy, in that they don't change the AtomArray or BVH data structuers when moved. Instead, a body has a ``self.pos`` attribute which is homogeneous transform from its initial position. Repositioned atoms can be produced with the property ``self.positioned_atoms``, which returns ``h.xform(self.pos, self.atoms)``. All functions behave as though the coordinates have actually been transformed, so you can safely think of the atoms represented by a Body or SymBody as actually moving around.

A SymBody holds a Body in self.asu, a position in self.asu, and a set of symmetric frames in self.frames. The position of the asu is changed via updates to self.asu.pos, and the global position of the particle is set by self.pos. The positioned_atoms of a symbody are also obtained via the property ``self.positioned_atoms``. For a SymBody, these positioned atoms are produced by an opration like the following. Note that while these transforms are associative, it it often helpful to read them from right to left... self.asu.pos moves the asu atoms, then all are xformed by self.frames, then the whole particle is moved by self.pos. (Note: the actual implementation goes left to right, which is usually more efficient, as the rightmost object tends to be large)

.. code-block:: python

    h.xform(self.pos, self.frames, self.asu.pos, self.asu.atoms)

The benefit of this somewhat complicated laziness is that both Body and SymBody can be freely copied around with operations like .movedby and .slide_into_contact without trouble. You can make thousands of them, or have symbodies with thousands of frames, without causing any trouble. The hasclash, nchash, contacts and slide operations are also very fast, typically you can do many thousands of these operations per second.

Note: some of the below is AI slop. It is tested for correctness, but may not be the best informative...

Body Examples
=============

Creating a Body
---------------
.. doctest::

    >>> import ipd
    >>> body = ipd.atom.body_from_file("1qys")
    >>> body.natom > 0
    True

Centering a Body
----------------
.. doctest::

    >>> centered = body.centered
    >>> import numpy as np
    >>> np.allclose(centered.com, [0, 0, 0], atol=1)
    True

Transforming a Body
-------------------
.. doctest::

    >>> from ipd.homog import hgeom as h
    >>> moved = body.movedby(h.trans([5, 0, 0]))
    >>> np.allclose(moved.pos[:3, 3], [5, 0, 0], atol=1e-3)
    True

Accessing Positioned Atoms
---------------------------
.. doctest::

    >>> pa = body.positioned_atoms
    >>> hasattr(pa, "coord")
    True

Detecting Clashes
-----------------
.. doctest::

    >>> b1 = body.centered
    >>> b2 = b1.movedby(h.trans([2, 0, 0]))
    >>> b1.hasclash(b2)
    True

Counting Clashes
----------------
.. doctest::

    >>> n = b1.nclash(b2, radius=2)
    >>> isinstance(n, int) and n > 0
    True

Getting Atom Coordinates via Slicing
------------------------------------
.. doctest::

    >>> coords = b1[:5]
    >>> coords.shape[0] <= 5
    True

Sliding Into Contact
--------------------
.. doctest::

    >>> b3 = b1.slide_into_contact(b2, [1,0,0], radius=3)
    >>> b3.hasclash(b2)
    False
    >>> b3 = b3.movedby([3,0,0])
    >>> b3.hasclash(b2)
    False

Analyzing Contacts
------------------
.. doctest::

    >>> contacts = b3.contacts(b2, radius=5)
    >>> contacts.total_contacts > 0
    True
    >>> contacts.nuniq1 > 0
    True

Iterating Over Contacts
-----------------------
.. doctest::

    >>> for isub1, isub2, sub1, sub2, idx1, idx2 in contacts:
    ...     break
    >>> isinstance(idx1, np.ndarray)
    True

SymBody Examples
================

Creating a SymBody
------------------
.. doctest::

    >>> sym = ipd.atom.symbody_from_file("1dxh", components="largest_assembly")
    >>> len(sym.frames) > 1
    True

Centering a SymBody
-------------------
.. doctest::

    >>> sym_centered = sym.centered
    >>> np.allclose(sym_centered.com, [0, 0, 0], atol=1)
    True

Transforming a SymBody
----------------------
.. doctest::

    >>> moved_sym = sym.movedby(h.trans([3, 0, 0]))
    >>> np.allclose(moved_sym.pos[:3, 3], [3, 0, 0], atol=1e-3)
    True

Self Clash Detection
--------------------
.. doctest::

    >>> matrix = sym.hasclash(sym)
    >>> isinstance(matrix, np.ndarray)
    True
    >>> matrix.shape[0] == len(sym)
    True

Counting SymBody Clashes
------------------------
.. doctest::

    >>> moved2 = sym.movedby([4 * sym.rg, 0, 0])
    >>> np.any(sym.hasclash(moved2)) == False
    np.True_
    >>> sym2 = sym.movedto([int(2.3 * sym.rg), 0, 0])
    >>> isinstance(sym.nclash(sym2), np.ndarray)
    True

SymBody Contacts
----------------
.. doctest::

    >>> contact4 = sym.contacts(sym2, radius=4)
    >>> len(contact4) > 0
    True

Sliding a SymBody into Contact
------------------------------
.. doctest::

    >>> slid = sym.slide_into_contact(sym2, along=[1, 0, 0], radius=2)
    >>> np.any(slid.hasclash(sym2))
    np.False_
    >>> np.any(slid.hasclash(sym2, radius=5))
    np.True_

Accessing SymBody Subunit Coordinates
-------------------------------------
.. doctest::

    >>> coords = sym[0][:5]
    >>> coords.shape[0] <= 5
    True

BodyContacts Summary
====================

Inspecting Contact Properties
-----------------------------
.. doctest::

    >>> bc = sym.contacts(sym2, radius=4)
    >>> bc
    SymContacts(ranges: (12, 12, 2) pairs: (1660, 2))
    >>> bc.nuniq  # num unique atoms making contacts
    155
    >>> isinstance(bc.total_contacts, int)
    True
    >>> isinstance(bc.mean_contacts, float)
    True
    >>> bc.min_contacts >= 0
    np.True_

Building a Contact Matrix Stack
-------------------------------
    see :ref:`contact_matrix_overview`
