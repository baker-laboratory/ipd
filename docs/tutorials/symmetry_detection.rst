.. _understanding_sym_detect:

================================
Understanding Symmetry Detection
================================

.. contents:: Table of Contents
   :depth: 3

Overview
---------

The `ipd.sym.sym_detect` module provides tools to automatically identify symmetry in molecular structures and transformation frames. The symmetry is inferred from spatial transformations between subunits (e.g., protein chains), and classified into cyclic, dihedral, helical, and cage symmetries (T, O, I).

This document explains:

- The `detect()` function
- The `SymInfo` result class
- How symmetry elements are extracted from transformation frames
- How equivalent symmetry axes are identified via `unique_symaxes`

Symmetry detection begins with either:

- A **stack of 4×4 transformation matrices** (e.g., from a symmetric assembly)
- A **list of Biotite AtomArray objects** (e.g., protein chains in a complex)

These are passed to `ipd.sym.detect`:

:py:func:`ipd.sym.sym_detect.detect`

The function delegates to:

- :func:`syminfo_from_frames()` — if given transformations
- :func:`syminfo_from_atomslist()` — if given atoms

The output is a `SymInfo` object containing all symmetry classification and geometric information.

.. _printed_syminfo:

Usage Example
--------------

.. doctest::

    >>> import ipd
    >>> chains = ipd.atom.get("2tbv", assembly="largest", het=False, chainlist=True)
    >>> sinfo = ipd.sym.detect(chains)
    >>> sinfo.symid
    'I'
    >>> sinfo.t_number
    3
    >>> sinfo  # doctest: +SKIP
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ SymInfo 2TBV                                                                 ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ │
    │ ┃ symid      ┃ guess      ┃ symcen        ┃ origin                         ┃ │
    │ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩ │
    │ │ I          │ I          │   0.000       │ [[-0.  1.  0.  0.]             │ │
    │ │            │            │   0.000       │  [-1. -0.  0. -0.]             │ │
    │ │            │            │  -0.000       │  [ 0. -0.  1. -0.]             │ │
    │ │            │            │               │  [ 0.  0.  0.  1.]]            │ │
    │ └────────────┴────────────┴───────────────┴────────────────────────────────┘ │
    │ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓ │
    │ ┃ has_translation           ┃ axes concurrent           ┃ axes dists       ┃ │
    │ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩ │
    │ │ False                     │ True                      │ [0. 0. 0.]       │ │
    │ └───────────────────────────┴───────────────────────────┴──────────────────┘ │
    │ ┏━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓ │
    │ ┃ T      ┃ asu rms   ┃ asu seq match     ┃ asu frames    ┃ all frames      ┃ │
    │ ┡━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩ │
    │ │    3   │ [2.061]   │ [0.942]           │ [3, 4, 4]     │ [60, 3, 4, 4]   │ │
    │ └────────┴───────────┴───────────────────┴───────────────┴─────────────────┘ │
    │ ┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓ │
    │ ┃ C  ┃ axis                       ┃ ang       ┃ cen             ┃ hel      ┃ │
    │ ┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩ │
    │ │ 5  │ [-0.     0.526  0.851]     │   1.257   │ [ 0.  0. -0.]   │  -0.000  │ │
    │ │ 2  │ [-0.  0.  1.]              │   3.142   │ [-0.  0. -0.]   │  -0.000  │ │
    │ │ 3  │ [0.357 0.    0.934]        │   2.094   │ [0. 0. 0.]      │   0.000  │ │
    │ └────┴────────────────────────────┴───────────┴─────────────────┴──────────┘ │
    │ ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┓ │
    │ ┃ Geom Tests           ┃ tol          ┃ frac        ┃ total    ┃ passes    ┃ │
    │ ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━┩ │
    │ │ angle                │   0.030      │   0.397     │  174     │   69      │ │
    │ │ helical_shift        │    1         │   1.000     │  177     │  177      │ │
    │ │ isect                │    2         │   1.000     │ 9677     │ 9677      │ │
    │ │ dot_norm             │   0.040      │   0.056     │ 9660     │  540      │ │
    │ │ misc_lineuniq        │    1         │ None        │    0     │    0      │ │
    │ │ nfold                │   0.300      │   0.800     │    5     │    4      │ │
    │ │ seqmatch             │   0.500      │ None        │    0     │    0      │ │
    │ │ matchsize            │   50         │ None        │    0     │    0      │ │
    │ │ rms_fit              │    4         │ None        │    0     │    0      │ │
    │ │ cageang              │   0.050      │   0.250     │    4     │    1      │ │
    │ └──────────────────────┴──────────────┴─────────────┴──────────┴───────────┘ │
    │ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ │
    │ ┃ worst seq match                             ┃ worst rms                  ┃ │
    │ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩ │
    │ │   1.000                                     │   0.000                    │ │
    │ └─────────────────────────────────────────────┴────────────────────────────┘ │
    └──────────────────────────────────────────────────────────────────────────────┘


Symmetry Detection Pipeline
-----------------------------

1. **Input Preparation**
   Frames or AtomArrays are preprocessed. If `order` is not given, chains are split automatically.

2. **Component Extraction**
   Chains are clustered using `find_components_by_seqaln_rmsfit()` to find unique biological units and generate frames between them.

3. **Symmetry Element Extraction**
   Given transformation frames, pairwise transforms are computed. Each transform is decomposed into axis, angle, center, and helical shift:

   .. code-block::

       axis, ang, cen, hel = h.axis_angle_cen_hel(frames)

4. **Axis Deduplication**
   Close or redundant axes are grouped using:

   :py:func:`ipd.homog.hgeom.unique_symaxes`

   This collapses symmetry elements that are equivalent under a tolerance.

5. **Classification**
   Symmetry is labeled based on:

   - Angular relationships between axes
   - Number of folds (e.g., C3 = 3-fold cyclic)
   - Whether symmetry is point, 1D/2D, helical, etc.

Unique Symaxes
^^^^^^^^^^^^^^^

:py:func:`ipd.homog.hgeom.unique_symaxes`

This function filters a list of axis/center/angle tuples into unique representatives across frames. It considers:

- Axis alignment (`dot_norm`)
- Closest approach (`isect`)
- Angular similarity (`angle`)
- Helical shift (`helical_shift`)
- Optional attributes like `nfold`, etc.

It supports two modes:

- `closest`: pick the line closest to a target
- `symaverage`: average close lines across symmetric copies

Symmetry from Atoms
^^^^^^^^^^^^^^^^^^^

If given a list of AtomArrays, `detect()` aligns them using sequence alignment and RMS fitting, then derives symmetry frames. This is managed by:

:py:func:`ipd.sym.sym_detect.syminfo_from_atomslist`

Internally, this calls:

:py:func:`ipd.atom.find_components_by_seqaln_rmsfit`

Symmetry from Frames
^^^^^^^^^^^^^^^^^^^^

If transformation matrices are already available:

:py:func:`ipd.sym.sym_detect.syminfo_from_frames`

Frames are decomposed, and redundant symmetry elements are eliminated via:

:py:func:`ipd.sym.sym_detect.symelems_from_frames`

This also uses `unique_symaxes` to find canonical axes.

The SymInfo Object
^^^^^^^^^^^^^^^^^^

:py:func:`ipd.sym.sym_detect.detect` returns a :py:class:`ipd.sym.sym_detect.SymInfo` instance containing all kinds of info about the symetry detection. :ref:`Printing <printed_syminfo>` it will provide a decent summary.



Advanced Topics
^^^^^^^^^^^^^^^

- **Tolerances**: Control how similar elements must be to merge.
- **Ideal Frames**: Testing with ideal symmetries (C3, D4, T, O, I) helps validate algorithms.
- **Point vs Helical Symmetry**: Automatically inferred from presence of translation and axis concurrency.
- **Pseudo-order**: Estimated from total transforms and asymmetric units.

Related Functions
^^^^^^^^^^^^^^^^^


    - :py:func:`ipd.sym.sym_detect.syminfo_from_atomslist`
    - :py:func:`ipd.sym.sym_detect.syminfo_from_frames`
    - :py:func:`ipd.sym.sym_detect.symelems_from_frames`
    - :py:func:`ipd.sym.sym_detect.syminfo_get_origin`
    - :py:func:`ipd.sym.sym_detect.syminfo_to_str`
    - :py:func:`ipd.sym.sym_detect.check_sym_combinations`

Conclusion
^^^^^^^^^^^

The symmetry detection system in IPD is highly flexible and powerful. It combines biological sequence matching with geometric reasoning to robustly identify symmetrical assemblies. The use of transform decomposition and axis consolidation enables consistent classification across structures with noise or imperfect symmetry.

