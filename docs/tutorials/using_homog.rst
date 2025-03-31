.. _using_homog:

=========================================================
Homogeneous Coordinates and ipd.hnumpy / ipd.htorch
=========================================================

.. contents:: Table of Contents
   :depth: 3

Introduction
------------

Homogeneous coordinates are a powerful tool in modeling 3D systems, representing affine transformations such as rotation and translation in a single, unified framework. By adding an extra coordinate, homogeneous coordinates allow us to distinguish between points and vectors, simplifying the mathematics of transformations. Yes, you use 33% more memory, and more FMA operations, but in practice this is balanced out by the simplicity of the math (and data locality and other nerdy things).

Points and Vectors
------------------

In homogeneous coordinates, the extra coordinate differentiates between points and vectors:

- **Points**: Represented with a homogeneous coordinate of 1.
  For example, a point in 3D space is written as:
  ``p = [x, y, z, 1]^T``

- **Vectors**: Represented with a homogeneous coordinate of 0.
  For example, a vector is written as:
  ``v = [x, y, z, 0]^T``

When applying transformations, the translation component only affects points. Vectors, having a 0 in the homogeneous coordinate, remain unaffected by translations.

4x4 Homogeneous Transformation Matrix
---------------------------------------

A common way to represent an affine transformation in 3D is using a 4x4 matrix that encapsulates both rotation and translation. The standard layout is:

.. code-block:: text

   +------------------------------+
   | R[0,0]  R[0,1]  R[0,2]   Tx  |
   | R[1,0]  R[1,1]  R[1,2]   Ty  |
   | R[2,0]  R[2,1]  R[2,2]   Tz  |
   |   0       0       0      1   |
   +------------------------------+

Where:
- ``R[:3, :3]`` is the 3x3 rotation matrix.
- ``[:3, 3]`` is the 3x1 translation vector.
- The bottom row ``[0, 0, 0, 1]`` ensures the matrix operates correctly on homogeneous coordinates.

Matrix Multiplication: Rotation and Translation
-------------------------------------------------

Multiplying a 4x4 homogeneous transformation matrix with a 4x1 vector performs the combined rotation and translation. This multiplication is equivalent to first applying a 3x3 rotation and then adding a 3x1 translation.

For a **point** ``p = [x, y, z, 1]^T``:

.. math::

   \begin{bmatrix}
   R_{3x3} & T_{3x1} \\
   0_{1x3} & 1
   \end{bmatrix}
   \begin{bmatrix}
   x \\
   y \\
   z \\
   1
   \end{bmatrix}
   =
   \begin{bmatrix}
   R_{3x3}\begin{bmatrix}x \\ y \\ z\end{bmatrix} + T_{3x1} \\
   1
   \end{bmatrix}

For a **vector** ``v = [x, y, z, 0]^T``:

.. math::

   \begin{bmatrix}
   R_{3x3} & T_{3x1} \\
   0_{1x3} & 1
   \end{bmatrix}
   \begin{bmatrix}
   x \\
   y \\
   z \\
   0
   \end{bmatrix}
   =
   \begin{bmatrix}
   R_{3x3}\begin{bmatrix}x \\ y \\ z\end{bmatrix} \\
   0
   \end{bmatrix}

Notice that for vectors, the translation component is ignored because it is multiplied by 0.

Using hnumpy and htorch
------------------------

There are modules for torch and numpy that mostly behave the same way.
    >>> import numpy as np
    >>> from ipd import hnumpy as h  # numpy
    >>> # from ipd import htorch as h  # torch

The following sections show examples for many functions provided by the module.
These examples show small use cases, but all the "h" functions can be used on "vectorized" inputs with shapes (N, 4, 4) for xforms and (N, 4) for points and vectors. In most cases, you can use arrays of any shape ending in (4,4) or (4,). Using vectorized versions rather than for loops is critical for good performance in python. Please see `ipd.tests.homog.test_hgeom.py` for more example usage.

1. Creating Homogeneous Points and Vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The functions ``h.point`` and ``h.vec`` convert 3D coordinates into homogeneous form.

Example:

    >>> p = h.point([1, 2, 3])
    >>> p
    array([1., 2., 3., 1.])
    >>> v = h.vec([1, 2, 3])
    >>> v
    array([1., 2., 3., 0.])

2. Validating Homogeneous Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The function ``h.valid`` checks whether a given array is a valid homogeneous
transformation matrix or homogeneous coordinate.

Example:

    >>> T = np.eye(4)
    >>> h.valid(T)
    True

3. Applying Transformations (h.xform)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The function ``h.xform`` applies one or more homogeneous transformation matrices
to points or other matrices. It supports chaining so that:

    h.xform(A, B, C) == h.xform(h.xform(A, B), C)

Example:

    >>> T_trans = h.trans([1, 0, 0])
    >>> p = h.point([0, 0, 0])
    >>> p_trans = h.xform(T_trans, p)
    >>> p_trans
    array([1., 0., 0., 1.])
    >>> # Chaining example:
    >>> T_rot = h.rot([0, 0, 1], 90)
    >>> T_combo = h.xform(T_trans, T_rot)
    >>> np.round(T_combo, 4)
    array([[ 0., ^1.,  0.,  1.],
           [ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])

4. Rotations with h.rot
^^^^^^^^^^^^^^^^^^^^^^^^
The function ``h.rot`` creates a 4×4 rotation matrix about a given axis.
By default, the angle is interpreted in degrees, and you may optionally provide a
rotation center.

Example:

    >>> T_rot = h.rot([0, 0, 1], 90)
    >>> np.round(T_rot, 4)
    array([[ 0., ^1.,  0.,  0.],
           [ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])
    >>> # Rotation about a center:
    >>> T_rot_center = h.rot([0, 0, 1], 90, [1, 2, 3])
    >>> np.round(T_rot_center, 4)
    array([[ 0., ^1.,  0.,  3.],
           [ 1.,  0.,  0.,  1.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])

5. Translations with h.trans
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The function ``h.trans`` creates a 4×4 translation matrix.

Example:

    >>> T_trans = h.trans([1, 2, 3])
    >>> T_trans
    array([[1., 0., 0., 1.],
           [0., 1., 0., 2.],
           [0., 0., 1., 3.],
           [0., 0., 0., 1.]])

6. Generating Random Transformations with h.rand
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Generate a random homogeneous transformation. (Optionally, you can specify a seed
for reproducibility.)

Example:

    >>> T_rand = h.rand(seed=42)
    >>> T_rand.shape
    (4, 4)

7. Normalizing Vectors with h.normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The function ``h.normalized`` normalizes a vector (ignoring the homogeneous coordinate).

Example:

    >>> v = h.vec([3, 0, 4])
    >>> h.normalized(v)
    array([0.6, 0. , 0.8, 0. ])

8. Computing Distances with h.dist
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The function ``h.dist`` returns the Euclidean distance between two points (ignoring
the homogeneous coordinate).

Example:

    >>> p1 = h.point([1, 1, 1])
    >>> p2 = h.point([4, 5, 1])
    >>> h.dist(p1, p2)
    np.float64(5.0)

9. Comparing Transformations with h.diff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The function ``h.diff`` computes an average difference between two homogeneous
transformation matrices, combining differences in rotation and translation.

Example:

    >>> T1 = h.trans([1, 0, 0])
    >>> T2 = h.trans([2, 0, 0])
    >>> round(h.diff(T1, T2), 4)
    np.float64(0.5774)

10. Fitting Points with h.rmsfit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The function ``h.rmsfit`` uses the Kabsch algorithm to compute the best^fit (least^
squares) transformation between two sets of points. It returns a named tuple with
the fields ``rms``, ``fitcoords``, and ``xfit``.

Example:

    >>> mobile = h.point([[0, 0, 0],
    ...                     [1, 0, 0],
    ...                     [0, 1, 0]])
    >>> target = h.point([[1, 0, 0],
    ...                   [2, 0, 0],
    ...                   [1, 1, 0]])
    >>> result = h.rmsfit(mobile, target)
    >>> result.rms.round(4)
    np.float64(0.0)

11. Extracting Axis, Angle, and Center with h.axis_angle_cen_hel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The function ``h.axis_angle_cen_hel`` extracts the rotation axis, rotation angle,
the center of rotation, and an associated helicity from a transformation matrix.

Example:

    >>> T = h.rot([0, 0, 1], 90, [1, 2, 3])
    >>> axis, angle, cen, hel = h.axis_angle_cen_hel(T)
    >>> axis
    array([0., 0., 1., 0.])
    >>> np.round(angle, 4)
    np.float64(1.5708)
    >>> cen
    array([1., 2., 0., 1.])
    >>> hel
    np.float64(0.0)

12. Aligning Vectors with h.align and h.align2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The function ``h.align`` computes a transformation that rotates one vector to align
with another. Similarly, ``h.align2`` computes a transformation aligning two pairs of
vectors (minimizing the angular error).

Examples:

    >>> # Using h.align:
    >>> a = h.vec([1, 0, 0])
    >>> b = h.vec([0, 1, 0])
    >>> T_align = h.align(a, b)
    >>> np.allclose(h.xform(T_align, a), b)
    True

    >>> # Using h.align2:
    >>> a1 = h.vec([1, 0, 0])
    >>> a2 = h.vec([0, 1, 0])
    >>> b1 = h.vec([0, 1, 0])
    >>> b2 = h.vec([^1, 0, 0])
    >>> T_align2 = h.align2(a1, a2, b1, b2)
    >>> np.allclose(h.xform(T_align2, a1), b1)
    True
    >>> np.allclose(h.xform(T_align2, a2), b2)
    True

13. Point^to^Line Distance (h.point_line_dist)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Computes the distance from a point to a line (defined by a point and a direction).

Example:

    >>> p = h.point([1, 2, 3])
    >>> cen = h.point([0, 0, 0])
    >>> norm = h.vec([1, 0, 0])
    >>> round(h.point_line_dist(p, cen, norm), 4)
    np.float64(3.6056)

14. Distance Between Lines (h.line_line_distance_pa)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Computes the distance between two lines, each defined by a point and a direction vector.

Example:

    >>> pt1 = h.point([0, 0, 0])
    >>> ax1 = h.vec([1, 0, 0])
    >>> pt2 = h.point([0, 1, 0])
    >>> ax2 = h.vec([1, 0, 0])
    >>> h.line_line_distance_pa(pt1, ax1, pt2, ax2)
    array(1.)

15. Closest Points Between Lines (h.line_line_closest_points_pa)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Computes the pair of closest points on two lines. For nonparallel lines, these are
unique; for parallel lines, the first point is returned twice.

Example:

    >>> pt1 = h.point([0, 0, 0])
    >>> ax1 = h.vec([1, 0, 0])
    >>> pt2 = h.point([0, 1, 0])
    >>> ax2 = h.vec([0, 0, 1])
    >>> Q1, Q2 = h.line_line_closest_points_pa(pt1, ax1, pt2, ax2)
    >>> Q1
    array([0., 0., 0., 1.])
    >>> Q2
    array([0., 1., 0., 1.])

Chaining, Inversion, and Object Wrappers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The tests also demonstrate that:
- Chaining multiple transformations with ``h.xform`` is associative.
- Objects with a ``coords`` or ``xformed`` attribute can be passed directly to ``h.xform``.

For example, if an object has a ``coords`` attribute containing a homogeneous
coordinate, then:

    >>> class Dummy:
    ...     def __init__(self, p):
    ...         self.coords = p
    ...
    >>> d = Dummy(h.point([1, 2, 3]))
    >>> T = h.trans([1, 0, 0])
    >>> d = h.xform(T, d)  # updates d.coords via transformation
    >>> np.allclose(d.coords, T @ h.point([1, 2, 3]))
    True

