"""
Module: ipd.homog.hgeom
=======================

This module provides functionality to create and manipulate 4x4 homogeneous
transformation matrices used throughout the IPD library for representing positions,
stubs, and symmetry frames. These transforms are essential when aligning, rotating,
or translating atomic bodies and symmetry bodies.

Key features include:
  - Generation of random homogeneous transformation matrices.
  - Construction of translation and rotation matrices.
  - Composition of multiple transformations.
  - Application of transforms to points, vectors, stubs, and higher-level structures.

Many of these functions are used to set up bodies or to manipulate AtomArrays, as well as
to test for clashes and contacts in a highly efficient manner using the rotationally
invariant bounding volume hierarchy (SphereBVH_double from hgeom). For large
symmetrical structures (e.g., virus capsids), this approach can be hundreds of times faster
than traditional alternatives such as the biotite cell list.

Usage Examples:
    >>> from ipd import hnumpy as h
    >>> import numpy as np
    >>> # Create a random transformation matrix (4x4)
    >>> T = h.rand()
    >>> T.shape
    (4, 4)

    >>> # Create a translation matrix that shifts by [1, 2, 3]
    >>> T_trans = h.trans([1.0, 2.0, 3.0])
    >>> # Create a rotation matrix: 90° about the z-axis at the origin
    >>> T_rot = h.rot([0, 0, 1], 90, [0, 0, 0])

    >>> # Compose the translation and rotation transforms
    >>> T_combined = h.xform(T_trans, T_rot)

    >>> # Apply the composed transformation to a point
    >>> pt = h.point([1, 0, 0])
    >>> pt_transformed = h.xform(T_combined, pt)
    >>> pt_transformed[0] > pt[0]  # Verify the transformation altered the x-coordinate
    np.True_

Additional Examples with Bodies and AtomArrays:
    >>> # Using AtomArrays and Bodies (assume these helper functions exist)
    >>> from ipd import atom, hnumpy as h
    >>> from ipd.atom import body
    >>> a = body . body_from_file('1A0J') . centered
    >>> b = body . body_from_file('1A0J') . centered
    >>> # Apply a random rotation to the Body
    >>> T_random = h.rand()
    >>> b_rotated = h.xform(T_random, b)
    >>> # Demonstrate contact checking (conceptual; actual return may vary)
    >>> contacts = b.contacts(a)
    >>> contacts.nuniq > 1000
    True

.. note::
    For more usage details, please see the tests in ipd/tests/homog/test_hgeom.py.
"""

import collections
import copy
import sys

import numpy as np

import ipd
from ipd.homog.npth_common import *

def as_tensor(array, **kw):
    """
    Convert input to a NumPy tensor.

    Args:
        array (array-like): Input data.
        **kw: Additional keyword arguments for `np.asarray`.

    Returns:
        np.ndarray: Converted array.

    Example:
        >>> import numpy as np
        >>> data = [[1, 2], [3, 4]]
        >>> tensor = as_tensor(data)
        >>> print(tensor)
        [[1 2]
         [3 4]]

    """
    if hasattr(array, 'data'): array = array.data
    return np.asarray(array, **kw)

as_tensors = ipd.dev.iterize_on_first_param(basetype=np.ndarray)(as_tensor)

def hconstruct(rot, trans=None):
    """
    Construct a homogeneous transformation matrix from rotation and translation.

    Args:
        rot (np.ndarray): Rotation matrix of shape (..., 3, 3).
        trans (np.ndarray, optional): Translation vector of shape (..., 3). Defaults to None.

    Returns:
        np.ndarray: Homogeneous transformation matrix of shape (..., 4, 4).

    Example:
        >>> rot = np.eye(3)
        >>> trans = np.array([1, 2, 3])
        >>> hmat = hconstruct(rot, trans)
        >>> print(hmat)
        [[1. 0. 0. 1.]
         [0. 1. 0. 2.]
         [0. 0. 1. 3.]
         [0. 0. 0. 1.]]
    """
    x = np.zeros((rot.shape[:-2] + (4, 4)))
    x[..., :3, :3] = rot[..., :3, :3]
    if trans is not None:
        x[..., :3, 3] = trans[..., :3]
    x[..., 3, 3] = 1
    return x

def isarray(x):
    """
    Check if input is a NumPy array or a PyTorch tensor.

    Args:
        x (any): Input object.

    Returns:
        bool: True if x is a NumPy array or PyTorch tensor, False otherwise.

    Example:
        >>> isarray(np.array([1, 2, 3]))
        True
    """
    if isinstance(x, np.ndarray):
        return True
    if 'torch' in sys.modules:
        import torch

        if isinstance(x, torch.Tensor):
            return True
    return False

def to_xyz(x):
    """
    Convert input to a 3D coordinate vector.

    Args:
        x (float or sequence of floats): Input value(s).

    Returns:
        np.ndarray: 3D coordinate vector.

    Example:
        >>> to_xyz(1)
        array([1., 1., 1.])
    """
    if isinstance(x, (int, float)):
        x = [x] * 3
    x = np.array(x, dtype=np.float64)
    return x

def hvalid(stuff, is_points=None, strict=False, **kw):
    """
    Validate a homogeneous transformation matrix or a set of coordinates.

    Args:
        stuff (np.ndarray): Input data.
        is_points (bool, optional): Whether input represents points. Defaults to None.
        strict (bool, optional): If True, enforce stricter validation. Defaults to False.
        **kw: Additional arguments for validation.

    Returns:
        bool: True if valid, False otherwise.

    Example:
        >>> hmat = np.eye(4)
        >>> hvalid(hmat)
        True
    """
    if stuff.shape[-2:] == (4, 4) and not is_points:
        return hvalid44(stuff, **kw)
    if stuff.shape[-2:] == (4, 2) and not is_points:
        return is_valid_rays(stuff)
    elif stuff.shape[-1] == 4 and strict:
        return np.allclose(stuff[..., 3], 0) or np.allclose(stuff[..., 3], 1)
    elif stuff.shape[-1] == 4:
        return np.all(np.logical_or(np.isclose(stuff[..., 3], 0), np.isclose(stuff[..., 3], 1)))
    elif stuff.shape[-1] == 3:
        return True
    return False

def hvalid_norm(x):
    normok = np.allclose(1, np.linalg.norm(x[..., :3, :3], axis=-1))
    normok &= np.allclose(1, np.linalg.norm(x[..., :3, :3], axis=-2))
    return np.all(normok)

def hvalid44(x, improper_ok=False, **kw):
    """
    Validate a 4x4 homogeneous transformation matrix.

    Args:
        x (np.ndarray): Input matrix.
        improper_ok (bool, optional): Whether to allow improper rotations. Defaults to False.
        **kw: Additional arguments for validation.

    Returns:
        bool: True if valid, False otherwise.

    Example:
        >>> hmat = np.eye(4)
        >>> hvalid44(hmat)
        True
    """
    if x.shape[-2:] != (4, 4):
        return False
    det = np.linalg.det(x[..., :3, :3])
    if improper_ok:
        det = np.abs(det)
    detok = np.allclose(det, 1.0, atol=1e-4)

    return all([np.allclose(x[..., 3, 3], 1), np.allclose(x[..., 3, :3], 0), detok])

def hscaled(scale, stuff, is_points=None):
    """
    Scale a homogeneous transformation matrix or a set of coordinates.

    Args:
        scale (float): Scale factor.
        stuff (np.ndarray): Input matrix or coordinates.
        is_points (bool, optional): Whether input represents points. Defaults to None.

    Returns:
        np.ndarray: Scaled matrix or coordinates.

    Example:
        >>> hmat = htrans([3, 2, 1])
        >>> scaled = hscaled(2.0, hmat)
        >>> print(scaled)
        [[1. 0. 0. 6.]
         [0. 1. 0. 4.]
         [0. 0. 1. 2.]
         [0. 0. 0. 1.]]
    """
    stuff = stuff.copy()
    if hvalid44(stuff):
        stuff[..., :3, 3] *= scale
    else:
        stuff[..., :3] *= scale
    return stuff

def hdist(x, y):
    """
    Compute the Euclidean distance between two homogeneous transformation matrices.

    Args:
        x(np . ndarray): First  object .
        y(np . ndarray): Second object .

    Returns:
        float: Euclidean distance.

    Example:
        >>> x = hpoint([1, 1, 3])
        >>> y = hpoint([2, 1, 3])
        >>> hdist(x, y)
        np.float64(1.0)

    """
    return hnorm(x - y)

def hdist2(x, y):
    return hnorm2(x - y)

def hdiff(x, y, lever=10.0):
    """
    Compute the average difference between two homogeneous transformation matrices.

    This function calculates the difference between two transformation matrices,
    considering both rotational and translational components.

    Args:
        x(np . ndarray): First  transformation matrix of shape(..., 4, 4) .
        y(np . ndarray): Second transformation matrix of shape(..., 4, 4) .
        lever (float, optional): Scaling factor for rotational difference. Defaults to 10.0.

    Returns:
        float: Average difference between matrices.

    Example:
        >>> x = hrot([1, 0, 0], [1  , 2  ])
        >>> y = hrot([1, 0, 0], [1.1, 2.1])
        >>> hdiff(x, y)
        array([[0.66639, 6.96916],
               [5.79954, 0.66639]])
    """

    shape1 = x . shape [: - 2]
    shape2 = y . shape [: - 2]
    a = x.reshape(shape1 + (1,) * len(shape1) + (4, 4))
    b = y.reshape((1,) * len(shape2) + shape2 + (4, 4))

    axyz = a[..., :3, :3] * lever + a[..., :3, 3, None]
    bxyz = b[..., :3, :3] * lever + b[..., :3, 3, None]

    diff = np.linalg.norm(axyz - bxyz, axis=-1)
    diff = np.mean(diff, axis=-1)

    return diff

def hxformx(*x, **kw):
    """
    Apply a homogeneous transformation to a matrix.

    Args:
        x (np.ndarray): Transformation matrix of shape (..., 4, 4).
        stuff (np.ndarray): Input matrix to be transformed.
        **kw: Additional arguments for transformation.

    Returns:
        np.ndarray: Transformed matrix.

    Example:
        >>> x = hrot([1, 0, 0], 90)
        >>> x2 = htrans([1, 0, 0])
        >>> hxformx(x, x2).round(4)
        array([[ 1.,  0.,  0.,  1.],
               [ 0.,  0., -1.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  1.]])
    """
    *x, stuff = x
    stuff = as_tensor(stuff)
    assert np.allclose(stuff[..., 3, :], [0, 0, 0, 1])
    result = hxform(*x, stuff, is_points=False, **kw)
    assert np.allclose(stuff[..., 3, :], [0, 0, 0, 1])
    return result

def hxformpts(*x, **kw):
    """
    Apply a homogeneous transformation to points.

    Args:
        x (np.ndarray): Transformation matrix of shape (..., 4, 4).
        stuff (np.ndarray): Points to be transformed.
        **kw: Additional arguments for transformation.

    Returns:
        np.ndarray: Transformed points.

    Example:
        >>> x = np.eye(4)
        >>> points = np.array([[1, 2, 3]])
        >>> hpoints = hpoint(points)
        >>> transformed = hxformpts(x, hpoints)
        >>> print(transformed)
        [[1. 2. 3. 1.]]
    """
    *x, stuff = x
    stuff, hdim = hpoint(stuff), stuff.shape[-1]
    assert np.allclose(stuff[..., 3], 1)
    result = hxform(*x, stuff, is_points=True, **kw)
    assert np.allclose(result[..., 3], 1)
    return result[..., :hdim]

def hxformvec(*x, **kw):
    """
    Apply a homogeneous transformation to vectors.

    Args:
        x (np.ndarray): Transformation matrix of shape (..., 4, 4).
        stuff (np.ndarray): Vectors to be transformed.
        **kw: Additional arguments for transformation.

    Returns:
        np.ndarray: Transformed vectors.

    Example:
        >>> x = hrot([0, 1, 0], 90)
        >>> vec = np.array([1, 0, 0])
        >>> hxformvec(x, vec).round(4)
        array([ 0.,  0., -1.])
    """
    *x, stuff = x
    stuff, hdim = hvec(stuff), stuff.shape[-1]
    assert np.allclose(stuff[..., 3], 0)
    result = hxform(*x, stuff, is_points=True, **kw)
    assert np.allclose(result[..., 3], 0)
    return result[..., :hdim]

def invxform(x, stuff, **kw):
    """
    Apply the inverse of a homogeneous transformation.

    Args:
        x (np.ndarray): Transformation matrix of shape (..., 4, 4).
        stuff (np.ndarray): Input matrix to be transformed.
        **kw: Additional arguments for transformation.

    Returns:
        np.ndarray: Transformed matrix using the inverse of `x`.

    Example:
        >>> x = htrans([1, 0, 0])
        >>> vec = np . array([[1, 2, 3, 0]])
        >>> pt  = np . array([[1, 2, 3, 1]])
        >>> newpt  = invxform(x, pt )
        >>> newvec = invxform(x, vec)
        >>> print(newpt, newvec)
        [[0. 2. 3. 1.]] [[1. 2. 3. 0.]]
    """
    return hxform(hinv(x), stuff, **kw)

def invxformpts(x, stuff, **kw):
    """
    Apply the inverse of a homogeneous transformation to points.

    Args:
        x (np.ndarray): Transformation matrix of shape (..., 4, 4).
        stuff (np.ndarray): Points to be transformed.
        **kw: Additional arguments for transformation.

    Returns:
        np.ndarray: Transformed points using the inverse of `x`.

    Example:
        >>> x = htrans([1, 0, 0])
        >>> points = np.array([[0, 0, 0]])
        >>> hpoints = hpoint(points)
        >>> transformed = invxformpts(x, hpoints)
        >>> print(transformed)
        [[-1.  0.  0.  1.]]
    """
    return hxformpts(hinv(x), stuff, **kw)

def invxformvec(x, stuff, **kw):
    """
    Apply the inverse of a homogeneous transformation to vectors.

    Args:
        x (np.ndarray): Transformation matrix of shape (..., 4, 4).
        stuff (np.ndarray): Vectors to be transformed.
        **kw: Additional arguments for transformation.

    Returns:
        np.ndarray: Transformed vectors using the inverse of `x`.

    Example:
        >>> x = hrot([1, 1, 0], 180)
        >>> vec = np.array([1, 0, 0])
        >>> invxformvec(x, vec).round(4)
        array([0., 1., 0.])
    """
    return hxformvec(hinv(x), stuff, **kw)

def product(*factors, **kw):
    """
    Compute the product of multiple homogeneous transformations.

    Args:
        *factors: Sequence of transformation matrices.
        **kw: Additional arguments for transformation.

    Returns:
        np.ndarray: Resulting transformation matrix.

    Example:
        >>> x = np . eye(4)
        >>> y = np . eye(4)
        >>> product(x, y)
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])
    """
    if len(factors) == 0: return np.eye(4)
    if len(factors) == 1: return factors[0]
    x, *xforms, stuff = factors
    for x1 in xforms:
        x = hxform(x, x1)
    return hxform(x.astype(stuff.dtype), stuff, **kw)

def hxform(*x, homogout='auto', **kw):
    if len(x) > 1:
        *x, stuff = x
        x = product(*x, **kw | {'is_points': False})
    else:
        raise TypeError('hxform missing required positional arguments hxform(x, stuff, ...)')
    if isinstance(stuff, list) and len(stuff) and not isinstance(stuff[0], (int, float, list, tuple)):
        return [hxform(x, v) for v in stuff]
    if isinstance(stuff, dict) and len(stuff) and not isinstance(stuff[0], (int, float, list, tuple)):
        return {k: hxform(x, v) for k, v in stuff.items()}
    if hasattr(stuff, 'xformed'): return stuff . xformed(x)
    if hasattr(stuff, 'movedby'): return stuff . movedby(x)
    orig   = None
    coords = None
    if hasattr(stuff, "coords"): coords = 'coords'
    elif hasattr(stuff, "coord"): coords = 'coord'
    if coords:
        isxarray = False
        if 'xarray' in sys.modules:
            import xarray

            # coords is perhaps poor choice of convention
            # xarray.DataArry has coords member already...
            # print('WARNING Deprivation of .coords convention in favor of .xformed method')
            isxarray = isinstance(stuff, xarray.DataArray)
        if not isxarray:
            if hasattr(stuff, "copy"): orig = stuff.copy()
            else: orig = copy.copy(stuff)
            stuff = getattr(stuff, coords)
            assert x.ndim in (2, 3)

    stuff, origstuff = np.asarray(stuff), stuff
    if stuff.dtype not in (np.float64, np.float32):
        stuff = stuff.astype(np.float64)
        # raise ValueError(f'unknown dtype {stuff.dtype} for type {type(origstuff)}')
    x = np.asarray(x).astype(stuff.dtype)

    nothomog = stuff.shape[-1] == 3
    if stuff.shape[-1] == 3:
        stuff = hpoint(stuff)

    # is this necessary? change ray convention?
    isray = stuff.shape[-1] == 2
    if isray:
        stuff = stuff.swapaxes(-1, -2)

    result = _hxform_impl(x, stuff, **kw)

    if isray:
        result = result.swapaxes(-1, -2)

    if homogout is False or homogout == 'auto' and nothomog:
        result = result[..., :3]

    if result.shape[-1] == 4 and not hvalid(result, **kw):
        ipd . icv(x     . shape)
        ipd . icv(stuff . shape)
        # ipd.icv(result)
        for x in result:
            if not hvalid(x, **kw):
                ipd.icv(x, np.linalg.det(x))
                assert 0, 'invalid transform'
        # this is a bad copout.. should make this check handle nans correctly
        if not stuff.shape[-2:] == (4, 1):
            raise ValueError(f'malformed homogeneous coords with shape {stuff.shape}, '
                             'if passing points, try is_points=True')

    if orig:
        # ipd.icv(result.shape)
        if result.ndim > 2:
            r = list()
            for x in result:
                if hasattr(orig, 'copy'):
                    o = orig.copy()
                else:
                    o = copy.copy(orig)
                setattr(o, coords, x)
                r.append(o)
            result = r
        else:
            setattr(orig, coords, result)
            result = orig

    assert result is not None

    return result

def _hxform_impl(
    x,
    stuff,
    *,
    outerprod='auto',
    flat=False,
    is_points='auto',
    improper_ok=False,
    uppertri=None,
    dtype=None,
):
    if is_points == 'auto':
        is_points = not hvalid44(stuff, improper_ok=improper_ok)
        if is_points:
            if stuff.shape[-1] != 4 and stuff.shape[-2:] == (4, 1):
                raise ValueError(f'hxform cant understand shape {stuff.shape}')

    if not is_points:
        if outerprod == 'auto':
            outerprod = x.shape[:-2] != stuff.shape[:-2]
        if outerprod:
            shape1 = x     . shape [: - 2]
            shape2 = stuff . shape [: - 2]
            a = x.reshape(shape1 + (1,) * len(shape2) + (4, 4))
            b = stuff.reshape((1,) * len(shape1) + shape2 + (4, 4))
            result = a @ b
            if flat:
                result = result.reshape(-1, 4, 4)
        else:
            result = x @ stuff
        if flat:
            result = result.reshape(-1, 4, 4)
    else:
        if outerprod == 'auto':
            outerprod = x.shape[:-2] != stuff.shape[:-1]

        if stuff.shape[-1] != 1:
            stuff = stuff[..., None]
        if outerprod:
            shape1 = x     . shape [: - 2]
            shape2 = stuff . shape [: - 2]
            # ipd.icv(x.shape, stuff.shape, shape1, shape2)
            a = x.reshape(shape1 + (1,) * len(shape2) + (4, 4))

            b = stuff.reshape((1,) * len(shape1) + shape2 + (4, 1))
            result = a @ b
        else:
            # try to match first N dimensions, outer prod the rest
            shape1 = x     . shape [: - 2]
            shape2 = stuff . shape [: - 2]
            sameshape = tuple()
            for i, (s1, s2) in enumerate(zip(shape1, shape2)):
                # ipd.icv(s1, s2)
                if s1 == s2:
                    shape1 = shape1 [1:]
                    shape2 = shape2 [1:]
                    sameshape = sameshape + (s1,)
                else:
                    break
            newshape1 = sameshape + shape1 + (1,) * len(shape2) + (4, 4)
            newshape2 = sameshape + (1,) * len(shape1) + shape2 + (4, 1)
            # ipd.icv(shape1, shape2, newshape1, newshape2)
            a = x     . reshape(newshape1)
            b = stuff . reshape(newshape2)
            result = a @ b

        result = result.squeeze(axis=-1)

        if flat:
            result = result.reshape(-1, 4)

    if uppertri is not None:
        assert result.ndim > 2
        assert result.shape[0] == result.shape[1]
        result = np.concatenate([x[i + uppertri :] for i, x in enumerate(result[:-uppertri])])

    return result

def is_valid_quat_rot(quat):
    assert quat.shape[-1] == 4
    return np.isclose(1, np.linalg.norm(quat, axis=-1))

def quat_to_upper_half(quat):
    ineg0 = quat[..., 0] < 0
    ineg1 = (quat[..., 0] == 0) * (quat[..., 1] < 0)
    ineg2 = (quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] < 0)
    ineg3 = (quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] == 0) * (quat[..., 3] < 0)
    # ipd.icv(ineg0.shape)
    # ipd.icv(ineg1.shape)
    # ipd.icv(ineg2.shape)
    # ipd.icv(ineg3.shape)
    ineg = ineg0 + ineg1 + ineg2 + ineg3
    quat = quat.copy()
    # TODO: is this right?
    quat[ineg] = -quat[ineg]
    return quat

@ipd.dev.preserve_random_state
def rand_quat(shape=()):
    if isinstance(shape, int):
        shape = (shape,)
    q = np.random.randn(*shape, 4)
    q /= np.linalg.norm(q, axis=-1)[..., np.newaxis]
    return quat_to_upper_half(q)

def rot_to_quat(xform):
    x = np.asarray(xform)
    t0, t1, t2 = x[..., 0, 0], x[..., 1, 1], x[..., 2, 2]
    tr = t0 + t1 + t2
    quat = np.empty(x.shape[:-2] + (4,))

    case0 = tr > 0
    S0 = np.sqrt(tr[case0] + 1) * 2
    quat[case0, 0] = 0.25 * S0
    quat[case0, 1] = (x[case0, 2, 1] - x[case0, 1, 2]) / S0
    quat[case0, 2] = (x[case0, 0, 2] - x[case0, 2, 0]) / S0
    quat[case0, 3] = (x[case0, 1, 0] - x[case0, 0, 1]) / S0

    case1 = ~case0 * (t0 >= t1) * (t0 >= t2)
    S1 = np.sqrt(1.0 + x[case1, 0, 0] - x[case1, 1, 1] - x[case1, 2, 2]) * 2
    quat[case1, 0] = (x[case1, 2, 1] - x[case1, 1, 2]) / S1
    quat[case1, 1] = 0.25 * S1
    quat[case1, 2] = (x[case1, 0, 1] + x[case1, 1, 0]) / S1
    quat[case1, 3] = (x[case1, 0, 2] + x[case1, 2, 0]) / S1

    case2 = ~case0 * (t1 > t0) * (t1 >= t2)
    S2 = np.sqrt(1.0 + x[case2, 1, 1] - x[case2, 0, 0] - x[case2, 2, 2]) * 2
    quat[case2, 0] = (x[case2, 0, 2] - x[case2, 2, 0]) / S2
    quat[case2, 1] = (x[case2, 0, 1] + x[case2, 1, 0]) / S2
    quat[case2, 2] = 0.25 * S2
    quat[case2, 3] = (x[case2, 1, 2] + x[case2, 2, 1]) / S2

    case3 = ~case0 * (t2 > t0) * (t2 > t1)
    S3 = np.sqrt(1.0 + x[case3, 2, 2] - x[case3, 0, 0] - x[case3, 1, 1]) * 2
    quat[case3, 0] = (x[case3, 1, 0] - x[case3, 0, 1]) / S3
    quat [case3, 1] = (x [case3, 0, 2] + x [case3, 2, 0]) / S3
    quat [case3, 2] = (x [case3, 1, 2] + x [case3, 2, 1]) / S3
    quat[case3, 3] = 0.25 * S3

    assert np.sum(case0) + np.sum(case1) + np.sum(case2) + np.sum(case3) == np.prod(xform.shape[:-2])

    return quat_to_upper_half(quat)

xform_to_quat = rot_to_quat

def quat_to_rot(quat, dtype='f8', shape=(3, 3)):
    quat = np.asarray(quat)
    assert quat.shape[-1] == 4
    qr = quat [..., 0]
    qi = quat [..., 1]
    qj = quat [..., 2]
    qk = quat [..., 3]
    outshape = quat.shape[:-1]
    rot = np.zeros(outshape + shape, dtype=dtype)
    rot[..., 0, 0] = 1 - 2 * (qj**2 + qk**2)
    rot[..., 0, 1] = 2 * (qi * qj - qk * qr)
    rot [..., 0, 2] = 2 * (qi * qk + qj * qr)
    rot [..., 1, 0] = 2 * (qi * qj + qk * qr)
    rot[..., 1, 1] = 1 - 2 * (qi**2 + qk**2)
    rot [..., 1, 2] = 2 * (qj * qk - qi * qr)
    rot [..., 2, 0] = 2 * (qi * qk - qj * qr)
    rot[..., 2, 1] = 2 * (qj * qk + qi * qr)
    rot[..., 2, 2] = 1 - 2 * (qi**2 + qj**2)
    return rot

def quat_to_xform(quat, dtype='f8'):
    r = quat_to_rot(quat, dtype, shape=(4, 4))
    r[..., 3, 3] = 1
    return r

def quat_multiply(q, r):
    q, r = np.broadcast_arrays(q, r)
    q0, q1, q2, q3 = np . moveaxis(q, - 1, 0)
    r0, r1, r2, r3 = np . moveaxis(r, - 1, 0)
    assert np.all(q1 == q[..., 1])
    t = np.empty_like(q)
    t[..., 0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
    t[..., 1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
    t[..., 2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
    t[..., 3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0
    return t

def guess_is_degrees(angle):
    return np.max(np.abs(angle)) > 2 * np.pi

def is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True

def fast_axis_of(xforms):
    if xforms.shape[-2:] == (4, 4):
        return np.stack(
            (
                xforms [..., 2, 1] - xforms [..., 1, 2],
                xforms [..., 0, 2] - xforms [..., 2, 0],
                xforms [..., 1, 0] - xforms [..., 0, 1],
                np.zeros(xforms.shape[:-2]),
            ),
            axis=-1,
        )
    if xforms.shape[-2:] == (3, 3):
        return np.stack(
            (
                xforms [..., 2, 1] - xforms [..., 1, 2],
                xforms [..., 0, 2] - xforms [..., 2, 0],
                xforms [..., 1, 0] - xforms [..., 0, 1],
            ),
            axis=-1,
        )
    else:
        raise ValueError('wrong shape for xform/rotation matrix: ' + str(xforms.shape))

def axis_of(xforms, tol=1e-7, debug=False):
    xdim = xforms.shape[-1]
    origshape = xforms.shape
    if xforms.ndim != 3:
        xforms = xforms.reshape(-1, xdim, xdim)

    axs = fast_axis_of(xforms)
    norm = np.linalg.norm(axs, axis=-1)
    is180 = norm < tol
    axs[~is180] = axs[~is180] / norm[~is180].reshape(-1, 1)
    if np.sum(is180) > 0:
        x180 = xforms[is180]
        is_ident = np.all(np.isclose(np.eye(3), x180[:, :3, :3], atol=tol), axis=(-2, -1))
        axs[np.where(is180)[0][is_ident]] = [1, 0, 0, 0]
        is180[np.where(is180)[0][is_ident]] = False
        x180 = x180[~is_ident]

        eig = np.linalg.eig(x180[..., :3, :3])
        eigval, eigvec = np.real(eig[0]), np.real(eig[1])
        eigval_is_1 = np.abs(eigval - 1) < 1e-6
        ixform, ieigval = np.where(eigval_is_1)

        # ipd.icv(ixform)
        # ipd.icv(ieigval)
        try:
            axs[is180, :3] = eigvec[ixform, :, ieigval]
        except Exception as e:
            # ipd.icv(is_ident)
            # ipd.icv(is180)
            # ipd.icv(x180.shape)
            # for a, b, c in zip(eigval, eigvec, eigval_is_1):
            # ipd.icv(a)
            # ipd.icv(b)
            # ipd.icv(c)
            # ipd.icv()
            raise e
        # assert 0

        if debug:
            n_unit_eigval_1 = np . sum(np . abs(eigval-1) < tol, axis = - 1) == 1
            n_unit_eigval_3 = np . sum(np . abs(eigval-1) < tol, axis = - 1) == 3
            np.all(np.logical_or(n_unit_eigval_1, n_unit_eigval_3))
            # assert np.allclose(np.all(np.sum(np.abs(eigval - 1) < tol, axis=-1) == 1)

    return axs.reshape(origshape[:-1])

def is_homog_xform(xforms):
    return (
        (xforms.shape[-2:] == (4, 4))
        and (np.allclose(1, np.linalg.det(xforms[..., :3, :3])))
        and (np.allclose(xforms[..., 3, :], [0, 0, 0, 1]))
    )

def hinv(xforms):
    """
    Compute the inverse of a homogeneous transformation matrix.

    Args:
        xforms (np.ndarray): Transformation matrix of shape (..., 4, 4).

    Returns:
        np.ndarray: Inverse of the input transformation matrix.

    Example:
        >>> x = ipd.hnumpy.trans([1, 2, 3])
        >>> inv = hinv(x)
        >>> print(inv)
        [[ 1.  0.  0. -1.]
         [ 0.  1.  0. -2.]
         [ 0.  0.  1. -3.]
         [ 0.  0.  0.  1.]]
    """
    return np.linalg.inv(xforms)

def hunique(xforms):
    """
    Check if transformation matrices are unique.

    Args:
        xforms (np.ndarray): Array of transformation matrices.

    Returns:
        bool: True if matrices are unique, False otherwise.

    Example:
        >>> xforms = np.array([np.eye(4), np.eye(4)])
        >>> hunique(xforms)
        np.False_
    """
    if len(xforms) == 0:
        return True
    diff = hdiff(xforms, xforms)
    np.fill_diagonal(diff, 9e9)
    # ipd.icv(diff.shape, np.min(diff))
    return 0.0001 < np.min(diff)

def axis_angle_of(xforms, debug=False):
    axis = axis_of (xforms, debug = debug)
    angl = angle_of(xforms, debug = debug)
    if angl.ndim == 0:
        if angl < 0: axis, angl = axis * -1, angl * -1
    else:
        axis [angl < 0] *= - 1
        angl [angl < 0] *= - 1
    return axis, angl

def axis_angle_hel_of(xforms):
    axis, angle = axis_angle_of(xforms)
    hel = hdot(axis, trans_of(xforms))
    return axis, angle, hel

def axis_angle_cen_hel_of(xforms):
    axis, angle, cen = axis_ang_cen_of(xforms)
    hel = hdot(axis, trans_of(xforms))
    return axis, angle, cen, hel

def angle_of(xforms, debug=False):
    tr = xforms[..., 0, 0] + xforms[..., 1, 1] + xforms[..., 2, 2]
    cos = (tr - 1.0) / 2.0
    angl = np.arccos(np.clip(cos, -1, 1))
    return angl

def angle_of_degrees(xforms, debug=False):
    tr = xforms[..., 0, 0] + xforms[..., 1, 1] + xforms[..., 2, 2]
    cos = (tr - 1.0) / 2.0
    angl = np.arccos(np.clip(cos, -1, 1))
    return np.degrees(angl)

def rot3(axis, angle=None, nfold=None, degrees='auto', dtype='f8', shape=(3, 3), **kw):
    """Angle will override nfold."""
    if angle is None:
        angle = 2 * np.pi / nfold
    angle = np.array(angle, dtype=dtype)

    axis  = np . array(axis , dtype = dtype)
    angle = np . array(angle, dtype = dtype)
    if degrees == 'auto':
        degrees = guess_is_degrees(angle)
    angle = angle * np.pi / 180.0 if degrees else angle
    if axis.shape and angle.shape and not is_broadcastable(axis.shape[:-1], angle.shape):
        raise ValueError('axis and angle not compatible: ' + str(axis.shape) + ' ' + str(angle.shape))
    axis /= np.linalg.norm(axis, axis=-1)[..., np.newaxis]
    a = np.cos(angle / 2.0)
    tmp = axis * -np.sin(angle / 2)[..., np.newaxis]
    b, c, d = tmp[..., 0], tmp[..., 1], tmp[..., 2]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    outshape = angle.shape if angle.shape else axis.shape[:-1]
    rot3 = np.zeros(outshape + shape, dtype=dtype)
    rot3[..., 0, 0] = aa + bb - cc - dd
    rot3[..., 0, 1] = 2 * (bc+ad)
    rot3 [..., 0, 2] = 2 * (bd - ac)
    rot3 [..., 1, 0] = 2 * (bc - ad)
    rot3[..., 1, 1] = aa + cc - bb - dd
    rot3 [..., 1, 2] = 2 * (cd + ab)
    rot3 [..., 2, 0] = 2 * (bd + ac)
    rot3[..., 2, 1] = 2 * (cd - ab)
    rot3[..., 2, 2] = aa + dd - bb - cc
    if shape == (4, 4):
        rot3[..., 3, 3] = 1.0
    return rot3

def hrot(axis, angle=None, center=None, dtype='f8', hel=0.0, **kw):
    """Angle will override nfold."""
    # if isinstance(axis, ipd.dev.Bunch):
    #   bunch = axis
    #   axis = bunch.axis
    #   if angle is None: angle = bunch.angle
    #   return
    axis = np.array(axis, dtype=dtype)
    if axis.shape[-1] == 2:
        assert center is None
        center = axis [..., 0]
        axis   = axis [..., 1]
    else:
        axis = axis
        center = np.array([0, 0, 0], dtype=dtype) if center is None else np.asarray(center, dtype=dtype)

    r = rot3(axis, angle, dtype=dtype, shape=(4, 4), **kw)
    if center.ndim > 1 and axis.ndim == 1:
        rshape, cshape = r.shape, center.shape
        r      = np . tile(r     , cshape [: - 1] + (1,) * len(rshape))
        center = np . tile(center, rshape [: - 2] + (1,) * len(cshape))
        assert center.ndim == 3
        center = center.swapaxes(0, 1)
        # ipd.icv(r.shape)
        # ipd.icv(center.shape)
        # x, y, z = center[..., 0], center[..., 1], center[..., 2]
        # ipd.icv(x.shape)
        # ipd.icv(r[..., 0, 3].shape)
        # assert 0
    x, y, z = center[..., 0], center[..., 1], center[..., 2]
    r [..., 0, 3] = x - r [..., 0, 0] * x - r [..., 0, 1] * y - r [..., 0, 2] * z
    r [..., 1, 3] = y - r [..., 1, 0] * x - r [..., 1, 1] * y - r [..., 1, 2] * z
    r [..., 2, 3] = z - r [..., 2, 0] * x - r [..., 2, 1] * y - r [..., 2, 2] * z
    r[..., 3, 3] = 1
    r[..., :3, 3] += hel * axis[..., :3]
    return r

def hpoint(point):
    point = as_tensor(point)
    if point.shape[-1] == 4:
        if np.allclose(point[..., 3], 1):
            # if True:
            return point.copy()
        else:
            return hpoint(point[..., :3])
    elif point.shape[-1] == 3:
        r = np.ones(point.shape[:-1] + (4,))
        r[..., :3] = point
        return r
    elif point.shape[-2:] == (4, 4):
        return point[..., :, 3].copy()
    else:
        raise ValueError('point must len 3 or 4')

def hpointorvec(point):
    point = as_tensor(point)
    if point.shape[-1] == 4:
        return point.copy()
    return hpoint(point)

def hvec(vec):
    vec = as_tensor(vec)
    if vec.shape[-1] == 4:
        vec = vec.copy()
        vec[..., 3] = 0
        return vec
    elif vec.shape[-1] == 3:
        r = np.zeros(vec.shape[:-1] + (4,))
        r[..., :3] = vec
        return r
    else:
        raise ValueError('vec must len 3 or 4')

def hcentered(coords, singlecom=False):
    origshape = coords.shape[:-1]
    if singlecom:
        coords = coords.reshape(-1, coords.shape[-1])
    coords = hpoint(coords).copy()
    com = hcom(coords)
    delta = com[..., None, :3]
    coords[..., :3] -= delta
    return coords.reshape(*origshape, 4)

def hcentered3(coords, singlecom=False):
    origshape = coords.shape[:-1]
    if singlecom:
        coords = coords.reshape(-1, coords.shape[-1])
    com = hcom(coords)
    delta = com[..., None, :3]
    coords = coords[..., :3] - delta
    return coords.reshape(*origshape, 3)

def hray(origin, direction):
    origin = hpoint(origin)
    direction = hnormalized(hvec(direction))
    s = np.broadcast(origin, direction).shape
    r = np.empty(s[:-1] + (4, 2))
    r[..., : origin.shape[-1], 0] = origin
    r[..., 3, 0] = 1
    r[..., :, 1] = direction
    return r

def hframe(u, v, w, cen=None):
    u, v, w = hpoint(u), hpoint(v), hpoint(w)
    assert u.shape == v.shape == w.shape
    if cen is None:
        cen = u
    cen = hpoint(cen)
    assert cen.shape == u.shape
    stubs = np.empty(u.shape[:-1] + (4, 4))
    stubs[..., :, 0] = hnormalized(u - v)
    stubs[..., :, 2] = hnormalized(hcross(stubs[..., :, 0], w - v))
    stubs[..., :, 1] = hcross(stubs[..., :, 2], stubs[..., :, 0])
    stubs[..., :, 3] = hpoint(cen[..., :])
    return stubs

def rot_of(xforms):
    return xforms[..., :3, :3]

def trans_of(xforms):
    return xforms[..., :, 3]

def xaxis_of(xforms):
    return xforms[..., :, 0]

def yaxis_of(xforms):
    return xforms[..., :, 1]

def zaxis_of(xforms):
    return xforms[..., :, 2]

def htrans(trans, dtype='f8', doto=None):
    if isinstance(trans, (int, float)):
        trans = np.array([trans, 0, 0])
    trans = np.asanyarray(trans)
    if trans.shape[-1] == 4:
        trans = trans[..., :3]
    if trans.shape[-1] != 3:
        raise ValueError('trans should be shape (..., 3)')
    tileshape = trans.shape[:-1] + (1, 1)
    t = np.tile(np.identity(4, dtype), tileshape)
    t[..., : trans.shape[-1], 3] = trans
    return t if doto is None else hxform(t, doto)

def hdot(a, b, outerprod=False):
    a = np . asanyarray(a)
    b = np . asanyarray(b)
    if outerprod:
        shape1 = a . shape [: - 1]
        shape2 = b . shape [: - 1]
        a = a.reshape((1,) * len(shape2) + shape1 + (-1,))
        b = b.reshape(shape2 + (1,) * len(shape1) + (-1,))
    return np.sum(a[..., :3] * b[..., :3], axis=-1)

def hcross(a, b):
    a = hvec(np . asanyarray(a))
    b = hvec(np . asanyarray(b))
    c = np.zeros(np.broadcast(a, b).shape, dtype=a.dtype)
    c[..., :3] = np.cross(a[..., :3], b[..., :3])
    return c

def hnorm(a):
    a = np.asanyarray(a)
    return np.sqrt(np.sum(a[..., :3] ** 2, axis=-1))

def hnorm2(a):
    a = hvec(a)
    return np.sum(a[..., :3] * a[..., :3], axis=-1)

def normalized_3x3(a):
    return a / np.linalg.norm(a, axis=-1)[..., np.newaxis]

def hnormalized(a, dtype=None):
    a = hvec(a)
    dtype = dtype or a.dtype
    if (not a.shape and len(a) == 3) or (a.shape and a.shape[-1] == 3):
        a, tmp = np.zeros(a.shape[:-1] + (4,), dtype=dtype), a
        a[..., :3] = tmp
    a2 = a.copy()
    a2[..., 3] = 0
    return a2 / hnorm(a2)[..., np.newaxis]

def is_valid_rays(r):
    r = np.asanyarray(r)
    if r.shape[-2:] != (4, 2):
        return False
    if np.any(r[..., 3, :] != (1, 0)):
        return False
    if np.any(abs(np.linalg.norm(r[..., :3, 1], axis=-1) - 1) > 0.000001):
        return False
    return True

@ipd.dev.preserve_random_state
def hrandpoint(shape=(), mean=0, std=1, seed=None):
    if isinstance(shape, int):
        shape = (shape,)
    p = hpoint(np.random.randn(*(shape + (3,))) * std + mean)
    return p

@ipd.dev.preserve_random_state
def rand_vec(shape=(), seed=None):
    if isinstance(shape, int):
        shape = (shape,)
    v = hvec(np.random.randn(*(shape + (3,))))
    if isinstance(shape, int):
        shape = (shape,)
    return v

@ipd.dev.preserve_random_state
def rand_unit(shape=(), seed=None):
    if isinstance(shape, int):
        shape = (shape,)
    v = hnormalized(np.random.randn(*(shape + (3,))))
    return v

def angle(u, v, outerprod=False):
    u, v = hnormalized(u), hnormalized(v)
    d = hdot(u, v, outerprod=outerprod)
    # todo: handle special cases... 1,-1
    return np.arccos(np.clip(d, -1, 1))

def angle_degrees(u, v):
    return angle(u, v) * 180 / np.pi

def line_angle(u, v, outerprod=False):
    a = angle(u, v, outerprod=outerprod)
    return np.minimum(a, np.pi - a)

def line_angle_degrees(u, v, outerprod=False):
    return np.degrees(line_angle(u, v, outerprod))

@ipd.dev.preserve_random_state
def hrandray(shape=(), cen=(0, 0, 0), sdev=1, seed=None, dtype=np.float64):
    if isinstance(shape, int):
        shape = (shape,)
    cen = np.asanyarray(cen)
    if cen.shape[-1] not in (3, 4):
        raise ValueError('cen must be len 3 or 4')
    shape = shape or cen.shape[:-1]
    cen = cen + np.random.randn(*(shape + (3,))) * sdev
    norm = np.random.randn(*(shape + (3,)))
    norm /= np.linalg.norm(norm, axis=-1)[..., np.newaxis]
    r = np.zeros(shape + (4, 2))
    r[..., :3, 0] = cen
    r[..., 3, 0] = 1
    r[..., :3, 1] = norm
    return r

@ipd.dev.preserve_random_state
def rand_xform_aac(shape=(), axis=None, ang=None, cen=None, seed=None, dtype=np.float64):
    if isinstance(shape, int):
        shape = (shape,)
    if axis is None:
        axis = rand_unit(shape)
    if ang is None:
        ang = np.random.rand(*shape) * np.pi  # todo: make uniform!
    if cen is None:
        cen = rand_point(shape)
    # q = rand_quat(shape)
    return hrot(axis, ang, cen)

@ipd.dev.preserve_random_state
def hrandsmall(shape=(), cart_sd=0.001, rot_sd=0.001, centers=None, seed=None, doto=None, dtype=np.float64):
    if isinstance(shape, int):
        shape = (shape,)
    axis = rand_unit(shape)
    ang = np.random.normal(0, rot_sd, shape) * np.pi
    if centers is None:
        centers = [0, 0, 0, 1]
    else:
        assert centers.shape[:-1] in ((), shape)
    x = hrot(axis, ang, centers, degrees=False).squeeze()
    trans = np.random.normal(0, cart_sd, x[..., :3, 3].shape)
    x[..., :3, 3] += trans
    return x.squeeze() if doto is None else hxform(x, doto)

rand_xform_small = hrandsmall

@ipd.dev.preserve_random_state
def hrand(shape=(), cart_cen=0, cart_sd=1, seed=None, dtype=np.float64):
    if isinstance(shape, int):
        shape = (shape,)
    q = rand_quat(
        shape,
    )
    x = quat_to_xform(q)
    x[..., :3, 3] = np.random.randn(*shape, 3) * cart_sd + cart_cen
    return np.ascontiguousarray(x)

rand_xform = hrand

@ipd.dev.preserve_random_state
def hrandrot(shape=(), seed=None, dtype=np.float64):
    if isinstance(shape, int):
        shape = (shape,)
    quat = rand_quat  (shape)
    rot  = quat_to_rot(quat )
    return np.ascontiguousarray(hconvert(rot))

@ipd.dev.preserve_random_state
def hrandrotsmall(shape=(), rot_sd=0.001, seed=None):
    if isinstance(shape, int):
        shape = (shape,)
    axis = rand_unit(shape)
    ang = np.random.normal(0, rot_sd, shape) * np.pi
    r = rot3(axis, ang, degrees=False).squeeze()
    return hconvert(r.squeeze())

def hrms(a, b):
    a = hpoint(a)
    b = hpoint(b)
    assert a.shape == b.shape
    return np.sqrt(np.sum(np.square(a - b)) / len(a))

def hparallel(a, b):
    return angle(a, b) < 0.001

def unhomog(stuff):
    return stuff[..., :3]

Result_hrmsfit = collections.namedtuple('Result_hrmsfit', 'rms fitcoords xfit')

@ipd.dev.timed
def hrmsfit(mobile, target):
    """Use kabsch method to get rmsd fit."""
    mobile = hpoint(mobile)
    target = hpoint(target)
    assert mobile.shape == target.shape
    assert mobile.ndim > 1

    mobile_cen = np . mean(mobile, axis = 0)
    target_cen = np . mean(target, axis = 0)
    mobile = mobile - mobile_cen
    target = target - target_cen
    # ipd.icv(mobile.shape)
    # ipd.icv(target.shape[-1] in (3, 4))
    covariance = mobile.T[:3] @ target[:, :3]
    V, S, W = np.linalg.svd(covariance)
    if 0 > np.linalg.det(V) * np.linalg.det(W):
        S = np.array([S[0], S[1], -S[2]], dtype=S.dtype)
        # S[-1] = -S[-1]
        # ipd.icv(S - S1)
        V = np.concatenate([V[:, :-1], -V[:, -1, None]], axis=1)
        # V[:, -1] = -V[:, -1]
        # ipd.icv(V - V1)
        # assert 0
    rot_m2t = hconvert(V @ W).T
    trans_m2t = target_cen - rot_m2t @ mobile_cen
    xform_mobile_to_target = hconvert(rot_m2t, trans_m2t)

    mobile = mobile + mobile_cen
    target = target + target_cen
    mobile_fit_to_target = hxformpts(xform_mobile_to_target, mobile)
    rms = hrms(target, mobile_fit_to_target)

    return Result_hrmsfit(rms, mobile_fit_to_target, xform_mobile_to_target)

def hproj(u, v):
    u = hvec       (u)
    v = hpointorvec(v)
    return hdot(u, v)[..., None] / hnorm2(u)[..., None] * u

def hcart(x):
    assert x.shape[-2:] == (4, 4)
    return x[..., :, 3]

def hcart3(x):
    assert x.shape[-2:] == (4, 4)
    return x[..., :3, 3]

def hori3(x):
    assert x.shape[-2:] == (4, 4)
    return x[..., :3, :3]

hori = hori3

def hprojperp(u, v):
    u = hvec       (u)
    v = hpointorvec(v)
    # return v - hdot(u, v)[..., None] / hnorm2(u)[..., None] * u
    return v - hproj(u, v)

def point_in_plane(plane, pt):
    return np.abs(hdot(plane[..., :3, 1], pt[..., :3] - plane[..., :3, 0])) < 0.000001

def ray_in_plane(plane, ray):
    assert ray.shape[-2:] == (4, 2)
    return point_in_plane(plane, ray[..., :3, 0]) * point_in_plane(plane, ray[..., :3, 0] + ray[..., :3, 1])

def hpointlineclose(point, cen, norm):
    point = hpoint     (point)
    cen   = hpoint     (cen  )
    norm  = hnormalized(norm )
    point = point - cen
    perp = hprojperp(norm, point)
    return perp + cen

def h_point_line_dist(point, cen, norm):
    point = hpoint     (point)
    cen   = hpoint     (cen  )
    norm  = hnormalized(norm )
    point = point - cen
    perp = hprojperp(norm, point)
    return hnorm(perp)

def closest_point_on_line(target, cen, norm):
    target = hpoint     (target)
    cen    = hpoint     (cen   )
    norm   = hnormalized(norm  )
    cen2point = target - cen
    return hdot(cen2point, norm)[..., None] * norm + cen

def intesect_line_plane(p0, n, l0, l):
    l = hm.hnormalized(l)
    d = hm.hdot(p0 - l0, n) / hm.hdot(l, n)
    return l0 + l * d

def intersect_planes(plane1, plane2):
    """
    find the 3D intersection of two planes

    status 0 -> intersection returned
    status 1 -> disjoint (no intersection)
    status 2 -> the two planes coincide

    Args:
        plane1: shape = (..., 4, 2)
        plane2: shape = (..., 4, 2)

    Returns:
        rays shape=(...,4,2), status

    """
    if not is_valid_rays(plane1):
        raise ValueError('invalid plane1')
    if not is_valid_rays(plane2):
        raise ValueError('invalid plane2')
    shape1, shape2 = np.array(plane1.shape), np.array(plane2.shape)
    if np.any((shape1 != shape2) * (shape1 != 1) * (shape2 != 1)):
        raise ValueError('incompatible shapes for plane1, plane2:')
    p1, n1 = plane1 [...,: 3, 0], plane1 [...,: 3, 1]
    p2, n2 = plane2 [...,: 3, 0], plane2 [...,: 3, 1]
    shape = tuple(np.maximum(plane1.shape, plane2.shape))
    u = np.cross(n1, n2)
    abs_u = np.abs(u)
    planes_parallel = np.sum(abs_u, axis=-1) < 0.000001
    p2_in_plane1 = point_in_plane(plane1, p2)
    status = np.zeros(shape[:-2])
    status[planes_parallel] = 1
    status[planes_parallel * p2_in_plane1] = 2
    d1 = - hdot(n1, p1)
    d2 = - hdot(n2, p2)
    amax = np.argmax(abs_u, axis=-1)
    sel0, sel1, sel2 = amax == 0, amax == 1, amax == 2
    n1a, n2a, d1a, d2a, ua = (x[sel0] for x in (n1, n2, d1, d2, u))
    n1b, n2b, d1b, d2b, ub = (x[sel1] for x in (n1, n2, d1, d2, u))
    n1c, n2c, d1c, d2c, uc = (x[sel2] for x in (n1, n2, d1, d2, u))

    ay = (d2a * n1a [..., 2] - d1a * n2a [..., 2]) / ua [..., 0]
    az = (d1a * n2a [..., 1] - d2a * n1a [..., 1]) / ua [..., 0]
    bz = (d2b * n1b [..., 0] - d1b * n2b [..., 0]) / ub [..., 1]
    bx = (d1b * n2b [..., 2] - d2b * n1b [..., 2]) / ub [..., 1]
    cx = (d2c * n1c [..., 1] - d1c * n2c [..., 1]) / uc [..., 2]
    cy = (d1c * n2c [..., 0] - d2c * n1c [..., 0]) / uc [..., 2]
    isect_pt = np.empty(shape[:-2] + (3,), dtype=plane1.dtype)
    isect_pt[sel0, 0] = 0
    isect_pt [sel0, 1] = ay
    isect_pt [sel0, 2] = az
    isect_pt [sel1, 0] = bx
    isect_pt[sel1, 1] = 0
    isect_pt [sel1, 2] = bz
    isect_pt [sel2, 0] = cx
    isect_pt [sel2, 1] = cy
    isect_pt[sel2, 2] = 0
    isect = hray(isect_pt, u)
    return isect, status

def axis_ang_cen_of_eig(xforms, debug=False):
    # raise NotImplementedError('this is a bad way to get rotation axis')
    axis, angle = axis_angle_of(xforms)
    # seems to numerically unstable??
    ev, cen = np.linalg.eig(xforms)
    cen = np.real(cen[..., 3])
    cen = cen / cen[..., 3][..., None]  # normalize homogeneous coord
    cen = cen - axis * np.sum(axis * cen)
    return axis, angle, cen

_axis_ang_cen_magic_points_numpy = np.array(
    [
        [-32.09501046777237, 3.36227004372687, 35.34672781477340, 1.0],
        [21.15113978202345, 12.55664537217840, -37.48294301885574, 1.0],
    ]
)

def axis_ang_cen_of_planes(xforms, debug=False, ident_match_tol=1e-8):
    """If angle is 0, will return axis along translation."""
    origshape = xforms.shape[:-2]
    xforms = xforms.reshape(-1, 4, 4)

    axis, angle = axis_angle_of(xforms)
    not_ident = np.abs(angle) > ident_match_tol
    cen = np.tile([0.0, 0.0, 0.0, 1.0], np.shape(angle)).reshape(*np.shape(angle), 4)

    if np.any(not_ident):
        xforms1 = xforms [not_ident]
        axis1   = axis   [not_ident]
        #  sketchy magic points...
        p1, p2 = _axis_ang_cen_magic_points_numpy
        tparallel = hdot(axis1, xforms1[..., :, 3])[..., None] * axis1

        q1 = xforms1 @ p1 - tparallel
        q2 = xforms1 @ p2 - tparallel
        n1 = hnormalized(q1 - p1)
        n2 = hnormalized(q2 - p2)
        c1 = (p1 + q1) / 2.0
        c2 = (p2 + q2) / 2.0
        plane1 = hray(c1, n1)
        plane2 = hray(c2, n2)
        isect, status = intersect_planes(plane1, plane2)
        cen1 = isect[..., :, 0]
        # ipd.icv(cen1)

        if len(cen) == len(cen1):
            cen = cen1
        else:
            cen[not_ident] = cen1

    # by convention, if no rotation, make axis along translation.
    trans = hcart(xforms)
    istransonly = np.logical_and(np.isclose(0, angle), np.any(~np.isclose(0, trans[:, :3]), axis=1))
    axis[istransonly] = hnormalized(trans[istransonly])

    axis = axis.reshape(*origshape, 4)
    angle = angle.reshape(origshape)
    cen = cen.reshape(*origshape, 4)

    return axis, angle, cen

axis_ang_cen_of = axis_ang_cen_of_planes

def line_line_distance_pa(pt1, ax1, pt2, ax2):
    pt1, pt2 = hpoint     (pt1), hpoint     (pt2)
    ax1, ax2 = hnormalized(ax1), hnormalized(ax2)
    n = abs(hdot(pt2 - pt1, hcross(ax1, ax2)))
    d = hnorm(hcross(ax1, ax2))
    r = np.zeros_like(n)
    i = abs(d) > 0.00001
    r[i] = n[i] / d[i]
    pp = hnorm(hprojperp(ax1, pt2 - pt1))
    dist = np.where(np.abs(hdot(ax1, ax2)) > 0.9999, pp, r)
    return dist

def line_line_distance(ray1, ray2):
    pt1, pt2 = ray1 [...,:, 0], ray2 [...,:, 0]
    ax1, ax2 = ray1 [...,:, 1], ray2 [...,:, 1]
    return line_line_distance_pa(pt1, ax1, pt2, ax2)

def line_line_closest_points_pa(pt1, ax1, pt2, ax2, verbose=0):
    pt1 = hpoint(pt1)
    ax1 = hvec  (ax1)
    pt2 = hpoint(pt2)
    ax2 = hvec  (ax2)
    assert pt1.shape == pt2.shape == ax1.shape == ax2.shape
    # origshape = pt1.shape
    # ipd.icv(pt1.shape)
    C21 = pt2 - pt1
    M = hcross(ax1, ax2)
    m2 = np.sum(M**2, axis=-1)[..., None]
    if np.allclose(m2, 0):
        return pt1, pt1
    R = hcross(C21, M / m2)
    t1 = hdot(R, ax2) [..., None]
    t2 = hdot(R, ax1) [..., None]
    Q1 = pt1 - t1 * ax1
    Q2 = pt2 - t2 * ax2

    if verbose:
        ipd . icv('C21', C21)
        ipd . icv('M'  , M  )
        ipd . icv('m2' , m2 )
        ipd . icv('R'  , R  )
        ipd . icv('t1' , t1 )
        ipd . icv('t2' , t2 )
        ipd . icv('Q1' , Q1 )
        ipd . icv('Q2' , Q2 )
    return Q1, Q2

hlinesisect = line_line_closest_points_pa

def line_line_closest_points(ray1, ray2, verbose=0):
    "currently errors if ax1==ax2"
    # pt1, pt2 = hpoint(pt1), hpoint(pt2)
    # ax1, ax2 = hnormalized(ax1), hnormalized(ax2)
    pt1, pt2 = ray1 [...,:, 0], ray2 [...,:, 0]
    ax1, ax2 = ray1 [...,:, 1], ray2 [...,:, 1]
    return line_line_closest_points_pa(pt1, ax1, pt2, ax2)

def dihedral(p1, p2, p3, p4):
    p1, p2, p3, p4 = hpoint(p1), hpoint(p2), hpoint(p3), hpoint(p4)
    a = hnormalized(p2 - p1)
    b = hnormalized(p3 - p2)
    c = hnormalized(p4 - p3)
    x = np.clip(hdot(a, b) * hdot(b, c) - hdot(a, c), -1, 1)
    y = np.clip(hdot(a, hcross(b, c)), -1, 1)
    return np.arctan2(y, x)

def align_around_axis(axis, u, v):
    return hrot(axis, -dihedral(u, axis, [0, 0, 0, 0], v))

def halign(a, b, doto=None):
    # if a.shape == b.shape and np.allclose(a, b): return np.eye(4)
    # x = hrot((hnormalized(a) + hnormalized(b)) / 2, np.pi)
    a, b = hvec(a), hvec(b)
    ax = hcross(a, b)
    if np.allclose(ax, 0):
        return np.eye(4)
    x = np.tile(np.eye(4), (*ax.shape[:-1], 1, 1))
    ok = ~np.all(np.isclose(ax, 0), axis=-1)
    # ipd.icv(ok.shape, x.shape, a.shape, b.shape)
    if a . ndim > 1: a = a [ok]
    if b . ndim > 1: b = b [ok]
    # ipd.icv(ok.shape, x.shape, a.shape, b.shape)
    tmp = align_around_axis(ax[ok], a, b)
    x[ok] = tmp
    # ipd.icv(x.shape)
    return x if doto is None else hxform(x, doto)

def halign2(a1, a2, b1, b2, doto=None, strict=False):
    "minimizes angular error"
    a1, a2, b1, b2 = (hnormalized(v) for v in (a1, a2, b1, b2))
    if strict:
        assert np.allclose(np.dot(a1, a2), np.dot(b1, b2), atol=0.001)
    aaxis = hnormalized(a1 + a2)
    baxis = hnormalized(b1 + b2)
    # baxis = np.where(hangle(aaxis, baxis) > , baxis, -baxis)
    spinaxis = (aaxis + baxis) / 2
    arbitrary = _axis_ang_cen_magic_points_numpy[0]
    spinaxis = np.where(hnorm(spinaxis) > 0.00001, spinaxis, hcross(aaxis, arbitrary))
    Xmiddle = hrot(spinaxis, np.pi)
    Xaround = align_around_axis(baxis, Xmiddle @ a1, b1)
    X = Xaround @ Xmiddle
    # ipd.icv(angle(b1, a1), angle(b2, a2), angle(b1, X @ a1), angle(b2, X @ a2))
    assert (angle(b1, a1) + angle(b2, a2)) + 0.001 >= (angle(b1, X @ a1) + angle(b2, X @ a2))
    return X if doto is None else hxform(X, doto)

def calc_dihedral_angle(p1, p2, p3, p4):
    p1, p2, p3, p4 = hpoint(p1), hpoint(p2), hpoint(p3), hpoint(p4)
    p1, p2, p3, p4 = p1.reshape(4), p2.reshape(4), p3.reshape(4), p4.reshape(4)
    # Calculate coordinates for vectors q1, q2 and q3
    q1 = np.subtract(p2, p1)  # b - a
    q2 = np.subtract(p3, p2)  # c - b
    q3 = np.subtract(p4, p3)  # d - c
    q1_x_q2 = hcross(q1, q2)
    q2_x_q3 = hcross(q2, q3)
    n1 = hnormalized(q1_x_q2)
    n2 = hnormalized(q2_x_q3)
    u1 = n2
    u3 = hnormalized(q2)
    u2 = hcross(u3, u1)
    cos_theta = np . sum(n1 * u1)
    sin_theta = np . sum(n1 * u2)
    theta = -np.arctan2(sin_theta, cos_theta)
    return theta

def rotation_around_dof_for_target_angle(target_angle, dof_angle, fix_to_dof_angle):
    assert fix_to_dof_angle < np.pi / 2
    assert dof_angle <= np.pi / 2 + 0.00001
    assert target_angle <= np.pi

    if target_angle + dof_angle < fix_to_dof_angle:
        return np.array([-12345.0])
    if dof_angle < 1e-6 or target_angle < 1e-6 or fix_to_dof_angle < 1e-6:
        return np.array([-12345.0])

    hdof  = np . sin(dof_angle   )
    l_dof = np . cos(dof_angle   )
    h_tgt = np . sin(target_angle)
    l_tgt = np . cos(target_angle)
    # ipd.icv('l_dof', l_dof)
    # ipd.icv('l_tgt', l_tgt)
    xdof = np . sin(fix_to_dof_angle) * l_dof
    ydof = np . cos(fix_to_dof_angle) * l_dof
    assert np.allclose(np.sqrt(xdof**2 + ydof**2), l_dof)
    ytgt = np.cos(target_angle)
    slope = -np.tan(np.pi / 2 - fix_to_dof_angle)

    # ipd.icv('ytgt', ytgt, 'xdof', xdof, 'ydof', ydof)

    yhat = ytgt
    xhat = xdof + (ytgt - ydof) * slope
    lhat = np.sqrt(xhat**2 + yhat**2)

    lhat = min(lhat, 1.0)

    # this caused occasional test failures
    # if lhat > 0.999999:
    #    if lhat > 1.000001:
    #       return np.array([-12345.0])
    #    else:
    #       return np.array([0.0])

    hhat = np.sqrt(1.0 - lhat**2)
    ahat = np.arcsin(hhat / hdof)

    # ipd.icv('xhat', xhat, 'yhat', yhat, 'slope', slope, 'lhat', lhat, 'hhat', hhat, 'ahat', ahat)

    # ipd.icv('ytgt', ytgt)
    # ipd.icv('xdof', xdof)
    # ipd.icv('ydof', ydof)
    # ipd.icv('xhat', xhat)
    # ipd.icv('yhat', yhat)
    # ipd.icv('ahat', ahat, np.degrees(ahat))

    return ahat

def xform_around_dof_for_vector_target_angle(fix, mov, dof, target_angle):
    if hdot(dof, fix) < 0:
        dof = -dof
    if angle(dof, mov) > np.pi / 2:
        mov = -mov
    dang = calc_dihedral_angle(fix, [0.0, 0.0, 0.0, 0.0], dof, mov)
    assert angle(dof, mov) <= np.pi / 2 + 0.000001
    ahat = rotation_around_dof_for_target_angle(target_angle, angle(mov, dof), angle(fix, dof))
    if ahat == -12345.0:
        return []
    elif ahat == 0:
        mov1 = (hrot(dof, 0.000 - dang) @ mov[..., None]).reshape(1, 4)
        mov2 = (hrot(dof, np.pi - dang) @ mov[..., None]).reshape(1, 4)
        if np.allclose(angle(fix, mov1), target_angle):
            return [hrot(dof, np.pi - dang)]
            return
        elif np.allclose(angle(fix, mov1), target_angle):
            return [hrot(dof, np.pi - dang)]
        else:
            return []
    else:
        angles = [-dang + ahat, -dang - ahat, np.pi - dang + ahat, np.pi - dang - ahat]
        moves = [(hrot(dof, ang + 0.000) @ mov[..., None]).reshape(1, 4) for ang in angles]
        if not (
            np.allclose(angle(moves[0], fix), angle(moves[1], fix))
            or np.allclose(angle(moves[2], fix), angle(moves[3], fix))
        ):
            return []

        if np.allclose(angle(moves[0], fix), target_angle):
            return [hrot(dof, angles[0]), hrot(dof, angles[1])]
        elif np.allclose(angle(moves[2], fix), target_angle):
            return [hrot(dof, angles[2]), hrot(dof, angles[3])]
        else:
            return []

def align_lines_isect_axis2(pt1, ax1, pt2, ax2, ta1, tp1, ta2, sl2, strict=True):
    """Zomg, point/axis reversed for second half of args..."""
    ## make sure to align with smaller axis choice
    assert np.allclose(np.linalg.norm(tp1[..., :3]), 0.0)
    if angle(ax1, ax2) > np.pi / 2:
        ax2 = -ax2
    if angle(ta1, ta2) > np.pi / 2:
        ta2 = -ta2
    if strict:
        assert np.allclose(angle(ta1, ta2), angle(ax1, ax2))
    if abs(angle(ta1, ta2)) < 0.01:
        assert 0, 'case not tested'
        # vector delta between pt2 and pt1
        d = hprojperp(ax1, pt2 - pt1)
        Xalign = halign2(ax1, d, ta1, sl2)  # align d to Y axis
        Xalign[..., :, 3] = -Xalign @ pt1
        slide_dist = (Xalign @ pt2)[..., 1]
    else:
        try:
            Xalign = halign2(ax1, ax2, ta1, ta2)
            # ipd.icv(Xalign @ ax1, ta1)
            # assert np.allclose(Xalign @ ax1, ta1, atol=0.0001)
            # assert np.allclose(Xalign @ ax2, ta2, atol=0.0001)
            # ipd.icv(Xalign)
        except AssertionError as e:
            ipd.icv('halign2 error')
            ipd . icv('   ', ax1)
            ipd . icv('   ', ax2)
            ipd . icv('   ', ta1)
            ipd . icv('   ', ta2)
            raise e
        Xalign[..., :, 3] = -Xalign @ pt1  ## move pt1 to origin
        Xalign[..., 3, 3] = 1
        cen2_0 = Xalign @ pt2  # moving pt2 by Xalign
        D = np.stack([ta1[:3], sl2[:3], ta2[:3]]).T
        A1offset, slide_dist, _ = np.linalg.inv(D) @ cen2_0[:3]
        # ipd.icv(A1offset, slide_dist)
        Xalign[..., :, 3] = Xalign[..., :, 3] - (A1offset * ta1)

    return Xalign, slide_dist

def scale_translate_lines_isect_lines(pt1, ax1, pt2, ax2, tp1, ta1, tp2, ta2):
    _pt1 = hpoint(pt1.copy())
    _ax1 = hnormalized(ax1.copy())
    _pt2 = hpoint(pt2.copy())
    _ax2 = hnormalized(ax2.copy())
    _tp1 = hpoint(tp1.copy())
    _ta1 = hnormalized(ta1.copy())
    _tp2 = hpoint(tp2.copy())
    _ta2 = hnormalized(ta2.copy())

    if abs(angle(_ax1, _ax2) - angle(_ta1, _ta2)) > 0.00001:
        _ta2 = -_ta2
    # ipd.icv(_ax1)
    # ipd.icv(_ax2)
    # ipd.icv(_ta1, ta1)
    # ipd.icv(_ta2)
    # ipd.icv(line_angle(_ax1, _ax2), line_angle(_ta1, _ta2))
    assert np.allclose(line_angle(_ax1, _ax2), line_angle(_ta1, _ta2))

    # scale target frame to match input line separation
    d1 = line_line_distance_pa(_pt1, _ax1, _pt2, _ax2)
    d2 = line_line_distance_pa(_tp1, _ta1, _tp2, _ta2)
    scale = np.array([d1 / d2, d1 / d2, d1 / d2, 1])
    _tp1 *= scale
    _tp2 *= scale

    # compute rotation to align line pairs, check "handedness" and correct if necessary
    xalign = halign2(_ax1, _ax2, _ta1, _ta2)
    a, b = line_line_closest_points_pa(_pt1, _ax1, _pt2, _ax2)
    c, d = line_line_closest_points_pa(_tp1, _ta1, _tp2, _ta2)
    _shift1 = xalign @ (b - a)
    _shift2 = d - c
    if hdot(_shift1, _shift2) < 0:
        if np.allclose(angle(_ax1, _ax2), np.pi / 2):
            xalign = halign2(-_ax1, _ax2, _ta1, _ta2)
        else:
            scale[:3] = -scale[:3]
            _tp1 *= - 1
            _tp2 *= - 1
            # rays = np.array([
            #    hm.hray(xalign @ pt1, xalign @ ax1),
            #    hm.hray(xalign @ pt2, xalign @ ax2),
            #    hm.hray(scale * tp1, scale * ta1),
            #    hm.hray(scale * tp2, scale * ta2),
            # ])
            # colors = [(1, 0, 0), (0, 0, 1), (0.8, 0.5, 0.5), (0.5, 0.5, 0.8)]
            # rp.viz.showme(rays, colors=colors, block=False)

    _pt1 = xalign @ _pt1
    _ax1 = xalign @ _ax1
    _pt2 = xalign @ _pt2
    _ax2 = xalign @ _ax2

    assert np.allclose(_ax1, _ta1, atol=1e-3) or np.allclose(-_ax1, _ta1, atol=1e-3)
    assert np.allclose(_ax2, _ta2, atol=1e-3) or np.allclose(-_ax2, _ta2, atol=1e-3)

    # move to overlap pa1,_ta1, aligning first axes
    delta1 = _tp1 - _pt1
    _pt1 += delta1
    _pt2 += delta1

    # delta align second axes by moving alone first
    pp = hprojperp(_ta2, _tp2 - _pt2)
    d = np.linalg.norm(pp)
    if d < 0.00001:
        delta2 = 0
    else:
        a = line_angle(_ta1, _ta2)
        l = d / np.sin(a)
        delta2 = l * hnormalized(hproj(_ta1, _tp2 - _pt2))
        if hdot(pp, delta2) < 0:
            delta2 *= -1
    _pt1 += delta2
    _pt2 += delta2

    xalign[:, 3] = delta1 + delta2
    xalign[3, 3] = 1

    if np.any(np.isnan(xalign)):
        ipd.icv('=============================')
        ipd.icv(xalign)
        ipd.icv(delta1, delta2)
    # rays = np.array([
    #    hm.hray(xalign @ pt1, xalign @ ax1),
    #    hm.hray(xalign @ pt2, xalign @ ax2),
    #    hm.hray(scale * tp1, scale * ta1),
    #    hm.hray(scale * tp2, scale * ta2),
    # ])
    # colors = [(1, 0, 0), (0, 0, 1), (0.8, 0.5, 0.5), (0.5, 0.5, 0.8)]
    # rp.viz.showme(rays, colors=colors, block=False)
    # assert 0

    return xalign, scale

def hcoherence(xforms, lever):
    xforms = xforms.copy()
    xforms[:, :3, :3] *= lever
    dist = xforms.reshape(1, -1, 4, 4) - xforms.reshape(-1, 1, 4, 4)
    dist = np.sqrt(np.sum(dist**2) / 4 / len(xforms))
    return dist

def hmean(xforms):
    q = rot_to_quat(xforms)
    idx = np.dot(q, [13, 7, 3, 1]) < 0
    if np.any(idx):
        q[idx] *= -1
    # ipd.icv('hmean')
    # ipd.icv(q)
    # for i in range(10):
    # qmean = np.mean(q, axis=0)
    # dot = hdot(qmean, q)
    # if np.all(dot >= 0):
    # q[dot < 0] *= -1
    # else:
    # assert 0, 'hmean cant find coherent quat mean'

    q = np.mean(q, axis=0)
    q = q / np.linalg.norm(q)
    r = quat_to_rot(q)
    t = np.mean(xforms[..., :3, 3], axis=0)
    x = hconstruct(r, t)
    # ipd.icv(x)
    return x

def hexpand(
    generators,
    depth=8,
    ntrials=1000_000,
    radius=9e9,
    cen=[0, 0, 0],
    deterministic=True,
):
    raise NotImplementedError('expand_xforms_rand lives in hgeom now')
    generators = np.asarray(generators).astype(np.float64)
    cen = np.asarray(cen).astype(np.float64)
    x, _ = ipd.homog.hgeom.expand_xforms_rand(
        generators,
        depth=depth,
        trials=ntrials,
        radius=radius,
        cen=cen,
        deterministic=deterministic,
    )
    return x

def hpow(xform, power):
    if power % 1 == 0:
        return hpow_int(xform, power)
    else:
        return hpow_float(xform, power)

def hpow_int(xform, power):
    if not power % 1 == 0:
        raise ValueError(f'power {power} is not integer')
    result = np.tile(np.eye(4), (*xform.shape[:-2], 1, 1))
    if power < 0:
        power = -power
        xform = hinv(xform)
    for i in range(power):
        result = hxform(xform, result)
    return result

def hpow_float(xform, power):
    # ipd.icv('get rot, axis, hel, cen')
    # ipd.icv('interp rot/hel around axis/cen')
    return NotImplemented

def hcom_flat(points):
    return np.mean(points, axis=-2)

def hcom(points, flat=False):
    assert len(points) > 0
    points = hpoint(points)
    if flat:
        return np.mean(points.reshape(-1, 4), axis=0)
    if points.shape == (4,):
        points = points.reshape(1, 4)
    oshape = points.shape
    points = points.reshape(-1, oshape[-2], 4)
    com = hcom_flat(points)
    com = com.reshape(*oshape[:-2], 4)
    return com

def hrog_flat(points):
    com = hcom_flat(points).reshape(-1, 1, 4)
    delta = np.linalg.norm(points - com, axis=2)
    rg = np.sqrt(np.mean(delta**2, axis=1))
    return rg

def hrog(points):
    points = hpoint(points)
    oshape = points.shape
    points = points.reshape(-1, *oshape[-2:])
    rog = hrog_flat(points)
    rog = rog.reshape(oshape[:-2])
    return rog

def hconvert(rot=np.eye(3), trans=None, **kw):
    if trans is None:
        trans = np.zeros(rot.shape[:-2] + (4,))
    trans = hpoint(trans)

    if rot.shape[-2:] == (3, 3):
        rot = np . concatenate([rot, np . zeros(rot . shape [:-2]+(3, 1))], axis = - 1)
        rot = np . concatenate([rot, np . zeros(rot . shape [:-2]+(1, 4))], axis = - 2)
        rot[..., 3, 3] = 1
        # assert hvalid(rot)

    assert rot.shape[-2:] == (4, 4)
    assert trans.shape[-1:] == (4, )

    h = np.concatenate([rot[:, :3], trans[:, None]], axis=1)
    h[..., 3, :3] = 0
    h[..., 3, 3] = 1
    return h

def lines_concurrent_isect(cen, axis, tol=1e-4):
    cen, axis = as_tensors([cen, axis])
    cen, axis = cen.reshape(-1, 4), axis.reshape(-1, 4)
    assert cen.shape == axis.shape
    if len(cen) == 1: return True, cen, np.array([0])
    pt1, pt2 = line_line_closest_points_pa(cen[0], axis[0], cen[1], axis[1])
    pt = (pt1 + pt2) / 2
    if not np.sqrt(np.sum((pt1 - pt2)**2)) < tol: return False, None, None
    if len(cen) == 2: return True, pt, hnorm(pt1 - pt2)
    dist = point_line_dist_pa(pt, cen, axis)
    return all(dist < tol), pt, dist

def lines_all_concurrent(cen, axis, tol=1e-4):
    """
    test if all lines defined by cen and axis intersect at a common point
    """
    return lines_concurrent_isect(cen, axis, tol=tol)[0]

def unique_symaxes(
    axis,
    cen,
    ang=None,
    hel=None,
    *extra,
    frames=np.eye(4)[None],
    target=None,
    tol=1e-3,
    lineuniq_method='symaverage',
    lever=100,
    debug=False,
    **kw,
):
    """
    return unique set of lines, where lines in different frames are considered the same.
    also checks anything in extras
    """
    assert axis.shape == cen.shape and axis.ndim == 2 and axis.shape[1] == 4
    tol = ipd.dev.Tolerances(tol, **kw)
    assert tol.angle.threshold < 0.1
    assert axis.shape == cen.shape
    if ang is None: ang = np . ones (len(cen))
    if hel is None: hel = np . zeros(len(cen))
    if target is None: target = normalized([10, .1, 999, 0])
    frames = frames.reshape(-1, 4, 4)
    ax = axis . reshape( - 1, axis . shape [ - 1])
    cn = cen  . reshape( - 1, cen  . shape [ - 1])
    an = ang . reshape(ang . shape [ - 1])
    hl = hel . reshape(hel . shape [ - 1])
    extra = list(extra)
    for i, e in enumerate(extra):
        if e.ndim == 1: extra[i] = e[:, None]
        extra[i] = e.reshape(-1, extra[i].shape[-1])
        assert len(e) == len(ax)
    uniq = []
    # ipd.icv(tol.isect, tol.dot_norm, tol.misc_lineuniq)
    for step in range(100):
        # ipd.icv(ax.shape, cn.shape, frames.shape)
        sax, scn = hxformvec(frames, ax[0]), hxformpts(frames, cn[0])
        pdist = h_point_line_dist(cn[:, None], scn[None], sax[None])
        adot = dot(sax[None], ax[:, None])
        same = 1 - np.abs(adot) < tol.dot_norm
        # print('dot  ', same.any(axis=1).sum())
        same &= pdist < tol.isect
        same = same.any(axis=1)
        # ipd.icv(same.shape, an.shape, (an[0]-an).shape)
        if an is not None: same &= np . abs(an [0] - an) < tol . angle
        if hl is not None: same &= np . abs(hl [0] - hl) < tol . helical_shift
        # print('pdist', same.sum())
        for e in extra:
            same &= np.sqrt(np.sum((e[0] - e) ** 2, axis=1)) < tol.misc_lineuniq
            # print('extra', same.sum())
        # print('final', same.sum())
        if debug:
            pdistmin, adotmax = pdist.min(1), np.abs(adot).max(1)
            ipd.print_table(ipd.dev.picklocals('pdistmin adotmax an hl', idx=same))
        assert np.any(same)
        if same.sum() == 1:
            sel = lambda x: x[same][0]
            pick = sel(ax), sel(cn), sel(an), sel(hl), *(sel(e) for e in extra)
        elif lineuniq_method == 'closest':
            which = np.argmin(h_point_line_dist(target, cn[same], ax[same]))
            sel = lambda x: x[same][which]
            if hdot(sel(ax), target) < 0: ax[same] *= -1
            pick = sel(ax), sel(cn), sel(an), sel(hl), *(sel(e) for e in extra)
        elif lineuniq_method == 'symaverage':
            cax, ccn = sym_closest_lines_if_already_close(ax[same], cn[same], frames, target=target)
            sel = lambda x: x[same].mean()
            pick = cax.mean(0), ccn.mean(0), sel(an), sel(hl), *(sel(e) for e in extra)
        uniq.append(pick)
        if np.all(same): break
        ax, cn, an, hl, extra = ax[~same], cn[~same], an[~same], hl[~same], [e[~same] for e in extra]
    else:
        raise ValueError('too many steps of unique_symaxes')
    result = tuple(np.stack([tup[i] for tup in uniq]) for i in range(len(uniq[0])))
    return result

def _target_axis_cen(target):
    if len(target) == 2 and target[1] is None: target = target[0]
    if len(target) == 2:
        if np.allclose(target[0, 3], 1) and np.allclose(target[0, 3], 0): target = target[[1, 0]]
        return hvec(target[0]), hpoint(target[1])
    if len(target) == 1: return hvec(target[0]), None
    if len(target) in (3, 4): return hvec(target), None
    raise ValueError('bad target, must be vec or (vec, point)')

def line_line_diff_pa(p1, n1, p2, n2, lever, directed=False):
    assert np . allclose(p1 [..., 3], 1) and np . allclose(n1 [..., 3], 0)
    assert np . allclose(p2 [..., 3], 1) and np . allclose(n2 [..., 3], 0)
    dist = line_line_distance_pa(p1, n1, p2, n2)
    dist += (angle if directed else line_angle)(n1, n2) * lever
    return dist

def sym_line_line_diff_pa(cn1, ax1, cn2, ax2, lever, frames):
    assert cn1.shape == ax1.shape and cn2.shape == ax2.shape
    assert cn1.ndim == cn2.ndim == 2
    symcn1 = hxform(frames, cn1) [:,:, None]
    symax1 = hxform(frames, ax1) [:,:, None]
    diff = line_line_diff_pa(symcn1, symax1, cn2, ax2, lever)
    return diff.min(0)

def sym_closest_lines_if_already_close(axis, cen, frames, target, lever=100):
    if len(axis) > 1:
        symaxis = xformvec(frames, axis [1:])
        symcen  = xformpts(frames, cen  [1:])
        symdist = line_line_diff_pa(cen[0], axis[0], symcen, symaxis, lever)
        closest = np.argmin(symdist, axis=0)
        axis [1:] = symaxis [closest, range(len(closest))]
        cen  [1:] = symcen  [closest, range(len(closest))]
    target = _target_axis_cen(target)
    axis[hdot(axis, axis[0]) < -0.1] *= -1
    tgtaxis, tgtcen = _target_axis_cen(target)
    if tgtcen is None: dist = angle(xformvec(frames, axis[0]), tgtaxis)
    else:
        dist = line_line_diff_pa(xformpts(frames, cen[0]),
                                 xformvec(frames, axis[0]),
                                 tgtcen ,
                                 tgtaxis,
                                 lever  ,
                                 directed=True)
    axis = xformvec(frames [np . argmin(dist)], axis)
    cen  = xformpts(frames [np . argmin(dist)], cen )
    return axis, cen

def uniqlastdim(x, tol=1e-4):
    x = np.asarray(x)
    origshape, x = x.shape, x.reshape(-1, x.shape[-1])
    uniq = []
    while len(x):
        uniq.append(x[0])
        different = np.sum((x[0][None] - x) ** 2, axis=-1) > tol * tol
        x = x[different]
    return np.stack(uniq)

def joinlastdim(u, v):
    return np.concatenate([u, v], axis=-1)

def get_dtype_dev(example, dtype=None, **_):
    if isinstance(example, list) and len(example) < 100:
        for e in example:
            if isinstance(e, np.ndarray):
                example = e
                break
    if dtype is None:
        if isinstance(example, np.ndarray): dtype = example.dtype
        else: dtype = th.float32
    return dict(dtype=dtype)

def toint(x):
    if isinstance(x, np.ndarray): return x.round().astype(int)
    return round(x)

def randtrans(*a, rot_sd=None, **kw):
    return hrandsmall(*a, rot_sd=0, **kw)

def tensor_in(x, targets, atol=1e-3):
    atol    = float(atol   )
    targets = list (targets)
    result = -np.ones_like(x, dtype=int)
    for i, t in enumerate(targets):
        result = np.where(np.isclose(x, t, atol=atol), i, result)
    return result

def allclose(a, b, atol=1e-3, **kw):
    if isinstance(a, (np.ndarray, list, int, float)):
        return ipd.kwcall(kw, np.allclose, a, b, atol=atol)
    if isinstance(a, dict) and isinstance(b, dict):
        with ipd.dev.capture_asserts() as result:
            assert a.keys() == b.keys()
            for k in a:
                assert allclose(a[k], b[k])
        return not result
    if (xr := sys.modules.get('xarray')) and isinstance(a, xr.DataArray):
        return ipd.kwcall(kw, np.allclose, a, b, atol=atol)
    if (xr := sys.modules.get('xarray')) and isinstance(a, xr.Dataset):
        if list(a.keys()) != list(b.keys()): return False
        if list(a.coords) != list(b.coords): return False
        for k in list(a.keys()) + list(a.coords):
            return ipd.kwcall(kw, np.allclose, a[k], b[k], atol=atol)
    raise TypeError(f'bad types for allclose {type(a)} {type(b)}')

def xinfo(xforms):
    return ipd.Bunch(zip('axis angle cen, hel'.split(), axis_angle_cen_hel_of(np.asarray(xforms))))

def radius_of_gyration(points, com=None):
    points = hpoint(points)
    com = hcom(points) if com is None else hpoint(com)
    delta2 = hdist2(points, com)
    rg = np.sqrt(np.mean(delta2))
    return rg

# compatibility with thgeom (torch version of these)
axis_angle = axis_angle_of
axis_angle_cen_hel = axis_angle_cen_hel_of
point_line_dist_pa = h_point_line_dist

# aliasing all that start with h
construct  = hconstruct
valid      = hvalid
valid_norm = hvalid_norm
valid44    = hvalid44
scaled     = hscaled
dist       = hdist
diff       = hdiff
xformx     = hxformx
xformpts   = hxformpts
xformvec   = hxformvec
xform      = hxform
inv    = hinv
unique = hunique
rot    = hrot
point  = hpoint
pointorvec = hpointorvec
vec      = hvec
centered = hcentered
centered3 = hcentered3
ray   = hray
frame = hframe
trans = htrans
dot   = hdot
cross = hcross
norm  = hnorm
norm2 = hnorm2
normalized = hnormalized
randpoint  = hrandpoint
randray    = hrandray
randsmall  = hrandsmall
rand    = hrand
randrot = hrandrot
randrotsmall = hrandrotsmall
rms      = hrms
parallel = hparallel
rmsfit   = hrmsfit
proj     = hproj
cart     = hcart
cart3    = hcart3
ori3     = hori3
ori      = hori
projperp = hprojperp
pointlineclose  = hpointlineclose
point_line_dist = h_point_line_dist
linesisect      = hlinesisect
align     = halign
align2    = halign2
coherence = hcoherence
mean      = hmean
expand    = hexpand
pow       = hpow
pow_int   = hpow_int
pow_float = hpow_float
com_flat  = hcom_flat
com       = hcom
rog_flat  = hrog_flat
rog       = hrog
convert   = hconvert
