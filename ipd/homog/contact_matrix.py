r"""
Efficient contact matrix processing for biological structure data.

This module provides the :class:`ContactMatrixStack`, which enables fast spatial
queries over a stack of symmetric residue-residue contact matrices. These matrices
are commonly used to represent pairwise atomic or residue interactions across
multiple structures (e.g., frames in a trajectory or members of an ensemble).

Key features
------------

- Efficient prefix-sum accumulation for fast region-based queries
- Fragment-based contact extraction with adjustable fragsize size and stride
- Subset enumeration and summarization
- Optional PyTorch acceleration for GPU computation

Cumulative sums, 1D basics
--------------------------

ContactMatrixStack uses a precomputed 2D cumsum array for efficient region-based queries. To explain
start with the 1D cumsum case. ``DATA`` is a 1D array and ``SUMS`` is the cumulative sum
of ``DATA`` (``np.cumsum(DATA)``. If we want the sum of ``DATA[i:j]`` we can compute it as
``SUMS[j] - SUMS[i]``.

.. code-block:: python

    Index:          0     1     2     3     4
    DATA:         [ 2 ] [ 7 ] [ 1 ] [ 3 ] [ 5 ]
                  \___________________________/
    Cumulative
    Sum (SUM):   [ 0 ] [ 2 ] [ 9 ] [ 10 ] [ 13 ] [ 18 ]
    Index:         ^     ^     ^      ^      ^      ^
                   0     1     2      3      4      5
                 (think of SUM[4] = sum of DATA[:4])

To find the sum of ``DATA[lb:ub]``, use ``SUM[ub] - SUM[lb]``. ``SUMS[ub]`` will be the sum of
``DATA[:ub]`` and ``SUMS[lb]`` will be the sum of ``DATA[:lb]``. So, ``SUM[ub] - SUM[lb]`` will be
the sum of ``DATA[lb:ub]``. The key observation is that ``SUM[ub]`` is the sum of elements ``DATA[0:ub]``, so it “overcounts” all elements up to ``lb``.    Subtracting ``SUM[lb]`` precisely
removes that overcount, leaving just the slice from ``lb`` to ``ub``.

Cumsum Performance Example
--------------------------
Setup

>>> import numpy as np
>>> data = np.random.rand(500)
>>> sums = np.zeros(len(data)+1)
>>> sums[1:] = np.cumsum(data)  # yes, it's really called that... in torch too.
>>> slice_sums1 = np.zeros((len(data), len(data)))
>>> slice_sums2 = np.zeros((len(data), len(data)))
>>> timer = ipd.dev.Timer('Cumsum Perf Example')

Compute all sums the brute force way

>>> for i in range(len(data)):
...     for j in range(i, len(data)):
...         slice_sums1[i, j] = data[i:j].sum()
>>> _ = timer.checkpoint('dumb way, summing explicitly like a barbarian')

Compute all sums using cumsum

>>> for i in range(len(data)):
...     for j in range(i, len(data)):
...         slice_sums2[i, j] = sums[j] - sums[i]
>>> _ = timer.checkpoint('kinda smarter way using cumsum')

Compute all sums using cumsum without loops

>>> slice_sums3 = np.maximum(0, sums[None, :-1] - sums[:-1, None])
>>> _ = timer.checkpoint('big brain way using cumsum and numpy broadcasting')

Verify results

>>> np.allclose(slice_sums1, slice_sums2)
True
>>> np.allclose(slice_sums1, slice_sums3)
True
>>> # check runtimes, note how much faster cumsum + broadcasting is
>>> timer.report(timecut=0)  # doctest: +SKIP
Timer.report: Cumsum Perf Example order=longest: summary: sum
   0.19970 * dumb way, summing explicitly like a barbarian
   0.03875 * kinda smarter way using cumsum
   0.00078 * big brain way using cumsum and numpy broadcasting

Cumulative sums, 2D
-------------------

Note how much faster the cumsum + broadcasting version was for the 1D version, almost 1000x faster.
It makes an even bigger difference in the 2D case because the arrays tend to be much larger.

.. figure:: ../_static/img/cumsum2d.png
   :scale: 67 %
   :alt: cumsum2d illustration

   Illustration of data 2D with pink region to be "summed" and 2D cumulative sum array from which four points are needed to computs the "sum:" ``sum = CSUM[ub1,ub2] (red point) + CSUM[lb1,lb2] (green point) - CUSM[ub1,lb2] (blue point) - CSUM[lb1,lb2] (blue point``.

The method :py:meth:`ContactMatrixStack.fragment_contact` uses this idea to compute the total contacts of all
pairs of fragments of a given length using a 2D cumsum array. The stride parameter allows for computing only evey Nth value. Note, even on large inputs, this function is fast enough to
compute every fragment pair, so stride is mainly useful as simple way to reduce redundancy.

>>> def fragment_contact(self, fragsize, stride=1):
...   result = (
...     self.cumsum[:, fragsize:         :stride, fragsize:         :stride] -
...     self.cumsum[:, fragsize:         :stride,         :-fragsize:stride] -
...     self.cumsum[:,         :-fragsize:stride, fragsize:         :stride] +
...     self.cumsum[:,         :-fragsize:stride,         :-fragsize:stride] )

This function retuns an ``S x M x N`` array containing the total contacts for all pairs of fragments for each contact matrix s in the stack: ``fragment1`` starting at m ending at ``m + fragsize``, to fragment2 starting at ``n`` and ending at ``n - fragsize``.

The method :py:meth:`ContactMatrixStack.topk_fragment_contact_by_subset_summary` uses the
arrays produced by
:py:meth:`ContactMatrixStack.fragment_contact` to search for subsets of subunits that
all "multibody" contacts by enumerating all subsets of contacting subunits, and taking
the minimum number of contacts for each fragment pair. See the example below.

ContactMatrixStack Example
---------------------------

Setup, reading in and positioning some data

>>> top7 = ipd.atom.body_from_file('1qys').centered
>>> dxh = ipd.atom.symbody_from_file('1dxh').centered
>>> dxh.contacts(top7).total_contacts  # lots, both are centered
3033
>>> top7 = top7.slide_into_contact(dxh, [1, 0, 0])  # just touching
>>> top7 = top7.movedby([15,0,0]) # now way clashing, but lots of contacts
>>> contacts = dxh.contacts(top7, radius=6)

Get best pair of fragment

>>> cmat = contacts.contact_matrix_stack()
>>> cmat
ContactMatrixStack(shape: (4, 92, 335) subs: [ 2  6  8 10])
>>> # 4 contact matrices, thus top7 contacts 4 (of 12) subunit in dxh
>>> pair_frag_contacts = cmat.fragment_contact(fragsize=20, stride=5)
>>> isub, itop7, idxh = np.unravel_index(np.argmax(pair_frag_contacts), pair_frag_contacts.shape)
>>> best_ncontact = pair_frag_contacts[isub, itop7, idxh]
>>> f'best frag pair is top7 resi {itop7}-{itop7+19} to dxh sub {cmat.subs[isub]} resi {idxh}-{idxh+19}'
'best frag pair is top7 resi 4-23 to dxh sub 8 resi 0-19'

Get fragments pairs with multiple subunit contacts

>>> args = dict(fragsize=10, stride=4, k=20, summary=np.min)
>>> bestfrags = cmat.topk_fragment_contact_by_subset_summary(**args)
>>> list(bestfrags.index.keys())
[(0, 2), (0,), (1,), (2,), (3,)]

bestfrags.index and bestfrags.vals are dicts mapping a set of subunits to fragment pairs that have contacts involving all the subunits. The subsets ``(0,), (1,), (2,), (3,)`` contain only one subunit, but there is one subset (0, 2), indicating fragment pairs that contact both subunit 0 and subunit 2.

>>> f'subunits in 1dxh {[int(cmat.subs[i]) for i in (0, 2)]}'
'subunits in 1dxh [2, 8]'
>>> bestfrags.index[0, 2].shape, bestfrags.vals[0,2].shape
((2, 7), (7,))
>>> np.concatenate([bestfrags.index[0, 2].T, bestfrags.vals[0,2][:,None]], axis=1)
array([[ 32, 112,  11],
       [ 28, 112,   6],
       [ 28, 116,   6],
       [ 36,   0,   1],
       [ 32,   0,   1],
       [ 36,   4,   1],
       [ 32,   4,   1]], dtype=int32)

This tells us that top7 resi ``32-51`` has **at least** 11 contacts to *both*
1dxh subunit 2 resi ``112-131`` *and* 1dxh subunit 8 resi ``112-131``. Lets get the atoms
and see if it's legit.

>>> top7frag = top7.positioned_atoms[np.isin(top7.atoms.res_id, range(32, 52))]
>>> dxhfrag1 = dxh.bodies[2].positioned_atoms[np.isin(dxh.bodies[2].atoms.res_id, range(112, 132))]
>>> dxhfrag2 = dxh.bodies[8].positioned_atoms[np.isin(dxh.bodies[8].atoms.res_id, range(112, 132))]
>>> ipd.atom.dump(top7frag, '/tmp/top7frag.cif')
>>> ipd.atom.dump(dxhfrag1, '/tmp/dxhfrag1.cif')
>>> ipd.atom.dump(dxhfrag2, '/tmp/dxhfrag2.cif')
>>> # ipd.showme(top7frag, name='top7', force=True)
>>> # ipd.showme(dxhfrag1, name='dxh1', force=True)
>>> # ipd.showme(dxhfrag2, name='dxh2', force=True)
>>> # ipd.showme(dxh, force=True)
>>> # ipd.showme(top7, force=True)

.. figure:: ../_static/img/contact_matrix_topk_frag__example.png
   :alt: Top7 / 1dxh fragment contact example

   Screenshot from pymol (as launched by ipd.showme). These contacts are't super good, but this
   is a totally arbitrary "dock" of top7 to 1dxh, not a real biological interface. (probably should
   have used a real biological interface for this example...) There may also be slightly better
   fragments if stride is set to 1.

Note: :py:func:`ipd.viz.pymol_viz.showme` (just call ipd.showme) is super useful for visualizing all kinds of things, mainly in pymol.
 It can show AtomArrays, Bodies, Symbodies, homogeneous transforms, stacks of xyz coords, symmetry
 elements, crystal lattices, etc etc. All you need is pymol in your conda environment, and runnable.

API docs
--------
"""

import numpy as np
import ipd

th = ipd.lazyimport('torch')

@ipd.struct
class ContactMatrixStack:
    """
    A stack of contact matrices with efficient region and fragment query operations.

    Attributes:
        contacts (np.ndarray): An array of shape (N, L, L) representing N square contact matrices.
        subs (np.ndarray): Optional metadata describing subsets (not used internally).
        cumsum (np.ndarray): Internal cumulative sum array of shape (N, L+1, L+1).

    Example:
        >>> import numpy as np
        >>> from ipd.homog import ContactMatrixStack
        >>> contacts = np.tril(np.ones((1, 5, 5)))
        >>> cms = ContactMatrixStack(contacts)
        >>> cms.contacts.shape
        (1, 5, 5)
    """

    contacts: np.ndarray
    subs: np.ndarray = None
    cumsum: np.ndarray = ipd.field(lambda: np.empty(0))

    def __post_init__(self):
        if len(self.contacts.shape) == 2: self.contacts = self.contacts[None]
        if self.subs is None: self.subs = np.arange(len(self.contacts))
        assert len(self.contacts.shape) == 3
        # assert self.contacts.shape[2] == self.contacts.shape[1], 'contacts must be square'
        self.update_cumsum()

    def update_cumsum(self):
        """
        Compute and store cumulative sums of each contact matrix for fast region queries.

        This method initializes a zero-padded prefix sum array with shape (N, L+1, L+1),
        where N is the number of matrices and L is the matrix size.

        Example:
            >>> import numpy as np
            >>> from ipd.homog import ContactMatrixStack
            >>> contacts = np.ones((1, 3, 3))
            >>> cms = ContactMatrixStack(contacts)
            >>> cms.cumsum[0, 3, 3]
            np.float64(9.0)
        """
        shape = self.contacts.shape
        self.cumsum = np.zeros((shape[0], shape[1] + 1, shape[2] + 1), dtype=self.contacts.dtype)
        self.cumsum[:, 1:, 1:] = np.cumsum(np.cumsum(self.contacts, axis=1), axis=2)

    def ncontact(self, lb, ub, lb2=None, ub2=None):
        r"""
        Compute total contact counts in rectangular subregions for each matrix.

        Args:
            lb (list of ranges): Lower bounds for rows (inclusive).
            ub (list of ranges): Upper bounds for rows (exclusive).
            lb2 (list of ranges, optional): Lower bounds for columns (defaults to lb).
            ub2 (list of ranges, optional): Upper bounds for columns (defaults to ub).

        Returns:
            np.ndarray: Contact count per region, shape (N, len(lb[0]))

        Example:
            >>> import numpy as np
            >>> contacts = np.ones((1, 20, 20))
            >>> cms = ipd.homog.ContactMatrixStack(contacts)
            >>> cms.ncontact(lb=10, ub=12, lb2=1, ub2=6)
            np.float64(10.0)
            >>> cms.ncontact(lb=[0,3,5], ub=[5,8,10], lb2=[10,13,15], ub2=[15,18,20])
            array([25., 25., 25.])
            >>> cms.ncontact(lb=[range(2, 8)], ub=[range(10, 16)])
            array([[64., 64., 64., 64., 64., 64.]])
            >>> contacts = np.ones((3, 20, 20))  # stack of 3 now
            >>> cms = ipd.homog.ContactMatrixStack(contacts)
            >>> cms.ncontact(lb=[range( 2, 8), range(6)    , range(1, 7)],
            ...              ub=[range(10,16), range(10,16), range(9,15)])
            array([[ 64.,  64.,  64.,  64.,  64.,  64.],
                   [100., 100., 100., 100., 100., 100.],
                   [ 64.,  64.,  64.,  64.,  64.,  64.]])
        """
        if lb2 is None: lb2 = lb
        if ub2 is None: ub2 = ub
        deref1, deref2 = False, False
        if ipd.isint(lb): lb, ub, lb2, ub2, deref1 = [lb], [ub], [lb2], [ub2], True
        if ipd.isint(lb[0]): lb, ub, lb2, ub2, deref2 = [lb], [ub], [lb2], [ub2], True
        if not ipd.homog.all_lte(lb, ub) or not ipd.homog.all_lte(lb2, ub2):
            raise ValueError('lb must be less than or equal to ub')
        idx = np.tile(np.arange(len(lb)), len(lb[0])).reshape(len(lb), len(lb[0]))
        A, B, C, D = (
            self.cumsum[idx, ub, ub2],
            self.cumsum[idx, ub, lb2],
            self.cumsum[idx, lb, ub2],
            self.cumsum[idx, lb, lb2],
        )
        result = A - B - C + D
        if deref2: result = result[0]
        if deref1: result = result[0]
        return result

    @ipd.dev.timed
    def fragment_contact(self, fragsize, stride=1):
        """
        Extract fragment contact maps using a sliding fragsize.

        Args:
            fragsize (int): Fragment size.
            stride (int): Distance between sliding windows.

        Returns:
            np.ndarray: Lower triangular matrix of shape (N, M, M), where
                        M = floor((L - fragsize) / stride) + 1

        Example:
            >>> import numpy as np
            >>> from ipd.homog import ContactMatrixStack
            >>> contacts = np.ones((1, 6, 6))
            >>> cms = ContactMatrixStack(contacts)
            >>> fc = cms.fragment_contact(fragsize=3, stride=1)
            >>> fc.shape
            (1, 4, 4)

        .. seealso::
            ncontact for detail on how the cumsum calculation works.
        """
        result = (self.cumsum[:, fragsize::stride, fragsize::stride] -
                  self.cumsum[:, fragsize::stride, :-fragsize:stride] -
                  self.cumsum[:, :-fragsize:stride, fragsize::stride] +
                  self.cumsum[:, :-fragsize:stride, :-fragsize:stride])
        return result

    @ipd.dev.timed
    def topk_fragment_contact_by_subset_summary(self, fragsize=20, k=13, summary=np.min, stride=1):
        """
        Compute top-k fragment contact values summarized across subsets.

        Args:
            fragsize (int): Fragment size.
            k (int): Number of top values to return.
            summary (Callable): Summary function applied across subset (e.g., np.min).
            stride (int): Stride for fragment sampling.

        Returns:
            ipd.Bunch: Dictionary-like object with keys `vals` and `index` for each subset.

        Example:
            >>> import numpy as np
            >>> from ipd.homog import ContactMatrixStack
            >>> contacts = np.random.rand(3, 50, 50)
            >>> cms = ContactMatrixStack(contacts)
            >>> result = cms.topk_fragment_contact_by_subset_summary(fragsize=10, k=5, stride=5)
            >>> isinstance(result.index, dict)
            True
        """
        result = ipd.Bunch(index=dict(), vals=dict())
        nwindow = self.fragment_contact(fragsize, stride)
        for i, sub in ipd.dev.subsetenum(range(len(self))):
            if not sub: continue
            vals = summary(nwindow[list(sub)], axis=0)
            idx = np.argsort((-vals).flat)[:k]
            idx = np.unravel_index(idx, vals.shape)
            result.vals[sub] = vals[idx]
            result.index[sub] = np.array(idx, dtype=np.int32) * stride
            result.index[sub] = result.index[sub][:, vals[idx] > 0]
            result.vals[sub] = result.vals[sub][vals[idx] > 0]
            if result.index[sub].size == 0:
                del result.index[sub]
                del result.vals[sub]
        return result

    def fragment_contact_torch(self, fragsize):
        cumsum = th.tensor(self.cumsum, device='cuda')
        result = (cumsum[:, fragsize:, fragsize:] - cumsum[:, fragsize:, :-fragsize] -
                  cumsum[:, :-fragsize, fragsize:] + cumsum[:, :-fragsize, :-fragsize])
        return th.tril(result)

    def topk_fragment_contact_by_subset_summary_torch(self, fragsize=20, k=13, summary=None):
        summary = summary or th.min
        result = ipd.Bunch(index=dict(), vals=dict())
        nwindow = self.fragment_contact_torch(fragsize)
        for i, sub in ipd.dev.subsetenum(range(len(self))):
            if not sub: continue
            vals = summary(nwindow[list(sub)].reshape(len(sub), -1), axis=0)
            vals = vals.values.reshape(nwindow.shape[1:])
            topk = th.topk(vals.flatten(), k)
            result.index[sub] = np.unravel_index(topk.indices.cpu().numpy(), vals.shape)
            result.vals[sub] = topk.values
            result.index[sub] = np.array(result.index[sub])
        return result

    def __len__(self):
        """
        Return the number of contact matrices.

        Returns:
            int: Number of matrices in the stack.

        Example:
            >>> import numpy as np
            >>> from ipd.homog import ContactMatrixStack
            >>> cms = ContactMatrixStack(np.ones((3, 10, 10)))
            >>> len(cms)
            3
        """
        return len(self.contacts)

    def __repr__(self):
        """
        Human-readable string representation.

        Returns:
            str: Summary of matrix shape and optional subsets.
        """
        return f'ContactMatrixStack(shape: {self.contacts.shape} subs: {self.subs})'

def is_contact_matrix(arg):
    """
    Check if an object is a valid 3D contact matrix stack.

    Args:
        arg (Any): Object to check.

    Returns:
        bool: True if it's a 3D square matrix with shape (N, L, L).

    Example:
        >>> import numpy as np
        >>> from ipd.homog import is_contact_matrix
        >>> is_contact_matrix(np.random.rand(2, 5, 5))
        True
    """
    if not isinstance(arg, np.ndarray): return False
    if len(arg.shape) != 3: return False
    if arg.shape[1] != arg.shape[2]: return False
    if arg.shape[2] < 5: return False
    return True

def rand_contacts(n, m=1, frac=0.2, cen=5, std=3, index_bias=0.0):
    """
    AI slop, very slow. Create a stack of m random symmetric contact matrices of size n x n.

    smoothed noise field that biases the contact probability in adjacent cells.
    For each off-diagonal cell (i,j), an effective probability is:
    p_eff = clip(frac + index_bias * noise[i, j], 0, 1)
    where the noise field is generated and smoothed for spatial correlation.

    Parameters:
        n (int): Size of each matrix (n x n).
        m (int): Number of matrices to generate.
        frac (float): Base probability for an off-diagonal contact to be nonzero.
        cen (float): Mean of the normal distribution for nonzero contacts.
        std (float): Standard deviation of the normal distribution for nonzero contacts.
        index_bias (float): A multiplier for a smoothed noise

    Returns:
        np.ndarray: A stack of m symmetric contact matrices with shape (m, n, n).
    """
    matrices = np.zeros((m, n, n))

    for k in range(m):
        # Generate a noise field in the range [-0.5, 0.5]
        noise = np.random.rand(n, n) - 0.5
        # Smooth the noise with a simple averaging of immediate neighbors
        noise_smoothed = (noise + np.roll(noise,
                                          (-2, -1, 1, 2, -2, -1, 1, 2), axis=(0, 0, 0, 0, 1, 1, 1, 1))) / 9.0
        # Symmetrize the noise field to ensure the effective probabilities are symmetric
        noise_sym = (noise_smoothed + noise_smoothed.T) / 2.0

        # Initialize the matrix for this iteration.
        mat = np.zeros((n, n))
        # Loop over the upper triangle (excluding the diagonal)
        for i in range(n):
            for j in range(i + 1, n):
                # Compute the effective probability (clip between 0 and 1)
                effective_prob = np.clip(frac + index_bias * noise_sym[i, j], 0, 1)
                if np.random.rand() < effective_prob:
                    value = np.random.normal(cen, std)
                    mat[i, j] = value
                    mat[j, i] = value  # enforce symmetry
        # Ensure the diagonal remains 0
        np.fill_diagonal(mat, 0)
        matrices[k] = mat

    return matrices
