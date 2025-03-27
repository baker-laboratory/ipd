import unittest
import pytest
import numpy as np

import ipd
from ipd.homog.contact_matrix import ContactMatrixStack

config_test = ipd.Bunch(
    re_only=[
        #
    ],
    re_exclude=[
        #
    ],
)

def main():
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
    )

def matrix_cumsum(cumsum, lb, ub, lb2, ub2):
    A, B, C, D = (
        cumsum[:, ub, ub2],
        cumsum[:, ub, lb2],
        cumsum[:, lb, ub2],
        cumsum[:, lb, lb2],
    )
    return A - B - C + D

def test_contacts_cumsum():
    # contacts = ipd.homog.rand_contacts(20, m=1, frac=0.3, cen=5.0, std=4.0, index_bias=0.5)
    contacts = np.random.rand(3, 20, 20)
    contacts += contacts.swapaxes(1, 2)
    mat = ipd.homog.ContactMatrixStack(contacts)
    assert np.allclose(mat.contacts[:, :4, :4].sum((1, 2)), mat.cumsum[:, 4, 4])
    assert np.allclose(mat.contacts[:, :8, :12].sum((1, 2)), mat.cumsum[:, 8, 12])

    lb, ub, lb2, ub2 = 1, 7, 8, 14
    test = matrix_cumsum(mat.cumsum, lb, ub, lb2, ub2)
    ref = mat.contacts[:, lb:ub, lb2:ub2].sum(axis=(1, 2))
    assert np.allclose(test, ref)

def test_ncontact():
    contacts = np.ones((1, 20, 20))
    cms = ipd.homog.ContactMatrixStack(contacts)
    cms.ncontact(lb=10, ub=12, lb2=1, ub2=6)
    np.float64(10.0)
    v = cms.ncontact(lb=[0, 3, 5], ub=[5, 8, 10], lb2=[10, 13, 15], ub2=[15, 18, 20])
    assert np.allclose(v, np.array([25., 25., 25.]))
    with pytest.raises(ValueError):
        v = cms.ncontact(lb=[range(10, 16)], ub=[range(2, 8)])
    v = cms.ncontact(lb=[range(0, 6)], ub=[range(2, 8)])
    assert np.allclose(v, np.array([[4., 4., 4., 4., 4., 4.]]))
    contacts = np.ones((3, 20, 20))  # stack of 3 now
    cms = ipd.homog.ContactMatrixStack(contacts)
    v = cms.ncontact(
        lb=[range(2, 8), range(6), range(1, 7)],
        ub=[range(10, 16), range(10, 16), range(9, 15)],
    )
    assert np.allclose(
        v,
        np.array([[64., 64., 64., 64., 64., 64.], [100., 100., 100., 100., 100., 100.],
                  [64., 64., 64., 64., 64., 64.]]))

def test_contacts_rand_m1():
    contacts = ipd.homog.rand_contacts(20, m=1, frac=0.3, cen=5.0, std=4.0, index_bias=0.5)
    mat = ipd.homog.ContactMatrixStack(contacts)
    lb = [range(0, 5)]
    # lb2 = [range(5, 10)]
    ub = [range(10, 15)]
    # ub2 = [range(15, 20)]
    count1 = mat.ncontact(lb, ub)
    for i in range(len(mat)):
        for j, l, u in ipd.dev.zipenum(lb[i], ub[i]):
            ic(mat.contacts[i, l:u, l:u].sum(), count1[i, j])
            assert np.isclose(mat.contacts[i, l:u, l:u].sum(), count1[i, j])

def test_contacts_allone_m1():
    # contacts = ipd.homog.rand_contacts(20, m=1, frac=0.3, cen=5.0, std=4.0, index_bias=0.5)
    contacts = np.ones((1, 20, 20))
    mat = ipd.homog.ContactMatrixStack(contacts)
    lb = [range(0, 5)]
    lb2 = [range(5, 10)]
    ub = [range(10, 15)]
    ub2 = [range(15, 20)]
    count1 = mat.ncontact(lb, ub, lb2, ub2)
    for i in range(len(mat)):
        for j, l, u, l2, u2 in ipd.dev.zipenum(lb[i], ub[i], lb2[i], ub2[i]):
            ic(mat.contacts[i, l:u, l2:u2].sum(), count1[i, j])
            assert np.isclose(mat.contacts[i, l:u, l2:u2].sum(), count1[i, j])

def test_contacts_rand_m3():
    contacts = ipd.homog.rand_contacts(20, m=1, frac=0.3, cen=5.0, std=4.0, index_bias=0.5)
    mat = ipd.homog.ContactMatrixStack(contacts)
    lb = [range(0, 5)] * len(mat)
    lb2 = [range(5, 10)] * len(mat)
    ub = [range(10, 15)] * len(mat)
    ub2 = [range(15, 20)] * len(mat)
    count1 = mat.ncontact(lb, ub, lb2, ub2)
    for i in range(len(mat)):
        for j, l, u, l2, u2 in ipd.dev.zipenum(lb[i], ub[i], lb2[i], ub2[i]):
            assert np.isclose(mat.contacts[i, l:u, l2:u2].sum(), count1[i, j])

def test_fragment_contact():
    # np.random.seed(0)
    contacts = ipd.homog.rand_contacts(100, m=3, frac=0.3, cen=5.0, std=4.0, index_bias=0.5)
    mat = ipd.homog.ContactMatrixStack(contacts)
    nfragsize = mat.fragment_contact(20)
    mins = [np.unravel_index(np.argsort((-nw).flat)[:10], nw.shape) for nw in nfragsize]
    for i, mn in enumerate(mins):
        prev = 9e9
        for s1, s2 in zip(*mn):
            tot = mat.contacts[i, s1:s1 + 20, s2:s2 + 20].sum()
            assert tot <= prev + 0.001
            prev = tot

def test_topk_fragment_contact_by_subset_summary():
    with ipd.dev.temporary_random_seed(0):
        contacts = np.random.rand(4, 1000, 1000) * 2
        mat = ipd.homog.ContactMatrixStack(contacts.astype(np.int32))
        topk = mat.topk_fragment_contact_by_subset_summary(fragsize=20, k=13, stride=4)
        assert np.all(topk.vals[(1, )] >= topk.vals[1, 2])
        assert np.all(topk.vals[1, 3] >= topk.vals[1, 2, 3])

# ------------------ Test Suite ------------------

def test_fragment_contact_sparse():
    contacts = np.zeros((1, 11, 11), dtype=int)
    contacts[0, 5, 5] = 1
    # ic(contacts)
    stack = ContactMatrixStack(contacts)
    assert np.all(stack.fragment_contact(1) == contacts)
    assert stack.fragment_contact(2).sum() == 4
    assert stack.fragment_contact(3).sum() == 9
    assert stack.fragment_contact(4).sum() == 16
    assert stack.fragment_contact(5).sum() == 25

class TestRandomContactMatrices(unittest.TestCase):
    """ai slop"""

    def setUp(self):
        # Use a fixed seed for reproducibility during testing.
        np.random.seed(42)
        self.n = 10
        self.m = 5
        self.fraction = 0.3
        self.center = 5.0
        self.std = 1.0

    def test_output_shape(self):
        mats = ipd.homog.rand_contacts(self.n, self.m, self.fraction, self.center, self.std, index_bias=0.0)
        self.assertEqual(mats.shape, (self.m, self.n, self.n))

    def test_symmetry(self):
        mats = ipd.homog.rand_contacts(self.n, self.m, self.fraction, self.center, self.std, index_bias=0.0)
        for mat in mats:
            self.assertTrue(np.allclose(mat, mat.T), "Matrix is not symmetric")

    def test_diagonal_zero(self):
        mats = ipd.homog.rand_contacts(self.n, self.m, self.fraction, self.center, self.std, index_bias=0.0)
        for mat in mats:
            self.assertTrue(np.all(np.diag(mat) == 0), "Diagonal entries are not zero")

    def test_zero_fraction(self):
        # With fraction = 0, there should be no contacts (all entries should be 0).
        mats = ipd.homog.rand_contacts(self.n, self.m, 0.0, self.center, self.std, index_bias=0.0)
        for mat in mats:
            self.assertTrue(np.all(mat == 0), "Matrix is not all zeros when fraction is 0")

    def test_full_fraction_no_bias(self):
        # With fraction = 1 and no index_bias, every off-diagonal entry should have a contact.
        mats = ipd.homog.rand_contacts(self.n, self.m, 1.0, self.center, self.std, index_bias=0.0)
        for mat in mats:
            # Get the off-diagonal entries (upper triangle) and ensure none are zero.
            off_diag = mat[np.triu_indices(self.n, k=1)]
            # It is highly unlikely that a sample from a normal distribution equals exactly 0.
            self.assertTrue(np.all(off_diag != 0), "Not all off-diagonals are nonzero when fraction is 1")

    def test_index_bias_effect(self):
        # Generate matrices with and without bias and compare their overall sums.
        mats_no_bias = ipd.homog.rand_contacts(self.n,
                                               self.m,
                                               self.fraction,
                                               self.center,
                                               self.std,
                                               index_bias=0.0)
        mats_with_bias = ipd.homog.rand_contacts(self.n,
                                                 self.m,
                                                 self.fraction,
                                                 self.center,
                                                 self.std,
                                                 index_bias=0.5)
        # Compute the sum of each matrix.
        sums_no_bias = np.array([np.sum(mat) for mat in mats_no_bias])
        sums_with_bias = np.array([np.sum(mat) for mat in mats_with_bias])
        # While randomness is involved, statistically the overall sums should differ when bias is applied.
        self.assertFalse(np.allclose(sums_no_bias, sums_with_bias),
                         "Index bias parameter did not affect the matrices as expected")

if __name__ == "__main__":
    main()
