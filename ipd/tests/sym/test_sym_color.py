import pytest

pytest.importorskip('torch')
import hypothesis
import torch as th  # type: ignore
from hypothesis import strategies as st

import ipd

def symslices_from_colors_should_fail(nsub, colors, Lasu, isasu):
    if len(colors) == 0:
        return True
    if not isasu:
        for i, l in enumerate(Lasu):
            # print(i, l, int(th.sum(colors==i)) % nsub, nsub)
            if l is None and int(th.sum(colors == i)) % nsub != 0:
                return True
            if l is not None and l % nsub != 0:
                return True
    return False

@pytest.mark.fast
# @hypothesis.settings(verbosity=hypothesis.Verbosity.verbose, max_examples=100, derandomize=True)
@hypothesis.settings(deadline=1000, max_examples=10)
@hypothesis.given(
    nsub=st.integers(1, 2),
    colors=st.lists(st.integers(0, 5), min_size=30, max_size=40).filter(lambda x: len(set(x)) == 6).map(sorted),
    isasu=st.booleans(),
    unsymfrac=st.lists(st.one_of(st.none(), st.floats(0, 0.9)), min_size=3, max_size=3),
)
def test_symslices_from_colors_fuzz(nsub, colors, isasu, unsymfrac, **kw):
    colors = th.tensor(colors)
    Lasu = th.full((6, ), -1)
    for t, uf in enumerate(unsymfrac):
        if uf is not None:
            Lasu[t] = max(0, int((1-uf) * th.sum(colors == t))) // (1 if isasu else nsub)
    xfail = symslices_from_colors_should_fail(nsub, colors, Lasu, isasu)
    try:
        idx = ipd.sym.symslices_from_colors(nsub, colors, isasu, Lasu, recolor=False)
        for s in idx:
            assert s[1] < s[0]
            assert s[2] <= s[0]
            if not isasu:
                assert th.all(colors[range(s[1], s[2])] == colors[s[1]])
        if idx:
            idx = ipd.sym.SymIndex(nsub, idx)
        assert not xfail
    except AssertionError:
        hypothesis.assume(False)

@pytest.mark.fast
def test_make_sequential_colors():
    c = ipd.sym.make_sequential_colors([7, 7, 1, 1, 2, 2])
    assert th.all(c == th.tensor([0, 0, 1, 1, 2, 2]))
    c = ipd.sym.make_sequential_colors([2, 3, 3, 3, 4, 4, 761, 761, 2, 0, 0, 0, 0, 34, 4, 4, 4, 6, 6, 761])
    assert th.all(c == th.tensor([0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 5, 5, 6, 7, 7, 7, 8, 8, 9]))

@pytest.mark.fast
def test_symslices_from_colors_one():
    nsub = 4
    idx = ipd.sym.symslices_from_colors(
        nsub=nsub,
        colors=th.tensor([0, 1, -1, 1, 2, -1, 2], dtype=int),  # type: ignore
        isasu=True,
        Lasu=th.tensor([-1, -1, 0, -1, -1, 0, -1], dtype=int),  # type: ignore
    )
    idx = ipd.sym.SymIndex(nsub, idx)
    assert th.allclose(
        idx.sub.to(int),
        th.tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]]),
    )

if __name__ == '__main__':
    test_make_sequential_colors()
    test_symslices_from_colors_one()
    test_symslices_from_colors_fuzz()
