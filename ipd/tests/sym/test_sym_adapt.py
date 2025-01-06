import pytest

pytest.importorskip('torch')
import ipd
from ipd.lazy_import import lazyimport
from ipd.sym.sym_adapt import _sym_adapt

th = lazyimport('torch')

from functools import partial

import numpy as np
import torch  # type: ignore

def main():
    test_dim_rearrange_basic()
    test_dim_rearrange_xyz()
    test_dim_rearrange_errors()
    test_sym_numpy()
    test_sym_adapt_tensor_1d()
    test_sym_adapt_tensor_2d()
    test_sym_adapt_tensor_2d_ambig()
    test_sym_adapt_tensor_3d()
    test_apply_symmetry_dispatch()

@pytest.mark.fast
def test_dim_rearrange_basic():
    for shape, keydim in [
        [(1, 1, 1, 2, 3, 4, 5), 2],
        [(1), 1],
        [(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 1],
        [(4, 7), 7],
        [(4, 5, 6, 7), 7],
        [(4, 7, 5, 6), 7],
    ]:
        t = th.arange(th.prod(th.tensor(shape))).reshape(shape)
        assert not ipd.sym.sym_adapt.tensor_is_xyz(t)
        t2, undo = ipd.sym.sym_adapt.tensor_keydims_to_front(t, keydim)
        assert t2.shape[0] == keydim
        t3 = ipd.sym.sym_adapt.tensor_undo_perm(t2, undo)
        if len(undo) == 1: assert t.data_ptr() == t3.data_ptr()
        assert th.all(t == t3)

@pytest.mark.fast
def test_dim_rearrange_xyz():
    for shape, keydim in [
        [(1, 9, 3), 9],
        [(1, 9, 1, 3), 9],
        [(12, 1, 3), 12],
    ]:
        t = th.arange(th.prod(th.tensor(shape)), dtype=float).reshape(shape)
        assert ipd.sym.sym_adapt.tensor_is_xyz(t)
        t2, undo = ipd.sym.sym_adapt.tensor_keydims_to_front(t, keydim)
        assert t2.shape[0] == keydim
        t3 = ipd.sym.sym_adapt.tensor_undo_perm(t2, undo)
        assert t.data_ptr() == t3.data_ptr()
        assert th.all(t == t3)

@pytest.mark.fast
def test_dim_rearrange_errors():
    for shape, keydim in [
        [(1, 1, 1, 2, 3, 4, 5), 9],
        [(1), 3],
    ]:
        t = th.arange(th.prod(th.tensor(shape))).reshape(shape)
        with pytest.raises(ValueError):
            t2, undo = ipd.sym.sym_adapt.tensor_keydims_to_front(t, keydim)

@pytest.mark.fast
def test_sym_adapt_tensor_1d():
    sym = ipd.sym.create_sym_manager(symid='c1', L=7)
    S = partial(_sym_adapt, sym=sym, isasym=None)
    S(th.randn(7, 3)).adapted[0].shape == (7, 1, 3)  # type: ignore
    S(th.randn(1, 1, 1, 7, 3, 4, 5, 6)).adapted[0].shape == (7, 60, 6)

    sym = ipd.sym.create_sym_manager(symid='c2')
    sym.idx = ipd.sym.SymIndex(sym.nsub, [(10, 0, 6)])

    sym.sym_adapt(th.randn(10, 3)).adapted.shape == (10, 1, 3)
    x = sym.sym_adapt(th.randn(7, 3))
    # ic(x.adapted.shape)
    assert x.adapted.shape == (10, 1, 3)
    assert x.reconstruct(x.adapted).shape == (10, 3)
    x = sym.sym_adapt(th.randn(1, 1, 7, 3, 2, 1))
    xcanon = x.adapted
    assert xcanon.shape == (10, 3, 2, 1, 1, 1)
    assert torch.all(xcanon[6:].cpu() == x.orig[0, 0, 3:].reshape(4, 3, 2, 1, 1, 1))
    assert x.reconstruct(xcanon).shape == (1, 1, 10, 3, 2, 1)

    xcanon = x.adapted
    result = x.reconstruct(xcanon)
    assert xcanon.shape == (10, 3, 2, 1, 1, 1)
    assert result.shape == (1, 1, 10, 3, 2, 1)
    assert torch.all(result[0, 0, 6:].cpu() == x.orig[0, 0, 3:])

@pytest.mark.fast
def test_sym_adapt_tensor_2d():
    sym = ipd.sym.create_sym_manager(symid='c2')
    sym.idx = ipd.sym.SymIndex(sym.nsub, [(10, 0, 6)])
    x = sym.sym_adapt(th.randn(1, 1, 7, 7, 3, 2, 1))
    xcanon = x.adapted
    result = x.reconstruct(xcanon)
    assert xcanon.shape == (10, 10, 3, 2, 1, 1, 1)
    assert result.shape == (1, 1, 10, 10, 3, 2, 1)
    assert torch.all(result[0, 0, :3, :3].cpu() == x.orig[0, 0, :3, :3])
    assert torch.all(result[0, 0, 6:, 6:].cpu() == x.orig[0, 0, 3:, 3:])

@pytest.mark.fast
def test_sym_adapt_tensor_2d_ambig():
    sym = ipd.sym.create_sym_manager(symid='c2')
    sym.idx = [(20, 0, 10)]
    x = sym.sym_adapt(th.randn(20, 20, 7))
    assert x.adapted.shape == (20, 20, 7)
    x = sym.sym_adapt(th.randn(20, 20, 10))
    assert x.adapted.shape == (20, 20, 10)

@pytest.mark.fast
def test_sym_adapt_tensor_3d():
    sym = ipd.sym.create_sym_manager(symid='c2')
    sym.idx = [(10, 0, 6)]
    # sym.sym_adapt(th.randn(10, 3)).adapted[0].shape == (10, 3)
    x = sym.sym_adapt(th.randn(1, 1, 1, 10, 2, 4, 3))
    # ic(x.adapted.shape)
    assert x.adapted.shape == (10, 2, 4, 3, 1, 1, 1)

def _dispatch_symfunc_on_type_shape(*a, **kw):
    """Take in args and decides what kind of symmetry func to use sortof a poor
    mans doubledispatch."""
    kw = ipd.dev.Bunch(kw, _strict=False)  # type: ignore
    if len(a) == 3:
        raise ValueError('apply_symmetry only accepts <= 2 args')
    if len(a) == 2:
        return ''
    if len(a) == 1:
        val = a[0]
        ext = '_' + val.__class__.__name__.split('.')[-1]
        if ext in ('_Tensor', '_ndarray'):
            if ext == '_Tensor' and not torch.is_floating_point(val): return '_seq'
            if ext == '_ndarray' and not np.issubdtype(val.dtype, float): return '_seq'
            if val.ndim >= 3: return ''
            if val.ndim == 2 and val.shape[1] in (3, 4): return ''
            if val.ndim == 2 and val.shape[1] not in (3, 4): return '_seq'
            if val.ndim == 1: return '_seq'
            return '_seq'
        else:
            return ext
    types = 'banana seq xyz pair'.split()
    if len(a) == 0:
        if kw.xyz is not None or kw.pair is not None:
            for t in types:
                if t in 'xyz pair'.split(): continue
                assert t not in kw
            return ''
        if kw.seq is not None:
            for t in types:
                if t == 'seq': continue
                print(t, kw, t not in kw)
                assert not kw[t]
            return '_seq'
        if kw.banana is not None:
            for t in types:
                if t == 'banana': continue
                assert not kw[t]
            return 'Banana'

@pytest.mark.fast
def test_apply_symmetry_dispatch():
    D = _dispatch_symfunc_on_type_shape
    xyz = torch.arange(10 * 4 * 3 * 3).reshape(1, -1, 3, 3).to(torch.float32)
    assert D(xyz) == ''
    assert D(xyz[0]) == ''
    assert D(xyz.reshape(-1)) == '_seq'
    assert D(xyz.reshape(-1, 1)) == '_seq'
    assert D(xyz.reshape(-1, 2)) == '_seq'
    assert D(xyz.reshape(-1, 3)) == ''
    assert D(xyz.reshape(-1, 4)) == ''
    xyz = xyz.to(torch.int16)
    assert D(xyz.reshape(-1, 1)) == '_seq'
    assert D(xyz.reshape(-1, 2)) == '_seq'
    assert D(xyz.reshape(-1, 3)) == '_seq'
    assert D(xyz.reshape(-1, 12)) == '_seq'

    assert D(xyz=8) == ''
    assert D(pair=8) == ''
    assert D(xyz=8, pair=8) == ''
    assert D(seq='arst') == '_seq'
    assert D(banana='arst') == 'Banana'
    assert D(xyz=8, foo=7) == ''
    with pytest.raises(AssertionError):
        D(xyz=8, seq=7)
    with pytest.raises(AssertionError):
        D(xyz=8, banana=7)
    with pytest.raises(AssertionError):
        D(seq=8, banana=3)

    class Foo:
        pass

    assert D(Foo()) == '_Foo'

@pytest.mark.fast
def test_sym_numpy():
    sym = ipd.tests.sym.create_test_sym_manager(['sym.symid=c2'])
    a = np.array(['foo', 'bar', '', '', '', '', '', '', ''], dtype='<U3')
    sym.idx = [len(a) * 2]
    assert np.all(sym(a) == np.concatenate([a, a]))
    a = np.arange(20)
    sym.idx = [len(a) * 2]
    assert np.all(sym(a) == np.concatenate([a, a]))

if __name__ == '__main__':
    main()
