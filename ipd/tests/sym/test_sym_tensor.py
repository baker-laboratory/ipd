import pytest

pytest.importorskip('torch')
import copy

from icecream import ic

from ipd.dev.lazy_import import lazyimport

th = lazyimport('torch')

import ipd
import ipd.sym.sym_tensor as st

ic.configureOutput(includeContext=False, contextAbsPath=True)

@pytest.mark.fast
def test_sym_tensor_copy():
    sym = ipd.tests.sym.create_test_sym_manager(symid='C2')
    sym.idx = [(5, 0, 4)]
    sym.idx.set_kind(th.zeros(5).to(int))  # type: ignore

    t = sym.tensor(th.arange(5))
    assert t.attr.sym is copy.deepcopy(t).attr.sym
    assert t.attr.sym is t.clone().attr.sym
    assert t.attr.sym is th.as_tensor(t).attr.sym
    assert t.attr.sym is sym.sym_adapt(t).new.attr.sym
    # ic(sym.sym_adapt(t).new.reshape(5))
    assert t.attr.sym is sym.sym_adapt(t).new.reshape(5).attr.sym

    assert len(t.full) == 5
    assert len(t.sym) == 4
    assert len(t.asym) == 3
    assert len(t.asu) == 2
    assert list(t.unsym) == [4]
    # with pytest.raises(st.SymTensorError):
    # t.asu.unsym

@pytest.mark.fast
def test_sym_tensor_sort_typemap_dense_1D():
    sym = ipd.tests.sym.create_test_sym_manager(symid='C3')
    sym.idx = [(10, 0, 3), (10, 5, 8)]
    sym.idx.set_kind(th.tensor([0, 0, 0, 10, 1, 2, 2, 2, 2, 12], dtype=int))  # type: ignore

    for type1 in st.subbasenames[st.Ordering]:
        cls = st.sym_tensor_types[f'Full{type1}All1DBasic']
        s = sym.tensor(th.arange(sym.idx.L), cls=cls)  # type: ignore

        assert hasattr(s, 'attr')
        assert isinstance(s, cls)
        assert isinstance(s, getattr(st, type1))
        assert th.all(s == s.attr.orig[getattr(sym.idx, type1.lower())])
        for type2 in st.subbasenames[st.Ordering]:
            type2l = type2.lower()
            t = getattr(s, type2l)
            assert hasattr(t, 'attr')
            assert t.__class__.__name__ == cls.__name__.replace(type1, type2), cls.__name__
            assert isinstance(t, getattr(st, type2))
            assert th.all(t == t.attr.orig[getattr(sym.idx, type2l)])
            orig = t.clone()
            new = th.randint(100, (len(orig), ))
            with s.sym_updates_off():
                setattr(s, type2l, new)
                assert th.all(getattr(s, type2l) == new)
                setattr(s, type2l, orig)

@pytest.mark.fast
def test_sym_tensor_res_typemap_dense_1D():
    sym = ipd.tests.sym.create_test_sym_manager(symid='C3')
    sym.idx = [(10, 0, 3), (10, 5, 8)]
    sym.idx.set_kind(th.tensor([0, 0, 0, 10, 1, 2, 2, 2, 2, 12], dtype=int))  # type: ignore

    for type1 in st.subbasenames[st.ChemType]:
        cls = st.sym_tensor_types[f'FullSliced{type1}1DBasic']
        s = sym.tensor(th.arange(sym.idx.L), cls=cls)  # type: ignore

        assert hasattr(s, 'attr')
        assert isinstance(s, cls)
        assert isinstance(s, getattr(st, type1))
        assert th.all(s == s.attr.orig[getattr(sym.idx, type1.lower())])
        for type2 in st.subbasenames[st.ChemType]:
            type2l = type2.lower()
            t = getattr(s, type2l)
            assert hasattr(t, 'attr')
            assert t.__class__.__name__ == cls.__name__.replace(type1, type2), cls.__name__
            assert isinstance(t, getattr(st, type2))
            assert th.all(t == t.attr.orig[getattr(sym.idx, type2l)])
            orig = t.clone()
            new = th.randint(100, (len(orig), ))
            with s.sym_updates_off():
                setattr(s, type2l, new)
                assert th.all(getattr(s, type2l) == new)
                setattr(s, type2l, orig)

@pytest.mark.fast
def test_sym_tensor_sym_typemap_dense_1D():
    sym = ipd.tests.sym.create_test_sym_manager(symid='C3')
    sym.idx = [(10, 0, 3), (10, 5, 8)]
    sym.idx.set_kind(th.tensor([0, 0, 0, 10, 1, 2, 2, 2, 2, 12], dtype=int))  # type: ignore

    for type1 in st.subbasenames[st.SymSub]:
        if type1 in ('Sub', 'NotSub'): continue
        cls = st.sym_tensor_types[f'{type1}SlicedAll1DBasic']
        s = sym.tensor(th.arange(sym.idx.L), cls=cls)  # type: ignore

        assert hasattr(s, 'attr')
        assert isinstance(s, cls)
        assert isinstance(s, getattr(st, type1))
        assert th.all(s == s.attr.orig[getattr(sym.idx, type1.lower())])
        for type2 in st.subbasenames[st.SymSub]:
            type2l = type2.lower()
            if type2 in ('Sub', 'NotSub'): continue
            t = getattr(s, type2l)
            assert hasattr(t, 'attr')
            assert t.__class__.__name__ == cls.__name__.replace(type1, type2), cls.__name__
            assert isinstance(t, getattr(st, type2))
            assert th.all(t == t.attr.orig[getattr(sym.idx, type2l)])
            orig = t.clone()
            new = th.randint(100, (len(orig), ))
            with s.sym_updates_off():
                setattr(s, type2l, new)
                assert th.all(getattr(s, type2l) == new)
                setattr(s, type2l, orig)
        for i in range(sym.nsub):
            t = s.sub(i)
            assert isinstance(t, st.Sub)  # type: ignore

            assert t.__class__.__name__ == cls.__name__.replace(type1, 'Sub'), cls.__name__
            assert th.all(t == t.attr.orig[getattr(sym.idx, 'sub')[i]])
            t = s.notsub(i)
            assert isinstance(t, st.NotSub)  # type: ignore

            assert t.__class__.__name__ == cls.__name__.replace(type1, 'NotSub'), cls.__name__
            assert th.all(t == t.attr.orig[getattr(sym.idx, 'notsub')[i]])
        with pytest.raises(st.SymTensorError):
            s.sub(sym.nsub + 1)

@pytest.mark.fast
def test_sym_tensor_2d():
    sym = ipd.tests.sym.create_test_sym_manager(symid='C3')
    sym.idx = [(10, 0, 3), (10, 5, 8)]
    sym.idx.set_kind(th.tensor([0, 0, 0, 10, 1, 2, 2, 2, 2, 12], dtype=int))  # type: ignore

    for type1 in st.subbasenames[st.SymSub]:
        if type1 in ('Sub', 'NotSub'): continue
        cls = st.sym_tensor_types[f'{type1}SlicedAll2DBasic']
        s = sym.tensor(th.arange(sym.idx.L**2).reshape(sym.idx.L, sym.idx.L), cls=cls)  # type: ignore

        assert hasattr(s, 'attr')
        assert isinstance(s, cls)
        assert isinstance(s, getattr(st, type1))
        assert th.all(s == sym.idx.slice2d(s.attr.orig, type1.lower(), dim=[0, 1]))  # type: ignore

        for type2 in st.subbasenames[st.SymSub]:
            type2l = type2.lower()
            if type2 in ('Sub', 'NotSub'): continue
            t = getattr(s, type2l)
            assert hasattr(t, 'attr')
            assert t.__class__.__name__ == cls.__name__.replace(type1, type2), cls.__name__
            assert isinstance(t, getattr(st, type2))
            # ic( t , sym.idx.slice2d(t.attr.orig, type2l, dim=[0,1]))
            assert th.all(t == sym.idx.slice2d(t.attr.orig, type2l, dim=[0, 1]))  # type: ignore

            orig = t.clone()
            new = th.randint(10000, 20000, (len(orig), len(orig)))
            setattr(s, type2l, new)
            # ic(s)
            assert th.all(getattr(s, type2l) == new)
            setattr(s, type2l, orig)

@pytest.mark.fast
def test_sym_tensor():
    sym = ipd.tests.sym.create_test_sym_manager(symid='C3')
    sym.idx = [(10, 0, 3), (10, 5, 8)]
    sym.idx.set_kind(th.tensor([0, 0, 0, 10, 1, 2, 2, 2, 2, 12], dtype=int))  # type: ignore

    t = sym.tensor(th.arange(10))

    t.asu = 17
    print(t)
    print(t == th.tensor([17, 17, 17, 3, 4, 17, 17, 17, 8, 9]))
    assert th.all(t == th.tensor([17, 17, 17, 3, 4, 17, 17, 17, 8, 9]))
    assert th.all(t.asym == th.tensor([17, 3, 4, 17, 8, 9]))
    t[:] = 0
    t.unsym[:] = 10
    assert th.all(t == th.tensor([0, 0, 0, 10, 10, 0, 0, 0, 10, 10]))
    # assert 0
    return
    sym = ipd.tests.sym.create_test_sym_manager(symid='C3')
    sym.idx = [(10, 0, 3), (10, 5, 8)]
    sym.idx.set_kind(['RES', 'RES', 'RES', 'RESGP', 'LIG', 'ATOMIZED', 'ATOMIZED', 'ATOMIZED', 'ATOMIZED', 'ATOMGP'])
    t = sym.tensor(th.tensor([11, 12, 13, 0, 1, 21, 22, 23, 2, 3]))
    # by SymSub
    print('>>> t.full')
    print(t.full)
    print('>>> t.sym')
    print(t.sym)
    print('>>> t.asu')
    print(t.asu)
    print('>>> t.asym')
    print(t.asym)
    print('>>> t.sub(0)')
    print(t.sub(0))
    print('>>> t.sub(1)')
    print(t.sub(1))
    print('>>> t.sub(2)')
    print(t.sub(2))
    print('>>> t.unsym')
    print(t.unsym)
    print('>>> t.notasu')
    print(t.notasu)
    # by ChemType

    print('>>> t.all')
    print(t.all)
    print('>>> t.res')
    print(t.res)
    print('>>> t.lig')
    print(t.lig)
    print('>>> t.notlig')
    print(t.notlig)
    print('>>> t.atomized')
    print(t.atomized)
    print('>>> t.gp')
    print(t.gp)

    t = sym.tensor(['RES', 'RES', 'RES', 'RESGP', 'LIG', 'ATOMIZED', 'ATOMIZED', 'ATOMIZED', 'ATOMIZED', 'ATOMGP'])
    print('>>> t.all.orig')
    print(t.all.orig)
    print('>>> t.res.orig')
    print(t.res.orig)
    print('>>> t.lig.orig')
    print(t.lig.orig)
    print('>>> t.notlig.orig')
    print(t.notlig.orig)
    print('>>> t.atomized.orig')
    print(t.atomized.orig)
    print('>>> t.gp.orig')
    print(t.gp.orig)
    # orderings
    print('>>> t.contig')
    print(t.contig)
    print('>>> t.sliced')
    print(t.sliced)

    sym.idx = [(10, 0, 9)]
    sym.idx.set_kind(th.zeros(10))
    print(sym.idx.all)
    t = sym.tensor(th.zeros(10, dtype=int))
    print(t)
    t[0] = 13
    print(t)
    t.asu[1:] = 17
    print(t)
    t.unsym = 7
    print(t)
    with t.sym_updates_off():
        t[:5] = th.arange(5)
    print(t)
    t._update()
    print(t)

    print(th.cat([t.contig.sub(i) for i in range(sym.nsub)]))
    print(t.contig == th.cat([t.contig.sub(i) for i in range(sym.nsub)]))

    # assert 0

@pytest.mark.fast
def test_sym_tensor_sparse():
    sym = ipd.tests.sym.create_test_sym_manager(symid='C3')
    sym.idx = [(10, 0, 3), (10, 5, 8)]
    sym.idx.set_kind(th.tensor([0, 0, 0, 10, 1, 2, 2, 2, 2, 12], dtype=int))  # type: ignore

    t = sym.tensor(th.arange(10), cls=st.sym_tensor_types.FullSlicedAllSparse1DBasic)

if __name__ == '__main__':
    test_sym_tensor()
    test_sym_tensor_sparse()
    test_sym_tensor_sym_typemap_dense_1D()
    test_sym_tensor_sort_typemap_dense_1D()
    test_sym_tensor_res_typemap_dense_1D()
    test_sym_tensor_copy()
    test_sym_tensor_2d()

    print('DONE')
