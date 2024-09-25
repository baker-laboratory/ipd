'''
SymTensor is a subclass of torch.Tensor that is used to store symmetrical tensors.

Make a SymTensor by calling symtensor(tensor) or SymManager.tensor(tensor). All SymTensors are a view
into a full tensor of all sym + unsym data. A SymTensor may show only a
portion of the data, but changes will be applied to the backend full tensor. A SymTensor is
comprised of 5 base types Payload, Shape, ChemType, SymSub, and Ordering. These base types are
used to create a variety of different subclasses that are used to store symmetrical tensors.

Payload Types:
    * XYZ: coordinate data which is symmetrized spatially
    * Pair: 2D pair data which is symmetrized by copying along diagonal stripes
    * Basic: flat data which is simply copied
    * Ptr: indices into a symmetric structure of matching "shape"

ChemTypes:
    * All: all data
    * Res: residue data
    * Lig: ligand data
    * Atomized: atomized residue data
    * GP: guideposts

SymSub Types:
    * Full: all data
    * Sym: symmetrical data
    * Asym: asymmetrical data (unsym + asu)
    * Asu: asymmetric unit data
    * Unsym: unsymmetrical data
    * Sub: subunit i data

Ordering Types:
    * Sliced: sliced data, orderind in the standard way by chem type
    * Contig: subunit-contiguous view that is easy to symmetrize

Shape Types:
    * OneDim: 1D tensors
    * TwoDim: 2D tensors
    * Sparse1D: sparse 1D tensors
    * Sparse2D: sparse 2D tensors


>>> from ipd.dev.lazy_import import lazyimport
th = lazyimport('torch')

>>> sym = ipd.tests.sym.create_test_sym_manager(symid='C3')
>>> sym.idx = [(10, 0, 3), (10, 5, 8)]
>>> sym.idx.set_kind(
...    ['RES', 'RES', 'RES', 'RESGP', 'LIG', 'ATOMIZED', 'ATOMIZED', 'ATOMIZED', 'ATOMIZED', 'ATOMGP'])
>>> t = sym.tensor(th.tensor([11, 12, 13, 0, 1, 21, 22, 23, 2, 3]))

Sliced ChemTypes, SymSub, and Orderings can be accessed thru properties:

>>> t.full
FullSlicedAll1DBasic([11, 12, 13,  0,  1, 21, 22, 23,  2,  3])
>>> t.sym
SymSlicedAll1DBasic([11, 12, 13, 21, 22, 23])
>>> t.asu
AsuSlicedAll1DBasic([11, 21])
>>> t.asym
AsymSlicedAll1DBasic([11,  0,  1, 21,  2,  3])
>>> t.sub(0)
SubSlicedAll1DBasic([11, 21])
>>> t.sub(1)
SubSlicedAll1DBasic([12, 22])
>>> t.sub(2)
SubSlicedAll1DBasic([13, 23])
>>> t.unsym
UnsymSlicedAll1DBasic([0, 1, 2, 3])
>>> t.notasu
NotAsuSlicedAll1DBasic([12, 13,  0,  1, 22, 23,  2,  3])

ChemType:

>>> t.all
FullSlicedAll1DBasic([11, 12, 13,  0,  1, 21, 22, 23,  2,  3])
>>> t.res
FullSlicedRes1DBasic([11, 12, 13])
>>> t.lig
FullSlicedLig1DBasic([1])
>>> t.notlig
FullSlicedNotLig1DBasic([11, 12, 13,  0, 21, 22, 23,  2,  3])
>>> t.atomized
FullSlicedAtomized1DBasic([21, 22, 23,  2])
>>> t.gp
FullSlicedGP1DBasic([0, 3])

Ordering sliced is the original ordering. contig is subunit-contiguout

>>> t.contig
FullContigAll1DBasic([11, 21, 12, 22, 13, 23])
>>> t.contig.sub(0)
SubContigAll1DBasic([11, 21])
>>> t.contig.sub(1)
SubContigAll1DBasic([12, 22])
>>> t.contig.sub(2)
SubContigAll1DBasic([13, 23])
>>> th.all(t.contig == th.cat([t.contig.sub(i) for i in range(sym.nsub)]))
FullContigAll1DBasic(True)
>>> t.contig.asu.sliced.full
FullSlicedAll1DBasic([11, 12, 13,  0,  1, 21, 22, 23,  2,  3])

The input can be a list or numpy object array, in which case the SymTensor will retain
pointers to the original data. The originals can be accessed via the .orig property

>>> t = sym.tensor(['RES', 'RES', 'RES', 'RESGP', 'LIG', 'ATOMIZED', 'ATOMIZED', 'ATOMIZED', 'ATOMIZED', 'ATOMGP'])
>>> t.all.orig[2:-3]
['RES', 'RESGP', 'LIG', 'ATOMIZED', 'ATOMIZED']
>>> t.res.orig
['RES', 'RES', 'RES']
>>> t.lig.orig
['LIG']
>>> t.notlig.orig
['RES', 'RES', 'RES', 'RESGP', 'ATOMIZED', 'ATOMIZED', 'ATOMIZED', 'ATOMIZED', 'ATOMGP']
>>> t.atomized.orig[1:3]
['ATOMIZED', 'ATOMIZED']
>>> t.gp.orig
['RESGP', 'ATOMGP']

SymTensors are automatically symmetrized:

>>> sym.idx = [(10,0,9)]
>>> sym.idx.set_kind(th.zeros(10))
>>> t = sym.tensor(th.zeros(10, dtype=int))
>>> t
FullSlicedAll1DBasic([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
>>> t[0] = 13
>>> t
FullSlicedAll1DBasic([13,  0,  0, 13,  0,  0, 13,  0,  0,  0])
>>> t.asu[1:] = 17
>>> t
FullSlicedAll1DBasic([13, 17, 17, 13, 17, 17, 13, 17, 17,  0])
>>> t.unsym = 7
>>> t
FullSlicedAll1DBasic([13, 17, 17, 13, 17, 17, 13, 17, 17,  7])

To disable automatic symmetriztion, use:

>>> with t.sym_updates_off():
...    t[:5] = th.arange(5)
>>> t # not symmetric
FullSlicedAll1DBasic([ 0,  1,  2,  3,  4, 17, 13, 17, 17,  7])
>>> t._update()
>>> t
FullSlicedAll1DBasic([0, 1, 2, 0, 1, 2, 0, 1, 2, 7])
'''

import sys
import ipd
import willutil as wu
from ipd.dev.lazy_import import lazyimport

th = lazyimport('torch')

from torch import Tensor as T
import functools
import numpy as np

def symtensor(sym, tensor, cls=None, symdims=None, idx=None, isidx=None):
    '''Create a SymTensor object from a tensor

        Args:
            tensor (torch.Tensor): the tensor to create a SymTensor from
            cls (type): the class to create the tensor from
            symdims (list): the dimensions that are symmetrical
            idx (SymIndex): index for the tensor, if sparse
            isidx (torch.Tensor): which entries are indices (pointers)
        Returns:
            SymTensor: a SymTensor object with the appropriates bases
        '''
    try:
        tensor, origval, origtype, val = th.as_tensor(tensor), None, None, None
    except (TypeError, ValueError):
        if isinstance(tensor, np.ndarray): assert tensor.ndim == 1
        tensor, origval, origtype, val = th.arange(len(tensor)), np.array(tensor), type(tensor), 'Ptr'

    symdims = symdims or sym.symdims(tensor, idx)
    if idx is None and len(symdims) and tensor.shape[symdims[0]] == sym.idx.Nasym:
        newshape = list(tensor.shape)
        for i in symdims:
            newshape[i] = sym.idx.L
        new = th.empty(newshape)
        new[sym.idx.asym] = tensor
    origshape = tensor.shape
    while len(tensor) == 1:
        tensor, symdims = tensor[0], symdims[1:]
    symdims = list(symdims)
    assert symdims == [0] or symdims == [0, 1]

    if val == 'Ptr': pass
    elif tensor.shape[-1] in (3, 4) and th.is_floating_point(tensor) and len(symdims) == 1: val = 'Xyz'
    elif len(symdims) == 2 and tensor.ndim > 2 and th.is_floating_point(tensor): val = 'Pair'
    elif isidx is not None: val = 'Ptr'
    else: val = 'Basic'

    if idx is not None: dim = 'Sparse1D'
    elif len(symdims) == 1: dim = '1D'
    elif len(symdims) == 2: dim = '2D'

    cls = cls or sym_tensor_types[f'FullSlicedAll{dim}{val}']
    symten = tensor.as_subclass(cls)
    attr = wu.Bunch(sym=sym,
                    ordering=th.arange(sym.idx.L, dtype=int),
                    symmask=th.ones(sym.idx.L, dtype=bool),
                    resmask=th.ones(sym.idx.L, dtype=bool),
                    origval=origval,
                    origtype=origtype,
                    origshape=origshape,
                    orig=tensor.clone(),
                    symdims=symdims,
                    noupdate=False,
                    symnoupdate=False)

    symten = cls(symten, attr)
    symten.attr.observers = set([symten])

    return symten

class SymTensor(th.Tensor):
    '''Base class for symmetrical tensors'''
    @classmethod
    def __torch_function__(cls, func, types, a=(), kw=None):
        '''__torch_function__ is called when a torch function is called on a SymTensor'''
        if kw is None: kw = {}
        # if hasattr(cls, func.__name__): getattr(cls, func.__name__)(*a, **kw)
        if func not in _TORCH_FUNCS or not all(issubclass(t, (T, SymTensor)) for t in types):
            result = super().__torch_function__(func, types, a, kw)
            if isinstance(result, T):
                result = result.set_attr(a)
                # ic(func, a, result)
            return result
        return _TORCH_FUNCS[func](*a, **kw)

    def __new__(cls, t: 'SymTensor', attr) -> 'SymTensor':
        # ic('SymTensor')
        assert isinstance(t, SymTensor)
        assert not t.as_subclass(T).requires_grad or not torch.is_grad_enabled()
        return t
        # new = t.as_subclass(cls).set_attr(t.attr)
        # new.attr.observers.add(new)
        # return new

    def set_attr(self, attr):
        if isinstance(attr, (list, tuple)):
            # for a in attr: print(a)
            attr = [a.attr for a in attr if hasattr(a, 'attr')]
            if attr: attr = attr[0]
            else: return self
        self.attr = attr.copy()
        return self

    # def clone(self):
    # return th.Tensor.clone(self).set_attr(self.attr)

    def result(self):
        if isinstance(self.attr.orig, list):
            return [self.attr.origval[i] for i in self]
        if isinstance(self.attr.orig, np.ndarray):
            assert self.attr.origval.ndim == 1
            return np.array([self.attr.origval[i] for i in self])
        return self.full.sliced.reshape(self.attr.origshape)

    # def reshape(self, *a, **kw):
    # return th.Tensor.reshape(self, *a, **kw).as_subclass(self.__class__).set_attr(self.attr)

    # def view(self, *a, **kw):
    # return th.Tensor.view(self, *a, **kw).as_subclass(self.__class__).set_attr(self.attr)

    # def __getitem__(self, slice):
    # return self.as_subclass(T).__getitem__(slice).as_subclass(self.__class__).set_attr(self.attr)

    def __setitem__(self, slice, val):
        assert hasattr(self, 'attr')
        # print('__setitem__')
        if isinstance(val, th.Tensor):
            if isinstance(slice, th.Tensor): ic(slice.shape)
        self.as_subclass(T)[slice] = val.as_subclass(T) if isinstance(val, th.Tensor) else val
        self._update()

    def __deepcopy__(self, memo):
        copy = self.clone()
        memo[id(self)] = copy
        return copy

    def __del__(self):
        '''__del__ is strange, sometimes stuff doesn't exist anymore'''
        if hasattr(self, 'attr') and hasattr(self.attr, 'observers'):
            if self.attr.observers and self in self.attr.observers:
                self.attr.observers.remove(self)

    def updates_off(self, name='noupdate'):
        class NoUpdate:
            def __init__(self, t):
                self.t = t

            def __enter__(self):
                self.t.attr[name] = True

            def __exit__(self, *args):
                self.t.attr[name] = False

        return NoUpdate(self)

    def sym_updates_off(self):
        return self.updates_off(name='symnoupdate')

    @property
    def orig(self):
        if self.attr.origval is not None:
            result = self.attr.origval[self.attr.idx]
            if self.attr.origtype is not None:
                if result.ndim == 0: result = [result]
                result = self.attr.origtype(result)
            return result
        return self

_TORCH_FUNCS = {}

def implements_torch(torch_function):
    """Register a torch function override for SymTensor"""
    def decorator(func):
        functools.update_wrapper(func, torch_function)
        _TORCH_FUNCS[torch_function] = func
        return func

    return decorator

@implements_torch(th.cat)
def cat(vals, *a, **kw):
    return th.cat([v.as_subclass(T) for v in vals], *a, **kw)

class Payload(SymTensor):
    pass

class XYZ(Payload):
    def _symmetrize_orig(self):
        if self.attr.symnoupdate or self.attr.noupdate: return
        with self.updates_off():
            self.attr.orig[:] = self.attr.sym.apply_symmetry(self.sym.contig).sliced.full

class Pair(Payload):
    def _symmetrize_orig(self):
        if self.attr.symnoupdate or self.attr.noupdate: return
        with self.updates_off():
            self.attr.orig[:] = self.attr.sym.apply_symmetry_pair(self.sym.contig).sliced.full

class Basic(Payload):
    def _symmetrize_orig(self):
        if self.attr.symnoupdate or self.attr.noupdate: return
        with self.updates_off():
            self.sym.contig._symmetrize_orig_dim_unsafe()._update(force=True)

class Ptr(Payload):
    def _symmetrize_orig(self):
        if self.attr.symnoupdate or self.attr.noupdate: return
        with self.updates_off():
            self.sym.contig._symmetrize_orig_dim_unsafe(ptr=True)

class Ordering(SymTensor):
    '''Base class for sort type masks: sortres, sortsub'''
    def __new__(cls, t: SymTensor, attr, **kw):
        # ic('Ordering')
        base = cls.__bases__[list(subbasenames.keys()).index(__class__)]
        sortname = (cls if cls.__bases__ == (__class__, ) else base).__name__.lower()
        attr.ordering = getattr(attr.sym.idx, sortname)
        # ic(attr.ordering, sortname)
        # print('Ordering ', cls)
        return super().__new__(cls, t, attr, **kw)

class ChemType(SymTensor):
    '''Base class for "residue" type masks: residue, lig, atomized, gp'''
    def __new__(cls, t: SymTensor, attr, **kw):
        # ic('ChemType')
        base = cls.__bases__[list(subbasenames.keys()).index(__class__)]
        resmask = (cls if cls.__bases__ == (__class__, ) else base).__name__.lower()
        # ic(attr.resmask)
        attr.resmask = getattr(attr.sym.idx, resmask)
        # print('ChemType ', cls)
        return super().__new__(cls, t, attr, **kw)

class SymSub(SymTensor):
    '''Base class for sub-sym mask types: Asym, Asu, Unsym, Sub, etc. Controls the symmask'''
    def __new__(cls, t: SymTensor, attr, *, isub=None, **kw):
        # ic('SymSub')
        base = cls.__bases__[list(subbasenames.keys()).index(__class__)]
        symmask = (cls if cls.__bases__ == (__class__, ) else base).__name__.lower()
        attr.symmask = getattr(attr.sym.idx, symmask)
        if issubclass(cls, (Sub, NotSub)):
            try:
                attr.symmask = attr.symmask[isub]
            except IndexError:
                raise SymTensorIndexError(f'no subunit {isub}')
        # ic('SymSub', cls.__name__, isub, id(attr.symmask), attr.symmask.sum())
        return super().__new__(cls, t, attr, **kw)

class Shape(SymSub, ChemType, Ordering):
    pass

class OneDim(Shape):
    '''Base class for 1D tensors'''
    def __new__(cls, t, attr, **kw):
        t = super().__new__(cls, t, attr, **kw)
        # ic('OneDim')
        # ic(attr.symmask.to(int))
        # ic(attr.resmask.to(int))
        # ic(attr.ordering)
        attr.idx = attr.ordering[(attr.symmask & attr.resmask)[attr.ordering]]
        new = attr.orig[attr.idx]
        new = new.as_subclass(cls).set_attr(attr)
        # ic('OneDim', new.__class__.__name__, id(attr.symmask), new.attr.idx.sum())
        return new

    def __setattr__(self, name, value):
        if isinstance(value, th.Tensor):
            getattr(self, name)[:] = value.as_subclass(T)
        elif isinstance(value, (int, float)):
            getattr(self, name)[:] = value
        else:
            self.__dict__[name] = value

    def _update_view(self):
        self.as_subclass(T).__setitem__(slice(None), self.attr.orig[self.attr.idx])

    def _update(self, force=False):
        if not force and self.attr.noupdate: return
        self.attr.orig[self.attr.idx] = self.as_subclass(T)
        self._symmetrize_orig()
        for o in self.attr.observers:
            o._update_view()

    def _symmetrize_orig_dim_unsafe(self, ptr=False):
        N = len(self) // self.attr.sym.nsub
        for i in range(1, self.attr.sym.nsub):
            val = self[:N]
            if ptr: val = self.attr.sym.idx.idx_asu_to_sub[i, val[:, self.attr.isidx].to(int)]
            self[i * N:(i + 1) * N] = val
        return self

class TwoDim(Shape):
    '''Base class for 2D tensors'''
    def __new__(cls, t, attr, **kw):
        t = super().__new__(cls, t, attr, **kw)
        # ic('TwoDim')
        attr.idx = attr.ordering[(attr.symmask & attr.resmask)[attr.ordering]]
        new = attr.sym.idx.slice2d(attr.orig, attr.idx, dim=attr.symdims)
        new = new.as_subclass(cls).set_attr(attr)
        return new

    def _update_view(self):
        assert hasattr(self, 'attr')
        val = self.attr.sym.idx.slice2d(self.attr.orig, self.attr.idx, dim=self.attr.symdims)
        th.Tensor.__setitem__(self, (slice(None), slice(None)), val)

    def _update(self, force=False):
        if not force and self.attr.noupdate: return
        self.attr.sym.idx.slice2d(self.attr.orig, self.attr.idx, self, dim=self.attr.symdims)
        for o in self.attr.observers:
            o._update_view()

    def slice2d(self, idx, value=None):
        assert hasattr(self, 'attr')
        result = self.attr.sym.idx.slice2d(self, idx, value, dim=self.attr.symdims)
        return result

    def __setattr__(self, name, value):
        if name != 'attr': assert hasattr(self, 'attr')
        if isinstance(value, th.Tensor):
            getattr(self, name)[:, :] = value.as_subclass(T)
        else:
            self.__dict__[name] = value

    def _symmetrize_orig_dim_unsafe(self, ptr=False):
        if self.attr.symnoupdate or self.attr.noupdate: return
        contig = self.sym.contig
        N = len(contig) // self.attr.sym.nsub
        for i in range(1, self.attr.sym.nsub):
            val = contig[:N, :N]
            if ptr: val = self.attr.sym.idx.idx_asu_to_sub[i, val[:, :, self.attr.isidx].to(int)]
            contig[i * N:(i + 1) * N, i * N:(i + 1) * N] = val
        return self

class Sparse1D(OneDim):
    pass

class Sparse2D(TwoDim):
    pass

# the real voodoo is below

M = sys.modules[__name__]

SymTensorError = type('SymTensorError', (Exception, ), {})
SymTensorTypeError = type('SymTensorTypeError', (SymTensorError, ), {})
SymTensorIndexError = type('SymTensorIndexError', (SymTensorError, ), {})

sym_tensor_types = wu.Bunch()
base_types = dict()
type_maps = dict()
subbasenames = dict()

subbasenames[SymSub] = ['Full', 'Sym', 'Asym', 'Asu', 'Unsym', 'Sub', 'NotAsym', 'NotAsu', 'NotSub']
subbasenames[Ordering] = ['Sliced', 'Contig']
subbasenames[ChemType] = ['All', 'Res', 'Lig', 'Atomized', 'GP', 'NotRes', 'NotLig', 'NotAtomized', 'NotGP']
subbasenames[Shape] = ['OneDim', 'TwoDim', 'Sparse1D', 'Sparse2D']

def make_sub_bases():
    for Base, subbases in list(subbasenames.items()):
        if Base is Shape: continue
        for name in subbases:
            if name in ('Sub', 'NotSub'): continue
            base_types[name] = type(name, (Base, ), {})

            def make_prop(name):
                return property(lambda self: type_maps[self.__class__][base_types[name]]
                                (self, self.attr.copy()))

            setattr(Base, name.lower(), make_prop(name))

    base_types['Sub'] = type('Sub', (SymSub, ), {})
    base_types['NotSub'] = type('NotSub', (SymSub, ), {})
    SymSub.sub = lambda self, i: type_maps[self.__class__][Sub](self, self.attr.copy(), isub=i)
    SymSub.notsub = lambda self, i: type_maps[self.__class__][NotSub](self, self.attr.copy(), isub=i)
    for k, v in base_types.items():
        setattr(M, k, v)

make_sub_bases()

def makeclass(bases):
    name = ''.join(b.__name__ for b in bases)
    name = name.replace('OneDim', '1D')
    name = name.replace('TwoDim', '2D')
    if not name in sym_tensor_types:
        sym_tensor_types[name] = type(name, tuple(bases), dict())
    return sym_tensor_types[name]

def allbasecombos():
    for valtype in (XYZ, Pair, Basic, Ptr):
        for dim in [getattr(M, n) for n in subbasenames[Shape]]:
            if valtype is Ptr and dim is TwoDim: continue
            if valtype is XYZ and dim is TwoDim: continue
            if valtype is Pair and dim is OneDim: continue
            for restype in [getattr(M, n) for n in subbasenames[ChemType]]:
                for symsub in [getattr(M, n) for n in subbasenames[SymSub]]:
                    for sort in [getattr(M, n) for n in subbasenames[Ordering]]:
                        yield [symsub, sort, restype, dim, valtype]

def make_all_classes():
    for bases in allbasecombos():
        cls = makeclass(bases)
        for i, (base, subs) in enumerate(subbasenames.items()):
            for sub in [getattr(M, n) for n in subbasenames[base]]:
                newbases = bases.copy()
                newbases[i] = sub
                type_maps.setdefault(cls, dict())[sub] = makeclass(newbases)

    # print(f'sym_tensor generated {len(sym_tensor_types.keys())} types')
    # print(f'{list(sym_tensor_types.keys())}')

make_all_classes()
