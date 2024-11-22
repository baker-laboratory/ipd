from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
import contextlib
import copy
from functools import singledispatch
import dataclasses
from typing import Any, Generic, TypeVar

import numpy as np

import ipd
from ipd.lazy_import import lazyimport
from ipd.sym.sym_kind import ShapeKind, SymKind, ValueKind

th = lazyimport('torch', warn=False)

T = TypeVar('T')

@singledispatch
def _sym_adapt(thing: Any, sym, isasym=None) -> 'SymAdapt':
    """Return a Symmable object that knows how to convert beteen input and a
    symmetrizable adapted form."""
    raise NotImplementedError(f"Don't know how to make SymAdapt for {type(thing)}")

@_sym_adapt.register(type(None))  # type: ignore
def _(*a, **kw):
    return None

with contextlib.suppress(ImportError):

    @_sym_adapt.register(th.Tensor)  # type: ignore
    def _(tensor, sym, isasym):
        if all(n is None for n in tensor.names):
            return SymAdaptTensor(tensor, sym, isasym)
        elif 'Lsparse' in tensor.names:
            return SymAdaptNamedSparseTensor(tensor, sym, isasym)
        else:
            return SymAdaptNamedDenseTensor(tensor, sym, isasym)

@_sym_adapt.register(np.ndarray)  # type: ignore
def _(ary, sym, isasym):
    if ary.dtype in (np.float64, np.float32, np.float16, np.complex64, np.complex128, np.int64, np.int32, np.int16,
                     np.int8, np.uint8, bool):
        return SymAdaptTensor(ary, sym, isasym, tlib='numpy')
    else:
        return SymAdaptNDArray(ary, sym, isasym)

class SymAdapt(ABC, Generic[T]):
    """You must define a subclass of SymAdapt for each type you want to symmetrize.

    Must have a kind and adapted property. See the :SymAdaptDataClass:`ipd.sim.SymAdaptDataClass` class for an example.
    These classes are not meant for the end user, they will be used internally by the sym manager
    """
    def __init_subclass__(cls, **kw):
        if not hasattr(cls, 'adapts'):
            raise TypeError(f'class {name} must define adapted type via adapts = ThingType')  # type: ignore
        if cls.adapts is not None:  # type: ignore

            @_sym_adapt.register(cls.adapts)  # type: ignore
            def _(thing, sym, isasym=None):
                return cls(thing, sym, isasym)  # type: ignore

    def __init__(self, x: T, sym: 'ipd.sym.SymmetryManager', isasym: bool):
        self.orig: T
        self.kind: ipd.sym.SymKind

    @abstractmethod
    def reconstruct(self, list_of_symmetrized) -> T:
        """Restore from dict of components that have been symmetrized."""

    # @property
    # def kind(self) -> SymKind:
    # '''return the SymKind that properly describes your components'''
    # return ipd.sym.SymKind(ipd.sym.ShapeKind.MAPPING, ipd.sym.ValueKind.MIXED)

    def __repr__(self):
        if isinstance(self.orig, (np.ndarray, th.Tensor)):  # type: ignore
            return f'{self.__class__.__name__}(orig={self.orig.shape}, new={self.new.shape})'  # type: ignore
        return f'{self.__class__.__name__}(kind={self.kind}, orig={type(self.orig)})'  # type: ignore

@_sym_adapt.register(SymAdapt)  # type: ignore
def _(thing, sym, isasym):
    return thing  # already sym adapted

class SymAdaptStr(SymAdapt):
    adapts = str

    def __init__(self, x, sym, isasym):
        self.orig = x
        self.sym = sym
        self.slice = sym.idx
        self.kind = SymKind(ShapeKind.ONEDIM, ValueKind.BASIC)

    @property
    def adapted(self):
        x = np.array(list(self.orig), dtype=object)
        if self.slice.Nasym == len(self.orig):
            x, old = np.zeros(self.slice.L, dtype=object), x
            x[self.slice.asym] = old
            return x
        return x

    def reconstruct(self, canon, asym=False):  # type: ignore
        return ''.join(canon)

class SymAdaptSequence(SymAdapt):
    adapts = Sequence

    def __init__(self, x, sym, isasym):
        self.orig = x
        self.sym = sym

    @property
    def kind(self):  # type: ignore
        if self.orig and isinstance(self.orig[0], (int, float, str)):
            return ipd.sym.SymKind(ipd.sym.ShapeKind.ONEDIM, ipd.sym.ValueKind.BASIC)  # type: ignore
        return ipd.sym.SymKind(ipd.sym.ShapeKind.SEQUENCE, ipd.sym.ValueKind.MIXED)  # type: ignore

    @property
    def adapted(self):
        if self.orig and isinstance(self.orig[0], (int, float, str)):
            return np.array(list(self.orig))
        return [self.sym.sym_adapt(x) for x in self.orig]

    def reconstruct(self, canonicals):  # type: ignore
        return type(self.orig)(canonicals)

class SymAdaptMap(SymAdapt):
    adapts = Mapping

    def __init__(self, x, sym, isasym):
        self.orig = x
        self.sym = sym
        self.kind = ipd.sym.SymKind(ipd.sym.ShapeKind.MAPPING, ipd.sym.ValueKind.MIXED)  # type: ignore
        self.adapted = copy.copy(self.orig)

    def reconstruct(self, canonicals):  # type: ignore
        return type(self.orig)(canonicals)

class SymAdaptDataClass(SymAdapt):
    """Base class for adapting dataclasses.

    All fields must be sym-adaptable and all tensor fields must have
    intepretable shapes, or dim names via add_tensor_dim_names
    """
    adapts = None

    def __init__(self, dataclass, sym, isasym):
        self.orig = dataclass
        self.sym = sym
        self.isasym = isasym
        self.kind = ipd.sym.SymKind(ipd.sym.ShapeKind.MAPPING, ipd.sym.ValueKind.MIXED)  # type: ignore
        # for f in dataclasses.fields(dataclass):
        #     v = getattr(dataclass, f.name)
        #     if v is None: continue
        #     if th.is_tensor(v): print(f'{f.name:15} {v.shape}')
        #     else: print(f'{f.name:15} {[x.shape for x in v]}')
        self.orig.add_tensor_dim_names()
        d = {f.name: getattr(self.orig, f.name) for f in dataclasses.fields(self.orig)}  # no deepcopy
        # print(f'SymAdaptDataClass.adapted: {self.orig.__class__.__name__}')
        for k in d:
            if th.is_tensor(d[k]): d[k] = d[k].rename(*d[k].names)  # shallow copy that keeps names
        self.orig.remove_tensor_dim_names()
        # for k, v in d.items():
        #     if v is None: continue
        #     if isinstance(v, SimpleSparseTensor):
        #         print(f'    {k:13} {str(v.val.dtype):13} {tuple(v.val.shape)} {v.val.names}', flush=True)
        #     elif th.is_tensor(v):
        #         print(f'    {k:13} {str(v.dtype):13} {tuple(v.shape)} {v.names}', flush=True)
        #     else:
        #         print(f'{f.name:15} {[x.shape for x in v]}')
        self.adapted = d

    def reconstruct(self, symparts, **kw):  # type: ignore
        # print(f'SymAdaptDataClass.reconstruct {self.orig.__class__.__name__}')
        for k, v in symparts.items():
            if v is None: continue
            if isinstance(v, SimpleSparseTensor):
                symparts[k] = v.val
            #     print(f'    {k:13} {v.val.dtype} {tuple(v.val.shape)} {v.val.names}', flush=True)
            # elif th.is_tensor(v):
            #     print(f'    {k:13} {str(v.dtype):13} {tuple(v.shape)} {v.names}', flush=True)
            # else:
            #     print(f'{k:15} {[x.shape for x in v]}')
        return type(self.orig)(**symparts)

# ipd libs shouldn't require torch
with contextlib.suppress(ImportError):

    @dataclasses.dataclass
    class SimpleSparseTensor:
        val: th.Tensor  # type: ignore
        idx: th.Tensor  # type: ignore
        isidx: slice
        kind: kind = None  # type: ignore

    # @_sym_adapt.register(SimpleSparseTensor)  # type: ignore
    # def _(sparse, sym):
    # return SymAdaptTensor(sparse.val, sym, idx=sparse.idx, isidx=sparse.isidx)

    def check_isasym(tensor, sym, isasym, idx):
        if isasym is not None: return isasym
        if sym.L in tensor.shape: return False
        if sym.Nasym in tensor.shape: return True
        assert idx is not None

    class SymAdaptNamedDenseTensor(SymAdapt):
        adapts = None

        def __init__(self, tensor, sym, isasym=None):  # sourcery skip: de-morgan
            if not ('L' in tensor.names or 'L1' in tensor.names or 'L2' in tensor.names):
                self.kind = SymKind(ShapeKind.SCALAR, ValueKind.BASIC)
                self.adapted = None
                self.orig = tensor
                return
            assert 'N3' not in tensor.names, 'no support for 3D'
            self.orig = tensor
            if 'L2' in tensor.names:
                self.kind = SymKind(ShapeKind.TWODIM, ValueKind.BASIC)
                if 'Pair' in tensor.names:
                    self.kind = SymKind(ShapeKind.TWODIM, ValueKind.PAIR)
                self.perm = tensor.align_to('L1', 'L2', ...)
            elif any(n.startswith('Idx') for n in tensor.names):
                self.kind = SymKind(ShapeKind.ONEDIM, ValueKind.INDEX)
                self.perm = tensor.align_to('L', ...)
            elif 'XYZ' in tensor.names:
                self.kind = SymKind(ShapeKind.ONEDIM, ValueKind.XYZ)
                self.perm = tensor.align_to('L', ..., 'XYZ')
            else:
                self.kind = SymKind(ShapeKind.ONEDIM, ValueKind.BASIC)
                self.perm = tensor.align_to('L', ...)
            assert self.perm.shape[0] in (sym.idx.L, sym.idx.Nasym)
            if self.perm.shape[0] == sym.idx.Nasym:
                assert isasym is None or isasym
                if self.kind.shapekind == ShapeKind.TWODIM:
                    newshape = (sym.idx.L**2, *self.perm.shape[2:])
                    self.adapted = th.zeros(newshape, dtype=tensor.dtype, device=tensor.device)
                    m = (sym.idx.asym[None] * sym.idx.asym[:, None]).view(-1)
                    self.adapted[m] = self.perm.rename(None).view(sym.idx.Nasym * sym.idx.Nasym, *self.perm.shape[2:])
                    self.adapted = self.adapted.view(sym.idx.L, sym.idx.L, *self.perm.shape[2:])
                else:
                    newshape = (sym.idx.L, *self.perm.shape[1:])
                    self.adapted = th.zeros(newshape, dtype=tensor.dtype, device=tensor.device)
                    self.adapted[sym.idx.asym] = self.perm.rename(None)
            else:
                assert not isasym
                assert self.perm.shape[0] == sym.L
                self.adapted = self.perm.rename(None)
            self.adapted = self.adapted.to(sym.device).to(self.orig.dtype)

        def reconstruct(self, x, **kw):  # type: ignore
            return x.rename(*self.perm.names).align_to(*self.orig.names).to(self.orig.device).to(
                self.orig.dtype).rename(None)

    class SymAdaptNamedSparseTensor(SymAdapt):
        adapts = None

        def __init__(self, tensor, sym, isasym):
            assert 'Lsparse' in tensor.names
            assert 1 == sum(n.startswith('IdxAll') for n in tensor.names)
            assert 'L' not in tensor.names and 'L1' not in tensor.names and 'XYZ' not in tensor.names
            self.orig = tensor
            self.isidx = True  # don't support XYZ and BASIC for now
            self.kind = SymKind(ShapeKind.SPARSE, ValueKind.INDEX)
            idxdim = [n for n in tensor.names if n.startswith('Idx')][0]
            self.perm = tensor.align_to('Lsparse', idxdim, ...)
            idx = self.perm[:, int(idxdim.replace('IdxAll', '').replace('Idx', ''))].rename(None).to(int)
            # ic(idx, self.perm.shape)
            # ic(tensor.shape, tensor.names, idxdim, idx)
            if sym.idx.is_sym_subsequence(idx):
                assert len(idx) == 0 or (0 <= idx.min() and idx.max() < sym.idx.L)
                self.asym = False
                self.adapted = SimpleSparseTensor(idx=idx, val=self.perm, isidx=self.isidx)  # type: ignore
                assert not isasym
            elif sym.idx.is_asym_subsequence(idx):
                assert len(idx) == 0 or (0 <= idx.min() and idx.max() < sym.idx.Nasym)
                assert isasym is None or isasym
                symidx = sym.idx.symidx(idx)
                dd = dict(dtype=self.orig.dtype, device=sym.device)
                symperm = th.zeros((len(symidx), *self.perm.shape[1:]), **dd)
                symperm[:len(self.perm)] = self.perm
                symperm.rename_(*self.perm.names)
                self.perm = symperm
                self.adapted = SimpleSparseTensor(idx=symidx, val=symperm, isidx=self.isidx)  # type: ignore
                if self.isidx:
                    self.adapted.val[:] = sym.idx.idx_asym_to_sym[self.adapted.val.to(int).rename(None)]
            else:
                raise ValueError(f'tensor {tensor.shape} not sym or asym compatible')
            assert len(self.adapted.idx) == len(self.adapted.val)
            self.adapted.idx = self.adapted.idx.to(sym.device)
            self.adapted.val = self.adapted.val.rename(None).to(sym.device).to(self.orig.dtype)

        def reconstruct(self, x, **kw):  # type: ignore
            # ic(self.perm.names, self.orig.names)
            x = x.val.rename(*self.perm.names).align_to(*self.orig.names).rename(None)
            return x.to(self.orig.device).to(self.orig.dtype)

    class SymAdaptNDArray(SymAdapt):
        """Symmetrizable ndarray."""
        adapts = np.ndarray

        def __init__(self, x, sym, isasym=None):
            """Handles object and str dtypes."""
            self.orig = x
            self.sym = sym
            self.isasym = isasym
            assert x.ndim > 0

        @property
        def kind(self):  # type: ignore
            return SymKind(ShapeKind.ONEDIM, ValueKind.BASIC)

        @property
        def adapted(self):
            if len(self.orig) == self.sym.idx.L:
                if self.isasym is not None: assert not self.isasym
                new = self.orig.copy()
            elif len(self.orig) == self.sym.idx.Nasym:
                if self.isasym is not None: assert self.isasym
                new = np.empty((self.sym.idx.L, *self.orig.shape[1:]), dtype=self.orig.dtype)
                new[self.sym.idx.asym.cpu()] = self.orig
            else:
                raise ValueError(f'unsupported length {len(self.orig)} L={self.sym.idx.L}, Lasym = {self.sym.idx.Nasym}')
            return new

        def reconstruct(self, ary):  # type: ignore
            return ary

    ########## SymAdaptTensor is kinda gross and depricated, trying to replace with the NamedTensor variant ###########

    def tensor_keydims_to_front(x, keydim):
        if tensor_is_xyz(x):
            undo = [x.shape]
            while x.shape[0] == 1:
                x = x[0]
            if x.ndim == 2: x = x[:, None, :]
            return x, undo
        altpos, undo = x.ndim + 1, [x.shape]
        while x.shape[0] != keydim and 0 < (altpos := altpos - 1):
            if x.shape[0] == 1:
                x = x.view(*x.shape[1:altpos], 1, *x.shape[altpos:])
            else:
                oldshape = x.shape
                x = x.swapdims(0, altpos - 1)
                undo.insert(0, [x.shape, (altpos - 1, 0), oldshape])
                # ic(undo[0])
        if x.shape[0] != keydim:
            raise ValueError(f'bad no keydim {keydim} in tensor shape {x.shape}')
        return x, undo

    def _resize(shape, oldL, newL):
        if not oldL or (oldL == newL): return shape
        shape = th.as_tensor(shape, dtype=int)
        shape[shape == oldL] = newL
        return tuple(shape)

    def tensor_undo_perm(x, undo, resizefrom=0):
        oldL = int(resizefrom)
        newL = int(x.shape[0])
        for oldshape, swap, newshape in undo[:-1]:
            # ic(x.shape, oldshape, swap, newshape)
            x = x.view(_resize(oldshape, oldL, newL)).swapdims(*swap).view(_resize(newshape, oldL, newL))
        return x.view(_resize(undo[-1], oldL, newL))

    def tensor_is_xyz(x):
        return 2 <= x.ndim < 5 and x.shape[-1] == 3 and th.is_floating_point(x)

    class SymAdaptTensor(SymAdapt):
        adapts = None

        def __init__(self, tensor, sym, isasym=None, idx=None, isidx=None, kind=None, tlib='torch'):
            '''Args:
                tensor: tensor to symmetrize
                sym: symmetrization manager
                idx: sparse indices of the tensor, if sparse
                isidx: if/which values are indices
            Attributes:
                symdims: dimensions to symmetrize
                symshape: shape of symmetrized tensor
                origasym: whether the tensor is asymmetric
                L: length of symmetrized tensor
                Nasym: length of asymmetric tensor
                sym: the symmetry manager
            '''
            self.tlib = tlib
            self._kind = kind
            self.isasym = check_isasym(tensor, sym, isasym, idx)
            self.idx = None if idx is None else th.as_tensor(idx, device=sym.device)
            self.isidx = isidx
            # self.L = sym.idx.L
            # self.Nasym = sym.idx.Nasym
            self.sym = sym
            self.orig = th.as_tensor(tensor)
            self.make_room_for_sym(self.orig)
            # self.symdims, self.symshape, self.asymshape, self.workshape = self.make_symdims()

        @property
        def kind(self) -> SymKind:  # type: ignore
            v = self.new.val if isinstance(self.new, SimpleSparseTensor) else self.new
            if self._kind is not None: valuekind = self._kind.valuekind
            elif tensor_is_xyz(self.orig): valuekind = ipd.sym.ValueKind.XYZ  # type: ignore
            elif self.isidx is not None: valuekind = ipd.sym.ValueKind.INDEX  # type: ignore
            elif len(self.symdims) == 2 and len(self.orig) == 1: valuekind = ipd.sym.ValueKind.PAIR  # type: ignore
            else: valuekind = ipd.sym.ValueKind.BASIC  # type: ignore
            if self._kind is not None: shapekind = self._kind.shapekind
            elif len(self.symdims) == 0: shapekind = ipd.sym.ShapeKind.SPARSE  # type: ignore
            elif len(self.symdims) == 1: shapekind = ipd.sym.ShapeKind.ONEDIM  # type: ignore
            elif len(self.symdims) == 2: shapekind = ipd.sym.ShapeKind.TWODIM  # type: ignore
            else: assert 0
            return SymKind(shapekind, valuekind)  # type: ignore

        @property
        def adapted(self):
            'convert to tensor ready for symmetrization'
            if self.idx is not None:
                assert isinstance(self.new, SimpleSparseTensor)
                assert len(self.new.idx) == len(self.new.val)
                adapted = self.new
                adapted.kind = self.kind
                if self.tlib == 'torch': adapted.val = adapted.val.to(self.sym.device)
            else:
                adapted = self.new.reshape(self.workshape).clone()  # type: ignore
                if self.tlib == 'torch': adapted = adapted.to(self.sym.device)
                if isinstance(self.isidx, (tuple, slice)):
                    adapted = adapted[self.isidx]

            # if isinstance(self.isidx, (tuple, slice)):
            # adapted.val = adapted.val[self.isidx]
            return adapted

        def reconstruct(self, x, asym=False, asu=False, unsym=False, symonly=False):  # type: ignore
            assert asym + asu + unsym <= 1
            if isinstance(x, SimpleSparseTensor):
                assert len(x.idx) == len(x.val)
                if self.tlib == 'torch': x.val = x.val.to(self.orig.device)
                if asym: x.val = x.val[self.sym.idx.asymidx(x.idx)].reshape(self.asymshape)
                elif asu: x.val = x.val[self.sym.idx.asymidx(x.idx)].reshape(self.asushape)
                elif unsym: x.val = x.val[self.sym.idx.asymidx(x.idx)].reshape(self.unsymshape)
                elif symonly: x.val = x.val[self.sym.idx.asymidx(x.idx)].reshape(self.symonlyshape)
                else: x.val = x.val.reshape(self.symshape)
                return x
            # ic(x.shape)
            # ic(self.sym.L)
            # ic(self.undo)
            new = tensor_undo_perm(x, self.undo, resizefrom=self.sym.Nasym if self.isasym else self.sym.L)
            if self.tlib == 'torch': new = new.to(self.orig.device)
            else: new = new.cpu().numpy()
            return new

        def make_room_for_sym(self, tensor):
            'if input is asymmetric, resize symdims to full symmetric size'
            self.symdims = None
            if self.idx is not None:
                s = self.sym.idx
                self._make_room_for_sym_sparse(tensor)
            else:
                self._make_room_for_sym_dense(tensor)

        def _make_room_for_sym_dense(self, tensor):
            # DENSE
            kw = {} if self.tlib == 'numpy' else dict(device=tensor.device)
            s = self.sym.idx
            keydim = s.Nasym if self.isasym else s.L
            tensor, self.undo = tensor_keydims_to_front(tensor, keydim)
            if tensor.shape[:2] == (s.Nasym, s.Nasym) and s.Nasym != s.L:
                if self.isasym is not None: assert self.isasym
                shape = (s.L * s.L, *tensor.shape[2:])
                if self.tlib == 'torch': self.new = th.zeros(shape, dtype=tensor.dtype, **kw)
                else: self.new = th.zeros(shape, dtype=tensor.dtype, **kw)
                m = (s.asym[None] * s.asym[:, None]).reshape(-1)
                self.new[m] = tensor.reshape(s.Nasym * s.Nasym, *tensor.shape[2:])
                self.new = self.new.reshape(s.L, s.L, *tensor.shape[2:])
                assert not self.isidx
            elif tensor.shape[0] == s.Nasym and s.Nasym != s.L:
                # ic(tensor.shape, s.Nasym, s.L)
                if self.isasym is not None: assert self.isasym
                shape = (s.L, *tensor.shape[1:])
                if self.tlib == 'torch': self.new = th.zeros(shape, dtype=tensor.dtype, **kw)
                else: self.new = th.zeros(shape, dtype=tensor.dtype, **kw)
                self.new[s.asym] = tensor
                if self.isidx is not None:
                    tosym = self.new[:, self.isidx].to(int)
                    for i, x in enumerate(tosym):
                        tosym[i] = s.idx_asym_to_sym[x]
                    self.new[:, self.isidx] = tosym
            elif tensor.shape[0] != s.L:
                raise ValueError(f'unsupported shape {tensor.shape} Nasym={s.Nasym}')
            else:
                self.new = tensor
            twod = tensor.shape[:2] in [(s.Nasym, s.Nasym), (s.L, s.L)]
            if twod: self._set_attrs_twodim(s)
            else: self._set_attrs_onedim(s)

        def _set_attrs_onedim(self, s):
            self.symdims = [0]
            self.symshape = list(self.new.shape)  # type: ignore
            self.workshape = self.new.shape  # type: ignore
            # if self.new.ndim > 1:
            # self.workshape = [self.new.shape[0], -1, self.new.shape[-1]]
            self.asymshape = [s.Nasym] + list(self.new.shape[1:])  # type: ignore
            self.asushape = [s.Nasu] + list(self.new.shape[1:])  # type: ignore
            self.unsymshape = [s.Nunsym] + list(self.new.shape[1:])  # type: ignore
            self.symonlyshape = [s.Nsymonly] + list(self.new.shape[1:])  # type: ignore

        def _set_attrs_twodim(self, s):
            self.symdims = [0, 1]
            self.symshape = list(self.new.shape)  # type: ignore
            self.workshape = self.new.shape  # type: ignore
            self.asymshape = [s.Nasym, s.Nasym] + list(self.new.shape[2:])  # type: ignore
            self.asushape = [s.Nasu, s.Nasu] + list(self.new.shape[2:])  # type: ignore
            self.unsymshape = [s.Nunsym, s.Nunsym] + list(self.new.shape[2:])  # type: ignore
            self.symonlyshape = [s.Nsymonly, s.Nsymonly] + list(self.new.shape[2:])  # type: ignore

        def _make_room_for_sym_sparse(self, tensor):
            # SPARSE
            s = self.sym.idx
            assert len(self.idx) == len(tensor)  # type: ignore
            if self.sym.idx.is_sym_subsequence(self.idx):
                if self.isasym is not None: assert not self.isasym
                self.asym = False
                self.newidx = self.idx.clone()  # type: ignore
                self.new = SimpleSparseTensor(idx=self.newidx, val=self.orig.clone(), isidx=self.isidx)  # type: ignore
                assert len(self.new.idx) == len(self.new.val)
            elif self.sym.idx.is_asym_subsequence(self.idx):
                if self.isasym is not None: assert self.isasym
                symidx = self.sym.idx.symidx(self.idx)
                self.asym = True
                self.orig = th.zeros((len(symidx), *self.orig.shape[1:]), dtype=self.orig.dtype, device=self.sym.device)
                self.orig[:len(tensor)] = tensor
                self.newidx = symidx
                self.new = SimpleSparseTensor(idx=self.newidx, val=self.orig, isidx=self.isidx)  # type: ignore
                if self.isidx is not None:
                    tosym = self.new.val[:, self.isidx].to(int)
                    for i, x in enumerate(tosym):
                        tosym[i] = s.idx_asym_to_sym[x]
                    self.new.val[:, self.isidx] = tosym
                assert len(self.new.idx) == len(self.new.val)
            else:
                ic(self.sym)  # type: ignore
                ic(self.idx)  # type: ignore
                raise ValueError('sparse indices are not asym or sym compatible')
            self.symdims = []
            self.symshape = self.new.val.shape
            if self.new.val.ndim > 1:
                self.workshape = [self.new.val.shape[0], -1, self.new.val.shape[-1]]
            self.workshape = self.new.val.shape
            self.asymshape = [len(self.sym.idx.asymidx(self.newidx)), *self.new.val.shape[1:]]
            self.asushape = [len(self.sym.idx.asuidx(self.newidx)), *self.new.val.shape[1:]]
            self.unsymshape = [len(self.sym.idx.unsymidx(self.newidx)), *self.new.val.shape[1:]]
            self.symonlyshape = [len(self.sym.idx.symonlyidx(self.newidx)), *self.new.val.shape[1:]]
