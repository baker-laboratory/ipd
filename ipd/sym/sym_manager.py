import sys
import contextlib
from abc import ABC, abstractmethod, ABCMeta
import copy
import itertools
from icecream import ic
import assertpy
import torch as th
import numpy as np
from itertools import repeat
import ipd
import willutil as wu
from willutil import h
from ipd.sym.sym_adapt import _sym_adapt
from ipd.sym import ShapeKind, ValueKind

_sym_managers = dict()
_default_sym_manager = 'base'

def set_default_sym_manager(kind):
    '''Set the default symmetry manager'''
    global _default_sym_manager
    _default_sym_manager = kind
    # ic('set_default_sym_manager', kind, _default_sym_manager)

class MetaSymManager(ABCMeta):
    '''
    Metaclass for SymmetryManager, ensures all subclasses are registered here even if in other modules
    '''
    def __init__(cls, cls_name, cls_bases, cls_dict):
        # sourcery skip: instance-method-first-arg-name
        '''Register the SymmetryManager subclass'''
        super(MetaSymManager, cls).__init__(cls_name, cls_bases, cls_dict)
        kind = cls.kind or cls_name
        if kind in _sym_managers:
            raise TypeError(f'multiple SymmetryManagers with same kind!'
                            f'trying to add {kind}:{cls_name} to:\n{_sym_managers}')
        _sym_managers[kind] = cls

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.post_init()
        return instance

class SymmetryManager(ABC, metaclass=MetaSymManager):
    """
    The SymmetryManager class encapsulates symmetry related functionality and parameters.

    It is extensible, allowing for new parameters and functionality to be added outside of the rf codebase. For example, for unbounded symmetries, translations and slide operations can be added without need to change the existing core symmetry code in ipd. With a SymmetryManager holding all relevant parameters, function signatures and other code can be cleaned up, removing up to seven parameters from many functions, and future additional parameters can be added without changing function signatures. Some places in the code now require that a SymmetryManager is present in self.sym, so a sym=None argument has been added to some classes __init__ functions. If no symmetry is specified, a no-op C1SymmetryManager is created via sym = create_sym_manager(). To create a symmetry manager based on config / command line, sym = create_sym_manager(conf) can be called. Symmetry is applied to various subjects via the __call__ operator: xyz = sym(xyz), seq = sym(seq), etc. Any SymmetryManager can also symmetrize arbitrary arrays like seq and the diffusion Indep object. Subclasses of SymmetryManager need only call super().__init__(*a,**kw) and implement the apply_symmetry method. apply_symmetry will be passed the correct slice containing only the coordinates that need to be symmetrized, already converted to the correct shape/dtype/device, along with all relevant parameters in kwargs. The kwargs will already be update based on the rfold and/or diffusion timestep as appropriate. Currently, kwargs provides the following and will also include any additions to sym.yaml. Most of these are also available via self.<whatever>, but extracting them from kwargs by adding a function argument is slightly more correct and more convenient.
    """
    kind = None

    def __init__(self, opt, device=None, **kw):
        '''Create a SymmetryManager'''
        super().__init__()
        self.opt = opt
        self.device = device or ('cuda' if th.cuda.is_available() else 'cpu')
        self.skip_keys = set()
        self._idx = None
        self._post_init_args = wu.Bunch(kw)
        self.add_properties()

    def post_init(self):
        if self._post_init_args.idx: self.idx = self._post_init_args.idx
        ipd.spy.sym_manager_created(self)

    def add_properties(self):
        locprops = dict(
            opt=[
                'nsub', 'symid', 'pseudo_cycle', 'sympair_method', 'fit', 'asu_to_best_frame',
                'symmetrize_repeats', 'sym_enabled', 'rfsym_enabled', 'sympair_enabled',
                'copy_main_block_template', 'ligand_is_symmetric'
            ],
            idx=[
                'L', 'Lasuprot', 'Lsymprot', 'masu', 'masym', 'msym', 'munsym', 'mnonprot', 'Nasu', 'Nasym',
                'Nsym', 'Nunsym'
            ],
        )
        for location, props in locprops.items():
            for prop in props:
                if location == 'opt': assertpy.assert_that(self.opt).contains(prop)

                def makeprop(loc, p):
                    return property(lambda slf: getattr(getattr(slf, loc), p))

                name = prop
                if name in ['masu', 'masym', 'msym', 'munsym', 'mnonprot']:
                    name = name[1:]
                setattr(self.__class__, prop, makeprop(location, name))

    @abstractmethod
    def apply_symmetry(self, xyz, pair=None, update_symmsub=False, **kw):
        '''All subclasses must implement this method. Calls will recieve only the part
        of the structure that needs to be symmetrized, and inputs will always be on the
        gpu, if cuda is available'''
        pass

    def __call__(self, thing=None, pair=None, key=None, isasym=None, **kw):
        '''
        This is the main entry point for applying symmetry to any object.

        The object can be a sequence, coordinates, or a pair of coordinates. The object will be
        passed to the appropriate method based on its type and shape. The method will be
        called with the object and all relevant symmetry parameters. The method should
        return the object with symmetry applied. If the object is a pair xyz,pair,
        the method should return a tuple of xyz,pair. If the object is a 'sequence', the
        method should return the sequence with the asu copies to the symmetric subs. 'sequence'
        can be anything with shape that starts with L
        '''

        kw = self.opt.to_bunch().sub(kw)
        if any([not self, key in self.skip_keys, thing is None]):
            if pair is not None: return thing, pair
            return thing

        thing = self.sym_adapt(thing, isasym=isasym)
        # print(f'sym {type(thing)}', key, flush=True)
        pair = self.sym_adapt(pair, isasym=isasym)
        assert thing
        kw.kind = thing.kind
        if pair:
            orig = thing.adapted
            newxyz, newpair = self.apply_sym_slices_xyzpair(thing, pair, **kw)
            self.move_unsym_to_match_asu(orig, newxyz)
            if self.symid.startswith('C') and self.opt.center_cyclic:
                newxyz[self.idx.kind < 1, :, 2] -= newxyz[self.idx.kind < 1, 1, 2].mean()
            newxyz = thing.reconstruct(newxyz)
            newpair = pair.reconstruct(newpair)
            newxyz[0] = ipd.sym.set_motif_placement_if_necessary(self, newxyz[0], **kw)
            # self.assert_symmetry_correct(newxyz, **kw)
            # self.assert_symmetry_correct(newpair, **kw)
            newxyz.__HAS_BEEN_SYMMETRIZED = True
            newpair.__HAS_BEEN_SYMMETRIZED = True
            return newxyz, newpair
        elif thing.kind.shapekind == ShapeKind.SEQUENCE:
            result = thing.reconstruct([self(x, **kw) for x in thing.adapted])
        elif thing.kind.shapekind == ShapeKind.MAPPING:
            result = thing.reconstruct(wu.Bunch({k: self(x, key=k, **kw) for k, x in thing.adapted.items()}))
        elif thing.kind.shapekind == ShapeKind.SCALAR:
            result = thing.orig
        else:
            result = thing.reconstruct(self.apply_sym_slices(thing, **kw))

        with contextlib.suppress(AttributeError):
            result.__HAS_BEEN_SYMMETRIZED = True
        return result

    def apply_sym_slices_xyzpair(self, xyzadaptor, pairadaptor, **kw):
        kw = wu.Bunch(kw)
        origxyz, xyz, kw['Lasu'] = self.to_contiguous(xyzadaptor, matchpair=True, **kw)
        origpair, pair, kw['Lasu'] = self.to_contiguous(pairadaptor, **kw)
        if origxyz.ndim == 2: xyz = xyz[:, None, :]
        pair = pair.squeeze(-1)
        xyz, pair = self.apply_symmetry(xyz, pair, opts=kw, **kw)
        xyz, pair = xyz.squeeze(0), pair.squeeze(0).unsqueeze(-1)
        if origxyz.ndim == 2: xyz = xyz[:, 0, :]
        xyzpair_on_subset = len(xyz) != len(origxyz)
        xyz = self.fill_from_contiguous(xyzadaptor, origxyz, xyz, matchpair=True, **kw)
        pair = self.fill_from_contiguous(pairadaptor, origpair, pair, **kw)
        xyz = self.move_unsym_to_match_asu(origxyz, xyz, move_all_nonprot=False)
        if xyzpair_on_subset:
            xyz = self(xyz, **kw.sub(fit=False, disable_all_fitting=True))
        ipd.spy.sym_xyzpair(xyz, pair=pair)
        return xyz, pair

    def apply_sym_slices(self, thing, **kw):
        adapted, contig, kw['Lasu'] = self.to_contiguous(thing, **kw)
        match thing.kind.valuekind:
            case ValueKind.XYZ:
                assert thing.kind.shapekind == ShapeKind.ONEDIM
                if adapted.ndim == 2: contig = contig[:, None, :]
                contig = self.apply_symmetry(contig, pair=None, opts=wu.Bunch(kw), **kw)
                if len(contig) == 1: contig = contig[0]
                if adapted.ndim == 2: contig = contig[:, 0, :]
            case ValueKind.INDEX:
                contig = self.apply_symmetry_index(adapted.idx, adapted.val, adapted.isidx, **kw)
            case ValueKind.BASIC:
                contig = self.apply_symmetry_scalar(thing.kind.shapekind, contig, **kw)
            case ValueKind.PAIR:
                assert thing.kind.shapekind == ShapeKind.TWODIM
                contig = self.apply_symmetry_pair(contig, **kw)
            case _:
                assert 0, f'bad kind {thing.kind}'
        if len(contig) == 1: contig = contig[0]
        result = self.fill_from_contiguous(thing, adapted, contig, **kw)
        if thing.kind.valuekind == ValueKind.XYZ:
            s = self.idx
            result = self.move_unsym_to_match_asu(adapted, result)
        return result

    def apply_symmetry_pair(self, pair, **kw):
        if not self.opt.symmetrize_repeats and not self.opt.sympair_enabled:
            return pair
        if kw['sympair_protein_only']:
            assert len(pair) == self.Lsymprot
            L, N = self.Lsymprot, self.Lasuprot
        else:
            L, N = self.Nsym, self.Nasu
        symmsub_k = self.opt.symmsub_k or self.nsub - 1
        groups = ipd.sym.find_symmsub_pair(L, N, symmsub_k, self.opt.pseudo_cycle)
        grouped = th.zeros([th.max(groups) + 1, N, N, *pair.shape[2:]], device=pair.device, dtype=pair.dtype)
        if self.opt.sympair_method == 'max': grouped -= 9e9

        for i, j in itertools.product(*[range(0, L, N)] * 2):
            group = groups[i // N, j // N]
            Nmembers = th.sum(groups == group)
            if group < 0: continue
            if self.opt.sympair_method == 'max':
                grouped[group] = th.maximum(grouped[group], pair[i:i + N, j:j + N])
            elif self.opt.sympair_method == 'mean':
                ic(grouped[group].shape)
                ic(pair[i:i + N, j:j + N].shape)
                grouped[group] += pair[i:i + N, j:j + N] / Nmembers
            else:
                raise NotImplementedError(f'unknown sympair_method {self.opt.sympair_method}')

        for i, j in itertools.product(*[range(0, L, N)] * 2):
            Nmembers = th.sum(groups == group)
            group = groups[i // N, j // N]
            if group < 0: continue
            pair[i:i + N, j:j + N] = grouped[group]

        return pair

    def apply_symmetry_index(self, idx, val, isidx, **kw):
        s = self.idx
        asu = val[s.asu[idx]]
        asuidx = idx[s.asu[idx]]
        asym = val[s.asym[idx]]
        asymidx = idx[s.asym[idx]]
        new = [asym]
        newidx = [asymidx]
        for i in range(1, self.nsub):
            new1 = asu.clone()
            new1[:, isidx] = s.idx_asu_to_sub.to(self.device)[i, asu[:, isidx].to(int)].to(asu.dtype)
            new.append(new1)
            newidx.append(s.idx_asu_to_sub.to(self.device)[i, asuidx])
        new = th.cat(new, 0)
        newidx = th.cat(newidx)
        assert th.allclose(newidx, idx)
        return new

    def apply_symmetry_scalar(self, shapekind, contig, **kw):
        N = len(contig) // self.nsub
        match shapekind:
            case ShapeKind.ONEDIM:
                for i in range(1, self.nsub):
                    contig[i * N:(i + 1) * N] = contig[:N]
            case ShapeKind.TWODIM:
                for i in range(1, self.nsub):
                    contig[i * N:(i + 1) * N, i * N:(i + 1) * N] = contig[:N, :N]
                # for i in range(1, self.nsub - 1):
                # contig[(i + 1) * N:(i + 2) * N, i * N:(i + 1) * N] = contig[N:2 * N, :N]
                # contig[i * N:(i + 1) * N, (i + 1) * N:(i + 2) * N] = contig[:N, N:2 * N]
        return contig

    def move_unsym_to_match_asu(self, orig, moved, move_all_nonprot=False):
        if not self.opt.move_unsym_with_asu: return moved
        tomove = self.munsym | (self.mnonprot if move_all_nonprot else False)
        # ic(move_all_nonprot)
        # ic(self.munsym)
        # ic(self.mnonprot)
        if not th.sum(tomove): return moved
        origasu = orig[self.masu, 0]
        movedasu = moved[self.masu, 0]
        unsym = orig[tomove]
        ic(origasu.shape, movedasu.shape, orig.shape, moved.shape)
        if len(unsym) and len(origasu) > 2 and not th.allclose(origasu, movedasu, atol=1e-3):
            rms, _, xfit = h.rmsfit(origasu, movedasu)
            moved[tomove] = h.xform(xfit, unsym)
            if rms > 1e-3:
                ic(orig)
                ic(moved)
                ic(rms)
                ic(th.where(self.idx.unsym)[0])
                ic(self.idx)
                import sys
                sys.exit()
                # wu.showme(origasu)
                # wu.showme(moveasu)
                assert rms < 1e-3
        return moved

    def to_contiguous(self, thing, matchpair=False, sympair_protein_only=None, **kw):
        if isinstance(thing, tuple):
            return tuple(self.make_contiguous(t) for t in thing)
        adapted = thing.adapted
        ctg = self.idx.contiguous
        if isinstance(adapted, np.ndarray): ctg = ctg.cpu().numpy()
        match thing.kind.shapekind:
            case ShapeKind.SPARSE:
                assert len(adapted.idx) == len(adapted.val)
                return adapted, adapted.val[self.idx.to_contiguous(adapted.idx)], self.Nasu
            case ShapeKind.ONEDIM:
                assert len(adapted) == self.L
                if sympair_protein_only and matchpair:
                    return adapted, adapted[:self.Lsymprot], self.Lsymprot // self.nsub
                return adapted, adapted[ctg], self.Nasu
            case ShapeKind.TWODIM:
                if sympair_protein_only:
                    return adapted, adapted[:self.Lsymprot, :self.Lsymprot], self.Lsymprot // self.nsub
                assert len(adapted) == self.L
                idx = th.cartesian_prod(ctg, ctg)
                shape = (len(ctg), len(ctg), *adapted.shape[2:])
                return adapted, adapted[idx[:, 0], idx[:, 1]].reshape(shape), self.Nasu

    def fill_from_contiguous(self, thing, orig, contig, matchpair=False, sympair_protein_only=None, **kw):
        ctg = self.idx.contiguous
        if isinstance(orig, np.ndarray): ctg = ctg.cpu().numpy()
        new = copy.deepcopy(orig)
        if isinstance(thing, tuple):
            return tuple(self.fill_from_contiguous(t) for t in thing)
        match thing.kind.shapekind:
            case ShapeKind.SPARSE:
                new.val[self.idx.to_contiguous(new.idx)] = contig
            case ShapeKind.ONEDIM:
                if sympair_protein_only and matchpair:
                    new[:self.Lsymprot] = contig
                else:
                    new[ctg] = contig
            case ShapeKind.TWODIM:
                if sympair_protein_only:
                    new[:self.Lsymprot, :self.Lsymprot] = contig
                else:
                    idx = th.cartesian_prod(ctg, ctg)
                    new[idx[:, 0], idx[:, 1]] = contig.reshape(-1, *contig.shape[2:])
        return new

    def extract(self, thing, mask, key=None, skip_keys=None, **kw):
        '''Extract the asu from an object. This should basically be the inverse
        of __call__. residues not involved with symmetry are included'''
        if skip_keys is None: skip_keys = []
        if key in skip_keys: return thing
        if thing is None: return None
        thing = self.sym_adapt(thing, isasym=False)
        match thing.kind.shapekind:
            case ShapeKind.SEQUENCE:
                return thing.reconstruct([self.extract(x, mask) for x in thing.adapted], **kw)
            case ShapeKind.MAPPING:
                d = {k: self.extract(x, mask, key=k, skip_keys=skip_keys) for k, x in thing.adapted.items()}
                return thing.reconstruct(d, **kw)
            case ShapeKind.ONEDIM:
                return thing.reconstruct(thing.adapted[mask], **kw)
            case ShapeKind.TWODIM:
                x = thing.adapted[mask[None] * mask[:, None]]
                # ic(x.shape, mask.sum(), mask.shape, kw)
                return thing.reconstruct(x.reshape(*[mask.sum()] * 2, *x.shape[1:]), **kw)
            case ShapeKind.SPARSE:
                assert len(thing.adapted.idx) == 0, 'spares not implemented yet'
                return thing.orig
            case _:
                raise ValueError(f'SymManager.extract: unknown thing {thing.kind}')

    def asym(self, thing, **kw):
        return self.extract(thing, self.masym, asym=True, **kw)

    def asu(self, thing, **kw):
        return self.extract(thing, self.masu, asu=True, **kw)

    def symdims(self, tensor, idx=None):
        '''try to guess which dimensions are symmetrical, could be 1 or 2'''
        if idx is None:
            symdims = th.where(th.tensor(tensor.shape) == self.idx.L)[0]
            dasym = th.where(th.tensor(tensor.shape) == self.idx.Nasym)[0]
            if len(dasym) > len(symdims): symdims = dasym
            if len(symdims) > 2: symdims = symdims[:2]
        else: symdims = th.where(th.tensor(tensor.shape) == len(idx))[0]
        natom = (14, 36)
        if len(symdims) == 2 and tensor.shape[symdims[1]] in natom and tensor.shape[-1] in (3, 4):
            symdims = symdims[:1]
        return symdims

    def tensor(self, *a, **kw):
        return ipd.sym.sym_tensor.symtensor(self, *a, **kw)

    def is_symmetrical(self, obj):
        if hasattr(obj, '__HAS_BEEN_SYMMETRIZED'): return True
        if self.idx is None: return False
        if th.is_tensor(obj):
            for n in obj.shape:
                if n == self.idx.L: return True
                if n != 1: continue
        return False

    @property
    def in_multistep_protocol(self):
        if 'rf_asym_only' not in self.opt._params: return False
        return self.opt._params['rf_asym_only'] is not False

    def multistep_adjusted_progress(self, t, T):
        return t / T, 1 / T
        asymsteps = self.opt._params['rf_asym_only'].diffuse_steps
        assert min(asymsteps) == 0 and max(asymsteps) + 1 == len(asymsteps)
        nasymstep = max(asymsteps) + 1
        if t > T - nasymstep: n, N = t - T + nasymstep, nasymstep
        else: n, N = t, T - nasymstep
        # else: n, N = t, T
        return n / N, 1 / N

    @property
    def idx(self):
        '''Return the idx of the symmetry managerm or a simple slice if None'''
        if not self._idx:
            try:
                # ic(self.opt.L,self.opt.Lasu,self.opt.nsub)
                L = self.opt.L or self.opt.Lasu * self.opt.nsub
                Lasu = self.opt.Lasu or L // self.opt.nsub
                nsub = self.opt.nsub or L // Lasu
                # ic(L,Lasu,nsub)
                self._idx = ipd.sym.SymIndex(nsub, [(L, 0, Lasu * nsub)])
            except (TypeError, AttributeError):
                return None
        return self._idx

    @idx.setter
    def idx(self, idx):
        '''Set the idx of the symmetry manager'''
        if isinstance(idx, ipd.sym.SymIndex):
            self._idx = idx
        elif self.nsub:
            self._idx = ipd.sym.SymIndex(self.nsub, idx)
        self._idx.to(self.device)

    def sym_adapt(self, thing, isasym=None):
        '''Return a SymAdapt object with metadata about the symmetry of the thing'''
        return _sym_adapt(thing, self, isasym)

    @property
    def is_dummy_sym(self):
        '''Return True if this is a dummy symmetry manager'''
        return False

    def assert_symmetry_correct(self, thing, **kw):
        if self.idx is None: return True
        return ipd.sym.symcheck(self, thing, **kw) if self else True

    def check(self, thing, **kw):
        try:
            self.assert_symmetry_correct(thing, **kw)
            return True
        except AssertionError as e:
            return False

    def reset(self):
        self.skip_keys.clear()
        self.opt.symmsub = None
        self._symmRs = self._symmRs.to(self.device)

    def is_on_symaxis(self, xyz):
        axes = wu.sym.axes(self.symid, all=True)
        onanyaxis = False
        for axis in itertools.chain(axes.values()):
            onanyaxis |= th.any(h.point_line_dist2(xyz, [0, 0, 0], axis) < 0.001)
        if not onanyaxis: return th.tensor([], dtype=int)
        if self.opt.subsymid is None:
            if len(axes) > 1: raise ValueError(f'atom on axes and dont know which subsymid {self.symid}')
            axes = axes[int(self.symid[1:])]
            if axes.ndim: axes = axes[None]
        onaxis = th.zeros(len(xyz), dtype=bool)
        for axis in axes:
            onaxis |= h.point_line_dist2(xyz, [0, 0, 0], axis) < 0.001
        return onaxis

    def update_px0(self, indep, px0):
        pass

    def __repr__(self):
        '''Return a string representation of the SymmetryManager'''
        return f'ipd.sym.{self.__class__.__name__}(symid="{self.opt.symid}", idx={self.idx})'

    def __bool__(self):
        '''Return True if symmetry is currently enabled. can be dynamic thru a run'''
        return not any([
            self.is_dummy_sym,
            not self.opt.sym_enabled,
            self.opt._in_rf2aa() and not self.opt.rfsym_enabled,
            self.opt._in_rf2aa() and self.opt.rf_asym_only,
        ])

    def __deepcopy__(self, memo):
        '''Deepcopy the SymmetryManager'''
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __contains__(self, k):
        '''Allow checking if a key is in the SymmetryManager'''
        return k in self.__dict__ or k in self.opt

    @property
    def symmRs(self):
        '''Return the symmetry matrices of the current symmsub'''
        return self._symmRs[self.symmsub]

    @property
    def allsymmRs(self):
        '''Return all symmetry matrices'''
        return self._symmRs

    def apply_initial_offset(self, x):
        return x

class C1SymmetryManager(SymmetryManager):
    """Basically a null symmetry manager, does not modify anything"""
    kind = 'C1'

    def __init__(self, opt=None, symid=None, idx=None, device=None, **kw):
        '''Create a C1SymmetryManager'''
        if symid: assert symid.upper() == 'C1'
        opt = opt or ipd.sym.get_sym_options(symid='C1')
        super().__init__(opt, device=device)
        self.opt.nsub = 1
        self._symmRs = th.eye(3)[None]
        self.symmsub = th.tensor([0])
        self.metasymm = [[th.tensor([0])], [1]]
        self.symmatrix = th.tensor([0])
        if idx: self.idx = idx

    def apply_symmetry(self, xyz, pair=None, **kw):
        '''no-op'''
        if xyz is None:
            return pair
        if pair is None:
            return xyz
        return xyz, pair

    @property
    def is_dummy_sym(self):
        '''Return True if this is a dummy symmetry manager'''
        return True

    def __bool__(self):
        '''Return False if this is a dummy symmetry manager'''
        return False

def create_sym_manager(conf=None, extra_params=None, kind=None, device=None, **kw):
    '''Create a symmetry manager based on the configuration
    Args:
        conf (dict, optional): Hydra conf
        extra_params (dict, optional): extra parameters
        kind (str, optional): symmetry manager kind
    Returns:
        SymmetryManager: a symmetry manager
    '''
    opt = ipd.sym.get_sym_options(conf, extra_params=extra_params)
    opt._add_params(**kw)
    opt = ipd.sym.process_symmetry_options(opt)
    global _default_sym_manager
    kind = kind or opt.get(kind, None) or _default_sym_manager
    if opt.symid == 'C1': kind = 'C1'
    sym = _sym_managers[kind](opt, device=device)
    ipd.sym.set_symmetry(sym)
    assert ipd.symmetrize is sym
    return sym

def check_sym_manager(sym, *a, **kw):
    '''Check if a symmetry manager is valid, and create one if it is not'''
    if sym is not None:
        assert isinstance(sym, SymmetryManager)
        return sym
    return create_sym_manager(*a, **kw)
