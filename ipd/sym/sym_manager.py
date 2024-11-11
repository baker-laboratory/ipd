import contextlib
import copy
import itertools
import typing
from abc import ABC, abstractmethod

import assertpy
import numpy as np
from icecream import ic
import torch as th

import ipd
from ipd.sym import ShapeKind, ValueKind
from ipd.sym.sym_adapt import _sym_adapt

T = typing.TypeVar('T')

class SymmetryManager(ABC, metaclass=ipd.sym.sym_factory.MetaSymManager):
    """The SymmetryManager class encapsulates symmetry related functionality
    and parameters.

    It is extensible, allowing for new parameters and functionality to
    be added outside of the rf codebase. For example, for unbounded
    symmetries, translations and slide operations can be added without
    need to change the existing core symmetry code in ipd. With a
    SymmetryManager holding all relevant parameters, function signatures
    and other code can be cleaned up, removing up to seven parameters
    from many functions, and future additional parameters can be added
    without changing function signatures. Some places in the code now
    require that a SymmetryManager is present in self.sym, so a sym=None
    argument has been added to some classes __init__ functions. If no
    symmetry is specified, a no-op C1SymmetryManager is created via sym
    = create_sym_manager(). To create a symmetry manager based on config
    / command line, sym = create_sym_manager(conf) can be called.
    Symmetry is applied to various subjects via the __call__ operator:
    xyz = sym(xyz), seq = sym(seq), etc. Any SymmetryManager can also
    symmetrize arbitrary arrays like seq and the diffusion Indep object.
    Subclasses of SymmetryManager need only call
    super().__init__(*a,**kw) and implement the apply_symmetry method.
    apply_symmetry will be passed the correct slice containing only the
    coordinates that need to be symmetrized, already converted to the
    correct shape/dtype/device, along with all relevant parameters in
    kwargs. The kwargs will already be update based on the rfold and/or
    diffusion timestep as appropriate. Currently, kwargs provides the
    following and will also include any additions to sym.yaml. Most of
    these are also available via self.<whatever>, but extracting them
    from kwargs by adding a function argument is slightly more correct
    and more convenient.
    """
    kind = None

    def __init__(self, opt, symid=None, device=None, **kw) -> None:
        """Create a SymmetryManager."""
        super().__init__()
        self.opt = opt
        # self.opt.symid =symid or self.opt.symid
        self.device = device or ('cuda' if th.cuda.is_available() else 'cpu')
        self.skip_keys = set()
        self._idx = None
        self._post_init_args = ipd.dev.Bunch(kw)
        self._frames = None
        self.add_properties()
        self.init(**kw)

    @abstractmethod
    def init(self, **kw) -> None:
        pass

    @abstractmethod
    def apply_symmetry(self, xyz: th.Tensor, pair=None, update_symmsub=False, **kw) -> th.Tensor:
        """All subclasses must implement this method.

        Calls will recieve only the part of the structure that needs to
        be symmetrized, and inputs will always be on the gpu, if cuda is
        available
        """
        pass

    def setup_for_symmetry(self, thing: T) -> T:
        """Default implementation no-op."""
        return thing

    def post_init(self) -> None:
        if 'idx' in self._post_init_args: self.idx = self._post_init_args.idx
        ipd.hub.sym_manager_created(self)

    def add_properties(self):
        locprops = dict(
            opt=[
                'nsub', 'symid', 'pseudo_cycle', 'sympair_method', 'fit', 'asu_to_best_frame', 'symmetrize_repeats',
                'sym_enabled', 'rfsym_enabled', 'sympair_enabled', 'copy_main_block_template', 'ligand_is_symmetric'
            ],
            idx=[
                'L', 'Lasuprot', 'Lsymprot', 'masu', 'masym', 'msym', 'munsym', 'mnonprot', 'Nasu', 'Nasym', 'Nsym',
                'Nunsym'
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

    def __call__(self, thing: T = None, pair=None, key=None, isasym=None, **kw) -> T:
        """This is the main entry point for applying symmetry to any object.

        The object can be a sequence, coordinates, or a pair of
        coordinates. The object will be passed to the appropriate method
        based on its type and shape. The method will be called with the
        object and all relevant symmetry parameters. The method should
        return the object with symmetry applied. If the object is a pair
        xyz,pair, the method should return a tuple of xyz,pair. If the
        object is a 'sequence', the method should return the sequence
        with the asu copies to the symmetric subs. 'sequence' can be
        anything with shape that starts with L
        """

        kw = self.opt.to_bunch().sub(kw)
        if any([not self, key in self.skip_keys, thing is None]):
            if pair is not None: return thing, pair
            return thing

        self._verify_index(thing)

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
            self.mark_symmetrical(newxyz, newpair)
            return newxyz, newpair
        elif thing.kind.shapekind == ShapeKind.SEQUENCE:
            result = thing.reconstruct([self(x, **kw) for x in thing.adapted])
        elif thing.kind.shapekind == ShapeKind.MAPPING:
            result = thing.reconstruct(ipd.Bunch({k: self(x, key=k, **kw) for k, x in thing.adapted.items()}))
        elif thing.kind.shapekind == ShapeKind.SCALAR:
            result = thing.orig
        elif th.is_tensor(thing.orig) and thing.orig.shape[-1] == 0:
            result = thing.orig
        else:
            result = thing.reconstruct(self.apply_sym_slices(thing, **kw))
        self.mark_symmetrical(result)
        return result

    def apply_symmetry_xyz_maybe_pair(self, xyz, pair=None, origxyz=None, **kw):
        xyz = self.apply_symmetry(xyz, pair=pair, opts=ipd.dev.Bunch(kw, _strict=False), **kw)
        if isinstance(xyz, tuple): xyz, pair = xyz
        if origxyz.ndim == 2: xyz = xyz[:, None, :]
        if len(xyz) == 1: xyz = xyz[0]
        return xyz if pair is None else (xyz, pair)

    def apply_sym_slices_xyzpair(self, xyzadaptor, pairadaptor, **kw):
        kw = ipd.dev.Bunch(kw)
        origxyz, xyz, kw['Lasu'] = self.to_contiguous(xyzadaptor, matchpair=True, **kw)
        origpair, pair, kw['Lasu'] = self.to_contiguous(pairadaptor, **kw)
        if origxyz.ndim == 2: xyz = xyz[:, None, :]
        pair = pair.squeeze(-1)
        xyz, pair = self.apply_symmetry_xyz_maybe_pair(xyz, pair=pair, origxyz=origxyz, **kw)
        xyz, pair = xyz.squeeze(0), pair.squeeze(0).unsqueeze(-1)
        xyzpair_on_subset = len(xyz) != len(origxyz)
        xyz = self.fill_from_contiguous(xyzadaptor, origxyz, xyz, matchpair=True, **kw)
        pair = self.fill_from_contiguous(pairadaptor, origpair, pair, **kw)
        xyz = self.move_unsym_to_match_asu(origxyz, xyz, move_all_nonprot=False)
        if xyzpair_on_subset:
            xyz = self(xyz, **kw.sub(fit=False, disable_all_fitting=True))
        ipd.hub.sym_xyzpair(xyz, pair=pair)
        return xyz, pair

    def apply_sym_slices(self, thing: T, **kw) -> T:
        adapted, contig, kw['Lasu'] = self.to_contiguous(thing, **kw)
        match thing.kind.valuekind:
            case ValueKind.XYZ:
                assert thing.kind.shapekind == ShapeKind.ONEDIM
                contig = self.apply_symmetry_xyz_maybe_pair(contig, pari=None, origxyz=adapted, **kw)
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
            result = self.move_unsym_to_match_asu(adapted, result)
        return result

    def apply_symmetry_pair(self, pair: th.Tensor, **kw) -> th.Tensor:
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
                # ic(grouped[group].shape)
                # ic(pair[i:i + N, j:j + N].shape)
                grouped[group] += pair[i:i + N, j:j + N] / Nmembers
            else:
                raise NotImplementedError(f'unknown sympair_method {self.opt.sympair_method}')

        for i, j in itertools.product(*[range(0, L, N)] * 2):
            Nmembers = th.sum(groups == group)
            group = groups[i // N, j // N]
            if group < 0: continue
            pair[i:i + N, j:j + N] = grouped[group]

        return pair

    def apply_symmetry_index(self, idx: T, val, isidx, **kw) -> T:
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

    def apply_symmetry_scalar(self, shapekind: ShapeKind, contig: th.Tensor, **kw) -> th.Tensor:
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
        # ic(origasu.shape, movedasu.shape, orig.shape, moved.shape)
        if len(unsym) and len(origasu) > 2 and not th.allclose(origasu, movedasu, atol=1e-3):
            rms, _, xfit = ipd.h.rmsfit(origasu, movedasu)
            moved[tomove] = ipd.h.xform(xfit, unsym)
            if rms > 1e-3:
                ic(orig)
                ic(moved)
                ic(rms)
                ic(th.where(self.idx.unsym)[0])
                ic(self.idx)
                import sys
                sys.exit()
                # ipd.showme(origasu)
                # ipd.showme(moveasu)
                assert rms < 1e-3
        return moved

    def to_contiguous(self, thing, matchpair=False, sympair_protein_only=None, **kw) -> tuple[th.Tensor, th.Tensor, int]:
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
            case _:
                raise ValueError(f'SymManager.to_contiguous: unknown thing {type(thing)}')

    def fill_from_contiguous(self, thing, orig, contig, matchpair=False, sympair_protein_only=None, **kw) -> th.Tensor:
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

    def extract(self, thing: T, mask: th.Tensor, key=None, skip_keys=None, **kw) -> T:
        """Extract the asu from an object.

        This should basically be the inverse of __call__. residues not
        involved with symmetry are included
        """
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

    def asym(self, thing: T, **kw) -> T:
        return self.extract(thing, self.masym, asym=True, **kw)

    def asu(self, thing: T, **kw) -> T:
        return self.extract(thing, self.masu, asu=True, **kw)

    def symdims(self, tensor: th.Tensor, idx=None) -> th.Tensor:
        """Try to guess which dimensions are symmetrical, could be 1 or 2."""
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

    def mark_symmetrical(self, *args):
        for a in args:
            with contextlib.suppress(AttributeError):
                a.__HAS_BEEN_SYMMETRIZED = True

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

    def multistep_adjusted_progress(self, t: float, T: float) -> tuple[float, float]:
        asymsteps = self.opt._params['rf_asym_only'].diffuse_steps
        assert min(asymsteps) == 0 and max(asymsteps) + 1 == len(asymsteps)
        nasymstep = max(asymsteps) + 1
        if t > T - nasymstep: n, N = t - T + nasymstep, nasymstep
        else: n, N = t, T - nasymstep
        # else: n, N = t, T
        # return n / N, 1 / Neo
        raise NotImplementedError('multistep_adjusted_progress not implemented')

    @property
    def idx(self) -> ipd.sym.SymIndex:
        """Return the idx of the symmetry managerm or a simple slice if
        None."""
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
        """Set the idx of the symmetry manager."""
        if isinstance(idx, ipd.sym.SymIndex):
            self._idx = idx
        elif self.nsub:
            self._idx = ipd.sym.SymIndex(self.nsub, idx)
        self._idx.to(self.device)

    def _verify_index(self, thing):
        if self.idx:
            # TODO: verify self.idx compatible with thing
            pass
        else:
            if isinstance(thing, th.Tensor):
                self.idx = len(thing)

    def sym_adapt(self, thing, isasym=None) -> ipd.sym.SymAdapt:
        """Return a SymAdapt object with metadata about the symmetry of the
        thing."""
        return _sym_adapt(thing, self, isasym)

    @property
    def is_dummy_sym(self):
        """Return True if this is a dummy symmetry manager."""
        return False

    def assert_symmetry_correct(self, thing, **kw):
        if self.idx is None: return True
        return ipd.sym.symcheck(self, thing, **kw) if self else True

    def check(self, thing, **kw):
        try:
            self.assert_symmetry_correct(thing, **kw)
            return True
        except AssertionError:
            return False

    def reset(self):
        self.skip_keys.clear()
        self.opt.symmsub = None
        self._symmRs = self._symmRs.to(self.device)

    def is_on_symaxis(self, xyz):
        if len(xyz) == 0: return None
        axes = ipd.sym.axes(self.symid, all=True)
        onanyaxis = False
        for axis in itertools.chain(axes.values()):
            onanyaxis |= th.any(ipd.h.point_line_dist2(xyz, [0, 0, 0], axis) < 0.001)
        if not onanyaxis: return th.tensor([], dtype=int)
        if self.opt.subsymid is None:
            if len(axes) > 1: raise ValueError(f'atom on axes and dont know which subsymid {self.symid}')
            axes = axes[int(self.symid[1:])]
            if axes.ndim: axes = axes[None]
        onaxis = th.zeros(len(xyz), dtype=bool)
        for axis in axes:
            onaxis |= ipd.h.point_line_dist2(xyz, [0, 0, 0], axis) < 0.001
        return onaxis

    def update_px0(self, indep, px0):
        pass

    def __repr__(self):
        """Return a string representation of the SymmetryManager."""
        return f'ipd.sym.{self.__class__.__name__}(symid="{self.opt.symid}", idx={self.idx})'

    def __bool__(self):
        """Return True if symmetry is currently enabled.

        can be dynamic thru a run
        """
        return not any([
            self.is_dummy_sym,
            not self.opt.sym_enabled,
            self.opt._in_rf2aa() and not self.opt.rfsym_enabled,
            self.opt._in_rf2aa() and self.opt.rf_asym_only,
        ])

    def __deepcopy__(self, memo):
        """Deepcopy the SymmetryManager."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __contains__(self, k):
        """Allow checking if a key is in the SymmetryManager."""
        return k in self.__dict__ or k in self.opt

    @property
    def symmRs(self):
        """Return the symmetry matrices of the current symmsub."""
        return self._symmRs[self.symmsub]

    @property
    def allsymmRs(self):
        """Return all symmetry matrices."""
        return self._symmRs

    @property
    def full_symmetry(self):
        if hasattr(self, '_full_symmetry'):
            return self._full_symmetry
        return self.allsymmRs

    def apply_initial_offset(self, x):
        return x + self.asucenvec.to(x.device).to(x.dtype) * self.opt.start_radius

    @property
    def frames(self):
        return self._frames

    @frames.setter
    def frames(self, frames):
        assert frames.shape[-2:] == (4, 4) and frames.ndim == 3
        self._frames = th.as_tensor(frames, device=self.device, dtype=th.float32)
        self.opt.nsub = len(frames)

    def recenter_by_chain(self, xyz, target):
        xyz = xyz.clone()
        for m in self.idx.subunit_masks:
            tgtcom = target[m, 1].mean(0)
            xyzcom = xyz[m, 1].mean(0)
            xyz[m] += tgtcom - xyzcom
        return xyz

class C1SymmetryManager(SymmetryManager):
    """Basically a null symmetry manager, does not modify anything."""
    kind = 'C1'

    def init(self, opt=None, symid=None, idx=None, device=None, **kw):
        super().init(**kw)
        """Create a C1SymmetryManager."""
        if symid: assert symid.upper() == 'C1'
        opt = opt or ipd.sym.get_sym_options(symid='C1')
        self.opt.nsub = 1
        self._symmRs = th.eye(3)[None]
        self.symmsub = th.tensor([0])
        self.metasymm = [[th.tensor([0])], [1]]
        self.symmatrix = th.tensor([0])
        if idx: self.idx = idx

    def apply_symmetry(self, xyz, pair=None, **kw):
        """no-op."""
        if xyz is None:
            return pair
        if pair is None:
            return xyz
        return xyz, pair

    @property
    def is_dummy_sym(self):
        """Return True if this is a dummy symmetry manager."""
        return True

    def __bool__(self):
        """Return False if this is a dummy symmetry manager."""
        return False
