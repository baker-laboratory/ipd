from ipd.dev.lazy_import import lazyimport

th = lazyimport('torch')

from ipd.sym.sym_manager import SymmetryManager, set_default_sym_manager
import ipd
import willutil as wu

class IpdSymmetryManager(SymmetryManager):
    """
    Implements default ipd symmetry operations.

    This class is the default symmetry manager for ipd. It implements the apply_symmetry method, which is the main entry point for applying symmetry to any object. The object can be a sequence, coordinates, or a pair xyz/pair. The object will be passed to the appropriate method based on its type and shape. The method will be called with the object and all relevant symmetry parameters. The method should return the object with symmetry applied. If the object is a pair xyz,pair, the method should return a tuple of xyz,pair. If the object is a 'sequence', the method should return the sequence with the asu copies to the symmetric subs. 'sequence' can be anything with shape that starts with L
    """
    kind = 'ipd'

    def __init__(self, *a, idx=None, **kw):
        '''Create an IpdSymmetryManager'''
        super().__init__(*a, **kw)
        self._symmRs = th.tensor(wu.sym.frames(self.symid)[:, :3, :3], dtype=th.float32, device=self.device)
        self.symmsub = th.arange(min(len(self._symmRs), self.opt.max_nsub))
        if self.symid == 'I' and self.opt.max_nsub == 4:
            self.asucen = th.as_tensor(wu.sym.canonical_asu_center('icos4')[:3], device=self.device)
        else:
            self.asucen = th.as_tensor(wu.sym.canonical_asu_center(self.symid)[:3], device=self.device)
        self.asucenvec = wu.h.normalized(self.asucen)
        if 'nsub' in self.opt and self.opt.nsub:
            # assert int(self.metasymm[1][0]) == self.opt.nsub
            if self.opt.has('Lasu'):
                self.opt.L = self.opt.Lasu * self.opt.nsub
            elif self.opt.has('repeat_length'):
                self.opt.L = self.opt.repeat_length * self.opt.nsub
                assert self.opt.L % self.opt.repeat_length == 0
                self.opt.Lasu = self.opt.L // self.opt.nsub
        self.opt.nsub = len(self.symmsub)
        self.post_init()

    def apply_symmetry(self, xyz, pair, opts, update_symmsub=False, disable_all_fitting=False, **kw):
        '''Apply symmetry to an object or xyz/pair'''
        kw = wu.Bunch(kw)
        kw.disable_all_fitting = disable_all_fitting
        xyz = ipd.sym.asu_to_best_frame_if_necessary(self, xyz, **kw)
        xyz = ipd.sym.set_particle_radius_if_necessary(self, xyz, **kw)
        xyz = ipd.sym.asu_to_canon_if_necessary(self, xyz, **kw)

        assert pair is None

        xyz = th.einsum('fij,raj->frai', self._symmRs[self.symmsub],
                        xyz[:len(xyz) // self.nsub]).reshape(-1, *xyz.shape[1:])
        return xyz

set_default_sym_manager('ipd')
