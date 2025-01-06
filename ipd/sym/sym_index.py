import math

from ipd.lazy_import import lazyimport
from ipd.sym.sym_slice import SymSlice

th = lazyimport('torch')

class SymIndex:
    """A collection of slices to symmetrize.

    Attributes:
        slices            (list)  : list of SymSlice objects
        L                 (int)   : total length
        asu               (Tensor): mask of the asu
        asym              (Tensor): mask of the asu, including unsym parts
        unsym             (Tensor): the unsymmetrized part
        sym               (Tensor): masks of symsubs
        asufit            (Tensor): mask of the asymmetric unit that is fit
        asunotfit         (Tensor): mask or the asymmetric unit not fit
        Nasym             (int)   : size of combined asu
        idx_asu_to_asym   (Tensor): map asu numbering to asym numbering
        idx_asym_to_asu   (Tensor): map asym numbering to asu numbering
        idx_asym_to_sym   (Tensor): map asym numbering to sym (full) numbering
        idx_sym_to_asym   (Tensor): map sym (full) numbering to asym numbering
        idx_sym_to_asu    (Tensor): map sym (full) numbering to asu numbering
        idx_sub_to_sym[i] (Tensor): map subunit i nubering to sym (full)
        idx_sym_to_sub[i] (Tensor): map sym numberint to subunit i numbering
        idx_sym_to_subnum (Tensor): subunit number of idx in sym numbering
        idx_sub_to_asu    (Tensor): map sym numbering to asu numbering, including all subs
        contiguous        (Tensor): map to contiguous subunits
        subnum            (Tensor): subunit number of idx in sym numbering
        subunit_masks     (list)  : list of masks for each subunit
        full              (Tensor): mask of the full structure
        notsym            (Tensor): mask of the nonsymmetric part
        notasym           (Tensor): mask of the not asymmetric part
        notasu            (Tensor): mask of the not asymmetric
        notsub            (Tensor): mask of the not subunit
        sliced            (Tensor): map to sliced numbering
        contig            (Tensor): map to contiguous numbering
        kind              (Tensor): residue kind, where 0 is 'standard'
    """
    def __init__(self, nsub: int, slices, debug=False):
        '''
        Args:
            nsub: the number of subunits
            slices: a list of slices
        '''
        self.nsub = nsub
        self.orig_input = slices
        if isinstance(slices, int): slices = [slices]
        self.slices = [SymSlice.make_symslice(s) for s in slices]
        self.slices[0].fit = True  # assume fit should always be checked on the first slice
        for s in self.slices:
            s.set_nsub(self.nsub)
        self.L = len(self.slices[0].mask)
        self.sanity_check_pre()
        self.unsym = th.ones(self.L, dtype=bool)
        for s in self.slices:
            self.unsym &= ~s.mask
        self.Nunsym = th.sum(self.unsym)
        self.Nsymonly = self.L - self.Nunsym
        self.symonly = ~self.unsym
        self.asu = th.zeros(self.L, dtype=bool)
        for s in self.slices:
            self.asu |= s.asu
        self.asym = self.asu | self.unsym
        self.Nasu = th.sum(self.asu)
        self.Nasym = th.sum(self.asym)
        self.Nsym = self.L - self.Nunsym
        self.unsymasu = self.unsym[self.asym]
        for s in self.slices:
            s.set_asu(self.asym)
        self.asufit = None
        self.asunotfit = self.unsym.clone()
        for s in self.slices:
            if s.fit: self.asufit = s.asu
            else: self.asunotfit |= s.asu
        self.fit = False if self.asufit is None else th.sum(self.asufit)
        self.L = len(self.slices[0].mask)
        self.Lasuprot = self.slices[0].Lasu
        self.Lsymprot = self.slices[0].Lsym
        self.nonprot = th.arange(self.L) >= (self.slices[1].beg if len(self.slices) > 1 else 9e9)
        self.sub = self.slices[0].sym
        for s in self.slices[1:]:
            self.sub |= s.sym
        self.sym = self.sub.max(0).values

        self.idx_asu_to_asym = th.where(self.asu[self.asym].to(int))[0]
        self.idx_asym_to_asu = -th.ones(self.Nasym, dtype=int)
        self.idx_asym_to_asu[self.idx_asu_to_asym] = th.arange(self.Nasu)
        self.idx_asym_to_sym = th.arange(self.L)[self.asym]
        self.idx_sym_to_asym = -th.ones(self.L, dtype=int)
        self.idx_sym_to_asym[self.asym] = th.arange(th.sum(self.asym))
        self.idx_sym_to_asu = th.where(self.unsym, -1, self.idx_sym_to_asym)
        self.idx_sub_to_sym = th.stack([th.where(self.sub[i])[0] for i in range(self.nsub)])
        self.idx_sym_to_sub = -th.ones((self.nsub, self.L), dtype=int)
        self.idx_asu_to_sub = -th.ones((self.nsub, self.L), dtype=int)
        for i in range(self.nsub):
            self.idx_sym_to_sub[i, self.sub[i]] = th.arange(self.Nasu)
            self.idx_asu_to_sub[i, self.asu] = th.where(self.sub[i])[0]
        self.subunit_masks = [m != -1 for m in self.idx_sym_to_sub]
        self.subnum = th.max(self.sub.to(int) * th.arange(nsub)[:, None], 0).values
        self.subnum[self.unsym] = -1
        self.idx_sub_to_asu = -th.ones(self.L, dtype=int)
        n = 0
        for s in self.slices:
            self.idx_sub_to_asu[s.mask] = n + th.arange(s.Lasu).repeat(nsub)
            n += s.Lasu
        self.contiguous = -th.ones(self.Nasu * self.nsub, dtype=int)
        # ic(self.sub.shape, len(self.contiguous), self.Nasu * self.nsub)
        for i in range(self.nsub):
            self.contiguous[i * self.Nasu:(i+1) * self.Nasu] = th.where(self.sub[i])[0]

        self.full = th.ones(self.L, dtype=bool)
        self.notsym = ~self.sym
        self.notasym = ~self.asym
        self.notasu = ~self.asu
        self.notsub = ~self.sub
        self.sliced = th.arange(self.L)
        self.contig = self.contiguous

        self.kind = th.zeros(self.L, dtype=int)

        if debug:
            print('full', self.full.to(int))
            # print('sym', self.sym.to(int))
            print('asym', self.asym.to(int))
            print('asu', self.asu.to(int))
            print('unsym', self.unsym.to(int))
            print('sub', self.sub.to(int))
            # assert 0
            # print('sym\n', self.sym.to(int))
            # print('idx_asu_to_asym', self.idx_asu_to_asym)
            # print('idx_asym_to_asu', self.idx_asym_to_asu)
            # print('idx_asym_to_sym', self.idx_asym_to_sym)
            # print('idx_sym_to_asym', self.idx_sym_to_asym)
            # print('idx_sym_to_asu', self.idx_sym_to_asu)
            # print('idx_sub_to_sym\n', self.idx_sub_to_sym)
            # print('idx_sym_to_sub\n', self.idx_sym_to_sub)
            # print('idx_subnum\n', self.subnum)
            # print('idx_sub_to_asu\n', self.idx_sub_to_asu)
            # print('contiguous    ', self.contiguous)

    def to(self, device):
        for k, v in list(self.__dict__.items()):
            if isinstance(v, th.Tensor):
                self.__dict__[k] = v.to(device)

    def getsubnum(self, idx):
        return self.subnum[idx]

    def is_sym_subsequence(self, idx, strict=True):
        strict = len({int(_) for _ in idx}) == len(idx)
        idx = th.as_tensor(idx).to(self.unsym.device)
        if th.all(self.unsym[idx]): return True
        # if idx.max() < len(self.idx_asym_to_sym) and th.all(self.asym[self.idx_asym_to_sym[idx]]): return False
        idx = th.as_tensor(idx, dtype=int)
        # ic(self.asym)
        replicates = th.bincount(idx)
        replicates = replicates[replicates != 0]
        if strict: assert len(replicates.unique()) == 1
        replicates = int(replicates[0])
        idx = th.as_tensor(idx, dtype=int)
        idx = idx[~self.unsym[idx]]
        subcount = th.bincount(self.subnum[idx])
        asucount = th.bincount(self.idx_sub_to_asu[idx])
        asucount = asucount[asucount != 0]
        # ic(self.idx_sub_to_asu[idx])
        # ic(idx, subcount, asucount)
        if len(subcount) != self.nsub: return False
        if len(subcount.unique()) != 1: return False
        if strict and len(asucount) != len(idx) // replicates // self.nsub: return False
        if strict and len(asucount.unique()) != 1: return False
        return True

    def is_asym_subsequence(self, idx):
        try:
            return th.all(self.idx_asym_to_sym.cpu()[idx.cpu()] >= 0)
        except IndexError:
            return False

    def is_asu_subsequence(self, idx):
        try:
            return th.all(self.idx_asu_to_sym[idx] >= 0)  # type: ignore
        except IndexError:
            return False

    def to_contiguous(self, idx):
        """Convert a subsequence index to a contiguous-subunit subsequence
        index."""
        return th.arange(len(idx))

        # assert th.all(0 <= idx)
        # assert th.all(self.L > idx)
        # idx2 = -th.ones(self.L, dtype=int)
        # idx2[idx] = idx
        # idx2 = idx2[self.contiguous]
        # idx2 = idx2[idx2 >= 0]
        # bc = th.bincount(self.subnum[idx2])
        # bc = bc[bc>0]
        # bcbc = th.bincount(bc)
        # bcbc = bcbc[bcbc>0]
        # ic(idx,self.subnum[idx2],bc,bcbc)
        # assert len(bc) == self.nsub
        # assert len(bcbc[bcbc != 0]) == 1
        # ic(idx)
        # ic(idx2)
        # remap = -th.ones(self.L, dtype=int)
        # remap[idx] = th.arange(len(idx), dtype=int)
        # idx2 = remap[idx2]
        # ic(idx2)
        # assert 0
        # return idx2

    def symidx(self, idx):
        new = self.idx_asym_to_sym[idx]
        assert th.all(new >= 0)
        asu = new[self.asu[new]]
        for i in range(1, self.nsub):
            # ic(self.idx_asu_to_sub[i, asu])
            new = th.cat([new, self.idx_asu_to_sub[i, asu]])
        return new

    def slice2d(self, t, idx, val=None, dim=[0, 1]):
        """Utility for 2d masking, takes care of reshaping nonsense.

        Args:
            t (torch.Tensor): the tensor to slice
            idx (torch.Tensor): the indices to slice, can be bool or indices
            val (torch.Tensor): the values to set (optional)
        """
        assert dim == [0, 1]
        idx = getattr(self, idx) if isinstance(idx, str) else idx
        if idx.dtype is th.bool:
            idx = (idx[None] * idx[:, None]).reshape(-1).to(t.device)
            if val is None:
                t = t.reshape(-1, *t.shape[2:])[idx]
                return t.reshape([int(math.sqrt(len(t)))] * 2)
            if th.is_tensor(val): val = val.reshape(-1, *val.shape[2:])
            t.view(-1, *t.shape[2:])[idx] = val
            return t.reshape(t.shape)
        else:
            idx = idx.to(int)
            idx = th.cartesian_prod(idx, idx).to(t.device)
            if val is None:
                t = t[idx[:, 0], idx[:, 1]]
                return t.reshape([int(math.sqrt(len(t)))] * 2)
            if th.is_tensor(val): val = val.reshape(-1, *val.shape[2:])
            t[idx[:, 0], idx[:, 1]] = val
            return t.reshape(t.shape)

    def asymidx(self, idx):
        idx = idx.to(self.asym.device)
        return idx[self.asym[idx]]

    def asuidx(self, idx):
        idx = idx.to(self.asym.device)
        return idx[self.asu[idx]]

    def unsymidx(self, idx):
        idx = idx.to(self.asym.device)
        return idx[self.unsym[idx]]

    def symonlyidx(self, idx):
        idx = idx.to(self.asym.device)
        return idx[self.symonly[idx]]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        return self.slices[i]

    def sanity_check_pre(self):
        assert 0 <= sum(s.fit for s in self.slices[1:]) <= 1
        assert all(s.sanity_check(self.nsub) for s in self.slices)
        assert all(len(s.mask) == self.L for s in self.slices)
        for i, s1 in enumerate(self.slices):
            for j, s2 in enumerate(self.slices[:i]):
                assert th.sum(s1.mask & s2.mask) == 0

    def sanity_check(self):
        assert 0 <= sum(s.fit for s in self.slices) <= 1
        assert all(not s.fit for s in self.slices[1:])
        assert all(s.sanity_check(self.nsub) for s in self.slices)
        assert all(len(s.mask) == self.L for s in self.slices)
        for i, s1 in enumerate(self.slices):
            for j, s2 in enumerate(self.slices[:i]):
                assert th.sum(s1.mask & s2.mask) == 0
        for s in self.slices:
            assert th.sum(s.mask & self.unsym) == 0
        A = th.arange
        assert th.all(self.idx_asym_to_asu[self.idx_asu_to_asym[A(self.Nasu)]] == A(self.Nasu))
        assert th.all(self.idx_sym_to_asym[self.idx_asym_to_sym[A(self.Nasym)]] == A(self.Nasym))
        for i in range(self.nsub):
            assert th.all(self.idx_sym_to_sub[i, self.idx_sub_to_sym[i, A(self.Nasu)]] == A(self.Nasu))
            assert th.all(self.idx_sym_to_sub[i, self.subnum == i] == A(self.Nasu))
            assert th.all(self.idx_asu_to_sub[i, self.asu] == th.where(self.sub[i])[0])
        s = self.subnum[self.contiguous]
        assert all(i == j for i, j in zip(s, sorted(s)))

    def __repr__(self):
        slices = '\n    '.join(repr(s) for s in self.slices)
        return f'ipd.sym.SymIndex(nsub={self.nsub}, slices=\n    {slices}'
        # return f'ipd.sym.SymIndex(nsub={self.nsub}, slices={self.orig_input})'
