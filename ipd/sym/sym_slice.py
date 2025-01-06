from dataclasses import dataclass

from ipd.lazy_import import lazyimport

th = lazyimport('torch')

@dataclass
class SymSlice:
    """A contiguous slice of an array to symmetrize."""
    @classmethod
    def make_symslice(cls, slice):
        """Construct a SymSlice from length, range, bools, or another
        SymSlice."""
        if isinstance(slice, (int, list, tuple)):
            return SymSlice(slice)
        elif isinstance(slice, SymSlice):
            return slice
        else:
            raise TypeError(f'Cannot convert {slice} to SymSlice')

    def __init__(self, mask, fit=False, kind=None):
        '''Args:
            mask: a boolean mask of the slice
            fit: whether to fit on this slice of coords
            kind: the kind of slice, prot=0, atom=1, gp=2
        Attributes:
            mask: the boolean mask of the slice
            asu: the mask of the asymmetric unit of this slice
            toasu: the mask of this slice within the asu
            beg: the beginning of the slice
            symend: the symend of the slice
            asuend: the symend of the asymmetric unit
            Lasu: the length of the asymmetric unit
            fit: whether to fit on this slice of coords
            kind: the kind of slice, prot=0, atom=1, gp=2
        '''
        if isinstance(mask, list) and len(mask) < 4: mask = tuple(mask)
        if isinstance(mask, (int, tuple)):
            if isinstance(mask, int): L, beg, symend = mask, 0, mask
            if isinstance(mask, tuple): L, beg, symend = mask
            mask = th.zeros(L, dtype=bool)  # type: ignore
            mask[range(beg, symend)] = True  # type: ignore
        self.mask = th.as_tensor(mask, dtype=bool)
        non0 = th.nonzero(self.mask)
        if len(non0):
            self.beg, self.end = int(non0[0]), int(non0[-1] + 1)
        else:
            self.beg, self.end = beg, beg  # type: ignore
        self.asuend = None
        self.fit = fit
        self.kind = kind
        self.L = len(self.mask)

    def to(self, device):
        for k, v in list(self.__dict__.items()):
            if isinstance(v, th.Tensor):
                self.__dict__[k] = v.to(device)

    def set_nsub(self, nsub):
        """Set the number of subunits in the symmetry."""
        self.Lasu = self.mask.sum() // nsub
        # ic(self.Lasu, self.mask.sum(), nsub)
        # assert self.Lasu
        if self.Lasu == 0: self.asuend, self.symend = 0, 0
        else: self.asuend = int(th.where(th.cumsum(self.mask, 0) == self.Lasu)[0][0]) + 1
        self.symend = self.beg + self.Lasu * nsub
        self.Lsym = self.Lasu * nsub
        self.asu = self.mask.clone()
        self.asu[self.asuend:] = False
        self.sym = th.zeros((nsub, self.L), dtype=bool)
        for i in range(nsub):
            self.sym[i, range(self.beg + i * self.Lasu, self.beg + (i+1) * self.Lasu)] = True

    def set_asu(self, toasu):
        """Set the asymmetric unit."""
        self.toasu = self.asu[toasu]

    def sanity_check(self, nsub):
        """Check that the slice is sane."""
        # assert self.Lasu
        N = th.sum(self.mask)
        # assert N
        if self.Lasu == 0: assert N == 0
        if N == 0: assert self.Lasu == 0
        else:
            assert N % self.Lasu == 0
            assert N // self.Lasu == nsub
        assert self.mask.ndim == 1
        assert len(self.mask) == len(self.asu)
        assert th.sum(self.mask[1:] != self.mask[:-1]) <= 2  # contiguous
        assert th.sum(self.asu[1:] != self.asu[:-1]) <= 2  # contiguous
        assert th.sum(self.asu) == self.Lasu
        assert self.end == self.symend
        return True

    def __repr__(self):
        return f'SymSlice(beg={self.beg:3}, end={self.end:3}, asuend={self.asuend:3}, symend={self.symend:3})'
