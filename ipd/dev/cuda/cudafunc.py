import abc

from ipd.lazy_import import lazyimport

th = lazyimport('torch')

class CudaFunc(abc.ABC):
    def __init__(self, arg, label):
        self.arg = th.as_tensor(arg).to('cuda').to(th.float32)
        self.label = label

    @abc.abstractmethod
    def reference_impl(self, dist):
        pass

    def __call__(self, dist):
        if isinstance(dist, (int, float)): dist = [dist]
        from ipd.voxel.voxel import _voxel
        dist = th.as_tensor(dist).to('cuda').to(th.float32)
        arg = th.as_tensor(self.arg).to('cuda').to(th.float32)
        return _voxel.eval_func(dist, self.label, arg)

class ClashFunc(CudaFunc):
    def __init__(self, radlow=3, radhi=4):
        assert radlow <= radhi
        super().__init__([radlow, radhi], 'clash')

    def reference_impl(self, dist):  # type: ignore
        if dist > self.arg[1]: return 0.0
        if dist < self.arg[0]: return 1.0
        return (self.arg[1] - dist) / (self.arg[1] - self.arg[0])

class ContactFunc(CudaFunc):
    def __init__(self, clashscore=10000, contactscore=-1, clashend=3, contactbeg=4, contactend=8, end=9):
        assert clashend <= contactbeg <= contactend <= end
        super().__init__([clashscore, contactscore, clashend, contactbeg, contactend, end], 'contact')

    def reference_impl(self, dist):  # type: ignore
        dist = float(dist)
        cl = float(self.arg[0])
        ct = float(self.arg[1])
        clend = float(self.arg[2])
        ctbeg = float(self.arg[3])
        ctend = float(self.arg[4])
        end = float(self.arg[5])
        if dist < clend: return cl
        if dist < ctbeg: return cl + (ct-cl) * (dist-clend) / (ctbeg-clend)
        if dist < ctend: return ct
        if dist < end: return ct * (end-dist) / (end-ctend)
        return 0
