import math

import gemmi  # type: ignore
import numpy as np
import torch as th  # type: ignore
from icecream import ic
from numba import cuda

import ipd
from ipd import h

_voxel = ipd.dev.lazyimport('ipd.voxel.voxel_cuda')

class Voxel:
    def __init__(
            self,
            xyz: th.Tensor,
            resl: float = 1,
            func: ipd.dev.cuda.CudaFunc = ipd.dev.cuda.ClashFunc(3, 4),  # type: ignore
            repulsive_only: th.Tensor = None,  # type: ignore
    ):
        assert th.cuda.is_available()
        self.xyz = th.as_tensor(xyz, device='cuda', dtype=th.float32)
        self.resl = float(resl)
        self.func = func
        self.create_threads = th.tensor([32, 2, 2])
        self.repulsive_only = th.empty(0, dtype=bool) if repulsive_only is None else repulsive_only  # type: ignore
        import ipd.samp.sampling_cuda
        self.boundcen, self.boundrad = ipd.samp.bounding_sphere(self.xyz)
        self.boundcen = h.point(self.boundcen.to('cuda'))
        self.create_grid()

    def create_grid(self):
        self.grid, self.lb = _voxel.create_voxel_grid(xyz=self.xyz,
                                                      resl=self.resl,
                                                      func=self.func.label,
                                                      funcarg=self.func.arg,
                                                      nthread=self.create_threads,
                                                      repulsive_only=self.repulsive_only)
        if not th.all(self.lb + self.resl < self.xyz.min(0)[0]):
            ic(self.lb)
            assert th.all(self.lb + self.resl < self.xyz.min(0)[0])

    def score(
        self,
        xyz,
        xyzpos=None,
        voxpos=None,
        symx=None,
        symclashdist=0,
        nthread=None,
        isinv=False,
        outerprod=False,
        boundscheck=False,
        repulsive_only=None,
    ):
        if symx is None:
            assert symclashdist == 0
            symx = th.empty(0, device='cuda')
        if xyzpos is None: xyzpos = th.eye(4)
        if voxpos is None: voxpos = th.eye(4)
        if xyzpos.ndim == 2 and voxpos.ndim == 2: outshape = ()
        elif xyzpos.ndim == 2: outshape = (len(voxpos), )
        elif voxpos.ndim == 2: outshape = (len(xyzpos), )
        else: outshape = (len(voxpos), len(xyzpos))
        if xyzpos.ndim == 2: xyzpos = xyzpos[None]
        xyzpos = xyzpos.to('cuda').to(th.float32)
        if voxpos.ndim == 2: voxpos = voxpos[None]
        voxpos = voxpos.to('cuda').to(th.float32)
        if not outerprod:
            outshape = (max(len(voxpos), len(xyzpos)), )
            if len(voxpos) == len(xyzpos):
                xyzpos = th.linalg.solve(voxpos, xyzpos).contiguous()
                voxpos = th.eye(4, device='cuda', dtype=th.float32)[None]
            else:
                len(voxpos) == 1 or len(xyzpos) == 1  # type: ignore
        if nthread is None:
            if len(xyzpos) == 1: nthread = th.tensor([256, 1, 1])
            elif len(voxpos) == 1: nthread = th.tensor([1, 256, 1])
            else: nthread = th.tensor([16, 16, 1])
        if isinv: voxposinv = voxpos
        else: voxposinv = th.linalg.inv(voxpos).contiguous()
        if boundscheck:
            assert outerprod is False
            assert isinv is False
            cen1, rad1 = voxpos @ self.boundcen, self.boundrad
            if xyz is self.xyz:
                cen2, rad2 = self.boundcen, self.boundrad
            else:
                cen2, rad2 = ipd.samp.bounding_sphere(xyz)
            cen2 = xyzpos @ h.point(cen2.to('cuda'))
            pad = self.func.arg[-1] + self.resl * math.sqrt(3) / 2
            ok = h.norm2(cen1 - cen2) < (rad1 + rad2 + pad)**2
            # ic(ok.sum() / len(ok))
            if ok.sum() == 0: return th.zeros(1, device='cuda')
            if len(voxposinv) > 1: voxposinv = voxposinv[ok]
            if len(xyzpos) > 1: xyzpos = xyzpos[ok]
            # ic(cen1.shape, cen2.shape, rad1, rad2)
        if repulsive_only is None: repulsive_only = th.empty(0, dtype=bool)  # type: ignore
        score = _voxel.score_voxel_grid(
            self.grid,
            voxposinv,
            xyz,
            xyzpos,
            self.lb,
            self.resl,
            nthread,
            symx=symx,
            symclashdist=symclashdist,
            repulsive_only=repulsive_only,
        ).reshape(-1)
        if boundscheck:
            # assert th.allclose(score[~ok], th.tensor(0.0), atol=1e-3)
            sok, score = score, th.zeros(len(ok), device='cuda')  # type: ignore
            score[ok] = sok  # type: ignore
        return score.reshape(outshape)

    def score_per_atom(self, xyz):
        assert xyz.ndim == 2
        return th.as_tensor([self.score(pt.to('cuda')[None]) for pt in xyz])

    def dump_ccp4(self, fname):
        self.ccp4().write_ccp4_map(fname)

    def ccp4(self):
        npgrid = self.grid.to(th.float32).cpu().numpy()
        grid = gemmi.FloatGrid(npgrid)
        grid.set_size(*npgrid.shape)
        bound = np.array(grid.shape) * self.resl
        grid.set_unit_cell(gemmi.UnitCell(*bound, 90, 90, 90))  # type: ignore
        assert grid.shape == npgrid.shape
        assert np.allclose(grid.spacing, self.resl)
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = grid
        ccp4.update_ccp4_header()
        return ccp4
        # print(ccp4)
        # ccp4.write_ccp4_map(fname)

@cuda.jit('void(f4[:, :], f4[:], f4[:], i4, f4, float16[:, :, :])', cache=True, fastmath=True)
def create_voxel_numba(xyz, lb, rad, irad, resl, vox):
    ixyz = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # type: ignore
    if ixyz >= len(xyz): return
    ix = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # type: ignore
    iy = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z  # type: ignore
    # icen = ((xyz[ixyz] - lb) / resl).astype(int)
    icenx = int(xyz[ixyz, 0] - lb[0]) / resl
    iceny = int(xyz[ixyz, 1] - lb[1]) / resl
    icenz = int(xyz[ixyz, 2] - lb[2]) / resl
    i = int(ix - irad - 1 + icenx)
    j = int(iy - irad - 1 + iceny)

    for k in range(icenz - irad, icenz + irad + 1):
        bcenx = lb[0] + float(i) * resl
        bceny = lb[1] + float(j) * resl
        bcenz = lb[2] + float(k) * resl
        dist = math.sqrt((bcenx - xyz[ixyz, 0])**2 + (bceny - xyz[ixyz, 1])**2 + (bcenz - xyz[ixyz, 2])**2)
        # val = 1.0 if dist < rad else 0.0
        if dist > rad[1]: val = 0.0
        elif dist < rad[0]: val = 1.0
        else: val = (rad[1] - dist) / (rad[1] - rad[0])
        if val: vox[i, j, k] += val
