import ipd
from ipd import h
from ipd.dev.lazy_import import lazyimport

th = lazyimport('torch')

def generate_O(coords):
    dd = dict(device=coords.device, dtype=coords.dtype)
    coords = h.point(coords, **dd)
    coords = coords.reshape(-1, coords.shape[-2], 4)
    _OCOORD = th.tensor([0.6123, 0.0101, -1.0739, 1.0000], **dd)
    stubs = h.frame(coords[:-1, 2], coords[:-1, 1], coords[1:, 0], **dd)
    o = h.xform(stubs, _OCOORD, homogout=True)
    o = th.cat([o, coords[-1:, 2]]).reshape(-1, 4)
    coords[:, 3] = o
    return coords[:, :5, :3].contiguous()

def relax_thresh_max(progress, thresh):
    mod = max(1, 0.3334 / (progress+0.001))
    return mod * thresh

def relax_thresh_min(progress, thresh):
    mod = min(1, progress / 0.3333)
    return mod * thresh

class SS(ipd.sieve.Sieve):
    from rf_diffusion.structure import assign_torch, get_bb_pydssp_seq_xyz_isgp_issm

    def __call__(self, progress, indep, xyz, cache, **kw):
        if 'ss' not in cache:
            xyz = generate_O(xyz)
            bb, isprot = get_bb_pydssp_seq_xyz_isgp_issm(indep.seq, xyz, indep.is_gp, indep.is_sm)
            cache['ss'] = assign_torch(bb).to(float)
        thresh = relax_thresh_min(progress, self.min_helix)
        val = cache['ss'][:, 0].mean()
        ok = val > thresh
        if not ok: print(f'SS  {progress:5.2f} {val:5.2f} {self.min_helix:5.2f} {thresh:5.2f}')
        return ok

class PCA(ipd.sieve.Sieve):
    def __call__(self, progress, xyz, cache, **kw):
        _, cache['pca'], _ = th.pca_lowrank(xyz[:, 1])
        thresh = relax_thresh_max(progress, self.max_bigsmallratio)
        mx, md, mn = cache['pca']
        ok = mx / mn < thresh
        if not ok: print(f'PCA {progress:5.2f} {mx/mn:5.2f} {self.max_bigsmallratio:5.2f} {thresh:5.2f} {ok}')
        return ok
