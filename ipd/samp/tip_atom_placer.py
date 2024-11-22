import numpy as np

from ipd.lazy_import import lazyimport

th = lazyimport('torch')

import dataclasses

from icecream import ic

import ipd
from ipd import h

ic.configureOutput(includeContext=False)

_sampling = ipd.dev.lazyimport('ipd.samp.sampling_cuda')

def rayframe(rays, cross=None, primary='z', device='cpu'):
    ori = rays[:, :, 1]
    cen = rays[:, :, 0] + 2.7*ori
    if cross is None: cross = h.randvec()
    return h.frame(cen, cen - ori, cen + cross, primary=primary, device=device)

@dataclasses.dataclass
class TipAtom:
    xyz: th.Tensor  # type: ignore
    don: th.Tensor  # shape = N, 4, 2 ray  # type: ignore
    acc: th.Tensor  # shape = N, 4, 2 ray  # type: ignore
    resn: str
    aname: np.array  # type: ignore
    _vizpos: th.Tensor = None  # type: ignore

    def __post_init__(self):
        self.xyz = th.as_tensor(self.xyz)
        if self.don is None and self.acc is None:
            # self.xyz = h.xform(h.inv(h.frame(self.xyz[0], self.xyz[1], self.xyz[2])), self.xyz)
            self.don, self.acc = find_don_acc(self.resn, self.aname, self.xyz)

    def donacc_frames(self):
        return h.inv(rayframe(self.don)), h.inv(rayframe(self.acc))

class TipAtomTarget:
    @staticmethod
    def from_pdb(fname, tgtres=None, clashthresh=2):
        pdb = ipd.pdb.readpdb(fname)
        xyz = th.stack([th.as_tensor(pdb.df.x), th.as_tensor(pdb.df.y), th.as_tensor(pdb.df.z)], dim=1)
        rn = [s.decode() for s in pdb.df.rn]
        an = [s.decode() for s in pdb.df.an]
        return ipd.samp.TipAtomTarget(pdb.df.ri, xyz, rn, an, tgtres, clashthresh, source=fname)

    def __init__(self, ires, xyz, resn, aname, tgtres, clashthresh, source=None):
        idx = [an[0] != 'H' for an in aname]
        self.aname = np.asarray(aname)[idx]
        self.ires = th.as_tensor(ires, dtype=th.int32)[idx]
        self.xyz = th.as_tensor(xyz)[idx]
        self.resn = np.asarray(resn)[idx]
        self.tgtres = th.unique(self.ires) if tgtres is None else tgtres
        self.source = source
        # self.vox = ipd.voxel.Voxel(self.xyz, resl=0.5, func=ipd.dev.cuda.ClashFunc(2.8, 3.3))
        self.find_don_acc(clashthresh)

    def place_tip_atoms(self, tips, **kw):
        vox = th.zeros((1, 1, 1), device='cuda')
        don = self.don.to('cuda')  # type: ignore
        acc = self.acc.to('cuda')  # type: ignore
        fdon, facc = self.donacc_frames('cuda')
        # ipd.showme(self, name='ref')
        for tip in tips:
            ic(tip.xyz.shape)
            _sampling.tip_atom_placer(vox, don, acc, tip.xyz.to('cuda'), tip.don.to('cuda'), tip.acc.to('cuda'), kw)
            for fd, fa in zip(fdon, facc):
                cgo = list()
                tip_atom_cgo(fd, fa, tip, cgo)
                pymol.cmd.load_cgo(cgo, tip.resn)
            # ipd.showme(tip)
        # assert 0

    def find_don_acc(self, clashthresh):
        self.don, self.acc = list(), list()
        for ir in self.tgtres:
            idx = self.ires == ir
            don, acc = find_don_acc(str(self.resn[idx][0]), self.aname[idx], self.xyz[idx])
            self.don.append(don)
            self.acc.append(acc)
        self.don = th.cat(self.don)
        self.acc = th.cat(self.acc)
        donpt = self.don[:, :3, 0] + 3 * self.don[:, :3, 1]
        accpt = self.acc[:, :3, 0] + 3 * self.acc[:, :3, 1]
        ddon = h.norm(donpt[:, None] - self.xyz[None]).min(1).values
        dacc = h.norm(accpt[:, None] - self.xyz[None]).min(1).values
        self.don = self.don[ddon >= clashthresh]
        self.acc = self.acc[dacc >= clashthresh]

        # ipd.showme(donpt)
        # ipd.showme(accpt)
        # donclash = self.vox.score_per_atom(donpt)
        # accclash = self.vox.score_per_atom(accpt)
        # ic(donclash, accclash, clashthresh)
        # self.don = self.don[donclash < clashthresh]
        # self.acc = self.acc[accclash < clashthresh]

    def donacc_frames(self, device='cpu'):
        return rayframe(self.don, device=device), rayframe(self.acc, device=device)

def get_tip_atom_groups():
    tip_atom_start = dict(asn=5, gln=6, asp=5, glu=6)
    rotset = ipd.pdb.rotamer.get_rotamerset()
    tips = list()
    # for resn in 'asp glu asn gln'.split():
    for resn in 'asp asn'.split():
        i = tip_atom_start[resn]
        rots = rotset.rotamers(resn.upper())
        tips.append(TipAtom(rots.coords[0, i:], None, None, resn, [x.strip() for x in rots.atomname[i:]]))
    return tips

def acc_doublebond_O(aname, xyz, oname, cname, bname):
    P = th.pi
    o = xyz[aname == oname]
    c = xyz[aname == cname]
    b = xyz[aname == bname]
    ax = h.cross(o - c, b - c)
    accp = h.point(o).repeat(3, 1)
    accv = h.normvec(h.xform(h.rot(ax, [P / 3, 0, -P / 3]), o - c).reshape(-1, 3))
    acc = th.stack([accp, accv], dim=2)
    assert acc.shape[1:] == (4, 2)
    return acc

def don_nh2(aname, xyz, nname, cname, bname):
    P = th.pi
    n = xyz[aname == nname]
    c = xyz[aname == cname]
    b = xyz[aname == bname]
    ax = h.cross(n - c, b - c)
    donp = h.point(n).repeat(2, 1)
    donv = h.normvec(h.xform(h.rot(ax, [P / 3, -P / 3]), n - c).reshape(-1, 3))
    don = th.stack([donp, donv], dim=2)
    assert don.shape[1:] == (4, 2)
    return don

def acc_n_ring(aname, xyz, nname, bname1, bname2):
    n = xyz[aname == nname]
    b1 = xyz[aname == bname1]
    b2 = xyz[aname == bname2]
    h.cross(n - b1, n - b2)
    accp = h.point(n)
    accv = h.normvec(n - (b1+b2) / 2)
    acc = th.stack([accp, accv], dim=2)
    assert acc.shape[1:] == (4, 2)
    return acc

def don_n_ring(aname, xyz, nname, bname1, bname2):
    return acc_n_ring(aname, xyz, nname, bname1, bname2)

def find_don_acc(resn, aname, xyz):
    aname = np.asarray(aname)
    don, acc = th.zeros((0, 4, 2)), th.zeros((0, 4, 2))
    if resn in 'asp glu'.split():
        acc = th.empty(6, 4, 2)
        acc[:3] = acc_doublebond_O(aname, xyz, aname[-1], aname[-3], aname[-2])
        acc[3:] = acc_doublebond_O(aname, xyz, aname[-2], aname[-3], aname[-1])
    elif resn in 'asn gln'.split():
        assert aname[-1].startswith('N')
        acc = acc_doublebond_O(aname, xyz, aname[-2], aname[-3], aname[-1])
        don = don_nh2(aname, xyz, aname[-1], aname[-3], aname[-2])
    elif resn == 'DC':
        acc = acc_doublebond_O(aname, xyz, 'O2', 'C2', 'N1')
        don = don_nh2(aname, xyz, 'N4', 'C4', 'C5')
    elif resn == 'DG':
        don = th.empty((3, 4, 2))
        don[:2] = don_nh2(aname, xyz, 'N2', 'C2', 'N3')
        don[2] = don_n_ring(aname, xyz, 'N1', 'C2', 'C6')
        acc = th.empty((5, 4, 2))
        acc[0] = acc_n_ring(aname, xyz, 'N7', 'C8', 'C5')
        acc[1] = acc_n_ring(aname, xyz, 'N3', 'C2', 'C4')
        acc[2:] = acc_doublebond_O(aname, xyz, 'O6', 'C6', 'C5')
    elif resn == 'DT':
        acc = th.empty((6, 4, 2))
        acc[:3] = acc_doublebond_O(aname, xyz, 'O2', 'C2', 'N1')
        acc[3:] = acc_doublebond_O(aname, xyz, 'O4', 'C4', 'C5')
        don = don_n_ring(aname, xyz, 'N3', 'C2', 'C4')
    elif resn == 'DA':
        acc = th.empty((3, 4, 2))
        don = don_nh2(aname, xyz, 'N6', 'C6', 'N1')  # sketchy
        acc[0] = acc_n_ring(aname, xyz, 'N1', 'C2', 'C6')
        acc[1] = acc_n_ring(aname, xyz, 'N3', 'C2', 'C4')
        acc[2] = acc_n_ring(aname, xyz, 'N7', 'C5', 'C8')

    don[:, :, 1] = h.normalized(don[:, :, 1])
    acc[:, :, 1] = h.normalized(acc[:, :, 1])
    assert don.shape[1:] == (4, 2)
    assert acc.shape[1:] == (4, 2)
    return don, acc

try:
    import pymol  # type: ignore

    def tip_atom_cgo(fdon, facc, tip, cgo, **kw):
        if fdon.ndim == 2: fdon = fdon[None]
        if facc.ndim == 2: facc = facc[None]
        atomcol = dict(C=(1, 1, 1), O=(1, 0, 0), N=(0, 0, 1))
        acol = [atomcol[n[0]] for n in tip.aname]
        for fd, fa in zip(fdon.cpu(), facc.cpu()):
            for rays, rcol, f, ftip in zip([tip.don, tip.acc], [(1, 1, 1), (1, 0, 0)], [fd, fa], tip.donacc_frames()):
                if len(rays) == 0: continue
                for ray, ft in zip(rays, ftip):
                    frame = f @ ft
                    for xyz, col in zip(h.xform(frame, tip.xyz), acol):
                        cgo += ipd.viz.cgo_sphere(cen=xyz, rad=0.5, col=col)
                    beg = ipd.homog.hxform(ft, ray[:, 0])
                    end = ipd.homog.hxform(ft, ray[:, 0] + 2 * ray[:, 1])
                    beg = ipd.homog.hxform(frame, ray[:, 0])
                    end = ipd.homog.hxform(frame, ray[:, 0] + 2 * ray[:, 1])
                    cgo += ipd.viz.cgo_cyl(beg, end, 0.1, col=rcol)

    @ipd.viz.pymol_scene
    @ipd.viz.pymol_load.register(TipAtom)
    def pymol_load_TipAtom(tip, name='TipAtom', **kw):
        cgo = list()
        tip_atom_cgo(th.eye(4), tip, cgo, **kw)
        pymol.cmd.load_cgo(cgo, name)

    @ipd.viz.pymol_scene
    @ipd.viz.pymol_load.register(TipAtomTarget)
    def pymol_load_TipAtomTarget(tgt, name='TipAtomTarget', **kw):
        atomcol = dict(C=(0.5, 0.5, 0.5), O=(1, 0, 0), N=(0, 0, 1), H=(1, 1, 1), P=(1, 0.5, 0.5))
        col = [atomcol[n[0]] for n in tgt.aname]
        if tgt.source:
            pymol.cmd.load(tgt.source, name + '_xyz')
            pymol.cmd.show('lines')
        else:
            ipd.viz.show_ndarray_point_or_vec(tgt.xyz, name=name + '_xyz', sphere=1, col=col, **kw)
        cgo = list()
        arrow = ipd.viz.cgo_cyl_arrow
        for don in tgt.don:
            cgo += arrow(don[:3, 0], don[:3, 0] + 3 * don[:3, 1], 0.1, col=(1, 1, 1), arrowlen=0)
        for acc in tgt.acc:
            cgo += arrow(acc[:3, 0], acc[:3, 0] + 3 * acc[:3, 1], 0.1, col=(1, 0, 0), arrowlen=0)
        pymol.cmd.load_cgo(cgo, name + '_hb')

except ImportError:
    pass
