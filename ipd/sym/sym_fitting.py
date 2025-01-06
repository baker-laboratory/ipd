from ipd.lazy_import import lazyimport

th = lazyimport('torch')
import ipd

def asu_com(sym, xyz, Lasu, **kw):
    """Calculate the center of mass of the asymmetric unit."""
    com = xyz[:Lasu].mean(dim=(0, 1))
    return com

def sym_com(sym, xyz, Lasu, **kw):
    """Calculate the center of mass of the symmetric unit."""
    com = asu_com(sym, xyz, Lasu, **kw)
    # ic(xyz.shape)
    # ic(com.shape)
    return th.einsum('sij,j->si', sym.full_symmetry, com)

def asu_to_best_frame_if_necessary(sym,
                                   xyz,
                                   Lasu,
                                   asu_to_best_frame=None,
                                   disable_all_fitting=None,
                                   asu_to_best_frame_min_dist_to_origin=7,
                                   **kw):
    """Align the asu if necessary."""
    if asu_to_best_frame and not disable_all_fitting:
        if th.any(th.isnan(xyz)): return xyz
        Natom = min(3, xyz.shape[1])
        com = sym_com(sym, xyz[:, :Natom], Lasu, **kw)
        if ipd.h.norm(com[0]) < asu_to_best_frame_min_dist_to_origin: return xyz  # type: ignore
        d2 = th.sum((com - sym.asucenvec.to(com.device))**2, dim=1)
        isub = th.argmin(d2)
        # ic(isub, com[isub], sym.asucenvec)
        if isub != 0:
            print(f'moving asu to sub {isub} {com[isub]}')
            # ipd.showme(xyz[:,1], delprev=False)
            xyz[:Lasu] = th.einsum('ij,raj->rai', sym.full_symmetry[isub], xyz[:Lasu])  # note ji inverts
            # ipd.showme(xyz[:,1], delprev=False)
            # assert 0
    return xyz

def asu_to_canon_if_necessary(sym, xyz, Lasu, asu_to_canon=None, disable_all_fitting=None, **kw):
    """Align the asu if necessary."""
    if asu_to_canon and not disable_all_fitting:
        if th.any(th.isnan(xyz)): return xyz
        xyz = asu_to_best_frame_if_necessary(sym, xyz, Lasu, True)
        oldcom = xyz[:Lasu].mean(dim=(0, 1))
        dist = th.cdist(xyz[None, :Lasu, 0], oldcom[None, None])[0]
        rg = th.sqrt(th.sum(th.square(dist)) / Lasu)
        rg_ratio = asu_to_canon if isinstance(asu_to_canon, float) else 1.2
        newcom = rg_ratio * rg * ipd.sym.canonical_asu_center(sym.symid, cuda=True)
        # print(newcom-oldcom)
        xyz[:Lasu] += newcom - oldcom
    return xyz

def set_particle_radius_if_necessary(sym, xyz, Lasu, force_radius=None, disable_all_fitting=None, **kw):
    """Set the particle radius if necessary."""
    if force_radius and not disable_all_fitting:
        if th.any(th.isnan(xyz)): return xyz
        # ic('set_particle_radius_if_necessary')
        Natom = min(3, xyz.shape[1])
        assert Lasu
        Lsym = Lasu * sym.nsub
        com = asu_com(sym, xyz[:, :Natom], Lasu, **kw)
        r = th.norm(com)
        if not 0.9 < r / force_radius < 1.1:
            delta = (force_radius-r) * com / r
            xyz[:Lasu] += delta
            xyz[:Lsym] = th.einsum('sij,raj->srai', sym.symmRs, xyz[:Lasu]).reshape(Lsym, *xyz.shape[1:])
            print(f'set particle radius to {force_radius} from {r}')
    return xyz

def set_motif_placement_if_necessary(sym, xyz, disable_all_fitting=None, **kw):
    """Set the particle radius if necessary."""
    if disable_all_fitting:
        return xyz
    if sym.opt.motif_position == "AB" and sym.opt.motif_fit:
        gpbb = th.stack([
            xyz[sym.idx.gpca - 1, 1],
            xyz[sym.idx.gpca, 1],
            xyz[sym.idx.gpca + 1, 1],
        ], dim=1)
        bb = xyz[sym.idx.kind == 0, :3]
        # import ipd
        # ipd.showme(gpbb.reshape(-1,3), 'gpbb', spheres=0.3)
        # ipd.showme(bb.reshape(-1,3), 'bb', spheres=0.3)
        # ic(gpbb.shape)
        # ic(bb.shape)
        Lasu = len(bb) // sym.nsub
        idx, rms, xform = ipd.fit.qcp_scan_AB(bb, gpbb, Lasu)
        # ic(idx, rms, R, T)
        # rms2, fit, xform = ipd.wrmsfit(gpbb.reshape(-1, 3), bb[idx].reshape(-1, 3))
        mask = th.logical_or(sym.idx.kind == 1, sym.idx.kind == 12)
        # import ipd
        # x = bb[idx]
        # y = th.einsum('ij,rj->ri', R, gpbb) + T
        # ic(rms2)
        # ipd.showme(y, sphere=0.5, name='fitcagp', col=(1,0,0))
        # ipd.showme(x, sphere=0.5, name='fitca', col=(1,0.6 ,0))
        # ipd.showme(fit, sphere=0.5, name='fitca', col=(1,1 ,0))

        # assert th.allclose(gpbb, xyz[sym.idx.gpca, 1])
        # ipd.showme(xyz[~mask, 1], name='prot0', sphere=0.3)
        # ipd.showme(xyz[mask, 1], name='gp0', col=(1, 0, 0), sphere=0.3)

        # xyz[mask] = th.einsum('ij,raj->rai', R, xyz[mask]) + T
        xyz[mask] = ipd.h.xform(xform, xyz[mask])  # type: ignore

        # ipd.showme(xyz[~mask, 1], name='prot1', sphere=0.3)
        # ipd.showme(xyz[mask, 1], name='gp1', col=(1, 0, 1), sphere=0.3)

        # assert 0

        # import pymol
        # pymol.cmd.delete('_ca*')
        # import ipd
        # ipd.showme(xyz[:,1], name='all', sphere=0.4)
        # ic(th.abs(x-y))

    return xyz
