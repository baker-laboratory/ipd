import ipd
from ipd.homog import *

def compute_canonical_asucen(sym, neighbors=None):
    import torch as th  # type: ignore
    from ipd import h
    sym = ipd.sym.map_sym_abbreviation(sym).lower()
    frames = h.tocuda(ipd.sym.frames(sym))
    x = h.randunit(int(5e5), device='cuda')
    symx = h.xform(frames[1:], x)
    d2 = th.sum((x[None] - symx)**2, dim=-1)

    mind2 = d2.min(dim=0)[0]
    if neighbors:
        ic(d2.shape)  # type: ignore
        sort = d2.sort(dim=0)[0]
        rank = sort[neighbors] - sort[neighbors - 1]
    else:
        rank = mind2
    # ic(d2.shape, mind2.shape)

    ibest = th.argmax(rank)
    best = x[ibest]
    dbest = th.sqrt(mind2[ibest])
    symbest = h.xform(frames, best)
    aln = th.sum(symbest * th.tensor([1, 2, 10], device='cuda'), dim=1)
    best = symbest[th.argmax(aln)] / dbest * 2
    if sym.startswith('c'):
        best = th.tensor([h.norm(best), 0, 0])
    return best.cpu().numpy()

def asufit(
    sym,
    coords,
    contact_coords=None,
    frames=None,
    objfunc=None,
    sampler=None,
    mc=None,
    cartsd=None,
    iterations=300,
    lever=None,
    temperature=1,
    thresh=0.000001,
    minradius=None,
    resetinterval=100,
    correctionfactor=2,
    showme=False,
    showme_accepts=False,
    dumppdb=False,
    verbose=False,
    **kw,
):
    ic("asufit", sym)  # type: ignore
    kw = ipd.dev.Bunch(kw)
    asym = ipd.dock.rigid.RigidBody(coords, contact_coords, **kw)
    if frames is None:
        frames = ipd.sym.frames(sym)
    bodies = [asym] + [ipd.dock.rigid.RigidBody(parent=asym, xfromparent=x, **kw) for x in frames[1:]]
    if kw.biasradial is None:
        kw.biasradial = ipd.sym.symaxis_radbias(sym, 2, 3)
    kw.biasdir = ipd.homog.hnormalized(asym.com())

    if lever is None:
        kw.lever = asym.rog() * 1.5
    if minradius is None:
        kw.minradius = ipd.homog.hnorm(asym.com()) * 0.5

    # if ipd.sym.is_known_xtal(sym):
    ObjFuncDefault = ipd.dock.rigid.RBLatticeOverlapObjective
    SamplerDefault = ipd.search.RBLatticeRBSampler
    # else:
    # ObjFuncDefault = ipd.dock.rigid.RBOverlapObjective
    # SamplerDefault = ipd.search.RBSampler

    if objfunc is None:
        objfunc = ObjFuncDefault(
            asym,
            bodies=bodies,
            sym=sym,
            **kw,
        )

    if sampler is None:
        sampler = SamplerDefault(
            cartsd=cartsd,
            center=asym.com(),
            **kw,
        )
    if mc is None:
        mc = ipd.MonteCarlo(objfunc, temperature=temperature, **kw)

    start = asym.state
    mc.try_this(start)
    initialscore = mc.best

    if showme:
        ipd.showme(bodies, name="start", pngi=0, **kw)
    ic(asym.scale())  # type: ignore
    if dumppdb:
        asym.dumppdb("debugpdbs/asufit_000000.pdb", **kw)
    # assert 0

    for i in range(iterations):
        if i % 50 == 0 and i > 0:
            asym.state = mc.beststate
            if i % resetinterval:
                asym.state = mc.startstate
            if mc.acceptfrac < 0.1:
                mc.temperature *= correctionfactor / resetinterval * 100
                sampler.cartsd /= correctionfactor / resetinterval * 100
            if mc.acceptfrac > 0.3:
                mc.temperature /= correctionfactor / resetinterval * 100
                sampler.cartsd *= correctionfactor / resetinterval * 100
            # ic(mc.acceptfrac, mc.best)

        pos, prev = sampler(asym.state)

        # adjust scale
        if i % 20:
            # for slidedir in [asym.com()]:
            # ipd.homog.htrans(-ipd.homog.hnormalized(slidedir))
            for i in range(10):
                contact = any([asym.contacts(b) for b in bodies[1:]])
                if contact:
                    break
                pos.scale -= 0.5
            for i in range(10):
                contact = any([asym.contacts(b) for b in bodies[1:]])
                if contact:
                    break
                pos.scale -= 0.5

        accept = mc.try_this(pos)
        if not accept:
            asym.state = prev
        else:
            if mc.best < thresh:
                # ic('end', i, objfunc(mc.beststate))
                # if showme: ipd.showme(bodies, name='mid%i' % i, **kw)
                return mc

            # if i % 10 == 0:
            if showme and showme_accepts:
                ipd.showme(bodies, name="mid%i" % i, pngi=i + 1, **kw)
            if dumppdb:
                asym.dumppdb(f"debugpdbs/asufit_{i+1:06}.pdb", **kw)

            if verbose:
                ic("accept", i, mc.last)  # type: ignore
        if mc.new_best_last:
            ic("best", i, mc.best)  # type: ignore
    assert mc.beststate is not None
    # ic('end', mc.best)
    initscore = objfunc(mc.startstate, verbose=True)
    stopscore = objfunc(mc.beststate, verbose=True)
    ic("init", initscore)  # type: ignore
    ic("stop", stopscore)  # type: ignore
    # ic(mc.beststate[:3, :3])
    # ic(mc.beststate[:3, 3])
    # ic(mc.beststate)
    asym.state = mc.beststate
    # ipd.pdb.dump_pdb_from_points('stopcoords.pdb', ipd.homog.hxform(mc.beststate.position, asym._coords))

    # ic('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # ic(bodies[0].contact_fraction(bodies[1]))
    # ic(bodies[0].contact_fraction(bodies[2]))
    if showme:
        ipd.showme(bodies, name="end", pngi=9999999, **kw)
    # if dumppdb:
    xyz = asym.coords
    cellsize = mc.beststate.scale
    ic(cellsize, xyz.shape)  # type: ignore
    TEST = 1
    if TEST:
        frames = ipd.hscaled(mc.beststate.scale, frames)
        xyz = ipd.homog.hxform(frames, xyz).reshape(len(frames), -1, 4, 4)
        if dumppdb:
            ipd.pdb.dumppdb(dumppdb, xyz, cellsize=cellsize, **kw)
    # ipd.pdb.dumppdb(f'debugpdbs/asufit_999999.pdb', xyz, cellsize=cellsize, **kw)
    # ipd.showme(bodies[0], name='pairs01', showcontactswith=bodies[1], showpairsdist=16, col=(1, 1, 1))
    # ipd.showme(bodies[0], name='pairs02', showcontactswith=bodies[2], showpairsdist=16, col=(1, 1, 1))
    # ipd.showme(bodies[1], name='pairs10', showcontactswith=bodies[0], showpairsdist=16, col=(1, 0, 0))
    # ipd.showme(bodies[2], name='pairs20', showcontactswith=bodies[0], showpairsdist=16, col=(0, 0, 1))

    # ic(coords.shape)
    # ipd.showme(hpoint(coords))
    # coords = asym.coords
    # ipd.showme(ipd.homog.hxform(mc.beststate, coords))

    return mc
