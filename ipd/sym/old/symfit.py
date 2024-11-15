import numpy as np

import ipd
from ipd import Bunch
from ipd import homog as hm

def align(coords, symelem, **kw):
    """Assumes shape (nchain, nres, natom, 3|4)"""
    assert len(coords) > 1
    if symelem.iscyclic:
        return aligncx(coords, symelem, **kw)
    else:
        msg = f"WARNING not aligning input on symelem: {symelem.label}"
        ipd.WARNME(msg)
        return coords
        raise NotImplementedError("hook up compute_symfit")

def aligncx(coords, symelem, rmsthresh=1, axistol=0.02, angtol=0.05, centol=1.0, **kw):
    """Leaves center at 0 regardless of symelem."""
    assert len(coords) == symelem.nfold

    coords = coords.reshape(symelem.nfold, -1, *coords.shape[-2:])
    origcoords = coords.copy()
    # ic(coords.shape)
    fitcoords = coords
    if coords.ndim > 3:  # ca only
        fitcoords = coords[:, :, 1]
    rms, _, xfits = zip(*[ipd.hrmsfit(fitcoords[i - 1], fitcoords[i]) for i in range(len(coords))])
    if max(rms) > rmsthresh:
        raise ValueError(f"subunits have high rms {rms}")
    axs, ang, cen = (np.array(x) for x in zip(*[ipd.homog.haxis_ang_cen_of(x) for x in xfits]))

    if np.max(ang - 2 * np.pi / symelem.nfold) > angtol:
        raise ValueError(f"sub rotation angles incoherent {ang}")
    if np.max(np.std(axs, axis=0)) > axistol:
        raise ValueError(f"sub rotation axes incoherent {axs}")
    if np.max(np.std(cen, axis=0)) > centol:
        raise ValueError(f"sub rotation centers incoherent {cen}")
    avgaxs = np.mean(axs, axis=0)
    avgang = np.mean(ang)
    avgcen = np.mean(cen, axis=0)
    # ic(avgaxs, avgang, avgcen)
    coords = ipd.homog.htrans(-avgcen, doto=coords)
    if ipd.homog.angle(avgaxs, symelem.axis) > 0.0001:
        coords = ipd.homog.halign(avgaxs, symelem.axis, doto=coords)
    com = ipd.homog.hcom(coords, flat=True)
    delta = -ipd.hproj(symelem.axis, ipd.homog.hvec(com))
    # delta += symelem.cen
    coords = ipd.homog.htrans(delta, doto=coords)

    xfit = ipd.homog.htrans(delta) @ ipd.homog.halign(avgaxs, symelem.axis) @ ipd.homog.htrans(-avgcen)
    coords2 = ipd.homog.hxform(xfit, origcoords)
    assert np.allclose(coords2, coords)

    return coords, xfit

def compute_symfit(
    sym,
    frames,
    *,
    lossterms=None,
    max_radius=100000.0,
    min_radius=0.0,
    penalize_redundant_cyclic_nth=0,  # nth closest
    target_sub_com=None,
    target_sub_com_testpoint=None,
    **kw,
):
    kw = ipd.dev.Bunch(kw)
    sym = sym.lower()

    iscyclic = sym.startswith("c")
    if not iscyclic and len(frames) < 3:
        raise ValueError(f"symmetry {sym} requires at least 3 subunits, {len(frames)} provided")
    if len(frames) > len(ipd.sym.sym_frames[sym]):
        raise ValueError(f"symmetry {sym} has at most {nnativeframes}, {len(frames)} supplied")  # type: ignore
    symops = ipd.sym.symops_from_frames(sym=sym, frames=frames, **kw)
    _checkpoint(kw, "symops_from_frames")

    cen1, cen2, axs1, axs2 = get_symop_pairs(symops, **kw)
    _checkpoint(kw, "get_symop_pairs")

    if iscyclic:
        return

    p, q, isect, center = get_symops_isect(sym, cen1, cen2, axs1, axs2, symops.nfold, **kw)
    _checkpoint(kw, "get_symops_isect")

    radius, rad_err, framedist_err = get_radius_err(frames, center, **kw)

    redundant_cyclic_err = 0
    if penalize_redundant_cyclic_nth:
        redundant_cyclic_err = get_redundant_cyclic_err(symops, penalize_redundant_cyclic_nth, **kw)

    cen_err = np.sqrt((np.sum((center - p)**2) + np.sum((center - q)**2)) / (len(q) + len(q)))
    op_hel_err = np.sqrt(np.mean(symops.hel**2))
    op_ang_err = np.sqrt(np.mean(symops.nfold_err**2))
    _checkpoint(kw, "post intersect stuff")

    xfit, axesfiterr = ipd.sym.symops_align_axes(sym, frames, symops, symops, center, radius, **kw)
    _checkpoint(kw, "align axes")

    symframes = ipd.sym.frames(sym)
    fitframes = ipd.homog.hxform(symframes, xfit)

    # moveme to func

    if target_sub_com is not None:
        coms = fitframes @ target_sub_com_testpoint
        diff = ipd.homog.hdot(ipd.homog.hnormalized(target_sub_com), ipd.homog.hnormalized(coms))
        ic(diff)  # type: ignore
        ic(np.arccos(diff) * 180 / 3.1416)  # type: ignore
        # assert 0
    else:
        diff = np.sum((fitframes - np.eye(4))**2, axis=(1, 2))

    ibest = np.argmin(diff)
    xfit = fitframes[ibest]

    # /moveme

    assert not lossterms
    loss = dict()
    loss["C"] = 1.0 * cen_err**2
    loss["H"] = 0.7 * op_hel_err**2
    loss["N"] = 1.2 * op_ang_err**2
    loss["A"] = 1.5 * axesfiterr**2
    # loss['R'] = 1.0 * rad_err**2
    loss["S"] = 0.01 * max(0, radius - max_radius)**2 + max(0, min_radius - radius)**2
    # loss['Q'] = 1.0 * quad_err**2
    # loss['M'] = 1.0 * missing_axs_err**2
    # loss['S'] = 1.0 * skew_axs_err**2
    loss["D"] = 0.1 * framedist_err
    loss["E"] = 0.1 * redundant_cyclic_err
    # print(framedist_err)
    lossterms = "CHNASDE"
    # lossterms = 'CHNARS'
    total_err = np.sqrt(np.sum(list(loss.values())))
    weighted_err = total_err
    if lossterms:
        weighted_err = np.sqrt(sum(loss[c] for c in lossterms))

    return SymFit(
        sym=sym,
        nframes=len(frames),
        frames=frames,
        symops=symops,
        center=center,
        opcen1=cen1,
        opcen2=cen2,
        opaxs1=axs1,
        opaxs2=axs2,
        iscet=isect,
        isect1=p,
        iscet2=q,
        radius=radius,
        xfit=xfit,
        cen_err=cen_err,
        symop_hel_err=op_hel_err,
        symop_ang_err=op_ang_err,
        axes_err=axesfiterr,
        total_err=total_err,
        weighted_err=weighted_err,
        redundant_cyclic_err=redundant_cyclic_err,
        losses=loss,
    )

_get_redundant_cyclic_err_warning = True

def get_redundant_cyclic_err(
    symops,
    penalize_redundant_cyclic_nth,
    penalize_redundant_cyclic_angle=None,  # degrees
    penalize_redundant_cyclic_weight=1.0,
    **kw,
):
    if symops.sym.startswith("c"):
        global _get_redundant_cyclic_err_warning
        if _get_redundant_cyclic_err_warning:
            print("WARNING: redundant_cyclic_err makes no sense when fitting cyclic symmetry")
            _get_redundant_cyclic_err_warning = False
    if penalize_redundant_cyclic_angle is None:
        penalize_redundant_cyclic_angle = ipd.sym.min_symaxis_angle(symops.sym) / 2
        penalize_redundant_cyclic_angle = np.degrees(penalize_redundant_cyclic_angle)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print('penalize_redundant_cyclic_angle', penalize_redundant_cyclic_angle)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    maxangrad = np.radians(penalize_redundant_cyclic_angle)
    # print(maxangrad)
    # ipd.showme(symops.axs)
    # ipd.showme(symops.cen)
    # print(symops.axs.shape)
    angs = ipd.line_angle(symops.axs, symops.axs, outerprod=True)
    angs = np.sort(angs, axis=-1)
    # print(angs)
    second_closest = angs[:, 2]
    # print(second_closest[:, None])

    err = (np.maximum(0, maxangrad - second_closest) / maxangrad)**2
    err *= penalize_redundant_cyclic_angle
    err *= penalize_redundant_cyclic_weight
    err = np.mean(err)
    # print(err[:, None])
    # print(err)
    return err
    # assert 0

class SymFitError(Exception):
    pass

def _checkpoint(kw, label):
    if "timer" in kw:
        kw["timer"].checkpoint(label)

class RelXformInfo(Bunch):
    pass

class SymOps(Bunch):
    pass

class SymFit(Bunch):
    pass

def rel_xform_info(frame1, frame2, **kw):
    # rel = np.linalg.inv(frame1) @ frame2
    rel = frame2 @ np.linalg.inv(frame1)
    # rot = rel[:3, :3]
    # axs, ang = hm.axis_angle_of(rel)
    axs, ang, cen = hm.axis_ang_cen_of(rel)

    framecen = (frame2[:, 3] + frame1[:, 3]) / 2
    framecen = framecen - cen
    framecen = hm.hproj(axs, framecen)
    framecen = framecen + cen

    inplane = hm.hprojperp(axs, cen - frame1[:, 3])
    # inplane2 = hm.hprojperp(axs, cen - frame2[:, 3])
    rad = np.sqrt(np.sum(inplane**2))
    if np.isnan(rad):
        print("isnan rad")
        print("xrel")
        print(rel)
        print("det", np.linalg.det(rel))
        print("axs ang", axs, ang)
        print("cen", cen)
        print("inplane", inplane)
        assert 0
    hel = np.sum(axs * rel[:, 3])
    return RelXformInfo(
        xrel=rel,
        axs=axs,
        ang=ang,
        cen=cen,
        rad=rad,
        hel=hel,
        framecen=framecen,
        frames=np.array([frame1, frame2]),
    )

def xform_update_symop(symop, xform, srad):
    frame1 = xform @ symop.frames[0]
    frame2 = xform @ symop.frames[1]
    result = rel_xform_info(frame1, frame2)
    for k, v in symop.items():
        if k not in result:
            result[k] = symop[k]
    scen = xform[:, 3]
    p = hm.hproj(result.axs, -result.cen) + result.cen
    d = np.linalg.norm(p[:3])
    e = 0
    if d < srad:
        e = np.sqrt(srad**2 - d**2)
    result.closest_to_cen = p
    result.isect_sphere = p + result.axs * e
    # ipd.showme([p, np.array([0, 0, 0, 1]), result])
    # assert 0

    return result

def cyclic_sym_err(pair, angle):
    hel_err = pair.hel
    errrad = min(10000, max(pair.rad, 1.0))
    ang_err = errrad * (angle - pair.ang)
    err = np.sqrt(hel_err**2 + ang_err**2)
    return err

def symops_from_frames(*, sym, frames, **kw):
    kw = ipd.dev.Bunch(kw)
    assert len(frames) > 1
    assert frames.shape[-2:] == (4, 4)
    pairs = dict()
    pairlist = list()
    keys, frame1, frame2 = list(), list(), list()
    for i, f1 in enumerate(frames):
        for j in range(i + 1, len(frames)):
            f2 = frames[j]
            keys.append((i, j))
            frame1.append(f1)
            frame2.append(f2)

    frame1 = np.stack(frame1)
    frame2 = np.stack(frame2)
    xrel = frame2 @ np.linalg.inv(frame1)
    axs, ang, cen = hm.axis_ang_cen_of(xrel)
    framecen = (frame2[:, :, 3] + frame1[:, :, 3]) / 2 - cen
    framecen = hm.hproj(axs, framecen) + cen
    inplane = hm.hprojperp(axs, cen - frame1[:, :, 3])
    rad = np.sqrt(np.sum(inplane**2, axis=-1))
    hel = np.sum(axs * xrel[:, :, 3], axis=-1)
    assert (len(frame1) == len(frame2) == len(xrel) == len(axs) == len(ang) == len(cen) == len(framecen) == len(rad) ==
            len(hel))
    errrad = np.minimum(10000, np.maximum(rad, 1.0))
    angdelta, err, closest = dict(), dict(), dict()
    point_angles = ipd.sym.sym_point_angles[sym]
    # print(point_angles)

    for n, tgtangs in point_angles.items():
        tgtangs = np.asarray(tgtangs)
        dabs = np.array([np.abs(ang - atgt) for atgt in tgtangs])
        d = np.array([atgt - ang for atgt in tgtangs])
        if len(dabs) == 1:
            dabs = dabs[0]
            d = d[0]
            atgt = tgtangs[0]
        elif len(dabs) == 2:
            d = np.where(dabs[0] < dabs[1], d[0], d[1])
            atgt = np.where(dabs[0] < dabs[1], tgtangs[0], tgtangs[1])
            dabs = np.where(dabs[0] < dabs[1], dabs[0], dabs[1])
        else:
            w = np.argmin(dabs, axis=0)
            atgt = tgtangs[w]
            d = d[w, np.arange(len(w))]
            dabs = dabs[w, np.arange(len(w))]

        ang_err = errrad * dabs
        err[n] = np.sqrt(hel**2 + ang_err**2)
        angdelta[n] = d
        closest[n] = np.argmin(dabs)

    errvals = np.stack(list(err.values()))
    w = np.argmin(errvals, axis=0)
    nfold = np.array(list(err.keys()))[w]
    nfold = np.array([ipd.sym.sym_nfold_map(nf) for nf in nfold])
    nfold = nfold.astype("i4")
    angdelta = np.array([angdelta[nf][i] for i, nf in enumerate(nfold)])
    nfold_err = np.min(errvals, axis=0)

    for nf in point_angles:
        if np.sum(nfold == nf) == 0:
            nfold[closest[nf]] = nf

    nfold = disambiguate_axes(sym, axs, nfold, **kw)

    only1nfold = sym == "d2" or sym.startswith("c")
    if not only1nfold and min(nfold) == max(nfold):
        raise SymFitError(f"sym {sym} all axes are same nfold {nfold[0]}")

    return SymOps(
        sym=sym,
        key=keys,
        frame1=frame1,
        frame2=frame2,
        xrel=xrel,
        axs=axs,
        ang=ang,
        cen=cen,
        rad=rad,
        hel=hel,
        framecen=framecen,
        nfold=nfold,
        nfold_err=nfold_err,
        angdelta=angdelta,
    )

def disambiguate_axes(sym, axis, nfold, noambigaxes=True, **kw):
    if ipd.sym.ambiguous_axes(sym) is None:
        return nfold
    if not noambigaxes:
        return nfold
    nfold1 = nfold.copy()

    angcut = np.pi / 12
    for ambignfold, maybenfold in reversed(ipd.sym.ambiguous_axes(sym)):
        if sym.lower().startswith("d") and ambignfold != 2:
            nfold1[nfold1 == ambignfold] = maybenfold
            continue
        ambigaxis = axis[nfold1 == ambignfold]
        maybeaxis = axis[nfold1 == maybenfold]
        # print('disambiguate_axes', ambigaxis.shape, maybeaxis.shape)
        dot = np.abs(hm.hdot(ambigaxis[None, :], maybeaxis[:, None]))
        try:
            maxdot = np.max(dot, axis=0)
        except Exception:
            raise SymFitError(f"missing axes: {nfold1}")
        maybe_so = maxdot > np.cos(angcut)  # theoretically pi/8 ro 22.5 deg

        nfold1[nfold1 == ambignfold] = np.where(maybe_so, maybenfold, ambignfold)

    if all(nfold1 == nfold1[0]):
        return nfold
    return nfold1

def stupid_pairs_from_symops(symops):
    pairs = dict()
    for i, k in enumerate(symops.key):
        pairs[k] = RelXformInfo(
            xrel=symops.xrel[i],
            axs=symops.axs[i],
            ang=symops.ang[i],
            cen=symops.cen[i],
            rad=symops.rad[i],
            hel=symops.hel[i],
            framecen=symops.framecen[i],
            frames=np.array([symops.frame1[i], symops.frame2[i]]),
            nfold=symops.nfold[i],
            nfold_err=symops.nfold_err[i],
        )
    return pairs

def get_symop_pairs(symops, **kw):
    cen1, cen2, axs1, axs2 = list(), list(), list(), list()
    nops = len(symops.axs)
    for i in range(nops):
        cen1.append(np.tile(symops.cen[i], nops - i - 1).reshape(-1, 4))
        axs1.append(np.tile(symops.axs[i], nops - i - 1).reshape(-1, 4))
        cen2.append(symops.cen[i + 1:])
        axs2.append(symops.axs[i + 1:])
        assert cen1[-1].shape == cen2[-1].shape
    cen1 = np.concatenate(cen1)
    cen2 = np.concatenate(cen2)
    axs1 = np.concatenate(axs1)
    axs2 = np.concatenate(axs2)
    return cen1, cen2, axs1, axs2

def get_symops_isect(sym, cen1, cen2, axs1, axs2, nfold, max_nan=0.9, isect_outliers_sd=3, **kw):
    # max_nan 0.9 totally arbitrary, downstream check for lacking info maybe better
    axes_angles = hm.line_angle(axs1, axs2)
    not_same_symaxis = axes_angles > ipd.sym.minsymang[sym]
    if not any(not_same_symaxis):
        # pick furthest apart axes to be distinct
        not_same_symaxis[np.argmax(axes_angles)] = True
        assert any(not_same_symaxis)
    p1np = cen1[not_same_symaxis]
    p2np = cen2[not_same_symaxis]
    a1np = axs1[not_same_symaxis]
    a2np = axs2[not_same_symaxis]

    p, q = hm.line_line_closest_points_pa(p1np, a1np, p2np, a2np)

    tot_nan = np.sum(np.isnan(p)) / 4
    if tot_nan / len(p) > max_nan:
        print("nan fail nfolds", nfold)
        raise SymFitError(f"{tot_nan/len(p)*100:7.3f}% of symops are parallel or cant be intersected")

    p = p[~np.isnan(p)].reshape(-1, 4)
    q = q[~np.isnan(q)].reshape(-1, 4)
    isect = (p+q) / 2
    assert len(isect) > 0
    center = np.mean(isect, axis=0)

    norm = np.linalg.norm(p - center, axis=-1)
    meannorm = np.mean(norm)
    sdnorm = np.std(norm)
    not_outlier = norm - meannorm < sdnorm * isect_outliers_sd
    if np.sum(not_outlier) > 5:
        center = np.mean(isect[not_outlier], axis=0)

    return p, q, isect, center

def get_radius_err(frames, center, **kw):
    dist = np.linalg.norm(frames[None, :, :3, 3] - frames[:, None, :3, 3], axis=-1)
    np.fill_diagonal(dist, 9e9)
    mindist = np.min(dist)
    radii = np.linalg.norm(frames[:, :, 3] - center, axis=-1)
    radius = np.mean(radii)
    if radius > 1e6:
        raise SymFitError(f"inferred radius is {radius}")
    rad_err = np.sqrt(np.mean(radii**2)) - radius
    if rad_err < -0.001:
        print("WARNING rad_err", rad_err, radii)
    framedist_err = max(0, radius/mindist - 5)
    return radius, rad_err, framedist_err

def best_axes_fit(sym, xsamp, nfolds, tgtaxes, tofitaxes, **kw):
    xsamp = xsamp[:, None]
    randtgtaxes = [(xsamp @ ax.reshape(1, -1, 4, 1)).squeeze(-1) for ax in tgtaxes]

    err = list()
    for i, (nf, tgt, fit) in enumerate(zip(nfolds, randtgtaxes, tofitaxes)):
        # print(nf, tgt.shape, fit.shape)
        if len(fit) == 0:
            continue
        n = np.newaxis
        nf = ipd.sym.sym_nfold_map(nf)
        dotall = hm.hdot(fit[n, n, :], tgt[:, :, n])
        if sym != "tet" or nf != 2:
            dotall = np.abs(dotall)
        # angall = np.arccos(dotall)
        # angmatch = np.min(angall, axis=1)
        # angerr = np.mean(angmatch**2, axis=1)
        dotmatch = np.max(dotall, axis=1)
        angerr = np.mean(np.arccos(dotmatch)**2, axis=1)
        err.append(angerr)
        assert np.all(angerr < 9e5)
    err = np.mean(np.stack(err), axis=0)
    bestx = xsamp[np.argmin(err)].squeeze()
    err = np.sqrt(np.min(err))
    return bestx, err

def symops_align_axes(
    sym,
    frames,
    opary,
    symops,
    center,
    radius,
    choose_closest_frame=False,
    align_ang_delta_thresh=0.001,
    alignaxes_more_iters=1.0,
    **kw,
):
    nfolds = list(ipd.sym.symaxes[sym].keys())

    if "2b" in nfolds:
        nfolds.remove("2b")  # what to do about T33?
    if "2c" in nfolds:
        nfolds.remove("2c")
    if "2d" in nfolds:
        nfolds.remove("2d")
    # print(nfolds)
    nfolds = list(map(ipd.sym.sym_nfold_map, nfolds))
    # print(nfolds)
    pang = ipd.sym.sym_point_angles[sym]
    # xtocen = np.eye(4)
    # xtocen[:, 3] = -center

    _checkpoint(kw, "symops_align_axes xtocen")
    # recenter frames without modification
    # for k in symops:
    # symops[k] = xform_update_symop(symops[k], xtocen, radius)
    # fitaxis = hm.hnormalized(symops[k].framecen)
    # fitaxis = hm.hnormalized(symops[k].isect_sphere)
    # fitaxis = hm.hnormalized(symops[k].axs)
    # symops[k].fitaxis = fitaxis

    # fitaxis = hm.hnormalized(symops[k].framecen)
    # fitaxis = hm.hnormalized(symops[k].isect_sphere)
    # opary.fitaxis = opary.isect_sphere
    # opary.fitaxis = opary.framecen
    opary.fitaxis = opary.axs
    # for i, a in enumerate(opary.fitaxis):
    #     cyccen = opary.framecen - center
    #     if np.sum(cyccen * a) < 0:
    #         opary.fitaxis[i] = -a

    # ipd.showme(symops)
    _checkpoint(kw, "symops_align_axes xform xform_update_symop")

    # nfold_axes = [{k: v for k, v in symops.items() if v.nfold == nf} for nf in nfolds]
    # allopaxes = np.array([op.fitaxis for op in symops.values()])
    # opnfold = np.array([op.nfold for op in symops.values()], dtype='i4')
    # sopaxes = [allopaxes[opnfold == nf] for nf in nfolds]
    # tgtaxes = [ipd.sym.symaxes_all[sym][nf] for nf in nfolds]
    allopaxes = opary.fitaxis
    opnfold = opary.nfold
    sopaxes = [allopaxes[opnfold == nf] for nf in nfolds]
    tgtaxes = [ipd.sym.symaxes_all[sym][nf] for nf in nfolds]

    if sym != "d2":
        assert sum([len(x) > 0 for x in sopaxes]) > 1, "must have at least 2 symaxes"

    # print([len(a) for a in sopaxes])
    # print('nfolds', nfolds)
    # print('tgtaxes', tgtaxes)
    # print('sopaxes', sopaxes)
    # assert 0
    assert len(sopaxes) > 0

    _checkpoint(kw, "symops_align_axes build arrays")
    # axes = symin
    # print('origtgtaxes', [a.shape for a in origtgtaxes])
    # print('tgtdaxes   ', [a.shape for a in tgtdaxes])
    # print('sopaxes     ', [a.shape for a in sopaxes])
    nsamp = int(20 * alignaxes_more_iters)
    xsamp = hm.rand_xform(nsamp, cart_sd=0)
    xfit, angerr = best_axes_fit(sym, xsamp, nfolds, tgtaxes, sopaxes)
    best = 9e9, np.eye(4)
    for i in range(int(20 * alignaxes_more_iters)):
        # _checkpoint(kw, 'symops_align_axes make xsamp pre')
        xsamp = hm.rand_xform_small(nsamp, rot_sd=angerr / 2, cart_sd=0) @ xfit
        # _checkpoint(kw, 'symops_align_axes make xsamp')
        xfit, angerr = best_axes_fit(sym, xsamp, nfolds, tgtaxes, sopaxes, **kw)
        # _checkpoint(kw, 'symops_align_axes best_axes_fit')
        delta = angerr - best[0]
        if delta < 0:
            best = angerr, xfit
            if delta > -align_ang_delta_thresh:
                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                break
        xfit = best[1]
        # if i % 1 == 0: print(angerr)

    # if False:
    if sym == "d2":
        # all nfolds are 2, must make sure not all aligned to same axis
        maxang = hm.line_angle(sopaxes[0][:, None], sopaxes[0][None])
        err = (np.pi / 2 - np.max(maxang)) * 3  # 3 is arbitrary
        best = best[0] + err, best[1]
        # print(maxang, best[0], err)
        # assert 0

    angerr, xfit = best
    assert angerr < 9e6

    # _checkpoint(kw, 'symops_align_axes check rand axes')

    xfit[:, 3] = center
    xfit = np.linalg.inv(xfit)
    axesfiterr = angerr * radius

    if choose_closest_frame:
        # assert 0
        axes_perm_choices = ipd.sym.sym_permute_axes_choices(sym)
        choices = axes_perm_choices[:, None] @ ipd.sym.sym_frames[sym][None, :]
        choices = choices.reshape(-1, 4, 4)
        # print(xfit)
        pos = (choices @ xfit @ frames[0])[:, :3, 3]
        # print(pos)
        dots = ipd.homog.hdot([3, 2, 1], pos)
        # print(dots)
        which_frame = np.argmax(dots)
        # print('choosing closest frame', which_frame)
        xfit = choices[which_frame] @ xfit
        # print(which_frame)
        assert xfit.shape == (4, 4)

        # assert 0

    # if sym == 'tet':
    #     fit = xfit @ frames
    #     cens = hm.hnormalized(fit[:, :, 3])
    #     upper = np.any(hm.angle(cens, [1, 1, 1]) > 1.91 / 2)
    #     if not upper:
    #         # rotate 3fold around -1 -1 -1 to 3fold around 1 1 1
    #         # this prevents mismatch with canonical tetrahedral 3fold position
    #         # tetframes @ frames can form octahedra
    #         xfit = hm.hrot([1, 1, -1], np.pi * 2 / 3) @ xfit
    #         fit = xfit @ frames
    #         cens = hm.hnormalized(fit[:, :, 3])
    #         upper = np.any(hm.angle(cens, [1, 1, 1]) > 1.91 / 2)
    #         # upper = np.any(np.all(cens > 0.0, axis=-1))
    #         if not upper:
    #             xfit = hm.hrot([1, 1, -1], np.pi * 2 / 3) @ xfit
    #     # [1, 1, 1],
    #     # [m, 1, 1],
    #     # [1, m, 1],
    #     # [1, 1, m],

    #     # lower = np.any(np.all(cens < +0.1, axis=-1))
    #     # if upper and lower:
    #     # assert 0
    #     # axesfiterr = 9e9
    #     # assert 0, 'can this happen and be reasonable?'

    # print(xfit.shape, axesfiterr * 180 / np.pi)

    # for i, ax in enumerate(sopaxes):
    #     col = [0, 0, 0]
    #     col[i] = 1
    #     ipd.showme(ax, col=col, usefitaxis=True, name='sopA')
    #     # ipd.showme(tgtaxes[i], col=col, usefitaxis=True, name='tgtA')
    #     ax = hm.hxform(xfit, ax)
    #     ipd.showme(ax, col=col, usefitaxis=True, name='nfoldB')
    return xfit, axesfiterr

def symfit_gradient(symfit):
    # sym=sym,
    # frames=frames,
    # symops=symops,
    # center=center,
    # opcen1=cen1,
    # opcen2=cen2,
    # opaxs1=axs1,
    # opaxs2=axs2,
    # iscet=isect,
    # isect1=p,
    # iscet2=q,
    # radius=radius,
    # xfit=xfit,
    # cen_err=cen_err,
    # symop_hel_err=op_hel_err,
    # symop_ang_err=op_ang_err,
    # axes_err=axesfiterr,
    # total_err=total_err,
    # weighted_err=weighted_err,

    # result = SymOps(
    #     key=keys,
    #     frame1=frame1,
    #     frame2=frame2,
    #     xrel=xrel,
    #     axs=axs,
    #     ang=ang,
    #     cen=cen,
    #     rad=rad,
    #     hel=hel,
    #     framecen=framecen,
    #     nfold=nfold,
    #     nfold_err=nfold_err,
    sop = symfit.symops

    print(list(symfit.symops.keys()))
    for i in range(len(sop.key)):
        print(
            sop.nfold[i],
            np.round(np.degrees(sop.ang[i] + sop.angdelta[i])),
        )
    cenforce = np.zeros
    frametorq = np.zeros(shape=(symfit.nframes, 4))

    for key, torq in zip(sop.key, optorq):  # type: ignore
        print(key, torq)
        frametorq[key[0]] -= torq
        frametorq[key[1]] += torq

    frameforce = np.zeros((symfit.nframes, 4))
    for key, force in zip(sop.key, sop.hel):
        frameforce[key[0]] += force
        frameforce[key[1]] -= force
    # SHOULD I ADD CART ROTATION FORCE HERE TOO?????

    # ipd.showme(sy,mfit.symops)

    # assert 0

def symfit_mc_play(
    sym=None,
    seed=None,
    random_frames=False,
    quiet=True,
    nframes=None,
    maxiters=500,
    goalerr=0.01,
    showme=False,
    scalesamp=1.0,
    scalecartsamp=1.0,
    scalerotsamp=1.0,
    scaletemp=1.0,
    max_cartsd=10,
    vizinterval=10,
    showsymdups=True,
    showsymops=False,
    showfulltraj=False,
    showopts=None,
    headless=False,
    **kw,
):
    kw = ipd.dev.Bunch(kw, _strict=False)  # type: ignore
    if "timer" not in kw:
        kw.timer = ipd.dev.Timer()

    import os

    if "PYTEST_CURRENT_TEST" in os.environ:
        showme = False

    if seed is None:
        seed = np.random.randint(2**32 - 1)
    np.random.seed(seed)
    print(f"------------------ SEED {seed} -------------------------", flush=True)

    # kw.sym = np.random.choice('tet oct icos'.split())
    # kw.nframes = len(ipd.sym.sym_frames[kw.sym])
    # kw.nframes = np.random.choice(6) + 6
    kw.sym = sym or "icos"
    if nframes is None:
        nframes = dict(
            d3=6,
            d5=6,
            tet=6,
            oct=7,
            icos=7,
        )[kw.sym]
    nframes = min(nframes, len(ipd.sym.sym_frames[sym]))

    kw.tprelen = kw.tprelen or 10
    kw.tprerand = kw.tprerand or 0
    kw.tpostlen = kw.tpostlen or 20
    kw.tpostrand = kw.tpostrand or 0
    kw.fuzzstdfrac = kw.fuzzstdfrac or 0.01  # frac of radian
    kw.cart_sd_fuzz = kw.cart_sd_fuzz or kw.fuzzstdfrac * kw.tprelen
    kw.rot_sd_fuzz = kw.rot_sd_fuzz or kw.fuzzstdfrac
    kw.remove_outliers_sd = kw.remove_outliers_sd or 3
    kw.penalize_redundant_cyclic_nth = 2
    kw.penalize_redundant_cyclic_angle = 10
    kw.penalize_redundant_cyclic_weight = 2.0

    # kw.choose_closest_frame = kw.choose_closest_frame or True
    showme_opts = showopts.sub(  # type: ignore
        _onlynone=True,
        spheres=0.4,
        showme=showme,
        vizfresh=True,
        weight=1,
        # xyzlen=[.4, .4, .4],
    )

    if random_frames:
        frames = hm.rand_xform(nframes, cart_sd=kw.tprelen)  #   @ frames
    else:
        frames, *_ = setup_test_frames(nframes, **kw)
    # frames = ipd.sym.sym_frames[kw.sym]

    # ipd.showme(frames)
    # assert 0

    # ipd.showme(frames, 'start', col=(1, 1, 1), **showme_opts)
    symfit = ipd.sym.compute_symfit(frames=frames, **kw)
    err0 = symfit.weighted_err
    frames = symfit.xfit @ frames
    if not quiet:
        print("start", symfit.weighted_err)
    # ipd.showme(ipd.sym.sym_frames[kw.sym][:, None] @ frames[None, :], name='symstart',
    # col=(1, 1, 0), rays=0.02, weight=0.3, **showme_opts)

    if showme:
        pairs = ipd.sym.stupid_pairs_from_symops(symfit.symops)
        ipd.showme(
            pairs,
            name="pairsstart",
            col="bycx",
            center=[0, 0, 0],
            **showme_opts,
        )

    import pymol  # type: ignore
    pymol.cmd.turn("x", 20)

    lowerr = symfit.weighted_err
    besterr = lowerr
    best = frames, None
    ipng = 0
    naccept, lastviz = 0, -999
    for isamp in range(maxiters):
        if isamp % 10 == 0:
            frames = best[0]
        if isamp % 100 == 0 and not quiet:
            print(f"{isamp:6} {symfit.weighted_err:7.3} {naccept / (isamp + 1):7.3} {lowerr:7.3} {symfit.radius:9.3}")
        cartsd = symfit.weighted_err / 45 * scalecartsamp * scalesamp
        cartsd = min(max_cartsd, cartsd)
        rotsd = cartsd / symfit.radius * scalerotsamp * scalesamp
        temp = symfit.weighted_err / 100 * scaletemp
        purturbation = hm.rand_xform_small(len(frames), cart_sd=cartsd, rot_sd=rotsd)

        assert np.max(purturbation[:, :, 3]) < 1e7
        purturbed = purturbation @ frames
        try:
            assert np.max(purturbed[:, :3, 3]) < 1e6
            symfit = ipd.sym.compute_symfit(frames=purturbed, **kw)
            # print('SCORE', isamp, symfit.weighted_err)
        except Exception as e:
            print("FAIL", isamp, "SEED", seed)
            print(repr(purturbed))
            # print(np.max(frames[:, :3, 3]))
            # print(np.max(purturbation[:, :3, 3]))
            # print(np.max(purturbed[:, :3, 3]))
            raise e

        candidate = symfit.xfit @ purturbed

        if np.isnan(symfit.weighted_err):
            break

        # symdupframes = ipd.sym.sym_frames[kw.sym][:, None] @ frames[None, :]
        # ipd.showme(symdupframes, name='xfitmc%05i' % isamp, col=None, **showme_opts)

        delta = symfit.weighted_err - lowerr
        if np.exp(-delta / temp) > np.random.rand():
            naccept += 1
            # frames = symfit.xfit @ candidate
            frames = candidate
            lowerr = symfit.weighted_err
            # if isamp % 20 == 0: print(f'accept rate {naccept/(isamp+1)}')
            # col = (isamp / maxiters, 1 - iasmp / maxiters, 1)
            # ipd.showme(candidate, name='mc%05i' % isamp, col=col, center=[0, 0, 0],
            # **showme_opts)b

            if showme and delta < 0 and isamp - lastviz > vizinterval:
                # pairs = ipd.sym.stupid_pairs_from_symops(symfit.symops)
                col = (isamp / maxiters, 1 - isamp/maxiters, 1)  #######

                if showsymdups:
                    vizfresh = not showfulltraj
                    vizfresh = True
                    # symdupframes = ipd.sym.sym_frames[kw.sym][:, None] @ frames[None, :]
                    symdupframes = frames
                    ipd.showme(symdupframes, name="xfitmc%05i" % isamp, col=None, **showme_opts.sub(vizfresh=vizfresh))

                    # os.makedirs('symfit_movie', exist_ok=True)
                    # pymol.cmd.png(f'symfit_movie/symdup_{ipng:04}.png', )

                # pymol.cmd.turn('y', 1)

                if showsymops:
                    vizfresh = not showfulltraj and not showsymdups
                    pairs = ipd.sym.stupid_pairs_from_symops(symfit.symops)
                    # print(type(pairs))
                    # print(pairs)
                    # pairs = {(0, 1): pairs[(0, 1)]}
                    del pairs[(0, 1)]
                    # assert 0
                    ipd.showme(
                        pairs,
                        name="pairsstop",
                        col="bycx",
                        center=[0, 0, 0],
                        **showme_opts.sub(vizfresh=vizfresh),
                    )
                    # assert 0
                    # os.makedirs('symfit_movie', exist_ok=True)
                    # fname = f'symfit_movie/symops_{ipng:04}.png'
                    # print('MOVIE FRAME', isamp, fname, flush=True)
                    # pymol.cmd.png(fname)
                ipng += 1

                # ipd.showme(frames, name='xfitmc%05ib' % isamp, col=None,
                # **showme_opts.sub(spheres=0.5, weight=1.5))
                # ipd.showme(pairs, name='mc%05i' % isamp, col='bycx', center=[0, 0, 0],
                # **showme_opts)
                lastviz = isamp

            if lowerr < besterr:
                besterr = lowerr
                best = symfit.xfit @ frames, symfit

                if symfit.total_err < goalerr * symfit.radius:
                    break
                # abserr = symframes_coherence(ipd.sym.sym_frames[kw.sym][:, None] @ frames[None, :])
                # if abserr < goalerr: break

    # pairs = ipd.sym.stupid_pairs_from_symops(symfit.symops)
    # ipd.showme(pairs, name='pairsstop', col='bycx', center=[0, 0, 0],
    # **showme_opts.sub(vizfresh=vizfresh))

    frames, symfit = best
    symdupframes = ipd.sym.sym_frames[kw.sym][:, None] @ frames[None, :]
    symdupframes = frames
    symerr = symframes_coherence(symdupframes)

    # os.system('yes | ffmpeg -i symfit_movie/symdup_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p symdup.mp4')
    # os.system('yes | ffmpeg -i symfit_movie/symops_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p symops.mp4')
    # print(f'symfit_mc_play SEED {seed:15} ERR {symfit.weighted_err:7.3}')

    # if showme:
    if False:
        symdupframes = ipd.sym.sym_frames[kw.sym][:, None] @ frames[None, :]
        ipd.showme(symdupframes, name="xfitmcfinal", col=None, **showme_opts)
        showme_opts.vizfresh = False
        ipd.showme(frames, name="xfitmc%05ib" % isamp, col=None, **showme_opts.sub(spheres=0.65, weight=1.5))
        pairs = ipd.sym.stupid_pairs_from_symops(symfit.symops, )
        ipd.showme(pairs, name="pairsstop", col="bycx", center=[0, 0, 0], **showme_opts)

    # print('symerr', symerr, isamp + 1)
    # assert 0
    # ipd.showme(frames, name='best', col=(1, 1, 1), center=[0, 0, 0], **showme_opts)
    # ipd.showme(symdupframes, name='xfitbest', col=(0, 0, 1), rays=0.1, **showme_opts)

    # t.report()

    return ipd.dev.Bunch(nsamp=isamp + 1, besterr=besterr, symerr=symerr, frames=frames, start_err=err0)  # type: ignore

def symframes_coherence(frames):
    frames = frames.reshape(-1, 4, 4)
    norms = np.linalg.norm(frames[:, None, :, 3] - frames[None, :, :, 3], axis=-1)
    np.fill_diagonal(norms, 9e9)
    normmin = np.sort(norms, axis=0)[2]  # should always be at leart 3 frames
    err = np.max(normmin, axis=0)
    # print(norms.shape, err)
    return err

def wrapper(*args, **kwargs):
    try:
        return symfit_mc_play(*args, **kwargs)
    except:  # noqa
        return ipd.dev.Bunch(nsamp=9999, besterr=999, symerr=999)

def symfit_parallel_convergence_trials(**kw):
    # ====== Octahedral convergence ======
    # perturbed from ideal
    # kw.tprelen = 10
    # kw.fuzzstdfrac = 0.1  # frac of radian
    # kw.cart_sd_fuzz = kw.fuzzstdfrac * kw.tprelen
    # kw.rot_sd_fuzz = kw.fuzzstdfrac
    #  5 iters  1811.2  fail 0.312  1.29 1.55 1.77 2.06 3.16
    #  6 iters  2076.0  fail 0.344  1.14 1.24 1.43 1.57 2.75
    #  7 iters  1914.9  fail 0.250  1.38 1.39 1.65 1.89 2.72
    #  8 iters  1230.7  fail 0.031  2.74 2.74 2.74 2.74 2.74
    #  9 iters  1542.4  fail 0.031  2.52 2.52 2.52 2.52 2.52
    # 10 iters  1823.3  fail 0.031  3.51 3.51 3.51 3.51 3.51
    # 11 iters  2454.2  fail 0.125  1.74 1.80 1.89 2.08 2.36
    # 12 iters  2850.6  fail 0.062  1.40 1.45 1.53 1.66 1.92
    # totally random frames:
    #  5 iters  9422.3  fail 0.938  5.53 999.00 999.00 999.00 999.00
    #  6 iters  8318.8  fail 0.812  0.75 2.76 999.00 999.00 999.00
    #  7 iters  4090.8  fail 0.312  1.42 2.53 3.63 501.94 999.00
    #  8 iters  4369.6  fail 0.344  0.87 0.94 1.79 999.00 999.00
    #  9 iters  5967.1  fail 0.531  0.44 1.66 2.61 999.00 999.00
    # 10 iters  5911.8  fail 0.438  0.87 1.04 1.45 2.94 999.00
    # 11 iters  7526.1  fail 0.656  0.59 1.14 2.17 3.93 999.00
    # 12 iters  9048.2  fail 0.844  0.47 1.03 1.39 5.37 999.00
    import concurrent.futures as cf
    from collections import defaultdict

    kw = ipd.dev.Bunch()
    ntrials = 32
    kw.goalerr = 0.1
    kw.maxiters = 10_000
    kw.quiet = True
    kw.showme = False
    seeds = list(np.random.randint(2**32 - 1) for i in range(ntrials))
    nnframesset = [5, 6, 7, 8, 9, 10, 11, 12]
    # nnframesset = [1, 2, 3, 4]
    # print('seeds', seeds)
    fut = defaultdict(dict)
    with cf.ProcessPoolExecutor() as exe:
        for inframes, nframes in enumerate(nnframesset):
            kw.nframes = nframes
            for iseed, seed in enumerate(seeds):
                kw.seed = seed
                # print('submit', terms, seed)
                # fut[nframes][seed] = exe.submit(symfit_mc_play, **kw)
                fut[nframes][seed] = exe.submit(wrapper, **kw)
        # for i, f in fut.items():
        # print(i, f.result())
        print("symfit_parallel_convergence_trials iters:")
        for nframes in nnframesset:
            niters = [f.result().nsamp for k, f in fut[nframes].items()]
            score = [f.result().symerr for k, f in fut[nframes].items()]
            badscores = [s for s in score if s > 3 * kw.goalerr]
            # badscores = []
            print(
                f"{nframes:4} iters {np.mean(niters):7.1f} ",
                f"fail {len(badscores)/ntrials:5.3f} ",
                " ".join(["%4.2f" % q for q in np.quantile(badscores, [0.0, 0.1, 0.25, 0.5, 1.0])] if badscores else "", ),
            )

def symfit_parallel_mc_scoreterms_trials(**kw):
    import concurrent.futures as cf
    from collections import defaultdict
    from itertools import chain, combinations

    termsset = list(chain(*(combinations("CHNA", i + 1) for i in range(4))))
    termsset = list(str.join("", combo) for combo in termsset)
    # termsset = ['C']

    kw = ipd.dev.Bunch()
    ntrials = 100
    kw.goalerr = 0.1
    kw.maxiters = 2000
    kw.quiet = True
    seeds = list(np.random.randint(2**32 - 1) for i in range(ntrials))
    # print('seeds', seeds)
    fut = defaultdict(dict)
    with cf.ProcessPoolExecutor() as exe:
        for iterms, terms in enumerate(termsset):
            kw.lossterms = terms
            for iseed, seed in enumerate(seeds):
                kw.seed = seed
                # print('submit', terms, seed)
                fut[terms][seed] = exe.submit(symfit_mc_play, **kw)
        # for i, f in fut.items():
        # print(i, f.result())
        print("symfit_parallel_mc_trials mean iters:")
        for terms in termsset:
            niters = [f.result().nsamp for k, f in fut[terms].items()]
            score = [f.result().symerr for k, f in fut[terms].items()]
            badscores = [s for s in score if s > 3 * kw.goalerr]
            # badscores = []
            print(
                f"{terms:4} iters {np.mean(niters):7.1f} ",
                f"fail {len(badscores)/ntrials:5.3f} ",
                " ".join(["%4.2f" % q for q in np.quantile(badscores, [0.0, 0.1, 0.25, 0.5, 1.0])] if badscores else "", ),
            )

def setup_test_frames(
    nframes,
    sym,
    cart_sd_fuzz,
    rot_sd_fuzz,
    tprelen=20,
    tprerand=0,
    tpostlen=10,
    tpostrand=0,
    noxpost=False,
    **kw,
):
    symframes = ipd.sym.sym_frames[sym]
    selframes = symframes[np.random.choice(len(symframes), nframes, replace=False), :, :]
    xpre = hm.rand_xform()
    xpre[:3, 3] = hm.rand_unit()[:3] * (tprelen + tprerand * (np.random.rand() - 0.5))
    xfuzz = hm.rand_xform_small(nframes, cart_sd=cart_sd_fuzz, rot_sd=rot_sd_fuzz)
    xpost = hm.rand_xform()
    xpost[:3, 3] = hm.rand_unit()[:3] * (tpostlen + tpostrand * (np.random.rand() - 0.5))
    if noxpost:
        xpost = np.eye(4)
    frames = xpost @ selframes @ xpre @ xfuzz  # move subunit
    radius = None
    return frames, xpre, xpost, xfuzz, radius

"""


   symop_hel_err
   symop_ang_err
   cen_align_err
   axes_fit_err
   radius_err
   redundant_cyclic_err


"""
