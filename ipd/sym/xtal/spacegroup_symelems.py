import itertools as it
from collections import defaultdict

import numpy as np
from opt_einsum import contract as einsum

import ipd
from ipd.homog.hgeom import (
    angle,
    h_point_line_dist,
    hdot,
    hnorm,
    hpoint,
    hvec,
    line_angle,
    line_line_closest_points_pa,
    line_line_distance_pa,
)
from ipd.sym.xtal.spacegroup_data import *
from ipd.sym.xtal.spacegroup_util import *
from ipd.sym.xtal.SymElem import *

def _inunit(p):
    x, y, z, w = p.T
    ok = (-0.001 < x) * (x < 0.999) * (-0.001 < y) * (y < 0.999) * (-0.001 < z) * (z < 0.999)
    return ok

def _flipaxs(a):
    if np.sum(a[:3] * [1, 1.1, 1.2]) < 0:
        a[:3] *= -1
    return a

def _compute_symelems(
    spacegroup,
    unitframes=None,
    lattice=None,
    aslist=False,
    find_alternates=True,
    profile=False,
):
    t = ipd.dev.Timer()

    if unitframes is None:
        unitframes = ipd.sym.xtal.sgframes(spacegroup, cellgeom="unit")

    latticetype = sg_lattice[spacegroup]
    if lattice is None:
        lattice = lattice_vectors(spacegroup, cellgeom="nonsingular")

    ncell = number_of_canonical_cells(spacegroup)
    f2cel = latticeframes(unitframes, lattice, cells=ncell - 2)
    f4cel = latticeframes(unitframes, lattice, cells=ncell)
    t.checkpoint("make_frames")

    # for f in f4cel:
    # if np.allclose(f[:3, :3], np.eye(3)) and f[0, 3] == 0 and f[2, 3] == 0:
    # print(f)

    # relframes = einsum('aij,bjk->abik', f2cel, ipd.)
    axs0, ang, cen0, hel0 = ipd.homog.axis_angle_cen_hel_of(f2cel)
    axs, cen, hel = axs0[:, :3], cen0[:, :3], hel0[:, None]
    frameidx = np.arange(len(f2cel))[:, None]
    tag0 = np.concatenate([axs, cen, hel, frameidx], axis=1).round(10)
    t.checkpoint("make_tags")

    # idx = np.isclose(ang, np.pi / 2)
    # ic(axs0[idx])
    # ic(ang[idx])
    # ic(cen0[idx])
    # ic(hel0[idx] / 1.7)
    # assert 0

    symelems = defaultdict(list)
    for nfold in [2, 3, 4, 6, -1, -2, -3, -4, -6]:
        screw, nfold = nfold < 0, abs(nfold)
        nfang = 2 * np.pi / nfold if nfold != 1 else 0

        # idx = np.isclose(ang, nfang, atol=1e-6)
        if screw:
            idx = np.logical_and(np.isclose(ang, nfang, atol=1e-6), ~np.isclose(0, hel[:, 0]))
        else:
            idx = np.logical_and(np.isclose(ang, nfang, atol=1e-6), np.isclose(0, hel[:, 0]))
        if np.sum(idx) == 0:
            continue

        # ic(nfold, screw, tag0[idx])
        t.checkpoint("make_tags")
        nftag = tag0[idx]
        nftag = nftag[np.lexsort(-nftag.T, axis=0)]
        nftag = np.unique(nftag, axis=0)
        nftag = nftag[np.argsort(-nftag[:, 0], kind="stable")]
        nftag = nftag[np.argsort(-nftag[:, 1], kind="stable")]
        nftag = nftag[np.argsort(-nftag[:, 2], kind="stable")]
        nftag = nftag[np.argsort(-nftag[:, 5], kind="stable")]
        nftag = nftag[np.argsort(-nftag[:, 4], kind="stable")]
        nftag = nftag[np.argsort(-nftag[:, 3], kind="stable")]

        d = np.sum(nftag[:, 3:6]**2, axis=1).round(6)
        nftag = nftag[np.argsort(d, kind="stable")]
        t.checkpoint("sort_tags")

        # remove symmetric dups
        keep = nftag[:1]
        for itag, tag in enumerate(nftag[1:]):
            symtags = _make_symtags(tag, f4cel)
            seenit = np.all(np.isclose(keep[None], symtags[:, None], atol=0.001), axis=2)
            if np.any(seenit):
                continue
            t.checkpoint("make_symtags")

            picktag = _pick_symelemtags(symtags, symelems)
            t.checkpoint("_pick_symelemtags")

            # picktag = None
            if picktag is None:
                keep = np.concatenate([keep, tag[None]])
            else:
                keep = np.concatenate([keep, picktag[None]])
        t.checkpoint("remove_symmetric_dups")

        for tag in keep:
            try:
                axs_, cen_, hel_, iframe = tag[:3], tag[3:6], tag[6], int(tag[7])
                se = SymElem(nfold, axs_, cen_, hel=hel_, lattice=lattice, isunit=True, latticetype=latticetype)
                seenit = symelems[se.label].copy()
                if screw and se.label[:2] in symelems:
                    seenit += symelems[se.label[:2]]
                if not any([_symelem_is_same(se, se2, f4cel) for se2 in seenit]):
                    symelems[se.label].append(se)
            except (ScrewError, OutOfUnitCellError):
                continue
        t.checkpoint("make_symelems")

    symelems = _shift_to_unitcell(symelems)
    t.checkpoint("_shift_to_unitcell")

    symelems = _symelem_remove_ambiguous_syms(symelems)
    t.checkpoint("_symelem_remove_ambiguous_syms")

    symelems = _pick_best_related_symelems(symelems, spacegroup, lattice, f4cel, f2cel, find_alternates)
    t.checkpoint("_pick_best_related_symelems")

    symelems = _remove_redundant_translations(symelems)
    t.checkpoint("_remove_redundant_translations")

    symelems = _remove_redundant_screws(symelems, f4cel, lattice)
    t.checkpoint("_remove_redundant_screws")

    newc11 = list()
    for e in symelems["C11"]:
        if e.hel < 1.0001:
            newc11.append(e)
    symelems["C11"] = newc11

    for k in list(symelems.keys()):
        if not symelems[k]:
            del symelems[k]

    if spacegroup == "P312":
        e = SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.0, 0.0, 0.0], lattice=lattice)
        symelems["C2"].append(e.tounit(lattice))

    if aslist:
        symelems = list(itertools.chain(*symelems.values()))

    if profile:
        t.report()

    return symelems

def _pick_best_related_symelems(symelems, spacegroup, lattice, f4cel, f2cel, find_alternates):
    # move back to unitcell postions, identical for cubic groups
    # 'nonsingular' lattice used to avoid cases where symelems only appear
    # in a particular lattice configuration
    newelems = defaultdict(list)
    for psym, elems in symelems.items():
        if elems[0].iscyclic and find_alternates:
            for ielem, unitelem in enumerate(elems):
                e = unitelem.tolattice(lattice)
                # unitelem = e.tounit(lattice)
                elem = _pick_alternate_elems(spacegroup, lattice, unitelem, f4cel, f2cel)
                if elem is None:
                    # newelems[psym].append((ielem - 9999, e))
                    print("!" * 80)
                    print(
                        f"WARNING {spacegroup} failed to find matches for element:\n{unitelem}\nWill be missing some symelements"
                    )
                    print("!" * 80, flush=True)
                    continue
                    from ipd.viz.pymol_viz import showme

                    showme(e, scale=10)
                    showme(f2cel, scale=10)
                    # assert 0
                    assert 0

                else:
                    newelems[psym].append(elem)
            # remove dups
            newelems[psym] = list({k: (k, v) for k, v in newelems[psym]}.values())
            # pick best
            newelems[psym] = [e for s, e in sorted(newelems[psym])]
        else:
            for unitelem in elems:
                assert not unitelem.iscyclic
                # unitelem = e.tounit(lattice)
                newelems[psym].append(unitelem)
    return newelems

def _find_compound_symelems(
    spacegroup,
    se=None,
    frames=None,
    frames2=None,
    frames1=None,
    aslist=False,
    lattice=None,
):
    timer = ipd.dev.Timer()
    if se is None:
        assert 0
        se = ipd.sym.xtal.symelems(spacegroup, asdict=False, screws=False)
    if lattice is None:
        lattice = lattice_vectors(spacegroup, cellgeom="nonsingular")
    se = [e.tolattice(lattice) for e in se if e.iscyclic]
    if frames is None:
        assert frames2 is None
        frames = ipd.sym.xtal.sgframes(spacegroup, cells=3, cellgeom="nonsingular")
        frames2 = ipd.sym.xtal.sgframes(spacegroup, cells=2, cellgeom="nonsingular")
        frames1 = ipd.sym.xtal.sgframes(spacegroup, cells=1, cellgeom="nonsingular")
    lattice = lattice_vectors(spacegroup, cellgeom="nonsingular")
    timer.checkpoint("start")
    isects = defaultdict(set)

    se_uniqaxisframes = _se_unique_axisline_frames(se, frames)
    timer.checkpoint("uniqaxisframes")

    for (i1, e1start), (i2, e2) in it.product(enumerate(se), enumerate(se)):
        # seene1line = list()
        for ifrm, frm in enumerate(se_uniqaxisframes[i1]):
            timer.checkpoint("misc")
            e1 = e1start.xformed(frm)
            timer.checkpoint("xformed")
            # for eseen in seene1line:
            # if np.isclose(0, line_line_distance_pa(e1.cen, e1.axis, eseen.cen, eseen.axis)):
            # ic('skip')
            # break
            # else:
            # seene1line.append(e1)
            # if e1.id == e2.id: continue
            axis, cen = e1.axis, e1.cen
            symcen = einsum("fij,j->fi", se_uniqaxisframes[i2], e2.cen)
            symaxis = einsum("fij,j->fi", se_uniqaxisframes[i2], e2.axis)
            taxis, tcen = [np.tile(x, (len(symcen), 1)) for x in (axis, cen)]  # type: ignore
            timer.checkpoint("symcenaxs")
            p, q = line_line_closest_points_pa(tcen, taxis, symcen, symaxis)
            timer.checkpoint("line_line_closest_points_pa")
            d = hnorm(p - q)
            p = (p+q) / 2
            ok = _inunit(p)
            ok = np.logical_and(ok, d < 0.001)
            if np.sum(ok) == 0:
                continue
            axis2 = symaxis[ok][0]  # type: ignore
            cen = p[ok][0]
            axis = einsum("fij,j->fi", se_uniqaxisframes[i2], axis)
            axis2 = einsum("fij,j->fi", se_uniqaxisframes[i2], axis2)
            cen = einsum("fij,j->fi", se_uniqaxisframes[i2], cen)
            axis = axis[_inunit(cen)]  # type: ignore
            axis2 = axis2[_inunit(cen)]  # type: ignore
            cen = cen[_inunit(cen)]  # type: ignore
            # ic(cen)
            pick = np.argmin(hnorm(cen - [0.003, 0.002, 0.001, 1]))
            axis = _flipaxs(axis[pick])
            axis2 = _flipaxs(axis2[pick])
            nf1, nf2 = e1.nfold, e2.nfold
            # D np.pi/2
            ang = angle(axis, axis2)
            if e2.nfold > e1.nfold:
                nf1, nf2 = e2.nfold, e1.nfold
                axis, axis2 = axis2, axis
            if np.isclose(ang, np.pi / 2):
                psym = f"D{nf1}"
            elif (nf1, nf2) == (3, 2) and np.isclose(ang, 0.9553166181245092):
                psym = "T"
            elif any([
                (nf1, nf2) == (3, 2) and np.isclose(ang, 0.6154797086703874),
                (nf1, nf2) == (4, 3) and np.isclose(ang, 0.9553166181245092),
                (nf1, nf2) == (4, 2) and np.isclose(ang, 0.7853981633974484),
            ]):
                psym = "O"
            # elif (nf1, nf2) in [(2, 2), (3, 3)]:
            # continue
            else:
                # print('SKIP', nf1, nf2, ang, flush=True)
                continue
            cen = cen[pick]
            t = tuple([f"{psym}{nf1}{nf2}", *cen[:3].round(9), *axis[:3].round(9), *axis2[:3].round(9)])
            isects[psym].add(t)
    timer.checkpoint("isects")
    # ic(isects['T'])
    # assert 0

    # remove redundant centers
    for psym in isects:
        isects[psym] = list(sorted(isects[psym]))  # type: ignore
        isects[psym] = list({t[1:4]: t for t in isects[psym]}.values())  # type: ignore
    # ic(isects['D2'])
    # assert 0
    compound = defaultdict(list)
    for psym in isects:
        seenit = list()
        for t in isects[psym]:
            nfold_, axis_, cen_, axis2_ = t[0], t[4:7], hpoint(t[1:4]), t[7:10]
            if any([hnorm(cen_ - s) < 0.0001 for s in seenit]):
                continue
            seenit.append(cen_)
            elem = SymElem(nfold_, axis_, cen_, axis2_, lattice=lattice, isunit=True)
            compound[psym].append(elem)

    timer.checkpoint("prune1")
    # newd2 = list()
    # for ed2 in compound['D2']:
    #    if not np.any([np.allclose(ed2.cen, eto.cen) for eto in it.chain(compound['T'], compound['O'], compound['D4'])]):
    #       newd2.append(ed2)
    # compound['D2'] = newd2
    # newd3 = list()
    # for ed3 in compound['D3']:
    #    if not np.any([np.allclose(ed3.cen, eo.cen) for eo in compound['O']]):
    #       newd3.append(ed3)
    # compound['D3'] = newd3
    # newd4 = list()
    # for ed4 in compound['D4']:
    #    if not np.any([np.allclose(ed4.cen, eo.cen) for eo in compound['O']]):
    #       newd4.append(ed4)
    # compound['D4'] = newd4
    compound = {k: v for k, v in compound.items() if len(v)}

    # returns unit elems
    compound = _pick_bestframe_compound_elems(spacegroup, compound, lattice, frames, frames2)
    timer.checkpoint("prune2")

    if aslist:
        compound = list(itertools.chain(*compound.values()))

    # timer.report()

    return compound

def _se_unique_axisline_frames(symelems, frames):
    uniqaxisframes = list()
    for ie, e in enumerate(symelems):
        seenit = list()
        xaxs = einsum("fij,j->fi", frames, e.axis)
        xcen = einsum("fij,j->fi", frames, e.cen)
        for ifrm, frame in enumerate(frames):
            for f, saxs, scen in seenit:
                d = line_line_distance_pa(xcen[ifrm], xaxs[ifrm], scen, saxs)  # type: ignore
                if np.isclose(0, d):
                    break
            else:
                seenit.append((frame, xaxs[ifrm], xcen[ifrm]))  # type: ignore
        uniqaxisframes.append(np.stack([f for f, a, c in seenit]))
    return uniqaxisframes

def _pick_bestframe_compound_elems(spacegroup, compound_elems, lattice, frames, frames2):
    newelems = defaultdict(list)
    allsyms = "I O T D6 D4 D3 D2".split()
    assert all(psym in allsyms for psym in compound_elems.keys())

    # remove overlapping elems in order of possible containment (e.g. D4 contains D2)
    hasuniquecen = list()
    for psym in "I O T D6 D4 D3 D2".split():
        if psym not in compound_elems:
            continue
        for uelem in compound_elems[psym]:
            assert uelem.isunit
            elem = uelem.tolattice(lattice)
            symcen = einsum("fij,j", frames, elem.cen)
            for elem2 in hasuniquecen:
                d = hnorm(symcen[None] - elem2.cen.reshape(1, 4))  # type: ignore
                if np.isclose(0, np.min(d)):
                    break
            else:
                # ic(elem)
                hasuniquecen.append(elem)

    bestplaced = list()
    for ielem, elem in enumerate(hasuniquecen):
        # try:
        # elem.matching_frames(frames2)
        # except ComponentIDError as cperr:
        # print('NOT ALL MATCHING FRAMES FOR', elem, len(cperr.match), 'of', len(elem.operators))
        # bestplaced.append(elem.tounit(lattice))
        # continue
        bestiframes, bestelem = [9e9], None
        bestbadiframes, bestbadelem = [9e9], None
        for iframe, elemframe in enumerate(frames):
            movedelem = elem.xformed(elemframe)
            try:
                iframes = movedelem.matching_frames(frames2)
                if np.max(iframes) < np.max(bestiframes):
                    bestiframes, bestelem = iframes, movedelem
            except ComponentIDError as cperr:
                # if elem.label == 'D2':
                # ic(cperr.match)
                if len(cperr.match) >= len(bestbadiframes):  # type: ignore
                    if np.max(cperr.match) < np.max(bestbadiframes):  # type: ignore
                        bestbadiframes, bestbadelem = cperr.match, movedelem  # type: ignore
                continue
        if bestelem is None:
            print(
                "NOT ALL MATCHING FRAMES FOR",
                elem,
                len(bestbadiframes),
                "of",
                len(elem.operators),
                bestbadiframes,
            )
            bestelem = bestbadelem
        bestelem = bestelem.tounit(lattice)  # type: ignore
        assert bestelem.isunit
        bestplaced.append(bestelem)

    # ic(bestplaced)
    # assert 0

    compound_elems = defaultdict(list)
    for elem in bestplaced:
        compound_elems[elem.label].append(elem)

    # _printelems(spacegroup, compound_elems)

    return compound_elems

# def _to_central_symelem(frames, elem, cen):
#    ic(elem)
#    ic(elem.matching_frames(frames))
#    ic(elem.matching_frames(ipd.sym.xtal.sgframes('I432', cells=2)))
#    # ipd.showme(elem, scale=10)
#    # ipd.showme(elem.operators, scale=10)
#    # ipd.showme(frames, scale=10)
#    if not elem.isdihedral: return elem
#    ic(len(frames))
#    for i, f in enumerate(frames):
#       cen = einsum('ij,j->i', f, elem.cen)
#       # if not _inunit(cen): continue
#       elem2 = SymElem(
#          elem.nfold,
#          einsum('ij,j->i', f, elem.axis),
#          cen,
#          einsum('ij,j->i', f, elem.axis2),
#       )

#       try:
#          m = np.max(elem2.matching_frames(frames))
#       except AssertionError:
#          pass
#       if 48 * 8 > m:
#          ic(elem)
#          assert 0
#    assert 0
#    return

#    # ic(elem)
#    cens = einsum('fij,j->fi', frames, elem.cen)
#    ipick = np.argmin(hnorm(cens - cen))
#    frame = frames[ipick]
#    elem = SymElem(
#       elem.nfold,
#       einsum('ij,j->i', frame, elem.axis),
#       einsum('ij,j->i', frame, elem.cen),
#       einsum('ij,j->i', frame, elem.axis2),
#    )
#    return elem

def _symelem_is_same(elem, elem2, frames):
    assert elem.iscyclic or elem.isscrew
    assert elem2.iscyclic or elem2.isscrew
    assert elem.label[:2] == elem2.label[:2]
    axis = einsum("fij,j->fi", frames, elem2.axis)
    axsame = np.all(np.isclose(axis, elem.axis), axis=1)  # type: ignore
    axsameneg = np.all(np.isclose(-axis, elem.axis), axis=1)  # type: ignore
    axok = np.logical_or(axsame, axsameneg)
    if not np.any(axok):
        return False
    frames = frames[axok]
    axis = axis[axok]  # type: ignore
    cen = einsum("fij,j->fi", frames, elem2.cen)
    censame = np.all(np.isclose(cen, elem.cen), axis=1)  # type: ignore
    # ic(censame.shape)
    if any(censame):
        return True
    # if any(censame):
    # ic(elem.axis, axis[censame])
    # ic(elem.cen, cen[censame])
    # ic(elem)
    # ic(elem2)
    # assert not any(censame)  # should have been filtered out already

    d = h_point_line_dist(elem.cen, cen, axis)
    return np.min(d) < 0.001

def _make_symtags(tag, frames):
    concat = np.concatenate
    tax, tcen = hvec(tag[:3]), hpoint(tag[3:6])
    thel = np.tile(tag[6], [len(frames), 1])
    tif = np.tile(tag[6], [len(frames), 1])

    # if is 21, 42, 63 screw, allow reverse axis with same helical shift
    c1 = (frames @ tax)[:, :3]
    c2 = (frames @ tcen)[:, :3]
    if np.any(np.isclose(thel, [0.5, np.sqrt(2) / 2])):
        # if is 21, 42, 63 screw, allow reverse axis with same helical shift
        symtags = concat([
            concat([c1, c2, +thel, tif], axis=1),
            concat([-c1, c2, -thel, tif], axis=1),
            concat([-c1, c2, +thel, tif], axis=1),
        ])
    else:
        symtags = concat([
            concat([c1, c2, +thel, tif], axis=1),
            concat([-c1, c2, -thel, tif], axis=1),
        ])

    return symtags

# def _make_symtags_torch(tag, frames, torch_device, t):
#    import torch
#    tag = torch.tensor(tag, device=torch_device).to(torch.float32)
#    # frames = torch.tensor(frames, device=torch_device).to(torch.float32)
#    concat = torch.cat
#    tax, tcen, thel = h.vec(tag[:3]), h.point(tag[3:6]), torch.tile(tag[6], [len(frames), 1])
#
#    # concat = np.concatenate
#    # tax, tcen, thel = hvec(tag[:3]), hpoint(tag[3:6]), np.tile(tag[6], [len(frames), 1])
#
#    const1 = torch.tensor(0.5, device=torch_device).to(torch.float32)
#    const2 = torch.tensor(np.sqrt(2.) / 2., device=torch_device).to(torch.float32)
#    c1 = (frames @ tax)[:, :3]
#    c2 = (frames @ tcen)[:, :3]
#    if torch.any(torch.isclose(thel, const1)) or torch.any(torch.isclose(thel, const2)):
#       # if is 21, 42, 63 screw, allow reverse axis with same helical shift
#       symtags = concat([
#          concat([c1, c2, +thel], axis=1),
#          concat([-c1, c2, -thel], axis=1),
#          concat([-c1, c2, +thel], axis=1),
#       ])
#    else:
#       symtags = concat([
#          concat([c1, c2, +thel], axis=1),
#          concat([-c1, c2, -thel], axis=1),
#       ])
#    return symtags

def _shift_to_unitcell(symelems):
    symelems = symelems.copy()
    for psym, elems in symelems.items():
        # if elems and not elems[0].isscrew: continue
        # if elems and not elems[0].ic: continue
        newelems = list()
        for e in elems:
            assert e.isunit
            cen = e.cen
            cen[:3] = cen[:3] % 1
            cen[:3][cen[:3] > 0.9999] = 0
            newelems.append(SymElem(e.nfold, e.axis, cen, hel=e.hel, screw=e.screw, frame=e.frame, isunit=True))
            # ic(e)
            # ax = list(sorted(np.abs(e.axis[:3])))
            # if np.allclose(ax, [0, 0, 1]):
        symelems[psym] = newelems

    #
    # elif np.allclose(ax, [0, np.sqrt(2)/2,np.sqrt(2)/2]):
    # e.cen[:3] = e.cen[:3] % 1.0
    # else:
    # assert 0

    return symelems

def _pick_symelemtags(symtags, symelems):
    # assert 0, 'this is incorrect somehow'

    # for i in [0, 1, 2]:
    #    symtags = symtags[np.argsort(-symtags[:, i], kind='stable')]
    # for i in [6, 5, 4, 3]:
    #    symtags = symtags[symtags[:, i] > -0.0001]
    #    symtags = symtags[symtags[:, i] < +0.9999]
    # for i in [5, 4, 3]:
    #    symtags = symtags[np.argsort(symtags[:, i], kind='stable')]
    # if len(symtags) == 0: return None
    # # ic(symtags)

    cen = [se.cen[:3] for psym in symelems for se in symelems[psym]]
    if cen and len(symtags):
        w = np.where(np.all(np.isclose(symtags[:, None, 3:6], np.stack(cen)[None]), axis=2))[0]
        if len(w) > 0:
            return symtags[w[0]]
    return None

    # d = np.sum(symtags[:, 3:6]**2, axis=1).round(6)
    # symtags = symtags[np.argsort(d, kind='stable')]
    # # ic(symtags[0])
    # return symtags[0]

def _symelem_remove_ambiguous_syms(symelems):
    symelems = symelems.copy()
    for sympair in ("c2/c4 c2/c6 c3/c6 c11/c21 c11/c31 c11/c32 c11/c41 c11/c42 c11/c43 "
                    "c11/c61 c11/c62 c11/c63 c11/c64 c11/c65 c21/c41 c21/c42 c21/c43 c21/c61 "
                    "c21/c62 c21/c63 c21/c64 c21/c65 c31/c61 "
                    "c31/c62 c31/c63 c31/c64 c31/c65 c32/c61 c32/c62 c32/c63 c32/c64 c32/c65 ").split():
        sym1, sym2 = sympair.upper().split("/")
        if sym2 in symelems:
            newc2 = list()
            for s2 in symelems[sym1]:
                for s in symelems[sym2]:
                    if np.allclose(s.axis, s2.axis) or np.allclose(s.axis, -s2.axis):
                        if h_point_line_dist(s.cen, s2.cen, s.axis) < 0.001:
                            break
                        # if np.allclose(s.cen, s2.cen): break

                else:
                    newc2.append(s2)
            symelems[sym1] = newc2

    for k in list(symelems.keys()):
        if not symelems[k]:
            del symelems[k]

    return symelems

def _pick_alternate_elems(sym, lattice, unitelem, frames, frames2):
    # checked_elems = list()
    # for i, unitelem in enumerate(elems):
    # alternate_elem_frames = elems[0].tolattice(lattice).operators

    best, argbest = (999999999, ), None
    seenit = list()
    assert unitelem.isunit
    elem0 = unitelem.tolattice(lattice)
    for j, elemframe in enumerate(frames2):
        if j == 0:
            assert np.allclose(elemframe, np.eye(4))
        try:
            elem = elem0.xformed(elemframe)
        except OutOfUnitCellError:
            continue
        try:
            iframes = elem.matching_frames(frames)
            # m = np.max(iframes)
            m = tuple(sorted(iframes))
            # ic(m)
            if m < best:
                # ic(iframes)
                # ic(j, m)
                best, argbest = m, elem
        except ComponentIDError:
            continue
    if argbest is None:
        return None
    return best, argbest.tounit(lattice)

    # for j, elemframe in enumerate(frames2):
    #    if j == 0: assert np.allclose(elemframe, np.eye(4))
    #    elem = unitelem.tolattice(lattice).xformed(elemframe)
    #    try:
    #       m = elem.matching_frames(frames)
    #       if np.any(m >= len(frames2)): raise ComponentIDError
    #       # compids = elem.frame_component_ids(frames, perms, sanitycheck=True)
    #       # opids = elem.frame_operator_ids(frames, sanitycheck=True)
    #       # opcompids = _make_operator_component_joint_ids(elem, elem, frames, opids, compids, sanitycheck=True)
    #    except ComponentIDError:
    #       print(f'{sym} checking alternate elem')
    #       continue
    #    return elem.tounit(lattice)
    #    # checked_elems.append(elem.tounit(lattice))
    #    # ic('success')
    #    # break
    # else:
    #    assert 0

# assert len(checked_elems) == len(elems)
# return checked_elems
def _remove_redundant_translations(symelems):
    if "C11" not in symelems:
        return symelems
    newsymelems = symelems.copy()
    del newsymelems["C11"]
    allelems = list(itertools.chain(*symelems.values()))
    newtrans = list()
    for t in symelems["C11"]:
        for e in allelems:
            if e.iscyclic and not np.allclose(line_angle(e.axis, t.axis), 0):
                # must be parallel to all cyclic
                break
            if e.isscrew and e.nfold > 1 and not np.allclose(hdot(e.axis, t.axis), 0):
                # must be perp to all cyclic
                break
        else:
            newtrans.append(t)
    newsymelems["C11"] = newtrans
    return newsymelems

def _remove_redundant_screws(symelems, frames, lattice):
    debug = False
    symelems = symelems.copy()
    for k in list(symelems.keys()):
        if not symelems[k]:
            del symelems[k]
    for psym, elems in symelems.items():
        if not elems[0].isscrew:
            continue
        # if elems[0].nfold == 1: continue
        newelems = list()
        elemssort = sorted(elems, key=lambda e: abs(e.hel))
        for ie, eunit in enumerate(elemssort):
            e = eunit.tolattice(lattice)
            symaxis = hxform(frames, e.axis)
            symcen = hxform(frames, e.cen)
            if debug:
                ipd.showme(symcen, scale=12, name=f"cen_{ie}")
            if debug:
                ipd.showme(e, scale=12, name=f"elem_{ie}")
            for ie2, e2unit in enumerate(newelems):
                e2 = e2unit.tolattice(lattice)
                if debug:
                    ipd.showme(e2, scale=12, name=f"other_{ie}{ie2}")
                d = line_line_distance_pa(symcen, symaxis, e2.cen, e2.axis)
                # d = h_point_line_dist(symcen, e2.cen, e2.axis)
                dup = np.logical_and(np.isclose(0, d, atol=1e-3), np.abs(hdot(symaxis, e2.axis)) > 0.999)
                if np.any(dup):
                    break
            else:
                newelems.append(eunit)
        symelems[psym] = newelems
    return symelems

def _printelems(sym, elems):
    print("-" * 80)
    print(sym)
    # print(f'   assert set(elems.keys()) == set(\'{" ".join(elems.keys())}\'.split())')
    print("   val = dict(")
    for k, v in elems.items():
        print(f"      {k}=[")
        for e in v:
            print(f"         {e},")
        print("      ],")
    print(")")

    print("-" * 80, flush=True)
