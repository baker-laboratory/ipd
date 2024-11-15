import functools

import numpy as np

import ipd
from ipd.dock.rigid.objective import tooclose_overlap

def asuslide(
    sym,
    coords,
    frames=None,
    axes=None,
    existing_olig=None,
    alongaxis=None,
    towardaxis=None,
    boundscheck=lambda x: True,
    printme=False,
    cellsize=1,
    isxtal=False,
    nbrs="auto",
    doscale=True,
    doscaleiters=True,
    iters=5,
    subiters=1,
    clashiters=5,
    receniters=2,
    step=10,
    scalestep=None,
    closestfirst=True,
    centerasu="toward_partner",
    centerasu_at_start=False,
    showme=False,
    scaleslides=1.0,
    iterstepscale=0.5,
    coords_to_asucen=False,
    along_extra_axes=[],
    xtalrad=0.5,
    timer=None,
    **kw,
):
    kw = ipd.dev.Bunch(kw)

    if isinstance(cellsize, (int, float)):
        cellsize = [float(cellsize)] * 3
    if not isinstance(cellsize, np.ndarray):
        cellsize = np.array(cellsize, dtype=np.float64)
    if printme:
        coordstr = (repr(coords).replace(" ", "").replace("\n", "").replace("\t", "").replace("float32", "np.float32"))
        framestr = repr(frames).replace(" ", "").replace("\n", "").replace("\t", "")
        print(f"""      #yapf: disable
      kw = {repr(dict(kw))}
      coords=np.{coordstr}
      # yapf: enable
      frames={framestr}
      slid = asuslide(sym='{sym}',coords=coords,frames=frames,axes={axes},existing_olig={existing_olig},alongaxis={alongaxis},towardaxis={towardaxis},printme=False,cellsize={repr(cellsize)},isxtal={isxtal},nbrs={repr(nbrs)},doscale={doscale},iters={iters},subiters={subiters},clashiters={clashiters},receniters={receniters},step={step},scalestep={scalestep},closestfirst={closestfirst},centerasu={repr(centerasu)},centerasu_at_start={centerasu_at_start},showme={showme},scaleslides={scaleslides},iterstepscale={iterstepscale},coords_to_asucen={coords_to_asucen},xtalrad={xtalrad},maxstep={kw.maxstep},**kw,)
      ipd.showme(slid)
      # slid.dump_pdb(f'/home/sheffler/DEBUG_slid_ipd.pdb')
      """)
    kw = ipd.dev.Bunch(kw)
    kw.showme = showme
    if timer is None:
        timer = ipd.dev.Timer()
    kw.timer = timer
    kw.boundscheck = boundscheck
    kw.scaleslides = scaleslides
    coords = ipd.homog.hpoint(coords).copy()
    coords = coords.reshape(-1, 4)
    if scalestep is None:
        scalestep = step
    if axes is None:
        axes0 = ipd.sym.axes(sym, cellsize=cellsize)
        if isinstance(axes0, dict):
            axes = [(ax, ipd.homog.hpoint([0, 0, 0])) for ax in axes0.values()]
        else:
            isxtal = True
            axes = [(elem.axis, elem.cen) for elem in axes0]
            doscale = True if doscale is None else doscale
        com = ipd.homog.hcom(coords)
        faxdist = [ipd.homog.hpointlinedis(com, ac[1], ipd.hscaled(cellsize, ac[0])) for ac in axes]
        faxorder = np.argsort(faxdist)
        axes = [axes[i] for i in faxorder]
        if not closestfirst:
            axes = list(reversed(axes))

    # ic(alongaxis, towardaxis)
    if alongaxis is None and towardaxis is None:
        alongaxis = not isxtal
        towardaxis = isxtal
    if alongaxis is True and towardaxis is None:
        towardaxis = False
    if alongaxis is False and towardaxis is None:
        towardaxis = True
    if towardaxis is True and alongaxis is None:
        alongaxis = False
    if towardaxis is False and alongaxis is None:
        alongaxis = True
    # ic(alongaxis, towardaxis)

    if frames is None:
        frames = ipd.sym.frames(sym, cellsize=cellsize, xtalrad=xtalrad, allowcellshift=True, **kw)

    # assert towardaxis
    # assert not alongaxis
    # clashfunc = tooclose_clash
    # kw.tooclosefunc = clashfunc
    # userfunc = functools.partial(kw.get('tooclosefunc', tooclose_overlap), printme=printme, **kw)
    kw.tooclosefunc = functools.partial(kw.get("tooclosefunc", tooclose_overlap), printme=printme, **kw)

    assembly = ipd.dock.rigid.RigidBodyFollowers(sym=sym,
                                                 coords=coords,
                                                 frames=frames,
                                                 recenter=True,
                                                 cellsize=cellsize,
                                                 **kw)
    if showme:
        ipd.showme(assembly, name="START", **kw)
    # assembly.dump_pdb(f'/home/sheffler/DEBUG_asuslide_start_{ipd.dev.datetimetag()}.pdb')

    cellsize0 = cellsize
    if centerasu_at_start:
        recenter_asu_frames(assembly, method="to_center", axis=axes, **kw)
        if printme:
            ic(f"recenter {centerasu}")  # type: ignore
        # if printme: ic(f'scale {cellsize}')
    if doscale and not alongaxis:
        cellsize = slide_scale(assembly, cellsize, step=scalestep, **kw)
    for i in range(iters):
        # if i >= clashiters: kw.tooclosefunc = userfunc
        # ic(step)
        for j in range(subiters):
            # ic('asuslide iter', i, j)
            if doscaleiters and not alongaxis:
                cellsize = slide_scale(assembly, cellsize, step=scalestep, **kw)
            # cellsize = slide_scale(assembly, cellsize, step=scalestep, **kw)
            for iax, (axis, axpos) in enumerate(axes):
                cellsize = slide_cellsize(assembly, cellsize, step=scalestep, **kw)
                timer.checkpoint("slide_cellsize")
                axis = ipd.homog.hnormalized(axis)
                axpos = ipd.hscaled(cellsize / cellsize0, ipd.homog.hpoint(axpos))
                if towardaxis:
                    # ic(axpos)
                    axisperp = ipd.homog.hnormalized(
                        ipd.hprojperp(axis,
                                      assembly.asym.com() - axpos - ipd.homog.hrandvec() / 1000))
                    # ic(axis, assembly.asym.com() - axpos, cellsize, axisperp)
                    if centerasu and i < receniters:
                        recenter_asu_frames(assembly, method=centerasu, axis=axisperp, **kw)
                        if printme:
                            ic(f"recenter {centerasu}")  # type: ignore
                    else:
                        # ic(axpos)
                        timer.checkpoint("slide_axis_perp_before")
                        slide = slide_axis(axisperp, assembly, perp=True, nbrs=None, step=step, **kw)  # type: ignore
                        # if printme: ic(f'slide along {axisperp[:3]} by {slide}')
                        timer.checkpoint("slide_axis_perp")
                if i < alongaxis:  # type: ignore
                    slide = slide_axis(axis, assembly, nbrs="auto", step=step, **kw)
                    timer.checkpoint("slide_axis")
                    # printme: ic(f'slide along {axis[:3]} by {slide}')
                if doscale and alongaxis:
                    slide = slide_axis(assembly.asym.com(), assembly, nbrs=None, step=step, **kw)  # type: ignore
                    timer.checkpoint("slide_axis")
                elif doscale:
                    cellsize = slide_cellsize(assembly, cellsize, step=scalestep, **kw)
                    timer.checkpoint("slide_cellsize")
                # if showme == 'pdb': assembly.dump_pdb(f'slide_i{i}_j{j}_iax{iax}.pdb')
            for axis in along_extra_axes:
                slide = slide_axis(axis, assembly, nbrs="auto", step=step, **kw)
                timer.checkpoint("slide_axis")
                # printme: ic(f'slide along {axis[:3]} by {slide}')
            if coords_to_asucen:
                cencoords = ipd.sym.coords_to_asucen(sym, assembly.asym.coords)
                # ic(ipd.homog.hcom(assembly.asym.coords))
                # ic(ipd.homog.hcom(cencoords))
                assembly.set_asym_coords(cencoords)
        step *= iterstepscale
        scalestep *= iterstepscale

    if showme:
        ipd.showme(assembly, name="FINISH", **kw)
    return assembly

def tooclose_by_symelem(assembly, tooclosefunc, **kw):
    if assembly.sym is None:
        return [tooclosefunc(assembly, **kw)]
    assoc = ipd.sym.xtal.symelem_associations(assembly.sym)
    tooclose = [bool(tooclosefunc(assembly, sa.nbrs, **kw)) for sa in assoc]
    return np.array(tooclose, dtype=bool)

def slide_axis(
    axis,
    assembly,
    nbrs="auto",
    tooclosefunc=None,
    perp=False,
    step=1.0,
    maxstep=100,
    showme=False,
    # symelemsibs=None,
    resetonfail=True,
    scaleslides=1.0,
    boundscheck=lambda x: True,
    nobadsteps=False,
    timer=None,
    **kw,
):
    axis = ipd.homog.hnormalized(axis)
    origpos = assembly.asym.position
    if nbrs == "auto":
        nbrs = assembly.get_neighbors_by_axismatch(axis, perp)
    elif nbrs == "all":
        nbrs = None
    # ic(repr(nbrs))

    # origisect = tooclose_by_symelem(assembly, tooclosefunc)
    # if perp and (np.all(origisect) or np.all(~origisect)):
    #    # either all overlapping or none, don't do perp axis move
    #    # better to scale
    #    return

    iflip, flip = 0, -1.0
    startsclose = tooclosefunc(assembly, nbrs, **kw)  # type: ignore
    if startsclose:
        iflip, flip = -1, 1.0

    clashnbrs = None
    if False:  # perp:
        # collect neighbors to "clash" check -- all not this point sym
        clashnbrs = []
        assoc = ipd.sym.xtal.symelem_associations(assembly.sym)
        ic(assoc)
        for i, sa in enumerate(assoc):
            isect = bool(tooclosefunc(assembly, sa.nbrs))
            axispos = ipd.hscaled(assembly.cellsize, sa.symelem.cen)
            pointtowards = ipd.homog.hdot(axis, axispos - assembly.asym.com()) > 0
            # ic(i, sa.symelem.nfold, axispos[0], isect, pointtowards)
            # ic(sa.symelem.axis)
            # ic(abs(ipd.homog.hdot(sa.symelem.axis, axis)))
            notperp = abs(ipd.homog.hdot(sa.symelem.axis, axis)) > 0.01
            if notperp or (isect and pointtowards):
                clashnbrs.extend(sa.nbrs)
                # ic(i, clashnbrs)
        ic(clashnbrs)
        nassoc = 1 + sum([len(sa.nbrs) for sa in assoc])
        for i in range(nassoc, len(assembly)):
            clashnbrs.append(i)

        assert not clashnbrs or clashnbrs[0] <= nassoc
        if not clashnbrs or clashnbrs[0] == nassoc and startsclose:
            assert 0
            # no neighbors to clash check
            return 0
    if timer:
        timer.checkpoint("slide_axis_loop")
    # ic(axis[0], flip, nbrs, clashnbrs)

    total = 0.0
    delta = ipd.homog.htrans(flip * step * axis)
    success = False
    lastclose = 1.0
    for i in range(maxstep):
        assembly.asym.moveby(delta)
        if not boundscheck(assembly):
            break
        total += flip * step
        if showme:
            ipd.showme(assembly, name="slideaxis%f" % axis[0], **kw)
        close = tooclosefunc(assembly, nbrs, **kw)  # type: ignore
        if iflip and nobadsteps and close - lastclose > 0.01:
            break
        lastclose = close
        if iflip + bool(close):
            success = True
            # assert 0
            break

        if clashnbrs is not None and tooclosefunc(assembly, clashnbrs):  # type: ignore
            assert 0
            break
    # ic('slideaxis', perp, success)
    if not success and resetonfail:
        assembly.asym.position = origpos
        if showme:
            ipd.showme(assembly, name="resetaxis%f" % axis[0], **kw)
        if timer:
            timer.checkpoint("slide_axis_end")
        return 0

    if iflip == 0:  # back off so no clash
        total -= flip * step
        assembly.asym.moveby(ipd.homog.hinv(delta))
        if showme:
            ipd.showme(assembly, name="backoffaxis%f" % axis[0], **kw)

    if iflip == 0 and abs(total) > 0.01 and scaleslides != 1.0:
        # if slid into contact, apply scaleslides (-1) is to undo slide
        # ic(total, total + (scaleslides - 1) * total)
        scaleslides_delta = ipd.homog.htrans((scaleslides-1) * total * axis)
        total += (scaleslides-1) * total

        assembly.asym.moveby(scaleslides_delta)
    if timer:
        timer.checkpoint("slide_axis_end")

    return total

def slide_scale(*a, **kw):
    return slide_cellsize(*a, scalecoords=True, **kw)

def slide_cellsize(
    assembly,
    cellsize,
    symelems=None,
    tooclosefunc=None,  # tooclose_clash,
    step=1.0,
    maxstep=100,
    showme=False,
    cellscalelimit=9e9,
    resetonfail=True,
    scaleslides=1.0,
    boundscheck=lambda x: True,
    scalecoords=None,
    nobadsteps=False,
    ignoreimmobile=True,
    mincellsize=0.1,
    maxcellsize=9e9,
    timer=None,
    **kw,
):
    if showme:
        ipd.showme(assembly, name="scaleinput", **kw)
    orig_scalecoords = assembly.scale_com_with_cellsize
    nbrs = None
    if scalecoords is not None:
        assembly.scale_com_with_cellsize = scalecoords
    elif ignoreimmobile and assembly.sym is not None:
        nbrs = []
        assoc = ipd.sym.xtal.symelem_associations(assembly.sym)
        # ic(assoc)
        for sa in assoc:
            if sa.symelem.mobile:
                nbrs.extend(sa.nbrs)
        nassoc = 1 + sum([len(sa.nbrs) for sa in assoc])
        for i in range(nassoc, len(assembly)):
            nbrs.append(i)
        # ic(nbrs)
        # assert 0

    step = ipd.to_xyz(step)
    # ic(cellsize, assembly.cellsize)

    if not np.allclose(cellsize, assembly.cellsize):
        raise ValueError(f"cellsize {cellsize} doesn't match assembly.cellsize {assembly.cellsize}")
    orig_cellsize = assembly.cellsize
    cellsize = assembly.cellsize

    iflip, flip = 0, -1.0
    startsclose = bool(tooclosefunc(assembly, nbrs, **kw))  # type: ignore
    if startsclose:
        # ic('scale cell tooclose')
        iflip, flip = -1, 1.0

    initpos = assembly.asym.position.copy()
    success, lastclose = False, 1.0
    for i in range(maxstep):
        close = tooclosefunc(assembly, nbrs, **kw)  # type: ignore
        # ic('SLIDE CELLSIZE', i, bool(close), startsclose, nbrs)
        if iflip + bool(close):
            # print(f'{i} {close}', flush=True)
            success = True
            break
        if iflip and nobadsteps and close - lastclose > 0.01:
            # assert 0
            break
        lastclose = close
        # ic(cellsize, flip, step)
        delta = (cellsize + flip*step) / cellsize
        # ic(delta)
        # ic(cellsize, flip * step)

        if np.min(np.abs(delta)) < 0.0001:
            print(f"bad delta {delta} cellsize {cellsize} flip {flip} step {step}", flush=True)
            success = False
            # assert 0
            break
        cellsize *= delta
        if any(cellsize < mincellsize) or any(cellsize > maxcellsize):
            success = False
            cellsize /= delta
            # assert 0
            break
        assembly.scale_frames(delta, safe=False)
        # changed = assembly.scale_frames(delta, safe=False)
        # if not changed:
        # assert i == 0
        # return cellsize
        if not boundscheck(assembly):
            break
        if showme:
            ipd.showme(assembly, name="scale", **kw)
        if np.all(cellsize / orig_cellsize > cellscalelimit):
            # assert 0
            success = True
            break
    # assert 0

    if not success:
        if resetonfail:
            assembly.scale_frames(orig_cellsize / cellsize)
        if showme:
            ipd.showme(assembly, name="resetscale", **kw)
        assembly.scale_com_with_cellsize = orig_scalecoords

        return orig_cellsize

    if iflip == 0 and success:  # back off
        delta = 1.0 / delta  # type: ignore
        assembly.scale_frames(delta, safe=False)
        cellsize *= delta

        if showme:
            ipd.showme(assembly, name="backoffscale", **kw)

    if iflip == 0 and scaleslides != 1.0 and np.sum((cellsize - orig_cellsize)**2) > 0.001:
        # if slid into contact, apply scaleslides (-1) is to undo slide

        newcellsize = (cellsize-orig_cellsize) * (scaleslides) + orig_cellsize
        newdelta = newcellsize / cellsize
        # ic(cellsize, newcellsize)
        cellsize = newcellsize

        assembly.scale_frames(newdelta, safe=False)

    assembly.scale_com_with_cellsize = orig_scalecoords
    # ic(cellsize)
    return cellsize

################################# NO ########################

def recenter_asu_frames(
    assembly,
    symelemsibs=None,
    method=None,
    axis=None,
    showme=False,
    resetonfail=True,
    **kw,
):
    """Symelemsibs is ???"""

    assert 0

    if symelemsibs is None:
        assert method == "to_center"
        assert axis is not None
        newcen = len(axis) * assembly.asym.com()
        for b in assembly.symbodies:
            newcen += b.com()
        newcen /= len(assembly) + len(axis) - 1
        assembly.asym.setcom(newcen)
        if showme:
            ipd.showme(assembly, name="recenasuabs", **kw)
        return

    origcom = assembly.asym.com()
    partnercom = assembly.asym.com()
    othercom = assembly.asym.com()
    for p in symelemsibs:
        partnercom += assembly.bodies[p].com()
    partnercom /= len(symelemsibs) + 1
    othercom = assembly.asym.com()
    for i in range(1, len(assembly)):
        if i not in symelemsibs:
            othercom += assembly.bodies[i].com()
    othercom /= len(assembly) - len(symelemsibs)
    comdir = ipd.homog.hnormalized(othercom - partnercom)
    halfdist = ipd.homog.hnorm(othercom - partnercom) / 2
    center = (partnercom+othercom) / 2
    # ipd.showme(origcom, name='com')
    # ipd.showme(partnercom, name='partnercom')
    # ipd.showme(othercom, name='othercom')
    # ipd.showme(center, name='center')

    if method == "to_center":
        newcen = center
    else:
        if resetonfail:
            if method == "toward_other":
                axis = ipd.homog.hnormalized(othercom - partnercom)
                dist = ipd.homog.hdot(axis, origcom - partnercom)
                dist = halfdist - dist
                newcen = origcom + axis*dist
            elif method == "toward_partner":
                axis = ipd.homog.hnormalized(axis)
                dist = halfdist / ipd.homog.hdot(axis, comdir)
                proj = axis * dist
                newcen = partnercom + proj
                # ipd.showme(axis, name='axis')
                # ipd.showme(ipd.hproj(axis, partnercom - othercom))
                # ipd.showme(proj, name='proj')
            else:
                raise ValueError(f'bad method "{method}"')

    # ipd.showme(newcen, name='newcen')
    pos = assembly.asym.setcom(newcen)  # type: ignore
    if showme:
        ipd.showme(assembly, name="recenterasu", **kw)

    # assert 0
