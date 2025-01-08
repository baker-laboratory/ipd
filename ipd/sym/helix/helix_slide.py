import functools

import numpy as np

import ipd

def helix_slide(
    helix,
    coords,
    cellsize,
    coils=2,
    contactfrac=0,
    step=10,
    maxstep=10,
    iters=4,
    breathe=[2.5, 1, 0.5, 0],
    showme=False,
    closest=9,
    scalefirst=False,
    **kw,
):
    assert np.allclose(cellsize[0], cellsize[1])
    cellsize = cellsize.copy()
    coords = coords.astype(np.float64)

    hframes = helix.frames(xtalrad=9e9, radius=cellsize[0], spacing=cellsize[2], coils=coils, closest=closest)
    assembly = ipd.dock.rigid.RigidBodyFollowers(  # type: ignore
        coords=coords,  # type: ignore
        frames=hframes,
        symtype="H",
        cellsize=cellsize,
        clashdis=8,
        contactdis=16)
    # if showme: assembly.dump_pdb(f'helix_slide____.pdb')
    hstep = np.array([0.00, 0.00, step])
    rstep = np.array([step, step, 0.00])
    sstep = np.array([step, step, step])
    tooclose = functools.partial(ipd.dock.rigid.tooclose_overlap, contactfrac=contactfrac)  # type: ignore
    # steps = [hstep, rstep]
    steps = [rstep, hstep]
    if scalefirst:
        steps = [sstep] + steps
    for i, expand in enumerate(breathe):
        for j in range(iters):
            for step in steps:
                scale = 1 + step / np.mean(assembly.cellsize) * expand

                cellsize = ipd.sym.slide_cellsize(  # type: ignore
                    assembly,
                    cellsize=cellsize,
                    step=step,
                    tooclosefunc=tooclose,
                    showme=showme,
                    maxstep=maxstep,
                    moveasymunit=False,
                    **kw,
                )

                if expand > 0:
                    cellsize = assembly.scale_frames(scale)
                if showme:
                    ipd.showme(assembly, **kw)

                # assembly.dump_pdb(f'helix_slide_{i}_{j}.pdb')
    return assembly
