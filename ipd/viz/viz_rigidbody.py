import numpy as np
from willutil.viz.pymol_viz import pymol_load, cgo_sphere
import willutil as wu
from willutil.rigid.rigidbody import RigidBody, RigidBodyFollowers


@pymol_load.register(RigidBodyFollowers)
def pymol_viz_RigidBodyFollowers(bodies, name="RigidBodyFollowers", state=None, addtocgo=None, **kw):
    wu.showme(bodies.bodies, name=name, topcolors=[(1, 1, 1)], **kw)
    return

    import pymol

    kw = wu.Bunch(kw)
    v = pymol.cmd.get_view()
    state["seenit"][name] += 1
    cgo = list()
    # col = get_different_colors
    pymol_viz_RigidBody(bodies.asym, state, name, addtocgo=cgo, **kw)
    for body in bodies.symbodies:
        pymol_viz_RigidBody(body, state, name, addtocgo=cgo, **kw)

    if addtocgo is None:
        pymol.cmd.load_cgo(cgo, f'{name}_{state["seenit"][name]}')
        pymol.cmd.set_view(v)
    else:
        addtocgo.extend(cgo)


@pymol_load.register(RigidBody)
def pymol_viz_RigidBody(
    body,
    name="rigidbody",
    state=None,
    addtocgo=None,
    showpairswith=None,
    showpairsdist=8,
    showcontactswith=None,
    vizsphereradius=2,
    col=(1, 1, 1),
    **kw,
):
    kw = wu.Bunch(kw)
    import pymol

    v = pymol.cmd.get_view()
    state["seenit"][name] += 1

    assert 0 == np.sum(np.isnan(body.coords))

    cgo = list()
    wu.showme(body.coords, addtocgo=cgo, sphere=vizsphereradius, col=col, **kw)

    if showcontactswith is not None:
        # ic(body.point_contact_count(showcontactswith, contactdist=showpairsdist))
        pairs = body.interactions(showcontactswith, contactdist=showpairsdist)
        # ic(len(set(pairs[:, 0])))
        # ic(len(set(pairs[:, 1])))

        crds = body.coords
        for i in set(pairs[:, 0]):
            cgo += cgo_sphere(crds[i], showpairsdist / 2, kw.col)

    if showpairswith is not None:
        assert 0
        crds1 = body.coords
        crds2 = showpairswith.coords
        pairs = body.interactions(showpairswith, contactdist=showpairsdist)
        for i, j in pairs:
            cgo += wu.viz.cgo_lineabs(crds1[i], crds2[j])

    if addtocgo is None:
        pymol.cmd.load_cgo(cgo, f'{name}_{state["seenit"][name]}')
        pymol.cmd.set_view(v)
    else:
        addtocgo.extend(cgo)
