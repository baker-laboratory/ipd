import numpy as np
import pytest

import ipd

# ic.configureOutput(includeContext=True)

def main():
    _gen_canonical_asu_cens()
    test_canonical_asu_center()
    test_asufit_I4132(showme=True)
    # test_asufit_P213(showme=True)
    # test_asufit_L6m322(showme=True)
    # test_asufit_L6_32(showme=True)
    # test_asufit_oct(showme=True)
    # test_asufit_icos(showme=True)
    ic("TEST asufit DONE")  # type: ignore

def _gen_canonical_asu_cens():
    for sym in 'oct'.split():
        for nnbr in (1, 2, 3, 4, 5, 6):
            # for sym in 'c2 c3 c4 c5 c6 c7 c8 c9 d2 d3 d4 d5 d6 tet oct icos'.split():
            cen = ipd.sym.compute_canonical_asucen(sym, neighbors=nnbr)
            print(f"    {sym}={repr(cen)},")
            ipd.showme(ipd.homog.hxform(ipd.sym.frames(sym), cen), kind="point", name=f'{sym}_{nnbr}')
    assert 0

@pytest.mark.fast
def test_canonical_asu_center():
    ic(ipd.sym.canonical_asu_center('C3'))  # type: ignore
    assert np.allclose(ipd.sym.canonical_asu_center('C3'), [1, 0, 0, 1])

@pytest.mark.xfail()
def test_asufit_oct(showme=False):
    sym = "oct"
    fname = "/home/sheffler/src/ipd/blob4h.pdb"
    pdb = ipd.pdb.readpdb(fname)
    pdb = pdb.subset(atomnames=["CA"], chains=["A"])
    xyz = np.stack([pdb.df["x"], pdb.df["y"], pdb.df["z"]]).T
    xyz[:, :3] -= ipd.homog.hcom(xyz)[:3]
    xyz_contact = None

    cendis = np.argsort(ipd.homog.hnorm(xyz - ipd.homog.hcom(xyz)[:3])**2)
    w = cendis[:int(len(xyz) * 0.5)]
    xyz_contact = xyz[w]
    # ipd.showme(xyz)
    # ipd.showme(xyz_contact)
    # assert 0

    ax2 = ipd.sym.axes(sym)[2]
    ax3 = ipd.sym.axes(sym)[3]
    # xyz = point_cloud(100, std=10, outliers=0)
    # xyz[:, :3] += 60 * (ipd.sym.axes(sym)[2] + ipd.sym.axes(sym)[3])[:3]
    # ipd.showme(xyz)
    primary_frames = [np.eye(4), ipd.homog.hrot(ax2, 180), ipd.homog.hrot(ax3, 120)]  # , ipd.homog.hrot(ax3, 240)]
    frames = ipd.sym.frames(sym, ontop=primary_frames)
    lever = ipd.hrog(xyz) * 1.5
    """"""
    with ipd.dev.Timer():
        ic("symfit")  # type: ignore
        # np.random.seed(7)
        mc = ipd.sym.asufit(
            sym,
            xyz,
            xyz_contact,
            symaxes=[ax3, ax2],
            frames=frames,
            showme=showme,
            showme_accepts=showme,
            vizfresh=True,
            headless=False,
            contactfrac=0.3,
            contactdist=10,
            clashdist=4,
            clashpenalty=0.1,
            cartsd=1.0,
            temperature=1.0,
            resetinterval=100,
            correctionfactor=1.5,
            iterations=1000,
            driftpenalty=0.0,
            anglepenalty=0.5,
            thresh=0.0,
            spreadpenalty=0.1,
            biasradial=4,
            usebvh=True,
            vizsphereradius=3,
            scoreframes=[(0, 1), (0, 2)],
            clashframes=[(1, 2), (1, 3), (2, 3)],
            lever=9e9,
        )
    assert np.allclose(
        mc.beststate.position,
        np.array([
            [9.63284623e-01, 2.49202698e-01, -9.99037016e-02, -1.32625492e01],
            [-2.65707884e-01, 9.38228112e-01, -2.21646860e-01, 5.46007841e01],
            [3.84974658e-02, 2.40054213e-01, 9.69995835e-01, 1.16772927e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]),
    )

@pytest.mark.xfail()
def test_asufit_I4132(showme=False):
    sym = "I4132_322"
    xtal = ipd.sym.xtal.Xtal(sym)

    scale = 140
    asucen = xtal.asucen(cellsize=scale)
    # np.random.seed(2)
    asucen = xtal.asucen(cellsize=scale)
    xyz = point_cloud(100, std=20, outliers=40)  # type: ignore
    xyz += ipd.homog.hvec(asucen)
    xyz[:, 0] += +0.000 * scale
    xyz[:, 1] += -0.030 * scale
    xyz[:, 2] += -0.020 * scale
    xyz = xyz[:, :3]

    # fname = '/home/sheffler/src/ipd/blob5h.pdb'
    # pdb = ipd.pdb.readpdb(fname)
    # # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
    # xyz, mask = pdb.coords()
    # xyz = xyz[:, :4].reshape(-1, 3)
    # xyz[:, :3] -= ipd.homog.hcom(xyz)[:3]
    # xyz[:, :3] += asucen[:3]

    cendis = np.argsort(ipd.homog.hnorm(xyz - ipd.homog.hcom(xyz)[:3])**2)
    w = cendis[:int(len(xyz) * 0.8)]
    xyz_contact = xyz[w]

    primary_frames = np.stack([
        np.eye(4),
        xtal.symelems[0].operators[1],
        xtal.symelems[1].operators[1],
        xtal.symelems[2].operators[1],
        xtal.symelems[0].operators[2],
    ])
    primary_frames = ipd.hscaled(scale, primary_frames)
    # ipd.showme(ipd.homog.hxform(primary_frames[0], xyz))
    # ipd.showme(ipd.homog.hxform(primary_frames[1], xyz))
    # ipd.showme(ipd.homog.hxform(primary_frames[2], xyz))
    # ipd.showme(ipd.homog.hxform(primary_frames[3], xyz))
    # ipd.showme(ipd.homog.hxform(primary_frames[4], xyz))
    # ipd.showme(xtal.symelems, scale=scale, symelemscale=0.5, name='cenelems')
    # assert 0
    # frames = ipd.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=scale, center=asucen, asucen=asucen,
    # radius=scale * 3)
    # ic(scale)
    frames = ipd.sym.frames(
        sym,
        ontop=primary_frames,
        cells=(-1, 1),
        cellsize=scale,
        center=asucen,
        asucen=asucen,
        radius=scale * 0.5,
    )
    # ic(frames.shape)
    # ic(frames)
    # ipd.showme(primary_frames, scale=1)
    # ipd.showme(frames)
    # assert 0

    if 0:
        # cenelems = xtal.central_symelems(target=[-0.1, -0.05, 0.1])
        # ic(xtal.symelems)
        # ic(cenelems)
        # assert 0
        # ipd.showme(cenelems, scale=scale, symelemscale=0.5, name='cenelems')

        ipd.showme(xtal.symelems, scale=scale, symelemscale=2)

        ipd.showme(ipd.homog.hxform(primary_frames[0], xyz), sphere=10, col=(1, 1, 1))
        ipd.showme(ipd.homog.hxform(primary_frames[1], xyz), sphere=10, col=(1, 0, 0))
        ipd.showme(ipd.homog.hxform(primary_frames[2], xyz), sphere=10, col=(0, 1, 0))
        ipd.showme(ipd.homog.hxform(primary_frames[3], xyz), sphere=10, col=(0, 0, 1))
        ipd.showme(ipd.homog.hxform(primary_frames[4], xyz), sphere=10, col=(1, 1, 0))

        from ipd.tests.sym.test_xtal import test_hxtal_viz  # type: ignore
        test_hxtal_viz(
            spacegroup="I4132_322",
            headless=False,
            # showpoints=ipd.homog.hcom(xyz),
            cells=(-1, 0),
            symelemscale=0.3,
            fansize=np.array([1.7, 1.2, 0.7]) / 3,
            fancover=10,
            symelemtwosided=True,
            showsymelems=True,
            scale=scale,
            pointradius=17,
        )

        # frames = ipd.sym.frames(sym, ontop=primary_frames, cells=3, cellsize=scale, center=asucen, radius=scale * 2)
        # ic(frames.shape)
        # assert 0
        # lever = ipd.hrog(xyz) * 1.5
        # assert 0

        # ipd.showme(
        #    list(ipd.homog.hxform(frames, xyz)),
        #    sphere=10,
        #    name='framepts',
        #    topcolors=[(0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)],
        #    chainbow=True,
        # )
        # assert 0

    # asucen = ipd.homog.hcom(xyz) / scale
    # xyz[:, :3] += scale

    frames = ipd.hscaled(1 / scale, frames)
    lever = ipd.hrog(xyz) * 1.5
    with ipd.dev.Timer():
        ic("symfit")  # type: ignore
        # np.random.seed(14)
        mc = ipd.sym.asufit(
            sym,
            xyz,
            xyz_contact,
            # symaxes=[3, 2],
            frames=frames,
            dumppdb=False,
            dumppdbscale=1,
            showme=showme,
            showme_accepts=showme,
            vizfresh=True,
            headless=False,
            spacegroup="I 41 3 2",
            # png='I4132_322',
            contactfrac=0.1,
            contactdist=10,
            clashdist=5,
            clashpenalty=10.1,
            cartsd=0.8,
            lever=1000,
            temperature=0.5,
            resetinterval=10000,
            correctionfactor=1.5,
            iterations=1000,
            nresatom=4,
            driftpenalty=0.2,
            anglepenalty=0.5,
            thresh=0.0,
            spreadpenalty=0.1,
            biasradial=1,
            usebvh=True,
            vizsphereradius=2,
            scale=scale,
            scalesd=1,
            scoreframes=[(0, 1), (0, 2), (0, 3)],
            clashframes=[(1, 2), (1, 3), (2, 3)],
            # topcolors=[(1, 1, 1)] + [(.9, .9, .9)] * 5),
            topcolors=[(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)],
            chainbow=False,
        )

    assert np.allclose(
        mc.beststate.position,
        np.array([
            [9.63284623e-01, 2.49202698e-01, -9.99037016e-02, -1.32625492e01],
            [-2.65707884e-01, 9.38228112e-01, -2.21646860e-01, 5.46007841e01],
            [3.84974658e-02, 2.40054213e-01, 9.69995835e-01, 1.16772927e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]),
    )

@pytest.mark.xfail()
def test_asufit_P213(showme=False):
    sym = "P 21 3"
    xtal = ipd.sym.xtal.Xtal(sym)
    scale = 100

    # xyz = point_cloud(100, std=30, outliers=20)
    asucen = xtal.asucen(use_olig_nbrs=True, cellsize=scale)
    # xyz += ipd.homog.hvec(asucen)
    # xyz[:, 2] += 30

    fname = "/home/sheffler/src/ipd/blob5h.pdb"
    pdb = ipd.pdb.readpdb(fname)
    # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
    xyz, mask = pdb.coords()
    xyz = xyz[:, :4].reshape(-1, 3)
    # ic(xyz.shape)
    # assert 0
    xyz[:, :3] -= ipd.homog.hcom(xyz)[:3]
    xyz[:, :3] += asucen[:3]
    cendis = np.argsort(ipd.homog.hnorm(xyz - ipd.homog.hcom(xyz)[:3])**2)
    w = cendis[:int(len(xyz) * 0.6)]
    xyz_contact = xyz[w]

    primary_frames = np.stack([
        ipd.hscaled(scale, np.eye(4)),
        xtal.symelems[0].operators[1],
        xtal.symelems[1].operators[1],
        xtal.symelems[0].operators[2],
        xtal.symelems[1].operators[2],
    ])
    primary_frames = ipd.hscaled(scale, primary_frames)
    # ic(xtal.symelems[0].cen)
    # ic(xtal.symelems[1].cen)
    # ic(asucen)

    # frames = ipd.sym.frames(sym, ontop=primary_frames, cells=[(-1, 0), (-1, 0), (-1, 0)])
    # frames = ipd.sym.frames(sym, ontop=primary_frames, cells=3, center=
    frames = ipd.sym.frames(
        sym,
        ontop=primary_frames,
        cells=(-1, 1),
        cellsize=scale,
        center=asucen,
        asucen=asucen,
        radius=scale * 0.8,
    )
    # ic(frames.shape)
    # ipd.showme(xtal.symelems, scale=scale, symelemscale=2)
    # ipd.showme(ipd.homog.hxform(primary_frames[0], xyz), sphere=10, col=(1, 1, 1))
    # ipd.showme(ipd.homog.hxform(primary_frames[1], xyz), sphere=10, col=(1, 0, 0))
    # ipd.showme(ipd.homog.hxform(primary_frames[2], xyz), sphere=10, col=(0, 1, 0))
    # ipd.showme(ipd.homog.hxform(primary_frames[3], xyz), sphere=10, col=(0, 0, 1))
    # ipd.showme(ipd.homog.hxform(primary_frames[4], xyz), sphere=10, col=(1, 1, 0))
    # ipd.showme(
    # list(ipd.homog.hxform(frames, xyz)),
    # sphere=10,
    # name='aoiresnt',
    # topcolors=[(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)],
    # )
    # assert 0

    frames = ipd.hscaled(1 / scale, frames)
    lever = ipd.hrog(xyz) * 1.5
    """"""
    with ipd.dev.Timer():
        ic("symfit")  # type: ignore
        # np.random.seed(7)
        mc = ipd.sym.asufit(
            sym,
            xyz,
            xyz_contact,
            spacegroup="P 21 3",
            nresatom=4,
            # symaxes=[3, 2],
            frames=frames,
            showme=showme,
            showme_accepts=showme,
            vizfresh=True,
            headless=False,
            contactfrac=0.1,
            contactdist=12,
            clashdist=6,
            clashpenalty=10.1,
            cartsd=1,
            temperature=0.5,
            resetinterval=200,
            correctionfactor=1.5,
            iterations=1000,
            driftpenalty=0.0,
            anglepenalty=0.5,
            thresh=0.0,
            spreadpenalty=0.1,
            biasradial=1,
            usebvh=True,
            vizsphereradius=2,
            scale=scale,
            scalesd=4,
            scoreframes=[(0, 1), (0, 2)],
            clashframes=[(1, 2), (1, 3), (2, 3)],
            # topcolors=[(1, 1, 1)] + [(.9, .9, .9)] * 5),
            topcolors=[(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)],
            # chainbow=True,
        )
    assert np.allclose(
        mc.beststate.position,
        np.array([
            [9.63284623e-01, 2.49202698e-01, -9.99037016e-02, -1.32625492e01],
            [-2.65707884e-01, 9.38228112e-01, -2.21646860e-01, 5.46007841e01],
            [3.84974658e-02, 2.40054213e-01, 9.69995835e-01, 1.16772927e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]),
    )

@pytest.mark.xfail()
def test_asufit_L6m322(showme=False):
    sym = "L6m_322"
    xtal = ipd.sym.xtal.Xtal(sym)
    scale = 70
    # xyz = point_cloud(100, std=30, outliers=20)
    # xyz += ipd.homog.hvec(xtal.asucen(cellsize=scale))
    # xyz[:, 2] += 20

    fname = "/home/sheffler/src/ipd/blob6h.pdb"
    pdb = ipd.pdb.readpdb(fname)
    # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
    xyz, mask = pdb.coords()
    xyz = xyz[:, :4].reshape(-1, 3)
    xyz[:, :3] -= ipd.homog.hcom(xyz)[:3]
    xyz[:, 0] += scale * 0.3
    xyz[:, 1] += scale * 0.1
    xyz[:, 2] += 18
    ss = np.array(list(ipd.pdb.dssp(xyz.reshape(-1, 4, 3))))
    xyz_contact = xyz.reshape(-1, 4, 3)[ss == "H"].reshape(-1, 3)
    ic(xyz_contact.shape)  # type: ignore
    primary_frames = np.stack([
        np.eye(4),
        xtal.symelems[0].operators[1],
        xtal.symelems[1].operators[1],
        xtal.symelems[2].operators[1],
    ])
    frames = ipd.sym.frames(sym, ontop=primary_frames)
    lever = ipd.hrog(xyz) * 1.5
    """"""
    for i in range(1):
        with ipd.dev.Timer():
            ic("symfit")  # type: ignore
            # np.random.seed(7)
            mc = ipd.sym.asufit(
                sym,
                xyz,
                xyz_contact,
                # symaxes=[3, 2],
                dumppdb=f"P6m32_{i:04}.pdb",
                frames=frames,
                showme=showme,
                showme_accepts=showme,
                vizfresh=True,
                headless=False,
                contactfrac=0.1,
                contactdist=12,
                clashdist=5,
                clashpenalty=10.1,
                cartsd=2,
                temperature=0.8,
                resetinterval=20000,
                correctionfactor=1.5,
                iterations=1000,
                driftpenalty=0.0,
                anglepenalty=0.5,
                thresh=0.0,
                spreadpenalty=0.1,
                biasradial=1,
                usebvh=True,
                vizsphereradius=2,
                scale=scale,
                scalesd=4,
                scoreframes=[(0, 1), (0, 2), (0, 3)],
                clashframes=[(1, 2), (1, 3), (2, 3)],
            )
    assert np.allclose(
        mc.beststate.position,  # type: ignore
        np.array([
            [9.63284623e-01, 2.49202698e-01, -9.99037016e-02, -1.32625492e01],
            [-2.65707884e-01, 9.38228112e-01, -2.21646860e-01, 5.46007841e01],
            [3.84974658e-02, 2.40054213e-01, 9.69995835e-01, 1.16772927e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]),
    )

@pytest.mark.xfail()
def test_asufit_L6_32(showme=False):
    sym = "L6_32"
    xtal = ipd.sym.xtal.Xtal(sym)
    scale = 100

    xyz = point_cloud(100, std=10, outliers=20)  # type: ignore
    xyz += ipd.homog.hvec(xtal.asucen()) * scale

    primary_frames = np.stack([np.eye(4), xtal.symelems[0].operators[1], xtal.symelems[1].operators[1]])
    frames = ipd.sym.frames(sym, ontop=primary_frames)
    lever = ipd.hrog(xyz) * 1.5

    ipd.showme(primary_frames)
    ipd.showme(frames)

    with ipd.dev.Timer():
        ic("symfit")  # type: ignore
        # np.random.seed(7)
        mc = ipd.sym.asufit(
            sym,
            xyz,
            # symaxes=[3, 2],
            frames=frames,
            showme=showme,
            showme_accepts=showme,
            vizfresh=True,
            headless=False,
            contactfrac=0.3,
            contactdist=16,
            cartsd=2,
            temperature=0.5,
            resetinterval=200,
            correctionfactor=1.5,
            iterations=1000,
            driftpenalty=0.0,
            anglepenalty=0.5,
            thresh=0.0,
            spreadpenalty=0.1,
            biasradial=1,
            usebvh=True,
            vizsphereradius=6,
            scale=scale,
            scalesd=4,
            scoreframes=[(0, 1), (0, 2)],
            clashframes=[(1, 2), (1, 3), (2, 3)],
        )
    assert np.allclose(
        mc.beststate.position,
        np.array([
            [9.63284623e-01, 2.49202698e-01, -9.99037016e-02, -1.32625492e01],
            [-2.65707884e-01, 9.38228112e-01, -2.21646860e-01, 5.46007841e01],
            [3.84974658e-02, 2.40054213e-01, 9.69995835e-01, 1.16772927e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]),
    )

@pytest.mark.xfail()
def test_asufit_icos(showme=False):
    sym = "icos"
    # fname = ipd.tests.testdata.test_data_path('pdb/x012.pdb')
    # pdb = ipd.pdb.readpdb(fname)
    # pdb = pdb.subset(atomnames=['CA'], chains=['A'])
    # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
    # xyz[:, :3] += ipd.homog.hcom(xyz)[:3]
    xyz = point_cloud(100, std=10, outliers=10)  # type: ignore
    xyz[:, :3] += 140 * ipd.homog.hnormalized(ipd.sym.axes(sym)[2] * 4 + ipd.sym.axes(sym)[3])[:3]
    ax2 = ipd.sym.axes("icos")[2]
    ax3 = ipd.sym.axes("icos")[3]
    primary_frames = [np.eye(4), ipd.homog.hrot(ax2, 180), ipd.homog.hrot(ax3, 120)]  # , ipd.homog.hrot(ax3, 240)]
    frames = ipd.sym.frames(sym, ontop=primary_frames)

    lever = ipd.hrog(xyz) * 1.5
    with ipd.dev.Timer():
        ic("symfit")  # type: ignore
        # np.random.seed(7)
        mc = ipd.sym.asufit(
            sym,
            xyz,
            symaxes=[ax3, ax2],
            frames=frames,
            showme=showme,
            showme_accepts=showme,
            vizfresh=True,
            contactfrac=0.2,
            contactdist=12,
            cartsd=1,
            temperature=1,
            resetinterval=100,
            correctionfactor=1.5,
            iterations=1000,
            vizsphereradius=10,
            driftpenalty=0.1,
            anglepenalty=0.1,
            thresh=0.0,
            spreadpenalty=0.1,
            biasradial=4,
            usebvh=True,
            scoreframes=[(0, 1), (0, 2)],
            clashframes=[(1, 2), (1, 3), (2, 3)],
        )
        ref = np.array([
            [0.99880172, 0.01937787, 0.04494025, -3.38724054],
            [-0.02529149, 0.99052266, 0.13500072, 1.91723184],
            [-0.04189831, -0.13597556, 0.98982583, 2.10409862],
            [0.0, 0.0, 0.0, 1.0],
        ])
        assert np.allclose(ref, mc.beststate.position)
        ic("test_asufit_icos PASS!")  # type: ignore

if __name__ == "__main__":
    main()
