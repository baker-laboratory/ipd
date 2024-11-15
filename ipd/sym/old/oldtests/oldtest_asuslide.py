import functools as ft

import numpy as np
import pytest
from numpy import array

import ipd
from ipd.sym.asuslide import asuslide  # type: ignore

pytest.skip(allow_module_level=True)

# ic.configureOutput(includeContext=True, contextAbsPath=True)

def main():
    # test_asuslide_L632()
    # assert 0

    manual_test()
    assert 0

    test_asuslide_p213()
    assert 0

    test_asuslide_L632()
    test_asuslide_L632_2()
    test_asuslide_L442()

    test_asuslide_F432()
    test_asuslide_I4132()
    test_asuslide_L632_ignoreimmobile()
    test_asuslide_P432_43()
    test_asuslide_P432_44()
    test_asuslide_I432()
    test_asuslide_I213()
    test_asuslide_p213()

    assert 0
    manual_test()
    assert 0

    test_asuslide_helix_case1()

    test_asuslide_case2()
    test_asuslide_helix_nfold1_2()
    test_asuslide_helix_nfold5()
    test_asuslide_helix_nfold3()
    test_asuslide_helix_nfold1()
    test_asuslide_I4132_clashframes()
    test_asuslide_F432()

    test_asuslide_oct()

    # test_asuslide_case2()
    # assert
    # test_asuslide_helix_nfold1_2()
    # test_asuslide_oct()
    # test_asuslide_L632_2(showme=True)
    # test_asuslide_P432_44(showme=True)
    # test_asuslide_P432_43(showme=True)
    assert 0

    test_asuslide_P432()  # type: ignore
    test_asuslide_P4132()  # type: ignore
    # asuslide_case3()

    # asuslide_case2()
    # asuslide_case1()
    # assert 0
    test_asuslide_L442()

    test_asuslide_I4132()

    test_asuslide_L632()

    test_asuslide_I213()

    ic("DONE")  # type: ignore

def manual_test():
    # yapf: disable
    kw = {'center': array([ 22.16297525,   9.5657125 , -24.6145983 ,   1.        ]), 'maxstep': 20, 'clashdis': 4.2, 'contactdis': 4.0, 'contactfrac': 0.014500000000000006, 'cellscalelimit': 1.5, 'strict': False}
    coords=np.array([[37.43444,16.600136,-36.849804],[34.127346,17.459654,-35.431736],[32.369286,15.066488,-37.71309],[34.490143,12.257697,-36.4415],[33.854485,13.158916,-32.85202],[30.14087,13.3336115,-33.421173],[30.158007,9.930897,-34.98829],[32.23455,8.527389,-32.194458],[29.95039,9.888692,-29.552744],[26.968216,8.4673815,-31.334763],[28.574718,5.0961127,-31.58422],[29.435669,5.1508093,-27.937641],[25.918219,6.01223,-26.96678],[24.533623,3.3079495,-29.147669],[26.920853,0.83473206,-27.691029],[26.083107,1.7542557,-24.1724],[22.423754,1.4482727,-24.85936],[22.800695,-1.9285806,-26.41853],[24.82718,-3.1842618,-23.54093],[22.429903,-1.9884397,-20.946342],[19.20533,-2.8035462,-22.704433],[18.457972,-5.5732207,-20.24991],[18.64516,-3.3609507,-17.291187],[15.429527,-1.5817798,-16.780115],[14.755223,0.86148727,-14.093442],[11.515039,2.637912,-13.728129],[13.028138,5.3479424,-11.628789],[15.5056925,6.254575,-14.264113],[12.756031,6.656026,-16.767748],[11.296263,9.68379,-15.092162],[14.608268,11.408663,-14.785076],[15.672325,10.704255,-18.307966],[12.374394,11.930729,-19.61201],[12.870153,15.178934,-17.799953],[16.291807,15.514817,-19.306164],[14.936408,14.857147,-22.750303],[12.49428,17.678102,-22.364908],[15.216918,20.020746,-21.301325],[17.48687,19.24535,-24.226528],[14.83651,19.628355,-26.848402],[17.264915,18.41478,-29.535057],[15.977524,15.582194,-31.580448],[19.0861,13.516757,-31.308044],[19.326181,14.126327,-27.635862],[15.729681,13.235167,-27.08639],[16.184309,10.021897,-28.96388],[19.206068,9.031187,-26.981613],[17.640123,9.982712,-23.699],[14.619852,7.861638,-24.431309],[16.93277,4.982046,-25.070734],[18.757181,5.5761847,-21.847363],[15.637167,5.614854,-19.739874],[15.938946,1.8690094,-19.367538],[19.45126,1.799174,-18.107029],[20.435474,2.3262575,-14.570786],[23.995745,1.1529319,-14.533105],[25.374428,3.3470116,-17.214006],[23.050564,6.2556853,-17.02632],[25.669886,8.84744,-16.293013],[28.01025,7.707162,-18.979683],[25.21269,7.596833,-21.463127],[24.125889,11.077479,-20.585724],[27.620346,12.378192,-21.008371],[27.911135,10.865114,-24.433369],[24.538767,12.190975,-25.415136],[25.384344,15.676883,-24.321796],[28.694895,15.590722,-26.093832],[27.010637,14.470794,-29.272808],[24.55835,17.29831,-28.983686],[27.293612,19.814505,-28.500967],[29.17836,18.490871,-31.461143],[26.098616,18.561434,-33.6082],[25.424032,22.13119,-32.7656],[28.950327,23.137213,-33.46722],[28.874958,21.414234,-36.795696]],dtype=np.float32)
    # yapf: enable
    frames = None
    foo = dict(
        contactdis=10,
        contactfrac=0.1,
        vizsphereradius=6,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
    )
    kw = ipd.dev.Bunch(kw)
    # ipd.contactdis = 10
    # ipd.contactfrac = 0.1
    slid = asuslide(
        sym="L6_32",
        coords=coords,
        frames=frames,
        axes=None,
        existing_olig=None,
        alongaxis=0,
        towardaxis=True,
        printme=False,
        cellsize=array([55.18552072, 55.18552072, 55.18552072]),
        isxtal=False,
        nbrs="auto",
        doscale=True,
        iters=3,
        subiters=1,
        clashiters=0,
        receniters=0,
        step=10,
        scalestep=None,
        closestfirst=True,
        centerasu="toward_other",
        centerasu_at_start=False,
        showme=True,
        scaleslides=1.0,
        iterstepscale=0.5,
        coords_to_asucen=False,
        xtalrad=0.0,
        **kw,
    )
    ipd.showme(slid)
    # slid.dump_pdb(f'/home/sheffler/DEBUG_slid_ipd.pdb')

@pytest.mark.fast
def test_asuslide_L632_2(showme=False):
    sym = "L6_32"
    xtal = ipd.sym.xtal.Xtal(sym)
    csize = 160
    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)
    asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
    xyz += ipd.homog.hvec(asucen)
    xyz[:, 1] -= 2

    primary_frames = np.stack([
        ipd.hscaled(csize, np.eye(4)),
        xtal.symelems[0].operators[1],
        xtal.symelems[0].operators[2],
        xtal.symelems[1].operators[1],
    ])
    primary_frames = ipd.hscaled(csize, primary_frames)
    frames = primary_frames

    slid = asuslide(
        sym,
        xyz,
        frames,
        showme=showme,
        maxstep=30,
        step=10,
        iters=5,
        clashiters=0,
        clashdis=8,
        contactdis=16,
        contactfrac=0.2,
        vizsphereradius=6,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
        scaleslides=1,
        resetonfail=True,
    )
    # ipd.showme(slid, vizsphereradius=6)
    ic(slid.cellsize)  # type: ignore
    ic(slid.asym.com())  # type: ignore
    assert np.allclose(slid.cellsize, 101.875)
    assert np.allclose(slid.asym.com(), [3.02441406e01, -1.27343750e00, 3.42139511e-16, 1.00000000e00])
    # assert np.allclose(slid.cellsize, 95)
    # assert np.allclose(slid.asym.com(), [25.1628825, -1.05965433, 0, 1])

@pytest.mark.fast
def test_asuslide_L632_ignoreimmobile(showme=False):
    sym = "L6_32"
    xtal = ipd.sym.xtal.Xtal(sym)
    csize = 160
    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)
    asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
    xyz += ipd.homog.hvec(asucen)
    xyz[:, 0] -= 30

    frames = xtal.cellframes(cellsize=csize)

    slid = asuslide(
        sym,
        xyz,
        frames,
        showme=False,
        maxstep=30,
        step=9,
        iters=3,
        subiters=1,
        doscale=True,
        doscaleiters=True,
        clashiters=0,
        clashdis=8,
        contactdis=12,
        contactfrac=0.1,
        vizsphereradius=6,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
        scaleslides=1,
        resetonfail=True,
        nobadsteps=False,
        ignoreimmobile=True,
        iterstepscale=0.5,
    )
    # ipd.showme(slid, vizsphereradius=6)
    ic(slid.cellsize)  # type: ignore
    ic(slid.asym.com())  # type: ignore
    assert np.allclose(slid.cellsize, 90.25)
    assert np.allclose(slid.asym.com(), [2.37973395e01, 1.44932026e-15, 7.30711512e-16, 1.00000000e00])

def asuslide_case4():
    sym = "P432"
    xtal = ipd.sym.xtal.Xtal(sym)
    # cellsize = 99.417
    cellsize = 76.38867528392643

    pdbfile = "/home/sheffler/project/diffusion/unbounded/preslide.pdb"
    pdb = ipd.pdb.readpdb(pdbfile).subset(chain="A")
    xyz = pdb.ca()
    fracremains = 1.0
    primaryframes = xtal.primary_frames(cellsize)
    cen = h.com(xyz.reshape(-1, xyz.shape[-1]))  # type: ignore
    frames = ipd.sym.frames(sym, ontop=primaryframes, cells=(-1, 1), cellsize=cellsize, center=cen, xtalrad=cellsize * 0.9)
    # frames = primaryframes
    cfracmin = 0.7
    cfracmax = 0.7
    cdistmin = 14.0
    cdistmax = 14.0
    t = 1
    slid = ipd.sym.asuslide(
        sym=sym,
        coords=xyz,
        frames=frames,
        # tooclosefunc=tooclose,
        cellsize=cellsize,
        maxstep=50,
        step=4,
        iters=4,
        subiters=4,
        clashiters=0,
        receniters=0,
        clashdis=4*t + 4,
        contactdis=14,
        contactfrac=0.1,
        cellscalelimit=1.5,
        # vizsphereradius=2,
        towardaxis=False,
        alongaxis=True,
        # vizfresh=False,
        # centerasu=None,
        centerasu="toward_other",
        # centerasu='closert',
        # centerasu_at_start=t > 0.8
        showme=False,
    )
    # ipd.showme(slid)

def asuslide_case3():
    sym = "P213_33"
    xtal = ipd.sym.xtal.Xtal(sym)
    # cellsize = 99.417
    cellsize = 115

    pdbfile = "/home/sheffler/project/diffusion/unbounded/preslide.pdb"
    pdb = ipd.pdb.readpdb(pdbfile).subset(chain="A")
    xyz = pdb.ca()
    fracremains = 1.0
    primaryframes = xtal.primary_frames(cellsize)
    # frames = ipd.sym.frames(sym, ontop=primaryframes, cells=(-1, 1), cellsize=cellsize, center=cen, xtalrad=cellsize * 0.5)
    frames = primaryframes
    cfracmin = 0.7
    cfracmax = 0.7
    cdistmin = 14.0
    cdistmax = 14.0
    t = 1
    slid = ipd.sym.asuslide(
        sym="P213_33",
        coords=xyz,
        frames=frames,
        # tooclosefunc=tooclose,
        cellsize=cellsize,
        maxstep=100,
        step=4*t + 2,
        iters=6,
        subiters=4,
        clashiters=0,
        receniters=0,
        clashdis=4*t + 4,
        contactdis=t * (cdistmax-cdistmin) + cdistmin,
        contactfrac=t * (cfracmax-cfracmin) + cfracmin,
        cellscalelimit=1.5,
        # vizsphereradius=2,
        towardaxis=True,
        alongaxis=False,
        # vizfresh=False,
        # centerasu=None,
        centerasu="toward_other",
        # centerasu='closert',
        # centerasu_at_start=t > 0.8
        showme=False,
    )
    # ipd.showme(slid)

@pytest.mark.fast
def test_asuslide_helix_case1(showme=False):
    showmeopts = ipd.dev.Bunch(vizsphereradius=4)

    np.random.seed(7084203)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=20)

    h = ipd.sym.helix.Helix(turns=15, phase=0.5, nfold=1)
    spacing = 50
    rad = 70
    hgeom = ipd.dev.Bunch(radius=rad, spacing=spacing, turns=2)
    cellsize = [hgeom.radius, hgeom.radius, hgeom.spacing]
    rb1 = ipd.sym.helix_slide(h, xyz, cellsize, iters=0, closest=9)
    rb2 = ipd.sym.helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=9)
    # rb3 = ipd.sym.helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=20)
    # ic(cellsize, rb1.cellsize, rb2.cellsize, rb3.cellsize)
    # assert 0

    # ipd.showme(rb1, **showmeopts)
    # ipd.showme(rb2, **showmeopts)
    # ipd.showme(rb3, **showmeopts)

    # ic(rb1.cellsize)
    ic(rb2.cellsize)  # type: ignore
    assert np.allclose(rb1.cellsize, [70, 70, 50])
    # assert np.allclose(rb2.cellsize, rb3.cellsize)
    assert np.allclose(rb2.cellsize, [113.7143553, 113.7143553, 44.31469973])

@pytest.mark.fast
def test_asuslide_helix_nfold1(showme=False):
    showmeopts = ipd.dev.Bunch(vizsphereradius=4)

    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)

    h = ipd.sym.helix.Helix(turns=15, phase=0.5, nfold=1)
    spacing = 70
    rad = h.turns * 0.8 * h.nfold * spacing / 2 / np.pi
    hgeom = ipd.dev.Bunch(radius=rad, spacing=spacing, turns=2)
    cellsize = [hgeom.radius, hgeom.radius, hgeom.spacing]

    rb1 = ipd.sym.helix_slide(h, xyz, cellsize, iters=0, closest=9)
    rb2 = ipd.sym.helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=9, showme=False, step=5)
    rb3 = ipd.sym.helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=20, step=5)

    # ic(cellsize, rb1.cellsize, rb2.cellsize, rb3.cellsize)
    # assert 0

    # ipd.showme(rb1, **showmeopts)
    # ipd.showme(rb2, **showmeopts)
    # ipd.showme(rb3, **showmeopts)

    ic(rb1.cellsize)  # type: ignore
    ic(rb2.cellsize)  # type: ignore
    ic(rb3.cellsize)  # type: ignore
    assert np.allclose(rb1.cellsize, [133.6901522, 133.6901522, 70.0])
    assert np.allclose(rb2.cellsize, rb3.cellsize)
    assert np.allclose(rb2.cellsize, [109.21284284, 109.21284284, 43.59816075])

@pytest.mark.fast
def test_asuslide_helix_nfold1_2():
    showmeopts = ipd.dev.Bunch(vizsphereradius=6)

    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)

    h = ipd.sym.helix.Helix(turns=8, phase=0.5, nfold=1)
    spacing = 70
    rad = h.turns * spacing / 2 / np.pi * 1.3
    hgeom = ipd.dev.Bunch(radius=rad, spacing=spacing, turns=2)
    cellsize = [hgeom.radius, hgeom.radius, hgeom.spacing]

    rb1 = ipd.sym.helix_slide(h, xyz, cellsize, iters=0, closest=20)
    rb2 = ipd.sym.helix_slide(h,
                              xyz,
                              cellsize,
                              contactfrac=0.3,
                              closest=20,
                              steps=30,
                              step=8.7,
                              iters=5,
                              showme=False,
                              **showmeopts)

    # ic(rb2.frames())

    # ipd.showme(rb1, **showmeopts)
    # ipd.showme(rb2, **showmeopts)

    ic(rb1.cellsize)  # type: ignore
    ic(rb2.cellsize)  # type: ignore
    assert np.allclose(rb1.cellsize, [115.86479857, 115.86479857, 70.0])
    assert np.allclose(rb2.cellsize, [55.93962805, 55.93962805, 38.53925788])

@pytest.mark.fast
def test_asuslide_helix_nfold3():
    showmeopts = ipd.dev.Bunch(vizsphereradius=4)

    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)

    h = ipd.sym.helix.Helix(turns=6, phase=0.5, nfold=3)
    spacing = 50
    rad = h.turns * spacing / 2 / np.pi
    hgeom = ipd.dev.Bunch(radius=rad, spacing=spacing, turns=2)
    cellsize = [hgeom.radius, hgeom.radius, hgeom.spacing]

    rb1 = ipd.sym.helix_slide(h, xyz, cellsize, iters=0, closest=20)
    rb2 = ipd.sym.helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=20, step=10, iters=5, showme=False)

    # ipd.showme(rb1, **showmeopts)
    # ipd.showme(rb2, **showmeopts)

    # ic(rb1.cellsize)
    ic(rb2.cellsize)  # type: ignore
    assert np.allclose(rb1.cellsize, [47.74648293, 47.74648293, 50.0])
    assert np.allclose(rb2.cellsize, [44.70186644, 44.70186644, 146.78939426])

@pytest.mark.fast
def test_asuslide_helix_nfold5():
    showmeopts = ipd.dev.Bunch(vizsphereradius=4)

    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)

    h = ipd.sym.helix.Helix(turns=4, phase=0.1, nfold=5)
    spacing = 40
    rad = h.turns * h.nfold * spacing / 2 / np.pi
    hgeom = ipd.dev.Bunch(radius=rad, spacing=spacing, turns=2)
    cellsize = [hgeom.radius, hgeom.radius, hgeom.spacing]

    rb = ipd.sym.helix_slide(h, xyz, cellsize, iters=0, closest=0)
    rb2 = ipd.sym.helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=9)
    rb3 = ipd.sym.helix_slide(h, xyz, rb2.cellsize, iters=0, closest=0)

    # ipd.showme(rb, **showmeopts)
    # ipd.showme(rb2, **showmeopts)
    # ipd.showme(rb3, **showmeopts)

    ic(rb.cellsize)  # type: ignore
    ic(rb2.cellsize)  # type: ignore
    assert np.allclose(rb.cellsize, [127.32395447, 127.32395447, 40.0])
    assert np.allclose(rb2.cellsize, [153.14643468, 153.14643468, 49.28047224])
    assert np.allclose(rb3.cellsize, rb2.cellsize)

@pytest.mark.fast
def test_asuslide_L442():
    sym = "L4_42"
    xtal = ipd.sym.xtal.Xtal(sym)
    csize = 160
    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)
    asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
    xyz += ipd.homog.hvec(asucen)
    xyz[:, 1] -= 2

    # pdbfile = '/home/sheffler/project/diffusion/unbounded/step10Bsym.pdb'
    # pdb = ipd.pdb.readpdb(pdbfile).subset(chain='A')
    # xyz = pdb.ca()

    primary_frames = np.stack([
        ipd.hscaled(csize, np.eye(4)),
        xtal.symelems[0].operators[1],
        xtal.symelems[0].operators[2],
        xtal.symelems[0].operators[3],
        xtal.symelems[1].operators[1],
    ])
    primary_frames = ipd.hscaled(csize, primary_frames)
    frames = primary_frames

    slid = asuslide(
        sym,
        xyz,
        frames,
        showme=False,
        maxstep=30,
        step=10,
        iters=10,
        clashiters=0,
        clashdis=8,
        contactdis=16,
        contactfrac=0.2,
        vizsphereradius=2,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
        cellscalelimit=1.2,
    )
    # ipd.showme(slid)
    ic(slid.cellsize, slid.asym.com())  # type: ignore
    assert np.allclose(slid.cellsize, 99.16625977)
    assert np.allclose(slid.asym.com(), [2.86722158e01, -1.14700730e00, 4.03010958e-16, 1.00000000e00])

@pytest.mark.fast
def test_asuslide_I4132_clashframes():
    sym = "I4132_322"
    xtal = ipd.sym.xtal.Xtal(sym)
    csize = 200
    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)
    asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
    xyz += ipd.homog.hvec(asucen)
    xyz[:, :3] -= 2

    primaryframes = np.stack([
        ipd.hscaled(csize, np.eye(4)),
        xtal.symelems[0].operators[1],
        xtal.symelems[0].operators[2],
        xtal.symelems[1].operators[1],
        xtal.symelems[2].operators[1],
    ])

    primaryframes = ipd.hscaled(csize, primaryframes)
    frames = ipd.sym.frames(sym,
                            ontop=primaryframes,
                            cells=(-1, 1),
                            cellsize=csize,
                            center=ipd.homog.hcom(xyz),
                            xtalrad=csize * 0.5)
    # frames = primaryframes

    tooclose = ft.partial(ipd.dock.rigid.tooclose_primary_overlap, nprimary=len(primaryframes))
    # tooclose = ipd.dock.rigid.tooclose_overlap

    slid = asuslide(
        sym,
        xyz,
        frames,
        showme=False,
        maxstep=30,
        step=5,
        iters=5,
        clashiters=0,
        clashdis=8,
        contactdis=16,
        contactfrac=0.3,
        vizsphereradius=6,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
    )  # , tooclosefunc=tooclose)
    # xtal.dump_pdb('test0.pdb', slid.asym.coords, cellsize=slid.cellsize, cells=0)
    # xtal.dump_pdb('test1.pdb', slid.asym.coords, cellsize=slid.cellsize, cells=(-1, 0), ontop='primary')
    # ipd.showme(slid)
    ic(slid.cellsize, slid.asym.com())  # type: ignore
    assert np.allclose(slid.cellsize, 180.390625)
    assert np.allclose(slid.asym.com(), [-4.80305991, 11.55346709, 28.23302801, 1.0])

    slid = asuslide(
        sym,
        xyz,
        frames,
        showme=False,
        maxstep=30,
        step=5,
        iters=5,
        clashiters=0,
        clashdis=8,
        contactdis=16,
        contactfrac=0.2,
        vizsphereradius=6,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
        tooclosefunc=tooclose,
    )
    # xtal.dump_pdb('test0.pdb', slid.asym.coords, cellsize=slid.cellsize, cells=0)
    # xtal.dump_pdb('test1.pdb', slid.asym.coords, cellsize=slid.cellsize, cells=(-1, 0), ontop='primary')
    # ipd.showme(slid)
    ic(slid.cellsize, slid.asym.com())  # type: ignore
    assert np.allclose(slid.cellsize, 241.25)
    assert np.allclose(slid.asym.com(), [-3.44916815, 14.59051223, 37.75725345, 1.0])

    # assert 0

def asuslide_case2():
    sym = "I4132_322"
    xtal = ipd.sym.xtal.Xtal(sym)
    # cellsize = 99.417
    cellsize = 115

    pdbfile = "/home/sheffler/project/diffusion/unbounded/step12Bsym.pdb"
    pdb = ipd.pdb.readpdb(pdbfile).subset(chain="A")
    xyz = pdb.ca()
    fracremains = 1.0
    primaryframes = xtal.primary_frames(cellsize)
    # frames = ipd.sym.frames(sym, ontop=primaryframes, cells=(-1, 1), cellsize=cellsize, center=cen, xtalrad=cellsize * 0.5)
    frames = primaryframes
    slid = ipd.sym.asuslide(
        sym=sym,
        coords=xyz,
        showme=True,
        frames=xtal.primary_frames(cellsize),
        cellsize=cellsize,
        maxstep=100,
        step=6*fracremains + 2,
        iters=6,
        clashiters=0,
        receniters=3,
        clashdis=4*fracremains + 2,
        contactdis=8*fracremains + 8,
        contactfrac=fracremains*0.3 + 0.3,
        # vizsphereradius=2,
        towardaxis=True,
        alongaxis=False,
        # vizfresh=False,
        # centerasu=None,
        centerasu="toward_other",
        # centerasu='closert',
        # centerasu_at_start=fracremains > 0.8
        # showme=True,
    )

    assert 0

def asuslide_case1():
    sym = "I4132_322"
    xtal = ipd.sym.xtal.Xtal(sym)
    # csize = 20
    # fname = '/home/sheffler/src/ipd/step2A.pdb'
    fname = "/home/sheffler/project/diffusion/unbounded/step-9Ainput.pdb"
    pdb = ipd.pdb.readpdb(fname)
    chainA = pdb.subset(chain="A")
    chainD = pdb.subset(chain="D")

    cachains = pdb.ca().reshape(xtal.nprimaryframes, -1, 4)
    csize = ipd.homog.hnorm(ipd.homog.hcom(chainD.ca()) * 2)
    ic(csize)  # type: ignore
    csize, shift = xtal.fit_coords(cachains, noshift=True)
    ic(csize)  # type: ignore
    # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
    xyz = chainA.ca()
    # xyz = pdb.ca()
    # xyz = xyz[:, :4].reshape(-1, 3)
    # ic(xyz.shape)

    # primary_frames = np.stack([
    # ipd.hscaled(csize, np.eye(4)),
    # xtal.symelems[0].operators[1],
    # xtal.symelems[0].operators[2],
    # xtal.symelems[1].operators[1],
    # ])
    # primary_frames = ipd.hscaled(csize, primary_frames)
    primary_frames = xtal.primary_frames(cellsize=csize)
    frames = primary_frames

    slid = asuslide(
        sym,
        xyz,
        frames,
        showme=True,
        printme=False,
        maxstep=100,
        step=10,
        iters=6,
        clashiters=0,
        receniters=3,
        clashdis=8,
        contactdis=16,
        contactfrac=0.5,
        vizsphereradius=2,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu="toward_other",
        centerasu_at_start=True,
    )
    ic(slid.cellsize)  # type: ignore
    assert 0
    # x = ipd.sym.xtal.Xtal(sym)
    # x.dump_pdb('test.pdb', slid.asym.coords, cellsize=slid.cellsize)
    # print(x)
    # ic(ipd.hcart3(slid.asym.globalposition))
    # assert np.allclose(slid.cellsize, 262.2992230399999)
    # assert np.allclose(ipd.hcart3(slid.asym.globalposition), [67.3001427, 48.96971455, 60.86220864])

@pytest.mark.fast
def test_asuslide_I213():
    sym = "I213"
    xtal = ipd.sym.xtal.Xtal(sym)
    csize = 200
    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)
    # asucen = xtal.asucen(method='closest', use_olig_nbrs=True, cellsize=csize)
    asucen = xtal.asucen(method="stored", cellsize=csize)
    xyz += ipd.homog.hvec(asucen)
    # xyz[:, 1] -= 2

    # ipd.showme(ipd.dock.rigid.RigidBodyFollowers(sym=sym, coords=xyz, cellsize=csize, xtalrad=0.7))
    # assert 0

    # primary_frames = np.stack([
    # ipd.hscaled(csize, np.eye(4)),
    # xtal.symelems[0].operators[1],
    # xtal.symelems[0].operators[2],
    # xtal.symelems[1].operators[1],
    # ])
    # primary_frames = ipd.hscaled(csize, primary_frames)
    frames = None  # xtal.primary_frames(cellsize=csize)

    slid = asuslide(
        sym,
        xyz,
        showme=False,
        frames=frames,
        maxstep=13,
        step=10,
        iters=3,
        clashiters=0,
        clashdis=8,
        contactdis=16,
        contactfrac=0.1,
        vizsphereradius=6,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
        xtalrad=0.6,
        iterstepscale=0.5,
    )
    # asym = ipd.dock.rigid.RigidBodyFollowers(sym=sym, coords=slid.asym.coords, cellsize=slid.cellsize,
    # frames=xtal.primary_frames(cellsize=slid.cellsize))
    # x = ipd.sym.xtal.Xtal(sym)
    # x.dump_pdb('test.pdb', slid.asym.coords, cellsize=slid.cellsize)
    # print(x)
    # ipd.showme(slid, vizsphereradius=6)
    # ipd.showme(asym, vizsphereradius=6)

    ic(slid.cellsize)  # type: ignore
    ic(slid.asym.com())  # type: ignore
    assert np.allclose(slid.cellsize, 142.5)
    assert np.allclose(slid.asym.com(), [82.59726537, 52.63939034, 90.46451613, 1.0])

    # frames = ipd.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=csize, center=asucen, asucen=asucen,
    # xtalrad=csize * 0.5)
    # slid2 = asuslide(sym, xyz, frames, showme=False, maxstep=50, step=10, iters=10, clashiters=0, clashdis=8,
    # contactdis=16, contactfrac=0.2, vizsphereradius=2, cellsize=csize, extraframesradius=1.5 * csize,
    # towardaxis=True, alongaxis=False, vizfresh=False, centerasu=False)

@pytest.mark.fast
def test_asuslide_L632():
    sym = "L6_32"
    xtal = ipd.sym.xtal.Xtal(sym)
    csize = 160
    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)
    asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
    xyz += ipd.homog.hvec(asucen)
    xyz[:, 1] -= 2

    primary_frames = np.stack([
        ipd.hscaled(csize, np.eye(4)),
        xtal.symelems[0].operators[1],
        xtal.symelems[0].operators[2],
        xtal.symelems[1].operators[1],
    ])
    primary_frames = ipd.hscaled(csize, primary_frames)
    frames = primary_frames

    slid = asuslide(
        sym,
        xyz,
        frames,
        showme=True,
        maxstep=20,
        step=10,
        iters=3,
        clashiters=0,
        clashdis=8,
        contactdis=10,
        contactfrac=0.1,
        vizsphereradius=6,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
    )
    # ipd.showme(slid)
    ic(slid.cellsize, slid.asym.com())  # type: ignore
    assert np.allclose(slid.cellsize, 97.5)
    assert np.allclose(slid.asym.com(), [2.89453125e01, -1.21875000e00, 3.27446403e-16, 1.00000000e00])

@pytest.mark.fast
def test_asuslide_I4132():
    sym = "I4132_322"
    xtal = ipd.sym.xtal.Xtal(sym)
    csize = 360
    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)
    asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
    xyz += ipd.homog.hvec(asucen)
    xyz[:, 1] -= 2

    primary_frames = np.stack([
        ipd.hscaled(csize, np.eye(4)),
        xtal.symelems[0].operators[1],
        xtal.symelems[0].operators[2],
        xtal.symelems[1].operators[1],
        xtal.symelems[2].operators[1],
    ])
    primary_frames = ipd.hscaled(csize, primary_frames)
    frames = primary_frames

    slid = asuslide(
        sym,
        xyz,
        frames,
        showme=False,
        maxstep=30,
        step=5,
        iters=3,
        clashiters=0,
        clashdis=8,
        contactdis=16,
        contactfrac=0.2,
        vizsphereradius=2,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
    )
    # ipd.showme(slid, vizsphereradius=6)
    ic(slid.cellsize)  # type: ignore
    ic(slid.asym.com())  # type: ignore
    # ic(ipd.hcart3(slid.asym.globalposition))
    # x = ipd.sym.xtal.Xtal(sym)
    # x.dump_pdb('test.pdb', slid.asym.coords, cellsize=slid.cellsize)
    assert np.allclose(slid.cellsize, 183.75)
    assert np.allclose(slid.asym.com(), [0.26694229, 19.87146628, 36.37601256, 1.0])

    slid2 = asuslide(
        sym,
        xyz,
        showme=False,
        maxstep=50,
        step=10,
        iters=3,
        clashiters=0,
        clashdis=8,
        contactdis=16,
        contactfrac=0.2,
        vizsphereradius=2,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
        xtalrad=0.5,
    )
    # ipd.showme(slid2, vizsphereradius=6)
    ic(slid.cellsize)  # type: ignore
    ic(slid.asym.com())  # type: ignore
    # ic(ipd.hcart3(slid.asym.globalposition))
    assert np.allclose(slid.cellsize, 183.75)
    assert np.allclose(slid.asym.com(), [0.26694229, 19.87146628, 36.37601256, 1.0])
    # assert 0

@pytest.mark.fast
def test_asuslide_p213():
    sym = "P 21 3"
    xtal = ipd.sym.xtal.Xtal(sym)
    csize = 180
    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)
    asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
    xyz += ipd.homog.hvec(asucen)
    xyz[:, 1] -= 2

    primary_frames = xtal.primary_frames(cellsize=csize)
    slid = asuslide(
        showme=0,
        sym=sym,
        coords=xyz,
        frames=primary_frames,
        maxstep=30,
        step=7,
        iters=5,
        subiters=3,
        contactdis=16,
        contactfrac=0.1,
        vizsphereradius=6,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
        iterstepscale=0.75,
    )
    # ipd.showme(slid)
    # slid.dump_pdb('test1.pdb')
    # ic(slid.bvh_op_count, len(slid.bodies))
    ic(slid.cellsize, slid.asym.com())  # type: ignore
    assert np.allclose(slid.cellsize, 161.5703125)
    assert np.allclose(slid.asym.com(), [81.45648685, 41.24336469, 62.20570401, 1.0])

    frames = xtal.frames(cells=(-1, 1), cellsize=csize, xtalrad=0.9)

    slid = asuslide(
        showme=0,
        sym=sym,
        coords=xyz,
        frames=frames,
        maxstep=30,
        step=7,
        iters=5,
        subiters=3,
        contactdis=16,
        contactfrac=0.1,
        vizsphereradius=6,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
        iterstepscale=0.75,
    )
    # slid.dump_pdb('test2.pdb')
    # ic(slid.bvh_op_count, len(slid.bodies))
    ic(slid.cellsize, slid.asym.com())  # type: ignore
    assert np.allclose(slid.cellsize, 161.5703125)
    assert np.allclose(slid.asym.com(), [81.45648685, 41.24336469, 62.20570401, 1.0])

    slid = asuslide(
        showme=0,
        sym=sym,
        coords=xyz,
        maxstep=30,
        step=7,
        iters=5,
        subiters=3,
        contactdis=16,
        contactfrac=0.1,
        vizsphereradius=6,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
        iterstepscale=0.75,
    )
    # ipd.showme(slid)
    # slid.dump_pdb('test3.pdb')
    # ic(slid.bvh_op_count, len(slid.bodies))
    # ic(slid.cellsize, slid.asym.com())
    # ic(ipd.hcart3(slid.asym.globalposition))
    # ic(slid.asym.tolocal)
    assert np.allclose(slid.cellsize, 161.5703125)
    assert np.allclose(slid.asym.com(), [81.45648685, 41.24336469, 62.20570401, 1.0])

@pytest.mark.fast
def test_asuslide_oct():
    sym = "oct"
    ax2 = ipd.sym.axes(sym)[2]
    ax3 = ipd.sym.axes(sym)[3]
    # axisinfo = [(2, ax2, (2, 3)), (3, ax3, 1)]
    axesinfo = [(ax2, [0, 0, 0]), (ax3, [0, 0, 0])]
    primary_frames = [np.eye(4), ipd.homog.hrot(ax2, 180), ipd.homog.hrot(ax3, 120), ipd.homog.hrot(ax3, 240)]
    # frames = primary_frames
    frames = ipd.sym.frames(sym, ontop=primary_frames)

    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)
    xyz += ax2 * 20
    xyz += ax3 * 20
    xyz0 = xyz.copy()

    slid = asuslide(
        showme=0,
        sym=sym,
        coords=xyz,
        frames=frames,
        axes=axesinfo,
        alongaxis=True,
        towardaxis=False,
        iters=3,
        subiters=3,
        contactfrac=0.1,
        contactdis=16,
        vizsphereradius=6,
    )
    ic(slid.asym.com(), slid.cellsize)  # type: ignore
    assert np.all(np.abs(slid.frames()[:, :3, 3]) < 0.0001)
    assert np.allclose(slid.asym.com(), [67.39961966, 67.39961966, 25.00882048, 1.0])
    assert np.allclose(slid.cellsize, [1, 1, 1])
    assert np.allclose(np.eye(3), slid.asym.position[:3, :3])
    # slid.dump_pdb('ref.pdb')

    slid = asuslide(
        showme=0,
        sym=sym,
        coords=xyz,
        frames=primary_frames,
        axes=axesinfo,
        alongaxis=True,
        towardaxis=False,
        iters=3,
        subiters=3,
        contactfrac=0.1,
        contactdis=16,
        vizsphereradius=6,
    )
    # ic(slid.asym.com(), slid.cellsize)
    assert np.all(np.abs(slid.frames()[:, :3, 3]) < 0.0001)
    assert np.allclose(slid.asym.com(), [67.39961966, 67.39961966, 25.00882048, 1.0])
    assert np.allclose(slid.cellsize, [1, 1, 1])
    assert np.allclose(np.eye(3), slid.asym.position[:3, :3])
    # slid.dump_pdb('test0.pdb')

    xyz = xyz0 - ax2*30
    slid2 = asuslide(
        showme=0,
        sym=sym,
        coords=xyz,
        frames=frames,
        alongaxis=True,
        vizsphereradius=6,
        contactdis=12,
        contactfrac=0.1,
        maxstep=20,
        iters=3,
        subiters=3,
        towardaxis=False,
        along_extra_axes=[[0, 0, 1]],
    )
    # ic(slid.asym.com(), slid.cellsize)
    assert np.all(np.abs(slid2.frames()[:, :3, 3]) < 0.0001)
    assert np.allclose(np.eye(3), slid2.asym.position[:3, :3])
    assert np.allclose(slid.asym.com(), [67.39961966, 67.39961966, 25.00882048, 1.0])
    # slid.dump_pdb('test1.pdb')

    xyz = xyz0 - ax2*20
    slid2 = asuslide(
        showme=0,
        sym=sym,
        coords=xyz,
        frames=primary_frames,
        alongaxis=True,
        vizsphereradius=6,
        contactdis=12,
        contactfrac=0.1,
        maxstep=20,
        iters=3,
        subiters=3,
        towardaxis=False,
    )
    # ic(slid.asym.com(), slid.cellsize)
    assert np.all(np.abs(slid2.frames()[:, :3, 3]) < 0.0001)
    assert np.allclose(np.eye(3), slid2.asym.position[:3, :3])
    assert np.allclose(slid.asym.com(), [67.39961966, 67.39961966, 25.00882048, 1.0])
    # slid.dump_pdb('test2.pdb')

@pytest.mark.fast
def test_asuslide_P432_44(showme=False):
    sym = "P_4_3_2"
    xtal = ipd.sym.xtal.Xtal(sym)
    csize = 200
    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)
    asucen = xtal.asucen(method="stored", cellsize=csize)
    xyz += ipd.homog.hvec(asucen)
    primary_frames = xtal.primary_frames(cellsize=csize)
    cen = ipd.homog.hcom(xyz)
    frames = ipd.sym.frames(
        sym,
        ontop=primary_frames,
        cells=(-1, 1),
        cellsize=csize,
        center=cen,
        asucen=asucen,
        xtalrad=0.9,
        strict=False,
    )

    # rbprimary = ipd.dock.rigid.RigidBodyFollowers(coords=xyz, frames=primary_frames)
    # ipd.showme(rbprimary)
    # rbstart = ipd.dock.rigid.RigidBodyFollowers(coords=xyz, frames=frames)
    # ipd.showme(rbstart)
    # frames = primary_frames

    slid = asuslide(
        sym,
        xyz,
        frames,
        showme=False,
        maxstep=10,
        step=10.123,
        iters=3,
        clashiters=0,
        clashdis=8,
        contactdis=16,
        contactfrac=0.2,
        vizsphereradius=6,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
        iterstepscale=0.5,
        resetonfail=True,
    )
    # ipd.showme(slid)
    ic(slid.cellsize, slid.asym.com())  # type: ignore
    assert np.allclose(slid.cellsize, 146.85425)
    assert np.allclose(slid.asym.com(), [18.666027, 37.33205399, 55.99808099, 1.0])

    asucen = xtal.asucen(method="stored", cellsize=csize)
    csize = 100
    slid = asuslide(
        sym,
        xyz,
        showme=False,
        maxstep=10,
        step=10.123,
        iters=3,
        clashiters=0,
        clashdis=8,
        contactdis=16,
        contactfrac=0.2,
        vizsphereradius=6,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
        iterstepscale=0.5,
        resetonfail=True,
        xtalrad=0.6,
    )
    # ipd.showme(slid)
    ic(slid.cellsize, slid.asym.com())  # type: ignore
    assert np.allclose(slid.cellsize, 145.5535)
    assert np.allclose(slid.asym.com(), [18.74043474, 37.48086949, 56.22130423, 1.0])

@pytest.mark.fast
def test_asuslide_P432_43(showme=False):
    sym = "P_4_3_2_43"
    xtal = ipd.sym.xtal.Xtal(sym)
    csize = 180
    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)
    asucen = xtal.asucen(method="stored", cellsize=csize)
    xyz += ipd.homog.hvec(asucen)
    primary_frames = xtal.primary_frames(cellsize=csize)
    cen = ipd.homog.hcom(xyz)
    frames = ipd.sym.frames(
        sym,
        ontop=primary_frames,
        cells=(-1, 1),
        cellsize=csize,
        center=cen,
        asucen=asucen,
        xtalrad=0.6,
        strict=False,
    )

    slid = asuslide(
        sym,
        xyz,
        frames,
        showme=showme,
        maxstep=10,
        step=10.123,
        iters=3,
        clashiters=0,
        clashdis=8,
        contactdis=16,
        contactfrac=0.2,
        vizsphereradius=6,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
        iterstepscale=0.5,
        resetonfail=True,
    )
    # ipd.showme(slid)
    ic(slid.cellsize)  # type: ignore
    ic(slid.asym.com())  # type: ignore
    assert np.allclose(slid.cellsize, 147.10025)
    assert np.allclose(slid.asym.com(), [18.69073977, 37.38147955, 56.07221932, 1.0])

@pytest.mark.fast
def test_asuslide_F432():
    sym = "F_4_3_2"
    xtal = ipd.sym.xtal.Xtal(sym)
    csize = 150
    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)
    asucen = xtal.asucen(method="stored", cellsize=csize)
    # ic(asucen / csize)
    # assert 0
    xyz += ipd.homog.hvec(asucen)

    # frames = ipd.sym.frames(sym, cells=None, cellsize=csize)
    frames = ipd.sym.frames(sym, cellsize=csize, cen=ipd.homog.hcom(xyz), xtalrad=0.5, strict=False)
    # ic(frames.shape)
    # assert 0

    slid = asuslide(
        sym,
        xyz,
        frames,
        showme=0,
        maxstep=30,
        step=10.3,
        iters=3,
        subiters=2,
        contactdis=10,
        contactfrac=0.1,
        vizsphereradius=6,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
        along_extra_axes=[[0, 0, 1]],
        iterstepscale=0.7,
    )
    ipd.showme(slid, vizsphereradius=6)
    ic(slid.cellsize, slid.asym.com())  # type: ignore
    assert np.allclose(slid.cellsize, 201.5)
    assert np.allclose(slid.asym.com(), [154.9535, 15.5155, 77.5775, 1.0])

    # cen = asucen
    # cen = ipd.homog.hcom(xyz + [0, 0, 0, 0])
    # frames = ipd.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=csize, center=asucen, asucen=cen,
    # xtalrad=csize * 0.7)
    # ic(len(frames))
    # assert 0

    # ipd.showme(slid2)

@pytest.mark.fast
def test_asuslide_I432():
    sym = "I_4_3_2"
    xtal = ipd.sym.xtal.Xtal(sym)
    csize = 250
    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)
    # ic(xyz.shape)
    # xyz += ipd.homog.hvec(xtal.asucen(method='stored', cellsize=csize))
    xyz += ipd.homog.hvec(xtal.asucen(method="closest", cellsize=csize))

    # p = xtal.primary_frames(cellsize=csize)
    # ic(p.shape)
    # ic(ipd.hcart3(p))
    # assert 0

    # ipd.showme(ipd.dock.rigid.RigidBodyFollowers(sym=sym, coords=xyz, cellsize=csize, xtalrad=0.7, strict=False))

    slid = asuslide(
        sym,
        xyz,
        showme=0,
        maxstep=30,
        step=10,
        iters=2,
        subiters=1,
        clashiters=0,
        clashdis=8,
        contactdis=12,
        contactfrac=0.2,
        vizsphereradius=6,
        cellsize=csize,
        towardaxis=True,
        alongaxis=False,
        vizfresh=False,
        centerasu=False,
        along_extra_axes=[],
        xtalrad=0.5,
        iterstepscale=0.5,
    )

    # ic(slid.bvh_op_count)
    # ipd.showme(slid)
    ic(slid.cellsize, slid.asym.com())  # type: ignore
    assert np.allclose(slid.cellsize, 165)
    assert np.allclose(slid.asym.com(), [48.91764706, 29.7, 13.97647059, 1.0])

    # cen = asucen
    # cen = ipd.homog.hcom(xyz + [0, 0, 0, 0])
    # frames = ipd.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=csize, center=asucen, asucen=cen,
    # xtalrad=csize * 0.7)
    # ic(len(frames))

    # ipd.showme(slid2)

@pytest.mark.fast
def test_asuslide_from_origin():
    from ipd.tests.testdata.misc.asuslide_misc import test_asuslide_case2_coords  # type: ignore

    def boundscheck_L632(bodies):
        return True

    sym = "L632"
    kw = {"maxstep": 40, "clashdis": 5.68, "contactdis": 12.0, "contactfrac": 0.05, "cellscalelimit": 1.5}
    csize = 1
    slid = asuslide(
        showme=1,
        sym=sym,
        coords=test_asuslide_case2_coords,
        axes=None,
        existing_olig=None,
        alongaxis=0,
        towardaxis=True,
        printme=False,
        cellsize=csize,
        isxtal=False,
        nbrs="auto",
        doscale=True,
        iters=2,
        subiters=2,
        clashiters=0,
        receniters=0,
        step=5.26,
        scalestep=None,
        closestfirst=True,
        centerasu="toward_other",
        centerasu_at_start=False,
        scaleslides=1.0,
        iterstepscale=0.75,
        coords_to_asucen=False,
        boundscheck=boundscheck_L632,
        nobadsteps=True,
        vizsphereradius=6,
        **kw,
    )
    ic(slid.asym.com(), slid.cellsize)  # type: ignore

@pytest.mark.fast
def test_asuslide_case2():
    from ipd.tests.testdata.misc.asuslide_misc import test_asuslide_case2_coords  # type: ignore
    sym = "L632"
    kw = {"maxstep": 40, "clashdis": 5.68, "contactdis": 12.0, "contactfrac": 0.05, "cellscalelimit": 1.5}
    xtal = ipd.sym.xtal.Xtal(sym)
    csize = 80

    frames = xtal.primary_frames(cellsize=csize)  # xtal.frames(cellsize=csize)
    slid = asuslide(
        showme=0,
        sym=sym,
        coords=test_asuslide_case2_coords,
        frames=frames,
        axes=None,
        existing_olig=None,
        alongaxis=0,
        towardaxis=True,
        printme=False,
        cellsize=csize,
        isxtal=False,
        nbrs="auto",
        doscale=True,
        iters=2,
        subiters=2,
        clashiters=0,
        receniters=0,
        step=5.26,
        scalestep=None,
        closestfirst=True,
        centerasu="toward_other",
        centerasu_at_start=False,
        scaleslides=1.0,
        iterstepscale=0.75,
        coords_to_asucen=False,
        nobadsteps=True,
        vizsphereradius=6,
        **kw,
    )
    # ipd.showme(slid)
    # ic(slid.asym.com(), slid.cellsize)
    assert np.allclose(slid.asym.com(), [18.33744584, 0.30792098, 3.55403141, 1])
    assert np.allclose(slid.cellsize, [58.96, 58.96, 58.96])
    slid = asuslide(
        showme=0,
        sym=sym,
        coords=test_asuslide_case2_coords,
        axes=None,
        existing_olig=None,
        alongaxis=0,
        towardaxis=True,
        printme=False,
        cellsize=csize,
        isxtal=False,
        nbrs="auto",
        doscale=True,
        iters=2,
        subiters=2,
        clashiters=0,
        receniters=0,
        step=5.26,
        scalestep=None,
        closestfirst=True,
        centerasu="toward_other",
        centerasu_at_start=False,
        scaleslides=1.0,
        iterstepscale=0.75,
        coords_to_asucen=False,
        nobadsteps=True,
        vizsphereradius=6,
        **kw,
    )
    # ipd.showme(slid)
    # ic(slid.asym.com(), slid.cellsize)
    assert np.allclose(slid.asym.com(), [18.33744584, 0.30792098, 3.55403141, 1])
    assert np.allclose(slid.cellsize, [58.96, 58.96, 58.96])

    # ic(test_asuslide_case2_coords.shape)
    # ic(slid.asym.coords.shape)
    # ic(slid.coords.shape)
    # slid.dump_pdb('ref.pdb')

    def boundscheck_L632(bodies):
        com = bodies.asym.com()
        if com[0] < 0:
            return False
        if com[0] > 4 and abs(np.arctan2(com[1], com[0])) > np.pi / 6:
            return False
        com2 = bodies.bodies[3].com()
        if com[0] > com2[0]:
            return False
        return True

    # coords = test_asuslide_case2_coords
    coords = ipd.hcentered(test_asuslide_case2_coords, singlecom=True)
    coords[..., 0] += 5
    # ipd.showme(test_asuslide_case2_coords[:, 1])
    # ic(ipd.homog.hcom(coords))
    slid = asuslide(
        showme=0,
        sym=sym,
        coords=coords,
        axes=None,
        existing_olig=None,
        alongaxis=0,
        towardaxis=True,
        printme=False,
        cellsize=csize,
        isxtal=False,
        nbrs="auto",
        doscale=True,
        iters=2,
        subiters=2,
        clashiters=0,
        receniters=0,
        step=5.26,
        scalestep=None,
        closestfirst=True,
        centerasu="toward_other",
        centerasu_at_start=False,
        scaleslides=1.0,
        iterstepscale=0.75,
        coords_to_asucen=True,
        nobadsteps=True,
        vizsphereradius=6,
        boundscheck=boundscheck_L632,
        **kw,
    )
    # slid.dump_pdb('test.pdb')
    # ipd.showme(slid)
    ic(slid.asym.com(), slid.cellsize)  # type: ignore
    ic("=======")  # type: ignore
    # don't know why this is unstable... generally off by a few thou
    assert np.allclose(slid.asym.com(), [1.81500000e01, -4.17462713e-04, 4.31305757e-15, 1.00000000e00], atol=0.1)
    assert np.allclose(slid.cellsize, [58.96, 58.96, 58.96], atol=0.01)

    coords = ipd.hcentered(test_asuslide_case2_coords, singlecom=True)
    coords[..., 0] += 5
    csize = 10
    slid = asuslide(
        showme=False,
        sym=sym,
        coords=coords,
        axes=None,
        existing_olig=None,
        alongaxis=0,
        towardaxis=True,
        printme=False,
        cellsize=csize,
        isxtal=False,
        nbrs="auto",
        doscale=True,
        iters=2,
        subiters=2,
        clashiters=0,
        receniters=0,
        step=5.26,
        scalestep=None,
        closestfirst=True,
        centerasu="toward_other",
        centerasu_at_start=False,
        scaleslides=1.0,
        iterstepscale=0.75,
        coords_to_asucen=False,
        nobadsteps=True,
        vizsphereradius=6,
        boundscheck=boundscheck_L632,
        **kw,
    )
    # ipd.showme(slid)
    ic(slid.asym.com(), slid.cellsize)  # type: ignore
    assert np.allclose(slid.asym.com(), [1.81500000e01, -4.17462713e-04, 4.31305757e-15, 1.00000000e00], atol=0.1)

    assert np.allclose(slid.cellsize, 57.34, atol=0.01)

if __name__ == "__main__":
    main()
    print("test_aluslide DONE")
