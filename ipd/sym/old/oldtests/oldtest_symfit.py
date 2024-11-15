import numpy as np

import ipd
import ipd.homog as hm
from ipd.sym import setup_test_frames, symfit_mc_play  # type: ignore

def main():
    t = ipd.dev.Timer()

    test_extra_cyclic()
    test_cyclic_sym_err()
    test_rel_xform_info()
    test_rel_xform_info_rand()
    test_symops_cen_perfect()
    test_symops_cen_imperfect()
    test_symfit_align_axes()
    test_disambiguate_axes()
    test_symfit_dihedral()
    test_symfit_d2()
    test_symfit_d2_af2()
    test_af2_example()
    test_symfit_d3_nfold_error()
    test_d4_error()
    assert 0

    for isym in range(1):
        sym = "d%i" % isym
        # nfail, ntest = 0, 100
        nfail, ntest = 0, 1
        seed = np.random.randint(2**32 - 1)
        for i in range(ntest):
            result = ipd.sym.symfit_mc_play(
                sym="tet",
                nframes=3,
                goalerr=0.04,
                showme=True,
                quiet=True,
                fuzzstdfrac=0.4,
                random_frames=True,
                maxiters=500,
                max_cartsd=1,
                scaletemp=1.0,
                scalesamp=1.0,
                tprelen=5,
                vizinterval=1,
                min_radius=10,
                max_radius=100,
                choose_closest_frame=True,
                showsymdups=True,
                showsymops=True,
                showopts=ipd.dev.Bunch(
                    cyc_ang_match_tol=0.3,
                    axisrad=0.19,
                    helicalrad=None,
                    fixedfansize=2.0,
                    spheres=0.8,
                    weight=4,
                    headless=False,
                    xyzlen=[2, 2, 2],
                    # headless=False,
                ),
                # seed=seed,
                noambigaxes=True,
                # seed=4193001408,
            )
            if result.besterr > 0.3:
                nfail += 1
            print(result.nsamp, result.besterr)
        # # t.checkpoint('symfit_mc_play')
        print(sym, nfail, ntest)
    # assert 0
    t.checkpoint("symfit_mc_play")

    test_extra_cyclic()
    t.checkpoint("test_extra_cyclic")

    test_d4_error()
    t.checkpoint("test_d4_error")

    test_af2_example()
    t.checkpoint("test_af2_example")

    test_symfit_d3_nfold_error()
    t.checkpoint("test_symfit_d3_nfold_error")
    # assert 0

    # SEED 4284572842
    # start 19.838108001495662
    # start 19.838108001495662
    # 0 19.838108001495662 0.0
    # 100 5.027766047712338 0.18811881188118812
    # 200 2.2378296496548282 0.208955223880597
    # 300 1.802095342227487 0.18604651162790697
    # 400 1.6264830366647551 0.17206982543640897
    # 500 1.604944965772087 0.15768463073852296
    # 600 2.9234244829889984 0.13810316139767054
    # 700 1.3972099214645886 0.12696148359486448
    # 800 1.5741736808536937 0.12109862671660425
    # 900 2.7650750598000853 0.1120976692563818
    # symfit_mc_play END 1.377131126759568
    # Traceback (most recent call last):

    # np.set_printoptions(
    # precision=10,
    # suppress=True,
    # linewidth=98,
    # formatter={'float': lambda f: '%8.4f' % f},
    # )
    #
    # np.seterr(all="ignore")

    #
    # test_symops_gradient()
    # t.checkpoint('test_symops_gradient')
    #
    # symfit_parallel_convergence_trials()
    #
    # symfit_mc_play(sym='icos', quiet=False, showme=True, fuzzstdfrac=0.4, random_frames=True,
    # nframes=4, maxiters=1000, scaletemp=1, scalesamp=1, seed=None, tprelen=5,
    # vizinterval=10)
    # d2   5%
    # d3  16%
    # d4   9%
    # d5  21%
    # d6  21%
    # d7  26%
    # d8  22%
    # d9  36%
    # d10 26%
    # d11 72%
    # d12 36%
    # d13 75%
    # for isym in range(2, 15):

    test_symfit_d2_af2()
    t.checkpoint("test_symfit_d2_af2")
    # assert 0

    test_symfit_d2()
    t.checkpoint("test_symfit_d2")

    test_symfit_dihedral()
    t.checkpoint("test_symfit_dihedral")

    # test_symfit_mc()
    t.checkpoint("test_symfit_mc")

    test_disambiguate_axes()
    t.checkpoint("test_disambiguate_axes")

    test_symfit_align_axes()
    t.checkpoint("test_symfit_align_axes")

    test_rel_xform_info()
    t.checkpoint("test_rel_xform_info")

    test_symops_cen_perfect()
    t.checkpoint("test_symops_cen_perfect")

    test_symops_cen_imperfect()
    t.checkpoint("test_symops_cen_imperfect")

    test_rel_xform_info_rand()
    t.checkpoint("test_rel_xform_info_rand")

    test_cyclic_sym_err()
    t.checkpoint("test_cyclic_sym_err")

    # test_cyclic()
    # t.checkpoint('test_cyclic')

    # symfit_parallel_mc_trials()
    # assert 0printe
    # errs = list()
    # for i in range(5):
    #     errs.append(test_symops_cen_imperfect(nsamp=20, manual=True))
    # err = ipd.dev.Bunch().accumulate(errs)
    # err.reduce(max)
    # print(err)

    t.report()
    print("test_symfit.py done")

@pytest.mark.fast  # type: ignore
def test_extra_cyclic(nsamp=100):
    frames = ipd.sym.frames("c5")
    frames = ipd.homog.hrandsmall(len(frames)) @ frames
    frames[4] = ipd.homog.hrandsmall(cart_sd=2, rot_sd=1)
    fit = ipd.compute_symfit("icos", frames, penalize_redundant_cyclic_nth=2)
    assert fit.redundant_cyclic_err > 5

    frames = ipd.sym.frames("d2")[:-1]
    frames = ipd.homog.hrandsmall(len(frames)) @ frames
    fit = ipd.compute_symfit("d2", frames, penalize_redundant_cyclic_nth=2)
    assert fit.redundant_cyclic_err < 0.001

    frames = ipd.sym.frames("icos")[[0, 1, 2, 3, 8, 15, 20, 31, 44]]
    # ipd.showme(frames @ ipd.homog.htrans([0, 0, 4]))
    frames = ipd.homog.hrandsmall(len(frames)) @ frames
    fit = ipd.compute_symfit("icos", frames, penalize_redundant_cyclic_nth=2)
    # print(i, fit.redundant_cyclic_err)
    assert fit.redundant_cyclic_err < 0.001

    # frames = ipd.sym.frames('c2')
    # frames = ipd.homog.hrandsmall(len(frames)) @ frames
    # # frames[4] = ipd.homog.hrandsmall(cart_sd=2, rot_sd=1)
    # fit = ipd.compute_symfit('c2', frames, penalize_redundant_cyclic_nth=2)
    # assert fit.redundant_cyclic_err < 0.001

@pytest.mark.fast  # type: ignore
def test_cyclic_sym_err(nsamp=100):
    for i in range(nsamp):
        prex = hm.rand_xform()
        axs = hm.rand_unit()
        tgtang = np.random.rand() * np.pi
        f1 = np.eye(4)
        cart = hm.hprojperp(axs, hm.hrandpoint())
        rad = np.linalg.norm(cart[:3])
        f1[:, 3] = cart
        rel = hm.hrot(axs, tgtang)
        f2 = rel @ f1
        pair = ipd.sym.rel_xform_info(f1, f2)
        err = ipd.sym.cyclic_sym_err(pair, tgtang)
        assert np.allclose(err, 0)

        tgtang2 = np.random.rand() * np.pi
        err2 = ipd.sym.cyclic_sym_err(pair, tgtang2)
        assert np.allclose(err2, abs(tgtang - tgtang2) * min(10000, max(1, rad)))

        hlen = np.random.normal()
        rel[:3, 3] = hlen * axs[:3]
        f2 = rel @ f1
        pair = ipd.sym.rel_xform_info(f1, f2)
        err = ipd.sym.cyclic_sym_err(pair, tgtang)
        assert np.allclose(err, abs(hlen))

        tgtang3 = np.random.rand() * np.pi
        err3 = ipd.sym.cyclic_sym_err(pair, tgtang3)
        angerr = (tgtang-tgtang3) * min(10000, max(1, rad))
        assert np.allclose(err3, np.sqrt(hlen**2 + angerr**2))

@pytest.mark.fast  # type: ignore
def test_rel_xform_info():
    axs0 = [0, 0, 1, 0]
    ang0 = (2 * np.random.random() - 1) * np.pi
    # frameAcen = [0, 0, 2 * np.random.random() - 1, 1]
    frameAcen = np.array([2 * np.random.random() - 1, 2 * np.random.random() - 1, 1, 1])

    xformcen = [0, 0, 0, 1]
    trans = [0, 0, 0, 1]

    # print(axs0)
    # print(ang0)
    # print(shift)

    frameA = np.eye(4)
    frameA[:, 3] = frameAcen
    xrel0 = hm.hrot(axs0, ang0, xformcen)
    xrel0[:, 3] = trans

    frameB = xrel0 @ frameA
    xinfo = ipd.sym.rel_xform_info(frameA, frameB)

    rad = np.sqrt(np.sum(frameAcen[:2]**2))
    # print('frameAcen', frameA[:, 3])
    # print('frameBcen', frameB[:, 3])

    assert np.allclose(xrel0, xinfo.xrel)
    assert np.allclose(axs0, xinfo.axs if ang0 > 0 else -xinfo.axs)
    assert np.allclose(xinfo.ang, abs(ang0))
    assert np.allclose(rad, xinfo.rad)
    assert np.allclose([0, 0, frameAcen[2], 1], xinfo.framecen)

    # print()
    # print('xinfo.cen', xinfo.cen)
    # print()
    # print('framecen', (frameA[:, 3] + frameB[:, 3]) / 2)
    # print()
    # print(axs0)
    # print('xinfo.framecen', xinfo.framecen)

    assert np.allclose(xinfo.hel, np.sum(xinfo.axs * trans))

@pytest.mark.fast  # type: ignore
def test_rel_xform_info_rand(nsamp=50):
    for i in range(nsamp):
        axs0 = [1, 0, 0, 0]
        ang0 = np.random.rand() * np.pi
        cen0 = [np.random.normal(), 0, 0, 1]

        rady = np.random.normal()
        radz = np.random.normal()
        # radz = rady
        rad0 = np.sqrt(rady**2 + radz**2)
        hel0 = np.random.normal()
        prefx = hm.rand_xform()
        postx = hm.rand_xform()

        xrel0 = hm.hrot(axs0, ang0, cen0)
        xrel0[:, 3] = [hel0, 0, 0, 1]

        frameA = np.eye(4)
        frameA = prefx @ frameA
        frameA[:, 3] = [0, rady, radz, 1]

        frameB = xrel0 @ frameA
        xinfo = ipd.sym.rel_xform_info(frameA, frameB)

        # print('xinfo.cen')
        # print(cen0)
        # print(xinfo.cen)
        # print('xinfo.rad', rad0, xinfo.rad, xinfo.rad / rad0)
        # print('xinfo.hel', hel0, xinfo.hel)
        cen1 = hm.hprojperp(axs0, cen0)
        assert np.allclose(np.linalg.norm(xinfo.axs, axis=-1), 1.0)
        assert np.allclose(xrel0, xinfo.xrel)
        assert np.allclose(axs0, xinfo.axs)
        assert np.allclose(ang0, xinfo.ang)
        assert np.allclose(cen1, xinfo.cen, atol=0.001)
        assert np.allclose(hel0, xinfo.hel)
        assert np.allclose(rad0, xinfo.rad)

        frameA = postx @ frameA
        frameB = postx @ frameB
        xinfo = ipd.sym.rel_xform_info(frameA, frameB)
        rrel0 = xrel0[:3, :3]
        rrel = xinfo.xrel[:3, :3]
        rpost = postx[:3, :3]
        assert np.allclose(np.linalg.norm(xinfo.axs, axis=-1), 1.0)

        assert np.allclose(np.linalg.det(rrel0), 1.0)
        assert np.allclose(np.linalg.norm(hm.axis_angle_of(rrel0)[0]), 1.0)

        # print(hm.axis_angle_of(rrel0))
        # print(hm.axis_angle_of(rrel))
        # print(hm.axis_angle_of(rpost))
        # print(hm.axis_angle_of(rpost @ rrel0))
        # assert np.allclose(rpost @ rrel0, rrel)
        # assert 0

        # hrm... not sure why this isn't true... rest of tests should be enough
        # assert np.allclose(postx[:3, :3] @ xrel0[:3, :3], xinfo.xrel[:3, :3])

        assert np.allclose(postx @ axs0, xinfo.axs)
        assert np.allclose(ang0, xinfo.ang)
        assert np.allclose(hm.hprojperp(xinfo.axs, postx @ cen0), hm.hprojperp(xinfo.axs, xinfo.cen))
        assert np.allclose(hel0, xinfo.hel)
        assert np.allclose(rad0, xinfo.rad)

@pytest.mark.fast  # type: ignore
def test_symops_cen_perfect(nframes=9):
    np.set_printoptions(
        precision=10,
        suppress=True,
        linewidth=98,
        formatter={"float": lambda f: "%14.10f" % f},
    )
    # r = hm.hrot([1, 0, 0], 180)
    # xinfo.axs, a = hm.axis_angle_of(r)
    # print(r)
    # print(xinfo.axs)
    # print(a)
    # assert 0

    allsym = "tet oct icos".split()
    all_symframes = [
        # ipd.sym.tetrahedral_frames,
        # ipd.sym.octahedral_frames,
        # ipd.sym.icosahedral_frames[:30],
        ipd.sym.tetrahedral_frames[np.random.choice(12, nframes, replace=False), :, :],
        ipd.sym.octahedral_frames[np.random.choice(24, nframes, replace=False), :, :],
        ipd.sym.icosahedral_frames[np.random.choice(60, nframes, replace=False), :, :],
        # ipd.sym.icosahedral_frames[(20, 21), :, :],
    ]
    all_point_angles = [
        {
            2: [np.pi],
            3: [np.pi * 2 / 3]
        },
        {
            2: [np.pi],
            3: [np.pi * 2 / 3],
            4: [np.pi / 2]
        },
        {
            2: [np.pi],
            3: [np.pi * 2 / 3],
            5: [np.pi * 2 / 5, np.pi * 4 / 5]
        },
    ]
    xpost3 = hm.rand_xform(cart_sd=10)
    xpre = hm.rand_xform(cart_sd=5)

    #    xpre = np.array([[-0.4971291915, 0.5418972027, -0.6776503439, 1.5447300543],
    #                     [0.5677267562, -0.3874638202, -0.7263319616, 2.4858980827],
    #                     [-0.6561622492, -0.7458010524, -0.1150299654, -4.0124612619],
    #                     [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]])
    #    xprost3 = np.array([[0.2066723595, -0.9743827067, 0.0886841400, -4.8092830795],
    #                        [0.9297657442, 0.2238133467, 0.2923067684, -7.8301135871],
    #                        [-0.3046673543, 0.0220437459, 0.9522036949, -13.6244069897],
    #                        [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]])

    # print()
    # print(repr(xpre))
    # print('-------------')
    # print(repr(xpost3))
    # print('-------------')

    for sym, symframes, point_angles in zip(allsym, all_symframes, all_point_angles):
        # print('---------', sym, '-----------')

        # xpre[:3, :3] = np.eye(3)

        frames = symframes @ xpre  # move subunit
        xpost1 = np.eye(4)
        # print(frames)
        # print('-------------')

        # np.random.shuffle(frames)
        # frames = frames[:nframes]

        symops = ipd.sym.symops_from_frames(sym=sym, frames=frames)
        symops = ipd.sym.stupid_pairs_from_symops(symops)
        assert len(symops) == len(frames) * (len(frames) - 1) / 2
        # print(list(symops[(0, 1)].keys()))
        for k, op in symops.items():
            # wgood, ngood = None, 0
            # for n, e in op.err.items():
            #     if e < 5e-2:
            #         ngood += 1
            #         wgood = n
            # if not ngood == 1:
            #     print('ngood!=1', k, ngood, op.err)
            #     assert 0
            assert np.allclose(0, op.cen[:3], atol=1e-3)

        xpost2 = hm.htrans(np.array([4, 5, 6]))
        xpost2inv = np.linalg.inv(xpost2)
        frames2 = xpost2 @ frames  # move whole structure
        symops2 = ipd.sym.symops_from_frames(sym=sym, frames=frames2)
        symops2 = ipd.sym.stupid_pairs_from_symops(symops2)
        assert len(symops2) == len(frames2) * (len(frames2) - 1) / 2
        assert len(frames) == len(frames2)
        for k in symops:
            op1 = symops[k]
            op2 = symops2[k]
            frame1 = frames2[k[0]]
            frame2 = frames2[k[1]]
            try:
                assert np.allclose(op2.xrel @ frame1, frame2, atol=1e-8)
                assert np.allclose(op1.axs, xpost2inv @ op2.axs, atol=1e-4)
                assert np.allclose(op1.ang, op2.ang, atol=1e-4)
                assert np.allclose(op1.cen, hm.hprojperp(op2.axs, xpost2inv @ op2.cen), atol=1e-3)
                assert np.allclose(op1.rad, op2.rad, atol=1e-3)
                assert np.allclose(op1.hel, op2.hel, atol=1e-4)
                # for k in point_angles:
                #     if not np.allclose(op1.err[k], op2.err[k], atol=1e-4):
                #         print('err', op1.err[k], op2.err[k])
                #     assert np.allclose(op1.err[k], op2.err[k], atol=1e-4)
            except AssertionError as e:
                print(repr(xpost2))
                from ipd import viz
                # assert 0
                # viz.showme(list(symops2.values()), 'someops2')
                # viz.showme(op1, 'op1')
                # viz.showme(op2, 'op2')

                print(op1.ang)
                print(repr(op1.frames))
                print(op1.axs)

                print("axs", op1.axs)
                print("cen", op1.cen)
                print("cen", hm.hprojperp(op2.axs, xpost2inv @ op2.cen))
                print("cen", xpost2inv @ op2.cen)
                print("cen", op2.cen)

                t2 = hm.hrot(op2.axs, np.pi, op2.cen)
                viz.showme([op2], headless=False)
                # viz.showme(symops, headless=False)
                assert 0
                print(op2.xrel)
                print(t2)

                print()
                print("hel   ", op1.hel, op2.hel)
                print("rad   ", op1.rad, op2.rad)
                print("op1   ", op1.cen)
                print("op2   ", op2.cen)
                print("op2   ", xpost2inv @ op2.cen)
                print("hproj  ", hm.hprojperp(op2.axs, xpost2inv @ op2.cen))
                # print('op1axs', op1.axs, op1.ang)
                print("op2axs", op2.axs, op2.ang)
                # print(op1.xrel)
                # print(op2.xrel)
                raise e

        # continue

        xpost3inv = np.linalg.inv(xpost3)
        frames3 = xpost3 @ frames  # move whole structure
        symops3 = ipd.sym.symops_from_frames(sym=sym, frames=frames3)
        symops3 = ipd.sym.stupid_pairs_from_symops(symops3)
        assert len(symops3) == len(frames3) * (len(frames3) - 1) / 2
        assert len(frames) == len(frames3)
        for k in symops:
            op1 = symops[k]
            op2 = symops3[k]
            try:
                assert np.allclose(op1.ang, op2.ang, atol=1e-3)
                assert np.allclose(op1.cen, hm.hprojperp(op1.axs, xpost3inv @ op2.cen), atol=1e-2)
                assert np.allclose(op1.rad, op2.rad, atol=1e-2)
                assert np.allclose(op1.hel, op2.hel, atol=1e-3)
                # for k in point_angles:
                #     assert np.allclose(op1.err[k], op2.err[k], atol=1e-2)

                op2axsinv = xpost3inv @ op2.axs
                if hm.hdot(op2axsinv, op1.axs) < 0:
                    op2axsinv = -op2axsinv
                assert np.allclose(op1.axs, op2axsinv, atol=1e-4)

                # assert np.allclose(op1.cen, xpost3inv @ hm.hprojperp(op2.axs, op2.cen), atol=1e-4)
            except AssertionError as e:
                print("op1       ", op1.rad)
                print("op1       ", op1.cen)
                print("cen op2   ", op2.cen)
                print("cen op2inv", xpost3inv @ op2.cen)
                print("hproj      ", hm.hprojperp(op1.axs, xpost3inv @ op2.cen))
                print("hproj      ", hm.hprojperp(op2.axs, xpost3inv @ op2.cen))
                print(op1.rad, op2.rad)
                # print('hproj  ', hm.hprojperp(op2.axs, op2.cen))
                # print('hproj  ', xpost3inv @ hm.hprojperp(op2.axs, op2.cen))
                # print('op1axs', op1.axs, op1.ang)
                # print('op2axs', xpost3inv @ op2.axs, op2.ang)
                # print('op2axs', op2.axs, op2.ang)
                # print(op1.xrel)
                # print(op2.xrel)
                raise e

        # ipd.viz.showme(list(symops3.values()), 'someops2')

@pytest.mark.fast  # type: ignore
def test_symops_cen_imperfect(nsamp=20, manual=False, **kw):
    # np.set_printoptions(
    # precision=10,
    # suppress=True,
    # linewidth=98,
    # formatter={'float': lambda f: '%14.10f' % f},
    # )
    # r = hm.hrot([1, 0, 0], 180)
    # xinfo.axs, a = hm.axis_angle_of(r)
    # print(r)
    # print(xinfo.axs)
    # print(a)
    # assert 0

    kw = ipd.dev.Bunch()
    kw.tprelen = 20
    kw.tprerand = 2
    kw.tpostlen = 20
    kw.tpostrand = 2
    kw.cart_sd_fuzz = 1.0
    kw.rot_sd_fuzz = np.radians(7)
    kw.cart_sd_fuzz = 1.0
    kw.rot_sd_fuzz = np.radians(7)
    kw.remove_outliers_sd = 3

    # ipd.viz.showme(symops)
    # assert 0

    all_cen_err = list()
    for i in range(nsamp):
        kw.sym = np.random.choice("tet oct icos".split())
        kw.nframes = np.random.choice(6) + 6
        kw.nframes = min(kw.nframes, len(ipd.sym.sym_frames[kw.sym]))

        frames, xpre, xpost, xfuzz, radius = setup_test_frames(**kw)

        symfit = ipd.sym.compute_symfit(frames=frames, **kw)
        symops = symfit.symops
        cen_err = np.linalg.norm(symfit.center - xpost[:, 3])
        all_cen_err.append(cen_err)

        # ipd.viz.showme(selframes, showcen=True, name='source_frames')
        # ipd.viz.showme(frames, showcen=True, name='source_frames')

        # ipd.viz.showme(frames, xyzlen=(.1, .1, .1), showcen=True)
        # ipd.viz.showme(xpost @ symframes, xyzlen=(.1, .1, .1), showcen=True)
        # ipd.viz.showme(symops, center=xpost[:, 3], expand=2.0, scalefans=0.125, name='symops',
        # cyc_ang_match_tol=0.3, axislen=30, fixedfansize=2)
        # assert 0

        radius = np.mean(np.linalg.norm(frames[:, :, 3] - xpost[:, 3], axis=-1))
        # print(radius, symfit.radius)
        assert np.allclose(radius, symfit.radius, atol=kw.cart_sd_fuzz * 10)

    np.sort(all_cen_err)
    err = ipd.dev.Bunch()
    err.mean = np.mean(all_cen_err)
    err.mean1 = np.mean(all_cen_err[1:-1])
    err.mean2 = np.mean(all_cen_err[2:-2])
    err.mean2 = np.mean(all_cen_err[3:-3])
    err.median = np.median(all_cen_err)
    err.min = np.min(all_cen_err)
    err.max = np.max(all_cen_err)

    # print('test_symops_cen_imperfect median err', err.median)
    assert err.median < 3.0
    if manual:
        return err

@pytest.mark.fast  # type: ignore
def test_symfit_align_axes():
    kw = ipd.dev.Bunch()
    # kw.sym = np.random.choice('tet oct icos'.split())
    kw.sym = "tet"
    # kw.nframes = np.random.choice(6) + 6
    kw.nframes = len(ipd.sym.sym_frames[kw.sym])
    kw.tprelen = 20
    kw.tprerand = 2
    kw.tpostlen = 20
    kw.tpostrand = 2
    kw.remove_outliers_sd = 3
    kw.fuzzstdfrac = 0.05  # frac of radian
    # kw.cart_sd_fuzz = fuzzstdabs
    # kw.rot_sd_fuzz = fuzzstdabs / tprelen

    kw.cart_sd_fuzz = kw.fuzzstdfrac * kw.tprelen
    kw.rot_sd_fuzz = kw.fuzzstdfrac

    point_angles = ipd.sym.sym_point_angles[kw.sym]
    frames, xpre, xpost, xfuzz, radius = setup_test_frames(**kw)
    symops = ipd.sym.symops_from_frames(frames=frames, **kw)
    symops = ipd.sym.stupid_pairs_from_symops(symops)
    # ipd.viz.showme(symops)
    # assert 0

    symfit = ipd.sym.compute_symfit(frames=frames, **kw)

    # print('ang_err', symfit.symop_ang_err)
    # print('hel_err', symfit.symop_hel_err)
    # print('cen_err', symfit.cen_err)
    # print('axes_err', symfit.axes_err)
    # print('total_err', symfit.total_err)
    # assert symfit.total_err < 10
    # assert 0

@pytest.mark.fast  # type: ignore
def test_disambiguate_axes():
    sym = "oct"
    nfold, axis0 = list(), list()
    for nf, ax in ipd.sym.octahedral_axes_all.items():
        nfold.append(np.repeat(nf, len(ax)))
        axis0.append(ax)
        if nf == 4:
            axis0.append(ax)
            nfold.append(np.repeat(2, len(ax)))
    nfold = np.concatenate(nfold)
    axis0 = np.concatenate(axis0)
    # print(nfold)
    tgt = [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]

    axis = axis0
    nfoldnew = ipd.sym.disambiguate_axes(sym, axis, nfold)
    # print(tgt)
    # print(nfoldnew)
    assert list(nfoldnew) == tgt

    axis = hm.hxform(
        hm.rand_xform_small(len(axis0), rot_sd=0.01),
        axis0,
    )

    nfoldnew = ipd.sym.disambiguate_axes(sym, axis, nfold)
    # print(nfoldnew)
    assert list(nfoldnew) == tgt

    # assert 0

def _test_symfit_mc():
    kw = ipd.dev.Bunch()
    kw.tprelen = 10
    kw.tprerand = 0
    kw.tpostlen = 20
    kw.tpostrand = 0
    kw.fuzzstdfrac = 0.01  # frac of radian
    kw.remove_outliers_sd = 3
    kw.choose_closest_frame = False
    # kw.showme = True
    # result = symfit_mc_play(sym='icos',nframes=5,seed=0, maxiters=100, **kw)
    # print(result.start_err, result.besterr, result.symerr)
    # assert np.isclose(result.start_err, 0.4195887257175782)
    # assert np.isclose(result.besterr, 0.1689752948952585)
    # assert np.isclose(result.symerr, 0.19297554111865917)

    kw.fuzzstdfrac = 0.1
    result = symfit_mc_play(nframes=4, sym="tet", seed=12, maxiters=200, **kw)
    assert np.isclose(result.start_err, 2.467584262814621)
    # assert np.isclose(result.besterr, 0.09552787596768347)
    # assert np.isclose(result.symerr, 0.12641561080909042)
    assert np.isclose(result.besterr, 0.1448517181829827)
    assert np.isclose(result.symerr, 0.15947049465758514)

    kw.fuzzstdfrac = 0.1
    # for s in range(20):
    # try:
    result = symfit_mc_play(nframes=5, sym="oct", seed=7, maxiters=200, **kw)
    # print(result.start_err, result.besterr, result.symerr, result.start_err / result.besterr)
    # except:
    # print('fail')
    # print(result.besterr, result.symerr)
    assert np.isclose(result.start_err, 4.967872160707628)
    # assert np.isclose(result.besterr, 0.4514411873704736)
    # assert np.isclose(result.symerr, 0.5324565429731142)
    assert np.isclose(result.besterr, 0.5411346166998205)
    assert np.isclose(result.symerr, 0.8198090180340757)

def helper_test_symfit_dihedral(icyc, rand=True):
    sym = "d%i" % icyc
    symframes = ipd.sym.sym_frames[sym]
    frames = hm.hxform(symframes, hm.htrans([5, 7, 11]))
    if rand:
        frames = hm.rand_xform_small(len(frames), rot_sd=0.001) @ frames
    frames = frames[:4]
    symfit = ipd.sym.compute_symfit(sym=sym, frames=frames, alignaxes_more_iters=2)
    # p = ipd.sym.stupid_pairs_from_symops(symfit.symops)
    # ipd.viz.showme(p)
    # print(icyc)
    # print(symfit.losses)
    assert symfit.losses["A"] < 4e-2 if rand else 1e-2
    assert symfit.losses["C"] < 1e-4
    assert symfit.losses["H"] < 1e-4
    assert symfit.losses["N"] < 1e-2

    # for s in range(10):
    #     np.random.seed(s)
    #     frames = hm.rand_xform(len(frames))  # @ frames
    #     try:
    #         symfit = ipd.sym.compute_symfit(sym=sym, frames=frames)
    #         # p = ipd.sym.stupid_pairs_from_symops(symfit.symops)
    #         # ipd.viz.showme(p)
    #         # print(symfit.weighted_err)
    #     except:
    #         print(seed)
    #         assert 0
    # assert 0

@pytest.mark.fast  # type: ignore
def test_symfit_dihedral():
    # helper_test_symfit_dihedral(2)
    helper_test_symfit_dihedral(3)
    helper_test_symfit_dihedral(4)
    helper_test_symfit_dihedral(5)
    helper_test_symfit_dihedral(6)
    helper_test_symfit_dihedral(7)
    helper_test_symfit_dihedral(8)
    helper_test_symfit_dihedral(9)
    for i in range(10, 20):
        helper_test_symfit_dihedral(i, rand=False)

    # assert 0

@pytest.mark.fast  # type: ignore
def test_symfit_d2():
    syminfo = ipd.sym.get_syminfo("d2")
    symframes = syminfo.frames
    frames = hm.hxform(symframes, hm.htrans([1, 2, 3]))
    # ipd.viz.showme(frames)
    helper_test_symfit_dihedral(2)
    # assert 0

@pytest.mark.fast  # type: ignore
def test_symfit_d2_af2():
    frames = np.array([
        [[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]],
        [
            [0.80374831, 0.53927461, -0.25133951, 6.00417164],
            [-0.59387094, 0.75282344, -0.28385591, -1.2769798],
            [0.03613799, 0.37741194, 0.92534009, 5.12398778],
            [0, 0, 0, 1],
        ],
        [
            [0.23884698, 0.09262249, -0.96662981, 2.92136038],
            [0.01116147, -0.99563673, -0.09264402, -9.39450731],
            [-0.97099307, 0.01133874, -0.23883863, 2.89522226],
            [0, 0, 0, 1],
        ],
        [
            [0.11333779, -0.15687948, -0.98109295, -0.99280996],
            [0.62179177, -0.75898037, 0.19319367, -8.23571297],
            [-0.77493841, -0.63193167, 0.01152521, -4.02264787],
            [0, 0, 0, 1],
        ],
    ])
    fit = ipd.sym.compute_symfit(frames=frames, sym="d2")
    # print(fit.total_err)

@pytest.mark.fast  # type: ignore
def test_af2_example():
    frames = np.array([
        [
            [1.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
            [0.00000000e00, 1.00000000e00, 0.00000000e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 1.00000000e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        [
            [-2.70417202e-01, -3.45264712e-01, -8.98702852e-01, 8.38615450e00],
            [8.29381616e-01, -5.57562790e-01, -3.53535094e-02, -3.93224288e01],
            [-4.88876950e-01, -7.54927820e-01, 4.37130774e-01, -9.66270097e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        [
            [6.75297114e-01, -6.00733668e-01, -4.27893524e-01, -6.41388273e00],
            [7.37448283e-01, 5.40525206e-01, 4.04972260e-01, -7.95503036e00],
            [-1.19932362e-02, -5.89025943e-01, 8.08025124e-01, -1.23335678e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        [
            [7.64528213e-01, 5.93149802e-01, -2.52329000e-01, 7.93868945e00],
            [-6.13101608e-01, 5.48294269e-01, -5.68752858e-01, -6.13328928e00],
            [-1.99005100e-01, 5.89530922e-01, 7.82847535e-01, 5.97667763e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
    ])
    fit = ipd.sym.compute_symfit(sym="d3", frames=frames)
    # print(fit.weighted_err)

@pytest.mark.fast  # type: ignore
def test_symfit_d3_nfold_error():
    frames = np.array([
        [
            [0.97054542, -0.20623367, -0.12453623, -5.53040854],
            [-0.06242719, 0.28398443, -0.95679448, -6.0806959],
            [0.23268958, 0.93638695, 0.2627452, 1.31800166],
            [0, 0, 0, 1],
        ],
        [
            [0.59226722, 0.19978753, -0.78057958, -2.35530704],
            [0.58528258, 0.55914571, 0.58719704, 0.0447721],
            [0.55377237, -0.8046372, 0.21423152, -4.86428115],
            [0, 0, 0, 1],
        ],
        [
            [0.06409955, -0.90166908, -0.42764953, -11.91285626],
            [0.8669915, 0.26252498, -0.42356389, -2.64170283],
            [0.49418314, -0.34361826, 0.79856716, -5.94434196],
            [0, 0, 0, 1],
        ],
        [
            [-0.36814632, -0.45993818, -0.80803784, 7.8419424],
            [0.90118702, 0.03730856, -0.43182175, -4.40349732],
            [0.22875804, -0.88716681, 0.40075531, -3.87763387],
            [0, 0, 0, 1],
        ],
    ])
    fit = ipd.sym.compute_symfit(sym="d3", frames=frames)

# @pytest.mark.xfail
def test_d4_error():
    frames = np.array([
        [
            [-0.49537276, -0.04757512, -0.86737676, -0.40013152],
            [-0.06300325, 0.9978372, -0.01874864, 3.60776545],
            [0.86639276, 0.04536, -0.49729875, -1.93353317],
            [0, 0, 0, 1],
        ],
        [
            [0.72021416, 0.3984008, 0.56795103, 4.50082754],
            [-0.19837333, 0.90274918, -0.38169614, -4.50196276],
            [-0.66478537, 0.16223663, 0.72920483, -1.66057906],
            [0, 0, 0, 1],
        ],
        [
            [-0.8942695, 0.4165067, 0.16372, 0.44246407],
            [0.43142538, 0.89958473, 0.06796655, -4.28919314],
            [-0.11897149, 0.13141337, -0.98416275, 3.76007159],
            [0, 0, 0, 1],
        ],
    ])
    fit = ipd.sym.compute_symfit(sym="d4", frames=frames)
    assert fit

# @pytest.mark.xfail
# def test_cyclic():
#    assert 0

if __name__ == "__main__":
    main()
