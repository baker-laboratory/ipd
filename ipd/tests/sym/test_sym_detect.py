import pytest
import numpy as np

import ipd
from ipd.homog import hgeom as h
from ipd.data.tests.numeric import sym_detect_test_frames

TEST_PDBS = [
    # '6u9d',
    '3sne',
    '1dxh',
    '1n0e',
    '1wa3',
    '1a2n',
    '1n0e',
    '1bfr',
    '1g5q'  #, '3woc', '7abl', '2tbv'
]
ALLSYMS = ['T', 'O', 'I'] + ['C%i' % i for i in range(2, 13)] + ['D%i' % i for i in range(2, 13)]

config_test = ipd.Bunch(
    re_only=[
        # 'test_sym_detect_pdb_1g5q',
        # r'test_chelsea_tube1',
        # r'test_sym_detect_frames_ideal_[^_]+$',
        # r'test_sym_detect_frames_ideal_xformed_[^_]+$',
        # r'test_sym_detect_frames_noised_[^_]+$',
        # r'.*symelem.*'
        # r'test.*_D\d+',
        # r'test.*noised.*'
    ],
    only=[
        # r'test_chelsea_tube1',
        # r'test_syminfo_from_frames_examples',
        # 'test_sym_detect_pdb_1n0e',
        # 'test_sym_detect_pdb_1bfr',
        # 'test_sym_detect_pdb_1g5q',
    ],
    # re_exclude=['test_sym_detect_1g5q'],
    exclude=[],
)

def main():
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
    )

def helper_test_frames(frames, symid, tol=None, origin=None, ideal=False, **kw):
    if ideal: tol = ipd.Tolerances(tol, **(ipd.sym.symdetect_ideal_tolerances | kw))
    else: tol = ipd.Tolerances(tol, **(ipd.sym.symdetect_default_tolerances) | kw)
    sinfo = ipd.sym.detect(frames, tol=tol, **kw)
    # print(sinfo)
    se = sinfo.symelem
    assert sinfo.symid == symid, f'{symid=}, {sinfo.symid=}'
    assert all(se.hel < tol.helical_shift)
    cendist = h.point_line_dist_pa(sinfo.symcen, se.cen, se.axis)
    assert cendist.max() < tol.isect
    ref = {k: v for k, v in ipd.sym.axes(symid).items() if isinstance(k, int)}
    assert sinfo.is_cyclic == (sinfo.symid[0] == 'C')
    assert sinfo.is_dihedral == (sinfo.symid[0] == 'D')
    if origin is not None:
        if sinfo.is_cyclic:
            # ipd.icv(h.line_angle(h.xform(origin, [0, 0, 1, 0]), sinfo.axis[0]), tol.line_angle)
            assert h.line_angle(h.xform(origin, [0, 0, 1, 0]), sinfo.axis[0]) < tol.line_angle
        elif sinfo.symid == 'D2':
            pass
        else:
            for nf, ax in ipd.sym.axes(sinfo.symid).items():
                if isinstance(nf, str): continue
                if nf == 2 and sinfo.is_dihedral and not sinfo.order % 4:
                    angs = h.line_angle(sinfo.nfaxis[nf], h.xform(origin, ax))
                    assert np.sum(np.abs(angs) < tol.line_angle) * 2 == angs.size
                else:
                    # if ideal:
                    ax1, ax2 = h.xform(h.inv(origin), sinfo.nfaxis[nf]), ax[None]
                    cn1, cn2 = h.point([[0, 0, 0]]), h.point([[0, 0, 0]])
                    frm = h.xform(h.inv(origin), frames)
                    diff = h.sym_line_line_diff_pa(cn1, ax1, cn2, ax2, lever=50, frames=frm)
                    ipd.icv(ax1, cn1, ax2, cn2, diff)
                    assert np.all(diff / 50 < tol.angle)

    return sinfo

def make_ideal_test_funcs():
    for symid0 in ALLSYMS:
        order = int(symid0[1:]) if len(symid0) > 1 else dict(T=12, O=24, I=60)[symid0]
        if symid0[0] == 'D': order *= 2
        is_cage = symid0[0] in 'TIO'

        def make_ideal_frame_funcs(symid):

            def func_ideal():
                helper_test_frames(ipd.sym.frames(symid), symid, ideal=True)

            def func_ideal_xformed():
                origin = h.rand(cart_sd=44)
                xframes = h.xform(origin, ipd.sym.frames(symid))
                sinfo = helper_test_frames(xframes, symid, origin=origin, ideal=True)

            def func_noised():
                frames = ipd.sym.frames(symid)
                noise = h.randsmall(len(frames), rot_sd=0.001, cart_sd=0.1)
                frames = h.xform(frames, noise)
                sinfo = helper_test_frames(frames, symid, ideal=False, helical_shift=3, isect=3, cageang=0.1)

            def func_noised_xformed():
                frames = ipd.sym.frames(symid)
                origin = h.rand(cart_sd=44)
                noise = h.randsmall(len(frames), rot_sd=0.001, cart_sd=0.1)
                frames = h.xform(origin, frames, noise)
                sinfo = helper_test_frames(frames, symid, ideal=False, helical_shift=3, isect=3, cageang=0.1)

            return func_ideal, func_ideal_xformed, func_noised, func_noised_xformed

        ideal, xformed, noised, noised_xformed = make_ideal_frame_funcs(symid0)

        ideal.__name__ = ideal.__qualname__ = f'test_sym_detect_frames_ideal_{symid0}'
        xformed.__name__ = xformed.__qualname__ = f'test_sym_detect_frames_ideal_xformed_{symid0}'
        noised.__name__ = noised.__qualname__ = f'test_sym_detect_frames_noised_{symid0}'
        noised_xformed.__name__ = noised.__qualname__ = f'test_sym_detect_frames_noised_xformed_{symid0}'
        globals()[f'test_sym_detect_frames_ideal_{symid0}'] = ideal
        globals()[f'test_sym_detect_frames_ideal_xformed_{symid0}'] = xformed
        if is_cage or order < 6:
            globals()[f'test_sym_detect_frames_noised_{symid0}'] = noised
            globals()[f'test_sym_detect_frames_noised_xformed_{symid0}'] = noised_xformed

def make_pdb_testfunc(pdbcode, path=''):
    tol = ipd.dev.Tolerances(**(ipd.sym.symdetect_default_tolerances | dict(
        default=1e-1,
        angle=0.04,
        helical_shift=4,
        isect=6,
        dot_norm=0.07,
        misc_lineuniq=0.2,
        rms_fit=3,
        nfold=0.2,
    )))
    if pdbcode == '3E47': tol.rms_fit.threshold = 7
    if pdbcode == '3E47': tol.seqmatch.threshold = 0.5

    def pdb_test_func(path=path, tol=tol):
        pytest.importorskip('biotite')
        tol.reset()
        symanno = ipd.pdb.sym_annotation(pdbcode)
        for id, symid in zip(symanno.id, symanno.sym):
            if symid == 'C1': continue
            atoms = ipd.atom.get(pdbcode, assembly=id, het=False, path=path, strict=False)
            sinfo = ipd.sym.detect(atoms, tol=tol, strict=False)
            # ic(sinfo.order, sinfo.pseudo_order)
            if not isinstance(sinfo, ipd.sym.SymInfo):
                sids = [si.symid for si in sinfo]
                assert len(sinfo) == 1, f'mutiple symmetrs detected: {sids}'

            if symid != sinfo.symid:
                if not any([
                        symid == 'C2' and sinfo.is_dihedral,
                        symid == 'C2' and sinfo.symid in 'TIO',
                        symid == 'C3' and sinfo.symid in 'TIO',
                        symid == 'C4' and sinfo.symid in 'O',
                        symid == 'C5' and sinfo.symid in 'I',
                ]):
                    print(sinfo)
                    assert symid == sinfo.symid, f'{symid=} detected as {sinfo.symid}'
            infer_t = sinfo.pseudo_order // sinfo.order
            err = f'T number mismatch {sinfo.t_number=}, {infer_t=} {sinfo.pseudo_order=} {sinfo.order=}'
            assert sinfo.t_number == infer_t, err

    pdb_test_func.__name__ = pdb_test_func.__qualname__ = f'test_sym_detect_pdb_{pdbcode}'
    # pdb_test_func = ipd.dev.timed(pdb_test_func)
    return pdb_test_func

ipd.pdb.download_test_pdbs(TEST_PDBS)
for code in TEST_PDBS:
    globals()[f'test_sym_detect_pdb_{code}'] = make_pdb_testfunc(code)
make_ideal_test_funcs()

def test_syminfo_from_atomslist():
    pytest.importorskip('biotite')
    atoms = ipd.pdb.readatoms(ipd.dev.package_testdata_path('pdb/L2_D1_C3_Apo.pdb'), chainlist=True)
    tol = ipd.dev.Tolerances(tol=1e-1,
                             angle=1e-2,
                             helical_shift=1,
                             isect=1,
                             dot_norm=0.04,
                             misc_lineuniq=1,
                             rms_fit=3,
                             nfold=0.2)
    syminfo = ipd.sym.syminfo_from_atomslist(atoms, tol=tol)
    # ipd.icv(syminfo.tol_checks)
    assert syminfo.symid == 'C3'

@ipd.dev.timed
def test_syminfo_from_frames_examples():
    for name, (symid, frames) in sym_detect_test_frames.items():
        tol = ipd.Tolerances(1e-1,
                             angle=1e-2,
                             helical_shift=1,
                             isect=1,
                             dot_norm=0.04,
                             misc_lineuniq=1,
                             nfold=0.2)
        helper_test_frames(frames, symid, tol=tol)

def test_symelems_from_frames_oct():
    frames0 = ipd.sym.frames('oct')
    ref = ipd.sym.symelems_from_frames(frames0)
    assert set(ref.nfold.data) == {2, 3, 4}
    pert = h.rand(cart_sd=0, dtype=np.float64)
    pert = np.array([[-0.8299, 0.4733, 0.2954, 0.0000], [-0.5578, -0.6963, -0.4516, 0.0000],
                     [-0.0080, -0.5396, 0.8419, 0.0000], [0.0000, 0.0000, 0.0000, 1.0000]],
                    dtype=np.float64)
    frames = h.xform(pert, frames0)
    symelem = ipd.sym.symelems_from_frames(frames)
    for nf, se in symelem.groupby('nfold', convert=np.array):
        assert len(se.ang) == 1
        assert se.axis.shape == (1, 4)

        refaxis = ref.axis[ref.nfold == nf]
        assert h.allclose(se.axis, h.xform(pert, refaxis)) or h.allclose(-se.axis, h.xform(pert, refaxis))
        assert h.allclose(se.cen, h.xform(pert, ref.cen[ref.nfold == nf]))

def test_symelems_from_frames_D2n(symid='D4'):
    frames = ipd.sym.frames(symid)
    refse = ipd.sym.symelems_from_frames(frames)
    se = ipd.sym.symelems_from_frames(frames)
    # ipd.dev.print_table(refse)
    # ipd.dev.print_table(se)
    assert h.allclose(refse, se)
    uniq, _, _, _ = h.unique_symaxes(se.axis, se.cen)
    assert len(uniq) == len(se.axis)

@pytest.mark.xfail
def test_chelsea_tube1():
    pytest.importorskip('biotite')
    atoms = ipd.atom.load(ipd.dev.package_testdata_path('pdb/chelsea_tube_1.pdb.gz'))
    sinfo = ipd.sym.detect(atoms, incomplete=True)
    print(sinfo)

if __name__ == '__main__':
    main()
