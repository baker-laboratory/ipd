import pytest
import numpy as np

import ipd
from ipd.homog import hgeom as h
from ipd.data.tests.numeric import sym_detect_test_frames

TEST_PDBS = ['7abl', '1dxh', '1n0e', '1wa3', '1a2n', '1n0e', '2tbv', '1bfr', '1g5q']
ALLSYMS = ['T', 'O', 'I'] + ['C%i' % i for i in range(2, 13)] + ['D%i' % i for i in range(2, 13)]

config_test = ipd.Bunch(
    re_only=[
        # r'test_sym_detect_frames_ideal_[^_]+$',
        # r'test_sym_detect_frames_ideal_xformed_[^_]+$',
        # r'.*symelem.*'
        # r'test.*_D\d+',
        r'test.*noised.*'
    ],
    only=[],
    # re_exclude=['test_sym_detect_1g5q'],
    exclude=[],
)

def main():
    ipd.tests.maintest(namespace=globals(), config=config_test, verbose=1)

def helper_test_frames(frames, symid, tol=None, origin=np.eye(4), ideal=False, **kw):
    if ideal: tol = ipd.Tolerances(tol, **ipd.sym.symdetect_ideal_tolerances)
    else: tol = ipd.Tolerances(tol, **ipd.sym.symdetect_default_tolerances)
    sinfo = ipd.sym.syminfo_from_frames(frames, tol=tol, **kw)
    # print(sinfo)
    se = sinfo.symelem
    assert sinfo.symid == symid, f'{symid=}, {sinfo.symid=}'
    assert all(se.hel < tol.helical_shift)
    cendist = h.point_line_dist_pa(sinfo.symcen, se.cen, se.axis)
    assert cendist.max() < tol.isect
    ref = {k: v for k, v in ipd.sym.axes(symid).items() if isinstance(k, int)}

    assert sinfo.is_cyclic == (sinfo.symid[0] == 'C')
    assert sinfo.is_dihedral == (sinfo.symid[0] == 'D')
    if sinfo.is_cyclic:
        ic(h.line_angle(h.xform(origin, [0, 0, 1, 0]), sinfo.axis[0]), tol.line_angle)
        assert h.line_angle(h.xform(origin, [0, 0, 1, 0]), sinfo.axis[0]) < tol.line_angle
    elif sinfo.symid == 'D2':
        pass
    else:
        for nf, ax in ipd.sym.axes(sinfo.symid).items():
            if isinstance(nf, str): continue
            if nf == 2 and sinfo.is_dihedral and not sinfo.order % 4:
                angs = h.line_angle(sinfo.nfaxis[nf], h.xform(origin, ax))
                ic(angs)

                ic(h.xform(h.inv(origin), sinfo.axis))
                assert np.sum(np.abs(angs) < tol.line_angle) * 2 == angs.size
            else:
                if ideal:
                    assert np.allclose(sinfo.nfaxis[nf], h.xform(origin, ax))

    return sinfo

for symid0 in ALLSYMS:

    def make_ideal_frame_funcs(symid):

        def func_ideal():
            helper_test_frames(ipd.sym.frames(symid), symid, ideal=True)

        def func_ideal_xformed():
            origin = h.rand(cart_sd=44)
            xframes = h.xform(origin, ipd.sym.frames(symid))
            sinfo = helper_test_frames(xframes, symid, origin=origin, ideal=True)

        def func_noised():
            frames = ipd.sym.frames(symid)
            noise = h.randsmall(len(frames), rot_sd=0.01, cart_sd=1)
            # frames = h.xform(frames, noise)
            assert h.valid44(frames)
            sinfo = helper_test_frames(frames, symid, ideal=False)

        return func_ideal, func_ideal_xformed, func_noised

    ideal, xformed, noised = make_ideal_frame_funcs(symid0)

    ideal.__name__ = ideal.__qualname__ = f'test_sym_detect_frames_ideal_{symid0}'
    xformed.__name__ = xformed.__qualname__ = f'test_sym_detect_frames_ideal_xformed_{symid0}'
    noised.__name__ = noised.__qualname__ = f'test_sym_detect_frames_noised_{symid0}'
    globals()[f'test_sym_detect_frames_ideal_{symid0}'] = ideal
    globals()[f'test_sym_detect_frames_ideal_xformed_{symid0}'] = xformed
    globals()[f'test_sym_detect_frames_noised_{symid0}'] = noised

test_sym_detect_frames_ideal_D12 = pytest.mark.xfail(test_sym_detect_frames_ideal_D12)
test_sym_detect_frames_ideal_xformed_D12 = pytest.mark.xfail(test_sym_detect_frames_ideal_xformed_D12)
test_sym_detect_frames_noised_D12 = pytest.mark.xfail(test_sym_detect_frames_noised_D12)

def make_pdb_testfunc(pdbcode):

    @pytest.mark.fast
    def pdb_test_func():
        pytest.importorskip('biotite')
        atoms = ipd.atom.load(ipd.dev.package_testcif_path(pdbcode), biounit='largest', het=False)
        tol = ipd.dev.Tolerances(**(ipd.sym.symdetect_default_tolerances | dict(
            default=1e-1,
            angle=9e-1,
            helical_shift=4,
            isect=6,
            dot_norm=0.07,
            misc_lineuniq=0.2,
            rms_fit=3,
            nfold=0.2,
        )))
        syminfo = ipd.sym.syminfo_from_atomslist(atoms, tol=tol)
        infersym = None
        ic(syminfo.frames.shape)
        if syminfo.order == 1: infersym = 'C1'
        elif syminfo.order == 12: infersym = 'T'
        elif syminfo.order == 24: infersym = 'O'
        elif syminfo.order == 60: infersym = 'I'
        elif syminfo.pseudo_order == 60: infersym = 'I'
        elif syminfo.pseudo_order == 24: infersym = 'O'
        elif syminfo.pseudo_order == 12: infersym = 'T'
        elif syminfo.symid[0] == 'C': infersym = f'C{syminfo.order}'
        elif syminfo.symid[0] == 'D': infersym = f'C{syminfo.order/2}'
        if infersym != syminfo.symid:
            ic(infersym, syminfo, syminfo.tol_checks)
        assert infersym == syminfo.symid, f'{infersym=} != {syminfo.symid}'
        infer_t = syminfo.pseudo_order // syminfo.order
        err = f'T number mismatch {syminfo.t_number=}, {infer_t=} {syminfo.pseudo_order=} {syminfo.order=}'
        assert syminfo.t_number == infer_t, err

    pdb_test_func.__name__ = pdb_test_func.__qualname__ = f'test_sym_detect_pdb_{pdbcode}'
    # pdb_test_func = ipd.dev.timed(pdb_test_func)
    return pdb_test_func

ipd.pdb.download_test_pdbs(TEST_PDBS)
for code in TEST_PDBS:
    globals()[f'test_sym_detect_{code}'] = make_pdb_testfunc(code)

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
    # ic(syminfo.tol_checks)
    assert syminfo.symid == 'C3'

@ipd.dev.timed
def test_syminfo_from_frames_examples():
    for name, (symid, frames) in sym_detect_test_frames.items():
        tol = ipd.Tolerances(1e-1, angle=1e-2, helical_shift=1, isect=1, dot_norm=0.04, misc_lineuniq=1, nfold=0.2)
        helper_test_frames(frames, symid, tol=tol)

@pytest.mark.fast
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
    for nf, se in symelem.groupby('nfold'):
        assert len(se.nfold) == 1
        refaxis = ref.axis[ref.nfold == nf].data
        assert h.allclose(se.axis, h.xform(pert, refaxis)) or h.allclose(-se.axis, h.xform(pert, refaxis))
        assert h.allclose(se.cen, h.xform(pert, ref.cen[ref.nfold == nf].data))

@pytest.mark.fast
def test_symelems_from_frames_D2n(symid='D4'):
    frames = ipd.sym.frames(symid)
    refse = ipd.sym.symelems_from_frames(frames)
    se = ipd.sym.symelems_from_frames(frames)
    ipd.dev.print_table(se)
    assert h.allclose(refse, se)
    uniq, _, _, _ = h.symmetrically_unique_lines(se.axis.data, se.cen.data)
    assert len(uniq) == len(se.axis)

if __name__ == '__main__':
    main()
