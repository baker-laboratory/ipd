import pytest
import numpy as np

import ipd
from ipd.homog import hgeom as h
from ipd.data.tests.numeric import sym_detect_test_frames

TEST_PDBS = ['7abl', '1dxh', '1n0e', '1wa3', '1a2n', '1n0e', '2tbv', '1bfr', '1g5q']

just = [
    # 'test_sym_detect_1dxh'
    # 'test_sym_detect_7abl',
    # 'test_sym_detect_1g5q',
]
exclude = [
    'test_sym_detect_1g5q',
]

def main():
    ipd.tests.maintest(namespace=globals(), just=just, exclude=exclude)

def make_pdb_testfunc(pdbcode):

    @pytest.mark.fast
    def pdb_test_func():
        pytest.importorskip('biotite')
        atoms = ipd.atom.load(ipd.dev.package_testcif_path(pdbcode), biounit='largest', het=False)
        tol = ipd.dev.Tolerances(**(ipd.sym.symdetect_default_tolerances | dict(
            default=1e-1,
            angle=9e-2,
            helical_shift=4,
            isect=2,
            dot_norm=0.04,
            misc_lineuniq=0.2,
            rms_fit=3,
            nfold=0.2,
        )))
        syminfo = ipd.sym.syminfo_from_atomslist(atoms, tol=tol)
        infersym = None
        if syminfo.order == 12: infersym = 'T'
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

    pdb_test_func.__name__ = pdb_test_func.__qualname__ = f'test_sym_detect_{pdbcode}'
    pdb_test_func = ipd.dev.timed(pdb_test_func)
    return pdb_test_func

ipd.pdb.download_test_pdbs(TEST_PDBS)
for code in TEST_PDBS:
    if f'test_sym_detect_{code}' in exclude: continue
    globals()[f'test_sym_detect_{code}'] = make_pdb_testfunc(code)

@ipd.dev.timed
@pytest.mark.fast
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

def helper_test_frames(frames, symid, tol=None, **kw):
    tol = ipd.Tolerances(tol, **kw)
    sinfo = ipd.sym.syminfo_from_frames(frames, tol=tol, **kw)
    # print(sinfo)
    se = sinfo.symelem
    assert sinfo.symid == symid, f'{symid=}, {sinfo.symid=}'
    assert all(se.hel < tol.helical_shift)
    cendist = h.point_line_dist_pa(sinfo.symcen, se.cen, se.axis)
    assert cendist.max() < tol.isect
    ref = {k: v for k, v in ipd.sym.axes(symid).items() if isinstance(k, int)}
    print('redo me')
    # for nf in ([] if symid in 'D2 '.split() else ref):
    # assert h.allclose(ref[nf], se.axis[se.nfold == nf][0])

@ipd.dev.timed
@pytest.mark.fast
def test_syminfo_from_frames_basic():
    allsyms = ['T', 'O', 'I'] + ['C%i' % i for i in range(2, 13)] + ['D%i' % i for i in range(2, 13)]
    for name, (symid, frames) in sym_detect_test_frames.items():
        tol = ipd.Tolerances(1e-1, angle=1e-2, helical_shift=1, isect=1, dot_norm=0.04, misc_lineuniq=1, nfold=0.2)
        helper_test_frames(frames, symid, tol=tol)
    for symid in allsyms:
        frames = ipd.sym.frames(symid)
        helper_test_frames(frames, symid)

@ipd.dev.timed
@pytest.mark.fast
def test_symaxis_closest_to():
    frames0 = ipd.sym.frames('oct')
    testaxes0 = [[1, 0, 0], [0, -1, 0], [1, 1, 0], [1, 1, 0], [-1, 0, -1], [1, 1, 1], [-1, -1, 1], [1, 1, 1]]
    golden0 = [[0.0000, 0.0000, 1.0000, 0.0000], [0.0000, 0.0000, 1.0000, 0.0000], [0.7071, 0.0000, 0.7071, 0.0000],
               [0.7071, 0.0000, 0.7071, 0.0000], [0.7071, 0.0000, 0.7071, 0.0000], [0.5774, 0.5774, 0.5774, 0.0000],
               [0.5774, 0.5774, 0.5774, 0.0000], [0.5774, 0.5774, 0.5774, 0.0000]]
    closeaxes, _which = ipd.sym.depricated_symaxis_closest_to(frames0, testaxes0)
    assert h.allclose(closeaxes, golden0, atol=1e-4)

    randrot = h.rand(cart_sd=0, dtype=np.float64)
    frames = h.xform(randrot, frames0)
    testaxes = h.xform(randrot, testaxes0)
    golden = h.xform(randrot, golden0)
    closeaxes, _which = ipd.sym.depricated_symaxis_closest_to(frames, testaxes)
    assert h.allclose(closeaxes, golden, atol=1e-4)

    randtrans = h.randtrans(cart_sd=1, dtype=np.float64)
    frames = h.xform(randtrans, frames0)
    testaxes = h.xformvec(randtrans, testaxes0)
    golden = h.xform(randtrans, golden0)
    closeaxes, _which = ipd.sym.depricated_symaxis_closest_to(frames, testaxes)
    assert h.allclose(closeaxes, golden, atol=1e-4)

@ipd.dev.timed
@pytest.mark.fast
def test_symelems_from_frames():
    frames0 = ipd.sym.frames('oct')
    ref = ipd.sym.symelems_from_frames(frames0)
    # ic(set(ref.nfold.data))
    assert set(ref.nfold.data) == {2, 3, 4}
    pert = h.rand(cart_sd=0, dtype=np.float64)
    # print(repr(pert))
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

if __name__ == '__main__':
    main()
