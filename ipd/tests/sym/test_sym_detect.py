import pytest
import numpy as np

import ipd
from ipd.homog import hgeom as h

TEST_PDBS = ['1dxh', '1g5q', '1n0e', '1wa3', '1a2n', '1n0e', '2tbv', '1bfr']

def main():
    ipd.tests.maintest(namespace=globals())
    # ipd.tests.maintest(namespace=globals(), just='test_sym_detect_1wa3')

def make_pdb_testfunc(pdbcode):

    @pytest.mark.fast
    def pdb_test_func():
        pytest.importorskip('biotite')
        atoms = ipd.pdb.readatoms(ipd.dev.package_testcif_path(pdbcode), biounit='largest', het=False)
        tol = ipd.dev.Tolerances(
            tol=1e-1,
            # angtol=1e-2,
            heltol=4,
            isectol=2,
            dottol=0.04,
            extratol=0.2,
            rmstol=3,
            nftol=0.2,
        )
        syminfo = ipd.sym.syminfo_from_atomslist(atoms, tol=tol)
        infersym = None
        if syminfo.order == 12: infersym = 'T'
        if syminfo.order == 24: infersym = 'O'
        if syminfo.order == 60: infersym = 'I'
        if syminfo.symid[0] == 'C': infersym = f'C{syminfo.order}'
        if syminfo.symid[0] == 'D': infersym = f'C{syminfo.order/2}'
        if infersym != syminfo.symid:
            ic(infersym, syminfo, syminfo.tol_checks)
        assert infersym == syminfo.symid, f'{infersym=} != {syminfo.symid}'

    pdb_test_func.__name__ = pdb_test_func.__qualname__ = f'test_sym_detect_{pdbcode}'
    pdb_test_func = ipd.dev.timed(pdb_test_func)
    return pdb_test_func

ipd.pdb.download_test_pdbs(TEST_PDBS)
for code in TEST_PDBS:
    globals()[f'test_sym_detect_{code}'] = make_pdb_testfunc(code)

@ipd.dev.timed
@pytest.mark.fast
def test_syminfo_from_atomslist():
    pytest.importorskip('biotite')
    atoms = ipd.pdb.readatoms(ipd.dev.package_testdata_path('pdb/L2_D1_C3_Apo.pdb'), chainlist=True)
    tol = ipd.dev.Tolerances(tol=1e-1, angtol=1e-2, heltol=1, isectol=1, dottol=0.04, extratol=1, rmstol=3, nftol=0.2)
    syminfo = ipd.sym.syminfo_from_atomslist(atoms, tol=tol)
    # ic(syminfo.tol_checks)
    assert syminfo.symid == 'C3'

def helper_test_frames(frames, symid, **kw):
    tol = ipd.dev.Tolerances(**kw)
    sinfo = ipd.sym.syminfo_from_frames(frames, **kw)
    # print(sinfo)
    se = sinfo.symelem
    assert sinfo.symid == symid, f'{symid=}, {sinfo.symid=}'
    assert all(se.hel < tol.heltol)
    cendist = h.point_line_dist_pa(sinfo.symcen, se.cen, se.axis)
    assert cendist.max() < tol.isectol
    ref = {k: v for k, v in ipd.sym.axes(symid).items() if isinstance(k, int)}
    print('redo me')
    # for nf in ([] if symid in 'D2 '.split() else ref):
    # assert h.allclose(ref[nf], se.axis[se.nfold == nf][0])

@ipd.dev.timed
@pytest.mark.fast
def test_syminfo_from_frames_basic():
    allsyms = ['T', 'O', 'I'] + ['C%i' % i for i in range(2, 13)] + ['D%i' % i for i in range(2, 13)]
    for name, (symid, frames) in test_frames.items():
        helper_test_frames(frames, symid, tol=1e-1, angtol=1e-2, heltol=1, isectol=1, dottol=0.04, extratol=1, nftol=0.2)
    for symid in allsyms:
        frames = ipd.sym.frames(symid)
        helper_test_frames(frames, symid, tol=1e-4)

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

test_frames = {
    '1bfr': ('O',
             np.array([[[1.00000000e+00, -2.77913159e-09, 5.57832394e-08, -2.86102295e-06],
                        [-3.27759615e-08, 1.00000012e+00, -3.96591275e-08, 2.38418579e-06],
                        [-1.90153671e-10, -2.47708272e-08, 9.99999940e-01, 3.81469727e-06],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[-8.03211555e-02, -5.51036537e-01, -8.30606580e-01, 5.60334473e+01],
                        [-5.52210450e-01, -6.69135630e-01, 4.97314036e-01, 1.55448914e-02],
                        [-8.29826713e-01, 4.98614281e-01, -2.50542283e-01, 6.21141167e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[9.15892899e-01, -1.74434677e-01, 3.61541569e-01, -5.68944550e+00],
                        [-3.53332013e-01, 7.71367550e-02, 9.32312369e-01, -1.24002495e+01],
                        [-1.90515712e-01, -9.81642723e-01, 9.01557226e-03, 2.27255898e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[-2.77258784e-01, -2.07700357e-01, -9.38076913e-01, 6.80853424e+01],
                        [-7.87873328e-01, 6.07948244e-01, 9.82582867e-02, 2.57122765e+01],
                        [5.49893975e-01, 7.66328633e-01, -3.32200497e-01, 1.25950708e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[8.32001925e-01, -5.27361810e-01, 1.72226280e-01, -5.68120956e-01],
                        [-5.27300239e-01, -8.48210692e-01, -4.99287322e-02, 1.00311375e+01],
                        [1.72414735e-01, -4.92742695e-02, -9.83791173e-01, 3.62318726e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[8.14674422e-02, -1.97131354e-02, -9.96481299e-01, 5.67412796e+01],
                        [5.52176058e-01, 8.33234966e-01, 2.86593474e-02, -2.26296368e+01],
                        [8.29737842e-01, -5.52567959e-01, 7.87666142e-02, -1.52151642e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[9.16279793e-01, -3.52025270e-01, -1.91074267e-01, 5.19862080e+00],
                        [-1.76030308e-01, 7.45879188e-02, -9.81554747e-01, 2.24169235e+01],
                        [3.59784067e-01, 9.33013678e-01, 6.37618871e-03, 1.35284348e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[2.79355526e-01, -3.64628673e-01, -8.88260484e-01, 4.46669922e+01],
                        [7.87467539e-01, -4.42329556e-01, 4.29231435e-01, -4.84140968e+01],
                        [-5.49413919e-01, -8.19384277e-01, 1.63565949e-01, 3.40988655e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[-2.80315161e-01, -9.59895015e-01, -5.00841951e-03, 4.34995918e+01],
                        [3.66856068e-01, -1.02307342e-01, -9.24635231e-01, -7.40398407e-01],
                        [8.87040198e-01, -2.61026740e-01, 3.80821764e-01, -2.21888180e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[5.56733131e-01, 7.94268072e-01, -2.43282288e-01, 2.74665565e+01],
                        [7.94316411e-01, -5.94727933e-01, -1.23934902e-01, -3.76184692e+01],
                        [-2.43124276e-01, -1.24244489e-01, -9.62005138e-01, 5.11654892e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[-2.73366511e-01, -7.91507185e-01, 5.46615064e-01, 3.25158577e+01],
                        [-2.08575025e-01, 6.03514791e-01, 7.69588590e-01, -1.11424942e+01],
                        [-9.39025104e-01, 9.63693559e-02, -3.30069333e-01, 6.54917984e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[5.43720927e-03, 9.52809453e-01, -3.03520292e-01, 5.11385498e+01],
                        [-9.55138803e-01, 9.48261619e-02, 2.80567408e-01, 2.49820824e+01],
                        [2.96109021e-01, 2.88378447e-01, 9.10580814e-01, -7.62539673e+00],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[-9.99984503e-01, -1.54577044e-03, -5.36618801e-03, 7.72662506e+01],
                        [-1.76952512e-03, -8.23701441e-01, 5.67021012e-01, -2.33117409e+01],
                        [-5.29650971e-03, 5.67021668e-01, 8.23685646e-01, 7.46256256e+00],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[8.56213719e-02, 5.49385786e-01, 8.31170321e-01, 2.09002991e+01],
                        [-1.55323073e-02, 8.34868550e-01, -5.50230145e-01, 1.17962265e+01],
                        [-9.96206462e-01, 3.42014506e-02, 8.00158158e-02, 5.83373604e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[-9.14358377e-01, 1.80472985e-01, -3.62461150e-01, 8.28019104e+01],
                        [1.79931581e-01, -6.20827675e-01, -7.63018668e-01, -2.09243774e-01],
                        [-3.62730265e-01, -7.62890875e-01, 5.35186231e-01, 1.92128296e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[2.74564773e-01, 2.02354550e-01, 9.40035641e-01, 9.05607986e+00],
                        [9.61548090e-01, -6.41781837e-02, -2.67032772e-01, -3.75308418e+01],
                        [6.29460718e-03, 9.77207124e-01, -2.12194636e-01, 3.21184769e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[-8.31783652e-01, 5.30234516e-01, -1.64277643e-01, 7.75552368e+01],
                        [5.29807329e-01, 6.70009136e-01, -5.19991755e-01, -1.09079876e+01],
                        [-1.65650010e-01, -5.19556284e-01, -8.38224947e-01, 4.30037079e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[-8.96685869e-02, 2.16332208e-02, 9.95736718e-01, 2.07517967e+01],
                        [1.89622957e-02, -9.99545753e-01, 2.34236624e-02, -1.35094995e+01],
                        [9.95791256e-01, 2.09817737e-02, 8.92177075e-02, -1.83519039e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[-9.17937636e-01, 3.45685303e-01, 1.94658965e-01, 7.19052277e+01],
                        [3.48384291e-01, 4.67656732e-01, 8.12357903e-01, -3.41256714e+01],
                        [1.89786702e-01, 8.13510120e-01, -5.49711287e-01, 3.12201214e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[-2.78692901e-01, 3.71562511e-01, 8.85591149e-01, 3.25664062e+01],
                        [-9.60344732e-01, -9.98433083e-02, -2.60326982e-01, 3.58615913e+01],
                        [-8.30742158e-03, -9.23023641e-01, 3.84653449e-01, 7.72256279e+00],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[2.75250822e-01, 9.61371124e-01, 1.63382164e-03, 3.41260376e+01],
                        [2.03354329e-01, -5.98833226e-02, 9.77272272e-01, -3.55291862e+01],
                        [9.39619243e-01, -2.68662632e-01, -2.11981833e-01, -1.13977871e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[-5.54342389e-01, -7.94145763e-01, 2.49071911e-01, 4.96655846e+01],
                        [-7.94233322e-01, 4.15295273e-01, -4.43534911e-01, 3.65669098e+01],
                        [2.48793185e-01, -4.43691492e-01, -8.60952973e-01, 2.80811043e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[2.80839175e-01, 7.88706303e-01, -5.46874702e-01, 4.44315643e+01],
                        [-3.61260712e-01, -4.41028744e-01, -8.21574271e-01, 2.29533978e+01],
                        [-8.89168382e-01, 4.28294420e-01, 1.61070362e-01, 5.49074783e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                       [[-4.28180769e-03, -9.55183804e-01, 2.95982689e-01, 2.62116013e+01],
                        [9.54320908e-01, 8.45244005e-02, 2.86579043e-01, -4.83275986e+01],
                        [-2.98753291e-01, 2.83689439e-01, 9.11189735e-01, 1.50957756e+01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]],
                      dtype=np.float32))
}
if __name__ == '__main__':
    main()
