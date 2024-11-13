import numpy as np
import pytest

import ipd

def main():
    test_helix_scaling()
    assert 0

    h = ipd.sym.helix.Helix(turns=11, nfold=1, turnsB=1, phase=0.5)
    ipd.showme(h, radius=3.8, spacing=1.3, coils=4)
    assert 0

    test_helix_9_1_1_r100_s40_p50_t2_d80_c7()
    test_helix_7_1_1_r80_s30_p20_t1_c7()
    test_helix_scaling()
    test_helix_params()
    test_helix_upper_neighbors()

@pytest.mark.fast
def test_helix_upper_neighbors():
    h = ipd.sym.helix.Helix(turns=9, nfold=1, turnsB=1, phase=0)
    ipd.showme(h)
    ipd.showme(h, closest=9)
    ipd.showme(h, closest=5, closest_upper_only=True)

@pytest.mark.fast
def test_helix_params():
    h = ipd.sym.helix.Helix(turns=9, nfold=1, turnsB=1, phase=0.001)
    h = ipd.sym.helix.Helix(turns=9, nfold=1, turnsB=1, phase=0.999)
    with pytest.raises(ValueError):
        h = ipd.sym.helix.Helix(turns=9, nfold=1, turnsB=1, phase=-0.001)
    with pytest.raises(ValueError):
        h = ipd.sym.helix.Helix(turns=9, nfold=1, turnsB=1, phase=1.001)

@pytest.mark.fast
def test_helix_scaling():
    pytest.skip()
    pytest.importorskip("willutil_cpp")
    h = ipd.sym.helix.Helix(turns=9, nfold=1, turnsB=1, phase=0)

    np.random.seed(7)
    xyz = ipd.homog.point_cloud(100, std=30, outliers=0)

    hframes = h.frames(xtalrad=9e8, closest=0, radius=100, spacing=40, coils=4)
    rb = ipd.dock.rigid.RigidBodyFollowers(coords=xyz, frames=hframes, symtype="H")
    origorig = rb.origins()
    origori = rb.orientations()

    scale = [1, 1, 2]
    rb.scale = scale
    assert np.allclose(rb.origins(), origorig * scale)
    assert np.allclose(rb.orientations(), origori)

    scale = [1.4, 1.4, 1]
    rb.scale = scale
    assert np.allclose(rb.origins(), origorig * scale)
    assert np.allclose(rb.orientations(), origori)

    scale = [1.4, 1.4, 0.5]
    rb.scale = scale
    assert np.allclose(rb.origins(), origorig * scale)
    assert np.allclose(rb.orientations(), origori)

    # for i in range(10):
    #    # rb.scale = [1, 1, 1 + i / 10]
    #    rb.scale = [1 + i / 10, 1 + i / 10, 1 - i / 20]
    #    ipd.showme(rb)
    #    ic(rb.origins())

    h = ipd.sym.helix.Helix(turns=9, nfold=1, turnsB=1, phase=0.5)
    hframes = h.frames(xtalrad=9e8, closest=0, radius=90, spacing=40, coils=4)
    rb = ipd.dock.rigid.RigidBodyFollowers(coords=xyz, frames=hframes, symtype="H")
    assert not np.allclose(rb.orientations(), origori)

@pytest.mark.skip
@pytest.mark.fast
@pytest.mark.skipif(np.__version__[0] == '2', reason='ipd.sym.Helix breaks on numpy 2')
def test_helix_9_1_1_r100_s40_p50_t2_d80_c7():
    pytest.importorskip('torch')
    h = ipd.sym.helix.Helix(turns=9, nfold=1, turnsB=1, phase=0.5)
    hframes = h.frames(xtalrad=80, closest=7, radius=100, spacing=40, coils=2)
    foo = np.array([
        [
            [5.46948158e-01, -8.37166478e-01, 0.00000000e00, 5.46948158e01],
            [8.37166478e-01, 5.46948158e-01, 0.00000000e00, 8.37166478e01],
            [0.00000000e00, 0.00000000e00, 1.00000000e00, -3.55555556e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        [
            [5.46948158e-01, 8.37166478e-01, 0.00000000e00, 5.46948158e01],
            [-8.37166478e-01, 5.46948158e-01, 0.00000000e00, -8.37166478e01],
            [0.00000000e00, 0.00000000e00, 1.00000000e00, 3.55555556e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        [
            [2.45485487e-01, -9.69400266e-01, 0.00000000e00, 2.45485487e01],
            [9.69400266e-01, 2.45485487e-01, 0.00000000e00, 9.69400266e01],
            [0.00000000e00, 0.00000000e00, 1.00000000e00, -7.55555556e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        [
            [-8.25793455e-02, 9.96584493e-01, 0.00000000e00, -8.25793455e00],
            [-9.96584493e-01, -8.25793455e-02, 0.00000000e00, -9.96584493e01],
            [0.00000000e00, 0.00000000e00, 1.00000000e00, 3.11111111e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        [
            [-8.25793455e-02, -9.96584493e-01, 0.00000000e00, -8.25793455e00],
            [9.96584493e-01, -8.25793455e-02, 0.00000000e00, 9.96584493e01],
            [0.00000000e00, 0.00000000e00, 1.00000000e00, -3.11111111e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        [
            [-6.77281572e-01, 7.35723911e-01, 0.00000000e00, -6.77281572e01],
            [-7.35723911e-01, -6.77281572e-01, 0.00000000e00, -7.35723911e01],
            [0.00000000e00, 0.00000000e00, 1.00000000e00, 2.66666667e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        [
            [-6.77281572e-01, -7.35723911e-01, 0.00000000e00, -6.77281572e01],
            [7.35723911e-01, -6.77281572e-01, 0.00000000e00, 7.35723911e01],
            [0.00000000e00, 0.00000000e00, 1.00000000e00, 5.77777778e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
    ])
    assert np.allclose(foo, hframes, atol=1e-4)

@pytest.mark.skip
@pytest.mark.fast
@pytest.mark.skipif(np.__version__[0] == '2', reason='ipd.sym.Helix breaks on numpy 2')
def test_helix_7_1_1_r80_s30_p20_t1_c7():
    pytest.importorskip('torch')
    h = ipd.sym.helix.Helix(turns=9, nfold=1, turnsB=1, phase=0.2)
    hframes = h.frames(closest=9, radius=80, spacing=30, coils=1)
    foo = np.array([
        [[1.0, 0.0, 0.0, 80.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        [
            [0.99068595, -0.13616665, 0.0, 79.25487568],
            [0.13616665, 0.99068595, 0.0, 10.89333193],
            [0.0, 0.0, 1.0, -30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.99068595, 0.13616665, 0.0, 79.25487568],
            [-0.13616665, 0.99068595, 0.0, -10.89333193],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.77571129, -0.63108794, 0.0, 62.05690326],
            [0.63108794, 0.77571129, 0.0, 50.48703555],
            [0.0, 0.0, 1.0, 3.33333333],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.77571129, 0.63108794, 0.0, 62.05690326],
            [-0.63108794, 0.77571129, 0.0, -50.48703555],
            [0.0, 0.0, 1.0, -3.33333333],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.8544194, 0.51958395, 0.0, 68.35355236],
            [-0.51958395, 0.8544194, 0.0, -41.566716],
            [0.0, 0.0, 1.0, -33.33333333],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.8544194, -0.51958395, 0.0, 68.35355236],
            [0.51958395, 0.8544194, 0.0, 41.566716],
            [0.0, 0.0, 1.0, 33.33333333],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.68255314, -0.73083596, 0.0, 54.60425146],
            [0.73083596, 0.68255314, 0.0, 58.46687714],
            [0.0, 0.0, 1.0, -26.66666667],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.68255314, 0.73083596, 0.0, 54.60425146],
            [-0.73083596, 0.68255314, 0.0, -58.46687714],
            [0.0, 0.0, 1.0, 26.66666667],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ])
    assert np.allclose(foo, hframes, atol=1e-4)

if __name__ == "__main__":
    main()
