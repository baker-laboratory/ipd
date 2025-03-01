import pytest
import numpy as np
import ipd

def main():
    ipd.tests.maintest(namespace=globals())

def test_tolerances_comparison_operators():
    tol = ipd.dev.Tolerances(1e-4, foo=4)
    tol.default < 1
    assert tol.default.n_checks == 1
    assert tol.default.n_passes == 1
    tol.default > 1
    assert tol.default.n_checks == 2
    assert tol.default.n_passes == 1
    tol.default >= np.ones(3)
    assert tol.default.n_checks == 5
    assert tol.default.n_passes == 1
    tol.default <= np.ones(5)
    assert tol.default.n_checks == 10
    assert tol.default.n_passes == 6
    5 > tol.foo
    assert tol.foo.n_checks == 1
    assert tol.foo.n_passes == 1
    5 < tol.foo
    assert tol.foo.n_checks == 2
    assert tol.foo.n_passes == 1
    5 >= tol.foo
    assert tol.foo.n_checks == 3
    assert tol.foo.n_passes == 2
    5.0 <= tol.foo
    assert tol.foo.n_checks == 4
    assert tol.foo.n_passes == 2

def test_tolerances_numpy():
    tol = ipd.dev.Tolerances(1e-4, foo=4)
    assert np.all(tol.foo > np.eye(4))

def test_tolerances_torch():
    tol = ipd.dev.Tolerances(1e-4, foo=4)
    if th := ipd.importornone('torch'):
        assert th.all(tol.foo > th.eye(4))

def test_tolerances_xarray():
    tol = ipd.dev.Tolerances(1e-4, foo=4)
    if xr := ipd.importornone('xarray'):
        assert np.all(tol.foo > xr.DataArray([1, 3, 3]))

def test_tolerances_numpy_right():
    tol = ipd.dev.Tolerances(1e-4, foo=4)
    assert np.all(np.eye(4) < tol.foo)

def test_tolerances_torch_right():
    tol = ipd.dev.Tolerances(1e-4, foo=4)
    if th := ipd.importornone('torch'):
        assert th.all(th.eye(4) < tol.foo)

def test_tolerances_xarray_right():
    tol = ipd.dev.Tolerances(1e-4, foo=4)
    if xr := ipd.importornone('xarray'):
        assert np.all(xr.DataArray([1, 3, 3]) < tol.foo)

def test_tolerances_xarray_dataset_unsupported():
    tol = ipd.dev.Tolerances(1e-4, foo=4)
    if xr := ipd.importornone('xarray'):
        ds = xr.Dataset(dict(foo=xr.DataArray([1, 3, 3])))
        with pytest.raises(TypeError):
            np.all(tol.foo > ds)

def test_tolerances_copy():
    tol = ipd.Tolerances(1)
    tol.foo == 1
    tol2 = tol.copy()
    tol2.foo == 2
    tol2.foo == 1
    tol.foo == 0
    assert tol.foo.n_checks == 2
    assert tol.foo.n_passes == 1
    assert tol2.foo.n_checks == 3
    assert tol2.foo.n_passes == 2

if __name__ == '__main__':
    main()
