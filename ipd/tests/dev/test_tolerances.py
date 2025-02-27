import numpy as np
import ipd

def main():
    ipd.tests.maintest(namespace=globals())

def test_Tolerances():
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
    5 <= tol.foo
    assert tol.foo.n_checks == 4
    assert tol.foo.n_passes == 2

if __name__ == '__main__':
    main()
