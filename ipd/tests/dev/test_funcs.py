import pytest

import ipd
import numpy as np

def main():
    ipd.tests.maintest(namespace=globals())

@pytest.mark.fast
def test_InfixOperator():
    th = pytest.importorskip('torch')
    x = ipd.dev.InfixOperator(lambda a, b: a * b)
    X = ipd.dev.InfixOperator(lambda a, b: a @ b)
    assert x(4, 3) == 12
    assert 4 * 3 == 4 | x | 3
    a = th.randn(16).reshape(4, 4)
    b = th.randn(16).reshape(4, 4)
    assert np.allclose(a @ b, X(a, b))
    assert np.allclose(a @ b, a | X | b)

if __name__ == '__main__':
    main()
