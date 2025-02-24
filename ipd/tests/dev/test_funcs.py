import pytest

import ipd

def main():
    ipd.tests.maintest(namespace=globals())

@pytest.mark.fast
def test_iterize():

    @ipd.dev.iterize_on_first_param
    def foo(a):
        return a * a

    assert foo(4) == 4 * 4
    assert foo([1, 2]) == [1, 4]

    @ipd.dev.iterize_on_first_param(basetype=str)
    def bar(a):
        return 2 * a

    assert bar('foo') == 'foofoo'
    assert bar(['a', 'b']) == ['aa', 'bb']
    assert bar(1.1) == 2.2

@pytest.mark.fast
def test_InfixOperator():
    th = pytest.importorskip('torch')
    x = ipd.dev.InfixOperator(lambda a, b: a * b)
    X = ipd.dev.InfixOperator(lambda a, b: a @ b)
    assert x(4, 3) == 12
    assert 4 * 3 == 4 | x | 3
    a = th.randn(16).reshape(4, 4)
    b = th.randn(16).reshape(4, 4)
    assert th.allclose(a @ b, X(a, b))
    assert th.allclose(a @ b, a | X | b)

if __name__ == '__main__':
    main()
