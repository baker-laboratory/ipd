import pytest

@pytest.mark.skip
def test_should_skip():
    raise RuntimeError('BOOM')

@pytest.mark.xfail
def test_should_xfail():
    assert 0

def test_should_throw_skip():
    raise pytest.skip.Exception('SKIP')

@pytest.mark.parametrize('a, b, golden', [
    (1, 1, True),
    (2, 2, True),
    (3, 4, False),
])
def test_func_with_pytest_parametrize(a, b, golden):
    assert golden == (a == b)
