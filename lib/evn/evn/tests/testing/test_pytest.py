import pytest

import evn
from evn.testing.pytest import *

config_test = evn.Bunch(
    re_only=[
        #
    ],
    re_exclude=[
        #
    ],
)


def main():
    evn.testing.quicktest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
    )


# ==== TESTS FOR has_pytest_mark ====


@pytest.mark.ci
def test_with_ci_mark():
    pass


def test_has_pytest_mark_positive():
    assert has_pytest_mark(test_with_ci_mark, 'ci') is True


def test_has_pytest_mark_negative():
    assert has_pytest_mark(test_with_ci_mark, 'skip') is False


def test_has_pytest_mark_no_marks():

    def test_func():
        pass

    assert has_pytest_mark(test_func, 'custom') is False


@pytest.mark.skip
def test_with_skip():
    pass


def test_no_pytest_skip_false():
    assert no_pytest_skip(test_with_skip) is False


def test_no_pytest_skip_true():

    def test_func():
        pass

    assert no_pytest_skip(test_func) is True


# ==== TESTS FOR get_pytest_params ====


@pytest.mark.parametrize('x, y', [(1, 2), (3, 4)])
def test_with_parametrize(x, y):
    assert y - x == 1


def test_get_pytest_params():
    args = get_pytest_params(test_with_parametrize)
    assert args == (['x', 'y'], [(1, 2), (3, 4)])


def test_get_pytest_params_none():

    def test_func():
        pass

    assert get_pytest_params(test_func) is None


# ==== USEFUL TEST UTILITIES ====


def is_skipped(func):
    """Utility function to check if a test is marked as skip."""
    return has_pytest_mark(func, 'skip')


def is_parametrized(func):
    """Utility function to check if a test is marked as parametrize."""
    return get_pytest_params(func) is not None


# === TEST UTILITIES ===


def test_is_skipped():
    assert is_skipped(test_with_skip) is True
    assert is_skipped(test_with_ci_mark) is False


def test_is_parametrized():
    assert is_parametrized(test_with_parametrize) is True

    def test_func():
        pass

    assert is_parametrized(test_func) is False


if __name__ == '__main__':
    main()
