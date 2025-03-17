import pytest

import ipd

config_test = ipd.Bunch(
    re_only=[
        #
    ],
    re_exclude=[
        #
    ],
)

def main():
    test_with_custom_mark()
    test_has_pytest_mark_positive()
    test_has_pytest_mark_negative()
    test_has_pytest_mark_no_marks()
    test_with_skip()
    test_no_pytest_skip_false()
    test_no_pytest_skip_true()
    # test_with_parametrize(x, y)
    test_get_pytest_params()
    test_get_pytest_params_none()
    test_is_skipped()
    test_is_parametrized()
    return
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
    )

# ==== TESTS FOR ipd.dev.has_pytest_mark ====

@pytest.mark.custom
def test_with_custom_mark():
    pass

def test_has_pytest_mark_positive():
    assert ipd.dev.has_pytest_mark(test_with_custom_mark, 'custom') is True

def test_has_pytest_mark_negative():
    assert ipd.dev.has_pytest_mark(test_with_custom_mark, 'skip') is False

def test_has_pytest_mark_no_marks():

    def test_func():
        pass

    assert ipd.dev.has_pytest_mark(test_func, 'custom') is False

# ==== TESTS FOR ipd.dev.no_pytest_skip ====

@pytest.mark.skip
def test_with_skip():
    pass

def test_no_pytest_skip_false():
    assert ipd.dev.no_pytest_skip(test_with_skip) is False

def test_no_pytest_skip_true():

    def test_func():
        pass

    assert ipd.dev.no_pytest_skip(test_func) is True

# ==== TESTS FOR ipd.dev.get_pytest_params ====

@pytest.mark.parametrize("x, y", [(1, 2), (3, 4)])
def test_with_parametrize(x, y):
    assert y - x == 1

def test_get_pytest_params():
    args = ipd.dev.get_pytest_params(test_with_parametrize)
    assert args == (["x", "y"], [(1, 2), (3, 4)])

def test_get_pytest_params_none():

    def test_func():
        pass

    assert ipd.dev.get_pytest_params(test_func) is None

# ==== USEFUL TEST UTILITIES ====

def is_skipped(func):
    """Utility function to check if a test is marked as skip."""
    return ipd.dev.has_pytest_mark(func, 'skip')

def is_parametrized(func):
    """Utility function to check if a test is marked as parametrize."""
    return ipd.dev.get_pytest_params(func) is not None

# === TEST UTILITIES ===

def test_is_skipped():
    assert is_skipped(test_with_skip) is True
    assert is_skipped(test_with_custom_mark) is False

def test_is_parametrized():
    assert is_parametrized(test_with_parametrize) is True

    def test_func():
        pass

    assert is_parametrized(test_func) is False

if __name__ == '__main__':
    main()
