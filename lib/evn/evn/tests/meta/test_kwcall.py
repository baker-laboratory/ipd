import unittest
import pytest

import evn

config_test = evn.Bunch(
    re_only=[],
    re_exclude=[],
)


def main():
    evn.testing.quicktest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
    )


def test_locals():
    foo, bar, baz = 1, 2, 3
    assert evn.meta.picklocals('foo bar', asdict=True) == dict(foo=1, bar=2)


def test_addreduce():
    assert evn.addreduce([[1], [2, 3], [4]]) == [1, 2, 3, 4]


@pytest.mark.xfail
def test_get_function_for_which_call_to_caller_is_argument():

    def FIND_THIS_FUNCTION(*a, **kw):
        ...

    def CALLED_TO_PRODUCE_ARGUMENT():
        uncle_func = evn.meta.get_function_for_which_call_to_caller_is_argument()
        assert uncle_func == FIND_THIS_FUNCTION

    FIND_THIS_FUNCTION(1, 2, CALLED_TO_PRODUCE_ARGUMENT(), 3)


def test_kwcheck():
    kw = dict(apple='apple', banana='banana', cherry='cherry')

    def foo(apple):
        return apple

    def bar(banana, cherry):
        return banana, cherry

    with pytest.raises(TypeError):
        foo(**kw)
    with pytest.raises(TypeError):
        bar(**kw)

    assert foo(**evn.kwcheck(kw, foo)) == evn.kwcall(kw, foo)
    assert bar(**evn.kwcheck(kw, bar)) == evn.kwcall(kw, bar)

    assert evn.kwcall(kw, bar, banana='bananums') == ('bananums', 'cherry')


def target_func(a, b, c=3):
    pass


class SomeClass:

    def method(self, param1, param2, optional=None):
        pass


class_method = SomeClass().method


def flexible_func(a, b, *args, c=3, **kwargs):
    pass


def test_kwcheck_explicit_function_filtering():
    """Test basic filtering with explicitly provided function."""
    kwargs = {'a': 1, 'b': 2, 'd': 4, 'e': 5}
    result = evn.kwcheck(kwargs, target_func, checktypos=False)

    # Should only keep keys that match target_func parameters
    assert result == {'a': 1, 'b': 2}


def test_kwcheck_checktypos_flag_disabled():
    """Test that no typo checking occurs when checktypos=False."""

    def func(alpha, beta, gamma=3):
        pass

    # 'alpho' is a potential typo for 'alpha'
    kwargs = {'alpho': 1, 'beta': 2}

    # Should not raise TypeError because checktypos=False
    result = evn.kwcheck(kwargs, func, checktypos=False)
    assert result == {'beta': 2}


def test_kwcheck_typo_detection():
    """Test that typos are detected and raise TypeError."""

    func = lambda alpha, beta, gamma=3: ...

    # 'alpho' is a potential typo for 'alpha'
    kwargs = {'alpho': 1, 'beta': 2}

    # Should raise TypeError due to 'alpho' being close to 'alpha'
    with pytest.raises(TypeError) as excinfo:
        evn.kwcheck(kwargs, func, checktypos=True)

    # Check that the error message contains both the typo and suggestion
    assert 'alpho' in str(excinfo.value)
    assert 'alpha' in str(excinfo.value)


def test_kwcheck_no_typo_for_dissimilar():
    """Test that dissimilar argument names don't trigger typo detection."""

    def func(first, second, third=3):
        pass

    # 'fourth' is not similar enough to any parameter
    kwargs = {'fourth': 4, 'first': 1}

    # Should not raise TypeError, just filter out 'fourth'
    result = evn.kwcheck(kwargs, func)
    assert result == {'first': 1}


def test_kwcheck_automatic_function_detection():
    """Test automatic detection of the calling function."""

    def func(x, y, z=3):
        pass

    kwargs = {'x': 1, 'y': 2, 'extra': 3}
    result = evn.kwcheck(kwargs, func)
    assert result == {'x': 1, 'y': 2}


def test_kwcheck_no_function_detection_error():
    """Test that an error is raised when function detection fails."""
    with pytest.raises(TypeError) as excinfo:
        evn.kwcheck({'a': 1})
    assert "Couldn't get function" in str(excinfo.value)


def test_kwcheck_method_as_function():
    """Test that evn.kwcheck works with methods as well as functions."""
    kwargs = {'param1': 'value1', 'param2': 'value2', 'other': 'value3'}

    result = evn.kwcheck(kwargs, class_method)
    assert result == {'param1': 'value1', 'param2': 'value2'}


def test_kwcheck_integration_with_function_call():
    """Test using evn.kwcheck directly in a function call (integration test)."""

    def function_with_specific_args(required, optional=None):
        return (required, optional)

    # Create a wrapper that simulates the actual usage pattern
    def call_with_kwcheck():
        kwargs = {'required': 'value', 'extra': 'ignored'}
        return function_with_specific_args(
            **evn.kwcheck(kwargs, function_with_specific_args))

    result = call_with_kwcheck()
    assert result == ('value', None)


def test_kwcheck_with_varargs_and_varkw():
    """Test with functions that use *args and **kwargs."""
    kwargs = {'a': 1, 'b': 2, 'c': 4, 'd': 5, 'e': 6}
    result = evn.kwcheck(kwargs, flexible_func)
    assert result == kwargs


def test_kwcheck_empty_kwargs():
    """Test with empty kwargs dictionary."""
    result = evn.kwcheck({}, target_func)
    assert result == {}


def test_kwcheck_all_kwargs_match():
    """Test when all kwargs match function parameters."""
    kwargs = {'a': 1, 'b': 2, 'c': 3}
    result = evn.kwcheck(kwargs, target_func)
    assert result == kwargs
    assert result is not kwargs  # Should be a copy, not the same object


def test_kwcheck_kwargs_with_none_values():
    """Test with None values in kwargs."""
    kwargs = {'a': None, 'b': None, 'd': None}
    result = evn.kwcheck(kwargs, target_func)
    assert result == {'a': None, 'b': None}


def test_kwcheck_with_kwargs_func():
    """Test with None values in kwargs."""
    func = lambda a, b, c=3, **kw: ...
    kwargs = {'a': None, 'b': None, 'd': None}
    result = evn.kwcheck(kwargs, func)
    assert result == kwargs


class TestFilterMapping(unittest.TestCase):

    def setUp(self):
        self.map = {
            'test_func1': lambda: 'func1',
            'test_func2': lambda: 'func2',
            'test_funcA': lambda: 'funcA',
            'test_funcB': lambda: 'funcB',
            'test_other': lambda: 'other',
            'normal_func': lambda: 'normal',
        }

    def test_default_behavior(self):
        map = self.map.copy()
        evn.meta.filter_namespace_funcs(map)
        assert 'test_func1' in map
        assert 'test_func2' in map
        assert 'test_funcA' in map
        assert 'test_funcB' in map
        assert 'test_other' in map
        assert 'normal_func' in map

    def test_only(self):
        map = self.map.copy()
        evn.meta.filter_namespace_funcs(map, only=('test_func1', ))
        assert 'test_func1' in map
        assert 'test_func2' not in map
        assert 'test_funcA' not in map
        assert 'test_funcB' not in map
        assert 'test_other' not in map

    def test_exclude(self):
        map = self.map.copy()
        evn.meta.filter_namespace_funcs(map, exclude=('test_func1', ))
        assert 'test_func1' not in map
        assert 'test_func2' in map

    def test_re_only(self):
        map = self.map.copy()
        evn.meta.filter_namespace_funcs(map, re_only=('test_func[0-9]', ))
        assert 'test_func1' in map
        assert 'test_func2' in map
        assert 'test_funcA' not in map
        assert 'test_funcB' not in map
        assert 'test_other' not in map

    def test_re_exclude(self):
        map = self.map.copy()
        evn.meta.filter_namespace_funcs(map, re_exclude=('test_func[0-9]', ))
        assert 'test_func1' not in map
        assert 'test_func2' not in map
        assert 'test_funcA' in map
        assert 'test_funcB' in map
        assert 'test_other' in map

    def test_re_only_letters(self):
        map = self.map.copy()
        evn.meta.filter_namespace_funcs(map, re_only=('test_func[A-Z]', ))
        assert 'test_func1' not in map
        assert 'test_func2' not in map
        assert 'test_funcA' in map
        assert 'test_funcB' in map
        assert 'test_other' not in map

    def test_combination_only_and_exclude(self):
        map = self.map.copy()
        evn.meta.filter_namespace_funcs(map,
                                        only=('test_func1', ),
                                        exclude=('test_func1', ))
        assert 'test_func1' not in map
        assert 'test_func2' not in map
        assert 'test_funcA' not in map
        assert 'test_funcB' not in map

    def test_combination_re_only_and_re_exclude(self):
        map = self.map.copy()
        evn.meta.filter_namespace_funcs(map,
                                        re_only=('test_func[0-9]', ),
                                        re_exclude=('test_func1', ))
        assert 'test_func1' not in map
        assert 'test_func2' in map
        assert 'test_funcA' not in map
        assert 'test_funcB' not in map


if __name__ == '__main__':
    main()
