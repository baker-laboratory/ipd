import unittest
import pytest

import ipd
from ipd import kwcheck

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

@pytest.mark.xfail
def test_get_function_for_which_call_to_caller_is_argument():

    def bar(foo):
        pass

    def foo():
        ic(ipd.dev.get_function_for_which_call_to_caller_is_argument())

    bar(foo())
    assert 0

@pytest.mark.fast
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

    assert foo(**ipd.kwcheck(kw, foo)) == ipd.kwcall(foo, kw)
    assert bar(**ipd.kwcheck(kw, bar)) == ipd.kwcall(bar, kw)

    assert ipd.kwcall(bar, kw, banana='bananums') == ('bananums', 'cherry')

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
    result = kwcheck(kwargs, target_func, checktypos=False)

    # Should only keep keys that match target_func parameters
    assert result == {'a': 1, 'b': 2}

def test_kwcheck_checktypos_flag_disabled():
    """Test that no typo checking occurs when checktypos=False."""

    def func(alpha, beta, gamma=3):
        pass

    # 'alpho' is a potential typo for 'alpha'
    kwargs = {'alpho': 1, 'beta': 2}

    # Should not raise TypeError because checktypos=False
    result = kwcheck(kwargs, func, checktypos=False)
    assert result == {'beta': 2}

def test_kwcheck_typo_detection():
    """Test that typos are detected and raise TypeError."""

    func = lambda alpha, beta, gamma=3: ...

    # 'alpho' is a potential typo for 'alpha'
    kwargs = {'alpho': 1, 'beta': 2}

    # Should raise TypeError due to 'alpho' being close to 'alpha'
    with pytest.raises(TypeError) as excinfo:
        kwcheck(kwargs, func, checktypos=True)

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
    result = kwcheck(kwargs, func)
    assert result == {'first': 1}

def test_kwcheck_automatic_function_detection():
    """Test automatic detection of the calling function."""

    def func(x, y, z=3):
        pass

    kwargs = {'x': 1, 'y': 2, 'extra': 3}
    result = kwcheck(kwargs, func)
    assert result == {'x': 1, 'y': 2}

def test_kwcheck_no_function_detection_error():
    """Test that an error is raised when function detection fails."""
    with pytest.raises(TypeError) as excinfo:
        kwcheck({'a': 1})
    assert "Couldn't get function" in str(excinfo.value)

def test_kwcheck_method_as_function():
    """Test that kwcheck works with methods as well as functions."""
    kwargs = {'param1': 'value1', 'param2': 'value2', 'other': 'value3'}

    result = kwcheck(kwargs, class_method)
    assert result == {'param1': 'value1', 'param2': 'value2'}

def test_kwcheck_integration_with_function_call():
    """Test using kwcheck directly in a function call (integration test)."""

    def function_with_specific_args(required, optional=None):
        return (required, optional)

    # Create a wrapper that simulates the actual usage pattern
    def call_with_kwcheck():
        kwargs = {'required': 'value', 'extra': 'ignored'}
        return function_with_specific_args(**kwcheck(kwargs, function_with_specific_args))

    result = call_with_kwcheck()
    assert result == ('value', None)

def test_kwcheck_with_varargs_and_varkw():
    """Test with functions that use *args and **kwargs."""
    kwargs = {'a': 1, 'b': 2, 'c': 4, 'd': 5, 'e': 6}
    result = kwcheck(kwargs, flexible_func)

    # Should keep 'a', 'b', 'c' (named params) but filter out 'd', 'e'
    assert result == {'a': 1, 'b': 2, 'c': 4}

def test_kwcheck_empty_kwargs():
    """Test with empty kwargs dictionary."""
    result = kwcheck({}, target_func)
    assert result == {}

def test_kwcheck_all_kwargs_match():
    """Test when all kwargs match function parameters."""
    kwargs = {'a': 1, 'b': 2, 'c': 3}
    result = kwcheck(kwargs, target_func)
    assert result == kwargs
    assert result is not kwargs  # Should be a copy, not the same object

def test_kwcheck_kwargs_with_none_values():
    """Test with None values in kwargs."""
    kwargs = {'a': None, 'b': None, 'd': None}
    result = kwcheck(kwargs, target_func)
    assert result == {'a': None, 'b': None}

import pytest
import functools
from pathlib import Path
from collections import namedtuple

# Assuming these are copied from your module
def iterizeable(arg, basetype=None):
    if basetype and isinstance(arg, basetype): return False
    if hasattr(arg, '__iter__'): return True
    return False

def iterize_on_first_param(*metaargs, **metakw):

    def deco(func):

        @functools.wraps(func)
        def wrapper(arg0, *args, **kw):
            if iterizeable(arg0, **metakw):
                return [func(a0, *args, **kw) for a0 in arg0]
            return func(arg0, *args, **kw)

        return wrapper

    if metaargs:  # handle case with no call/args
        assert callable(metaargs[0])
        assert not metakw
        return deco(metaargs[0])
    return deco

iterize_on_first_param_path = iterize_on_first_param(basetype=(str, Path))

# Define a custom iterable type for testing
class CustomIterable(namedtuple('CustomIterable', ['items'])):

    def __iter__(self):
        return iter(self.items)

@pytest.mark.fast
class TestIterizeOnFirstParam(unittest.TestCase):
    """Test suite for the iterize_on_first_param decorator."""

    def setUp(self):
        """Set up test functions with the decorator applied in different ways."""
        # Basic decorator without arguments
        @iterize_on_first_param
        def square(x):
            return x * x

        self.square = square

        @iterize_on_first_param
        def multiply(x, y):
            return x * y

        self.multiply = multiply

        # Decorator with basetype=str
        @iterize_on_first_param(basetype=str)
        def get_length(x):
            return len(x)

        self.get_length = get_length

        # Using the pre-configured path decorator
        @iterize_on_first_param_path
        def process_path(path):
            return f"Processing {path}"

        self.process_path = process_path

        # For metadata preservation test
        def original_func(x):
            """Test docstring."""
            return x

        self.original_func = original_func

        # Test data
        self.scalar = 5
        self.list_data = [1, 2, 3]
        self.tuple_data = (4, 5, 6)
        self.empty_list = []
        self.custom_iterable = CustomIterable([1, 2, 3])
        self.nested_lists = [[1, 2], [3, 4]]
        self.string = "hello"
        self.string_list = ["hello", "world"]
        self.path_obj = Path("sample.txt")
        self.path_list = [Path("file1.txt"), Path("file2.txt")]

    def tearDown(self):
        """Clean up after each test."""
        # No specific cleanup needed for these tests
        pass

    def test_scalar_input(self):
        """Test with a scalar input value."""
        assert self.square(self.scalar) == 25
        assert self.multiply(3, 4) == 12

    def test_list_input(self):
        """Test with a list input for the first parameter."""
        assert self.square(self.list_data) == [1, 4, 9]
        assert self.multiply(self.list_data, 2) == [2, 4, 6]

    def test_tuple_input(self):
        """Test with a tuple input for the first parameter."""
        assert self.square(self.tuple_data) == [16, 25, 36]
        assert self.multiply(self.tuple_data, 3) == [12, 15, 18]

    def test_empty_iterable(self):
        """Test with an empty iterable."""
        assert self.square(self.empty_list) == []
        assert self.multiply(self.empty_list, 10) == []

    def test_custom_iterable(self):
        """Test with a custom iterable type."""
        assert self.square(self.custom_iterable) == [1, 4, 9]
        assert self.multiply(self.custom_iterable, 5) == [5, 10, 15]

    def test_basetype_exclusion(self):
        """Test that basetyped objects are treated as scalars."""
        # String should be treated as scalar when basetype=str
        assert self.get_length(self.string) == 5
        assert self.get_length(self.string_list) == [5, 5]

    def test_multiple_basetype_exclusion(self):
        """Test with multiple basetype exclusions."""
        # String should be treated as scalar with path decorator
        assert self.process_path(self.string) == f"Processing {self.string}"
        # Path object should be treated as scalar with path decorator
        assert self.process_path(self.path_obj) == f"Processing {self.path_obj}"
        # List of strings should be processed element-wise
        assert self.process_path(["file1.txt", "file2.txt"]) == ["Processing file1.txt", "Processing file2.txt"]
        # List of Path objects should be processed element-wise
        expected = [f"Processing {self.path_list[0]}", f"Processing {self.path_list[1]}"]
        assert self.process_path(self.path_list) == expected

    def test_nested_iterables(self):
        """Test handling of nested iterables."""
        # Define a custom function that handles lists for this test
        @iterize_on_first_param
        def sum_list(x):
            return sum(x) if isinstance(x, list) else x

        assert sum_list(self.nested_lists) == [3, 7]

    def test_decorator_preserves_metadata(self):
        """Test that the decorator preserves function metadata."""
        decorated = iterize_on_first_param(self.original_func)

        assert decorated.__name__ == "original_func"
        assert decorated.__doc__ == "Test docstring."

    def test_generator_input(self):
        """Test with a generator expression as input."""
        gen = (i for i in range(1, 4))
        assert self.square(gen) == [1, 4, 9]

    def test_set_input(self):
        """Test with a set as input."""
        # Note: Sets are unordered, so we need to check membership rather than exact equality
        result = self.square({1, 2, 3})
        assert set(result) == {1, 4, 9}
        assert len(result) == 3

@pytest.mark.fast
class TestIterizeableFunction(unittest.TestCase):
    """Test suite for the iterizeable helper function."""

    def setUp(self):
        """Set up test data."""
        self.list_data = [1, 2, 3]
        self.string = "hello"
        self.integer = 42
        self.path_obj = Path("test.txt")

    def test_basic_iterizeable(self):
        """Test basic iterizeable function without basetype."""
        assert iterizeable(self.list_data) is True
        assert iterizeable(self.string) is True  # String is iterable
        assert iterizeable(self.integer) is False

    def test_iterizeable_with_basetype(self):
        """Test iterizeable function with basetype parameter."""
        # String should not be considered iterable when basetype includes str
        assert iterizeable(self.string, basetype=str) is False
        assert iterizeable(self.list_data, basetype=str) is True

        # Path should not be considered iterable when basetype includes Path
        assert iterizeable(self.path_obj, basetype=Path) is False

        # Multiple basetypes
        assert iterizeable(self.string, basetype=(str, Path)) is False
        assert iterizeable(self.path_obj, basetype=(str, Path)) is False
        assert iterizeable(self.list_data, basetype=(str, Path)) is True

class TestFilterMapping(unittest.TestCase):

    def setUp(self):
        self.map = {
            'test_func1': lambda: "func1",
            'test_func2': lambda: "func2",
            'test_funcA': lambda: "funcA",
            'test_funcB': lambda: "funcB",
            'test_other': lambda: "other",
            'normal_func': lambda: "normal",
        }

    def test_default_behavior(self):
        map = self.map.copy()
        ipd.dev.filter_namespace_funcs(map)
        assert 'test_func1' in map
        assert 'test_func2' in map
        assert 'test_funcA' in map
        assert 'test_funcB' in map
        assert 'test_other' in map
        assert 'normal_func' in map

    def test_only(self):
        map = self.map.copy()
        ipd.dev.filter_namespace_funcs(map, only=('test_func1', ))
        assert 'test_func1' in map
        assert 'test_func2' not in map
        assert 'test_funcA' not in map
        assert 'test_funcB' not in map
        assert 'test_other' not in map

    def test_exclude(self):
        map = self.map.copy()
        ipd.dev.filter_namespace_funcs(map, exclude=('test_func1', ))
        assert 'test_func1' not in map
        assert 'test_func2' in map

    def test_re_only(self):
        map = self.map.copy()
        ipd.dev.filter_namespace_funcs(map, re_only=('test_func[0-9]', ))
        assert 'test_func1' in map
        assert 'test_func2' in map
        assert 'test_funcA' not in map
        assert 'test_funcB' not in map
        assert 'test_other' not in map

    def test_re_exclude(self):
        map = self.map.copy()
        ipd.dev.filter_namespace_funcs(map, re_exclude=('test_func[0-9]', ))
        assert 'test_func1' not in map
        assert 'test_func2' not in map
        assert 'test_funcA' in map
        assert 'test_funcB' in map
        assert 'test_other' in map

    def test_re_only_letters(self):
        map = self.map.copy()
        ipd.dev.filter_namespace_funcs(map, re_only=('test_func[A-Z]', ))
        assert 'test_func1' not in map
        assert 'test_func2' not in map
        assert 'test_funcA' in map
        assert 'test_funcB' in map
        assert 'test_other' not in map

    def test_combination_only_and_exclude(self):
        map = self.map.copy()
        ipd.dev.filter_namespace_funcs(map, only=('test_func1', ), exclude=('test_func1', ))
        assert 'test_func1' not in map
        assert 'test_func2' not in map
        assert 'test_funcA' not in map
        assert 'test_funcB' not in map

    def test_combination_re_only_and_re_exclude(self):
        map = self.map.copy()
        ipd.dev.filter_namespace_funcs(map, re_only=('test_func[0-9]', ), re_exclude=('test_func1', ))
        assert 'test_func1' not in map
        assert 'test_func2' in map
        assert 'test_funcA' not in map
        assert 'test_funcB' not in map

if __name__ == '__main__':
    main()
