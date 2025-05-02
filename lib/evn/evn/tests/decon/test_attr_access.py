from pathlib import Path
from collections import namedtuple
import unittest
import pytest

import evn


def main():
    evn.testing.quicktest(namespace=globals())


def test_iterize():

    @evn.iterize_on_first_param
    def Foo(a):
        return a * a

    assert Foo(4) == 4 * 4
    assert Foo([1, 2]) == [1, 4]


def test_iterize_basetype():

    @evn.iterize_on_first_param(basetype=str)
    def bar(a):
        return 2 * a

    assert bar('foo') == 'foofoo'
    assert bar(['a', 'b']) == ['aa', 'bb']
    # ic(bar('a b'))
    assert bar('a b') == ['aa', 'bb']
    assert bar(1.1) == 2.2


def test_iterize_asdict():

    @evn.iterize_on_first_param(basetype=str, asdict=True)
    def baz(a):
        return 2 * a

    assert baz('foo') == 'foofoo'
    assert baz(['a', 'b']) == dict(a='aa', b='bb')
    assert baz('a b') == dict(a='aa', b='bb')
    assert baz(1.1) == 2.2


def test_iterize_asbunch():

    @evn.iterize_on_first_param(basetype=str, asbunch=True)
    def baz(a):
        return 2 * a

    assert baz('foo') == 'foofoo'
    assert isinstance(baz(['a', 'b']), evn.Bunch)
    assert baz(['a', 'b']) == dict(a='aa', b='bb')
    assert baz('a b') == dict(a='aa', b='bb')
    assert baz(1.1) == 2.2
    assert baz([1, 2]) == {1: 2, 2: 4}


def test_iterize_allowmap():

    @evn.iterize_on_first_param(basetype=str, asbunch=True)
    def foo(a):
        return 2 * a

    with pytest.raises(TypeError):
        foo(dict(a=1, b=2))

    @evn.iterize_on_first_param(basetype=str, asbunch=True, allowmap=True)
    def bar(a):
        return 2 * a

    assert bar(dict(a=1, b=2)) == dict(a=2, b=4)


def test_iterize_basetype_string():

    class mylist(list):
        pass

    @evn.iterize_on_first_param(basetype='str')
    def foo(a):
        return 2 * a

    with pytest.raises(TypeError):
        foo(dict(a=1, b=2))

    @evn.iterize_on_first_param(basetype='mylist')
    def bar(a):
        return len(a)

    assert bar([]) == []
    assert bar([[], []]) == [0, 0]
    assert bar(mylist([[], []])) == 2
    # assert bar(e/[dict(a=1, b=2)]) == ['a', 'b']


# Define a custom iterable type for testing
class CustomIterable(namedtuple('CustomIterable', ['items'])):

    def __iter__(self):
        return iter(self.items)


class TestIterizeOnFirstParam(unittest.TestCase):
    """Test suite for the evn.iterize_on_first_param decorator."""

    def setUp(self):
        """Set up test functions with the decorator applied in different ways."""

        # Basic decorator without arguments
        @evn.iterize_on_first_param
        def square(x):
            return x * x

        self.square = square

        @evn.iterize_on_first_param
        def multiply(x, y):
            return x * y

        self.multiply = multiply

        # Decorator with basetype=str
        @evn.iterize_on_first_param(basetype=str)
        def get_length(x):
            return len(x)

        self.get_length = get_length

        # Decorator with basetype=str
        @evn.iterize_on_first_param(basetype=str, nonempty=True)
        def remove_first(x):
            return x[1:] if x else ''

        self.remove_first = remove_first

        # Using the pre-configured path decorator
        @evn.iterize_on_first_param_path
        def process_path(path):
            return f'Processing {path}'

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
        self.string = 'hello'
        self.string_list = ['hello', 'world']
        self.path_obj = Path('sample.txt')
        self.path_list = [Path('file1.txt'), Path('file2.txt')]

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
        assert self.square(self.tuple_data) == (16, 25, 36)
        assert self.multiply(self.tuple_data, 3) == (12, 15, 18)

    def test_empty_iterable(self):
        """Test with an empty iterable."""
        assert self.square(self.empty_list) == []
        assert self.multiply(self.empty_list, 10) == []

    def test_custom_iterable(self):
        """Test with a custom iterable type."""
        assert self.square(self.custom_iterable) == CustomIterable([1, 4, 9])
        assert self.multiply(self.custom_iterable,
                             5) == CustomIterable([5, 10, 15])

    def test_basetype_exclusion(self):
        """Test that basetyped objects are treated as scalars."""
        # String should be treated as scalar when basetype=str
        assert self.get_length(self.string) == 5
        assert self.get_length(self.string_list) == [5, 5]

    def test_multiple_basetype_exclusion(self):
        """Test with multiple basetype exclusions."""
        # String should be treated as scalar with path decorator
        assert self.process_path(self.string) == f'Processing {self.string}'
        # Path object should be treated as scalar with path decorator
        assert self.process_path(
            self.path_obj) == f'Processing {self.path_obj}'
        # List of strings should be processed element-wise
        assert self.process_path(
            ['file1.txt',
             'file2.txt']) == ['Processing file1.txt', 'Processing file2.txt']
        # List of Path objects should be processed element-wise
        expected = [
            f'Processing {self.path_list[0]}',
            f'Processing {self.path_list[1]}'
        ]
        assert self.process_path(self.path_list) == expected

    def test_nested_iterables(self):
        """Test handling of nested iterables."""

        # Define a custom function that handles lists for this test
        @evn.iterize_on_first_param
        def sum_list(x):
            return sum(x) if isinstance(x, list) else x

        assert sum_list(self.nested_lists) == [3, 7]

    def test_decorator_preserves_metadata(self):
        """Test that the decorator preserves function metadata."""
        decorated = evn.iterize_on_first_param(self.original_func)

        assert decorated.__name__ == 'original_func'
        assert decorated.__doc__ == 'Test docstring.'

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

    def test_remove_first_nonempty(self):
        """Test with a non-empty iterable."""
        assert self.remove_first(self.string) == 'ello'
        assert self.remove_first(self.string_list) == ['ello', 'orld']
        assert self.remove_first(self.string_list +
                                 ['a', '']) == ['ello', 'orld']


class TestIterizeableFunction(unittest.TestCase):
    """Test suite for the evn.is_iterizeable helper function."""

    def setUp(self):
        """Set up test data."""
        self.list_data = [1, 2, 3]
        self.string = 'hello'
        self.integer = 42
        self.path_obj = Path('test.txt')

    def test_basic_iterizeable(self):
        """Test basic evn.is_iterizeable function without basetype."""
        assert evn.is_iterizeable(self.list_data) is True
        assert evn.is_iterizeable(self.string) is False
        assert evn.is_iterizeable(self.string,
                                  basetype=None) is True  # String is iterable
        assert evn.is_iterizeable(self.integer) is False

    def test_iterizeable_with_basetype(self):
        """Test evn.is_iterizeable function with basetype parameter."""
        # String should not be considered iterable when basetype includes str
        assert evn.is_iterizeable(self.string, basetype=str) is False
        assert evn.is_iterizeable(self.list_data, basetype=str) is True

        # Path should not be considered iterable when basetype includes Path
        assert evn.is_iterizeable(self.path_obj, basetype=Path) is False

        # Multiple basetypes
        assert evn.is_iterizeable(self.string, basetype=(str, Path)) is False
        assert evn.is_iterizeable(self.path_obj, basetype=(str, Path)) is False
        assert evn.is_iterizeable(self.list_data, basetype=(str, Path)) is True


def test_subscriptable_for_attributes__getitem__():

    @evn.subscriptable_for_attributes
    class Foo:
        a, b, c = 6, 7, 8

    assert Foo()['a'] == 6
    assert Foo()['a b'] == (6, 7)


def test_subscriptable_for_attributes_enumerate():

    @evn.subscriptable_for_attributes
    class Foo:

        def __init__(self):
            self.a, self.b, self.c = range(6), range(1, 7), range(10, 17)

    foo = Foo()
    for (i, a, b, c), e, f, g in zip(foo.enumerate('a b c'), range(6),
                                     range(1, 7), range(10, 17)):
        assert a == e and b == f and c == g


def test_subscriptable_for_attributes_enumerate_noarg():

    @evn.subscriptable_for_attributes
    class Foo:

        def __init__(self):
            self.a, self.b, self.c = range(6), range(1, 7), range(10, 17)

    foo = Foo()
    for (i, a, b, c), e, f, g in zip(foo.enumerate(), range(6), range(1, 7),
                                     range(10, 17)):
        assert a == e and b == f and c == g


def test_subscriptable_for_attributes_groupby():

    @evn.subscriptable_for_attributes
    class Foo:

        def __init__(self):
            self.a, self.b, self.c, self.group = range(6), range(1, 7), range(
                10, 17), 'aaabbb'

    foo = Foo()
    # for g, a, b, c in foo.groupby('group', 'a b c'):
    # ic(g, a, b, c)
    v = list(foo.groupby('group', 'a c'))
    assert v == [('a', (0, 1, 2), (10, 11, 12)),
                 ('b', (3, 4, 5), (13, 14, 15))]
    v = list(foo.groupby('group'))
    assert v == [
        ('a', evn.Bunch(a=(0, 1, 2), b=(1, 2, 3), c=(10, 11, 12))),
        ('b', evn.Bunch(a=(3, 4, 5), b=(4, 5, 6), c=(13, 14, 15))),
    ]


def test_subscriptable_for_attributes_fzf():

    @evn.subscriptable_for_attributes
    class Foo:

        def __init__(self):
            self.london, self.france, self.underpants = 'london', 'france', 'underpants'
            self._ignored = 'ignored'
            self.redundand1, self.redundand2 = 'fo'

    foo = Foo()
    assert foo.fzf('lon') == 'london'
    assert foo.fzf('fr') == 'france'
    assert foo.fzf('underpants') == 'underpants'
    assert foo.fzf('undpant loon frnc') == ('underpants', 'london', 'france')
    with pytest.raises(AttributeError):
        foo.fzf('notthere')
    with pytest.raises(AttributeError):
        foo.fzf('lndon')  # first two must match
    with pytest.raises(AttributeError):
        foo.fzf('')
    with pytest.raises(AttributeError):
        foo.fzf('_ignor')
    with pytest.raises(AttributeError):
        foo.fzf('redun')
    assert foo.fzf('red1') == 'f'


def test_getitem_picklable():

    @evn.subscriptable_for_attributes
    class Foo:

        def __init__(self):
            self.a, self.b, self.c = range(6), range(1, 7), range(10, 17)

    foo = Foo()
    assert foo.pick('a b').keys() == {'a', 'b'}


def test_safe_lru_cache():
    ncompute = 0

    @evn.safe_lru_cache(maxsize=32)
    def example(x):
        nonlocal ncompute
        ncompute += 1
        return x * 2

    example(2)  #  Computing 2
    example(2)  # No print (cached)
    example([1, 2, 3])  #  Computing [1, 2, 3]
    example([1, 2, 3])  #  Computing [1, 2, 3] (because list is unhashable)
    assert ncompute == 3


def test_safe_lru_cache_noarg():
    ncompute = 0

    @evn.safe_lru_cache
    def example(x):
        nonlocal ncompute
        ncompute += 1
        return x * 2

    example(2)  #  Computing 2
    example(2)  # No print (cached)
    example([1, 2, 3])  #  Computing [1, 2, 3]
    example([1, 2, 3])  #  Computing [1, 2, 3] (because list is unhashable)
    assert ncompute == 3


def test_is_safe_lru_cache_necessary():

    @evn.ft.lru_cache
    def example(x):
        return x * 2

    with pytest.raises(TypeError):
        example([1, 2, 3])


if __name__ == '__main__':
    main()
