from collections import OrderedDict
import operator
import unittest
import pytest
import evn

np = evn.lazyimport('numpy')

config_test = evn.Bunch(
    # re_only=['test_generic_get_items'],
    re_exclude=[], )


def main():
    evn.testing.quicktest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
        use_test_classes=True,
    )


def test_generic_get_items():
    foo = dict(a=1, b_=3)
    assert evn.decon.generic_get_items(foo) == [('a', 1)]

    class Foo:
        pass

    foo = Foo()
    foo.a, foo.b, foo._c = 1, 1, 1
    assert evn.decon.generic_get_items(foo) == [('a', 1), ('b', 1)]
    assert evn.decon.generic_get_items([0, 1, 2]) == [(0, 0), (1, 1), (2, 2)]

    class Bar:
        a: int = 1
        _b: int = 2
        c_: int = 3

    bar = Bar()
    assert evn.decon.generic_get_items(bar) == [('a', 1)]


@evn.item_wise_operations
class EwiseDict(dict):
    pass


def test_item_wise_no_args():

    @evn.item_wise_operations
    class EwiseDictonly(dict):
        pass

    if evn.installed.numpy:
        assert 'npwise' in dir(EwiseDictonly)
    assert 'dictwise' not in dir(EwiseDictonly)
    assert 'mapwise' in dir(EwiseDictonly)
    assert 'valwise' in dir(EwiseDictonly)


def test_item_wise_resulttypes():
    with pytest.raises((TypeError, KeyError)):

        @evn.item_wise_operations(result_types='foo')
        class EwiseDictBad(dict):
            pass

    @evn.item_wise_operations(result_types='np dict')
    class EwiseDictonly(dict):
        pass

    if evn.installed.numpy:
        assert 'npwise' in dir(EwiseDictonly)
    assert 'dictwise' in dir(EwiseDictonly)
    assert 'mapwise' not in dir(EwiseDictonly)
    assert 'valwise' not in dir(EwiseDictonly)
    instance = EwiseDictonly()
    assert hasattr(instance, 'npwise')
    assert hasattr(instance, 'dictwise')
    assert not hasattr(instance, 'mapwise')
    assert not hasattr(instance, 'valwise')


def test_item_wise():
    b = EwiseDict(zip('abcdefg', ([] for _ in range(7))))
    assert all(b.valwise == [])
    r = b.mapwise.append(1)
    assert all(b.valwise == [1])
    assert b['a'] == [1]


def test_item_wise_accum():
    b = EwiseDict(zip('abcdefg', range(7)))
    assert isinstance(b.mapwise + 10, evn.Bunch)
    assert isinstance(b.valwise + 10, list)
    if evn.installed.numpy:
        assert isinstance(b.npwise + 10, np.ndarray)


def test_item_wise_multi():
    b = EwiseDict(zip('abcdefg', ([] for _ in range(7))))
    assert b.mapwise == []
    with pytest.raises(ValueError):
        r = b.mapwise.append(1, 2)
    b.mapwise.append(*range(7))
    assert list(b.values()) == [[i] for i in range(7)]


def test_item_wise_equal():
    b = EwiseDict(zip('abcdefg', ([] for _ in range(7))))
    assert b.mapwise == []
    b.mapwise.append(*range(7))
    eq4 = b.mapwise == [4]
    assert list(eq4.values()) == [0, 0, 0, 0, 1, 0, 0]
    assert not any((b.mapwise == 3).values())


def test_item_wise_add():
    b = EwiseDict(zip('abcdefg', range(7)))
    assert (b.valwise == 4) == [0, 0, 0, 0, 1, 0, 0]
    assert not any(b.valwise == 'ss')
    c = b.mapwise + 7
    b.mapwise += 7
    assert b == c
    if evn.installed.numpy:
        d = b.npwise - 4
        e = 4 - b.npwise
        assert np.all(d == -e)


def test_item_wise_contains():
    b = EwiseDict(zip('abcdefg', [[i] for i in range(7)]))
    with pytest.raises(ValueError):
        contains = b.valwise.__contains__(4)
    contains = b.valwise.contains(4)
    assert contains == [0, 0, 0, 0, 1, 0, 0]


def test_item_wise_contained_by():
    b = EwiseDict(zip('abcdefg', range(7)))
    contained = b.valwise.contained_by([1, 2, 3])
    assert contained == [0, 1, 1, 1, 0, 0, 0]


def test_item_wise_indexing():
    if evn.installed.numpy:
        dat = np.arange(7 * 4).reshape(7, 4)
        b = EwiseDict(zip('abcdefg', dat))
        indexed = b.npwise[1]
        assert np.all(indexed == dat[:, 1])


def test_item_wise_slicing():
    if evn.installed.numpy:
        dat = np.arange(7 * 4).reshape(7, 4)
        b = EwiseDict(zip('abcdefg', dat))
        indexed = b.npwise[1:3]
        assert np.all(indexed == dat[:, 1:3])


def test_item_wise_call_operator():
    if evn.installed.numpy:
        dat = np.arange(7 * 4).reshape(7, 4)
        b = EwiseDict(zip('abcdefg', dat))
        c = b.mapwise(lambda x: list(map(int, x)))
        d = c.mapwise(np.array, dtype=float)
        assert np.all(b.npwise == d)


@evn.item_wise_operations
@evn.mutablestruct
class Foo:
    a: list
    b: list

    def c(self):
        pass


def test_item_wise_attrs():
    foo = Foo(a=[], b=[])
    foo.mapwise.append(5, 7)
    assert foo.a == [5], foo.b == [7]
    with pytest.raises(ValueError):
        foo.mapwise.append(1, 2, 3, 4)


@evn.item_wise_operations
@evn.struct
class Bar:
    a: list
    b: list

    def c(self):
        pass


@pytest.mark.skip
def test_item_wise_slots():
    foo = Bar(a=[], b=[])
    foo.mapwise.append(5, 7)
    assert foo.a == [5], foo.b == [7]
    with pytest.raises(ValueError):
        foo.mapwise.append(1, 2, 3, 4)


def test_item_wise_kw_call():
    x = Foo([], [])
    x.mapwise.append(dict(b=2, a=1))
    x.mapwise.append(**dict(b=2, a=1))
    # x.mapwise.append(a=1, b=2)
    assert x.a == [1, 1] and x.b == [2, 2]


############################ ai gen tests ######################


class TestElementWiseOperations(unittest.TestCase):
    """Test cases for item_wise_operations decorator and related functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dict = EwiseDict({
            'a': 1,
            'b': 2,
            'c': 3,
            '_hidden': 4,  # should be skipped in element-wise operations
        })

        # For testing container operations
        self.test_container_dict = EwiseDict({
            'a': [1, 2, 3],
            'b': [2, 3, 4],
            'c': [3, 4, 5]
        })
        # For testing with objects

        @evn.item_wise_operations
        class Metrics(dict):
            pass

        self.metrics = Metrics({
            'accuracy': 0.95,
            'precision': 0.87,
            'recall': 0.92,
            'f1': 0.89
        })

    def test_item_wise_basic(self):
        """Test basic mapwise operations."""
        # Test addition with single value
        result = self.test_dict.mapwise.__add__(10)
        self.assertIsInstance(result, evn.Bunch)
        self.assertEqual(result.a, 11)
        self.assertEqual(result.b, 12)
        self.assertEqual(result.c, 13)
        self.assertNotIn('_hidden', result)

        # Test with custom method
        result = self.test_dict.mapwise.__getattr__(lambda x, y: x * y)(5)
        self.assertEqual(result.a, 5)
        self.assertEqual(result.b, 10)
        self.assertEqual(result.c, 15)

    def test_valwise_basic(self):
        """Test basic valwise operations."""
        # Test multiplication with single value
        result = self.test_dict.valwise.__mul__(2)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [2, 4, 6])

        # Test with no arguments (calls method with no args)
        test_dict = EwiseDict({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        result = test_dict.valwise.__getattr__(len)()
        self.assertEqual(result, [2, 2, 2])

    def test_npwise_basic(self):
        """Test basic npwise operations."""
        # Test subtraction
        if evn.installed.numpy:
            result = self.test_dict.npwise.__sub__(1)
            self.assertIsInstance(result, np.ndarray)
            assert np.allclose(result, np.array([0, 1, 2]))

            # Test negative operation (unary)
            result = self.test_dict.npwise.__neg__()
            assert np.allclose(result, np.array([-1, -2, -3]))

    def test_multiple_args(self):
        """Test operations with multiple arguments."""
        # Test with exact number of arguments
        result = self.test_dict.mapwise.__getattr__(operator.add)(10, 20, 30)
        self.assertEqual(result.a, 11)
        self.assertEqual(result.b, 22)
        self.assertEqual(result.c, 33)

        # Test with wrong number of arguments
        with self.assertRaises(TypeError):
            self.test_dict.mapwise.__getattr__(operator.add)([10, 20])

    def test_binary_operations(self):
        """Test binary operations (add, sub, mul, etc.)."""
        # Addition
        result = self.test_dict.mapwise + 5
        self.assertEqual(result.a, 6)
        self.assertEqual(result.b, 7)
        self.assertEqual(result.c, 8)

        # Subtraction
        result = self.test_dict.mapwise - 1
        self.assertEqual(result.a, 0)
        self.assertEqual(result.b, 1)
        self.assertEqual(result.c, 2)

        # Right subtraction (special case)
        # result = 10 - self.test_dict.mapwise
        # self.assertEqual(result.a, 9)
        # self.assertEqual(result.b, 8)
        # self.assertEqual(result.c, 7)

        # Multiplication
        result = self.test_dict.mapwise * 3
        self.assertEqual(result.a, 3)
        self.assertEqual(result.b, 6)
        self.assertEqual(result.c, 9)

        # Division
        result = self.test_dict.mapwise / 2
        self.assertEqual(result.a, 0.5)
        self.assertEqual(result.b, 1.0)
        self.assertEqual(result.c, 1.5)

    def test_container_operations(self):
        """Test container operations (contains, contained_by)."""
        # Contains operation
        result = self.test_container_dict.mapwise.contains(2)
        self.assertEqual(result.a, True)
        self.assertEqual(result.b, True)
        self.assertEqual(result.c, False)

        # Contained_by operation
        container = [1, 2, 3, 4]
        testmap = EwiseDict(a=1, b=3, c=7)
        result = testmap.mapwise.contained_by(container)
        self.assertEqual(result.a,
                         True)  # all elements in [1,2,3] are in container
        self.assertEqual(result.b,
                         True)  # all elements in [2,3,4] are in container
        self.assertEqual(result.c, False)  # 5 is not in container

        # Test direct __contains__ (should raise error)
        with self.assertRaises(ValueError):
            2 in self.test_container_dict.mapwise

    def test_method_calls(self):
        """Test calling methods on elements."""
        dict_of_lists = EwiseDict({
            'a': [1, 2, 3],
            'b': [4, 5],
            'c': [6, 7, 8, 9]
        })

        # Call len() on each element
        result = dict_of_lists.mapwise.__getattr__('__len__')()
        self.assertEqual(result.a, 3)
        self.assertEqual(result.b, 2)
        self.assertEqual(result.c, 4)

    def test_accumulators(self):
        """Test different accumulator types."""
        # BunchAccumulator (mapwise)
        mapwise_result = self.metrics.mapwise * 100
        self.assertIsInstance(mapwise_result, evn.Bunch)
        self.assertEqual(mapwise_result.accuracy, 95)
        self.assertEqual(mapwise_result.precision, 87)

        # ListAccumulator (valwise)
        valwise_result = self.metrics.valwise * 100
        self.assertIsInstance(valwise_result, list)
        self.assertEqual(valwise_result, [95, 87, 92, 89])

        # NumpyAccumulator (npwise)
        # npwise_result = self.metrics.npwise * 100
        # self.assertIsInstance(npwise_result, np.ndarray)
        # np.testing.assert_array_almost_equal(npwise_result, np.array([95, 87, 92, 89]))

    def test_set_values(self):
        """Test setting values through the descriptor."""
        # Using a mapping
        self.test_dict.mapwise = {'a': 10, 'b': 20, 'c': 30}
        self.assertEqual(self.test_dict['a'], 10)
        self.assertEqual(self.test_dict['b'], 20)
        self.assertEqual(self.test_dict['c'], 30)

        # Using a sequence
        test_dict2 = EwiseDict({'x': 0, 'y': 0, 'z': 0})
        test_dict2.mapwise = [5, 6, 7]
        self.assertEqual(test_dict2['x'], 5)
        self.assertEqual(test_dict2['y'], 6)
        self.assertEqual(test_dict2['z'], 7)

    def test_real_world_scenario(self):
        """Test a realistic scenario with the decorator."""

        @evn.item_wise_operations
        class ExperimentResults(OrderedDict):
            pass

        # Create test data simulating experiment results
        results = ExperimentResults({
            'exp1': {
                'accuracy': 0.85,
                'runtime': 120
            },
            'exp2': {
                'accuracy': 0.92,
                'runtime': 150
            },
            'exp3': {
                'accuracy': 0.78,
                'runtime': 90
            },
        })

        # Extract a specific metric across all experiments
        get_accuracy = lambda experiment_data: experiment_data['accuracy']
        accuracies = results.valwise.__getattr__(get_accuracy)()
        self.assertEqual(accuracies, [0.85, 0.92, 0.78])

        # Compute average runtime
        if evn.installed.numpy:
            get_runtime = lambda experiment_data: experiment_data['runtime']
            runtimes = results.npwise.__getattr__(get_runtime)()
            self.assertEqual(np.mean(runtimes), 120.0)

        # Find best experiment by accuracy
        accuracies_dict = results.mapwise.__getattr__(get_accuracy)()
        best_exp = max(accuracies_dict.items(), key=lambda x: x[1])[0]
        self.assertEqual(best_exp, 'exp2')


if __name__ == '__main__':
    main()
