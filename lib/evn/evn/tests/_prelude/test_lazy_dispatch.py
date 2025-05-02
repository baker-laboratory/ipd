# test_lazy_dispatch.py
import sys
import types
import pytest

import evn
from evn._prelude.lazy_dispatch import GLOBAL_DISPATCHERS, lazydispatch, LazyDispatcher

def main():
    import evn
    evn.testing.quicktest(namespace=globals())

def test_dispatch_deco():

    @lazydispatch
    def foo(obj):
        print(obj)

    assert isinstance(foo, LazyDispatcher)

def test_dispatch_deco_nest():

    @lazydispatch(object, scope='local')
    def foo(obj):
        print(obj)

    assert isinstance(foo, LazyDispatcher)

def test_dispatch_global_registry():
    GLOBAL_DISPATCHERS.clear()

    @lazydispatch(object, scope='local')
    def describe(obj):
        return 'default'

    @lazydispatch(list, scope='local')
    def describe(obj):
        return 'list'

    assert len(GLOBAL_DISPATCHERS) == 1

def test_dispatchers_match():
    GLOBAL_DISPATCHERS.clear()

    @lazydispatch(object, scope='local')
    def describe(obj):
        return 'default'

    describe1 = describe
    assert isinstance(describe1, LazyDispatcher)

    @lazydispatch(list, scope='local')
    def describe(obj):
        return 'list'

    describe2 = describe
    assert isinstance(describe2, LazyDispatcher)
    assert describe1 is evn.first(GLOBAL_DISPATCHERS.values())
    assert describe1 is describe2

    assert describe([1, 2]) == 'list', str(describe([1, 2]))
    assert describe(42) == 'default', str(describe(42))

def test_dispatch_default():
    GLOBAL_DISPATCHERS.clear()

    @lazydispatch(object, scope='local')
    def describe(obj):
        return 'default'

    describe1 = describe

    @lazydispatch(list, scope='local')
    def describe(obj):
        return 'list'

    describe2 = describe
    assert describe1 is describe2

    assert describe([1, 2]) == 'list'
    assert describe(42) == 'default'

def test_lazy_registration_numpy():
    numpy = pytest.importorskip('numpy')
    if 'numpy' in sys.modules: del sys.modules['numpy']

    @lazydispatch(object, scope='local')
    def describe(obj):
        return 'default'

    @lazydispatch('numpy.ndarray', scope='local')
    def describe(obj):
        return f'ndarray({obj.size})'

    assert describe(numpy.arange(3)) == 'ndarray(3)'

def test_scope_local_disambiguation():

    @lazydispatch(object, scope='local')
    def action(obj):
        return 'default'

    @lazydispatch('builtins.int', scope='local')
    def action(obj):
        return 'int'

    assert action(123) == 'int'
    assert action('hi') == 'default'

def test_unresolved_type_skips():

    @lazydispatch(object, scope='local')
    def handler(obj):
        return 'base'

    @lazydispatch('ghost.Type', scope='local')
    def handler(obj):
        return 'ghost'

    class Other:
        pass

    assert handler(Other()) == 'base'

def test_missing_dispatcher_errors():
    with pytest.raises(ValueError):

        @lazydispatch('foo.   Bar')
        def nothing(obj):
            return 'fail'

        print(nothing(5))

def test_predicate_registration():
    GLOBAL_DISPATCHERS.clear()

    @lazydispatch(object, scope='local')
    def describe(obj):
        return 'default'

    @lazydispatch(predicate=lambda x: isinstance(x, tuple), scope='local')
    def describe(obj):
        return 'tuple'

    assert callable(describe), str(describe)
    assert describe(5) == 'default'
    assert describe((15, 13)) == 'tuple'

def test_lazydispatch_int():

    @lazydispatch(int)
    def int_func(obj):
        return obj + 1

    assert int_func(5) == 6

def test_lazydispatch_int_type():

    @lazydispatch(int)
    def int_func2(obj):
        return obj + 1

    assert int_func2(5) == 6

    @lazydispatch(type)
    def int_func2(obj):
        return f'type: {str(obj)}'

    assert int_func2(5) == 6
    assert int_func2(int) == 'type: <class \'int\'>'

def test_lazydispatch_int_type_pred():

    @lazydispatch(int)
    def int_func3(obj):
        return obj + 1

    assert int_func3(5) == 6

    @lazydispatch(type)
    def int_func3(obj):
        return f'type: {str(obj)}'

    assert int_func3(5) == 6
    assert int_func3(int) == 'type: <class \'int\'>'

    @lazydispatch(predicate=lambda x: isinstance(x, list))
    def int_func3(obj):
        return f'list: {str(obj)}'

    assert int_func3(5) == 6
    assert int_func3(int) == 'type: <class \'int\'>'
    assert int_func3([1, 2, 3]) == 'list: [1, 2, 3]'

def test_lazydispatch_int_type_pred_func():

    @lazydispatch(object)
    def int_func4(obj):
        return str(obj)

    assert int_func4(5) == '5'

    @lazydispatch(type)
    def int_func4(obj):
        return f'type: {str(obj)}'

    assert int_func4(5) == '5'
    assert int_func4(int) == 'type: <class \'int\'>'

    @lazydispatch(predicate=lambda x: isinstance(x, list))
    def int_func4(obj):
        return f'list: {str(obj)}'

    assert int_func4(5) == '5'
    assert int_func4(int) == 'type: <class \'int\'>'
    assert int_func4([1, 2, 3]) == 'list: [1, 2, 3]'

    # return 0
    @lazydispatch(types.FunctionType)
    def int_func4(obj):
        return f'func: {str(obj)}'

    assert int_func4(5) == '5'
    assert int_func4(int) == 'type: <class \'int\'>'
    assert int_func4([1, 2, 3]) == 'list: [1, 2, 3]'
    assert int_func4(lambda: 'lambda').startswith('func:')

if __name__ == '__main__':
    main()
