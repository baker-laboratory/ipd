import types
import pytest
import evn


def main():
    evn.testing.quicktest(
        namespace=globals(),
        config=evn.Bunch(re_only=[], re_exclude=[]),
        verbose=1,
        check_xfail=False,
        chrono=False,
    )


def test_deco():
    result = ''

    @evn.make_decorator
    def foo(func, *args, **kwargs):
        nonlocal result
        result += 'prefoo'
        return func(*args, **kwargs)

    def baz():
        pass

    @foo
    def bar():
        nonlocal result
        result += 'bar'

    bar()
    assert result == 'prefoobar'  # Check the result of the decorator


def test_deco_config_default():
    result = ''

    @evn.make_decorator(msg='baz')
    def foo(func, *args, msg, **kwargs):
        nonlocal result
        result += msg
        return func(*args, **kwargs)

    @foo
    def bar():
        nonlocal result
        result += 'bar'

    bar()
    assert result == 'bazbar'  # Check the result of the decorator
    result = ''


def test_deco_config():
    result = ''

    @evn.make_decorator(msg='baz')
    def foo(func, *args, msg=None, **kwargs):
        nonlocal result
        result += msg
        return func(*args, **kwargs)

    @foo(msg='aaaaaa')
    def aaa():
        nonlocal result
        result += 'bar'

    aaa()
    assert result == 'aaaaaabar'  # Check the result of the decorator with different message


def test_deco_not_callable_error():
    with pytest.raises(TypeError):

        @evn.make_decorator('baz')
        def foo(func, *args, **kwargs):
            return func(*args, **kwargs)


def test_deco_config_kwargs_error():
    result = ''

    @evn.make_decorator(msg='baz')
    def foo(func, *args, msg='', **kwargs):
        nonlocal result
        result += msg
        return func(*args, **kwargs)

    with pytest.raises(TypeError):

        @foo(msg2='aaaaaa')
        def aaa():
            nonlocal result
            result += 'bar'


def test_deco_config_args_error():
    result = ''

    @evn.make_decorator(msg='baz')
    def foo(func, *args, msg='', **kwargs):
        nonlocal result
        result += msg
        return func(*args, **kwargs)

    with pytest.raises(TypeError):

        @foo(msg2='aaaaaa')
        def aaa():
            nonlocal result
            result += 'bar'


def test_deco_method():

    @evn.make_decorator(extra=0)
    def plus_this(func, *args, extra, **kwargs):
        return func(*args, **kwargs) + extra

    class Foo:

        @plus_this(extra=3)
        def add(self, a, b):
            return a + b

        def mul(self, a, b):
            return a * b

    foo = Foo()
    assert foo.add(1, 2) == 6
    assert foo.mul(1, 2) == 2


def test_deco_class():

    class Foo:

        def add(self, a, b):
            return a + b

        def mul(self, a, b):
            return a * b

    foo = Foo()
    assert foo.add(1, 2) == 3
    assert foo.mul(1, 2) == 2

    @evn.make_decorator(extra=0)
    def plus_this(func, *args, extra, **kwargs):
        return func(*args, **kwargs) + extra

    @plus_this(extra=5)
    class Bar:

        def add(self, a, b):
            return a + b

        def mul(self, a, b):
            return a * b

    bar = Bar()
    assert bar.add(1, 2) == 8
    assert bar.mul(1, 2) == 7


def test_basic_function_decorator():
    log = []

    @evn.make_decorator
    def logger(func, *args, **kwargs):
        log.append(f'calling {func.__name__}')
        return func(*args, **kwargs)

    @logger
    def foo():
        log.append('foo ran')
        return 42

    result = foo()
    assert result == 42
    assert log == ['calling foo', 'foo ran']


def test_configurable_decorator_default_and_override():
    log = []

    @evn.make_decorator(prefix='>> ')
    def trace(func, *args, prefix, **kwargs):
        log.append(prefix + func.__name__)
        return func(*args, **kwargs)

    @trace
    def one():
        log.append('one')

    @trace(prefix='** ')
    def two():
        log.append('two')

    one()
    two()
    assert log == ['>> one', 'one', '** two', 'two']


def test_strict_mode_disallows_unknown_config():

    @evn.make_decorator(msg='ok', strict=True)
    def f(func, *args, msg, **kwargs):
        return func(*args, **kwargs)

    with pytest.raises(TypeError):

        @f(extra='bad')
        def nope():
            pass


def test_non_callable_userwrap_raises():
    with pytest.raises(TypeError):
        evn.make_decorator(123)


def test_decorator_metadata_preserved():

    @evn.make_decorator
    def dummy(func, *args, **kwargs):
        return func(*args, **kwargs)

    @dummy
    def my_func():
        """This is a docstring."""
        return 7

    assert my_func.__name__ == 'my_func'
    assert my_func.__doc__ == 'This is a docstring.'
    assert isinstance(my_func, types.FunctionType)  # Still a function


def test_decorator_on_instance_method():

    @evn.make_decorator(extra=1)
    def bump(func, *args, extra, **kwargs):
        return func(*args, **kwargs) + extra

    class Thing:

        @bump(extra=3)
        def do(self, x):
            return x

    t = Thing()
    assert t.do(4) == 7


def test_decorator_on_class_entirely():

    @evn.make_decorator(suffix=1)
    def plus(func, *args, suffix, **kwargs):
        return func(*args, **kwargs) + suffix

    @plus(suffix=5)
    class Math:

        def add(self, x, y):
            return x + y

        def mul(self, x, y):
            return x * y

    m = Math()
    assert m.add(1, 2) == 8
    assert m.mul(2, 3) == 11


def test_classmethod_and_staticmethod_wrapping():
    calls = []

    @evn.make_decorator(tag='')
    def logcall(func, *args, tag, **kwargs):
        calls.append(f'{tag}:{func.__name__}')
        return func(*args, **kwargs)

    @logcall(tag='X')
    class Example:

        @classmethod
        def cls_method(cls):
            return 'cls'

        @staticmethod
        def stat_method():
            return 'stat'

    assert Example.cls_method() == 'cls'
    assert Example.stat_method() == 'stat'
    assert calls == ['X:cls_method', 'X:stat_method']


def test_nested_configuration_application():

    @evn.make_decorator(greeting='hi')
    def greeter(func, *args, greeting, **kwargs):
        return f'{greeting}, {func(*args, **kwargs)}'

    @greeter
    def name():
        return 'Alice'

    @greeter(greeting='hello')
    def name2():
        return 'Bob'

    assert name() == 'hi, Alice'
    assert name2() == 'hello, Bob'


if __name__ == '__main__':
    main()
