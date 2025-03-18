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
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
    )

def test_get_all_annotations():

    class Base:
        a: int

    class Child(Base):
        b: str

    assert ipd.dev.get_all_annotations(Child) == {'a': int, 'b': str}

def test_eval_fstring():
    assert ipd.dev.eval_fstring('{x + y}', {'x': 2, 'y': 3}) == '5'

def test_printed_string():
    assert ipd.dev.printed_string('hello') == 'hello\n'

def test_strip_duplicate_spaces():
    assert ipd.dev.strip_duplicate_spaces('hello  world') == 'hello world'

def test_tobytes():
    assert ipd.dev.tobytes('hello') == b'hello'
    assert ipd.dev.tobytes(b'hello') == b'hello'

def test_tostr():
    assert ipd.dev.tostr(b'hello') == 'hello'
    assert ipd.dev.tostr('hello') == 'hello'

def test_toname():
    assert ipd.dev.toname('hello') == 'hello'
    assert ipd.dev.toname('hello#') is None

def test_toidentifier():
    assert ipd.dev.toidentifier('valid_name') == 'valid_name'
    assert ipd.dev.toidentifier('123invalid') is None

def test_find_close_argnames():
    for arg, pool, ref in [
        ('apple', ['apple', 'ape', 'apply', 'banana'], ['apple', 'apply']),
        ('dog', ['cat', 'car', 'bat'], []),
    ]:
        result = ipd.dev.find_close_argnames(arg, pool, n=99)
        assert result == ref, f'{arg} {result}'

if __name__ == '__main__':
    main()
