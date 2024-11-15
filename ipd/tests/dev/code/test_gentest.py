import os
import pathlib

from assertpy import assert_that as at
import pytest

import ipd

def main():
    ipd.tests.maintest(globals())

@pytest.mark.xfail
def test_qualname_of_file():
    with pytest.raises(ValueError):
        ipd.dev.qualname_of_file('foo')
    for fname, expected in [
        ('sourcefile.py', 'sourcefile'),
        ('/foo/bar/baz.py', 'baz'),
        ('/home/sheffler/rfdsym/rf_diffusion/foo.py', 'rf_diffusion.foo'),
        ('/home/sheffler/rfdsym/lib/rf2aa/rf2aa/foo/bar.py', 'rf2aa.foo.bar'),
        ('/home/sheffler/rfdsym/lib/rf2aa/lib/ipd/foo.py', 'foo'),
            # ('/home/sheffler/rfdsym/lib/rf2aa/lib/ipd/ipd/foo/bar/baz.py', 'ipd.foo.bar.baz'),
    ]:
        if expected.startswith('rf2aa') and 'rf2aa' not in __file__: continue
        if expected.startswith('rf_diffusion') and 'rf_diffusion' not in __file__: continue
        at(ipd.dev.qualname_of_file(fname)).is_equal_to(expected)

def test_make_testfile(tmpdir):
    sourcefile = os.path.join(tmpdir, 'sourcefile.py')
    testfile = os.path.join(tmpdir, 'testfile.py')
    pathlib.Path(sourcefile).write_text(CODE)
    os.system(f'cp {ipd.projdir}/../pyproject.toml {tmpdir}')
    ipd.dev.make_testfile(sourcefile, testfile)
    code = pathlib.Path(testfile).read_text()
    code = code.replace(sourcefile, 'SOURCEFILE')
    pathlib.Path(testfile).write_text(code)
    pathlib.Path(f'{tmpdir}/expected.py').write_text(EXPECTED)
    diff = ipd.dev.run(f'diff -wB {tmpdir}/expected.py {tmpdir}/testfile.py', errok=True)
    if diff:
        print('-' * 80)
        print(code)
        print('-' * 80)
        print(diff)
        print('-' * 80)
    assert code == EXPECTED or not diff

CODE = """
from pathlib import Path
from os.path import exists
from collections import namedtuple

Return = namedtuple('Return', 'value error')

def some_func(a, *, b=5) -> None:
    pass

def some_other_func() -> Return:
    pass

class Bar:
    def baz():
        pass
    def double(fug: int) -> int:
        return fug * 2

class SomeOtherClass:
    def triple(bar: Bar) -> Bar:
        return bar
"""

EXPECTED = """
import pytest

import ipd

def main():
    ipd.tests.testmain(namespace=globals())

def test_some_func():
    # some_func(a, *, b=5) -> None
    assert 0

def test_some_other_func():
    # some_other_func() -> sourcefile.Return
    assert 0

def test_Bar():
    # Bar.baz()
    # Bar.double(fug: int) -> int
    assert 0

def test_SomeOtherClass():
    # SomeOtherClass.triple(bar: sourcefile.Bar) -> sourcefile.Bar
    assert 0

# please develop a comprehensive set of pytest tests, including edge cases and input validation, for the code in file:
# SOURCEFILE, specifically the functions, classes and methods specified below:
# please develop a comprehensive set of pytest tests, including edge cases and input validation, for the function some_func with the following signature: sourcefile.some_func(a, *, b=5) -> None
# please develop a comprehensive set of pytest tests, including edge cases and input validation, for the function some_other_func with the following signature: sourcefile.some_other_func() -> sourcefile.Return
# please write a comprehensive set of pytest tests, including edge cases and input validation, for the class Bar with the following member function signatures:
#        sourcefile.Bar.baz()
#        sourcefile.Bar.double(fug: int) -> int
# please write a comprehensive set of pytest tests, including edge cases and input validation, for the class SomeOtherClass with the following member function signatures:
#        sourcefile.SomeOtherClass.triple(bar: sourcefile.Bar) -> sourcefile.Bar

if __name__ == '__main__':
    main()

""".lstrip()

if __name__ == '__main__':
    main()
