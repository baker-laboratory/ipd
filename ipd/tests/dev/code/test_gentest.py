import os
import pathlib
import tempfile
import ipd

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_make_testfile(tmpdir)

def test_make_testfile(tmpdir):
    sourcefile = os.path.join(tmpdir, 'sourcefile.py')
    testfile = os.path.join(tmpdir, 'testfile.py')
    pathlib.Path(sourcefile).write_text(CODE)
    os.system(f'cp {ipd.projdir}/../pyproject.toml {tmpdir}')
    ipd.dev.code.make_testfile(sourcefile, testfile)
    code = pathlib.Path(testfile).read_text()
    if code != EXPECTED:
        pathlib.Path(f'{tmpdir}/expected.py').write_text(EXPECTED)
        print(code)
        os.system(f'diff {tmpdir}/expected.py {tmpdir}/testfile.py')
    assert code == EXPECTED

CODE = '''
from pathlib import Path
from os.path import exists

def some_func(a, *, b=5) -> None:
    pass

class Bar:
    def baz():
        pass
    def double(fug: int) -> int:
        return fug * 2
'''

EXPECTED = '''import pytest

def main():
    test_some_func()
    test_Bar()

def test_some_func():
    # some_func(a, *, b=5) -> None
    assert 0

def test_Bar():
    # Bar.baz()
    # Bar.double(fug: int) -> int
    assert 0

if __name__ == '__main__':
    main()
'''

if __name__ == '__main__':
    main()
