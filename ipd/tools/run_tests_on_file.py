"""
usage: python run_tests_on_file.py [project name(s)] [file.py]

This script exists for easy editor integration with python test files. Dispatch:

1. If the file has a main block, run it with python
2. If the file is a test_* file without a main block, run it with pytest
3. If the file is not a test_* file and does not have a main block, look for a test_* file in tests with the same path. for example rf_diffusion/foo/bar.py will look for rf_diffusion/tests/foo/test_bar.py
4. If none of the above, or no file specified, run pytest

_overrides can be set to manually specipy a command for a file
_file_mappings can be set to mannually map a file to another file
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from time import perf_counter
from assertpy import assert_that

# set to manually specipy a command for a file
_overrides = {
    # "foo.py": "PYTHONPATH=.. python foo/bar.py -baz"
}

# set to mannually map a file to another file
_file_mappings = {
    'pymol_selection_algebra.lark': ['ipd/tests/sel/test_sel_pymol.py'],
}

# postprocess command
_post = defaultdict(lambda: "")

def get_args(sysargv):
    parser = argparse.ArgumentParser()
    parser.add_argument("projects", type=str, nargs='+', default='')
    parser.add_argument("testfile", type=str, default='')
    parser.add_argument("--pytest", action='store_true')
    parser.add_argument("--quiet", action='store_true')
    parser.add_argument("--filter_build_log", action='store_true')
    args = parser.parse_args(sysargv[1:])
    return args.__dict__

def file_has_main(fname):
    "check if file has a main block"
    if not os.path.exists(fname): return False
    with open(fname) as inp:
        for line in inp:
            if line.startswith("if __name__ == ") and not line.strip().endswith('{# in template #}'):
                return True
    return False

def test():
    tfile = testfile_of(['foo'], '/a/b/c/d/foo/e/f/g', 'h.py', debug=True)
    assert_that(tfile).is_equal_to('/a/b/c/d/foo/tests/e/f/g/test_h.py')

    tfile = testfile_of(['foo'], 'a/b/c/d/foo/e/f/g', 'h.py', debug=True)
    assert_that(tfile).is_equal_to('a/b/c/d/foo/tests/e/f/g/test_h.py')

    tfile = testfile_of(['foo', 'bar', 'baz'], '/a/foo/b/bar/c/baz/d', 'file.py', debug=True)
    assert_that(tfile).is_equal_to('/a/foo/b/bar/c/baz/tests/d/test_file.py')

    tfile = testfile_of(['foo', 'bar', 'baz'], 'a/foo/b/bar/c', 'file.py', debug=True)
    assert_that(tfile).is_equal_to('a/foo/b/bar/tests/c/test_file.py')

    tfile = testfile_of(['foo', 'bar', 'baz'], 'a/foo/b', 'file.py', debug=True)
    assert_that(tfile).is_equal_to('a/foo/tests/b/test_file.py')

    tfile = testfile_of(['foo', 'bar', 'baz'], 'foo/foo', 'file.py', debug=True)
    assert_that(tfile).is_equal_to('foo/foo/tests/test_file.py')

    tfile = testfile_of(['foo', 'bar', 'baz'], 'a/b/c', 'file.py', debug=True)
    assert_that(tfile).is_equal_to('tests/a/b/c/test_file.py')

    tfile = testfile_of(['foo', 'bar', 'baz'], '', 'file.py', debug=True)
    assert_that(tfile).is_equal_to('tests//test_file.py')

    print(__file__, 'tests pass')

def rindex(lst, val):
    try:
        return len(lst) - lst[-1::-1].index(val) - 1
    except ValueError:
        return -1

def testfile_of(projects, path, bname, debug=False, **kw) -> str:
    "find testfile for a given file"
    if bname.startswith('_'): return None  # type: ignore
    root = '/' if path and path[0] == '/' else ''
    spath = path.split('/')
    i = max(rindex(spath, proj) for proj in projects)
    # assert i >= 0, f'no {" or ".join(projects)} dir in {path}'
    if i < 0:
        pre, post = '', f'{path}/'
    else:
        proj = spath[i]
        # print(spath[:i + 1], spath[i + 1:])
        pre, post = spath[:i + 1], spath[i + 1:]
        pre = f'{os.path.join(*pre)}/' if pre else ''
        post = f'{os.path.join(*post)}/' if post else ''
    # print(pre, post)
    t = f'{root}{pre}tests/{post}test_{bname}'
    return t

def dispatch(
        projects,
        fname,
        pytest_args='-x --disable-warnings -m "not nondeterministic" --doctest-modules',
        file_mappings=dict(),
        overrides=dict(),
        strict=True,
        pytest=False,
        **kw,
):
    "dispatch command for a given file. see above"

    fname = os.path.relpath(fname)
    module_fname = '' if fname[:5] == 'test_' else fname
    path, bname = os.path.split(fname)

    if bname in overrides:
        oride = overrides[bname]
        return oride, _post[bname]

    if bname in file_mappings:
        assert len(file_mappings[bname]) == 1
        fname = file_mappings[bname][0]
        path, bname = os.path.split(fname)
    if not strict and bname in file_mappings:
        assert len(file_mappings[bname]) == 1
        bname = file_mappings[bname][0]
        path, bname = os.path.split(bname)

    if not file_has_main(fname) and not bname.startswith("test_"):
        testfile = testfile_of(projects, path, bname, **kw)
        if testfile:
            if not os.path.exists(testfile) and fname.endswith('.py'):
                print('autogen test file', testfile)
                os.system(f'{sys.executable} -mipd code make_testfile {fname} {testfile}')
                os.system(f'subl {testfile}')
            fname = testfile
            path, bname = os.path.split(fname)

    if bname == os.path.basename(__file__):
        test()
        sys.exit()

    if pytest or (not file_has_main(fname) and bname.startswith("test_")):
        cmd = f"{sys.executable} -m pytest {pytest_args} {module_fname} {fname}"
    elif fname.endswith(".py") and bname != 'conftest.py':
        cmd = f"PYTHONPATH=. {sys.executable} " + fname
    else:
        cmd = f"{sys.executable} -mpytest {pytest_arg --doctest-moduless}"
    return cmd, _post[bname]

def main(projects, quiet=False, filter_build_log=False, **kw):
    t = perf_counter()
    cmd, post = dispatch(projects, kw['testfile'], **kw) if kw['testfile'] else (f'{sys.executable} -mpytest',
                                                                                 '')
    if not quiet:
        print("call:", sys.argv)
        print("cwd:", os.getcwd())
        print("cmd:", cmd)
        print(f"{' run_tests_on_file.py running cmd in cwd ':=^69}")
        sys.stdout.flush()
    os.system(cmd)
    os.system(post)
    t = perf_counter() - t
    if filter_build_log:
        p = Path('sublime_build.log')
        assert p.exists()

    print(f"{f' run_tests_on_file.py done, time {t:7.3f} ':=^69}")

if __name__ == '__main__':
    args = get_args(sys.argv)
    main(file_mappings=_file_mappings, overrides=_overrides, **args)
