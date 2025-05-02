"""
usage: python run_tests_for_file.py [project name(s)] [file.py]
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
from time import perf_counter

t_start = perf_counter()

import sys
import subprocess
from ninja_import import ninja_import
from fnmatch import fnmatch
import functools
from collections import defaultdict
from assertpy import assert_that
from io import StringIO

spo = ninja_import('evn.tool.filter_python_output')
# set to manually specipy a command for a file
_overrides = {
    'noxfile.py': 'nox -- 3.13 all',
    'pyproject.toml': 'uv run validate-pyproject pyproject.toml',
}
# set to mannually map a file to another file
_file_mappings = {
    # 'pymol_selection_algebra.lark': ['evn/tests/sel/test_sel_pymol.py'],
    '*.sublime-project': 'ide/validate_sublime_project.py',
}
# postprocess command
_post = defaultdict(lambda: '')

def get_args(sysargv):
    """get command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('projects', type=str, nargs='+', default='')
    parser.add_argument('inputfile', type=str, default='')
    parser.add_argument('--pytest', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--uv', action='store_true')
    parser.add_argument('--python', type=str, default=sys.executable)
    parser.add_argument('--filter-output', action='store_true')
    args = parser.parse_args(sysargv[1:])
    return args.__dict__

@functools.cache
def file_has_main(fname):
    "check if file has a main block"
    if not os.path.exists(fname):
        return False
    with open(fname) as inp:
        for line in inp:
            if all([
                    line.startswith('if __name__ == '),
                    not line.strip().endswith('{# in template #}'),
                    'ignore' not in line,
            ]):
                return True
    return False

def test_testfile_of():
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

def testfile_of(projects, path, basename, debug=False, **kw) -> str:
    "find testfile for a given file"
    if basename.startswith('_'):
        return None  # type: ignore
    root = '/' if path and path[0] == '/' else ''
    spath = path.split('/')
    i = max(rindex(spath, proj) for proj in projects)
    # assert i >= 0, f'no {' or '.join(projects)} dir in {path}'
    if i < 0:
        pre, post = '', f'{path}/'
    else:
        # proj = spath[i]
        # print(spath[:i + 1], spath[i + 1:])
        pre, post = spath[:i + 1], spath[i + 1:]
        pre = f'{os.path.join(*pre)}/' if pre else ''
        post = f'{os.path.join(*post)}/' if post else ''
    # print(pre, post)
    t = f'{root}{pre}tests/{post}test_{basename}'
    return t

# def locate_fname(fname):
#     'locate file in sys.path'
#     if os.path.exists(fname): return fname
#     candidates = [fn for fn in evn.project_files() if fn.endswith(fname)]
#     if len(candidates) == 1: return candidates[0]
#     if len(candidates) == 0: raise FileNotFoundError(f'file {fname} not found in git project')
#     raise FileNotFoundError(f'file {fname} found ambiguous {candidates} in git project')
def dispatch(
        projects,
        fname,
        file_mappings=dict(),
        overrides=dict(),
        strict=True,
        **kw,
):
    "dispatch command for a given file. see above"
    # fname = locate_fname(fname)
    fname = os.path.relpath(fname)
    module_fname = '' if fname[:5] == 'test_' else fname
    path, basename = os.path.split(fname)
    for pattern in file_mappings:
        if fnmatch(fname, pattern):
            fname = file_mappings[pattern]
            path, basename = os.path.split(fname)
    if basename in overrides:
        return overrides[basename], _post[basename]
    if not strict and basename in file_mappings:
        assert len(file_mappings[basename]) == 1
        basename = file_mappings[basename][0]
        path, basename = os.path.split(basename)
    if not file_has_main(fname) and not basename.startswith('test_'):
        if testfile := testfile_of(projects, path, basename, **kw):
            if not os.path.exists(testfile) and fname.endswith('.py'):
                print('autogen test file', testfile)
                os.system(f'{sys.executable} -mevn create testfile {fname} {testfile}')
                os.system(f'subl {testfile}')
                sys.exit()
            fname = testfile
            path, basename = os.path.split(fname)
    cmd, post = make_build_cmds(fname, module_fname, **kw)
    return cmd, post

def make_build_cmds(
    fname,
    module_fname,
    uv,
    pytest_args='-x --disable-warnings -m "not nondeterministic" --doctest-modules --durations=7',
    pytest=False,
    python=None,
    verbose=False,
    **kw,
):
    basename = os.path.basename(fname)
    if uv: python = f'uv run --extra dev --python {python}'
    pypath = f'PYTHONPATH={":".join(p for p in sys.path if "python3" not in p)}'
    has_main = file_has_main(fname)
    is_test = basename.startswith('test_')
    if not is_test and 'doctest-mod' not in pytest_args:
        pytest_args += ' --doctest-modules'
    if fname.endswith('.rst'):
        cmd = f'{pypath} {python} -m doctest {module_fname}'
    elif is_test and (pytest or not has_main):
        if module_fname == fname: fname = ''
        if is_test and not has_main: print('running pytest because no main')
        cmd = f'{pypath} {python} -m pytest {pytest_args} {module_fname} {fname}'
    elif fname.endswith('.py') and has_main and basename != 'conftest.py':
        cmd = f'{pypath} {python} {fname}'
    else:
        cmd = f'{pypath} {python} -mpytest {pytest_args}'
    return cmd, _post[basename]

def run_commands(cmds, out):
    exitcodes = {}
    output = ''
    for cmd in cmds:
        if out is sys.__stdout__:
            exitcodes.setdefault(cmd, []).append(os.system(cmd))
        else:
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            output += result.stdout
            exitcodes.setdefault(cmd, []).append(result.returncode)
    return output

def main(projects, quiet=False, filter_output=False, inputfile=None, **kw):
    stdout = sys.stdout
    if filter_output: sys.__stderr__ = sys.stdout = sys.stderr = StringIO()
    try:
        if inputfile:
            cmd, post = dispatch(projects, inputfile, **kw)
            if os.path.basename(inputfile) == os.path.basename(__file__):
                test_testfile_of()
                sys.exit()
        else:
            cmd, post = f'{sys.executable} -mpytest', ''
        if not quiet:
            print('call:', sys.argv)
            print('cwd:', os.getcwd())
            print('cmd:', cmd)
        print(f'{" run_tests_for_file.py running cmd in cwd ":=^69}')

        output = run_commands([cmd, post], sys.stdout)
        print(output, end='')

        t_total = perf_counter() - t_start
        print(f'{f" run_tests_for_file.py done, time {t_total:7.3f} ":=^69}', flush=True)
    finally:
        if isinstance(sys.stdout, StringIO):
            sys.stdout.seek(0)
            stdout.write(sys.stdout.read())

if __name__ == '__main__':
    args = get_args(sys.argv)
    main(file_mappings=_file_mappings, overrides=_overrides, **args)
