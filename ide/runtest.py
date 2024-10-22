"""
usage: python runtests.py [file.py]

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
from collections import defaultdict
from time import perf_counter

from icecream import ic

# set to manually specipy a command for a file
_overrides = {
    # "foo.py": "PYTHONPATH=.. python foo/bar.py -baz"
}

# set to mannually map a file to another file
_file_mappings = {
    # 'sym_slice.py': ['tests/sym/test_sym_indep.py'],
    # 'sym_slice.py': ['../rf2aa/tests/sym/test_rf2_sym_manager.py'],
    # 'sym_adapt.py': ['../rf2aa/tests/sym/test_rf2_sym_manager.py'],
    # 'sym_manager.py': ['../rf2aa/tests/sym/test_rf2_sym_manager.py'],
    # 'sym_manager.py': ['../rf2aa/tests/sym/test_sym_manager.py'],
    # 'sym_manager.py': ['../rf2aa/tests/sym/test_sym_check.py'],
    # 'sym_adapt.py': ['../rf2aa/tests/sym/test_sym_check.py'],
    # 'sym_slice.py': ['../rf2aa/tests/sym/test_sym_check.py'],
    # 'sym_adapt.py': ['../rf2aa/tests/sym/test_sym_manager.py'],
    # 'sym_util.py': ['../rf2aa/tests/sym/test_rf2_sym_manager.py'],
    # 'thgeom.py': ['../rf2aa/tests/sym/test_rf2_sym_manager.py'],
}

# postprocess command
_post = defaultdict(lambda: "")

def get_args(sysargv):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--projname", default='', help='name of project')
    parser.add_argument("testfile", type=str, default='')
    args = parser.parse_args(sysargv[1:])
    return args.__dict__

def file_has_main(fname):
    "check if file has a main block"
    with open(fname) as inp:
        for line in inp:
            if line.startswith("if __name__ == "):
                return True
    return False

def testfile_of(path, bname, **kw):
    "find testfile for a given file"
    ic(path, bname)
    if path.startswith('../rf2aa/'):
        t = f'../rf2aa/tests/{path[9:]}/test_{bname}'
    elif path.startswith('../lib/rf2aa/'):
        t = f'../lib/rf2aa/rf2aa/tests/{path[19:]}/test_{bname}'
    elif path.startswith('../../rf/rfsym/rf2aa/'):
        t = f'../rf2aa/tests/{path[21:]}/test_{bname}'
    else:
        t = f'tests/{path}/test_{bname}'
    ic(t)
    if os.path.exists(t):
        return t

def dispatch(
        fname,
        pytest_args='--disable-warnings -m "not nondeterministic"',
        file_mappings=dict(),
        overrides=dict(),
        strict=True,
        **kw,
):
    "dispatch command for a given file. see above"

    fname = os.path.relpath(fname)
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
        testfile = testfile_of(path, bname, **kw)
        if testfile:
            fname = testfile
            path, bname = os.path.split(fname)

    if not file_has_main(fname) and bname.startswith("test_"):
        cmd = "PYTHONPATH=.. CUDA_VISIBLE_DEVICES='' pytest {pytest_args} {fname}".format(**vars())
    elif fname.endswith(".py") and bname != 'conftest.py':
        cmd = f"PYTHONPATH=.. {sys.executable} " + fname
    else:
        cmd = "PYTHONPATH=.. CUDA_VISIBLE_DEVICES='' pytest {pytest_args}".format(**vars())
    return cmd, _post[bname]

def main(**kw):
    t = perf_counter()

    post = ""
    if not kw['testfile']:
        cmd = "pytest"
    else:
        if kw['testfile'].endswith(__file__):
            cmd = ""
        else:
            cmd, post = dispatch(
                kw['testfile'],
                **kw,
            )

    # print("call:", sys.argv)
    print("cwd:", os.getcwd())
    print("cmd:", cmd)
    print(f"{' ide/runtests.py running cmd in cwd ':=^80}")
    sys.stdout.flush()
    # if 1cmd.startswith('pytest '):
    os.putenv("NUMBA_OPT", "1")
    # os.putenv('NUMBA_DISABLE_JIT', '1')

    # print(cmd)
    os.system(cmd)

    print(f"{' main command done ':=^80}")
    os.system(post)
    t = perf_counter() - t
    print(f"{f' runtests.py done, time {t:7.3f} ':=^80}")

if __name__ == '__main__':
    args = get_args(sys.argv)
    main(file_mappings=_file_mappings, overrides=_overrides, **args)
