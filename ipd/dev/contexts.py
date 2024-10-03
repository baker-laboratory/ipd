from contextlib import contextmanager
import os
import sys

@contextmanager
def redirect(stdout=sys.stdout, stderr=sys.stderr):
    _out, _err = sys.stdout, sys.stderr
    try:
        sys.stdout.flush(), sys.stderr.flush()
        sys.stdout, sys.stderr = stdout, stderr
        yield None
    finally:
        sys.stdout.flush(), sys.stderr.flush()
        sys.stdout, sys.stderr = _out, _err

@contextmanager
def cd(path):
    oldpath = os.getcwd()
    try:
        os.chdir(path)
        yield None
    finally:
        os.chdir(oldpath)
