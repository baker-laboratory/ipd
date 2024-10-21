from contextlib import contextmanager
import os
import sys
import io

@contextmanager
def cast(cls, self):
    try:
        orig, self.__class__ = self.__class__, cls
        yield self
    finally:
        self.__class__ = orig

@contextmanager
def redirect(stdout=sys.stdout, stderr=sys.stderr):
    _out, _err = sys.stdout, sys.stderr
    try:
        sys.stdout.flush(), sys.stderr.flush()
        if stderr == 'stdout': stderr = stdout
        if stdout is None: stdout = io.StringIO()
        if stderr is None: stderr = io.StringIO()
        sys.stdout, sys.stderr = stdout, stderr
        yield stdout, stderr
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
