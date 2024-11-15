import atexit
import io
import os
import sys
import traceback
from contextlib import contextmanager

def onexit(func):
    def wrapper(*args, **kw):
        return func(*args, **kw)

    atexit.register(wrapper)
    return wrapper

@contextmanager
def cast(cls, self):
    try:
        orig, self.__class__ = self.__class__, cls
        yield self
    finally:
        self.__class__ = orig  # type: ignore

@contextmanager
def redirect(stdout=sys.stdout, stderr=sys.stderr):
    _out, _err = sys.stdout, sys.stderr
    try:
        sys.stdout.flush(), sys.stderr.flush()  # type: ignore
        if stderr == 'stdout': stderr = stdout
        if stdout is None: stdout = io.StringIO()
        if stderr is None: stderr = io.StringIO()
        sys.stdout, sys.stderr = stdout, stderr
        yield stdout, stderr
    finally:
        sys.stdout.flush(), sys.stderr.flush()  # type: ignore
        sys.stdout, sys.stderr = _out, _err

@contextmanager
def cd(path):
    oldpath = os.getcwd()
    try:
        os.chdir(path)
        yield None
    finally:
        os.chdir(oldpath)

class TracePrints(object):
    def __init__(self):
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write("Writing %r\n" % s)
        traceback.print_stack(file=self.stdout)

    def flush(self):
        self.stdout.flush()

@contextmanager
def trace_prints():
    tp = TracePrints()
    with redirect(stdout=tp):
        yield tp
