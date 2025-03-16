import atexit
import io
import numpy as np
import os
import sys
import traceback
from contextlib import contextmanager

import ipd

def onexit(func, msg=None, **metakw):

    def wrapper(*args, **kw):
        if msg is not None: print(msg)
        return func(*args, **(metakw | kw))

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
        if stdout is None: stdout = io.StringIO()
        if stderr == 'stdout': stderr = stdout
        elif stderr is None: stderr = io.StringIO()
        sys.stdout, sys.stderr = stdout, stderr
        yield stdout, stderr
    finally:
        sys.stdout.flush(), sys.stderr.flush()  # type: ignore
        sys.stdout, sys.stderr = _out, _err

@contextmanager
def capture_stdio():
    with redirect(None, 'stdout') as (out, err):
        try:
            yield out
        finally:
            out.seek(0)
            err.seek(0)

@contextmanager
def stdio():
    with redirect(sys.__stdout__, sys.__stderr__) as (out, err):
        try:
            yield out, err
        finally:
            pass

@contextmanager
def nocontext():
    try:
        yield None
    finally:
        pass

@contextmanager
def cd(path):
    oldpath = os.getcwd()
    try:
        os.chdir(path)
        yield None
    finally:
        os.chdir(oldpath)

@contextmanager
def openfiles(*fnames, **kw):
    files = [ipd.dev.openfile(f, **kw) for f in fnames]
    if len(files) == 1: files = files[0]
    try:
        yield files
    finally:
        ipd.dev.closefiles(files)

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

@contextmanager
def np_printopts(**kw):
    npopt = np.get_printoptions()
    try:
        np.set_printoptions(**kw)
        yield None
    finally:
        np.set_printoptions(**{k: npopt[k] for k in kw})

def np_compact(precision=4, suppress=True, **kw):
    return np_printopts(precision=precision, suppress=suppress, **kw)

@contextmanager
def temporary_random_seed(seed=None):
    randstate = np.random.get_state()
    if seed is not None: np.random.seed(seed)
    try:
        yield None
    finally:
        if seed is not None: np.random.set_state(randstate)

@contextmanager
def capture_asserts():
    errors = []
    try:
        yield errors
    except AssertionError as e:
        errors.append(e)
    finally:
        pass

@contextmanager
def catchall():
    errors = []
    try:
        yield errors
    except Exception as e:
        errors.append(e)
    finally:
        pass
