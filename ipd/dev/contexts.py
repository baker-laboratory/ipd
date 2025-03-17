"""
===============================
Context Managers Utility Module
===============================

This module provides a collection of useful and versatile **context managers**
for handling various runtime behaviors. These include:

- **Redirection of stdout/stderr**
- **Dynamic class casting**
- **Automatic file handling**
- **Temporary working directory changes**
- **Capturing asserts and exceptions**
- **Random seed state preservation**
- **Debugging tools (tracing prints, capturing stdio)**
- **Suppressing optional imports**

### **💡 Why Use These Context Managers?**
Context managers allow you to **manage resources safely and concisely**,
ensuring proper cleanup regardless of errors. This module provides **custom utilities**
not found in Python's standard library.

---

## **📌 Usage Examples**
### **Redirect stdout and stderr**
```python
with redirect(stdout=open("output.log", "w")):
    print("This will be written to output.log")
"""

import atexit
import io
import numpy as np
import os
import sys
import traceback
import contextlib

import ipd

def onexit(func, msg=None, **metakw):

    def wrapper(*args, **kw):
        if msg is not None: print(msg)
        return func(*args, **(metakw | kw))

    atexit.register(wrapper)
    return wrapper

@contextlib.contextmanager
def cast(cls, self):
    try:
        orig, self.__class__ = self.__class__, cls
        yield self
    finally:
        self.__class__ = orig  # type: ignore

@contextlib.contextmanager
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

@contextlib.contextmanager
def capture_stdio():
    with redirect(None, 'stdout') as (out, err):
        try:
            yield out
        finally:
            out.seek(0)
            err.seek(0)

@contextlib.contextmanager
def stdio():
    """useful as temporary escape hatch with io capuring contexts"""
    with redirect(sys.__stdout__, sys.__stderr__) as (out, err):
        try:
            yield out, err
        finally:
            pass

@contextlib.contextmanager
def nocontext():
    try:
        yield None
    finally:
        pass

@contextlib.contextmanager
def cd(path):
    oldpath = os.getcwd()
    try:
        os.chdir(path)
        yield None
    finally:
        os.chdir(oldpath)

@contextlib.contextmanager
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

@contextlib.contextmanager
def trace_prints():
    tp = TracePrints()
    with redirect(stdout=tp):
        yield tp

@contextlib.contextmanager
def np_printopts(**kw):
    npopt = np.get_printoptions()
    try:
        np.set_printoptions(**kw)
        yield None
    finally:
        np.set_printoptions(**{k: npopt[k] for k in kw})

def np_compact(precision=4, suppress=True, **kw):
    return np_printopts(precision=precision, suppress=suppress, **kw)

@contextlib.contextmanager
def temporary_random_seed(seed=None):
    randstate = np.random.get_state()
    if seed is not None: np.random.seed(seed)
    try:
        yield None
    finally:
        if seed is not None: np.random.set_state(randstate)

@contextlib.contextmanager
def capture_asserts():
    errors = []
    try:
        yield errors
    except AssertionError as e:
        errors.append(e)
    finally:
        pass

@contextlib.contextmanager
def catchall():
    errors = []
    try:
        yield errors
    except Exception as e:
        errors.append(e)
    finally:
        pass

def optional_imports():
    return contextlib.suppress(ImportError)
