"""
===============================
Context Managers Utility Module
===============================

This module provides a collection of useful and versatile **context managers**
for handling various runtime behaviors. These include:

- **Redirection of stdout/stderr:** Easily capture or redirect print output.
- **Dynamic class casting:** Temporarily change an object's class.
- **Automatic file handling:** Open multiple files and ensure proper cleanup.
- **Temporary working directory changes:** Change directory and automatically revert.
- **Capturing asserts and exceptions:** Capture exceptions for later inspection.
- **Random seed state preservation:** Temporarily set a random seed for reproducibility.
- **Debugging tools:** Trace print statements with stack traces and capture stdio.
- **Suppressing optional imports:** Cleanly handle optional imports without crashing.

### **💡 Why Use These Context Managers?**
Context managers allow you to **manage resources safely and concisely**,
ensuring proper cleanup regardless of errors. This module provides **custom utilities**
not found in Python's standard library, which can be extremely useful in testing,
debugging, and experimental setups.

---

## **📌 Usage Examples**

### **Redirect stdout and stderr**
>>> with redirect(stdout=open("/tmp/output.log", "w")):
...    print("This will be written to output.log")

### **Temporarily Change Working Directory**
>>> import os
>>> curdir = os.getcwd()
>>> with cd("/tmp"):
...     assert os.getcwd().startswith('/tmp')
>>> curdir == os.getcwd()
True

### **Capture Standard Output**
>>> with capture_stdio() as captured:
...     print("Captured output")
>>> print("Captured text:", captured.read(), end='')
Captured text: Captured output

### **Set a Temporary Random Seed**
>>> import numpy as np
>>> with temporary_random_seed(42):
...    print(np.random.rand(3))
[0.37454 0.95071 0.73199]

>>> # Outside the context, the previous random state is restored.

### **Capture Assertion Errors**
>>> with capture_asserts() as errors:
...     assert False, "This assertion error will be captured"
>>> print("Captured errors:", errors)
Captured errors: [AssertionError('This assertion error will be captured')]

### **Suppress Optional Imports**
>>> with optional_imports():
...     import some_optional_module  # Will not raise ImportError if module is absent.
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
def set_class(cls, self):
    try:
        orig, self.__class__ = self.__class__, cls
        yield self
    finally:
        self.__class__ = orig  # type: ignore

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
def catch_em_all():
    errors = []
    try:
        yield errors
    except Exception as e:
        errors.append(e)
    finally:
        pass

@contextlib.contextmanager
def redirect(stdout=sys.stdout, stderr=sys.stderr):
    """
    Temporarily redirect the stdout and stderr streams.

    Parameters:
        stdout (file-like or None): Target for stdout (default: sys.stdout).
        stderr (file-like, 'stdout', or None): Target for stderr (default: sys.stderr).

    Yields:
        tuple: (stdout, stderr) during redirection.
    """
    _out, _err = sys.stdout, sys.stderr
    try:
        sys.stdout.flush(), sys.stderr.flush()
        if stdout is None:
            stdout = io.StringIO()
        if stderr == 'stdout':
            stderr = stdout
        elif stderr is None:
            stderr = io.StringIO()
        sys.stdout, sys.stderr = stdout, stderr
        yield stdout, stderr
    finally:
        sys.stdout.flush(), sys.stderr.flush()
        sys.stdout, sys.stderr = _out, _err

@contextlib.contextmanager
def cd(path):
    """
    Temporarily change the working directory.

    Parameters:
        path (str): Target directory.

    Yields:
        None
    """
    oldpath = os.getcwd()
    try:
        os.chdir(path)
        yield None
    finally:
        os.chdir(oldpath)

@contextlib.contextmanager
def capture_stdio():
    """
    Capture standard output and error.

    Yields:
        io.StringIO: The captured stdout buffer.
    """
    with redirect(None, 'stdout') as (out, err):
        try:
            yield out
        finally:
            out.seek(0)
            err.seek(0)

@contextlib.contextmanager
def temporary_random_seed(seed=None):
    """
    Temporarily set a numpy random seed.

    Parameters:
        seed (int, optional): The seed to set.

    Yields:
        None
    """
    randstate = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        yield None
    finally:
        if seed is not None:
            np.random.set_state(randstate)

@contextlib.contextmanager
def capture_asserts():
    """
    Capture AssertionErrors.

    Yields:
        list: A list of captured AssertionErrors.
    """
    errors = []
    try:
        yield errors
    except AssertionError as e:
        errors.append(e)
    finally:
        pass

def optional_imports():
    """
    Suppress ImportError.

    Returns:
        contextlib.suppress(ImportError)
    """
    return contextlib.suppress(ImportError)

_original_print = print
_current_indent = 0

def indented_print(*args, **kwargs):
    """
    A replacement for the print function that adds indentation.
    """
    sep = kwargs.get('sep', ' ')
    indent_str = ' ' * _current_indent
    content = sep.join(str(arg) for arg in args)
    indented_content = indent_str + content.replace('\n', '\n' + indent_str)
    _original_print(indented_content, **kwargs)

@contextlib.contextmanager
def indent(spaces):
    """
    A context manager that indents all print statements within its scope.

    Args:
        spaces (int): The number of spaces to add to the current indentation level.
    """
    global _current_indent
    if sys.modules['builtins'].print is _original_print:
        sys.modules['builtins'].print = indented_print
    _current_indent += spaces
    try:
        yield
    finally:
        _current_indent -= spaces
        if _current_indent == 0:
            sys.modules['builtins'].print = _original_print
