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
```python
with redirect(stdout=open('output.log', 'w')):
    print('This will be written to output.log')
```

### **Temporarily Change Working Directory**
```python
import os

print('Current directory:', os.getcwd())
with cd('/tmp'):
    print('Inside /tmp:', os.getcwd())
print('Reverted directory:', os.getcwd())
```

### **Capture Standard Output**
```python
with capture_stdio() as captured:
    print('Captured output')
# Use captured.getvalue() to retrieve the captured text.
print('Captured text:', captured.getvalue())
```

### **Capture Assertion Errors**
```python
with capture_asserts() as errors:
    assert False, 'This assertion error will be captured'
print('Captured errors:', errors)
```

### **Suppress Optional Imports**
```python
with optional_imports():
    import some_optional_module  # Will not raise ImportError if module is absent.
```
"""

import atexit
import io
import os
import sys
import traceback
import contextlib

__the_real_stdout__ = sys.__stdout__
__the_real_stderr__ = sys.__stderr__
import evn


def onexit(func, msg=None, **metakw):

    def wrapper(*args, **kw):
        if msg is not None:
            print(msg)
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
def force_stdio():
    """useful as temporary escape hatch with io capuring contexts"""
    with redirect(__the_real_stdout__, __the_real_stderr__) as (out, err):
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


class TraceWrites(object):

    def __init__(self, preset):
        self.stdout = sys.stdout
        self.preset = preset
        self.log = []

    def write(self, s):
        stack = os.linesep.join(traceback.format_stack())
        stack = evn.filter_python_output(stack,
                                         preset=self.preset,
                                         arrows=False)
        self.log.append(f'\nA WRITE TO STDOUT!: "{s}"{os.linesep}')
        self.log.append(stack)

    def flush(self):
        self.stdout.flush()

    def printlog(self):
        self.stdout.write(os.linesep.join(self.log))


@contextlib.contextmanager
def trace_writes_to_stdout(preset='aggressive'):
    tp = TraceWrites(preset)
    with redirect(stdout=tp, after=lambda: tp.printlog()):
        yield tp


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
def redirect(
    stdout: evn.IO = sys.stdout,
    stderr: evn.IO = sys.stderr,
    after: evn.Callable = evn.NoOp,
):
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
        if after:
            after()


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
def just_stdout():
    try:
        yield sys.stdout
    finally:
        pass


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


@contextlib.contextmanager
def modloaded(pkg):
    try:
        if pkg in sys.modules:
            yield sys.modules[pkg]
        else:
            yield None
    finally:
        pass


@contextlib.contextmanager
def cd_project_root():
    """
    Change to the project root directory.

    Yields:
        bool: True if the project root exists, False otherwise.
    """
    if root := evn.projroot:
        with cd(root):
            yield True
    else:
        yield False


@contextlib.contextmanager
def np_printopts(**kw):
    np = evn.maybeimport('numpy')
    if not np:
        yield None
        return
    npopt = np.get_printoptions()
    try:
        np.set_printoptions(**kw)
        yield None
    finally:
        np.set_printoptions(**{k: npopt[k] for k in kw})


def np_compact(precision=4, suppress=True, **kw):
    return np_printopts(precision=precision, suppress=suppress, **kw)
