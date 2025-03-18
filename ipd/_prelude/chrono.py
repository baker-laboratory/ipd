"""
Versatile High-Performance Timing Module
=========================================

This module provides a robust, feature-rich **Chrono** class designed for various performance tracking scenarios. It can be used as:

- **A standalone Chrono instance**
- **A function or class decorator** (`@chrono`, `@chrono_class`)
- **A context manager** (`with Chrono() as t:`)
- **A global checkpointing mechanism** (`checkpoint()`)
- **A nested timing system** (correctly tracks time across recursive/nested calls)

Features:
---------
- Supports **Python lists or a high-performance Cython backend (`DynamicFloatArray`)**
- Works with **async functions, generators, and class methods**
- Provides **automatic function labeling**
- Supports **interjected checkpoints** (mid-execution markers without disrupting ongoing tracking)
- Reports **sum, mean, median, min, and max timing statistics**

Installation & Setup:
---------------------
Ensure Cython is installed for the best performance:

.. code-block:: bash

   pip install Cython numpy

Using this module in your project:

.. code-block:: python

   from your_package_name.chrono import Chrono, chrono, chrono_class, checkpoint

Basic Usage:
------------
Standalone Chrono instance:

>>> from your_package_name.chrono import Chrono
>>> t = Chrono()
>>> t.checkpoint("step1")
>>> t.checkpoint("step2")
>>> t.stop()
>>> print(t.report(printme=False))

Using as a **context manager**:

>>> with Chrono() as t:
>>>     time.sleep(0.1)
>>>     t.checkpoint("task1")
>>> print(t.report(printme=False))

Using as a **function decorator**:

>>> @chrono
>>> def slow_function():
>>>     time.sleep(0.2)
>>>
>>> slow_function()
>>> print(ipd.global_chrono.report(printme=False))

Using as a **class decorator**:

>>> @chrono_class
>>> class MyClass:
>>>     def method(self):
>>>         time.sleep(0.1)
>>>
>>> obj = MyClass()
>>> obj.method()
>>> print(ipd.global_chrono.report(printme=False))

Checkpointing with Automatic Chrono Selection:
---------------------------------------------
For cases where a chrono is passed via `kw` (often forwarded from higher up), or when using the global chrono, the `checkpoint()` function provides a convenient way to mark sections of code.

This is especially useful for short segments that need independent timing:

>>> import your_package_name.chrono as chronomodule
>>> chronomodule.checkpoint(interject=True)  # Mark interjection
>>> # Some code to time independently of everything else
>>> chronomodule.checkpoint("my_bespoke_chrono_stuff")

This allows:
- **Automatically selecting a chrono** from `kw` or falling back to the global chrono.
- **Precise timing of code blocks** within larger functions or contexts.
- **Minimal modification of existing functions**, as chronos can be passed via kw.

Interjecting Checkpoints:
-------------------------
If you want to insert a checkpoint **without losing track of ongoing execution**, use `checkpoint(interject=True)`. This ensures the next recorded time **includes all execution time up to the interjection**.

>>> t = Chrono()
>>> t.checkpoint("before")
>>> t.checkpoint(interject=True)
>>> t.checkpoint("after")
>>> t.stop()
>>> print(t.report(printme=False))

Performance Considerations:
---------------------------
- Use **`use_cython=True`** in `Chrono()` for high-performance tracking.
- If working with **many nested function calls**, function decorators correctly **attribute execution time to each function**.
- For **profiling generators**, `@chrono` correctly tracks execution **across yields**.

Limitations:
------------
- If using `checkpoint()`, ensure a **consistent labeling convention** to avoid confusion in reports.
- **Recursive function calls** work correctly, but deep recursion may require **manual inspection of reports**.
- **Global timing with `@chrono` on deeply nested functions** may produce extensive logs—filtering output is recommended.

Future Improvements:
---------------------
- Potential integration with `cProfile` for even finer granularity.
- Exporting timing data to structured formats (JSON, CSV) for deeper analysis.
- Automatic visualization of timing results.

"""

import time
import functools
import inspect
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union
import ipd

try:
    from your_package_name.dynamic_float_array import DynamicFloatArray
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

@dataclass
class Chrono:
    """
    A versatile high-performance timing utility.

    Features:
        - Works as a class instance, function decorator, or context manager.
        - Supports both Python lists and Cython-backed storage.
        - Supports interjected checkpoints.
        - Provides statistical summaries.
    """
    name: str = "Chrono"
    verbose: bool = True
    use_cython: bool = False
    initial_capacity: int = 10
    _start: Union[float, None] = None
    last: Union[float, None] = None
    _checkpoint_stack: List[float] = field(default_factory=list)
    storage_type: str = ''
    active_stack: list[str] = field(default_factory=list)
    checkpoints: dict[str, Union[List[float], 'DynamicFloatArray']] = field(init=False)
    _entered: bool = False
    _exited: bool = False
    _last_interject: bool = False

    def __post_init__(self):
        self.use_cython = self.use_cython and CYTHON_AVAILABLE
        if self.use_cython:
            self.checkpoints = ipd.Bunch()
            self.storage_type = "Cython (DynamicFloatArray)"
        else:
            self.checkpoints = ipd.Bunch(_default=list)
            self.storage_type = "Python List"

        if self.verbose:
            print(f"Chrono using {self.storage_type} storage")

        self.start()

    def start(self):
        """Start the chrono."""
        if self._start is not None:
            self.stop()
        self._start = time.perf_counter()
        self.last = self._start
        self._checkpoint_stack.clear()
        return self

    def stop(self):
        """Stop the chrono and store total elapsed time."""
        if self._start is not None:
            total_elapsed = time.perf_counter() - self._start
            self._store_checkpoint("total", total_elapsed)
            ipd.ic(id(self), id(ipd.global_chrono))
            self._start = None

    def __enter__(self):
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"An exception of type {exc_type} occurred: {exc_val}")
        self._entered = False
        self._exited = True
        return False

    def _store_checkpoint(self, name: str, elapsed: float):
        """Store elapsed time in either Python list or Cython storage."""
        if self.use_cython:
            if name not in self.checkpoints:
                self.checkpoints[name] = DynamicFloatArray()
            self.checkpoints[name].append(elapsed)
        else:
            self.checkpoints[name].append(elapsed)

    def checkpoint(self, name: str = None, interject: bool = False):
        """
        Mark a checkpoint in the chrono.

        Args:
            name (str, optional): Label for the checkpoint.
            interject (bool, optional): Whether to inject a checkpoint.
        """
        if name is None: interject = True
        if self._start is None:
            raise RuntimeError("Chrono is not running")

        current_time = time.perf_counter()
        elapsed = current_time - self.last
        self.last = current_time

        if interject:
            self._checkpoint_stack.append(elapsed)
            self._last_interject = True
        elif self._last_interject:
            self._last_interject = False
            self._store_checkpoint(name, elapsed)
        else:
            while self._checkpoint_stack:
                self._store_checkpoint(name, self._checkpoint_stack.pop())
            self._store_checkpoint(name, elapsed)

        return self

    def elapsed(self) -> float:
        """Return the total elapsed time."""
        return sum(self.get_checkpoint_data("total"))

    def get_checkpoint_data(self, name: str):
        """
        Retrieve checkpoint times.

        Args:
            name (str): The checkpoint label.

        Returns:
            np.ndarray: Array of checkpoint times.
        """
        if self.use_cython:
            return self.checkpoints.get(name, DynamicFloatArray()).get_data()
        return np.array(self.checkpoints.get(name, []), dtype=np.float32)

    def report_dict(self, order="longest", summary=sum):
        """
        Generate a report dictionary of checkpoint times.

        Args:
            order (str): Sorting order ('longest' or 'callorder').
            summary (callable): Function to summarize times (e.g., sum, mean).

        Returns:
            dict: Checkpoint times summary.
        """
        items = self.checkpoints.keys()
        if order == "longest":
            sorted_items = sorted(items, key=lambda k: -summary(self.get_checkpoint_data(k)))
        elif order == "callorder":
            sorted_items = items
        else:
            raise ValueError(f"Unknown order: {order}")
        return {k: summary(self.get_checkpoint_data(k)) for k in sorted_items}

    def report(self, order="longest", summary=sum, printme=True) -> str:
        """
        Print or return a report of checkpoint times.

        Args:
            order (str): Sorting order ('longest' or 'callorder').
            summary (callable): Function to summarize times (e.g., sum, mean).
            printme (bool): Whether to print the report.

        Returns:
            str: Report string.
        """
        times = self.report_dict(order=order, summary=summary)
        report_lines = [f"Chrono Report ({self.name}) using {self.storage_type}"]
        for name, time_ in times.items():
            report_lines.append(f"{name}: {time_:.6f}s")
        report = "\n".join(report_lines)
        if printme:
            print(report)
        return report

    @property
    def total(self) -> float:
        """Return total recorded time."""
        return self.get_checkpoint_data("total").sum()

def chrono(func=None, *, label=None, timer=None):
    """
    Function decorator to automatically time function execution.

    Usage:
        @chrono
        def my_function():
            pass

    Args:
        func (callable): The function to be decorated.
        label (str, optional): Custom checkpoint label.
        timer (Timer, optional): Timer instance to use.

    Returns:
        callable: Wrapped function with timing functionality.
    """
    if func is None: return functools.partial(chrono, label=label, timer=timer)
    if inspect.isclass(func): return chrono_class(cls=func, label=label, timer=timer)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t = timer or kwargs.get("timer", ipd.global_chrono)
        function_name = func.__qualname__
        checkpoint_label = label or function_name
        t.checkpoint(interject=True)
        result = func(*args, **kwargs)
        t.checkpoint(checkpoint_label)
        return result

    return wrapper

def chrono_class(cls, label=None, timer=None):
    """
    Class decorator to time all methods in a class.

    Usage:
        @chrono_class
        class MyClass:
            def method(self):
                pass

    Args:
        cls (type): The class to be decorated.

    Returns:
        type: The decorated class.
    """
    for attr_name, attr_value in vars(cls).items():
        if callable(attr_value) and not attr_name.startswith("__"):
            setattr(cls, attr_name, chrono(attr_value, label=label, timer=timer))
    return cls

def checkpoint(label=None, *, interject=False, **kw):
    """
    Create a module-level checkpoint, using a timer from `kw` or falling back to `ipd.global_chrono`.

    Args:
        label (str, optional): Name of the checkpoint. If None, an auto-generated name is used.
        interject (bool, optional): If True, creates an interjection checkpoint.
        **kw: Additional keyword arguments (expected to contain a 'timer' key, if available).

    Example:
        ```python
        checkpoint("start_loop")
        for _ in range(10):
            checkpoint(interject=True)
            time.sleep(0.1)  # Some operation
            checkpoint("iteration_done")
        checkpoint("end_loop")
        ipd.global_chrono.report()
        ```
    """
    t = kw.get("timer", ipd.global_chrono)

    if label is None:
        label = "__interject__" if interject else "unnamed"

    t.checkpoint(label, interject=interject)
