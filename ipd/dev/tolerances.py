"""
Tolerance tracking and checking utilities for numerical comparisons.

This module provides a `Tolerances` class for managing multiple named tolerances,
each represented by a `Checker` object that tracks the number of comparisons made
and how many passed. Tolerances are typically used in testing scenarios where
approximate comparisons are needed.

Example usage:
    >>> from ipd.dev import Tolerances
    >>> import numpy as np
    >>> # Initialize with a default tolerance and some custom ones
    >>> tol = Tolerances(1e-4, position=1e-3, angle=1.0, energy=1e-5)
    >>> # Make various comparisons
    >>> tol.position < 0.002         # True
    True
    >>> tol.position > 1e-4          # True
    True
    >>> tol.angle >= 0.5             # True
    True
    >>> tol.angle <= 2.0             # True
    True
    >>> tol.energy > np.array([1e-6, 1e-4])    # False (one fails)
    array([ True, False])
    >>> np.array([1e-5, 1e-6]) < tol.energy    # One passes, one fails (reverse operator)
    array([False,  True])
    >>> # Check the history of comparisons
    >>> hist = tol.check_history()
    >>> for name, h in hist.items():
    ...     print(f"{name:10s}: tol={h.tol}, passed={h.passes}/{h.total} ({h.frac})")
    position  : tol=0.001, passed=2/2 (1.0)
    angle     : tol=1.0, passed=2/2 (1.0)
    energy    : tol=1e-05, passed=2/4 (0.5)
"""

import copy
import sys

import numpy as np

import ipd

class Tolerances:
    """
    Container for named numerical tolerances with comparison tracking.

    This class is useful for managing and checking multiple named tolerance
    thresholds, such as in unit tests where different components may have
    varying precision requirements. Each named tolerance is backed by a
    `Checker` that tracks the number of comparisons made and how many passed.

    Parameters
    ----------
    tol : float or Tolerances, optional
        A default tolerance value, or an existing `Tolerances` object to clone.
    default : float, optional
        Default threshold to use for unnamed tolerances. Ignored if `tol` is a `Tolerances` instance.
    **kw : float
        Named tolerance values to initialize, such as `length=1e-3`, `angle=0.1`.

    Attributes
    ----------
    kw : dict
        Raw keyword arguments specifying named tolerances.
    checkers : dict[str, Checker]
        Dictionary mapping names to `Checker` objects.
    _default_tol : float
        Default tolerance used for any undeclared checker.

    Examples
    --------
    >>> tol = Tolerances(1e-4, length=1e-3, angle=0.1)
    >>> tol.length < 0.002
    True
    >>> tol.length > 0.0005
    True
    >>> tol.angle == 0.1
    True
    >>> 0.05 < tol.angle  # reverse operator
    True
    >>> 0.001 < tol.unset_tolerance  # will use default
    False
    >>> 0.00001 < tol.unset_tolerance  # will use default
    True
    >>> ipd.print_table(tol.check_history(), precision=4)
    ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━┓
    ┃ key             ┃ tol     ┃ frac    ┃ total ┃ passes ┃
    ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━┩
    │ length          │  0.0010 │  1.0000 │    2  │    2   │
    │ angle           │  0.1000 │  1.0000 │    2  │    2   │
    │ unset_tolerance │  0.0001 │  0.5000 │    2  │    1   │
    └─────────────────┴─────────┴─────────┴───────┴────────┘
    """

    ...

    def __init__(self, tol=None, default=None, **kw):
        if isinstance(tol, Tolerances):
            self.kw = tol.kw | kw
            self.checkers = tol.checkers
            self._default_tol = tol._default_tol
        else:
            if tol is not None:
                assert default is None, 'usage is Tolerances(tol, default, **kw) or Tolerances(default, **kw)'
                default = tol
            self.kw = kw
            self.checkers = {}
            self._default_tol = default or 1e-4
        self.checkers = {name: Checker(tol, 0, 0) for name, tol in kw.items()} | self.checkers

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        if not name.startswith('_') and 'checkers' in self.__dict__:
            if name not in self.__dict__['checkers']:
                threshold = float(self.kw.get(name, self._default_tol))
                self.checkers[name] = Checker(threshold)
            return self.checkers[name]
        raise AttributeError(f'Tolerances object has no attribute {name}')

    def reset(self):
        """
        Reset the pass/fail history of all checkers.

        Returns:
            self
        """
        for c in self.checkers.values():
            c.n_checks = 0
            c.n_passes = 0
        return self

    def check_history(self):
        """
        Return a summary of all comparisons made.

        Returns:
            ipd.Bunch: Mapping of tolerance names to stats including:
                - tol: Threshold value
                - frac: Fraction of comparisons that passed
                - total: Number of checks
                - passes: Number of passing checks
        """
        history = ipd.Bunch()
        for k, c in self.checkers.items():
            frac = round(c.n_passes / c.n_checks, 3) if c.n_checks > 0 else None
            history[k] = ipd.Bunch(tol=c.threshold, frac=frac, total=c.n_checks, passes=c.n_passes)
        return history

    def copy(self):
        """
        Return a deep copy of the Tolerances object.

        Returns:
            Tolerances: A copy with independent Checker objects.
        """
        return copy.deepcopy(self)

    def __repr__(self):
        with ipd.dev.capture_stdio() as out:
            if hist := self.check_history():
                ipd.dev.print_table(hist, key='Tolerances')
        return out.read()

@ipd.struct
class Checker:
    """
    Encapsulates a numerical threshold and tracks pass/fail statistics from comparisons.

    Supports scalar and array-based comparison using the standard operators:
    `<`, `<=`, `>`, `>=`, `==`. These comparisons increment internal counters
    to track the number of checks and how many passed.

    Fields
    ------
    threshold : float
        The numeric threshold to compare against.
    n_checks : int
        Number of comparisons made.
    n_passes : int
        Number of comparisons that passed.

    Notes
    -----
    Reverse comparison operators (e.g., `5 > tol.foo`) are supported via
    special methods (`__rgt__`, etc.). If used with `xarray.Dataset`, a
    `TypeError` is raised.

    Examples
    --------
    >>> c = Checker(0.5)
    >>> c < 0.7
    True
    >>> c.n_checks, c.n_passes
    (1, 1)

    >>> c > 1.0
    False
    >>> c.n_checks, c.n_passes
    (2, 1)
    """

    ...

    threshold: float
    n_checks: int = 0
    n_passes: int = 0

    def _record(self, result):
        """
        Update internal counters based on result and return the input.

        Parameters:
            result (bool | np.ndarray | torch.Tensor): The outcome of the comparison.

        Returns:
            The input result.
        """
        if isinstance(result, bool):
            self.n_checks += 1
            self.n_passes += result
        else:
            xr = sys.modules.get('xarray')
            if xr and isinstance(result, xr.Dataset):
                raise TypeError('Tolerances doesnt support whole xarray.Dataset comparisons')
            assert str(result.dtype)[-4:] == 'bool'
            self.n_checks += len(result)
            self.n_passes += np.sum(np.array(result))
        return result

    def __float__(self):
        return self.threshold

    def __gt__(self, val):
        return self._record(self.threshold > val)

    def __ge__(self, val):
        return self._record(self.threshold >= val)

    def __lt__(self, val):
        return self._record(self.threshold < val)

    def __le__(self, val):
        return self._record(self.threshold <= val)

    def __rgt__(self, val):
        return self._record(self.threshold < val)

    def __rge__(self, val):
        return self._record(self.threshold <= val)

    def __rlt__(self, val):
        return self._record(self.threshold > val)

    def __rle__(self, val):
        return self._record(self.threshold >= val)

    def __eq__(self, val):
        return self._record(self.threshold == val)

    # Aliases for explicit method-based comparisons
    gt, ge, lt, le = __gt__, __ge__, __lt__, __le__
