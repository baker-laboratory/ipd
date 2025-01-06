"""
======================================================
This file defines a DynamicParameter management system
======================================================

The DynamicParameters class allows parameter values to change through a run based on the design number, diffusion T number, and iterative simulator block number. Boolean parameters can be set by specifying particular iterations, or a range of values. Numerical parameters can be set by specifying example values for particular iterations, then 1D or 2D spline smoothing is used to get the value for intermediate iterations. Parameters are accessed by attribute, e.g. by accessing symopts.fit, you will get a true/false value depending on where you are in the run. Behavior is specified in config or command line, with no extra logic needed in the code that accesses the parameter. For example, ``sym.fit='rfold:[7,13,27,35]'`` will set fit to true on blocks 7, 13, 27 and 35, ``sym.fit='diffuse:[[12,27]]'`` will set sym.fit to true on diffusion steps 12 thru 27 inclusive, ``sym.fit='rfold:[[0,0.5]]*diffuse:[[0,10]]'`` will set fit to true for the first half of the rosettafold blocks when the diffusion step is 10 or less. DynamicParameters implements a Mapping, so it can be used for kwargs like so, foo(a,b,**symopts).

Examples
--------

A central DynamicParameter manager class creates and contains all parameters and is
created for a run with 10 designs, 50 diffusion steps, and 40 total rfold steps like so

    >>> params = DynamicParameters(ndesign=10, ndiffuse=50, nrfold=40)

New parameters are created with factory member functions of DynamicParameter
with a specified name and will depend on any combination of the current design,
diffusion, and/or rfold step. For example:

    >>> params.newparam_false_on_steps(name='done_on_first_3_steps', diffuse=[0, 1, 2])

and then accessed like this from anywhere using the DynamicParameter object

    >>> StepObserver().set_nstep(diffuse=50) # system will call this automatically
    >>> StepObserver().set_step(diffuse=50) # system will call this automatically
    >>> params.done_on_first_3_steps
    False
    >>> StepObserver().set_step(diffuse=48) # system will call this automatically
    >>> params.done_on_first_3_steps
    False
    >>> StepObserver().set_step(diffuse=47) # system will call this automatically
    >>> params.done_on_first_3_steps
    True

The factory functions newparam_false_on_steps and newparam_false_on_steps create new
boolean parameters that are true/false for specified steps in the protocol

    >>> params.newparam_true_on_steps('name1', diffuse=[1, 3])
    >>> params.newparam_true_on_steps('name2', diffuse=[-1, -3])
    >>> params.newparam_true_on_steps('name3', diffuse=[-1, -3])
    >>> params.newparam_false_on_steps('name4', design=1, rfold=[1, 3])
    >>> params.newparam_true_on_steps('name5', design=1, rfold=_NotIn(1, 3))

floats can be used as a fraction

    >>> params.newparam_true_on_steps('name6', diffuse=[0.33])
    >>> params.newparam_true_on_steps('name7', diffuse=[0.01, -0.33])

The factory functions newparam_true_in_range, newparam_false_in_range create new
boolean parameters based on ranges of steps. The range is inclusive. Floats can be
used to specify, for example, true when the fraction of steps done is between 0.3 and 0.8

    >>> params.newparam_true_in_range('name8', diffuse=(0.3, 0.8))

or explicit step numbers can be used for ex: the first 5 steps

    >>> params.newparam_true_in_range('name9', rfold=[(0, 4)])
    >>> StepObserver().set_step(rfold=2)
    >>> params.name9
    True
    >>> StepObserver().set_step(rfold=5)
    >>> params.name9
    False

multiple ranges can be specified

    >>> params.newparam_false_in_range('name10', rfold=[(0, 2), (4, 5)])

python range objects can also be used, eg the first 10 odd steps

    >>> params.newparam_false_in_range('name11', rfold=range(1, 20, 2))

Floating point parameters can be created with newparam_spline_1d and newparam_spline_2d
with two points, basically linear interp. (Do we need integer valued parameters? What
would be a good way to specify those? I'm kinda planning to round spline values for int
params that need to vary, but I dunno if that will be a good approach)

this example has values n+1, where n is the step number and N is total steps

    >>> N = 7
    >>> params.newparam_spline_1d('name12', diffuse=[(0, 1), (1, N + 1)])

with 3 points, a quadratic fit
this example has values (n+1)**2, where n is the step number and N is total steps

    >>> params.newparam_spline_1d('name13', rfold=[(0, 0), (0.5, 4**2), (1, 8**2)])

with 4 or more points, cubic spline fit
this example has values (n+1)**3, where n is the step number and N is total steps

    >>> params.newparam_spline_1d('name14', design=[(0, 0), (0.25, 2**3), (0.5, 4**3), (1, 8**3)])

can also do fancy 2D triangulation-based interploation to vary rfold params
over the course of an individual diffusion trajectory
values are specified as triples of (diffuse_step, rfold_step, value)
there must be at least 3, and anything outside their convex hull will
have>>>  an undefined value, so the points must cover the unit square [0-1,0-1]
(you'll get an informative error). I think it's probably best to always give values
for the four corners exactly

    >>> params.newparam_spline_2d('twod', diffuse_rfold=[
    ...     (0.0, 0.0, 0),
    ...     (0.0, 1.0, 50),
    ...     (1.0, 0.0, 50),
    ...     (1.0, 1.0, 100),
    ... ])

"""

import contextlib
import copy
import sys
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    import pytest

if TYPE_CHECKING:
    import pytest

import ipd
from ipd.observer.observer import Observer

Step = namedtuple('Step', 'design diffuse rfold')

class StepObserver(Observer):
    """Can be used to track the progress of a run, design step, diffusion step
    and rfold block step."""
    _subject = None

    # initialize once
    def __new__(cls):
        if not cls._subject:
            cls._subject = super().__new__(cls)
            cls._subject.observers = list()
            cls._subject._nstep = None
        # if not cls._subject._nstep and conf:
        return cls._subject

    def set_config(self, conf):  # type: ignore
        self.__class__._subject.set_nstep(  # type: ignore
            design=conf.inference.num_designs,
            diffuse=conf.diffuser.T,
            rfold=40,
        )

    def _add_observer(self, observer):
        """Add an observer to the list of observers."""
        self.observers.append(observer)  # type: ignore
        observer._subject = self
        if self._nstep:
            observer._set_nstep(self._nstep.design, self._nstep.diffuse, self._nstep.rfold)

    def set_nstep(self, design=None, diffuse=None, rfold=None):
        # ic('set_nstep', design, diffuse, rfold)
        self._nstep = Step(design, diffuse, rfold)
        for observer in self.observers:  # type: ignore
            observer._set_nstep(design, diffuse, rfold)

    def set_step(self, design=None, diffuse=None, rfold=None):
        # ic('set_step', design, diffuse, rfold)
        if diffuse is not None:
            diffuse = self._nstep.diffuse - diffuse
        for observer in self.observers:  # type: ignore
            observer._set_step(design, diffuse, rfold)

    def rfold_begin(self):
        for observer in self.observers:  # type: ignore
            observer._rfold_begin()

    def rfold_end(self):
        for observer in self.observers:  # type: ignore
            observer._rfold_end()

    def rfold_iter_begin(self, tag, **kw):
        # ic('rfold_iter_begin', tag)
        for observer in self.observers:  # type: ignore
            observer._rfold_iter_begin(tag, **kw)

class DynamicParameters(Mapping):
    """A central class that manages all dynamic parameters for a run."""
    def __init__(self, ndesign=None, ndiffuse=None, nrfold=40, _testing=False):
        self._step = Step(None, None, None)
        self._nstep = Step(ndesign, ndiffuse, nrfold)
        self._rfold_tags = set()
        self._params = {}
        self._strict = (ndesign is not None and ndiffuse is not None and nrfold is not None)
        self._parsed_params = {}
        if not _testing: StepObserver()._add_observer(self)
        self._sanity_check()

    ################## factory funcs for the various dynparam types #####################
    def copy(self):
        return copy.copy(self)

    def parse_dynamic_param(self, name, value, overwrite=False):
        if not isinstance(value, str) or not (value.count(':') or value.count('spline')):
            self.newparam_constant(name, value, overwrite=overwrite)
        elif value.startswith('spline'):
            raise NotImplementedError
        elif value.startswith(('rfold:', 'diffuse:', 'design:')):
            self._parsed_params[name] = value
            try:
                args = {}
                for val in value.split('*'):
                    k, v, *a = val.split(':')
                    if a:
                        assert len(a) == 1
                        a = ipd.dev.safe_eval(a[0])
                    levels = a if a else (True, False)
                    args[k] = ipd.dev.safe_eval(v)
                if any(isinstance(v[0], (tuple, list)) for v in args.values()):
                    self.newparam_true_in_range(name, overwrite=overwrite, levels=levels, **args)  # type: ignore
                else:
                    self.newparam_true_on_steps(name, overwrite=overwrite, levels=levels, **args)  # type: ignore
            except ValueError as e:
                raise ValueError(f'bad dynam param string "{value}"') from e
        else:
            raise ValueError(f'bad dynam param string "{value}"')

    def newparam_true_on_steps(self, name, design=None, diffuse=None, rfold=None, levels=(True, False), **kw):
        self._add_param(name, _TrueOnIters(self, design, diffuse, rfold, levels), **kw)

    def newparam_false_on_steps(self, name, design=None, diffuse=None, rfold=None, levels=(True, False), **kw):
        self._add_param(name, _Not(_TrueOnIters(self, design, diffuse, rfold, levels)), **kw)

    def newparam_true_in_range(self, name, design=None, diffuse=None, rfold=None, levels=(True, False), **kw):
        if design is not None and isinstance(design[0], (tuple, list)):
            design = _ranges_to_sets(design, self._nstep.design)
        if diffuse is not None and isinstance(diffuse[0], (tuple, list)):
            diffuse = _ranges_to_sets(diffuse, self._nstep.diffuse)
        if rfold is not None and isinstance(rfold[0], (tuple, list)):
            rfold = _ranges_to_sets(rfold, self._nstep.rfold)
        param = _TrueOnIters(self, design, diffuse, rfold, levels)
        self._add_param(name, param, **kw)

    def newparam_false_in_range(self, name, **kw):
        self.newparam_true_in_range(name, **kw)
        self._params[name] = _Not(self._params[name])

    def newparam_spline_1d(self, name, design=None, diffuse=None, rfold=None, **kw):
        self._add_param(name, _Spline1D(self, design, diffuse, rfold, **kw))

    def newparam_spline_2d(self, name, diffuse_rfold, **kw):
        self._add_param(name, _Spline2D(self, diffuse_rfold, **kw))

    def newparam_constant(self, name, value, **kw):
        self._add_param(name, value, **kw)

    ######################################################################################

    def _sanity_check(self):
        # print(id(self), self._step, flush=True)
        assert 'design' not in self._params
        assert 'diffuse' not in self._params
        assert 'rfold' not in self._params
        assert 'tag' not in self._params
        assert self._nstep.design is not None
        assert self._nstep.diffuse is not None
        assert self._nstep.rfold is not None

        if 'pytest' not in sys.modules:
            if self._step.design is not None: assert 0 <= self._step.design < self._nstep.design
            if self._step.diffuse is not None: assert 0 <= self._step.diffuse < self._nstep.diffuse
            if self._step.rfold is not None: assert -1 <= self._step.rfold < self._nstep.rfold

        for p in self._params.values():
            if isinstance(p, DynamicParam):
                p._sanity_check()

    def _add_param(self, name, param, overwrite=False):
        if name.startswith('_'):
            raise ValueError(f'DynamicParameter names cant start with "_" {name}')
        if name.startswith('newparam'):
            raise ValueError(f'DynamicParameter names cant start with "newparam" {name}')
        if not overwrite and name in self._params:
            raise ValueError(f'DynamicParameter called "{name}" already exists')
        self._params[name] = param

    def _add_params(self, overwrite=True, **kw):
        for k, v in kw.items():
            self._add_param(k, v, overwrite=overwrite)

    def has(self, k):
        'like contains, except returns false for None values'
        return k in self._params and self._params[k] is not None

    def istrue(self, k):
        return k in self._params and self._params[k] is True

    def __contains__(self, k):
        if k != '_params':
            return k in self._params  # and self._params[k] is not None
        return False

    def __getattr__(self, k):
        if k != '_params' and k in self._params:
            param = self._params[k]
            return param.value() if isinstance(param, DynamicParam) else param
        raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, k))

    def __getitem__(self, k):
        if k != '_params' and k in self._params:
            param = self._params[k]
            return param.value() if isinstance(param, DynamicParam) else param
        raise KeyError("%r object has no key %r" % (self.__class__.__name__, k))

    def __setattr__(self, k, v):
        if k.startswith('_'): self.__dict__[k] = v
        elif '_params' in self.__dict__: self._params[k] = v
        else: raise AttributeError(f'cant overwrite attribute {k} to {v}')

    def __setitem__(self, k, v):
        if k.startswith('_'): self.__dict__[k] = v
        elif '_params' in self.__dict__: self._params[k] = v
        else: raise KeyError(f'cant overwrite key {k} to {v}')

    def __iter__(self):
        return iter(self._params)

    def __len__(self):
        return len(self._params)

    def _set_nstep(self, design=None, diffuse=None, rfold=None):
        for name, param in self._parsed_params.items():
            self.parse_dynamic_param(name, param, overwrite=True)
        design = self._nstep.design if design is None else design
        diffuse = self._nstep.diffuse if diffuse is None else diffuse
        rfold = self._nstep.rfold if rfold is None else rfold
        self._nstep = Step(design, diffuse, rfold)
        self._sanity_check()
        return self

    def _set_step(self, design=9e9, diffuse=9e9, rfold=9e9):
        self._step = self._get_step(design, diffuse, rfold)
        self._sanity_check()
        return self

    def _get_step(self, design=9e9, diffuse=9e9, rfold=9e9):
        design = self._step.design if design == 9e9 else design
        diffuse = self._step.diffuse if diffuse == 9e9 else diffuse
        rfold = self._step.rfold if rfold == 9e9 else rfold
        if design and design < 0: design += self._nstep.design
        if diffuse and diffuse < 0: diffuse += self._nstep.diffuse
        if rfold and rfold < 0: rfold += self._nstep.rfold
        return Step(design, diffuse, rfold)

    def _rfold_iter_begin(self, tag):  # sourcery skip: class-extract-method
        # assert tag not in self._rfold_tags, f'iter recorded twice! {self._step} {tag}'
        self._rfold_tags.add(tag)
        rfold = 0 if self._step.rfold is None else self._step.rfold + 1
        self._step = Step(self._step.design, self._step.diffuse, rfold)
        self._sanity_check()
        return self

    def _rfold_begin(self):
        self._step = Step(self._step.design, self._step.diffuse, -1)
        self._sanity_check()
        return self

    def _rfold_end(self):
        self._step = Step(self._step.design, self._step.diffuse, None)
        self._rfold_tags.clear()
        self._sanity_check()
        return self

    def _in_rf2aa(self):
        return self._step.rfold is not None

    @property
    def tag(self):
        try:
            return "_".join(f'{_:02}' for _ in self._step)
        except TypeError:
            return "_".join(str(_) for _ in self._step)

    def to_bunch(self):
        return ipd.dev.Bunch(self, tag=self.tag)

    # @property
    # def progress(self):
    # return [s / max(1, ts - 1) for s, ts in zip(self._step, self._nstep)]

    @property
    def design(self):
        return self._step.design

    @property
    def diffuse(self):
        return self._step.diffuse

    @property
    def rfold(self):
        return self._step.rfold

    def __repr__(self):
        mxlen = max(len(k) for k in self.keys())
        s = f'DynamicParameters{self._step},\n                  _nstep={self._nstep})\n'
        notset = []
        for k, v in self._params.items():
            if v is None:
                notset.append(k)
            else:
                s += f'   {k} {"."*(mxlen-len(k))} {v}\n'
        if notset:
            s += 'Parameters that are None:\n'
            s += '   ' + ', '.join(notset) + '\n'

        if len(s) < 80: s = s.replace('\n', '')
        return s

def _ranges_to_sets(thing, n):
    if thing is None: return None
    if not isinstance(thing[0], Iterable): thing = [thing]
    thing = [list(t) for t in thing]
    for i in range(len(thing)):
        if isinstance(thing[i], range): continue
        assert len(thing[i]) == 2
        if any(isinstance(x, float) for x in thing[i]):
            thing[i] = [float(x) for x in thing[i]]
        thing[i] = [int(n*x - 0.001) if isinstance(x, float) else x for x in thing[i]]
        thing[i] = [x + n if x < 0 else x for x in thing[i]]
        thing[i] = range(thing[i][0], thing[i][1] + 1)  # type: ignore
    s = set()
    for t in thing:
        s = s.union(set(t))
    return s

def _as_set(thing, n):
    if thing is None: return thing
    invert = False
    if isinstance(thing, _NotIn):
        thing = thing.vals
        assert thing is not None, '_Not(None) no make sense... excludues everything'
        invert = True
    try:
        thing = set(thing)
    except TypeError:
        thing = {thing}
    if any(isinstance(x, float) for x in thing):
        thing = {float(x) for x in thing}  # type: ignore
    # process neg vals from end
    if isinstance(thing, range): thing = list(thing)
    # thing = [list(x) if isinstance(x,range) else x for x in thing]
    thing = {x + 1 if (x < 0 and isinstance(x, float)) else x for x in thing}  # type: ignore
    thing = {int(n * x) if isinstance(x, float) else x for x in thing}
    thing = {x if x >= 0 else x + n for x in thing}  # type: ignore
    if invert: thing = {_ for _ in range(n) if _ not in thing}
    return thing

class _NotIn:
    def __init__(self, *vals):
        self.vals = vals

class DynamicParam(ABC):
    """Parameter that can change values through a process."""
    def __init__(self, manager=None):
        self.manager = manager

    def attach_manager(self, manager):
        self.manager = manager
        if hasattr(self, 'parent'):
            getattr(self, 'parent').manager = manager

    def __str__(self):
        with contextlib.suppress(TypeError):
            self.value()
        return f'{self.__class__.__name__}'

    @property
    def progress(self):
        return self.manager.progress()  # type: ignore

    @abstractmethod
    def value(self):
        pass

    def _sanity_check(self):
        pass

class _Not(DynamicParam):
    def __init__(self, parent):
        self.parent = parent
        self.manager = parent.manager

    def value(self):  # type: ignore
        return not self.parent.value()

class _TrueOnIters(DynamicParam):
    def __init__(self, manager, design, diffuse, rfold, levels=(True, False)):
        super().__init__(manager)
        ndesign, ndiffuse, nrfold = self.manager._nstep  # type: ignore
        self.design_steps = _as_set(design, ndesign)
        self.diffuse_steps = _as_set(diffuse, ndiffuse)
        self.rfold_steps = _as_set(rfold, nrfold)
        self.levels = levels

    def value(self):
        design_step, diffuse_step, rfold_step = self.manager._step  # type: ignore
        if self.design_steps is not None:
            # assert design_step is not None, 'DynamicParam missing _step info'
            if design_step is not None and design_step not in self.design_steps: return self.levels[1]
            if design_step is None: return self.levels[1]
        if self.diffuse_steps is not None:
            # assert diffuse_step is not None, 'DynamicParam missing _step info'
            if diffuse_step is not None and diffuse_step not in self.diffuse_steps: return self.levels[1]
            if diffuse_step is None: return self.levels[1]
        if self.rfold_steps is not None:
            # assert rfold_step is not None, 'DynamicParam missing _step info'
            if rfold_step is not None and rfold_step not in self.rfold_steps: return self.levels[1]
            if rfold_step is None: return self.levels[1]
        return self.levels[0]

    def __str__(self):
        s = super().__str__()
        extra = '' if self.levels == (True, False) else f' levels = {self.levels}'
        if self.design_steps: s += f' design:  {str(self.design_steps)}{extra}'
        if self.diffuse_steps: s += f' diffuse: {str(self.diffuse_steps)}{extra}'
        if self.rfold_steps: s += f' rfold:   {str(self.rfold_steps)}{extra}'
        return s

class _Spline1D(DynamicParam):
    def __init__(self, manager, design, diffuse, rfold, **kw):
        pytest.importorskip('scipy')
        import numpy as np
        from scipy.interpolate import CubicSpline  # type: ignore
        super().__init__(manager)
        if 1 != sum([design is not None, diffuse is not None, rfold is not None]):
            raise ValueError('add_spline_1d requires exactly one of design, diffuse, or rfold ')
        self.which, vals, n = 'design', design, self.manager._nstep.design  # type: ignore
        if diffuse is not None: self.which, vals, n = 'diffuse', diffuse, self.manager._nstep.diffuse  # type: ignore
        if rfold is not None: self.which, vals, n = 'rfold', rfold, self.manager._nstep.rfold  # type: ignore
        x, y = [np.array(_) for _ in zip(*vals)]
        if not np.all(np.logical_and(-0.001 <= x, x <= 1.001)):
            raise ValueError(f'interpolation points {x} must be 0 <= x <= 1')
        self.spline = CubicSpline(x, y, **kw)
        self.interpvals = self.spline(np.arange(n) / (n-1))

    def value(self):  # type: ignore
        design_step, diffuse_step, rfold_step = self.manager._step  # type: ignore
        if self.which == 'design': return self.interpvals[design_step]
        elif self.which == 'diffuse': return self.interpvals[diffuse_step]
        elif self.which == 'rfold': return self.interpvals[rfold_step]
        else: assert self.which in 'design diffuse rfold'.split()

class _Spline2D(DynamicParam):
    def __init__(self, manager, diffuse_rfold, **kw):
        pytest.importorskip('scipy')
        import numpy as np
        from scipy.interpolate import CloughTocher2DInterpolator  # type: ignore
        from scipy.spatial import QhullError  # type: ignore
        super().__init__(manager)
        x, y, z = [np.array(_) for _ in zip(*diffuse_rfold)]
        if len(x) < 3:
            raise ValueError('need at least 3 points to interpolate from')
        xy = np.stack([x, y], axis=-1)
        if not np.all(np.logical_and(-0.001 <= x, x <= 1.001)):
            raise ValueError(f'interpolation points {x} must be 0 <= x <= 1')
        if not np.all(np.logical_and(-0.001 <= y, y <= 1.001)):
            raise ValueError(f'interpolation points {y} must be 0 <= y <= 1')
        self.interp = CloughTocher2DInterpolator(xy, z)
        xx, yy = np.meshgrid(np.linspace(0, 1, manager._nstep.diffuse), np.linspace(0, 1, manager._nstep.rfold))
        try:
            self.interpvals = self.interp(xx, yy).astype(np.float32).T
            assert not np.any(np.isnan(self.interpvals))
        except (QhullError, AssertionError):
            msg = ('=' * 80)
            msg += ('\nconvex hull of interpolation points doesnt cover the unit square')
            msg += ('\nINTERP POINTS')
            for v in diffuse_rfold:
                msg += '\n' + str(v)
            msg += '\n'
            msg += '=' * 80
            raise ValueError(msg)

    def value(self):
        _, diffuse_step, rfold_step = self.manager._step  # type: ignore
        return self.interpvals[diffuse_step, rfold_step]
