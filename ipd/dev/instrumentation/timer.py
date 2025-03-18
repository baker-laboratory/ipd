import collections
import functools
import inspect
import logging
import os
import statistics
import time

import ipd

log = logging.getLogger(__name__)

_summary_types = dict(
    sum=sum,
    mean=statistics.mean,
    min=min,
    max=max,
    median=statistics.median,
)

class _TimerGetter:

    def __init__(self, timer, summary):
        self.timer = timer
        self.summary = summary

    def __getattr__(self, name):
        if name in ("timer", "checkpoints"):
            raise AttributeError
        if name in self.timer.checkpoints:
            return self.summary(self.timer.checkpoints[name])
        raise AttributeError("Timer has no attribute named: " + name)

    def __getitem__(self, name):
        return getattr(self, name)

class Timer:

    def __init__(
        self,
        name="Timer",
        verbose=True,
        start=True,
    ):
        self.name = name
        self.verbose = verbose
        self.sum = _TimerGetter(self, sum)
        self.mean = _TimerGetter(self, statistics.mean)
        self.min = _TimerGetter(self, min)
        self.max = _TimerGetter(self, max)
        self.median = _TimerGetter(self, statistics.median)
        self._start = 0
        self.checkpoints = collections.defaultdict(list)
        if start: self.start()
        self._in_interjection = None

    def start(self):
        return self.__enter__()

    def stop(self):
        return self.__exit__()

    def __enter__(self):
        if self._start:
            if time.perf_counter() - self._start > 0.001:
                raise ValueError("Timer already started")
        if self.verbose:
            log.debug(f"Timer {self.name} intialized")
        self._start = time.perf_counter()
        self.last = self._start
        self.lastname = "start"
        return self

    def checkpoint(
        self,
        name=None,
        verbose=False,
        keeppriorname=False,
        autolabel=False,
        interject=False,
    ):
        if name is None or interject: name = '__interject__'
        elif not keeppriorname: self.lastname = str(name)
        name = str(name)
        if autolabel: name = name + "$$$$"
        t = time.perf_counter()

        if name == '__interject__':
            self._in_interjection = '__waiting__'
        elif self._in_interjection == '__waiting__':
            self._in_interjection = name
        elif self._in_interjection and name != self._in_interjection:
            interjection_credit = sum(self.checkpoints.pop('__interject__'))
            self.checkpoints[name].append(interjection_credit)
            self._in_interjection = None

        self.checkpoints[name].append(t - self.last)
        self.last = t
        if self.verbose or verbose:
            log.debug(f"{self.name} checkpoint {name} iter {len(self.checkpoints[name])}" +
                      f"time {self.checkpoints[name][-1]}")
        return self

    def elapsed(self) -> float:
        return sum(self.checkpoints['total'])

    def __exit__(
        self,
        type=None,
        value=None,
        traceback=None,
    ):
        self.checkpoints["total"].append(time.perf_counter() - self._start)
        # self._start = None
        if self.verbose:
            log.debug(f"Timer {self.name} finished")
            self.report()

    def __getattr__(self, name):
        if name == "checkpoints":
            raise AttributeError
        if name in self.checkpoints:
            return self.checkpoints[name]
        raise AttributeError("Timer has no attribute named: " + name)

    def alltimes(self, name):
        return self.checkpoints[name]

    def report_dict(
        self,
        order="longest",
        summary="sum",
        timecut: float = 0,
    ):
        if not callable(summary):
            if summary not in _summary_types:
                raise ValueError("unknown summary type: " + str(summary))
            summary = _summary_types[summary]
        if order == "longest":
            reordered = sorted(self.checkpoints.items(), key=lambda kv: -summary(kv[1]))
            report = {k: summary(v) for k, v in reordered}
        elif order == "callorder":
            report = {k: summary(v) for k, v in self.checkpoints.items()}
        else:
            raise ValueError("Timer, unknown order: " + order)
        return {k: v for k, v in report.items() if v > timecut}

    def report(
        self,
        order="longest",
        summary="sum",
        namelen=60,
        precision="10.5f",
        printme=True,
        scale=1.0,
        timecut=0.001,
        file=None,
        pattern="",
    ):
        if len(self.checkpoints) <= 1: timecut = 0
        namelen = min(namelen, max(len(n.rstrip("$")) for n in self.checkpoints) if self.checkpoints else 0)
        lines = [f"Times(name={self.name}, order={order}, summary={summary}):"]
        times = self.report_dict(order=order, summary=summary, timecut=timecut)

        if not times: times["total$$$$"] = time.perf_counter() - self._start
        for checkpoint, t in times.items():
            if not checkpoint.count(pattern): continue
            a = " " if checkpoint.endswith("$$$$") else "*"
            cpstr = checkpoint.rstrip("$")[-namelen:]
            lines.append(f'{t*scale:{precision}} {a} {cpstr:<{namelen}}')
            if scale == 1000: lines[-1] += "ms"
        r = os.linesep.join(lines)
        if printme:
            print()
            if file is None:
                print(r, flush=True)
            else:
                with open(file, "w") as out:
                    out.write(r + os.linesep)
        return r

    @property
    def total(self):
        if "total" in self.checkpoints:
            return sum(self.checkpoints["total"])
        return time.perf_counter() - self._start

    def __str__(self):
        return self.report(printme=False)

    def __repr__(self):
        return str(type(self))

    def merge(self, others):
        if isinstance(others, Timer):
            others = [others]
        for other in others:
            for k, v in other.checkpoints.items():
                self.checkpoints[k].extend(v)

def checkpoint(kw={},
               label=None,
               funcbegin=False,
               dont_mod_label=False,
               filename=None,
               clsname=None,
               funcname=None,
               interject=False):
    t = None
    if isinstance(kw, Timer): t = kw
    elif "timer" in kw: t = kw["timer"]
    else: t = ipd.dev.global_timer
    if interject: return t.checkpoint(interject=True)
    # autogen_label = False
    istack = 1 + int(funcbegin)
    func = funcname or inspect.stack()[istack][3]
    fn = filename or os.path.basename(inspect.stack()[istack][1])
    fulllabel = label
    clsname = f'{clsname}.' if clsname else ''
    if not dont_mod_label:
        fulllabel = f"{fn}:{clsname}{func}"
    if label:
        fulllabel += f":{label}"
    t.checkpoint(fulllabel, autolabel=label is None)

def timed_func(func, *, label=None):
    if '__file__' in func.__globals__:
        filen = os.path.basename(func.__globals__["__file__"])
    else:
        filen = '???'
    funcn = func.__qualname__

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*a, **kw):
            kwarg = dict(label=label, filename=filen, funcname=funcn)
            # should interject instead? refactor to use stack?
            checkpoint(kw, funcbegin=True, **kwarg)
            val = await func(*a, **kw)
            checkpoint(kw, **kwarg)
            return val
    else:

        @functools.wraps(func)
        def wrapper(*a, **kw):
            kwarg = dict(label=label, filename=filen, funcname=funcn)
            # should interject instead? refactor to use stack?
            checkpoint(kw, funcbegin=True, **kwarg)
            val = func(*a, **kw)
            checkpoint(kw, **kwarg)
            return val

    return wrapper

def timed_class(cls, *, label=None):
    # label = label or rs
    for k, v in vars(cls).items():
        if callable(v) and not inspect.isclass(v):  # skip inner classes
            setattr(cls, k, timed_func(v))

    return cls

def timed(thing=None, *, label=None, name=None):
    if thing is None:
        return functools.partial(timed, label=label)
    if name: thing.__qualname__ = name
    if inspect.isclass(thing):
        return timed_class(thing, label=label)
    else:
        return timed_func(thing, label=label)
