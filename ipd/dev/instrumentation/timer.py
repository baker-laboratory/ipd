import collections
import functools
import inspect
import logging
import os
import statistics
import time

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
        self._start = None
        self.checkpoints = collections.defaultdict(list)
        if start:
            self.start()

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
    ):
        name = str(name)
        if name is None:
            name = self.lastname
        elif not keeppriorname:
            self.lastname = name
        if autolabel:
            name = name + "$$$$"
        t = time.perf_counter()
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
        self.checkpoints["total"].append(time.perf_counter() - self._start)  # type: ignore
        self._start = None
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
        timecut=0,
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
        namelen=None,
        precision="10.5f",
        printme=True,
        scale=1.0,
        timecut=0,
        file=None,
        pattern="",
    ):
        if namelen is None:
            namelen = max(len(n.rstrip("$")) for n in self.checkpoints) if self.checkpoints else 0
        lines = [f"Times(name={self.name}, order={order}, summary={summary}):"]
        times = self.report_dict(order=order, summary=summary, timecut=timecut)
        if not times:
            times["total$$$$"] = time.perf_counter() - self._start  # type: ignore
        for cpoint, t in times.items():
            if not cpoint.count(pattern):
                continue
            a = " " if cpoint.endswith("$$$$") else "*"
            lines.append(f'    {cpoint.rstrip("$"):>{namelen}} {a} {t*scale:{precision}}')
            if scale == 1000:
                lines[-1] += "ms"
        r = os.linesep.join(lines)
        if printme:
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
        return time.perf_counter() - self._start  # type: ignore

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

def checkpoint(kw, label=None, funcbegin=False, dont_mod_label=False, filename=None, clsname=None, funcname=None):
    t = None
    if isinstance(kw, Timer):
        t = kw
    elif "timer" in kw:
        t = kw["timer"]
    else:
        t = global_timer
    # autogen_label = False
    istack = 1 + int(funcbegin)
    func = funcname or inspect.stack()[istack][3]
    fn = filename or os.path.basename(inspect.stack()[istack][1])
    fulllabel = label
    clsname = f'{clsname}.' if clsname else ''
    if not dont_mod_label:
        fulllabel = f"{fn}:{clsname}{func}"
    if label:
        fulllabel += f":{label}"  # type: ignore
    t.checkpoint(fulllabel, autolabel=label is None)

def timed_func(func, *, label=None):
    if '__file__' in func.__globals__:
        filen = os.path.basename(func.__globals__["__file__"])
    else:
        filen = '???'
    funcn = func.__qualname__

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*a, **kw):  # type: ignore
            kwarg = dict(label=label, filename=filen, funcname=funcn)
            checkpoint(kw, funcbegin=True, **kwarg)  # type: ignore
            val = await func(*a, **kw)
            checkpoint(kw, **kwarg)  # type: ignore
            return val
    else:

        @functools.wraps(func)
        def wrapper(*a, **kw):
            kwarg = dict(label=label, filename=filen, funcname=funcn)
            checkpoint(kw, funcbegin=True, **kwarg)  # type: ignore
            val = func(*a, **kw)
            checkpoint(kw, **kwarg)  # type: ignore
            return val

    return wrapper

def timed_class(cls, *, label=None):
    # label = label or rs
    for k in cls.__dict__:
        v = getattr(cls, k)
        if callable(v) and not inspect.isclass(v):  # skip inner classes
            setattr(cls, k, timed_func(v))

    return cls

def timed(thing=None, *, label=None):
    if thing is None:
        return functools.partial(timed, label=label)

    if inspect.isclass(thing):
        return timed_class(thing, label=label)
    else:
        return timed_func(thing, label=label)

global_timer = Timer()
