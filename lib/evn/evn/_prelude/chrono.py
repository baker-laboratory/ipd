from contextlib import contextmanager
import types
import typing as t
from time import perf_counter
from dataclasses import dataclass, field

import evn
from evn._prelude.make_decorator import make_decorator

@dataclass
class Chrono:
    name: str = 'Chrono'
    verbose: bool = False
    start_time: float = field(default_factory=perf_counter)
    scopestack: list = field(default_factory=list)
    times: dict[str, list] = field(default_factory=dict)
    times_tot: dict[str, list] = field(default_factory=dict)
    entered: bool = False
    _pre_checkpoint_name: str | None = None
    stopped: bool = False
    debug: bool = False

    def __post_init__(self):
        self.start()

    def clear(self):
        self.times.clear()
        self.times_tot.clear()
        self.scopestack.clear()
        self.start()

    def start(self):
        assert not self.stopped
        self.scopestack.append(TimerScope(self.name))

    def stop(self):
        """Stop the chrono and store total elapsed time."""
        assert not self.stopped
        self.exit_scope(self.name)
        self.store_finished_scope(TimerScope('total', 0, self.elapsed()))
        self.stopped = True

    def __enter__(self):
        if not self.entered: self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type: print(f'An exception of type {exc_type} occurred: {exc_val}')
        self.stop()

    def scope(self, scopekey) -> t.ContextManager:
        name = self.check_scopekey(scopekey, 'scope')

        @contextmanager
        def cm(chrono=self):
            chrono.enter_scope(name)
            yield
            chrono.exit_scope(name)

        return cm()

    def store_finished_scope(self, scope):
        assert not (self.stopped or scope.stopped)
        # print(scope.name, evn.ident.hash(scope), t)
        time, tottime = scope.final()
        self.times.setdefault(scope.name, []).append(time)
        self.times_tot.setdefault(scope.name, []).append(tottime)

    def scope_name(self, obj: 'str|object') -> str:
        if isinstance(obj, str): return obj
        if hasattr(obj, '__module__'):
            return f'{obj.__module__}.{obj.__qualname__}'.replace('.<locals>', '')
        return f'{obj.__qualname__}'.replace('.<locals>', '')

    def enter_scope(self, scopekey):
        self._pre_checkpoint_name = None
        name = self.check_scopekey(scopekey, 'enter_scope')
        if self.scopestack:
            self.scopestack[-1].subscope_begins()
        self.scopestack.append(TimerScope(name))

    def exit_scope(self, scopekey: str | object):
        self._pre_checkpoint_name = None
        name = self.check_scopekey(scopekey, 'exit_scope')
        if not self.scopestack: raise RuntimeError('Chrono is not running')
        err = f'exiting scope: {name} doesnt match: {self.scopestack[-1].name}'
        assert self.scopestack[-1].name == name, err
        self.store_finished_scope(self.scopestack.pop())
        if self.scopestack: self.scopestack[-1].subscope_ends()

    def checkpoint(self, scopekey: str | object):
        oldname = self.scopestack[-1].name
        newname = self.check_scopekey(scopekey, 'checkpoint')
        if not self._pre_checkpoint_name: self._pre_checkpoint_name = oldname
        self.scopestack[-1].name = newname
        self.store_finished_scope(self.scopestack.pop())
        self.scopestack.append(TimerScope(self._pre_checkpoint_name))

    def check_scopekey(self, scopekey, label):
        if self.debug: print(label, scopekey)
        assert not self.stopped
        return self.scope_name(scopekey)

    def elapsed(self) -> float:
        """Return the total elapsed time."""
        return perf_counter() - self.start_time

    def report_dict(self, order='longest', summary=sum, tottime=False):
        """
        Generate a report dictionary of
         times.

        Args:
            order (str): Sorting order ('longest' or 'callorder').
            summary (callable): Function to summarize times (e.g., sum, mean).

        Returns:
            dict: Checkpoint times summary.
        """
        times = self.times_tot if tottime else self.times
        keys = times.keys()
        if order == 'longest':
            sorted_items = sorted(keys, key=lambda k: times[k], reverse=True)
        elif order == 'callorder':
            sorted_items = keys
        else:
            raise ValueError(f'Unknown order: {order}')
        return {k: summary(times[k]) for k in sorted_items}

    def report(self, order='longest', summary=sum, printme=True) -> str:
        """
        Print or return a report of
         profile.

        Args:
            order (str): Sorting order ('longest' or 'callorder').
            summary (callable): Function to summarize profile (e.g., sum, mean).
            printme (bool): Whether to print the report.

        Returns:
            str: Report string.
        """
        profile = self.report_dict(order=order, summary=summary)
        report_lines = [f'Chrono Report ({self.name})']
        report_lines.extend(f'{name}: {time_:.6f}s' for name, time_ in profile.items())
        report = '\n'.join(report_lines)
        if printme:
            print(report)
        return report

    def find_times(self, name):
        return next((v for k, v in self.times.items() if name in k), None)

@dataclass
class TimerScope:
    name: str
    start_time: float = field(default_factory=perf_counter)
    sub_start: float = field(default_factory=perf_counter)
    total: float = 0
    subtotal: float = 0
    stopped: bool = False
    debug: bool = False

    def final(self):
        if self.debug: print('final', self.name, perf_counter(), self.sub_start, self.subtotal)
        assert self.sub_start < 9e8
        self.stopped, subtotal = True, perf_counter() - self.sub_start + self.subtotal
        return subtotal, perf_counter() - self.start_time

    def subscope_begins(self):
        if self.debug: print('subscope_begins', self.name)
        self.subtotal += perf_counter() - self.sub_start
        self.sub_start = 9e9

    def subscope_ends(self):
        if self.debug: print('subscope_ends', self.name)
        self.sub_start = perf_counter()

evn.chronometer = Chrono('main')

def chrono_enter_scope(name, **kw):
    global chronometer
    chrono = kw.get('chrono', evn.chronometer)
    chrono.enter_scope(name, **kw)

def chrono_exit_scope(name, **kw):
    global chronometer
    chrono = kw.get('chrono', evn.chronometer)
    chrono.exit_scope(name, **kw)

def chrono_checkpoint(name, **kw):
    global chronometer
    chrono = kw.get('chrono', evn.chronometer)
    chrono.checkpoint(name, **kw)

@make_decorator(chrono=evn.chronometer)
def chrono(wrapped, *args, chrono=None, **kw):
    chrono2: Chrono = kw.get('chrono', chrono)
    chrono2.enter_scope(wrapped)
    result = wrapped(*args, **kw)
    chrono2.exit_scope(wrapped)
    if not isinstance(result, types.GeneratorType):
        return result
    return _generator_proxy(wrapped, result, chrono2)

def _generator_proxy(gener, wrapped, chrono):
    try:
        geniter = iter(gener)
        while True:
            chrono.enter_scope(wrapped)
            item = next(geniter)
            chrono.exit_scope(wrapped)
            yield item
    except StopIteration:
        pass
    finally:
        chrono.exit_scope(wrapped)
        if hasattr(gener, 'close'):
            gener.close()
