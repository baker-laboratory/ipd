"""
Observer pattern implementation for method dispatch in the IPD framework.

This module provides a dynamic event dispatch system using a centralized
:class:`Subject` instance (`hub`) and auto-registered singleton observer classes
derived from :class:`Observer`.

The key idea is that calling `hub.some_method(...)` dispatches the call to all
registered observers that define a method named `some_method`.

Use cases include debug reporting, instrumentation, UI hooks, and logging.

.. code-block:: python

    class MyObserver(Observer):
        def set_config(self, conf):
            self.conf = conf

        def event_happened(self, info, methodname=None):
            print("Received:", info)

    hub._debug = True
    hub.event_happened("hello")
    # Output (example):
    # === Observer Warnings ===
    # no observers have registered for shutdown. all methods registered: {...}
    # =========================

    print(hub.usage)
    # {'event_happened': ['MyObserver']}
"""

import abc
import inspect
import re

null_func = lambda *a, **kw: None

class ObserverError(Exception):
    """Raised when observer registration or dispatch fails."""

class ObserverMethod:
    """
    Proxy for a method call on a :class:`Subject`. Dispatches to all observers.

    This class is returned when accessing `hub.methodname`. When called,
    it checks which observers implement that method, and calls it on each of them.

    Handles special case logic (e.g. for `debug`, `set_config`, `shutdown`).

    .. warning::
        This proxy does not return a combined result — only a list of individual results.

    :param subject: The subject owning the observers
    :type subject: Subject
    :param method: The method name to proxy
    :type method: str
    :param kw: Default keyword arguments to merge into call
    :type kw: dict
    """
    def __init__(self, subject, method, **kw):
        self.subject = subject
        self.method = method
        self.kw = kw

    def __call__(self, *args, strict=False, **kw):
        """
        Call this observer proxy, dispatching to all observers that implement the method.

        :param args: Positional arguments passed to the observer methods
        :param strict: If True, raise an error when no observers implement the method
        :param kw: Additional keyword arguments to merge with defaults
        :return: List of return values from all matching observer methods
        :rtype: list
        :raises ObserverError: If method is missing and `strict` is True
        """
        if self.method in ('set_config', 'shutdown'):
            getattr(self.subject, f'_{self.method}')(*args, **kw)
        if self.method not in self.subject._allmethods:
            warn = f'no observers have registered for {self.method}. all methods registered: {self.subject._allmethods}'
            if strict:
                raise ObserverError(warn)
            if warn not in self.subject._warnings:
                self.subject._warnings.add(warn)
        results = []
        mergekw = {**self.kw, **kw}
        if self.subject.call_is_allowed(*args, methodname=self.method, **mergekw):
            for observer in self.subject._observers.values():
                try:
                    _METHOD_ = getattr(observer, self.method)
                except AttributeError:
                    continue
                if _METHOD_.__name__ != self.method:
                    mergekw['methodname'] = self.method
                results.append(_METHOD_(*args, **mergekw))
                self.subject._usage.setdefault(self.method, []).append(observer.__class__.__name__)
        return results

def process_regex(patterns):
    """Compile string or list of strings into regex patterns."""
    if isinstance(patterns, str):
        patterns = [patterns]
    return list(map(re.compile, patterns))

class Subject:
    """
    Central registry and dispatch controller for all :class:`Observer` instances.

    The singleton `hub` instance of this class controls observer registration and
    method dispatch. Calling `hub.foo()` will forward the call to all observers
    that implement `foo`.

    Debug-related calls are filtered using regex and level settings stored in
    `conf.viz` (a Hydra/OmegaConf config).
    """
    def __init__(self, conf=None):
        self._observers = dict()
        self._allmethods = set()
        self._warnings = set()
        self._usage = dict()
        self._debug = False
        self._debug_level = 0
        self._debug_regex = self._debug_not_regex = self._debug_always_regex = ''

    def _register_instance(self, observer: 'Observer'):
        """
        Internal method for observer auto-registration.

        :param observer: An instance of an :class:`Observer` subclass
        :raises ObserverError: If duplicate or invalid observer registration
        """
        cls = observer.__class__
        if cls in self._observers:
            raise ObserverError(f'{cls} is already an observer')
        if not isinstance(observer, Observer):
            raise ObserverError(f'{observer} is not an Observer')
        for method in inspect.getmembers(observer, predicate=inspect.ismethod):
            if method[0] == '__init__' or method[0].startswith('_'):
                continue
            self._allmethods.add(method[0])
        self._observers[cls] = observer

    def _set_config(self, conf):
        """Configure debug settings from `conf.viz`."""
        if 'viz' in conf:
            self._debug_level = conf.viz.debug if conf else 0
            self._debug_regex = process_regex(conf.viz.debug_regex if conf else '')
            self._debug_not_regex = process_regex(conf.viz.debug_not_regex if conf else '')
            self._debug_always_regex = process_regex(conf.viz.debug_always_regex if conf else '')
            for k in conf.viz:
                if k != 'settings':
                    self._allmethods.add(k)

    def _shutdown(self):
        """
        Print out observer warnings and method usage if `_debug` is enabled.
        """
        if not self._debug:
            return
        if self._warnings:
            print("=== Observer Warnings ===")
            for warning in sorted(self._warnings):
                print(warning)
            print("=========================")
        if self._usage:
            print("=== Observer Usage Summary ===")
            for method, users in sorted(self._usage.items()):
                print(f"{method}: {sorted(set(users))}")
            print("==============================")

    def call_is_allowed(self, *args, methodname='', **kw):
        """
        Determine whether an observer call is permitted (used for debug throttling).

        :param methodname: Method name to check
        :param args: Positional args (used to infer debug name)
        :param kw: Keyword args (must include `name` for debug)
        :return: True if allowed, False if suppressed
        :rtype: bool
        """
        if methodname == 'debug':
            if 'name' not in kw and args and isinstance(args[0], str):
                kw['name'] = args[0]
            if 'name' not in kw:
                raise ValueError('ipd debug functions must be called with a name argument')
            lvlok = kw.get('lvl', 100) <= self._debug_level
            reok, forbid, force = (any(r.search(kw['name']) for r in pat) for pat in (
                self._debug_regex,
                self._debug_not_regex,
                self._debug_always_regex,
            ))
            return force or (lvlok and reok and not forbid)
        return True

    def __getattr__(self, name: str):
        """
        Proxy undefined method calls to :class:`ObserverMethod`.

        Special case: names starting with `debugNN` are treated as debug-level calls.

        :param name: Method name
        :return: An :class:`ObserverMethod` instance
        :raises AttributeError: If accessing a private method
        """
        if name.startswith('_'):
            raise AttributeError(f'Subject has no attribute {name}')
        elif name.startswith('debug'):
            lvl = int(name[5:]) if name[5:].isdigit() else 100
            return ObserverMethod(self, 'debug', lvl=lvl)
        else:
            return ObserverMethod(self, name)

    def __getitem__(self, cls):
        """Access the singleton instance of a given Observer class."""
        return self._observers[cls]

    @property
    def usage(self):
        """Returns a dictionary of methods and the observers that handled them."""
        return self._usage

hub = Subject()

def hub_init_hydra(conf):
    """Reset or initialize `hub` with a new configuration."""
    global hub
    hub = Subject(conf)

class Observer(abc.ABC):
    """
    Base class for observer components that hook into the `hub`.

    Subclasses are instantiated once and auto-registered. Must define
    a `set_config()` method to configure themselves.

    Instances are accessible via `hub[MyObserver]` or `MyObserver()`.

    Example::

        class Logger(Observer):
            def set_config(self, conf):
                self.conf = conf

            def debug(self, msg, name=None, lvl=0, methodname=None):
                print(f"{name}: {msg}")
    """
    _instances = dict()

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        hub._register_instance(cls())

    def __new__(cls, *args, **kw):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls, *args, **kw)
        return cls._instances[cls]

    def set_config(self, conf, **kw):
        """Abstract method. Observers should configure themselves here."""
        pass
