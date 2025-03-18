import abc
import inspect
import re

null_func = lambda *a, **kw: None

class ObserverError(Exception):
    pass

class ObserverMethod:
    """This class is used to call a method on all observers that have that
    method."""

    def __init__(self, subject, method, **kw):
        self.subject = subject
        self.method = method
        self.kw = kw

    def __call__(self, *args, strict=False, **kw):
        if self.method in 'set_config shutdown'.split():
            getattr(self.subject, f'_{self.method}')(*args, **kw)
        if self.method not in self.subject._allmethods:
            warn = f'no observers have registered for {self.method}. all methods registered: {self.subject._allmethods}'
            if strict: raise ObserverError(warn)
            if warn not in self.subject._warnings: self.subject._warnings.add(warn)
        results = list()
        mergekw = {**self.kw, **kw}
        if self.subject.call_is_allowed(*args, methodname=self.method, **mergekw):
            for observer in self.subject._observers.values():
                try:
                    _METHOD_ = getattr(observer, self.method)
                except AttributeError:
                    continue
                # for callable members not defined as metods e.g. self.alias = self.original
                if _METHOD_.__name__ != self.method: kw['methodname'] = self.method
                results.append(_METHOD_(*args, **mergekw))
        return results

def process_regex(patterns):
    if isinstance(patterns, str): patterns = [patterns]
    return list(map(re.compile, patterns))

class Subject:
    """This class is used to register observers and call their methods."""

    def __init__(self, conf=None):
        self._observers = dict()
        self._allmethods = set()
        self._warnings = set()
        self._debug_level = 0
        self._debug_regex = self._debug_not_regex = self._debug_always_regex = ''

    def _register_instance(self, observer: 'Observer'):
        cls = observer.__class__
        if cls in self._observers:
            raise ObserverError(f'{cls} is already an observer')
        if not isinstance(observer, Observer):
            raise ObserverError(f'{observer} is not an Observer')
        for method in inspect.getmembers(observer, predicate=inspect.ismethod):
            if method[0] == '__init__': continue
            if method[0][0] == '_': continue
            self._allmethods.add(method[0])
        self._observers[cls] = observer

    def _set_config(self, conf):
        """Register methods listed in conf.viz."""
        if 'viz' in conf:
            self._debug_level = conf.viz.debug if conf else 0
            self._debug_regex = process_regex(conf.viz.debug_regex if conf else '')
            self._debug_not_regex = process_regex(conf.viz.debug_not_regex if conf else '')
            self._debug_always_regex = process_regex(conf.viz.debug_always_regex if conf else '')
            for k, v in conf.viz.items():
                if k != 'settings':
                    self._allmethods.add(k)

    def _shutdown(self):
        pass
        # for e in self._warnings:
        # print(e)

    def call_is_allowed(self, *args, methodname='', **kw):
        if methodname == 'debug':
            if 'name' not in kw and args and isinstance(args[0], str): kw['name'] = args[0]
            if 'name' not in kw: raise ValueError('ipd debug functions must be called with a name argument')
            lvlok = kw.get('lvl', 100) <= self._debug_level
            reok, forbid, force = (any(r.search(kw['name']) for r in pat) for pat in (
                self._debug_regex,
                self._debug_not_regex,
                self._debug_always_regex,
            ))
            return force or (lvlok and reok and not forbid)
        return True

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(f'Subject has no attribute {name}')
        elif name.startswith('debug'):
            lvl = int(name[5:]) if name[5:].isdigit() else 100
            return ObserverMethod(self, 'debug', lvl=lvl)
        else:
            return ObserverMethod(self, name)

    def __getitem__(self, cls):
        return self._observers[cls]

hub = Subject()

def hub_init_hydra(conf):
    global hub
    hub = Subject(conf)

class Observer(abc.ABC):
    """Base class for all Observers, must define set_config and use it to
    configure themselves.

    A single instance of each subclass of Observer will be created and registered by the hub
    Subject. These instances will be accessible via hub[MyObserver]. For example, defining

    >>> class MyObserver(Observer):
    ...     def set_config(self, conf):
    ...         self.conf = conf
    ...     def thing_I_react_to(self, arg1, arg2, methodname):
    ...         print(f'{self} reacting to thing_I_react_to with {arg1} and {arg2}')

    will cause ipd.hub.thing_I_react_to(1,2) to call it's instance of thing_I_react_to
    the instance can be accessed via ipd.hub[MyObserver] or MyObserver()
    """
    _instances = dict()

    def __init_subclass__(cls, **kw):
        super().__init__(cls, **kw)  # type: ignore
        hub._register_instance(cls())

    def __new__(cls, *args, **kw):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls, *args, **kw)
        return cls._instances[cls]

    def set_config(self, conf, **kw):
        pass
