import abc
import inspect

class ObserverError(Exception):
    pass

class ObserverMethod:
    """This class is used to call a method on all observers that have that
    method."""
    def __init__(self, subject, method):
        self.subject = subject
        self.method = method

    def __call__(self, *args, strict=False, **kw):
        if self.method in 'set_config shutdown'.split():
            getattr(self.subject, f'_{self.method}')(*args, **kw)
        if self.method not in self.subject._allmethods:
            warn = f'no observers have registered for {self.method}. all methods registered: {self.subject._allmethods}'
            if strict: raise ObserverError(warn)
            if warn not in self.subject._warnings: self.subject._warnings.add(warn)
        results = list()
        for observer in self.subject._observers.values():
            try:
                _METHOD_ = getattr(observer, self.method)
            except AttributeError:
                continue
            if _METHOD_.__name__ != self.method: kw['methodname'] = self.method
            results.append(_METHOD_(*args, **kw))
        return results

class Subject:
    """This class is used to register observers and call their methods."""
    def __init__(self):
        self._observers = dict()
        self._allmethods = set()
        self._warnings = set()

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
            for k, v in conf.viz.items():
                if k != 'settings':
                    self._allmethods.add(k)

    def _shutdown(self):
        pass
        # for e in self._warnings:
        # print(e)

    def __getattr__(self, k: str):
        if k == '_observers':
            raise AttributeError
        return ObserverMethod(self, k)

    def __getitem__(self, cls):
        return self._observers[cls]

hub = Subject()

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
