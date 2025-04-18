import evn
from ipd.observer import observer

def main():
    evn.testing.quicktest(globals())

class LogObserver(observer.Observer):

    def set_config(self, conf):
        self.conf = conf

    def ping(self):
        return "pong"

    def debug(self, name, lvl=0, methodname=None):
        return f"debug: {name}"

def test_singleton_behavior():
    a = LogObserver()
    b = LogObserver()
    assert a is b
    assert observer.hub[LogObserver] is a

def test_method_dispatch():
    observer.hub._debug = False
    results = observer.hub.ping()
    assert results == ["pong"]

def test_debug_level_filtering():
    observer.hub._debug_level = 5
    observer.hub._debug_regex = [__import__("re").compile("foo.*")]
    observer.hub._debug_not_regex = []
    observer.hub._debug_always_regex = []
    out = observer.hub.debug3("foobar")
    assert out == ["debug: foobar"]

def test_warning_for_unimplemented():
    observer.hub._debug = False
    observer.hub._warnings.clear()
    observer.hub._allmethods.clear()
    observer.hub._observers.clear()
    observer.hub._register_instance(LogObserver())
    observer.hub.missing_method(strict=False)
    assert any("missing_method" in w for w in observer.hub._warnings)

def test_usage_tracking_and_shutdown():
    with evn.capture_stdio() as cap:
        observer.hub._debug = True
        observer.hub._warnings.clear()
        observer.hub._usage.clear()
        observer.hub.ping()
        observer.hub._shutdown()
    captured = cap.read()
    assert "Observer Usage Summary" in captured
    assert "ping" in captured

class ObserverTest(observer.Observer):

    def __init__(self):
        super().__init__()
        self.foobar_called = False
        self.idx_called = None

    def set_config(self, conf):
        pass

    def foobar(self):
        self.foobar_called = True

    def idx(self, i):
        self.idx_called = i

# def test_observer_cls():
#     agent = ipd.hub[ObserverTest]
#     assert not agent.foobar_called
#     ipd.hub.blah(check_is_registered_method=False)
#     with pytest.raises(ipd.observer.ObserverError):  # type: ignore
#         ipd.hub.blah(strict=True)
#     assert not agent.foobar_called
#     ipd.hub.foobar()
#     assert agent.foobar_called
#     ipd.hub.idx(7)
#     assert agent.idx_called == 7
#     assert ObserverTest() is ObserverTest()
#     assert ObserverTest() is ipd.hub[ObserverTest]

if __name__ == '__main__':
    main()
