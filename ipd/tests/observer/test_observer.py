import pytest
import ipd

class ObserverTest(ipd.observer.Observer):
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

@pytest.mark.fast
def test_observer():
    agent = ipd.spy[ObserverTest]
    assert not agent.foobar_called
    ipd.spy.blah(check_is_registered_method=False)
    with pytest.raises(ipd.observer.ObserverError):
      ipd.spy.blah(strict=True)
    assert not agent.foobar_called
    ipd.spy.foobar()
    assert agent.foobar_called
    ipd.spy.idx(7)
    assert agent.idx_called == 7
    assert ObserverTest() is ObserverTest()
    assert ObserverTest() is ipd.spy[ObserverTest]

if __name__ == '__main__':
    test_observer()
