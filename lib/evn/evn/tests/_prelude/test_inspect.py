# test_inspect.py

import types
import builtins
import pytest
from evn._prelude import inspect as insp

def main():
    import evn
    evn.testing.quicktest(namespace=globals())

def test_summary_basic_types():
    insp42 = insp.summary(42, debug=True)
    print(insp42)
    assert insp42 == '42'
    assert insp.summary("hello") == 'hello'
    assert insp.summary([1, 2]) == "[1, 2]"
    assert "Type: int" in insp.summary(int)
    assert "Type: list" in insp.summary(list)

def test_summary_function_and_method():

    def dummy_fn():
        pass

    class Dummy:

        def method(self):
            pass

    assert "dummy_fn" in insp.summary(dummy_fn)
    assert "Dummy.method" in insp.summary(Dummy().method)

def test_summary_numpy_array():
    np = pytest.importorskip("numpy")
    a = np.arange(5)
    b = np.arange(100)
    assert insp.summary(a) == str(a)
    assert insp.summary(b).startswith("ndarray[")

def test_summary_torch_tensor():
    torch = pytest.importorskip("torch")
    t = torch.arange(5)
    t_large = torch.arange(100)
    assert insp.summary(t) == str(t)
    assert insp.summary(t_large).startswith("Tensor[")

def test_diff_basic_sets():
    assert insp.diff({1, 2}, {2, 3}) == {1, 3}

def test_show_prints(monkeypatch, capsys):
    called = {}

    def fake_inspect(obj, **kw):
        called["obj"] = obj

    monkeypatch.setattr(insp, "show_impl", fake_inspect)
    result = insp.show("hello")
    assert called["obj"] == "hello"
    assert result is None

def test_inspect_alias(monkeypatch):
    called = {}

    def fake_show(obj, **kw):
        called["obj"] = obj

    monkeypatch.setattr(insp, "show_impl", fake_show)
    assert insp.inspect("world") is None
    assert called["obj"] == "world"

def test_trace_decorator(capsys):
    insp.evn.show_trace = True

if __name__ == '__main__':
    main()
