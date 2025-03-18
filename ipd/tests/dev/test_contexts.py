import os
import glob
import ipd

def main():
    ipd.tests.maintest(namespace=globals())

def test_nocontext():
    with ipd.dev.nocontext() as foo:
        assert foo is None

def test_cast():
    # cast(cls, self)
    ...

def test_redirect():
    # redirect(stdout=<_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>, stderr=<_io.TextIOWrapper name='<stderr>' mode='w' encoding='utf-8'>)
    ...

def test_cd():
    prev = os.getcwd()
    with ipd.dev.cd('/'):
        assert os.getcwd() == '/'
    assert os.getcwd() == prev

def test_openfiles():
    # openfiles(*fnames, **kw)
    pdbfiles = glob.glob(ipd.dev.package_testdata_path('pdb/*'))
    with ipd.dev.openfiles(pdbfiles) as files:
        pass
    assert all(f.closed for f in files)

def test_trace_prints():
    with ipd.dev.capture_stdio() as log:
        with ipd.dev.trace_prints():
            print('76125455762317357825346521683745')
    log = log.read()
    assert '76125455762317357825346521683745' in log
    assert 'test_trace_prints' in log
    assert 'test_contexts.py' in log

def test_capture_asserts():
    with ipd.dev.capture_asserts() as errors:
        assert 1, 'true'
        assert 0, 'foo'
        assert 0, 'bar'
    assert errors
    assert len(errors) == 1
    # assert errors[0] == AssertionError('foo')

if __name__ == '__main__':
    main()
