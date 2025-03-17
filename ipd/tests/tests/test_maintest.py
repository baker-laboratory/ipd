import ipd
import ipd.tests.tests.maintest_test_namespace as testns

def main():
    test_maintest()

def test_maintest():
    result = ipd.tests.maintest(testns)
    assert 3 == len(result.passed)
    assert 1 == len(result.skipexcn)
    assert 0 == len(result.errored)
    assert 0 == len(result.failed)
    assert 1 == len(result.xfailed)

if __name__ == '__main__':
    main()
