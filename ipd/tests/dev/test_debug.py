import pytest
import ipd

config_test = ipd.Bunch(
    re_only=[
        #
    ],
    re_exclude=[
        #
    ],
)

def main():
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
    )

def test_ic_config():
    ipd.ic.configureOutput(includeContext=True)
    with ipd.dev.capture_stdio() as printed:
        ipd.ic(7)
        with ipd.dev.ic_config(includeContext=False):
            ipd.ic(8)
        ipd.ic(9)
    result = printed.read()
    print(result)
    assert result.count('ic|') == 3
    assert result.count('test_debug.py') == 2

@pytest.mark.xfail
def test_bypass_stdio_redirect():
    with ipd.dev.capture_stdio() as printed:
        print('a', flush=True)
        ipd.icm(1)
        print('b', flush=True)
        ipd.icv(1)
        print('c', flush=True)
        ipd.ic(1)
        print('d', flush=True)
    output = printed.read()
    assert output.count('test_bypass_stdio_redirect') == 1

if __name__ == '__main__':
    main()
