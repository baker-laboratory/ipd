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
    ic.configureOutput(includeContext=True)
    with ipd.dev.capture_stdio() as printed:
        ipd.ic(7)
        with ipd.dev.ic_config(includeContext=False):
            ipd.ic(8)
        ipd.ic(9)
    result = printed.read()
    print(result)
    assert result.count('ic|') == 3
    assert result.count('test_debug.py') == 2

if __name__ == '__main__':
    main()
