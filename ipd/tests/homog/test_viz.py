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

def test_can_showme():
    assert not ipd.homog.can_showme(None)
    assert not ipd.homog.can_showme(ipd.homog.hrand(10))

if __name__ == '__main__':
    main()
