import ipd

def main():
    ipd.tests.maintest(namespace=globals())

def test_subscripable_for_attributes():

    @ipd.dev.subscripable_for_attributes
    class foo:
        a, b, c = 6, 7, 8

    assert foo()['a'] == 6
    assert foo()['a b'] == (6, 7)

if __name__ == '__main__':
    main()
