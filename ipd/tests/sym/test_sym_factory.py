import pytest

import ipd

th = pytest.importorskip('torch')

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

def test_create_test_sym_manager():
    assert ipd.sym.create_sym_manager().symid == 'C1'
    assert ipd.sym.create_sym_manager(symid='c3').symid == 'C3'

if __name__ == '__main__':
    main()
