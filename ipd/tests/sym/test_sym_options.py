import pytest
from icecream import ic

import ipd

@pytest.mark.fast
def test_sym_options():
    opt = ipd.sym.get_sym_options()
    ic(opt)

if __name__ == '__main__':
    test_sym_options()
