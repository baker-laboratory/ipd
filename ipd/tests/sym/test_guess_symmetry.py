import pytest

import ipd

def main():
    test_guess_from_fnames()

@pytest.mark.fast
def test_guess_from_fnames():
    assert 'C3' == ipd.sym.guess_sym_from_fnames(['foo_c3_bar.pdb', 'c3.pdb'])
    assert None is ipd.sym.guess_sym_from_fnames(['foo_c3_bar.pdb', 'c3.pdb', 'icos.pdb'])

if __name__ == '__main__':
    main()
