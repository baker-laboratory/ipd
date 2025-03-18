import pytest
import numpy as np
import ipd

def main():
    ipd.tests.maintest(namespace=globals())

bunch = ipd.Bunch(
    dot_norm=ipd.Bunch(frac=0.174, tol=0.04, total=282, passes=49),
    isect=ipd.Bunch(frac=0.149, tol=1.0, total=302, passes=45),
    angle=ipd.Bunch(frac=0.571, tol=0.09, total=42, passes=24),
    helical_shift=ipd.Bunch(frac=1.0, tol=1.0, total=47, passes=47),
    axistol=ipd.Bunch(frac=0.412, tol=0.1, total=17, passes=7),
    nfold=ipd.Bunch(frac=1.0, tol=0.2, total=5, passes=5),
    cageang=ipd.Bunch(frac=0.5, tol=0.1, total=2, passes=1),
)

def test_make_table_dict_of_dict():
    ipd.dev.print_table(bunch)

def test_summary_numpy():
    assert ipd.dev.summary(np.arange(3)) == "[0 1 2]"
    assert ipd.dev.summary(np.arange(300)) == "ndarray[300]"

def test_summary_atomarray():
    pytest.importorskip('biotite')
    atoms = ipd.atom.get('1qys')
    assert ipd.dev.summary(atoms) == "AtomArray(692)"

# def test_print_table_bunch():
# ipd.dev.print_table_bunch(bunch)

if __name__ == '__main__':
    main()
