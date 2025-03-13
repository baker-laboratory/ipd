from typing import TypeVar

import numpy as np

import ipd
from ipd import basic_typevars, Frames44, FramesN44

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

def test_basic_typevars():
    # Test T and R type variables
    typevars = list(basic_typevars(['T', 'R']))
    assert len(typevars) == 2
    assert type(typevars[0]).__name__ == 'TypeVar'
    assert type(typevars[1]).__name__ == 'TypeVar'

    # Test class type variable
    typevars = list(basic_typevars(['C']))
    assert len(typevars) == 1
    C = TypeVar('C')
    C = type[C]
    assert str(typevars[0]) == str(C)

    # Test callable type variable
    typevars = list(basic_typevars(['F']))
    assert len(typevars) == 1
    assert callable(typevars[0])

def test_frames44_instancecheck():
    # Valid 4x4 matrix
    x = np.zeros((4, 4))
    assert isinstance(x, Frames44)

    # Invalid shape
    y = np.zeros((3, 4))
    assert not isinstance(y, Frames44)

    # Invalid dimensions
    z = np.zeros((5, 5))
    assert not isinstance(z, Frames44)

def test_framesn44_instancecheck():
    # Valid batch of 4x4 matrices
    x = np.zeros((10, 4, 4))
    assert isinstance(x, FramesN44)

    # Single 4x4 matrix (not batch)
    y = np.zeros((4, 4))
    assert not isinstance(y, FramesN44)

    # Incorrect shape for batch
    z = np.zeros((10, 3, 4))
    assert not isinstance(z, FramesN44)

    # Incorrect dimensions
    w = np.zeros((1, 4, 4, 4))
    assert not isinstance(w, FramesN44)

if __name__ == '__main__':
    main()
