import numpy as np

import ipd
from ipd.sym.diffusion_sym import *  # type: ignore

def main():
    test_symmatrix12()
    test_symmatrix24()
    test_symmatrix60()

@pytest.mark.fast  # type: ignore
def test_symmatrix12():
    symmatrix = ipd.sym.symframes.tet_symmatrix
    frames = ipd.homog.hconvert(ipd.sym.symframes.tet_Rs)
    assert ipd.homog.hvalid(frames)
    first = frames[symmatrix[0]] @ ipd.homog.hinv(frames[0])
    for i in range(len(symmatrix)):
        x = frames[symmatrix[i]] @ ipd.homog.hinv(frames[i])
        assert np.allclose(first, x)

@pytest.mark.fast  # type: ignore
def test_symmatrix24():
    symmatrix = ipd.sym.symframes.oct_symmatrix
    frames = ipd.homog.hconvert(ipd.sym.symframes.oct_Rs)
    assert ipd.homog.hvalid(frames)
    first = frames[symmatrix[0]] @ ipd.homog.hinv(frames[0])
    for i in range(len(symmatrix)):
        x = frames[symmatrix[i]] @ ipd.homog.hinv(frames[i])
        assert np.allclose(first, x)

@pytest.mark.fast  # type: ignore
def test_symmatrix60():
    symmatrix = ipd.sym.symframes.icos_symmatrix
    frames = ipd.homog.hconvert(ipd.sym.symframes.icos_Rs)
    assert ipd.homog.hvalid(frames)
    first = frames[symmatrix[0]] @ ipd.homog.hinv(frames[0])
    for i in range(len(symmatrix)):
        x = frames[symmatrix[i]] @ ipd.homog.hinv(frames[i])
        assert np.allclose(first, x)

if __name__ == "__main__":
    main()
