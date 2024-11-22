import pytest
import ipd
from ipd.lazy_import import lazyimport

th = lazyimport('torch')

pytest.importorskip('ipd.voxel.voxel_cuda')

def main():
    test_cudafunc_clash()
    test_cudafunc_contact_10()
    test_cudafunc_contact()
    test_cudafunc_clash_on_gpu()
    test_cudafunc_contact_on_gpu()
    print('test_cuda DONE')

@pytest.mark.fast
def test_cudafunc_clash():
    func = ipd.dev.cuda.ClashFunc(3, 4)
    assert func.reference_impl(0) == 1
    assert func.reference_impl(3) == 1
    assert func.reference_impl(3.5) == 0.5
    assert func.reference_impl(4) == 0
    assert func.reference_impl(10) == 0

@pytest.mark.fast
def test_cudafunc_clash_on_gpu():
    func = ipd.dev.cuda.ClashFunc(3, 4)
    dist = th.arange(0, 10.001, 0.5).to('cuda').to(th.float32)
    ref = th.tensor([func.reference_impl(x) for x in dist])
    tst = func(dist)
    assert th.allclose(tst.cpu(), ref)

@pytest.mark.fast
def test_cudafunc_contact():
    func = ipd.dev.cuda.ContactFunc()
    # for f in th.arange(0,10.01,0.25):
    # print(f'{f}) == {func.reference_impl(f)}')
    assert func.reference_impl(0.0) == 10000.0
    assert func.reference_impl(3.0) == 10000.0
    assert func.reference_impl(3.25) == 7499.75
    assert func.reference_impl(3.5) == 4999.5
    assert func.reference_impl(3.75) == 2499.25
    assert func.reference_impl(4.0) == -1.0
    assert func.reference_impl(8.0) == -1.0
    assert func.reference_impl(8.25) == -0.75
    assert func.reference_impl(8.5) == -0.5
    assert func.reference_impl(8.75) == -0.25
    assert func.reference_impl(9.0) == 0
    assert func.reference_impl(10.0) == 0

@pytest.mark.fast
def test_cudafunc_contact_10():
    func = ipd.dev.cuda.ContactFunc(clashscore=10, contactscore=-1, clashend=3, contactbeg=4, contactend=8, end=9)
    assert func.reference_impl(0.00) == 10.00
    assert func.reference_impl(3.00) == 10.00
    assert func.reference_impl(3.25) == 7.25
    assert func.reference_impl(3.50) == 4.50
    assert func.reference_impl(3.75) == 1.75
    assert func.reference_impl(4.00) == -1.00
    assert func.reference_impl(7.75) == -1.00
    assert func.reference_impl(8.00) == -1.00
    assert func.reference_impl(8.25) == -0.75
    assert func.reference_impl(8.50) == -0.50
    assert func.reference_impl(8.75) == -0.25
    assert func.reference_impl(9.00) == 0.00
    assert func.reference_impl(9.25) == 0.00

@pytest.mark.fast
def test_cudafunc_contact_on_gpu():
    func = ipd.dev.cuda.ContactFunc()
    dist = th.arange(0, 10.001, 0.5).to('cuda').to(th.float32)
    ref = th.tensor([func.reference_impl(x) for x in dist])
    tst = func(dist)
    # ic(dist)
    # ic(tst)
    assert th.allclose(tst.cpu(), ref)

if __name__ == '__main__':
    main()
