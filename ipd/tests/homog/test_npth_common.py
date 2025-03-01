import ipd

th, xr, np = ipd.importornone('torch xarray numpy')

def main():
    ipd.tests.maintest(namespace=globals())

def test_is_tensor():
    assert not ipd.homog.is_tensor(1)
    if np: assert ipd.homog.is_tensor(np.eye(4))
    if th: assert ipd.homog.is_tensor(th.eye(4))
    if xr: assert ipd.homog.is_tensor(xr.DataArray([1, 2, 3]))

def test_is_xform_stack():
    assert not ipd.homog.is_xform_stack(6)
    assert not ipd.homog.is_xform_stack(np.eye(4))
    assert ipd.homog.is_xform_stack(np.eye(4)[None])

if __name__ == '__main__':
    main()
