import numpy as np
import pytest

import ipd
from ipd.observer.dynamic_parameters import _NotIn

def main():
    test_dynparams_constant()
    test_dynparams_spline2D()
    test_dynparams_spline1D()
    test_dynparam_bool_in_range()
    test_dynparam_bool_on_steps()
    test_dynparam_steps()
    test_dynparam_parse()
    test_dynparam_askwargs()
    print('test_dynparam PASS')

@pytest.mark.fast
def test_dynparams_constant():
    dynp = ipd.dev.DynamicParameters(ndesign=1, ndiffuse=50, nrfold=40, _testing=True)
    SS = dynp._set_step
    dynp.newparam_constant('foo', 'FOO')
    assert SS(diffuse=0, rfold=39).foo == 'FOO'
    dynp.newparam_constant('fortytwo', 42)
    assert SS(diffuse=0, rfold=39).fortytwo == 42

@pytest.mark.fast
def test_dynparams_spline2D():
    dynp = ipd.dev.DynamicParameters(ndesign=1, ndiffuse=50, nrfold=40, _testing=True)
    SS = dynp._set_step
    dynp.newparam_spline_2d('twod', diffuse_rfold=[
        (0.0, 0.0, 0),
        (0.0, 1.0, 50),
        (1.0, 0.0, 50),
        (1.0, 1.0, 100),
    ])
    assert np.allclose(SS(diffuse=0, rfold=0).twod, 0.0)  # type: ignore
    assert np.allclose(SS(diffuse=0, rfold=39).twod, 50.0)  # type: ignore
    assert np.allclose(SS(diffuse=49, rfold=0).twod, 50.0)  # type: ignore
    assert np.allclose(SS(diffuse=49, rfold=39).twod, 100.0)  # type: ignore
    assert np.allclose(SS(diffuse=7, rfold=9).twod, 18.68132)  # type: ignore
    assert np.allclose(SS(diffuse=13, rfold=9).twod, 24.80377)  # type: ignore
    assert np.allclose(SS(diffuse=45, rfold=7).twod, 54.89272)  # type: ignore
    assert np.allclose(SS(diffuse=28, rfold=30).twod, 67.03297)  # type: ignore
    assert np.allclose(SS(diffuse=43, rfold=37).twod, 91.313446)  # type: ignore
    with pytest.raises(ValueError):
        dynp.newparam_spline_2d('linear1', diffuse_rfold=[(0, 0, 1)])
    with pytest.raises(ValueError):
        dynp.newparam_spline_2d('linear1', diffuse_rfold=[(0, 0, 1), (1, 1, 0)])
    with pytest.raises(ValueError):
        dynp.newparam_spline_2d('linear1', diffuse_rfold=[(0, 0, 1), (1, 1, 0), (1, 0, 1)])

@pytest.mark.fast
def test_dynparams_spline1D():
    dynp = ipd.dev.DynamicParameters(ndesign=9, ndiffuse=9, nrfold=9, _testing=True)
    SS = dynp._set_step

    dynp.newparam_spline_1d('linear1', diffuse=[(0, 1), (1, 9)])
    for i in range(9):
        assert np.allclose(SS(diffuse=i).linear1, i + 1)  # type: ignore
    dynp.newparam_spline_1d('square', rfold=[(0, 0), (0.5, 16), (1, 64)])
    for i in range(9):
        assert np.allclose(SS(rfold=i).square, i**2)  # type: ignore
    dynp.newparam_spline_1d('cube', design=[(0, 0), (0.25, 2**3), (0.5, 4**3), (1, 8**3)])
    for i in range(9):
        assert np.allclose(SS(design=i).cube, i**3)  # type: ignore
    with pytest.raises(ValueError):
        dynp.newparam_spline_1d('test', diffuse=7, rfold=9)
    with pytest.raises(ValueError):
        dynp.newparam_spline_1d('test')
    with pytest.raises(ValueError):
        dynp.newparam_spline_1d('test', diffuse=[(-1, 0), (1, 0)])

@pytest.mark.fast
def test_dynparam_bool_in_range():
    dynp = ipd.dev.DynamicParameters(ndesign=5, ndiffuse=7, nrfold=11, _testing=True)
    SS = dynp._set_step

    #
    dynp.newparam_true_in_range('range1', diffuse=(0.5, 0.5))
    # ic(dynp.params['range1'].diffuse_steps)
    assert SS(diffuse=0).range1 is False
    assert SS(diffuse=1).range1 is False
    assert SS(diffuse=2).range1 is False
    assert SS(diffuse=3).range1 is True
    assert SS(diffuse=4).range1 is False
    assert SS(diffuse=5).range1 is False
    assert SS(diffuse=6).range1 is False

    dynp.newparam_true_in_range('range2', rfold=[(0, 2), (4, 5)])
    # ic(dynp.params['range2'].diffuse_steps)
    assert SS(rfold=0).range2 is True
    assert SS(rfold=1).range2 is True
    assert SS(rfold=2).range2 is True
    assert SS(rfold=3).range2 is False
    assert SS(rfold=4).range2 is True
    assert SS(rfold=5).range2 is True
    assert SS(rfold=6).range2 is False

    dynp.newparam_false_in_range('range3', rfold=[(0, 2), (4, 5)])
    # ic(dynp.params['range3'].diffuse_steps)
    assert SS(rfold=0).range3 is not True
    assert SS(rfold=1).range3 is not True
    assert SS(rfold=2).range3 is not True
    assert SS(rfold=3).range3 is not False
    assert SS(rfold=4).range3 is not True
    assert SS(rfold=5).range3 is not True
    assert SS(rfold=6).range3 is not False

@pytest.mark.fast
def test_dynparam_bool_on_steps():
    dynp = ipd.dev.DynamicParameters(ndesign=5, ndiffuse=7, nrfold=11, _testing=True)
    SS = dynp._set_step

    dynp.newparam_true_on_steps('toi', diffuse=[1, 3])

    assert dynp.toi is False
    assert SS(diffuse=1).toi is True
    assert SS(diffuse=3).toi is True
    assert SS(diffuse=0).toi is False
    assert SS(diffuse=2).toi is False
    assert SS(diffuse=4).toi is False
    assert SS(diffuse=5).toi is False
    assert SS(diffuse=6).toi is False

    dynp.newparam_false_on_steps('foi', design=1, rfold=[1, 3])
    for i in range(7):
        for j in [0, 2]:
            assert SS(design=j, rfold=i).foi
    SS(design=1)
    assert not SS(rfold=1).foi
    assert not SS(rfold=3).foi
    assert SS(rfold=0).foi
    assert SS(rfold=2).foi
    assert SS(rfold=4).foi
    assert SS(rfold=5).foi
    assert SS(rfold=6).foi

    dynp.newparam_true_on_steps('foinv', design=1, rfold=_NotIn(1, 3))
    for i in range(7):
        for j in [0, 2]:
            assert SS(design=j, rfold=i).foi
    SS(design=1)
    assert not SS(rfold=1).foinv
    assert not SS(rfold=3).foinv
    assert SS(rfold=0).foinv
    assert SS(rfold=2).foinv
    assert SS(rfold=4).foinv
    assert SS(rfold=5).foinv
    assert SS(rfold=6).foinv

    with pytest.raises(ValueError):
        dynp.newparam_true_on_steps('toi', diffuse=[-1, -3])

    dynp.newparam_true_on_steps('neg', diffuse=[-1, -3])
    # ic(dynp.params['neg'].diffuse_steps)
    assert not SS(diffuse=0).neg
    assert not SS(diffuse=1).neg
    assert not SS(diffuse=2).neg
    assert not SS(diffuse=3).neg
    assert SS(diffuse=4).neg
    assert not SS(diffuse=5).neg
    assert SS(diffuse=6).neg

    dynp.newparam_true_on_steps('float1', diffuse=[0.33])
    # ic(dynp.params['float1'].diffuse_steps)
    assert SS(diffuse=0).float1 is False
    assert SS(diffuse=1).float1 is False
    assert SS(diffuse=2).float1 is True
    assert SS(diffuse=3).float1 is False
    assert SS(diffuse=4).float1 is False
    assert SS(diffuse=5).float1 is False
    assert SS(diffuse=6).float1 is False

    dynp.newparam_true_on_steps('float2', diffuse=[0.01, -0.33])
    # ic(dynp.params['float2'].diffuse_steps)
    assert SS(diffuse=0).float2 is True
    assert SS(diffuse=1).float2 is False
    assert SS(diffuse=2).float2 is False
    assert SS(diffuse=3).float2 is False
    assert SS(diffuse=4).float2 is True
    assert SS(diffuse=5).float2 is False
    assert SS(diffuse=6).float2 is False

@pytest.mark.fast
def test_dynparam_steps():
    dynp = ipd.dev.DynamicParameters(ndesign=12, ndiffuse=50, nrfold=40, _testing=True)
    dynp._set_step(design=7)
    assert dynp._step == (7, None, None)
    dynp._set_step(diffuse=13)
    assert dynp._step == (7, 13, None)
    dynp._set_step(diffuse=-12)
    assert dynp._step == (7, 38, None)
    dynp._set_step(design=-1)
    assert dynp._step == (11, 38, None)

    dynp._rfold_iter_begin('foobar')
    assert dynp._step == (11, 38, 0)
    dynp._rfold_iter_begin('foobar2')
    assert dynp._step == (11, 38, 1)

    # with pytest.raises(AssertionError):
    # dynp._rfold_iter_begin('foobar2')

@pytest.mark.fast
def test_dynparam_parse():
    dp = ipd.dev.DynamicParameters(ndesign=1, ndiffuse=10, nrfold=10, _testing=True)
    with pytest.raises(SyntaxError):
        dp.parse_dynamic_param('foo', 'rfold:[(2 4)]')
    with pytest.raises(ValueError):
        dp.parse_dynamic_param('foo', 'foo:[(2,4)]')

    dp.parse_dynamic_param('foo', 'rfold:[(2,4)]')
    with pytest.raises(ValueError):
        dp.parse_dynamic_param('foo', 'rfold:[(2,4)]')
    assert not dp.foo
    assert dp._set_step(rfold=1).foo is False
    assert dp._set_step(rfold=2).foo is True
    assert dp._set_step(rfold=3).foo is True
    assert dp._set_step(rfold=4).foo is True
    assert dp._set_step(rfold=5).foo is False

    dp.parse_dynamic_param('bar', 'rfold:[(0.5,1)]')
    assert dp._set_step(rfold=3).bar is False
    assert dp._set_step(rfold=7).bar is True
    assert dp._set_step(rfold=9).bar is True

    dp.parse_dynamic_param('baz', 'rfold:[(-4,-1)]*diffuse:[(0,0.5)]')
    assert dp._set_step(rfold=1, diffuse=2).baz is False
    assert dp._set_step(rfold=5, diffuse=2).baz is False
    assert dp._set_step(rfold=6, diffuse=2).baz is True
    assert dp._set_step(rfold=8, diffuse=2).baz is True
    assert dp._set_step(rfold=9, diffuse=2).baz is True
    assert dp._set_step(rfold=1, diffuse=7).baz is False
    assert dp._set_step(rfold=5, diffuse=7).baz is False
    assert dp._set_step(rfold=6, diffuse=7).baz is False
    assert dp._set_step(rfold=8, diffuse=7).baz is False
    assert dp._set_step(rfold=9, diffuse=7).baz is False

@pytest.mark.fast
def test_dynparam_askwargs():
    def foo(a, b, c):
        return int(a) + int(b) + int(c)

    dp = ipd.dev.DynamicParameters(ndesign=1, ndiffuse=10, nrfold=10, _testing=True)
    dp.newparam_true_on_steps('a', rfold=0, diffuse=0)
    dp.newparam_true_on_steps('b', rfold=0, diffuse=0)
    dp.newparam_true_on_steps('c', rfold=1, diffuse=1)
    dp._set_step(diffuse=0, rfold=0)
    assert foo(**dp) == 2
    dp._set_step(diffuse=1, rfold=1)
    assert foo(**dp) == 1
    dp._set_step(diffuse=0, rfold=1)
    assert foo(**dp) == 0

if __name__ == '__main__':
    main()
