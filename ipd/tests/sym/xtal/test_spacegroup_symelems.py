import itertools

import pytest

import ipd
from ipd.sym.xtal.spacegroup_symelems import (
    _compute_symelems,
    _find_compound_symelems,
    _printelems,
    _remove_redundant_screws,
)
from ipd.sym.xtal.SymElem import SymElem, showsymelems

pytest.skip(allow_module_level=True)

def main():
    # test_symelems_R3()rpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdb

    # test_symelems_P622()
    # assert 0

    # test_symelems_P622()
    # assert 0

    # for l in 'C21 C31 C32 C41 C42 C43 C61 C62 C63 C64 C65'.split():
    #    print('\n', l)
    #    for k, v in ipd.sym.xtal.sg_symelem_dict.items():
    #       # if not any(x.iscyclic for x in v):
    #       # if not ipd.sym.xtal.latticetype(k) == 'MONOCLINIC': continue
    #       # if not any(x.isdihedral for x in v): continue
    #       if not any(x.label == l for x in v): continue
    #       print(k, end=' ')
    #       # for e in v:
    #       # print(e)
    # print(flush=True)
    # assert 0

    # T
    # P23 F23 I23 P4232 F432 F4132
    # O
    # P432 F432 I432

    sym = "P3221"
    for e in ipd.sym.xtal.symelems(sym):
        print(e, flush=True)
    # assert 0
    showsymelems(
        sym,
        ipd.sym.xtal.symelems(sym),
        scan=20,
        weight=8,
        offset=0.0,  # type: ignore
        # lattice=ipd.sym.xtal.lattice_vectors(sym, [1.8, 1.6, 1.7, 70, 70, 110], strict=False),
        lattice=ipd.sym.xtal.lattice_vectors(sym, [1, 1.6, 1.7, 70, 70, 110], strict=False),
    )
    assert 0

    # test_symelems_P312()

    # test_symelems_P1()
    # test_symelems_P41()
    # test_symelems_P43()

    # test_compound_elems_R32()
    # test_compound_elems_P222()
    # test_compound_elems_F23(1)
    # test_compound_elems_P23()
    # test_compound_elems_I23()
    # test_compound_elems_P213()
    # test_compound_elems_P4132()
    # test_compound_elems_I4132()
    # test_compound_elems_P432()
    # test_compound_elems_I432()
    # test_compound_elems_F432()
    # test_compound_elems_F4132()

    # test_compound_elems_P3121()
    # test_compound_elems_P212121()
    # test_compound_elems_P31()
    # test_compound_elems_P32()
    # test_compound_elems_P213()
    # test_compound_elems_P3221()
    # test_compound_elems_P41()
    # test_compound_elems_P41212()
    # test_compound_elems_P4232()
    # test_compound_elems_P43()
    # test_compound_elems_P43212()
    # test_compound_elems_P4332()
    # test_compound_elems_P6()
    # test_compound_elems_P61()
    # test_compound_elems_P6122()
    # test_compound_elems_P62()
    # test_compound_elems_P63()
    # test_compound_elems_P64()
    # test_compound_elems_P65()
    # test_compound_elems_P6522()
    # test_compound_elems_I213()rpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdb
    # test_compound_elems_I23()rpxdock_P3221_Crpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdb2_c21004657999819pmsave_0006_asym.pdb
    # test_compound_elems_I4()rpxdock_P3221_C2_c21004657999819pmsarpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdbrpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdbvrpxdock_P3221_C2_c21004657999819pmrpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdbsrpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdbarpxdock_P3221_C2_c21004657999819pmsrpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdbave_0006_asym.pdbrpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdbrpxdock_P3221_C2_c21004657999819pmsrpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdbave_0006_asym.pdbrpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdbrpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdbrpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdbrpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdbve_0006_asym.pdbrpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdbrpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdbrpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdbe_0006_asym.pdb
    # test_compound_elems_I41()

    assert 0

    # test_symelems_P1211()rpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdb
    # test_symelems_P2221()
    # test_symelems_P21212()
    # test_symelems_P1()
    # test_symelems_C121()
    # test_symelems_P3()
    # test_symelems_P222()rpxdock_P3221_C2_c21004657999819pmsave_0006_asym.pdb
    # test_symelems_P23()
    # test_symelems_F23()
    # test_symelems_R32()

    # test_symelems_P3121()
    # test_symelems_P212121()
    # test_symelems_P31()
    # test_symelems_P32()
    # test_symelems_P213()
    # test_symelems_P3221()
    # test_symelems_P41()
    # test_symelems_P41212()
    # test_symelems_P4132()
    # test_symelems_P4232()
    # test_symelems_P43()
    # test_symelems_P432()
    # test_symelems_P43212()
    # test_symelems_P4332()
    # test_symelems_P6()
    # test_symelems_P61()
    # test_symelems_P6122()
    # test_symelems_P62()
    # test_symelems_P63()
    # test_symelems_P64()
    # test_symelems_P65()
    # test_symelems_P6522()
    # test_symelems_I213()
    # test_symelems_I23()
    # test_symelems_I4()
    # test_symelems_I41()
    # test_symelems_I4132()
    # test_symelems_I432()
    # test_symelems_F4132()
    # test_symelems_F432()

    # test_remove_redundant_screws()

    ic("PASS test_spacegroup_symelems")  # type: ignore

# yapf: disable


@pytest.mark.fast
def test_symelems_P622(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[1.0, -0.57735, -0.0], cen=[0.0, 0.0, 0.0], label='C2'),
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], label='C2'),
          SymElem(2, axis=[1.0, -0.57735, -0.0], cen=[0.0, 0.0, 0.5], label='C2'),
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.5], label='C2'),
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[0, 0, 1], cen=[-0.3333333333423178, 0.3333333333333333, 0.0], label='C3'),
       ],
       C6=[
          SymElem(6, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C6'),
       ],
       C21=[
          SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.5], hel=0.5, label='C21'),  # type: ignore
       ],
    )
    helper_test_symelem('P622', val, debug, **kw)


@pytest.mark.xfail
def test_symelems_P312(debug=False, **kw):
    val = dict(   )
    helper_test_symelem('P312', val, debug, **kw)



@pytest.mark.fast
def test_compound_elems_R32(debug=False, **kw):
    sym = 'R32'
    val = dict(
       D3=[
          SymElem(3, axis=[0, 0, 1], axis2=[1.0, 1.0, 0.0], cen=[0.0, 0.0, 0.0], label='D3'),
          SymElem(3, axis=[0, 0, 1], axis2=[0.0, 1.0, 0.0], cen=[0.0, 0.0, 0.5], label='D3'),
       ],
    )
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P222(debug=False, **kw):
    sym = 'P222'
    val = dict(
       D2=[
          SymElem(2, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.0, 0.0, 0.0], label='D2'),
          SymElem(2, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.0, 0.0, 0.5], label='D2'),
          SymElem(2, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.0, 0.5, 0.0], label='D2'),
          SymElem(2, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.0, 0.5, 0.5], label='D2'),
          SymElem(2, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.0], label='D2'),
          SymElem(2, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.5], label='D2'),
          SymElem(2, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.5, 0.5, 0.0], label='D2'),
          SymElem(2, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.5, 0.5, 0.5], label='D2'),
       ],
    )
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P3121(debug=False, **kw):
    sym = 'P3121'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P212121(debug=False, **kw):
    sym = 'P212121'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P31(debug=False, **kw):
    sym = 'P31'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P32(debug=False, **kw):
    sym = 'P32'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P213(debug=False, **kw):
    sym = 'P213'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P3221(debug=False, **kw):
    sym = 'P3221'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P41(debug=False, **kw):
    sym = 'P41'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P41212(debug=False, **kw):
    sym = 'P41212'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P4232(debug=False, **kw):
    sym = 'P4232'
    val = dict(
       T=[
          SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0], label='T'),
       ],
       D3=[
          SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, -0.0], cen=[0.25, 0.25, 0.25], label='D3'),
          SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, -0.0], cen=[0.75, 0.75, 0.75], label='D3'),
       ],
       D2=[
          SymElem(2, axis=[1, 0, 0], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.5], label='D2'),
          SymElem(2, axis=[0, 1, 1], axis2=[-0.0, -1.0, 1.0], cen=[0.25, 0.0, 0.5], label='D2'),
          SymElem(2, axis=[1, 0, 1], axis2=[0.0, 1.0, 0.0], cen=[0.0, 0.25, 0.5], label='D2'),
       ],
    )
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P43(debug=False, **kw):
    sym = 'P43'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)


@pytest.mark.fast
def test_compound_elems_P4332(debug=False, **kw):
    sym = 'P4332'
    val = dict(
       D3=[
          SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, -0.0], cen=[0.125, 0.125, 0.125], label='D3'),
          SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, -0.0], cen=[0.625, 0.625, 0.625], label='D3'),
       ],
    )
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P6(debug=False, **kw):
    sym = 'P6'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P61(debug=False, **kw):
    sym = 'P61'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P6122(debug=False, **kw):
    sym = 'P6122'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P62(debug=False, **kw):
    sym = 'P62'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P63(debug=False, **kw):
    sym = 'P63'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P64(debug=False, **kw):
    sym = 'P64'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P65(debug=False, **kw):
    sym = 'P65'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P6522(debug=False, **kw):
    sym = 'P6522'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_I213(debug=False, **kw):
    sym = 'I213'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)


@pytest.mark.fast
def test_compound_elems_I4(debug=False, **kw):
    sym = 'I4'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_I41(debug=False, **kw):
    sym = 'I41'
    val = dict()
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P432(debug=False, **kw):
    sym = 'P432'
    val = dict(
       O=[
          SymElem('O43', axis=[0, 0, 1], axis2=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0], label='O'),
          SymElem('O43', axis=[1, 0, 0], axis2=[1.0, 1.0, 1.0], cen=[0.5, 0.5, 0.5], label='O'),
       ],
       D4=[
          SymElem(4, axis=[0, 0, 1], axis2=[1.0, 0.0, 0.0], cen=[0.0, 0.0, 0.5], label='D4'),
          SymElem(4, axis=[1, 0, 0], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.5, 0.5], label='D4'),
       ],
    )
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_I432(debug=False, **kw):
    sym = 'I432'
    val = dict(
       O=[
          SymElem('O43', axis=[0, 0, 1], axis2=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0], label='O'),
       ],
       D4=[
          SymElem(4, axis=[0, 0, 1], axis2=[1.0, 0.0, 0.0], cen=[0.0, 0.0, 0.5], label='D4'),
       ],
       D3=[
          SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, -0.0], cen=[0.25, 0.25, 0.25], label='D3'),
       ],
       D2=[
          SymElem(2, axis=[1, 0, 0], axis2=[-0.0, -1.0, 1.0], cen=[0.25, 0.0, 0.5], label='D2'),
       ],
    )
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_F432(debug=False, **kw):
    sym = 'F432'
    val = dict(
       O=[
          SymElem('O43', axis=[0, 0, 1], axis2=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0], label='O'),
          SymElem('O43', axis=[1, 0, 0], axis2=[1.0, 1.0, 1.0], cen=[0.5, 0.5, 0.5], label='O'),
       ],
       T=[
          SymElem('T32', axis=[1, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.25, 0.25, 0.25], label='T'),
       ],
       D2=[
          SymElem(2, axis=[1, 0, 0], axis2=[-0.0, -1.0, 1.0], cen=[0.0, 0.25, 0.25], label='D2'),
       ],
    )
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_F4132(debug=False, **kw):
    sym = 'F4132'
    val = dict(
       T=[
          SymElem('T32', axis=[1, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.0, 0.0, 0.0], label='T'),
          SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 1.0, 0.0], cen=[0.5, 0.5, 0.5], label='T'),
       ],
       D3=[
          SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, -0.0], cen=[0.125, 0.125, 0.125], label='D3'),
          SymElem(3, axis=[1, 1, 1], axis2=[-1.0, -0.0, 1.0], cen=[0.625, 0.625, 0.625], label='D3'),
       ],
    )
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P23(debug=False, **kw):
    sym = 'P23'
    val = dict(
       T=[
          SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0], label='T'),
          SymElem('T32', axis=[1, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.5, 0.5, 0.5], label='T'),
       ],
       D2=[
          SymElem(2, axis=[1, 0, 0], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.5], label='D2'),
          SymElem(2, axis=[0, 1, 0], axis2=[1.0, 0.0, 0.0], cen=[0.0, 0.5, 0.5], label='D2'),
       ],
    )
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_I23(debug=False, **kw):
    sym = 'I23'
    val = dict(
       T=[
          SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0], label='T'),
       ],
       D2=[
          SymElem(2, axis=[1, 0, 0], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.5], label='D2'),
       ],
    )
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_F23(debug=False, **kw):
    sym = 'F23'
    val = dict(T=[
       SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0], label='T'),
       SymElem('T32', axis=[1, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.5, 0.5, 0.5], label='T'),
       SymElem('T32', axis=[1, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.25, 0.25, 0.25], label='T'),
       SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 1.0, 0.0], cen=[0.75, 0.75, 0.75], label='T'),
    ], )
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_P4132(debug=False, **kw):
    sym = 'P4132'
    val = dict(D3=[
       SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, -0.0], cen=[0.375, 0.375, 0.375], label='D3'),
       SymElem(3, axis=[1, 1, 1], axis2=[-0.0, -1.0, 1.0], cen=[0.875, 0.875, 0.875], label='D3'),
    ], )
    helper_test_symelem(sym, val, debug, compound=True, **kw)

@pytest.mark.fast
def test_compound_elems_I4132(debug=False, **kw):
    sym = 'I4132'
    val = dict(
       D3=[
          SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, -0.0], cen=[0.125, 0.125, 0.125], label='D3'),
          SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, -0.0], cen=[0.375, 0.375, 0.375], label='D3'),
       ],
       D2=[
          SymElem(2, axis=[0, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.125, 0.0, 0.25], label='D2'),
          SymElem(2, axis=[1, 0, 1], axis2=[-1.0, -0.0, 1.0], cen=[0.25, 0.375, 0.5], label='D2'),
       ],
    )
    helper_test_symelem(sym, val, debug, compound=True, **kw)

# def test_compound_elems_P4132(showme=False):
#    ic('test_compound_elems_P4132')
#    sym = 'P4132'
#    elems = ipd.sym.xtal.symelems(sym, asdict=True)
#    celems = _find_compound_symelems(sym)
#    if showme: showsymelems(sym, elems)
#    if showme: showsymelems(sym, celems)
#    # print(repr(celems), flush=True)
#    assert set(elems.keys()) == set('D3'.split())
#    assert celems['D3'] == [SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, 0.0], cen=[0.375, 0.375, 0.375], label='D3')]

# def test_compound_elems_F4132(showme=False):
#    ic('test_compound_elems_F4132')
#    sym = 'F4132'
#    elems = ipd.sym.xtal.symelems(sym, asdict=True)
#    celems = _find_compound_symelems(sym)

#    # for k, v in celems.items():
#    #    print(k)
#    #    for x in v:
#    #       print(x, flush=True)

#    if showme: showsymelems(sym, elems)
#    if showme: showsymelems(sym, celems)
#    assert set(elems.keys()) == set('T D3'.split())
#    print(repr(celems), flush=True)
#    assert elems['T'] == [SymElem('T32', axis=[1, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.0, 0.0, 0.0], label='T')]
#    assert elems['D3'] == [SymElem(3, axis=[1, 1, 1], axis2=[0.0, -1.0, 1.0], cen=[0.125, 0.125, 0.125], label='D3')]

# def test_compound_elems_P213(showme=False):
#    ic('test_compound_elems_P213')
#    sym = 'P213'
#    elems = _find_compound_symelems(sym)

#    # for k, v in celems.items():
#    #    print(k)
#    #    for x in v:
#    #       print(x, flush=True)

#    # if showme: showsymelems(sym, elems)
#    if showme: showsymelems(sym, celems)

#    assert elems == {}

def helper_test_symelem(sym, eref=None, debug=False, compound=False, **kw):
    if compound:
        otherelems = ipd.sym.xtal.symelems(sym, asdict=True)
        # otherelems = {}
        symelems = list(itertools.chain(*otherelems.values()))
        elems0 = _find_compound_symelems(sym, symelems)
    else:
        otherelems = {}
        elems0 = _compute_symelems(sym, profile=debug)

    etst = elems0.copy()
    eref = eref.copy()  # type: ignore
    if 'C11' in etst: del etst['C11']  # type: ignore
    if 'C11' in eref: del eref['C11']

    ok = True
    if eref is not None:
        vkey = set(eref.keys())
        tkey = set(etst.keys())  # type: ignore
        key = sorted(vkey.intersection(tkey))
        for k in vkey - tkey:
            ok = False
            print(sym, 'MISSING', k)
        for k in tkey - vkey:
            ok = False
            print(sym, 'EXTRA', k)
        for k in key:
            tval = ipd.dev.UnhashableSet(etst[k])
            vval = ipd.dev.UnhashableSet(eref[k])
            x = vval.difference(tval)
            if x:
                ok = False
                print(sym, k, 'MISSING')
                for v in x:
                    print('  ', v)
            x = tval.difference(vval)
            if x:
                ok = False
                print(sym, k, 'EXTRA')
                for v in x:
                    print('  ', v)
            x = tval.intersection(vval)
            if x:
                print(sym, k, 'COMMON')
                for v in x:
                    print('  ', v)

    if not ok or debug:
        _printelems(sym, etst)
        showsymelems(sym, {**otherelems, **elems0}, scale=12, scan=12, offset=0, **kw)  # type: ignore
        assert ok

    if not compound:
        storedelems = ipd.sym.xtal.symelems(sym, asdict=True)
        # assert elems0 == storedelems

    assert not debug

@pytest.mark.fast
def test_symelems_F4132(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], label='C2'),
          SymElem(2, axis=[-1, 0, 1], cen=[0.25, 0.125, 0.0], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
       ],
       C31=[
          SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.166666667, 0.0], hel=0.577350269, label='C31'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.0, 0.166666667], hel=1.154700538, label='C32'),  # type: ignore
       ],
       C41=[
          SymElem(4, axis=[0, 1, 0], cen=[0.25, 0.0, 0.0], hel=0.25, label='C41'),  # type: ignore
       ],
       C43=[
          SymElem(4, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.75, label='C43'),  # type: ignore
       ],
    )
    helper_test_symelem('F4132', val, debug, **kw)

@pytest.mark.fast
def test_symelems_F432(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.0, 0.0], label='C2'),
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.25], label='C2'),
          SymElem(2, axis=[0, -1, 1], cen=[0.0, 0.0, 0.5], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
       ],
       C4=[
          SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C4'),
       ],
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.166666667, 0.0], hel=0.577350269, label='C31'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.0, 0.166666667], hel=1.154700538, label='C32'),  # type: ignore
       ],
       C42=[
          SymElem(4, axis=[0, 0, 1], cen=[0.25, 0.25, 0.0], hel=0.5, label='C42'),  # type: ignore
       ],
    )
    helper_test_symelem('F432', val, debug, **kw)

@pytest.mark.fast
def test_symelems_F23(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.25], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
       ],
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[-1, 1, 1], cen=[0.16666666666666666, 0.16666666666666666, 0.0], hel=0.5773502691896257, label='C31'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[-1, 1, 1], cen=[0.16666666666666666, 0.0, 0.16666666666666666], hel=1.154700538368877, label='C32'),  # type: ignore
       ],
    )
    helper_test_symelem('F23', val, debug, **kw)

@pytest.mark.fast
def test_symelems_I432(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.0, 0.0], label='C2'),
          SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.5, 0.25], label='C2'),
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.5], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
       ],
       C4=[
          SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C4'),
       ],
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.25, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 1], cen=[0.25, 0.0, 0.0], hel=0.707106781, label='C21'),  # type: ignore
          SymElem(2, axis=[-1, 1, 0], cen=[0.5, 0.0, 0.0], hel=0.707106781, label='C21'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[1, -1, 1], cen=[0.333333333, 0.333333333, 0.0], hel=0.577350269, label='C31'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[1, 1, -1], cen=[0.333333333, 0.0, 0.333333333], hel=1.154700538, label='C32'),  # type: ignore
       ],
       C42=[
          SymElem(4, axis=[0, 1, 0], cen=[0.5, 0.0, 0.0], hel=0.5, label='C42'),  # type: ignore
       ],
    )
    helper_test_symelem('I432', val, debug, **kw)

@pytest.mark.fast
def test_symelems_R32(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0.57735, 1.0, 0.0], cen=[1.0, 1.0, -1e-09], label='C2'),
          SymElem(2, axis=[-0.57735, 1.0, 0.0], cen=[0.333333333, 0.166666667, 0.166666667], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[0, 0, 1], cen=[1e-09, 1e-09, 0.333333334], label='C3'),
       ],
       C21=[
          SymElem(2, axis=[1, 0, 0], cen=[0.083333333, 0.166666667, 0.166666667], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.0], hel=0.5, label='C21'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[0, 0, 1], cen=[0.333333334, 0.0, 0.0], hel=0.666666667, label='C32'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.0, 0.0], hel=0.333333333, label='C31'),  # type: ignore
       ],
    )
    helper_test_symelem('R32', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P1211(debug=False, **kw):
    val = dict(
       C21=[
          SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.5], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.5], hel=0.5, label='C21'),  # type: ignore
       ],
       C11=[
          SymElem(1, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),  # type: ignore
       ],
    )
    helper_test_symelem('P1211', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P2221(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], label='C2'),
          SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.25], label='C2'),
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.5, 0.0], label='C2'),
          SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], label='C2'),
       ],
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.5, 0.0], hel=0.5, label='C21'),  # type: ignore
       ],
    )
    helper_test_symelem('P2221', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P21212(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, -1.0], label='C2'),
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], label='C2'),
       ],
       C21=[
          SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.5], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.5], hel=0.5, label='C21'),  # type: ignore
       ],
    )
    helper_test_symelem('P21212', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P1(debug=False, **kw):
    val = dict()
    helper_test_symelem('P1', val, debug, **kw)

@pytest.mark.fast
def test_symelems_C121(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], label='C2'),
          SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.5], label='C2'),
       ],
       C21=[
          SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.5], hel=0.5, label='C21'),  # type: ignore
       ],
    )
    helper_test_symelem('C121', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P3(debug=False, **kw):
    val = dict(C3=[
       SymElem(3, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C3'),
       SymElem(3, axis=[0, 0, 1], cen=[-0.333333333, 0.333333335, 0.0], label='C3'),
       SymElem(3, axis=[0, 0, 1], cen=[0.333333334, 0.666666666, 0.0], label='C3'),
    ], )
    helper_test_symelem('P3', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P222(debug=False, **kw):
    val = dict(C2=[
       SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
       SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], label='C2'),
       SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], label='C2'),
       SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.5], label='C2'),
       SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.5], label='C2'),
       SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], label='C2'),
       SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.5, 0.0], label='C2'),
       SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.5, 0.5], label='C2'),
       SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], label='C2'),
       SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.0], label='C2'),
       SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.5], label='C2'),
       SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.5, 0.0], label='C2'),
    ], )
    helper_test_symelem('P222', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P23(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.5], label='C2'),
          SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.5], label='C2'),
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.5, 0.5], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
       ],
       C31=[
          SymElem(3, axis=[1, -1, 1], cen=[0.333333333, 0.333333333, 0.0], hel=0.577350269, label='C31'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[1, 1, -1], cen=[0.333333333, 0.0, 0.333333333], hel=1.154700538, label='C32'),  # type: ignore
       ],
    )
    helper_test_symelem('P23', val, debug, **kw)

@pytest.mark.fast
def test_symelems_I213(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.25, 0.0], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
       ],
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], hel=0.5, label='C21'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.166666667, 0.0], hel=0.577350269, label='C31'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.333333333, 0.666666667], hel=1.154700538, label='C32'),  # type: ignore
       ],
    )
    helper_test_symelem('I213', val, debug, **kw)

@pytest.mark.fast
def test_symelems_I23(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
          SymElem(2, axis=[1, 0, 0], cen=[0.5, 0.0, 0.5], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
       ],
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.25, 0.0], hel=0.5, label='C21'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[1, -1, 1], cen=[0.333333333, 0.333333333, 0.0], hel=0.577350269, label='C31'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[1, 1, -1], cen=[0.333333333, 0.0, 0.333333333], hel=1.154700538, label='C32'),  # type: ignore
       ],
    )
    helper_test_symelem('I23', val, debug, **kw)

@pytest.mark.fast
def test_symelems_I4(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], label='C2'),
       ],
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.75, 0.75, 0.0], hel=0.5, label='C21'),  # type: ignore
       ],
       C4=[
          SymElem(4, axis=[0, 0, 1], cen=[-0.5, 0.5, 1.0], label='C4'),
       ],
       C42=[
          SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], hel=0.5, label='C42'),  # type: ignore
       ],
    )
    helper_test_symelem('I4', val, debug, **kw)

@pytest.mark.fast
def test_symelems_I41(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.5, 0.0], label='C2'),
       ],
       C41=[
          SymElem(4, axis=[0, 0, 1], cen=[0.25, 0.75, 0.0], hel=0.25, label='C41'),  # type: ignore
       ],
       C43=[
          SymElem(4, axis=[0, 0, 1], cen=[0.25, 0.25, 0.0], hel=0.75, label='C43'),  # type: ignore
       ],
    )
    helper_test_symelem('I41', val, debug, **kw)

@pytest.mark.fast
def test_symelems_I4132(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.0, 0.25], label='C2'),
          SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.25, 0.125], label='C2'),
          SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.75, 0.375], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
       ],
       C21=[
          SymElem(2, axis=[0, 1, 1], cen=[0.125, 0.25, 0.0], hel=0.707106781, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 1], cen=[0.375, 0.0, 0.25], hel=0.707106781, label='C21'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.166666667, 0.0], hel=0.577350269, label='C31'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.333333333, 0.666666667], hel=1.154700538, label='C32'),  # type: ignore
       ],
       C41=[
          SymElem(4, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.25, label='C41'),  # type: ignore
       ],
       C43=[
          SymElem(4, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], hel=0.75, label='C43'),  # type: ignore
       ],
    )
    helper_test_symelem('I4132', val, debug, **kw)

@pytest.mark.fast
def test_symelems_R3(debug=False, **kw):
    val = dict(
       C3=[
          SymElem(3, axis=[0, 0, 1], cen=[1e-09, 1e-09, 0.333333334], label='C3'),
       ],
       C32=[
          SymElem(3, axis=[0, 0, 1], cen=[0.333333334, 0.0, 0.0], hel=0.666666667, label='C32'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[0, 0, 1], cen=[0.333333334, 0.333333333, 0.0], hel=0.333333333, label='C31'),  # type: ignore
       ],
    )
    helper_test_symelem('R3', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P3121(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0.57735, 1.0, 0.0], cen=[0.0, 0.0, 1e-09], label='C2'),
          SymElem(2, axis=[-0.57735, 1.0, 0.0], cen=[0.0, 0.0, 0.166666667], label='C2'),
       ],
       C21=[
          SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.333333333], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.833333333], hel=0.5, label='C21'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.333333333, label='C31'),  # type: ignore
          SymElem(3, axis=[0, 0, 1], cen=[0.666666666, 0.333333333, 0.0], hel=0.333333333, label='C31'),  # type: ignore
       ],
    )
    helper_test_symelem('P3121', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P212121(debug=False, **kw):
    val = dict(C21=[
       SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
       SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.0], hel=0.5, label='C21'),  # type: ignore
       SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.25], hel=0.5, label='C21'),  # type: ignore
       SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], hel=0.5, label='C21'),  # type: ignore
       SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.5, 0.0], hel=0.5, label='C21'),  # type: ignore
       SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.5], hel=0.5, label='C21'),  # type: ignore
    ], )
    helper_test_symelem('P212121', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P31(debug=False, **kw):
    val = dict(
       C31=[
          SymElem(3, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.333333333, label='C31'),  # type: ignore
          SymElem(3, axis=[0, 0, 1], cen=[0.666666666, 0.333333333, 0.0], hel=0.333333333, label='C31'),  # type: ignore
          SymElem(3, axis=[0, 0, 1], cen=[0.333333333, 0.666666666, 0.0], hel=0.333333333, label='C31'),  # type: ignore
       ],
       C11=[
          SymElem(1, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),  # type: ignore
       ],
    )
    helper_test_symelem('P31', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P32(debug=False, **kw):
    val = dict(
       C32=[
          SymElem(3, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.666666667, label='C32'),  # type: ignore
          SymElem(3, axis=[0, 0, 1], cen=[0.666666666, 0.333333333, 0.0], hel=0.666666667, label='C32'),  # type: ignore
          SymElem(3, axis=[0, 0, 1], cen=[0.333333333, 0.666666666, 0.0], hel=0.666666667, label='C32'),  # type: ignore
       ],
       C11=[
          SymElem(1, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),  # type: ignore
       ],
    )
    helper_test_symelem('P32', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P213(debug=False, **kw):
    val = dict(
       C3=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
       ],
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], hel=0.5, label='C21'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[-1, 1, 1], cen=[0.16666666666666666, 0.16666666666666666, 0.0], hel=0.5773502691896257, label='C31'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.3333333333333333, 0.6666666666666666], hel=1.154700538368877, label='C32'),  # type: ignore
       ],
    )
    helper_test_symelem('P213', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P3221(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0.57735, 1.0, 0.0], cen=[0.0, 0.0, -1e-09], label='C2'),
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.166666667], label='C2'),
       ],
       C21=[
          SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.166666667], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.666666667], hel=0.5, label='C21'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.666666667, label='C32'),  # type: ignore
          SymElem(3, axis=[0, 0, 1], cen=[0.666666666, 0.333333333, 0.0], hel=0.666666667, label='C32'),  # type: ignore
       ],
    )
    helper_test_symelem('P3221', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P41(debug=False, **kw):
    val = dict(
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
       ],
       C41=[
          SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.25, label='C41'),  # type: ignore
          SymElem(4, axis=[0, 0, 1], cen=[0.5, 0.5, 0.0], hel=0.25, label='C41'),  # type: ignore
       ],
       C11=[
          SymElem(1, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),  # type: ignore
       ],
    )
    helper_test_symelem('P41', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P41212(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.0, 0.75], label='C2'),
       ],
       C21=[
          SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.5, 0.5], hel=0.707106781, label='C21'),  # type: ignore
       ],
       C41=[
          SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], hel=0.25, label='C41'),  # type: ignore
       ],
    )
    helper_test_symelem('P41212', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P4132(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.75, 0.375], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
       ],
       C21=[
          SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 1], cen=[0.125, 0.25, 0.0], hel=0.707106781, label='C21'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.166666667, 0.0], hel=0.577350269, label='C31'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.333333333, 0.666666667], hel=1.154700538, label='C32'),  # type: ignore
       ],
       C41=[
          SymElem(4, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.25, label='C41'),  # type: ignore
       ],
    )
    helper_test_symelem('P4132', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P4232(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
          SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.5, 0.25], label='C2'),
          SymElem(2, axis=[1, 0, 0], cen=[0.5, 0.0, 0.5], label='C2'),
          SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.5, 0.5], label='C2'),
          SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.5, 0.75], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
       ],
       C21=[
          SymElem(2, axis=[0, 1, 1], cen=[0.25, 0.0, 0.0], hel=0.707106781, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 1], cen=[0.75, 0.0, 0.0], hel=0.707106781, label='C21'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[1, -1, 1], cen=[0.333333333, 0.333333333, 0.0], hel=0.577350269, label='C31'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[1, 1, -1], cen=[0.333333333, 0.0, 0.333333333], hel=1.154700538, label='C32'),  # type: ignore
       ],
       C42=[
          SymElem(4, axis=[0, 1, 0], cen=[0.5, 0.0, 0.0], hel=0.5, label='C42'),  # type: ignore
          SymElem(4, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C42'),  # type: ignore
       ],
    )
    helper_test_symelem('P4232', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P43(debug=False, **kw):
    val = dict(
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
       ],
       C43=[
          SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.75, label='C43'),  # type: ignore
          SymElem(4, axis=[0, 0, 1], cen=[0.5, 0.5, 0.0], hel=0.75, label='C43'),  # type: ignore
       ],
       C11=[
          SymElem(1, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),  # type: ignore
       ],
    )
    helper_test_symelem('P43', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P432(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.0, 0.0], label='C2'),
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.5], label='C2'),
          SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.0, 0.5], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
       ],
       C4=[
          SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C4'),
          SymElem(4, axis=[1, 0, 0], cen=[0.0, -0.5, 0.5], label='C4'),
       ],
       C21=[
          SymElem(2, axis=[-1, 0, 1], cen=[0.5, 0.5, 0.0], hel=0.707106781, label='C21'),  # type: ignore
          SymElem(2, axis=[-1, 1, 0], cen=[0.5, 0.0, 0.0], hel=0.707106781, label='C21'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[1, -1, 1], cen=[0.333333333, 0.333333333, 0.0], hel=0.577350269, label='C31'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[1, 1, -1], cen=[0.333333333, 0.0, 0.333333333], hel=1.154700538, label='C32'),  # type: ignore
       ],
    )
    helper_test_symelem('P432', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P43212(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[1, 1, 0], cen=[0.5, 0.5, 0.0], label='C2'),
       ],
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.875], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.75, 0.125], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.5, 0.0], hel=0.7071067811865476, label='C21'),  # type: ignore
       ],
       C43=[
          SymElem(4, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.75, label='C43'),  # type: ignore
       ],
    )
    helper_test_symelem('P43212', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P4332(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.25, 0.125], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
       ],
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 1], cen=[0.375, 0.0, 0.25], hel=0.707106781, label='C21'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.166666667, 0.0], hel=0.577350269, label='C31'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.333333333, 0.666666667], hel=1.154700538, label='C32'),  # type: ignore
       ],
       C43=[
          SymElem(4, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], hel=0.75, label='C43'),  # type: ignore
       ],
    )
    helper_test_symelem('P4332', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P6(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], label='C2'),
       ],
       C3=[
          SymElem(3, axis=[0, 0, 1], cen=[-0.333333332, 0.333333334, 0.0], label='C3'),
       ],
       C6=[
          SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], label='C6'),
       ],
       C11=[
          SymElem(1, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),  # type: ignore
       ],
    )
    helper_test_symelem('P6', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P61(debug=False, **kw):
    val = dict(
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.333333333, 0.0], hel=0.333333333, label='C31'),  # type: ignore
       ],
       C61=[
          SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.166666667, label='C61'),  # type: ignore
       ],
       C11=[
          SymElem(1, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),  # type: ignore
       ],
    )
    helper_test_symelem('P61', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P6122(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], label='C2'),
       ],
       C21=[
          SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
       ],
       C31=[
          SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.333333333, 0.0], hel=0.333333333, label='C31'),  # type: ignore
       ],
       C61=[
          SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.166666667, label='C61'),  # type: ignore
       ],
    )
    helper_test_symelem('P6122', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P62(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], label='C2'),
       ],
       C32=[
          SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.333333333, 0.0], hel=0.666666667, label='C32'),  # type: ignore
       ],
       C62=[
          SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.333333333, label='C62'),  # type: ignore
       ],
    )
    helper_test_symelem('P62', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P63(debug=False, **kw):
    val = dict(
       C3=[
          SymElem(3, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C3'),
          SymElem(3, axis=[0, 0, 1], cen=[-0.333333332, 0.333333334, 0.5], label='C3'),
       ],
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
       ],
       C63=[
          SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.5, label='C63'),  # type: ignore
       ],
    )
    helper_test_symelem('P63', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P64(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
          SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], label='C2'),
       ],
       C31=[
          SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.333333333, 0.0], hel=0.333333333, label='C31'),  # type: ignore
       ],
       C64=[
          SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.666666667, label='C64'),  # type: ignore
       ],
    )
    helper_test_symelem('P64', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P65(debug=False, **kw):
    val = dict(
       C21=[
          SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.333333333, 0.0], hel=0.666666667, label='C32'),  # type: ignore
       ],
       C65=[
          SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.833333333, label='C65'),  # type: ignore
       ],
       C11=[
          SymElem(1, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),  # type: ignore
       ],
    )
    helper_test_symelem('P65', val, debug, **kw)

@pytest.mark.fast
def test_symelems_P6522(debug=False, **kw):
    val = dict(
       C2=[
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], label='C2'),
       ],
       C21=[
          SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
       ],
       C32=[
          SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.333333333, 0.0], hel=0.666666667, label='C32'),  # type: ignore
       ],
       C65=[
          SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.833333333, label='C65'),  # type: ignore
       ],
    )
    helper_test_symelem('P6522', val, debug, **kw)

@pytest.mark.fast
def test_remove_redundant_screws():
    sym = 'P212121'
    f4cel = ipd.sym.xtal.sgframes(sym, cells=6, cellgeom='nonsingular')
    lattice = ipd.sym.xtal.lattice_vectors(sym, cellgeom='nonsingular')
    elems = {
       'C21': [
          SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 0, 1], cen=[-0.25, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, -0.25, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.25], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, -0.25], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.5, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 0, 1], cen=[-0.25, 0.5, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.5], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, -0.5], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, -0.25, 0.5], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, -0.25, -0.5], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.75, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.75], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.75, 0.5], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.75, -0.5], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 0, 1], cen=[0.25, 1.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 0, 1], cen=[-0.25, 1.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.75], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 1.25, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 1.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, -0.25, 1.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 1.25, 0.5], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 1.25, -0.5], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.75, 1.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 1.25], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 1.25], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 1.25, 1.0], hel=0.5, label='C21'),  # type: ignore
       ]
    }
    # ic(f4cel.shape)
    # ic(lattice)
    elems2 = _remove_redundant_screws(elems, f4cel, lattice)
    # ic(elems2)
    assert elems2 == {
       'C21': [
          SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 0, 1], cen=[-0.25, 0.0, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[1, 0, 0], cen=[0.0, -0.25, 0.0], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.25], hel=0.5, label='C21'),  # type: ignore
          SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, -0.25], hel=0.5, label='C21'),  # type: ignore
       ]
    }

if __name__ == '__main__':
    main()
