import numpy as np
import pytest
from icecream import ic

import ipd
from ipd.sym.permutations import symframe_permutations_torch
from ipd.sym.xtal.spacegroup_symelems import _compute_symelems, _find_compound_symelems
from ipd.sym.xtal.SymElem import *
from ipd.sym.xtal.SymElem import _make_operator_component_joint_ids

def main():
    test_screw_elem()
    test_screw_elem_frames()

    # mcdock_bug1()
    # assert 0
    check_frame_opids()

@pytest.mark.fast
def test_screw_elem_frames():
    f31a = ipd.homog.hrot([0, 0, 1], 120) @ ipd.homog.htrans([0, 0, 1 / 3])
    assert ipd.sym.xtal.symelem_of(f31a) == SymElem(
        3,
        axis=[0, 0, 1],
        cen=[0.0, 0.0, 0.0],
        hel=1 / 3,  # type: ignore
        label="C31")  # type: ignore
    f31b = ipd.homog.hrot([0, 0, -1], 240) @ ipd.homog.htrans([0, 0, 1 / 3])
    assert np.allclose(f31a, f31b)
    ipd.sym.xtal.symelem_of(f31a) == ipd.sym.xtal.symelem_of(f31b)  # type: ignore
    f31b = ipd.homog.hrot([0, 0, -1], 240) @ ipd.homog.htrans([0, 0, 1 / 3])

    f32a = ipd.homog.hrot([0, 0, 1], 120) @ ipd.homog.htrans([0, 0, 2 / 3])
    assert ipd.sym.xtal.symelem_of(f32a) == SymElem(
        3,
        axis=[0, 0, 1],
        cen=[0.0, 0.0, 0.0],
        hel=2 / 3,  # type: ignore
        label="C32")  # type: ignore
    f32b = ipd.homog.hrot([0, 0, 1], 240) @ ipd.homog.htrans([0, 0, 1 / 3])
    assert ipd.sym.xtal.symelem_of(f32b) == SymElem(
        3,
        axis=[0, 0, 1],
        cen=[0.0, 0.0, 0.0],
        hel=2 / 3,  # type: ignore
        label="C32")  # type: ignore

@pytest.mark.fast
def test_screw_elem():
    ic("test_screw_elem")
    S2 = np.sqrt(2)
    S3 = np.sqrt(3)

    assert SymElem(1, [0, 0, 1], hel=1).label == "C11"
    assert SymElem(2, [0, 0, 1], hel=0.5).label == "C21"  # type: ignore
    assert SymElem(2, [0, 1, 1], hel=S2 / 2).label == "C21"
    assert SymElem(2, [1, 1, 1], hel=S3 / 2).label == "C21"
    with pytest.raises(ScrewError):
        print(SymElem(2, [1, 1, 1], hel=1))
    with pytest.raises(ScrewError):
        print(SymElem(2, [0, 0, 1], hel=1))
    with pytest.raises(ScrewError):
        print(SymElem(1, [0, 0, 1], hel=0.5))  # type: ignore
    with pytest.raises(ScrewError):
        print(SymElem(1, [1, 2, 3], hel=0.5))  # type: ignore
    assert SymElem(3, [0, 0, 1], hel=1 / 3).label == "C31"  # type: ignore
    assert SymElem(3, [0, 0, -1], hel=2 / 3).label == "C32"  # type: ignore
    assert SymElem(3, [1, 1, 1], hel=S3 * 1 / 3).label == "C31"
    assert SymElem(3, [1, 1, 1], hel=S3 * 2 / 3).label == "C32"
    with pytest.raises(ScrewError):
        print(SymElem(3, [0, 0, 1], hel=1))

    assert SymElem(4, [0, 0, 1], hel=0.25).label == "C41"  # type: ignore
    assert SymElem(4, [0, 0, 1], hel=0.50).label == "C42"  # type: ignore
    assert SymElem(4, [0, 0, 1], hel=0.75).label == "C43"  # type: ignore
    assert SymElem(4, [0, 0, 1], hel=-0.5).label == "C42"  # type: ignore
    assert SymElem(4, [0, 0, 1], hel=-0.25).label == "C43"  # type: ignore
    with pytest.raises(ScrewError):
        print(SymElem(4, [0, 0, 1], hel=-0.51))  # type: ignore
    with pytest.raises(ScrewError):
        print(SymElem(4, [0, 0, 1], hel=1))

    assert SymElem(6, [0, 0, 1], hel=1 / 6).label == "C61"  # type: ignore
    assert SymElem(6, [0, 1, 0], hel=2 / 6).label == "C62"  # type: ignore
    assert SymElem(6, [1, 0, 0], hel=3 / 6).label == "C63"  # type: ignore
    assert SymElem(6, [0, 1, 0], hel=4 / 6).label == "C64"  # type: ignore
    assert SymElem(6, [0, 0, 1], hel=5 / 6).label == "C65"  # type: ignore
    for i in range(100):
        x = ipd.homog.hrand()
        x31 = ipd.homog.hinv(x) @ ipd.homog.hrot([0, 0, 1], 120) @ ipd.homog.htrans([0, 0, 1]) @ x
        x32 = ipd.homog.hinv(x) @ ipd.homog.hrot([0, 0, 1], 240) @ ipd.homog.htrans([0, 0, 1]) @ x
        assert np.allclose(1, ipd.homog.axis_angle_cen_hel_of(x31)[3])
        assert np.allclose(-1, ipd.homog.axis_angle_cen_hel_of(x32)[3])

def mcdock_bug1():
    sym = "I4132"
    elems = ipd.sym.xtal.symelems(sym)
    ic(elems)

    frames4 = ipd.sym.frames(sym, sgonly=True, cells=4)
    f = frames4[ipd.sym.xtal.sg_symelem_frame444_opcompids_dict[sym][:, 1, 1] == 109]
    ipd.showme(f, scale=10)
    ic(f)

def check_frame_opids():
    sym = "P3"

    # unitframes = ipd.sym.xtal.sgframes(sym, cellgeom='unit')
    n_std_frames = 4
    n_min_frames = 2
    frames = ipd.sym.xtal.sgframes(sym, cells=n_std_frames, cellgeom="nonsingular")
    frames2 = ipd.sym.xtal.sgframes(sym, cells=n_std_frames - 2, cellgeom="nonsingular")

    lattice = ipd.sym.xtal.lattice_vectors(sym, cellgeom="nonsingular")
    ic(lattice)
    # ic(unitframes.shape)
    ic(frames.shape)

    elems = _compute_symelems(sym, aslist=True)
    # for e in elems:
    # ic(e)
    # elems = _check_alternate_elems(sym, lattice, elems, frames, frames2)
    # for e in elems:
    # ic(e)
    # assert 0
    celems = _find_compound_symelems(sym, elems, aslist=True)
    for e in elems + celems:
        ic(e)
        # ipd.showme(e.tolattice(lattice), scale=10)
    # perms = ipd.sym.xtal.sgpermutations(sym, cells=4)

    scale = 10

    # ipd.showme(frames, scale=10)
    # assert 0

    perms = symframe_permutations_torch(frames, maxcols=len(frames2))

    # elems = ipd.sym.xtal.symelems(sym, cyclic=True, screws=False)

    for i, unitelem in enumerate(elems + celems):
        alternate_elem_frames = elems[0].tolattice(lattice).operators
        # alternate_elem_frames = [np.eye(4), ipd.homog.hrot([0, 0, 1], 120)]
        for j, elemframe in enumerate(alternate_elem_frames):
            # if True:
            # if not elem.issues: continue

            # elem = elems[2]
            # ic(elem)
            # ic(elem.kind)
            # ic(elem.isoct)
            # ic(elem.cen)
            ic(i)
            # ic(unitelem)
            # ic(elemframe)
            elem = unitelem.tolattice(lattice).xformed(elemframe)
            ic(elem)
            # continue

            # ipd.showme(elem, scale=scale, name='ref', symelemscale=5)
            # offset = ipd.homog.htrans([.02, .025, .03])
            # ipd.showme(elem.operators @ elem.origin @ offset, scale=scale)
            # ipd.showme(ipd.hscaled(scale, elem.cen))
            # ipd.sym.showsymelems(sym, [elem], scale=scale)
            # ipd.showme(elem.operators @ offset, scale=scale)
            # ipd.showme(frames2 @ offset, scale=scale)

            # compids = elem.frame_component_ids_bycenter(frames, sanitycheck=False)elem
            try:
                compids = elem.frame_component_ids(frames, perms, sanitycheck=True)
            except ComponentIDError:
                print("!" * 80, flush=True)
                print("elem has bad componentids, trying an alternate position")
                print("!" * 80, flush=True)
                continue

            opids = elem.frame_operator_ids(frames, sanitycheck=True)
            opcompids = _make_operator_component_joint_ids(elem, elem, frames, opids, compids, sanitycheck=True)

            print("SUCCESS", flush=True)
            break

        if i != 8:
            continue
        # tmp = ipd.homog.hxformpts(ipd.hscaled(100, frames[opcompids == 109]), ipd.hscaled(100, elem.cen + elem.axis))
        # ic(tmp)
        # assert 0
        # ids = opids
        # ids = compids
        ids = opcompids  # type: ignore
        offset = ipd.homog.htrans([0.002, 0.0025, 0.003])
        seenit = np.empty((0, 4))
        for i in range(np.max(ids)):
            assert np.sum(ids == i) > 0
            compframes = frames[ids == i]
            ipd.showme(compframes @ elem.origin @ offset, scale=scale, name=f"id{i}")  # type: ignore
            # cens = einsum('fij,j->fi', compframes, elem.origin[:, 3])
            # assert not np.any(np.all(np.isclose(cens[None], seenit[:, None]), axis=2))
            # seenit = np.concatenate([cens, seenit])

    assert 0

if __name__ == "__main__":
    main()
