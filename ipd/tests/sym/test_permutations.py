import itertools

import numpy as np
import pytest

import ipd
from ipd.sym.xtal.spacegroup_symelems import _compute_symelems

def main():
    # test_symelems_I432(showme=False)
    # assert 0

    # for k, v in ipd.sym.xtal.sg_symelem_frame_ids_dict.items():
    # print(k, v.shape)
    # assert 0

    # WIP_P23_perm()
    # test_icos_perm()
    test_icos_perm()
    ic("PASS test_permutations")  # type: ignore

def WIP_opcompid():
    f = ipd.sym.frames("P23", cells=4)
    ic(ipd.sym.xtal.symelems("P23"))  # type: ignore
    for ielem, se in enumerate(ipd.sym.xtal.symelems("P23")):
        fcompid = ipd.sym.xtal.sg_symelem_frame444_compids_dict["P23"][:, ielem]
        fopid = se.frame_operator_ids(f)
        ids = fcompid.copy()
        for i in range(np.max(fopid)):
            fcids = fcompid[fopid == i]
            idx0 = fcompid == fcids[0]
            for fcid in fcids[1:]:
                idx = fcompid == fcid
                ids[idx] = min(min(ids[idx]), min(ids[idx0]))
        for i, id in enumerate(sorted(set(ids))):
            ids[ids == id] = i
        for i in range(max(ids)):
            ic(f[ids == i, :3, 3])  # type: ignore
        assert 0

def WIP_P23_perm():
    frames = ipd.sym.xtal.sgframes("P23", cells=4)
    # semap = ipd.sym.xtal.symelems('P23')
    semap = _compute_symelems("P23")

    selems = list(itertools.chain(*semap.values()))

    perms = ipd.sym.symframe_permutations_torch(frames)

    compid = -np.ones((len(frames), len(selems)), dtype=np.int32)
    for ielem, se in enumerate(selems):
        compid[:, ielem] = se.frame_component_ids(frames, perms)

    ielem = 4
    ecen, eaxs = selems[ielem].cen, selems[ielem].axis
    ic(selems[ielem])  # type: ignore
    for icomp in range(np.max(compid[:, ielem])):
        # ic(np.max(frames[:, :3, 3]))
        selframes = compid[:, ielem] == icomp
        assert len(selframes)
        testf = frames[selframes] @ ipd.homog.htrans(ecen) @ ipd.homog.halign([0, 0, 1], eaxs)
        # ic(testf.shape)
        # print(testf[:, :3, 3])
        # ipd.showme(testf)

    # sym = 'I4132'
    # frames = ipd.sym.frames('I4132', sgonly=True, cells=5)
    # perms = ipd.sym.symframe_permutations_torch(frames)
    # perms = ipd.load('/home/sheffler/WILLUTIL_SYM_PERMS_I4132_5_int32.pickle')
    # unitframes = ipd.sym.xtal.sgframes('I4132', cellgeom='unit')
    # ic(unitframes.shape)

    # f = ipd.sym.xtal.sgframes('I213', cells=2, cellgeom=[10])
    # ipd.showme(f @ ipd.homog.htrans([0, 0, 0]) @ ipd.homog.halign([0, 0, 1], [1, 1, 1]),**vizopt)
    # ipd.showme(f @ ipd.homog.htrans([5, -5, 0]) @ ipd.homog.halign([0, 0, 1], [1, -1, 1]),**vizopt)
    # ipd.showme(f @ ipd.homog.halign([0, 0, 1], [1, -1, 1]),**vizopt)

    # ipd.save(perms, '/home/sheffler/WILLUTIL_SYM_PERMS_I4132_5.pickle')
    # assert 0

    assert 0

@pytest.mark.fast
def test_icos_perm():
    frames = ipd.sym.frames("icos")
    perms = ipd.sym.permutations("icos")
    pts = ipd.homog.hxform(frames, [1, 2, 3])
    d = ipd.homog.hnorm(pts[0] - pts)
    for i, perm in enumerate(perms):
        xinv = ipd.homog.hinv(frames[i])
        for j, p in enumerate(perm):
            assert np.allclose(xinv @ frames[p], frames[j])
        assert np.allclose(d, ipd.homog.hnorm(pts[i] - pts[perm]))

if __name__ == "__main__":
    main()
