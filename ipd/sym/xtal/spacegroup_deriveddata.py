REBUILD_SPACEGROUP_DATA = False
# REBUILD_SPACEGROUP_DATA = True

import itertools

import numpy as np
from icecream import ic

from ipd.dev import have_package_data, load_package_data, save_package_data
from ipd.sym.permutations import symframe_permutations_torch
from ipd.sym.xtal.spacegroup_data import *
from ipd.sym.xtal.spacegroup_symelems import _compute_symelems, _find_compound_symelems
from ipd.sym.xtal.spacegroup_util import *
from ipd.sym.xtal.SymElem import ComponentIDError, _make_operator_component_joint_ids

def _get_spacegroup_data():
    sgdata = dict()
    if have_package_data('spacegroup_data'):
        sgdata = load_package_data('spacegroup_data')

    sg_frames_dict = sgdata.get('sg_frames_dict', dict())  # type: ignore
    sg_cheshire_dict = sgdata.get('sg_cheshire_dict', dict())  # type: ignore
    sg_symelem_dict = sgdata.get('sg_symelem_dict', dict())  # type: ignore
    sg_permutations444_dict = sgdata.get('sg_permutations444_dict', dict())  # type: ignore
    sg_symelem_frame444_opids_dict = sgdata.get('sg_symelem_frame444_opids_dict', dict())  # type: ignore
    sg_symelem_frame444_compids_dict = sgdata.get('sg_symelem_frame444_compids_dict', dict())  # type: ignore
    sg_symelem_frame444_opcompids_dict = sgdata.get('sg_symelem_frame444_opcompids_dict', dict())  # type: ignore
    if not REBUILD_SPACEGROUP_DATA:
        return sgdata
    #

    # del sg_symelem_dict['P1211']
    # del sg_permutations444_dict['P1211']
    # del sg_symelem_frame444_opcompids_dict['P1211']

    # sg_symelem_dict = dict()
    # sg_symelem_frame444_opcompids_dict = dict()

    #

    from ipd.sym.xtal import spacegroup_frames

    sg_improper = dict()

    ichiral = 0
    for isym, (sym, symtag) in enumerate(sg_tag.items()):
        if sym in sgdata: continue  # type: ignore
        assert REBUILD_SPACEGROUP_DATA

        if symtag in sg_lattice:
            sg_lattice[sym] = sg_lattice[symtag]
        else:
            sg_lattice[symtag] = sg_lattice[sym]

        if sym not in sg_frames_dict:
            tmp, sg_cheshire_dict[sym] = getattr(spacegroup_frames, f'symframes_{symtag}')()
            frames = np.zeros((len(tmp), 4, 4))
            frames[:, 3, 3] = 1
            frames[:, 0, :3] = tmp[:, 0:3]
            frames[:, 1, :3] = tmp[:, 3:6]
            frames[:, 2, :3] = tmp[:, 6:9]
            frames[:, :3, 3] = tmp[:, 9:]
            frames[frames[:, 0, 3] > 0.999, 0, 3] -= 1
            frames[frames[:, 1, 3] > 0.999, 1, 3] -= 1
            frames[frames[:, 2, 3] > 0.999, 2, 3] -= 1
            assert np.sum(frames == 12345) == 0
            sg_frames_dict[sym] = frames
            sg_imporper = not np.allclose(1, np.linalg.det(frames))
        frames = sg_frames_dict[sym]
        # if not sg_imporper:

        # if not (sg_lattice[sym] == 'CUBIC' and sg_is_chiral(sym)): continue
        if not sg_is_chiral(sym):
            continue

        print('-' * 40, ichiral, sym, '-' * 40, flush=True)
        ichiral += 1
        n_std_cells = number_of_canonical_cells(sym)
        latticevec = lattice_vectors(sym, 'nonsingular')
        stdframes = latticeframes(frames, latticevec, n_std_cells)
        stdframes2 = latticeframes(frames, latticevec, n_std_cells - 2)
        stdframes1 = latticeframes(frames, latticevec, n_std_cells - 3)

        update = False
        IERROR = -900_000_000
        if sym not in sg_symelem_dict:
            update = True
            # print(sym, 'detect symelems', flush=True)
            print('_compute_symelems', flush=True)
            sg_symelem_dict[sym] = _compute_symelems(sym, frames)
            sg_symelem_dict[sym] = list(itertools.chain(*sg_symelem_dict[sym].values()))  # flatten  # type: ignore
            print('_find_compound_symelems', flush=True)
            celems = _find_compound_symelems(sym, sg_symelem_dict[sym], stdframes, stdframes2, stdframes1)
            sg_symelem_dict[sym] += list(itertools.chain(*celems.values()))  # type: ignore
            for ise, e in enumerate(sg_symelem_dict[sym]):
                e.index = ise
                print(f'{ise:2}', e.label, e, flush=True)
        # len(frames)*8 keeps only enough perm frames for 2x2x2 cell
        # if n_std_cells != 4 or sym not in sg_permutations444_dict:
        if sym not in sg_permutations444_dict:
            print('permutations', flush=True)
            # print(sym, 'compute permutations', flush=True)
            sg_permutations444_dict[sym] = symframe_permutations_torch(stdframes, maxcols=len(frames) * 8)
        perms = sg_permutations444_dict[sym]
        nops = len(sg_symelem_dict[sym])
        if sym not in sg_symelem_frame444_opcompids_dict:
            print('compute op/comp ids', flush=True)
            update = True
            # print('rebuild symelem frameids', sym, flush=True)
            sg_symelem_frame444_opids_dict[sym] = -np.ones((len(stdframes), nops), dtype=np.int32)
            sg_symelem_frame444_compids_dict[sym] = -np.ones((len(stdframes), nops), dtype=np.int32)
            sg_symelem_frame444_opcompids_dict[sym] = -np.ones((len(stdframes), nops, nops), dtype=np.int32)
            for ielem, unitelem in enumerate(sg_symelem_dict[sym]):
                elem = unitelem.tolattice(latticevec)
                if not (elem.iscyclic or elem.isdihedral):
                    continue
                # print(sym, elem, flush=True)
                sg_symelem_frame444_opids_dict[sym][:, ielem] = elem.frame_operator_ids(stdframes)

                try:
                    sg_symelem_frame444_compids_dict[sym][:, ielem] = elem.frame_component_ids(stdframes, perms)
                except ComponentIDError:
                    print('!' * 80)
                    print('ERROR making component ids for symelem', sym, ielem)
                    print(elem)
                    print('probably not all SymElem operators contained in 2x2x2 cells')
                    print('this remains mysterious')
                    elem.issues.append('This element breaks standard component id system')
                    sg_symelem_frame444_compids_dict[sym][:, ielem] = IERROR
                    for jelem, se2 in enumerate(sg_symelem_dict[sym]):
                        sg_symelem_frame444_opcompids_dict[sym][:, ielem, jelem] = IERROR
                        IERROR += 1
                    # assert 0
                    continue

                for jelem, elem2 in enumerate(sg_symelem_dict[sym]):
                    fopid = sg_symelem_frame444_opids_dict[sym][:, ielem]
                    fcompid = sg_symelem_frame444_compids_dict[sym][:, jelem]
                    # if not elem.iscompound or not elem2.iscompound: continue
                    # ic(elem, elem2)
                    ids = fcompid.copy()
                    for ifopid in range(np.max(fopid)):
                        fcids = fcompid[fopid == ifopid]
                        idx0 = fcompid == fcids[0]
                        for fcid in fcids[1:]:
                            idx = fcompid == fcid
                            ids[idx] = min(min(ids[idx]), min(ids[idx0]))
                    for iid, id in enumerate(sorted(set(ids))):
                        ids[ids == id] = iid
                    sg_symelem_frame444_opcompids_dict[sym][:, ielem, jelem] = ids
                    try:
                        opcompids = _make_operator_component_joint_ids(elem, elem2, stdframes, fopid, fcompid)
                        assert np.allclose(opcompids, sg_symelem_frame444_opcompids_dict[sym][:, ielem, jelem])
                    except ComponentIDError:
                        print('!' * 80)
                        print('ERROR checking operator component joint ids for symelem', sym, ielem, jelem)
                        print('      could be issues with op/comp ids for elem pair:', sym, ielem, jelem)
                        print(elem)
                        print(elem2, flush=True)
                        # assert 0
                        continue

        if update:
            sgdata = dict(
                sg_frames_dict=sg_frames_dict,
                sg_cheshire_dict=sg_cheshire_dict,
                sg_symelem_dict=sg_symelem_dict,
                sg_permutations444_dict=sg_permutations444_dict,
                sg_symelem_frame444_opids_dict=sg_symelem_frame444_opids_dict,
                sg_symelem_frame444_compids_dict=sg_symelem_frame444_compids_dict,
                sg_symelem_frame444_opcompids_dict=sg_symelem_frame444_opcompids_dict,
            )
            ic('saving spacegroup data')
            save_package_data(sgdata, 'spacegroup_data.pickle')

    return sgdata

_sgdata = _get_spacegroup_data()
sg_frames_dict = _sgdata['sg_frames_dict']  # type: ignore
sg_cheshire_dict = _sgdata['sg_cheshire_dict']  # type: ignore
sg_symelem_dict = _sgdata['sg_symelem_dict']  # type: ignore
sg_permutations444_dict = _sgdata['sg_permutations444_dict']  # type: ignore
sg_symelem_frame444_opids_dict = _sgdata['sg_symelem_frame444_opids_dict']  # type: ignore
sg_symelem_frame444_compids_dict = _sgdata['sg_symelem_frame444_compids_dict']  # type: ignore
sg_symelem_frame444_opcompids_dict = _sgdata['sg_symelem_frame444_opcompids_dict']  # type: ignore
# def makedata2():
#   sg_symelem_frame444_opids_dict = dict()
#   sg_symelem_frame444_compids_sym_dict = dict()
#   sg_symelem_frame444_opcompids_dict = dict()
#   for k, v in sg_symelem_dict.items():
#      if sg_lattice[k] == 'CUBIC' and sg_is_chiral(k):
#
#         frames4 = latticeframes(sg_frames_dict[k], np.eye(3), 4)
#         nops = len(sg_symelem_dict[k])
#         perms = sg_permutations444_dict[k]
#         sg_symelem_frame444_opids_dict[k] = -np.ones((len(frames4), nops), dtype=np.int32)
#         sg_symelem_frame444_compids_dict[k] = -np.ones((len(frames4), nops), dtype=np.int32)
#         sg_symelem_frame444_opcompids_dict[k] = -np.ones((len(frames4), nops, nops), dtype=np.int32)
#         for ielem, se in enumerate(sg_symelem_dict[k]):
#            ic(k, ielem)
#            sg_symelem_frame444_opids_dict[k][:, ielem] = se.frame_operator_ids(frames4)
#            sg_symelem_frame444_compids_dict[k][:, ielem] = se.frame_component_ids(frames4, perms)
#            for jelem, se2 in enumerate(sg_symelem_dict[k]):
#               fopid = sg_symelem_frame444_opids_dict[k][:, ielem]
#               fcompid = sg_symelem_frame444_compids_dict[k][:, jelem]
#
#               ids = fcompid.copy()
#               for i in range(np.max(fopid)):
#                  fcids = fcompid[fopid == i]
#                  idx0 = fcompid == fcids[0]
#                  for fcid in fcids[1:]:
#                     idx = fcompid == fcid
#                     ids[idx] = min(min(ids[idx]), min(ids[idx0]))
#               for i, id in enumerate(sorted(set(ids))):
#                  ids[ids == id] = i
#
#               sg_symelem_frame444_opcompids_dict[k][:, ielem, jelem] = ids
#   sgdata = (
#      sg_frames_dict,
#      sg_cheshire_dict,
#      sg_symelem_dict,
#      sg_permutations444_dict,
#      sg_symelem_frame444_opids_dict,
#      sg_symelem_frame444_compids_dict,
#      sg_symelem_frame444_opcompids_dict,
#   )
#   save_package_data(sgdata, 'spacegroup_data')
#   assert 0
# makedata2()
