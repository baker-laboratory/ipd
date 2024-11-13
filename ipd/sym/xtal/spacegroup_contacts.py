import itertools

import numpy as np

import ipd

def check_if_symelems_complete(spacegroup, symelems, depth=60, radius=5, trials=100000, fudgefactor=0.9):
    latticevec = ipd.sym.xtal.lattice_vectors(spacegroup, "nonsingular")

    generators = list()
    for unitelem in symelems:
        elem = unitelem.tolattice(latticevec)
        ops = elem.make_operators_screw()
        generators.append(ops)
        # ipd.showme(ipd.homog.htrans([0.02, 0.03, 0.04]) @ ops)
    generators = np.concatenate(generators)

    frames, _ = ipd.homog.hcom.geom.expand_xforms_rand(generators, depth=depth, radius=radius, trials=trials)
    frames = ipd.sym.tounitcell(latticevec, frames)
    x, y, z = frames[:, :3, 3].T
    nunitcell = np.sum((-0.001 < x) * (x < 1.999) * (-0.001 < y) * (y < 1.999) * (-0.001 < z) * (z < 1.999))
    nunitcell_target = 8 * ipd.sym.copies_per_cell(spacegroup)

    # ic(nunitcell, nunitcell_target)
    if nunitcell >= nunitcell_target * fudgefactor:
        # print('\t'.join(e.label for e in symelems))
        # for e in symelems:
        # print(e)
        # print(frames.shape, nunitcell, nunitcell_target)
        return True
    else:
        return False

def minimal_spacegroup_cover_symelems(spacegroup, maxelems=5, noscrew=False, nocompound=False):
    allsymelems = ipd.sym.xtal.symelems(spacegroup)
    if noscrew:
        allsymelems = [e for e in allsymelems if not e.isscrew]
    if nocompound:
        allsymelems = [e for e in allsymelems if not e.iscompound]
    max_combo = min(len(allsymelems), maxelems)
    ipd.printheader("compute covers", spacegroup, len(allsymelems), max_combo, padstart=10)

    generators = list()
    for ncombo in range(2, max_combo + 1):
        for combo in itertools.product(*[list(range(len(allsymelems)))] * ncombo):
            if any([combo[i] >= combo[i + 1] for i in range(ncombo - 1)]):
                continue
            genelems = [allsymelems[i] for i in combo]
            # ic(genelems)
            complete = check_if_symelems_complete(spacegroup, genelems)
            if complete:
                generators.append(genelems)

                # if all([
                # len(genelems) < 3,
                # not all([e.isscrew for e in genelems]),
                # not any([e.iscompound for e in genelems]),
                # ]):
                for se in genelems:
                    print(se)
                print()
                # else:
                # print(len(genelems))

        if generators:
            break

    return generators
