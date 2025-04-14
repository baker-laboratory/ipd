import sys
import argparse
import functools
import operator
import ipd
import ipd.homog.hgeom as h

bs = ipd.lazyimport('biotite.structure', 'bs')

def get_args(sysargv):
    """get command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, nargs=1)
    parser.add_argument('files', type=str, nargs='+')
    return parser.parse_args(sysargv[1:])

def main():
    args = get_args(sys.argv)
    if mode == 'abbas':
        return build_from_components_abbas(*args.files)
    raise ValueError(f"Unknown mode {args.mode}")

def build_from_components_abbas(atoms1: 'list[bs.AtomArray]', atoms2: 'list[bs.AtomArray]', tol=0.1, **kw):
    """
    this is currently bespoke for a case abbas had... would like to make more general
    """
    tol = ipd.dev.Tolerances(tol, **kw)
    rms, _, xfit = h.rmsfit(atoms2[0].coord, atoms1[0].coord)
    if rms > tol.rms_fit: return None
    for i, a2 in enumerate(atoms2):
        atoms2[i].coord = h.xform(xfit, a2.coord)
        atoms2[i].chain_id[:] = 'ABCDEFGHIJK'[i + len(atoms1)]

    sinfo1 = ipd.sym.syminfo_from_atomslist(atoms1, tol=tol, **kw)
    sinfo2 = ipd.sym.syminfo_from_atomslist(atoms2, tol=tol, **kw)
    se1, se2 = sinfo1.symelem, sinfo2.symelem

    p1, p2 = h.line_line_closest_points_pa(se1.cen, se1.axis, se2.cen, se2.axis)
    cen = (p1+p2) / 2
    axes = ipd.sym.axes('I')

    joint = functools.reduce(operator.add, atoms1 + atoms2[1:])
    joint.coord -= cen[0, :3]
    x = h.halign2(se1.axis[0], se2.axis[0], axes[3], axes[5])
    joint.coord = h.xform(x, joint.coord)

    return joint

if __name__ == '__main__':  # ignore
    main()
