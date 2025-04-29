import sys
import argparse
import numpy as np
import ipd
import ipd.homog.hgeom as h

bs = ipd.lazyimport('biotite.structure', 'bs')

def get_args(sysargv):
    """get command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    parser.add_argument('files', type=str, nargs='+')
    parser.add_argument('-o', '--output', type=str, default='out.pdb')
    return parser.parse_args(sysargv[1:])

def main():
    args = get_args(sys.argv)
    if args.mode == 'abbas':
        return build_from_components_abbas(args.files, output=args.output)
    raise ValueError(f"Unknown mode {args.mode}")

def get_component_syminfo(fname, atoms, tol, **kw):
    sinfo = ipd.sym.syminfo_from_atomslist(atoms, tol=tol, **kw)
    junk = []
    if isinstance(sinfo, list):
        junk = [s for s in sinfo if s.symid == 'C1']
        sinfo = [s for s in sinfo if s.symid != 'C1']
        assert len(sinfo) == 1, f'more than one component in {fname}'
        sinfo = sinfo[0]
    print(f'------------------------ {fname} ---------------------------')
    print(sinfo)
    for j in junk:
        print(f'mysterious extra junk in component {fname}:')
        print('   ', j.symelem)
    return sinfo

def build_from_components_abbas(files, output=None, tol=0.1, **kw):
    """
    this is currently bespoke for a case abbas had... would like to make more general
    """
    fname1, fname2 = files
    atoms1, atoms2 = (ipd.atom.load(f, chainlist=True) for f in [fname1, fname2])
    tol = ipd.dev.Tolerances(tol, **kw)
    rms, _, xfit = h.rmsfit(atoms2[0].coord, atoms1[0].coord)
    if rms > tol.rms_fit: return None
    for i, a2 in enumerate(atoms2):
        atoms2[i].coord = h.xform(xfit, a2.coord)
        atoms2[i].chain_id[:] = 'ABCDEFGHIJK'[i + len(atoms1)] # type:ignore

    sinfo1 = get_component_syminfo(fname1, atoms1, tol=tol, **kw)
    sinfo2 = get_component_syminfo(fname2, atoms2, tol=tol, **kw)
    se1, se2 = sinfo1.symelem, sinfo2.symelem
    print(se1.nfold[0], se2.nfold[0], 'shold be 3 5')

    p1, p2 = h.line_line_closest_points_pa(se1.cen, se1.axis, se2.cen, se2.axis)
    cen = (p1+p2) / 2
    axes = ipd.sym.axes('I')
    joint = ipd.atom.join(atoms1 + atoms2[1:])
    joint.coord -= cen[0, :3]
    seang = h.angle_degrees(se1.axis[0], se2.axis[0])
    if seang > 90: se2.axis[0] = -se2.axis[0]
    seang = h.angle_degrees(se1.axis[0], se2.axis[0])
    symang = h.angle_degrees(axes[3], axes[5])
    x = h.halign2(se1.axis[0], se2.axis[0], axes[3], axes[5])
    print('components angle:', seang)
    print('target angle:', symang)
    print(
        np.array([
            [se1.axis[0], se2.axis[0]],
            [axes[3], axes[5]],
            [h.xform(x, se1.axis[0]), h.xform(x, se2.axis[0])],
        ]).swapaxes(0, 1))
    joint = h.xform(x, joint)

    if output:
        print('dumping to:', output)
        output, ext = output.rsplit('.', 1)
        ipd.atom.dump(joint, f'{output}_components.{ext}')
        asu = ipd.atom.Body(joint[joint.chain_id == 'A'])
        sym = ipd.atom.SymBody(asu, ipd.sym.frames('I'))
        sym.dump(f'{output}_icos.{ext}')

    return joint

if __name__ == '__main__':  # ignore
    main()
