import functools
import operator
import ipd
import ipd.homog.hgeom as h

# @dataclass
# class SymBuild:
#     symid: str
#     syminfo: ipd.sym.SymInfo
#     component_syminfo: list[ipd.sym.SymInfo]

def build_from_components_abbas(atoms1: 'list[AtomArray]', atoms2: 'list[AtomArray]', tol=0.1, **kw):
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
    se1, se2 = sinfo1.symelem.sel(index=0), sinfo2.symelem.sel(index=0)

    p1, p2 = h.line_line_closest_points_pa(se1.cen, se1.axis, se2.cen, se2.axis)
    cen = (p1+p2) / 2
    axes = ipd.sym.axes('I')

    joint = functools.reduce(operator.add, atoms1 + atoms2[1:])
    joint.coord -= cen[:3]
    x = h.halign2(se1.axis, se2.axis, axes[3], axes[5])
    joint.coord = h.xform(x, joint.coord)

    return joint
    # joint = atoms1 + atoms2
    # sinfo = ipd.sym.syminfo_from_atoms(joint, tol=tol, **kw)
    # ipd.icv(sinfo.symid)
    # ipd.icv(sinfo.symcen)
    # ipd.icv(sinfo.symelem)
    # return SymBuild('none', None, [sinfo1, sinfo2])
