import itertools
import numpy as np
import willutil as wu

try:
    import pymol
    from pymol import cgo, cmd
except ImportError:
    pass

_numcom = 0
_numvec = 0
_numray = 0
_numline = 0
_numseg = 0

def showcom(sel="all"):
    global _numcom
    cen = com(sel)
    print("Center of mass: ", c)
    mycgo = [
        cgo.COLOR,
        1.0,
        1.0,
        1.0,
        cgo.SPHERE,
        cen[0],
        cen[1],
        cen[2],
        1.0,
    ]  # white sphere with 3A radius
    pymol.cmd.load_cgo(mycgo, "com%i" % _numcom)
    _numcom += 1

def cgo_sphere(cen, rad=1, col=(1, 1, 1)):
    cen = wu.homog.hpoint(cen).reshape(-1, 4)
    # white sphere with 3A radius
    # ic(col)
    mycgo = [cgo.COLOR, col[0], col[1], col[2]]
    for c in cen:
        mycgo.extend([cgo.SPHERE, c[0], c[1], c[2], rad])
    return mycgo

def showsphere(cen, rad=1, col=(1, 1, 1), lbl=""):
    v = pymol.cmd.get_view()
    if not lbl:
        global _numvec
        lbl = "sphere%i" % _numvec
        _numvec += 1
    mycgo = cgo_sphere(cen=cen, rad=rad, col=col)
    pymol.cmd.load_cgo(mycgo, lbl)
    pymol.cmd.set_view(v)

def cgo_vecfrompoint(axis, cen, col=(1, 1, 1), lbl=""):
    OBJ = [
        cgo.BEGIN,
        cgo.LINES,
        cgo.COLOR,
        col[0],
        col[1],
        col[2],
        cgo.VERTEX,
        cen[0],
        cen[1],
        cen[2],
        cgo.VERTEX,
        cen[0] + axis[0],
        cen[1] + axis[1],
        cen[2] + axis[2],
        cgo.END,
    ]
    return OBJ

def showvecfrompoint(axis, cen, col=(1, 1, 1), lbl=""):
    if not lbl:
        global _numray
        lbl = "ray%i" % _numray
        _numray += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    mycgo = cgo_vecfrompoints(axis, cen, col)
    pymol.cmd.load_cgo(mycgo, lbl)
    # pymol.cmd.load_cgo([cgo.COLOR, col[0],col[1],col[2],
    #             cgo.SPHERE,   cen[0],       cen[1],       cen[2],    0.08,
    #             cgo.CYLINDER, cen[0],       cen[1],       cen[2],
    #                       cen[0] +axis[0], cen[1] +axis[1], cen[2] +axis[2], 0.02,
    #               col[0],col[1],col[2],col[0],col[1],col[2],], lbl)
    pymol.cmd.set_view(v)

def cgo_segment(c1, c2, col=(1, 1, 1)):
    OBJ = [
        cgo.BEGIN,
        cgo.LINES,
        cgo.COLOR,
        col[0],
        col[1],
        col[2],
        cgo.VERTEX,
        c1[0],
        c1[1],
        c1[2],
        cgo.VERTEX,
        c2[0],
        c2[1],
        c2[2],
        cgo.END,
    ]
    # pymol.cmd.load_cgo([cgo.COLOR, col[0],col[1],col[2],
    #             cgo.CYLINDER, c1[0],     c1[1],     c1[2],
    #                           c2[0],     c2[1],     c2[2], 0.02,
    #               col[0],col[1],col[2],col[0],col[1],col[2],], lbl)
    return OBJ

def showsegment(c1, c2, col=(1, 1, 1), lbl=""):
    if not lbl:
        global _numseg
        lbl = "seg%i" % _numseg
        _numseg += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    pymol.cmd.load_cgo(cgo_segment(c1=c1, c2=c2, col=col), lbl)
    # pymol.cmd.load_cgo([cgo.COLOR, col[0],col[1],col[2],
    #             cgo.CYLINDER, c1[0],     c1[1],     c1[2],
    #                           c2[0],     c2[1],     c2[2], 0.02,
    #               col[0],col[1],col[2],col[0],col[1],col[2],], lbl)
    pymol.cmd.set_view(v)

def cgo_cyl(c1, c2, rad, col=(1, 1, 1), col2=None):
    col2 = col2 or col
    return [  # cgo.COLOR, col[0],col[1],col[2],
        9.0,  #        cgo.CYLINDER,
        float(c1[0]),
        float(c1[1]),
        float(c1[2]),
        float(c2[0]),
        float(c2[1]),
        float(c2[2]),
        rad,
        col[0],
        col[1],
        col[2],
        col2[0],
        col2[1],
        col2[2],
    ]

def showcyl(c1, c2, rad, col=(1, 1, 1), col2=None, lbl=""):
    if not lbl:
        global _numseg
        lbl = "seg%i" % _numseg
        _numseg += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    pymol.cmd.load_cgo(cgo_cyl(c1=c1, c2=c2, rad=rad, col=col, col2=col2), lbl)
    pymol.cmd.set_view(v)

def showline(axis, cen, col=(1, 1, 1), lbl="", oneside=False):
    if not lbl:
        global _numline
        lbl = "line%i" % _numline
        _numline += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    OBJ = [
        cgo.BEGIN,
        cgo.LINES,
        cgo.COLOR,
        col[0],
        col[1],
        col[2],
        cgo.VERTEX,
        cen[0] if oneside else cen[0] - axis[0],
        cen[1] if oneside else cen[1] - axis[1],
        cen[2] if oneside else cenc[2] - axis[2],
        cgo.VERTEX,
        cen[0] + axis[0],
        cen[1] + axis[1],
        cen[2] + axis[2],
        cgo.END,
    ]
    pymol.cmd.load_cgo(OBJ, lbl)
    pymol.cmd.set_view(v)

def cgo_lineabs(axis, cen, col=(1, 1, 1)):
    return [
        cgo.BEGIN,
        cgo.LINES,
        cgo.COLOR,
        col[0],
        col[1],
        col[2],
        cgo.VERTEX,
        cen[0],
        cen[1],
        cen[2],
        cgo.VERTEX,
        axis[0],
        axis[1],
        axis[2],
        cgo.END,
    ]

def showlineabs(axis, cen, col=(1, 1, 1), lbl=""):
    if not lbl:
        global _numline
        lbl = "line%i" % _numline
        _numline += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    mycgo = cgo_lineabs(axis, cen, col)
    pymol.cmd.load_cgo(mycgo, lbl)
    pymol.cmd.set_view(v)

def cgo_fan(
    axis,
    cen,
    rad,
    arc,
    col=(1, 1, 1),
    col2=None,
    startpoint=[1, 2, 3, 1],
    thickness=0.0,
    showstart=True,
    randspread=0,
    fanshift=0.0,
    ntri=50,
):
    if arc > 10:
        arc = np.radians(arc)
    col2 = col2 or col
    rot = wu.homog.hrot(axis, arc / (ntri + 0), cen)

    # ic(startpoint - cen)
    dirn = wu.homog.hprojperp(axis, startpoint - cen)
    dirn = wu.homog.hnormalized(dirn)
    cen = cen + fanshift * axis
    pt1 = cen + dirn * rad  # - thickness * axis * 0.5

    shift = randspread * (np.random.rand() - 0.5) * axis
    cen += shift
    pt1 += shift

    obj = []
    # obj += cgo_sphere(startpoint, 0.1)
    # obj += cgo_sphere(pt1, 0.1, col)

    for i in range(ntri):
        # yapf: disable
        pt2 = rot @ pt1
        if i%2 == 0:
            pt2 += thickness* axis
        else:
            pt2 -= thickness* axis
        obj += [
             cgo.BEGIN,
             cgo.TRIANGLES,
             cgo.COLOR,    col[0],  col[1],  col[2],
             cgo.ALPHA, 1,
             cgo.NORMAL,  axis[0], axis[1], axis[2],
             cgo.VERTEX,  cen [0], cen [1], cen [2],
             cgo.NORMAL,  axis[0], axis[1], axis[2],
             cgo.VERTEX,  pt1 [0], pt1 [1], pt1 [2],
             cgo.NORMAL,  axis[0], axis[1], axis[2],
             cgo.VERTEX,  pt2 [0], pt2 [1], pt2 [2],
             cgo.END,
          ]



        pt1 = pt2

        # yapf: enable
    return obj

def showfan(axis, cen, rad, arc, col=(1, 1, 1), lbl="", **kw):
    if not lbl:
        global _numseg
        lbl = "seg%i" % _numseg
        _numseg += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    pymol.cmd.load_cgo(cgo_fan(axis=axis, cen=cen, rad=rad, arc=arc, col=col, **kw), lbl)
    pymol.cmd.set_view(v)

def showaxes():
    v = pymol.cmd.get_view()
    obj = [
        cgo.BEGIN, cgo.LINES, cgo.COLOR, 1.0, 0.0, 0.0, cgo.VERTEX, 0.0, 0.0, 0.0, cgo.VERTEX, 20.0, 0.0, 0.0,
        cgo.COLOR, 0.0, 1.0, 0.0, cgo.VERTEX, 0.0, 0.0, 0.0, cgo.VERTEX, 0.0, 20.0, 0.0, cgo.COLOR, 0.0, 0.0,
        1.0, cgo.VERTEX, 0.0, 0.0, 0.0, cgo.VERTEX, 00, 0.0, 20.0, cgo.END
    ]
    pymol.cmd.load_cgo(obj, "axes")

def cgo_cyl_arrow(c1, c2, r, col=(1, 1, 1), col2=None, arrowlen=4.0):
    c1, c2 = wu.hpoint(c1), wu.hpoint(c2)
    if not col2:
        col2 = col
    CGO = []
    # c1.round0()
    # c2.round0()
    CGO.extend(cgo_cyl(c1, c2, rad=r, col=col, col2=col2))
    dirn = wu.hnormalized(c2 - c1)
    perp = wu.hnormalized(wu.hprojperp(dirn, [0.2340790923, 0.96794275, 0.52037438472304783]))
    if arrowlen > 0:
        arrow1 = c2 - dirn * arrowlen + perp * 2.0
        arrow2 = c2 - dirn * arrowlen - perp * 2.0
        # -dirn to shift to sphere surf
        CGO.extend(cgo_cyl(c2 - dirn * 3.0, arrow1 - dirn * 3.0, rad=r, col=col2))
        # -dirn to shift to sphere surf
        CGO.extend(cgo_cyl(c2 - dirn * 3.0, arrow2 - dirn * 3.0, rad=r, col=col2))
    return CGO

def showcube(*args, **kw):
    cmd.delete("CUBE")
    v = cmd.get_view()
    mycgo = cgo_cube(*args, **kw)
    cmd.load_cgo(mycgo, "CUBE")
    cmd.set_view(v)

def showcell(*args, **kw):
    cmd.delete("CELL")
    v = cmd.get_view()
    mycgo = cgo_cell(*args, **kw)
    cmd.load_cgo(mycgo, "CELL")
    cmd.set_view(v)

def cgo_cell(lattice, r=0.03):
    ic(lattice)
    lattice = lattice.T
    a = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        lattice[0],
        lattice[0],
        lattice[1],
        lattice[1],
        lattice[2],
        lattice[2],
        lattice[0] + lattice[1],
        lattice[0] + lattice[2],
        lattice[1] + lattice[2],
    ]
    b = [
        lattice[0],
        lattice[1],
        lattice[2],
        lattice[0] + lattice[1],
        lattice[0] + lattice[2],
        lattice[1] + lattice[0],
        lattice[1] + lattice[2],
        lattice[2] + lattice[0],
        lattice[2] + lattice[1],
        lattice[0] + lattice[1] + lattice[2],
        lattice[0] + lattice[1] + lattice[2],
        lattice[0] + lattice[1] + lattice[2],
    ]

    mycgo = [[cgo.CYLINDER, a[i][0], a[i][1], a[i][2], b[i][0], b[i][1], b[i][2], r, 1, 1, 1, 1, 1, 1]
             for i in range(len(a))]
    mycgo = list(itertools.chain(*mycgo))

    return mycgo

def cgo_cube(lb=[-10, -10, -10], ub=[10, 10, 10], r=0.03, xform=np.eye(4)):
    if isinstance(lb, (float, int)):
        lb = [lb] * 3
    if isinstance(ub, (float, int)):
        ub = [ub] * 3
    a = [
        wu.homog.hxform(xform, [ub[0], ub[1], ub[2]]),
        wu.homog.hxform(xform, [ub[0], ub[1], lb[2]]),
        wu.homog.hxform(xform, [ub[0], lb[1], lb[2]]),
        wu.homog.hxform(xform, [ub[0], lb[1], ub[2]]),
        wu.homog.hxform(xform, [lb[0], ub[1], ub[2]]),
        wu.homog.hxform(xform, [lb[0], ub[1], lb[2]]),
        wu.homog.hxform(xform, [lb[0], lb[1], lb[2]]),
        wu.homog.hxform(xform, [lb[0], lb[1], ub[2]]),
        wu.homog.hxform(xform, [lb[0], ub[1], ub[2]]),
        wu.homog.hxform(xform, [lb[0], ub[1], lb[2]]),
        wu.homog.hxform(xform, [lb[0], lb[1], ub[2]]),
        wu.homog.hxform(xform, [lb[0], lb[1], lb[2]]),
    ]
    b = [
        wu.homog.hxform(xform, [ub[0], ub[1], lb[2]]),
        wu.homog.hxform(xform, [ub[0], lb[1], lb[2]]),
        wu.homog.hxform(xform, [ub[0], lb[1], ub[2]]),
        wu.homog.hxform(xform, [ub[0], ub[1], ub[2]]),
        wu.homog.hxform(xform, [lb[0], ub[1], lb[2]]),
        wu.homog.hxform(xform, [lb[0], lb[1], lb[2]]),
        wu.homog.hxform(xform, [lb[0], lb[1], ub[2]]),
        wu.homog.hxform(xform, [lb[0], ub[1], ub[2]]),
        wu.homog.hxform(xform, [ub[0], ub[1], ub[2]]),
        wu.homog.hxform(xform, [ub[0], ub[1], lb[2]]),
        wu.homog.hxform(xform, [ub[0], lb[1], ub[2]]),
        wu.homog.hxform(xform, [ub[0], lb[1], lb[2]]),
    ]
    mycgo = [[cgo.CYLINDER, a[i][0], a[i][1], a[i][2], b[i][0], b[i][1], b[i][2], r, 1, 1, 1, 1, 1, 1]
             for i in range(len(a))]
    mycgo = list(itertools.chain(*mycgo))

    # yapf: disable
    #   l=10#*sqrt(3)
    #   m=-l
    #   mycgo += [
    #      cgo.CYLINDER, 0,0,0, l, 0, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, l, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, 0, l, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, m, 0, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, m, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, 0, m, r, 1, 1, 1, 1, 1, 1,
    #
    #      cgo.CYLINDER, 0,0,0, l, l, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, l, l, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, l, 0, l, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, m, l, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, m, l, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, m, 0, l, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, m, m, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, m, m, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, m, 0, m, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, l, m, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, l, m, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, l, 0, m, r, 1, 1, 1, 1, 1, 1,
    #
    #   ]
    # yapf: enable
    return mycgo
