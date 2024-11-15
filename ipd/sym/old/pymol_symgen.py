import functools
import inspect
import itertools
import os
import re
import sys

from ipd.sym.pymol_xyzmath import RAD, SYMOCT, SYMTET, Mat, Ux, Uz, Vec, Xform, isvec, projperp, randnorm  # type: ignore
try:
    from ipd.viz.pymol_cgo import cgo_cyl, cgo_sphere, cgo_segment  # noqa
    import pymol  # type: ignore
    from pymol import cmd  # type: ignore
except ImportError:
    pymol = None
    cmd = None

_symelem_nshow = 0

def hacky_xtal_maker(
    G,
    cell,
    depth=4,
    mindepth=0,
    symdef=1,
    shownodes=1,
    verbose=False,
    showcell=False,
    **kw,
):
    print("hacky_xtal_maker")
    v = cmd.get_view()  # type: ignore
    CEN = [g.cen for g in G]
    FN = list()
    tag = "test" if "tag" not in kw else kw["tag"]
    for d in range(mindepth, depth + 1):
        symtrie = generate_sym_trie(G, depth=d)
        # buildcgo = BuildCGO( nodes=[ CEN1+Vec(2,3,4), CEN2+Vec(2,4,3), ] )
        nodes = []
        if "component_pos" in list(kw.keys()):
            raise NotImplementedError("component_pos is no longer used")
            # nodes = kw["component_pos"][:1]
        buildcgo = BuildCGO(nodes=nodes, label=tag + "_DEPTH%i"%d, **kw)
        symtrie.visit(buildcgo)
        buildcgo.show()
        if shownodes:
            find_nodes = ComponentCenterVisitor(symelems=G, label=tag + "_NODES%i"%d, **kw)
            symtrie.visit(find_nodes)
            FN.append(find_nodes)
            if symdef:
                sdef_string = FN[-1].make_symdef(**kw)
                if verbose:
                    print("==================== SYMDEF (dump to " + tag + "_" + str(d) + ".sym) ====================")
                    print(sdef_string)
                    print("=====================================================================")
                with open(tag + "_" + str(d) + ".sym", "w") as out:
                    out.write(sdef_string)
                if "symdef_check" in kw and kw["symdef_check"]:
                    sdef = RosettaSymDef()
                    sdef.parse(sdef_string)
                    sdef.show("SYMDEF_" + tag)
    for fn in FN:
        fn.show(**kw)  # dumb order hack for pymol up/dn
    cmd.disable("all")  # type: ignore
    cmd.enable(tag + "_DEPTH%i" % (depth))  # type: ignore
    cmd.enable(tag + "_NODES%i" % (depth))  # type: ignore
    count = CountFrames()
    symtrie.visit(count)  # type: ignore
    if verbose:
        print("N Frames:", count.count)

    if showcell:
        cube(Vec(0, 0, 0), cell * Vec(1, 1, 1))  # type: ignore
    cmd.set_view(v)  # type: ignore

class PymolSymElem(object):
    """Docstring for PymolSymElem."""
    def __init__(
            self,
            kind,
            axis=Vec(0, 0, 1),
            cen=Vec(0, 0, 0),
            axis2=Vec(1, 0, 0),
            col=None,
            input_xform=None,
    ):
        super(PymolSymElem, self).__init__()
        self.kind = kind
        self.axis = axis.normalized()
        self.cen = cen
        self.axis2 = axis2.normalized()
        self.col = col
        self.frames = []
        self.input_xform = input_xform
        if self.kind.startswith("C"):
            assert not input_xform
            self.nfold = int(self.kind[1:])
            for i in range(self.nfold):
                deg = i * 360.0 / self.nfold
                self.frames.append(RAD(self.axis, deg, cen))
        elif self.kind.startswith("D"):
            assert not input_xform
            self.nfold = int(self.kind[1:])
            assert abs(axis.dot(axis2)) < 0.00001
            for i in range(self.nfold):
                deg = i * 360.0 / self.nfold
                cx = RAD(self.axis, deg, cen)
                self.frames.append(cx)
                self.frames.append(RAD(self.axis2, 180.0, cen) * cx)
        elif self.kind == "T":
            self.frames = [
                Xform(Mat(Vec(1, 0, 0), Vec(0, 1, 0), Vec(0, 0, 1)), Vec(0, 0, 0)),
                Xform(Mat(Vec(0, 0, 1), Vec(1, 0, 0), Vec(0, 1, 0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(0, 0, 1), Vec(-1, 0, 0), Vec(0, -1, 0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(0, 0, -1), Vec(1, 0, 0), Vec(0, -1, 0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(0, 0, -1), Vec(-1, 0, 0), Vec(0, 1, 0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(0, 1, 0), Vec(0, 0, 1), Vec(1, 0, 0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(0, 1, 0), Vec(0, 0, -1), Vec(-1, 0, 0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(0, -1, 0), Vec(0, 0, 1), Vec(-1, 0, 0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(0, -1, 0), Vec(0, 0, -1), Vec(1, 0, 0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(1, 0, 0), Vec(0, -1, 0), Vec(0, -0, -1)), Vec(0, 0, 0)),
                Xform(Mat(Vec(-1, 0, 0), Vec(0, 1, 0), Vec(0, 0, -1)), Vec(0, 0, 0)),
                Xform(Mat(Vec(-1, 0, 0), Vec(0, -1, 0), Vec(0, 0, 1)), Vec(0, 0, 0)),
            ]
            if input_xform:
                xc = Xform(cen) * input_xform
            else:
                xc = Xform(cen)
            for i, x in enumerate(self.frames):
                self.frames[i] = xc * x * (~xc)
        elif self.kind == "O":
            self.frames = [
                Xform(Mat(Vec(+1, +0, -0), Vec(+0, +1, +0), Vec(+0, -0, +1)), Vec(0, 0, 0)),
                Xform(Mat(Vec(+0, +1, +0), Vec(+1, +0, -0), Vec(-0, +0, -1)), Vec(0, 0, 0)),
                Xform(Mat(Vec(+0, -0, +1), Vec(+1, +0, -0), Vec(-0, +1, +0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(+1, +0, -0), Vec(+0, -0, +1), Vec(+0, -1, -0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(-0, +0, -1), Vec(+0, +1, +0), Vec(+1, -0, -0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(-0, +1, +0), Vec(+0, -0, +1), Vec(+1, +0, -0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(+0, +1, +0), Vec(-0, +0, -1), Vec(-1, +0, +0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(+0, -0, +1), Vec(-0, +1, +0), Vec(-1, -0, +0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(+0, -1, -0), Vec(+1, +0, +0), Vec(+0, -0, +1)), Vec(0, 0, 0)),
                Xform(Mat(Vec(+1, -0, -0), Vec(-0, +0, -1), Vec(+0, +1, +0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(+1, +0, +0), Vec(+0, -1, -0), Vec(-0, +0, -1)), Vec(0, 0, 0)),
                Xform(Mat(Vec(-0, +0, -1), Vec(+1, -0, -0), Vec(-0, -1, -0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(-1, +0, +0), Vec(+0, +1, -0), Vec(-0, +0, -1)), Vec(0, 0, 0)),
                Xform(Mat(Vec(-1, -0, +0), Vec(+0, +0, +1), Vec(-0, +1, +0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(+0, -0, +1), Vec(+0, -1, -0), Vec(+1, +0, +0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(+0, +1, -0), Vec(-1, +0, +0), Vec(+0, -0, +1)), Vec(0, 0, 0)),
                Xform(Mat(Vec(+0, +0, +1), Vec(-1, -0, +0), Vec(+0, -1, -0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(+0, -1, -0), Vec(-0, -0, +1), Vec(-1, -0, -0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(-0, -1, -0), Vec(-0, +0, -1), Vec(+1, -0, -0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(-0, +0, -1), Vec(-1, +0, +0), Vec(+0, +1, -0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(+0, +0, -1), Vec(-0, -1, -0), Vec(-1, +0, +0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(-1, +0, +0), Vec(-0, +0, -1), Vec(-0, -1, +0)), Vec(0, 0, 0)),
                Xform(Mat(Vec(-1, -0, -0), Vec(+0, -1, -0), Vec(-0, -0, +1)), Vec(0, 0, 0)),
                Xform(Mat(Vec(+0, -1, -0), Vec(-1, -0, -0), Vec(+0, +0, -1)), Vec(0, 0, 0)),
            ]
            if input_xform:
                xc = Xform(cen) * input_xform
            else:
                xc = Xform(cen)
            for i, x in enumerate(self.frames):
                self.frames[i] = (xc) * x * (~xc)
        assert self.frames
        if not self.frames[0] == Xform():
            if verbose:  # type: ignore
                print(self.kind, self.frames[0].pretty())
            assert self.frames[0] == Xform()

    def show(self, label=None, **kwargs):
        if not label:
            global _symelem_nshow
            label = "SymElem_%i" % _symelem_nshow
            _symelem_nshow += 1
        pymol.cmd.delete(label)  # type: ignore
        v = pymol.cmd.get_view()  # type: ignore
        CGO = self.cgo(**kwargs)
        pymol.cmd.load_cgo(CGO, label)  # type: ignore
        pymol.cmd.set_view(v)  # type: ignore

    def cgo(
        self,
        length=20.0,
        radius=0.5,
        vizsphereradius=1.5,
        col=None,
        showshape=0,
        **kwargs,
    ):
        if not col and self.col:
            col = self.col
        CGO = []
        x = Xform()
        if "xform" in kwargs:
            x = kwargs["xform"]
        if self.kind[0] in "CD":
            if not col:
                if self.kind == "C2":
                    col = (1.0, 0.0, 0.0)
                elif self.kind == "C3":
                    col = (0.0, 1.0, 0.0)
                elif self.kind == "C4":
                    col = (0.0, 0.0, 1.0)
                elif self.kind == "C5":
                    col = (1.0, 0.7, 0.8)
                elif self.kind == "C6":
                    col = (1.0, 1.0, 0.0)
                elif self.kind == "D2":
                    col = (1.0, 0.0, 1.0)
                elif self.kind == "D3":
                    col = (0.0, 1.0, 1.0)
                elif self.kind == "D4":
                    col = (1.0, 0.5, 0.0)
                elif self.kind == "D6":
                    col = (0.5, 1.0, 0.0)
                else:
                    raise NotImplementedError("unknown symelem kind " + self.kind)
            a = self.axis * length / 2.0
            c = self.cen
            c1, c2 = self.cen - a, self.cen + a
            c = x * c
            c1 = x * c1
            c2 = x * c2
            c.round0()
            c1.round0()
            c2.round0()
            CGO.extend([
                # pymol.cgo.BEGIN, pymol.cgo.LINES,
                # pymol.cgo.COLOR, col[0], col[1],col[2],
                # pymol.cgo.VERTEX, self.cen.x-a.x, self.cen.y-a.y, self.cen.z-a.z,
                # pymol.cgo.VERTEX, self.cen.x+a.x, self.cen.y+a.y, self.cen.z+a.z,
                # pymol.cgo.END,
                pymol.cgo.CYLINDER,  # type: ignore
                c1.x,
                c1.y,
                c1.z,
                c2.x,
                c2.y,
                c2.z,
                radius,
                # pymol.cgo.CYLINDER, 0,     50,    0,      0,    -50,     0, radius,
                col[0],
                col[1],
                col[2],
                col[0],
                col[1],
                col[2],
                # pymol.cgo.SPHERE,
                # c.x,
                # c.y,
                # c.z,
                # vizsphereradius
            ])
            if self.kind.startswith("D"):
                for i in range(self.nfold):
                    xtmp = self.frames[2 * i].R
                    if self.nfold == 2 and i == 1:
                        xtmp = RAD(self.axis, 90)
                    a = xtmp * self.axis2 * length / 2.0
                    c = self.cen
                    c1, c2 = self.cen - a, self.cen + a
                    if "xform" in kwargs:
                        x = kwargs["xform"]
                        c = x * c
                        c1 = x * c1
                        c2 = x * c2
                    c.round0()
                    c1.round0()
                    c2.round0()
                    r = radius if self.nfold == 2 else radius
                    CGO.extend([
                        pymol.cgo.CYLINDER,  # type: ignore
                        c1.x,
                        c1.y,
                        c1.z,
                        c2.x,
                        c2.y,
                        c2.z,
                        r,
                        col[0],
                        col[1],
                        col[2],
                        col[0],
                        col[1],
                        col[2],
                    ])
        elif self.kind == "T":
            cen = x * self.cen
            cen.round0()
            CGO.extend(cgo_sphere(cen, 1.6 * vizsphereradius, col=(0.5, 0.5, 1)))  # type: ignore
            seen2, seen3 = list(), list()
            for f in self.frames:
                c2b = x.R * f.R * (Vec(1, 0, 0).normalized() * length / 2.0)
                c3a = x.R * f.R * (-Vec(1, 1, 1).normalized() * length / 2.0)
                c3b = x.R * f.R * (Vec(1, 1, 1).normalized() * length / 2.0)
                c2b.round0()
                c3a.round0()
                c3b.round0()
                if c2b not in seen2:
                    CGO.extend(cgo_cyl(cen, cen + c2b, radius, col=(1, 0, 0)))  # type: ignore
                    seen2.append(c2b)
                if c3a not in seen3:
                    CGO.extend(cgo_cyl(cen + c3a, cen + c3b, radius, col=(0, 1, 0)))  # type: ignore
                    seen3.append(c3a)
                    seen3.append(c3b)
        elif self.kind == "O":
            cen = x * self.cen
            cen.round0()
            CGO.extend(cgo_sphere(cen, 1.6 * vizsphereradius, col=(0.5, 0.5, 1)))  # type: ignore
            seen2, seen3, seen4 = list(), list(), list()
            for f in self.frames:
                c2a = x.R * f.R * (-Vec(1, 1, 0).normalized() * length / 2.0)
                c2b = x.R * f.R * (Vec(1, 1, 0).normalized() * length / 2.0)
                c3a = x.R * f.R * (-Vec(1, 1, 1).normalized() * length / 2.0)
                c3b = x.R * f.R * (Vec(1, 1, 1).normalized() * length / 2.0)
                c4a = x.R * f.R * (-Vec(1, 0, 0).normalized() * length / 2.0)
                c4b = x.R * f.R * (Vec(1, 0, 0).normalized() * length / 2.0)
                c2a.round0()
                c2b.round0()
                c3a.round0()
                c3b.round0()
                c4a.round0()
                c4b.round0()
                if c2b not in seen2:
                    CGO.extend(cgo_cyl(cen + c2a, cen + c2b, radius, col=(1, 0, 0)))  # type: ignore
                    seen2.append(c2a)
                    seen2.append(c2b)
                if c3a not in seen3:
                    CGO.extend(cgo_cyl(cen + c3a, cen + c3b, radius, col=(0, 1, 0)))  # type: ignore
                    seen3.append(c3a)
                    seen3.append(c3b)
                if c4a not in seen4:
                    CGO.extend(cgo_cyl(cen + c4a, cen + c4b, radius, col=(0, 0, 1)))  # type: ignore
                    seen4.append(c4a)
                    seen4.append(c4b)
        if showshape:
            if self.kind == "C2":
                axs = x.R * self.axis
                cen = x * self.cen
                p1 = x.R * projperp(self.axis, Vec(7, 3, 1)).normalized() * 30.0
                p2 = RAD(axs, 180.0) * p1
                p3 = x.R * projperp(self.axis, Vec(1, 3, 7)).normalized() * 30.0
                p4 = RAD(axs, 180.0) * p3
                p1 = cen + p1
                p2 = cen + p2
                p3 = cen + p3
                p4 = cen + p4
                axs.round0()
                p1.round0()
                p2.round0()
                p3.round0()
                p4.round0()
                # if verbose: print p1,p2,p3,p4
                CGO.extend([
                    pymol.cgo.BEGIN,  # type: ignore
                    pymol.cgo.TRIANGLES,  # type: ignore
                    pymol.cgo.COLOR,  # type: ignore
                    col[0],  # type: ignore
                    col[1],  # type: ignore
                    col[2],  # type: ignore
                    pymol.cgo.ALPHA,  # type: ignore
                    1,
                    pymol.cgo.NORMAL,  # type: ignore
                    axs.x,
                    axs.y,
                    axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p1.x + axs.x / 10.0,
                    p1.y + axs.y / 10.0,
                    p1.z + axs.z / 10.0,
                    pymol.cgo.NORMAL,  # type: ignore
                    axs.x,
                    axs.y,
                    axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p2.x + axs.x / 10.0,
                    p2.y + axs.y / 10.0,
                    p2.z + axs.z / 10.0,
                    pymol.cgo.NORMAL,  # type: ignore
                    axs.x,
                    axs.y,
                    axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p3.x + axs.x / 10.0,
                    p3.y + axs.y / 10.0,
                    p3.z + axs.z / 10.0,
                    pymol.cgo.NORMAL,  # type: ignore
                    axs.x,
                    axs.y,
                    axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p1.x + axs.x / 10.0,
                    p1.y + axs.y / 10.0,
                    p1.z + axs.z / 10.0,
                    pymol.cgo.NORMAL,  # type: ignore
                    axs.x,
                    axs.y,
                    axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p2.x + axs.x / 10.0,
                    p2.y + axs.y / 10.0,
                    p2.z + axs.z / 10.0,
                    pymol.cgo.NORMAL,  # type: ignore
                    axs.x,
                    axs.y,
                    axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p4.x + axs.x / 10.0,
                    p4.y + axs.y / 10.0,
                    p4.z + axs.z / 10.0,
                    pymol.cgo.COLOR,  # type: ignore
                    1 - col[0],  # type: ignore
                    1 - col[1],  # type: ignore
                    1 - col[2],  # type: ignore
                    pymol.cgo.NORMAL,  # type: ignore
                    -axs.x,
                    -axs.y,
                    -axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p1.x - axs.x / 10.0,
                    p1.y - axs.y / 10.0,
                    p1.z - axs.z / 10.0,
                    pymol.cgo.NORMAL,  # type: ignore
                    -axs.x,
                    -axs.y,
                    -axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p2.x - axs.x / 10.0,
                    p2.y - axs.y / 10.0,
                    p2.z - axs.z / 10.0,
                    pymol.cgo.NORMAL,  # type: ignore
                    -axs.x,
                    -axs.y,
                    -axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p3.x - axs.x / 10.0,
                    p3.y - axs.y / 10.0,
                    p3.z - axs.z / 10.0,
                    pymol.cgo.NORMAL,  # type: ignore
                    -axs.x,
                    -axs.y,
                    -axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p1.x - axs.x / 10.0,
                    p1.y - axs.y / 10.0,
                    p1.z - axs.z / 10.0,
                    pymol.cgo.NORMAL,  # type: ignore
                    -axs.x,
                    -axs.y,
                    -axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p2.x - axs.x / 10.0,
                    p2.y - axs.y / 10.0,
                    p2.z - axs.z / 10.0,
                    pymol.cgo.NORMAL,  # type: ignore
                    -axs.x,
                    -axs.y,
                    -axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p4.x - axs.x / 10.0,
                    p4.y - axs.y / 10.0,
                    p4.z - axs.z / 10.0,
                    pymol.cgo.END,  # type: ignore
                ])
            if self.kind == "C3":
                axs = x.R * self.axis
                cen = x * self.cen
                p1 = projperp(axs, Vec(1, 2, 3)).normalized() * 35.0
                p2 = RAD(axs, 120.0) * p1
                p3 = RAD(axs, 240.0) * p1
                p1 = cen + p1
                p2 = cen + p2
                p3 = cen + p3
                CGO.extend([
                    pymol.cgo.BEGIN,  # type: ignore
                    pymol.cgo.TRIANGLES,  # type: ignore
                    pymol.cgo.COLOR,  # type: ignore
                    col[0],  # type: ignore
                    col[1],  # type: ignore
                    col[2],  # type: ignore
                    pymol.cgo.ALPHA,  # type: ignore
                    1,
                    pymol.cgo.NORMAL,  # type: ignore
                    axs.x,
                    axs.y,
                    axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p1.x + axs.x / 10.0,
                    p1.y + axs.y / 10.0,
                    p1.z + axs.z / 10.0,
                    pymol.cgo.NORMAL,  # type: ignore
                    axs.x,
                    axs.y,
                    axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p2.x + axs.x / 10.0,
                    p2.y + axs.y / 10.0,
                    p2.z + axs.z / 10.0,
                    pymol.cgo.NORMAL,  # type: ignore
                    axs.x,
                    axs.y,
                    axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p3.x + axs.x / 10.0,
                    p3.y + axs.y / 10.0,
                    p3.z + axs.z / 10.0,
                    pymol.cgo.COLOR,  # type: ignore
                    1 - col[0],  # type: ignore
                    1 - col[1],  # type: ignore
                    1 - col[2],  # type: ignore
                    pymol.cgo.NORMAL,  # type: ignore
                    -axs.x,
                    -axs.y,
                    -axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p1.x - axs.x / 10.0,
                    p1.y - axs.y / 10.0,
                    p1.z - axs.z / 10.0,
                    pymol.cgo.NORMAL,  # type: ignore
                    -axs.x,
                    -axs.y,
                    -axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p2.x - axs.x / 10.0,
                    p2.y - axs.y / 10.0,
                    p2.z - axs.z / 10.0,
                    pymol.cgo.NORMAL,  # type: ignore
                    -axs.x,
                    -axs.y,
                    -axs.z,
                    pymol.cgo.VERTEX,  # type: ignore
                    p3.x - axs.x / 10.0,
                    p3.y - axs.y / 10.0,
                    p3.z - axs.z / 10.0,
                    pymol.cgo.END,  # type: ignore
                ])
        return CGO

    def __eq__(self, other):
        return (self.kind == other.kind and self.cen == other.cen and self.axis == other.axis
                and self.axis2 == other.axis2)

    def __str__(self):
        return self.kind + " " + str(self.axis)

class SymElemPosition(object):
    """Docstring for SymElemPosition."""
    def __init__(self, symelem, xform):
        super(SymElemPosition, self).__init__()
        self.symelem = symelem
        self.xform = xform

    def __eq__(self, other):
        return self.symelem == other.symelem and self.xform == other.xform

class SymElemGroupManager(object):
    """Docstring for SymElemGroupManager."""
    def __init__(self):
        super(SymElemGroupManager, self).__init__()
        self.node2elems = dict()
        self.elem2nodes = dict()

    def add_symelem(self, symelem, xform, node):
        xelem = SymElemPosition(symelem, xform)
        seenit = False
        for oldelem in list(self.elem2nodes.keys()):
            if oldelem == xelem:
                xelem = oldelem
                assert not seenit
                seenit = True
        if xelem not in list(self.elem2nodes.keys()):
            self.elem2nodes[xelem] = list()
        self.elem2nodes[xelem].append(node)
        if node not in list(self.node2elems.keys()):
            self.node2elems[node] = list()
        self.node2elem[node].append(xelem)  # type: ignore

class SymTrieNode(object):
    """Docstring for SymTrieNode."""
    def __init__(self, generators, ielem, iframe, depth, position):
        super(SymTrieNode, self).__init__()
        self.generators = generators
        self.ielem = ielem
        self.iframe = iframe
        self.depth = depth
        self.children = []
        self.parent = None
        self.position = position

    def add_child(self, node):
        self.children.append(node)

    def visit(self, visitor, depth=0, xform=Xform()):
        parentxform = xform
        if self.parent:
            if not self.parent.position == parentxform:
                if verbose:  # type: ignore
                    print("parentxform mismatch!", self.parent.position.pretty())
                if verbose:  # type: ignore
                    print("parentxform mismatch!", parentxform.pretty())
                assert self.parent.position == parentxform
        xform = parentxform * self.generators[self.ielem].frames[self.iframe]
        visitor(self, depth=depth, xform=xform, parentxform=parentxform)
        for c in self.children:
            c.visit(visitor, depth=depth + 1, xform=xform)

    def __str__(self):
        return "elem %2i frame %2i depth %2i nchild %2i" % (
            self.ielem,
            self.iframe,
            self.depth,
            len(self.children),
        )

class SymTrieSanityCheckVisitor(object):
    """Docstring for SymTrieSanityCheckVisitor."""
    def __init__(self):
        super(SymTrieSanityCheckVisitor, self).__init__()
        self.seenit = list()  # [ Xform() ]

    def __call__(self, STN, **kwargs):
        assert STN.position not in self.seenit
        self.seenit.append(STN.position)
        for c in STN.children:
            assert c.parent is STN
        if STN.parent:
            assert STN in STN.parent.children

def generate_sym_trie_recurse(
    generators,
    depth,
    opts,
    body,
    heads,
    newheads,
    igen,
    verbose=False,
):
    if depth < 1:
        return
    if verbose:
        print("GEN SYM TRIE", "depth", depth, "igen", igen, "heads", len(heads), "newheads", len(newheads))
    newnewheads = []
    # for ielem, elem in enumerate(generators):
    ielem = igen
    elem = generators[igen]
    for iframe, frame in enumerate(elem.frames):
        if iframe == 0:
            continue  # skip identity
        # if verbose: print "FRAME",frame.pretty()
        for head, xpos in itertools.chain(newheads, heads):
            candidate = xpos * frame
            # if verbose: print candidate.pretty()
            seenit = False
            for seennode, seenframe in itertools.chain(newnewheads, newheads, heads, body):
                if candidate == seenframe:
                    seenit = True
                    # assert ~candidate * seenframe == Xform()
                    break
            if not seenit:
                newhead = SymTrieNode(generators, ielem, iframe, head.depth + 1, position=candidate)
                head.add_child(newhead)
                newhead.parent = head
                newnewheads.append((newhead, candidate))
                # if verbose: print "NEWHEAD",candidate

    newheads.extend(newnewheads)

    if igen + 1 == len(generators):
        body.extend(heads)
        heads = newheads
        newheads = []

    # if verbose: print len(newheads),igen,len(generators)
    if depth > 1:  # and newheads:
        generate_sym_trie_recurse(generators, depth - 1, opts, body, heads, newheads, (igen+1) % len(generators))

def generate_sym_trie(generators, depth=10, opts=None, verbose=False):
    raise NotImplementedError("some bug needs to be fixed")
    if opts is None:
        opts = dict()
    if verbose:
        print("NEW SYM TRIE")
    root = SymTrieNode(generators, 0, 0, 0, Xform())
    heads = [
        (root, Xform()),
    ]
    body = list()
    newheads = list()
    generate_sym_trie_recurse(generators, depth, opts, body, heads, newheads, 0)
    sanitycheck = SymTrieSanityCheckVisitor()
    root.visit(sanitycheck)
    return root

##########################################################################
##########################################################################
####################################### move the above to SymTrie!!! #####
##########################################################################
##########################################################################

newpath = os.path.dirname(inspect.getfile(inspect.currentframe()))  # script directory  # type: ignore
if newpath not in sys.path:
    sys.path.append(newpath)
# from xyzMath import Vec, Mat, Xform, RAD, projperp, Ux, Uy, Uz
# from wills_pymol_crap.pymol_util import cgo_sphere, cgo_segment, cgo_cyl
# from SymTrie import PymolSymElem, SymTrieNode, generate_sym_trie

# run /Users/sheffler/pymol/una.py; make_d3oct("test*","o33*",depth=3)

def makesym(frames0, sele="all", newobj="MAKESYM", depth=None, maxrad=9e9, n=9e9, verbose=False):
    v = cmd.get_view()  # type: ignore
    cmd.delete(newobj)  # type: ignore
    sele = "((" + sele + ") and (not TMP_makesym_*))"
    selechains = cmd.get_chains(sele)  # type: ignore
    if verbose:
        print(selechains)
    if not depth:
        frames = frames0
    else:
        frames = expand_xforms(frames0, N=depth, maxrad=maxrad)  # type: ignore
    # order on COM transform dis
    cen = com(sele)  # type: ignore
    frames = sorted(frames, key=lambda x: cen.distance(x * cen))

    # make new objs
    for i, x in enumerate(frames):
        if i >= n:
            break
        # if verbose: print i, x.pretty()
        tmpname = "TMP_makesym_%i" % i
        cmd.create(tmpname, sele)  # type: ignore
        for j, c in enumerate(selechains):
            cmd.alter(tmpname + " and chain " + c, "chain='%s'" % ROSETTA_CHAINS[len(selechains) * i + j])  # type: ignore
        xform(tmpname, x)  # type: ignore
    cmd.create(newobj, "TMP_makesym_*")  # type: ignore
    cmd.delete("TMP_makesym_*")  # type: ignore
    cmd.set_view(v)  # type: ignore
    # util.cbc()

def makecx(sel="all", name="TMP", n=5, axis=Uz):
    if sel == "all":
        for i, o in enumerate(cmd.get_object_list()):  # type: ignore
            makecx(sel=o, name="TMP%i" % i, n=n, axis=axis)
        return
    v = cmd.get_view()  # type: ignore
    cmd.delete("TMP__C%i_*" % n)  # type: ignore
    chains = ROSETTA_CHAINS  # type: ignore
    for i in range(n):
        cmd.create("TMP__C%i_%i" % (n, i), sel + " and (not TMP__C%i_*)"%n)  # type: ignore
    for i in range(n):
        rot("TMP__C%i_%i" % (n, i), axis, -360.0 * float(i) / float(n))  # type: ignore
    for i in range(n):
        cmd.alter("TMP__C%i_%i" % (n, i), "chain = '%s'" % chains[i])  # type: ignore
    util.cbc("TMP__C*")  # type: ignore
    # for i in range(n): cmd.alter("TMP__C%i_%i"%(n, i),"resi=str(int(resi)+%i)"%(1000*i));
    # util.cbc("TMP__C*")
    cmd.create(name, "TMP__*")  # type: ignore
    cmd.delete("TMP__*")  # type: ignore
    cmd.set_view(v)  # type: ignore
    cmd.disable(sel)  # type: ignore
    cmd.enable(name)  # type: ignore

def mofview():
    cmd.set("sphere_scale", 0.3)  # type: ignore
    cmd.hide("ev")  # type: ignore
    # cmd.show('sti')
    # cmd.show('lines', 'name n+ca+c')
    cmd.show("sti", "resn asp+das+cys+dcs+his+dhi+glu+dgu+zn+bpy and not hydro and not name n+c+o")  # type: ignore
    # cmd.show('sti', 'resn cys and name HG')
    cmd.show("sph", "name ZN")  # type: ignore
    # cmd.show('car')
    cmd.show("sti", "name n+ca+c+cb")  # type: ignore
    cmd.show("sph", "name cb and not resn asp+das+cys+dcs+his+dhi+glu+dgu")  # type: ignore
    # util.cbag('all')
    # cmd.color('green', 'name N')

    cmd.unbond("name zn", "all")  # type: ignore
    cmd.bond("name zn", "(not elem H+C) within 3 of name zn")  # type: ignore
    showline(Vec(-2, -1, 1) * 20, Vec(0, 0, 0))  # type: ignore
    showline(Vec(-1, -1, 0) * 20, Vec(0, 0, 0))  # type: ignore
    showaxes()  # type: ignore
    cmd.show("cgo")  # type: ignore
    # makec3(axis=Vec(1, 1, 1))
    # cmd.hide('sti')
    # util.cbag()
    # cmd.zoom()
    # cmd.show('lin')
    # cmd.show('sti', 'resn asp+das+cys+dcs+his+dhi+glu+dgu+zn')

if cmd is not None:
    cmd.extend("mofview", mofview)

def makedx(sel="all", n=2, name=None):
    if not name:
        name = sel.replace("+", "").replace(" ", "") + "_D%i"%n
    cmd.delete(name)  # type: ignore
    v = cmd.get_view()  # type: ignore
    cmd.delete("_TMP_D%i_*" % n)  # type: ignore
    ALLCHAIN = ROSETTA_CHAINS  # type: ignore
    chains = cmd.get_chains(sel)  # type: ignore
    for i in range(n):
        dsel = "_TMP_D%i_%i" % (n, i)
        dsel2 = "_TMP_D%i_%i" % (n, n + i)
        cmd.create(dsel, sel + " and (not _TMP_D%i_*)"%n)  # type: ignore
        rot(dsel, Uz, 360.0 * float(i) / float(n))  # type: ignore
        cmd.create(dsel2, dsel)  # type: ignore
        rot(dsel2, Ux, 180.0)  # type: ignore
        for ic, c in enumerate(chains):
            cmd.alter(  # type: ignore
                "((%s) and chain %s )" % (dsel, c),  # type: ignore
                "chain = '%s'" % ALLCHAIN[len(chains) * (i) + ic])  # type: ignore
            cmd.alter(  # type: ignore
                "((%s) and chain %s )" % (dsel2, c),  # type: ignore
                "chain = '%s'" % ALLCHAIN[len(chains) * (i+n) + ic])  # type: ignore
    cmd.create(name, "_TMP_D*")  # type: ignore
    util.cbc(name)  # type: ignore
    cmd.delete("_TMP_D*")  # type: ignore
    cmd.set_view(v)  # type: ignore
    cmd.disable(sel)  # type: ignore
    cmd.enable(name)  # type: ignore

for i in range(2, 21):
    globals()["makec%i" % i] = functools.partial(makecx, n=i)
for i in range(2, 21):
    globals()["maked%i" % i] = functools.partial(makedx, n=i)

def makecxauto():
    for o in cmd.get_object_list():  # type: ignore
        n = int(re.search("_C\d+_", o).group(0)[2:-1])  # type: ignore
        makecx(o, n)  # type: ignore

def maketet(sel="all", name="TET", n=12):
    makesym(frames0=SYMTET, sele=sel, newobj=name, n=n)

def makeoct(sel="all", name="OCT", n=24):
    makesym(frames0=SYMOCT, sele=sel, newobj=name, n=n)

def makeicos(sel="all", name="ICOS", n=60):
    makesym(frames0=SYMICS, sele=sel, newobj=name, n=n)  # type: ignore

def make_d3oct(d3, cage, cage_trimer_chain="A", depth=4, maxrad=9e9):
    if verbose:  # type: ignore
        print(
            cmd.super(  # type: ignore
                "((" + cage + ") and (chain " + cage_trimer_chain + "))",  # type: ignore
                "((" + d3 + ") and (chain A))"))  # type: ignore
    zcagecen = com(cage + " and name ca").z  # type: ignore
    if verbose:  # type: ignore
        print(zcagecen)
    # return
    x = alignvectors(Vec(1, 1, 1), Vec(1, -1, 0), Vec(0, 0, 1), Vec(1, 0, 0))  # type: ignore
    # if verbose: print x * Vec(1,1,1), x*Vec(1,-1,0)
    # RAD(Ux,180), RAD(Uy,120),
    G = [
        RAD(Ux, 180),
        RAD(Uz, 120),
        RAD(x * Vec(1, 0, 0), 90, Vec(0, 0, zcagecen)),
        RAD(x * Vec(1, 1, 0), 180, Vec(0, 0, zcagecen)),
    ]
    makesym(G, sele="((" + d3 + ") and ((chain A+B) and name CA))", depth=depth, maxrad=maxrad)
    cmd.show("sph", "MAKESYM")  # type: ignore
    # cmd.disable("all")
    cmd.enable("MAKESYM")  # type: ignore

def make_d3tet(d3, cage, cage_trimer_chain="A", depth=4, maxrad=9e9):
    if verbose:  # type: ignore
        print(
            cmd.super(  # type: ignore
                "((" + cage + ") and (chain " + cage_trimer_chain + "))",  # type: ignore
                "((" + d3 + ") and (chain A))"))  # type: ignore
    zcagecen = com(cage + " and name ca").z  # type: ignore
    if verbose:  # type: ignore
        print(zcagecen)
    # return
    x = alignvectors(Vec(1, 1, 1), Vec(1, -1, 0), Vec(0, 0, 1), Vec(1, 0, 0))  # type: ignore
    # if verbose: print x * Vec(1,1,1), x*Vec(1,-1,0)
    # RAD(Ux,180), RAD(Uy,120),
    G = [
        RAD(Ux, 180),
        RAD(Uz, 120),
        RAD(x * Vec(1, 0, 0), 180, Vec(0, 0, zcagecen)),
    ]
    makesym(G, sele="((" + d3 + ") and ((chain A+B) and name CA))", depth=depth, maxrad=maxrad)
    cmd.show("sph", "MAKESYM")  # type: ignore
    # cmd.disable("all")
    cmd.enable("MAKESYM")  # type: ignore

def print_node(node, **kwargs):
    if verbose:  # type: ignore
        print(kwargs["depth"] * "    ", node, kwargs["xform"].pretty())

def show_node(node, **kwargs):
    if node.iframe == 1:
        node.show(xform=kwargs["xform"])

class CountFrames(object):
    """Docstring for CountFrames."""
    def __init__(self):
        super(CountFrames, self).__init__()
        self.count = 0

    def __call__(self, *args, **kwkwargs):
        self.count += 1

def cgo_cyl_arrow(c1, c2, rad, col=(1, 1, 1), col2=None, arrowlen=4.0):
    if not col2:
        col2 = col
    CGO = []
    c1.round0()
    c2.round0()
    CGO.extend(cgo_cyl(c1, c2 + randnorm() * 0.0001, rad=rad, col=col, col2=col2))  # type: ignore
    dirn = (c2 - c1).normalized()
    perp = projperp(dirn, Vec(0.2340790923, 0.96794275, 0.52037438472304783)).normalized()
    arrow1 = c2 - dirn*arrowlen + perp*2.0
    arrow2 = c2 - dirn*arrowlen - perp*2.0
    # -dirn to shift to sphere surf
    CGO.extend(cgo_cyl(c2 - dirn*3.0, arrow1 - dirn*3.0, rad=rad, col=col2))  # type: ignore
    # -dirn to shift to sphere surf
    CGO.extend(cgo_cyl(c2 - dirn*3.0, arrow2 - dirn*3.0, rad=rad, col=col2))  # type: ignore
    return CGO

class BuildCGO(object):
    """Docstring for BuildCGO."""
    def __init__(
        self,
        nodes,
        maxrad=9e9,
        origin=Vec(0, 0, 0),
        bbox=[Vec(-9e9, -9e9, -9e9), Vec(9e9, 9e9, 9e9)],
        showlinks=False,
        showelems=True,
        label="BuildCGO",
        arrowlen=10.0,
        **kwargs,
    ):
        super(BuildCGO, self).__init__()
        self.nodes = nodes
        self.CGO = list()  # cgo_sphere(Vec(0, 0, 0), 3.0)
        self.jumps = set()
        self.maxrad = maxrad
        self.origin = origin
        self.bbox = bbox
        self.bbox[0].x, self.bbox[1].x = (
            min(self.bbox[0].x, self.bbox[1].x),
            max(self.bbox[0].x, self.bbox[1].x),
        )
        self.bbox[0].y, self.bbox[1].y = (
            min(self.bbox[0].y, self.bbox[1].y),
            max(self.bbox[0].y, self.bbox[1].y),
        )
        self.bbox[0].z, self.bbox[1].z = (
            min(self.bbox[0].z, self.bbox[1].z),
            max(self.bbox[0].z, self.bbox[1].z),
        )
        self.showlinks = showlinks
        self.showelems = showelems
        self.label = label
        self.arrowlen = arrowlen
        self.colors = list()
        self.colors = [
            (1, 1, 0),
            (0, 1, 1),
            (1, 0, 1),
            (0.5, 0.5, 0.5),
        ]
        self.kwargs = kwargs
        if self.bbox[0].x < self.bbox[1].x:
            self.bbox[0].x, self.bbox[1].x = self.bbox[1].x, self.bbox[0].x
        if self.bbox[0].y < self.bbox[1].y:
            self.bbox[0].y, self.bbox[1].y = self.bbox[1].y, self.bbox[0].y
        if self.bbox[0].z < self.bbox[1].z:
            self.bbox[0].z, self.bbox[1].z = self.bbox[1].z, self.bbox[0].z

    def bounds_check(self, v):
        # print('bounds check', v)
        if self.origin.distance(v) > self.maxrad:
            return False
        if not self.bbox[0].x <= v.x <= self.bbox[1].x:
            return False
        if not self.bbox[0].y <= v.y <= self.bbox[1].y:
            return False
        if not self.bbox[0].z <= v.z <= self.bbox[1].z:
            return False
        return True

    def __call__(self, node, **kwargs):
        x = kwargs["xform"]

        cencen = Vec(0, 0, 0)
        px = kwargs["parentxform"]
        if self.nodes:
            pcen = px * self.nodes[-1]
            # pcen = px*self.nodes[0]

        # show "nodes"
        for icen, cen in enumerate(self.nodes):
            xcen = x * cen
            if pcen:  # type: ignore
                self.jumps.add(pcen.distance(xcen))
            if self.showlinks:
                if self.bounds_check(pcen) or self.bounds_check(xcen):  # type: ignore
                    # if verbose: print "DEBUG",icen,px.pretty(),px==Xform()
                    if icen != 0 or node.parent:  # skip node 0 for root
                        self.add_segment(pcen, xcen, icen)  # type: ignore
            if self.bounds_check(xcen):
                self.add_sphere(x * (cen + Vec(0, 0, 0)), 2.0, text="%s%i" % ("ABCD"[icen], node.depth), icol=icen)
                self.add_sphere(x * (cen + Vec(2, 0, 0)), 2.0, text="%s%i" % ("ABCD"[icen], node.depth), icol=icen)
                self.add_sphere(x * (cen + Vec(0, 2, 0)), 2.0, text="%s%i" % ("ABCD"[icen], node.depth), icol=icen)
            pcen = xcen

        # show symelems
        if self.showelems:
            for elem in node.generators:
                if self.bounds_check(x * elem.cen):
                    mergeargs = dict(list(kwargs.items()) + list(self.kwargs.items()))
                    self.add_symelem(elem, x, **mergeargs)

    def add_symelem(self, elem, x, **kwargs):
        # should add duplicate checks here
        self.CGO.extend(elem.cgo(**kwargs))

    def add_sphere(self, cen, rad, text="", icol=0):
        # should add duplicate checks here
        cen.round0()
        if text:
            pos = [cen.x + 1.0, cen.y + 1.0, cen.z + 1.0]
            v = cmd.get_view()  # type: ignore
            axes = [[v[0], v[3], v[6]], [v[1], v[4], v[7]], [v[2], v[5], v[8]]]
            # pymol.cgo.wire_text(self.CGO,pymol.vfont.plain,pos,text,axes)
        self.CGO.extend(cgo_sphere(cen, rad, col=self.colors[icol]))  # type: ignore

    def add_segment(self, c1, c2, icol):
        # should add duplicate checks here
        if c1.distance(c2) < 1.0:
            return
        self.CGO.extend(cgo_cyl_arrow(c1, c2, rad=0.5, col=self.colors[max(0, icol - 1)], col2=self.colors[icol]))

    def show(self, verbose=False, **kwargs):
        v = cmd.get_view()  # type: ignore
        cmd.delete(self.label)  # type: ignore
        cmd.load_cgo(self.CGO, self.label)  # type: ignore
        cmd.set_view(v)  # type: ignore
        # for i,c in enumerate(self.nodes):
        # showsphere(c,1.5,col=self.colors[i])
        if verbose:
            print(self.jumps)

# TODO move to xyzMath

class VecDict(object):
    """Docstring for VecDict."""
    def __init__(self):
        super(VecDict, self).__init__()
        self.keys_ = list()
        self.values_ = list()

    def keys(self):
        return tuple(self.keys_)

    def values(self):
        return tuple(self.values_)

    def items(self):
        return zip(self.keys_, self.values_)

    def __getitem__(self, key):
        i = self.keys_.index(key)
        return self.values_[i]

    def __setitem__(self, key, val):
        try:
            i = self.keys_.index(key)
            self.values_[i] = val
        except ValueError:
            assert len(self.keys_) == len(self.values_)
            self.keys_.append(key)
            self.values_.append(val)

class ComponentCenterVisitor(object):
    """Docstring for ComponentCenterVisitor."""
    def __init__(self, symelems, extranodes=[], label="NODES", colors=list(), showlinks=1, **kwargs):
        super(ComponentCenterVisitor, self).__init__()
        # if len(symelems) > 2:
        #   raise NotImplementedError("num components > 2 not working yet... BFS fails")
        self.symelems = symelems
        CCs = [elem.cen for elem in symelems]
        CCs.extend(extranodes)
        self.primaryCCs = CCs
        self.label = label
        self.priCCtoCClist = VecDict()
        self.priCCtoCCframes = VecDict()
        for n in self.primaryCCs:
            assert isvec(n)
            self.priCCtoCClist[n] = [n]
            self.priCCtoCCframes[n] = VecDict()
        self.parentmap = None
        self.childmap = None
        self.colors = colors
        self.colors.extend(((1, 1, 0), (0, 1, 1), (1, 0, 1), (0.7, 0.7, 0.7)))
        self.showlinks = showlinks

    def __call__(self, sym_trie_node, xform, parentxform, **kwargs):
        assert list(self.priCCtoCCframes.keys()) == list(self.priCCtoCClist.keys())
        for priCC in list(self.priCCtoCClist.keys()):
            CC = xform * priCC
            CClist = self.priCCtoCClist[priCC]
            CCframes = self.priCCtoCCframes[priCC]
            if CC not in CClist:
                CClist.append(CC)
            if CC not in list(CCframes.keys()):
                CCframes[CC] = list()
            CCframes[CC].append(sym_trie_node)

    def makeCCtree(self, dhint=1.001):
        root = self.priCCtoCClist[self.primaryCCs[0]][0]
        self.parentmap = VecDict()
        self.parentmap[root] = None
        self.childmap = dict()
        visited, queue = list(), [root]
        lowest_d = 9e9
        while queue:
            CCparent = queue.pop(0)
            # if verbose: print "NEW VERTEX",CCparent
            if CCparent in visited:
                continue
            assert CCparent not in visited
            visited.append(CCparent)
            closedist, closest = 9e9, list()
            for priCC, CClist in list(self.priCCtoCClist.items()):
                for CC in CClist:
                    if CC in list(self.parentmap.keys()):
                        continue
                    d = CC.distance(CCparent)
                    if closedist - d > 0.01:
                        # if verbose: print "  new closedist",d
                        closest, closedist = list(), d
                    if abs(closedist - d) < 0.01:
                        closest.append(CC)
            lowest_d = min(lowest_d, closedist)
            for v in visited:
                assert v not in closest
            # if verbose: print "  ",CCparent,closedist,len(closest)
            if closedist > lowest_d * dhint:
                continue
            queue.extend(closest)
            for CC in closest:
                # assert CC not in list(self.parentmap.keys())
                self.parentmap[CC] = CCparent
                if CCparent not in self.childmap:
                    self.childmap[CCparent] = list()
                self.childmap[CCparent].append(CC)

    def check_jumps(self, verbose=False):
        if not self.parentmap:
            self.makeCCtree()
        jset = VecDict()
        for c, p in list(self.parentmap.items()):  # type: ignore
            if p:
                jset[c - p] = True
        jsetsort = VecDict()
        for k in list(jset.keys()):
            xyz = sorted((abs(round(k.x, 5)), abs(round(k.y, 5)), abs(round(k.z, 5))))
            # if verbose: print xyz
            jsetsort[Vec(xyz[0], xyz[1], xyz[2])] = True
        if verbose:
            print("UNIQUE PERMUTED JUMPS:")
        for k in list(jsetsort.keys()):
            if verbose:
                print("  ", k)

    def sanitycheck(self):
        if not self.parentmap:
            self.makeCCtree()
        self.check_jumps()
        assert list(self.priCCtoCCframes.keys()) == list(self.priCCtoCClist.keys())
        for priCC in self.primaryCCs:
            assert priCC in list(self.priCCtoCClist.keys())
        for priCC in list(self.priCCtoCClist.keys()):
            assert priCC in self.primaryCCs
        for priCC, CClist in list(self.priCCtoCClist.items()):
            for i1, n1 in enumerate(CClist):
                if n1 not in list(self.parentmap.keys()):  # type: ignore
                    pass
                    # if verbose: print("NOT IN PARENTMAP:", n1)
                for i2, n2 in enumerate(CClist):
                    assert (i1 == i2) == (n1 == n2)  # xor
        for priCC, CCframes in list(self.priCCtoCCframes.items()):
            for CC, STNs in list(CCframes.items()):
                for stn in STNs:
                    assert stn.position * priCC == CC

    def show(
            self,
            component_pos=(Vec(0, -4, 4), Vec(0, 3, 3), Vec(11, 9, 3), Vec(9, 3, 11)),
            showframes=True,
            **kwargs,
    ):
        self.sanitycheck()
        if not self.parentmap:
            self.makeCCtree()
        CGO = []
        for ipn, itms in enumerate(self.priCCtoCClist.items()):
            pn, CClist = itms
            for n in CClist:
                # CGO.extend(cgo_sphere(n, rad=2.2, col=self.colors[ipn]))
                # if n in list(self.parentmap.keys()) and self.parentmap[n] and self.showlinks:
                # CGO.extend(cgo_cyl_arrow(self.parentmap[n], n, rad=0.8, col=self.colors[ipn]))
                if showframes and ipn < len(component_pos):
                    for stn in self.priCCtoCCframes[pn][n]:
                        cn = stn.position * (pn + component_pos[ipn])
                        cnx = stn.position * (pn + component_pos[ipn] + Vec(3, 0, 0))
                        cny = stn.position * (pn + component_pos[ipn] + Vec(0, 2, 0))
                        CGO.extend(cgo_sphere(cn, rad=2.5, col=self.colors[ipn]))  # type: ignore
                        CGO.extend(cgo_sphere(cnx, rad=1.7, col=self.colors[ipn]))  # type: ignore
                        CGO.extend(cgo_sphere(cny, rad=1.2, col=self.colors[ipn]))  # type: ignore
                        if self.showlinks:
                            CGO.extend(cgo_cyl_arrow(n, cn, rad=0.3, col=self.colors[ipn], arrowlen=2.0))
        v = cmd.get_view()  # type: ignore
        cmd.delete(self.label)  # type: ignore
        cmd.load_cgo(CGO, self.label)  # type: ignore
        cmd.set_view(v)  # type: ignore

    def make_symdef(self, **kwargs):
        if "one_component" not in kwargs:
            kwargs["one_component"] = False

        XYZ_TEMPLATE = (r"xyz  %-30s  %+012.9f,%+012.9f,%+012.9f  %+012.9f,%+012.9f,%+012.9f  %+014.9f,%+014.9f,%+014.9f" +
                        "\n")
        if not self.parentmap:
            self.makeCCtree()
        scale = 1.0
        if "symdef_scale" in kwargs:
            scale = kwargs["symdef_scale"]
        # for k,v in self.parentmap.items(): if verbose: print "parentmap:",k,v
        node2num = VecDict()
        for ip, val in enumerate(self.priCCtoCCframes.items()):
            priCC, CCframes = val
            for icc, val2 in enumerate(CCframes.items()):
                CC, STNs = val2
                node2num[CC] = (ip, icc)
        Sxyz = ""
        edges = list()
        celldofjumps = list()
        compdofjumps = [list() for i in self.symelems]
        SUBAs = list()
        SUBs = [list() for i in self.symelems]
        for ip, val in enumerate(self.priCCtoCCframes.items()):
            if ip >= len(self.symelems):
                continue
            priCC, CCframes = val
            for icc, val2 in enumerate(CCframes.items()):
                Sxyz += "# virtuals for comp%i cen%i\n" % (ip, icc)
                CC, STNs = val2
                assert len(STNs) > 0
                PCC = None if CC not in list(self.parentmap.keys()) else self.parentmap[CC]  # type: ignore
                if PCC:
                    PCCName = "CMP%02i_CEN%03i" % node2num[PCC]
                if PCC:
                    PCCDofBegName = PCCName + "_CELLDofBeg_%i_%i" % (icc, ip)  # NOTE icc FIRST!!!!  # type: ignore
                if PCC:
                    PCCDofEndName = PCCName + "_CELLDofEnd_%i_%i" % (icc, ip)  # NOTE icc FIRST!!!  # type: ignore
                if True:
                    CCDofBegName = "CMP%02i_CEN%03i" % (ip, icc)
                if True:
                    CCDofEndName = "CMP%02i_CEN%03i_CmpDofEnd" % (ip, icc)
                # if True: CCDOFName  = "CMP%02i_CEN%03i_COMPDOF" %(ip,icc)
                if PCC:
                    DIR = (CC - PCC).normalized()
                if PCC:
                    DIR2 = projperp(DIR, Vec(1, 2, 3)).normalized()  # type: ignore
                if True:
                    ELEMDIR = STNs[0].position.R * self.symelems[ip].axis
                if True:
                    ELEMDIR2 = projperp(ELEMDIR, Vec(1, 2, 3)).normalized()
                if PCC:
                    Sxyz += XYZ_TEMPLATE % (
                        PCCDofBegName,  # type: ignore
                        DIR.x,  # type: ignore
                        DIR.y,  # type: ignore
                        DIR.z,  # type: ignore
                        DIR2.x,  # type: ignore
                        DIR2.y,  # type: ignore
                        DIR2.z,  # type: ignore
                        PCC.x * scale,
                        PCC.y * scale,
                        PCC.z * scale,
                    )
                if PCC:
                    Sxyz += XYZ_TEMPLATE % (
                        PCCDofEndName,  # type: ignore
                        DIR.x,  # type: ignore
                        DIR.y,  # type: ignore
                        DIR.z,  # type: ignore
                        DIR2.x,  # type: ignore
                        DIR2.y,  # type: ignore
                        DIR2.z,  # type: ignore
                        CC.x * scale,
                        CC.y * scale,
                        CC.z * scale,
                    )
                if True:
                    Sxyz += XYZ_TEMPLATE % (
                        CCDofBegName,
                        ELEMDIR.x,
                        ELEMDIR.y,
                        ELEMDIR.z,
                        ELEMDIR2.x,
                        ELEMDIR2.y,
                        ELEMDIR2.z,
                        CC.x * scale,
                        CC.y * scale,
                        CC.z * scale,
                    )
                if True:
                    Sxyz += XYZ_TEMPLATE % (
                        CCDofEndName,
                        ELEMDIR.x,
                        ELEMDIR.y,
                        ELEMDIR.z,
                        ELEMDIR2.x,
                        ELEMDIR2.y,
                        ELEMDIR2.z,
                        CC.x * scale,
                        CC.y * scale,
                        CC.z * scale,
                    )
                if PCC:
                    edges.append((PCCName, PCCDofBegName))  # type: ignore
                if PCC:
                    edges.append((PCCDofBegName, PCCDofEndName))  # type: ignore
                if PCC:
                    edges.append((PCCDofEndName, CCDofBegName))  # type: ignore
                if True:
                    edges.append((CCDofBegName, CCDofEndName))
                # if True: edges.append( (CCDofEndName, CCDOFName) )
                if PCC:
                    celldofjumps.append((PCCDofBegName, PCCDofEndName))  # type: ignore
                if True:
                    compdofjumps[ip].append((CCDofBegName, CCDofEndName))
                for isub, stn in enumerate(STNs):
                    SUBName = "CMP%02i_CEN%03i_SUB%03i" % (ip, icc, isub)
                    if ip == 0:
                        SUBAs.append(SUBName)
                    SUBs[ip].append((SUBName, stn.position))
                    # edges.append( (CCDOFName , SUBName) )
                    edges.append((CCDofEndName, SUBName))
                    # edges.append( (SUBName, "SUBUNIT %s"%("ABCDEFG"[ip]) ) )
                    SX = stn.position.R * Vec(1, 0, 0)
                    SY = stn.position.R * Vec(0, 1, 0)
                    SO = stn.position * priCC
                    Sxyz += XYZ_TEMPLATE % (
                        SUBName,
                        SX.x,
                        SX.y,
                        SX.z,
                        SY.x,
                        SY.y,
                        SY.z,
                        SO.x * scale,
                        SO.y * scale,
                        SO.z * scale,
                    )
                edges.append((None, None))  # spacer
                Sxyz += "\n"

        # dummy jumps, sometimes needed by dumb rosetta protocols
        Sxyz += "xyz DUMMY_VIRT 1,0,0 0,1,0 0,0,0\n\n"  # for stupid reasons

        celldofjumps = sorted(celldofjumps)

        s = "symmetry_name FUBAR\n\nsubunits %i\n\nnumber_of_interfaces %i\n\n" % (len(SUBAs), len(SUBAs) - 1)
        s += "E = 2*%s" % SUBAs[0]
        for suba in SUBAs[1:]:
            s += " + 1*(%s:%s)" % (SUBAs[0], suba)
        s += "\n\nanchor_residue COM\n\n"

        s += "#####################################################################################\n"
        s += "########################## Virtual Coordinate Frames ################################\n"
        s += "#####################################################################################\n\n"
        s += "virtual_coordinates_start\n\n"
        s += Sxyz
        s += "virtual_coordinates_stop\n\n"

        s += "#####################################################################################\n"
        s += "##################################### Jumps #########################################\n"
        s += "#####################################################################################\n\n"
        for p, c in edges:
            if p and c:
                s += "connect_virtual %-57s %-25s %-25s\n" % (
                    "JUMP__%s__to__%s" % (p, c.replace(" ", "")),
                    p,
                    c,
                )
            else:
                s += "\n"

        s += "################# SUBUNIT Jumps ############################\n\n"
        subunit_group_map = dict()
        for SUBname, x in SUBs[0]:
            p = SUBname
            c = "SUBUNIT A"
            if kwargs["one_component"]:
                c = "SUBUNIT"
            jname = "JUMP__%s__to__%s" % (p, c.replace(" ", ""))
            s += "connect_virtual %-57s %-25s %-25s\n" % (jname, p, c)
            if "A" not in subunit_group_map:
                subunit_group_map["A"] = list()
            subunit_group_map["A"].append(jname)
        s += "\n"
        for icomp, subs in enumerate(SUBs):
            if icomp == 0 or kwargs["one_component"]:
                continue
            for SUBname0, x0 in SUBs[0]:
                for SUBname, x in subs:
                    if x != x0:
                        continue
                    p = SUBname
                    chain = "ABCDEFG"[icomp]
                    c = "SUBUNIT " + chain
                    jname = "JUMP__%s__to__%s" % (p, c.replace(" ", ""))
                    # "# shares xform with primary "+SUBname0 )
                    s += "connect_virtual %-57s %-25s %-25s %s\n" % (jname, p, c, "")
                    if chain not in subunit_group_map:
                        subunit_group_map[chain] = list()
                    subunit_group_map[chain].append(jname)
            s += "\n"

        s += "#####################################################################################\n"
        s += "##################################### DOFs ##########################################\n"
        s += "#####################################################################################\n\n"

        s += "########################### stupid dummy DOF ################################\n\n"
        s += "connect_virtual DUMMY_JUMP CMP00_CEN000 DUMMY_VIRT\n"
        s += "set_dof DUMMY_JUMP x\n\n"

        if len(celldofjumps) > 0:
            s += "################# CELL DOFs ############################\n\n"
            s += "set_dof   JUMP__%s__to__%s   x\n\n" % celldofjumps[0]

        s += "################# COMPONENT DOFs ############################\n\n"
        for icomp, dofs in enumerate(compdofjumps):
            if not dofs:
                continue
            if icomp > 0 and kwargs["one_component"]:
                continue
            if self.symelems[icomp].kind[0] != "C":
                s += "#NOTCYCLIC# "
            s += "set_dof   JUMP__%s__to__%s   x angle_x\n\n" % dofs[0]

        s += "#####################################################################################\n"
        s += "################################## JUMP GROUPS ######################################\n"
        s += "#####################################################################################\n\n"

        s += "################# CELL DOF JUMP GROUPS ############################\n\n"

        s += "set_jump_group   GROUP_CELLDOF"
        for p, c in celldofjumps:
            s += "   JUMP__%s__to__%s" % (p, c)
        s += "\n\n"

        s += "################# COMPONENT DOF JUMP GROUPS ############################\n\n"

        for idof, dofs in enumerate(compdofjumps):
            if not dofs:
                continue
            if idof > 0 and kwargs["one_component"]:
                continue
            s += "set_jump_group   GROUP_COMP_%i_DOF" % idof
            for p, c in dofs:
                s += "   JUMP__%s__to__%s" % (p, c)
            s += "\n\n"

        s += "################# SUBUNIT JUMP GROUPS ############################\n\n"

        for chain, jumps in list(subunit_group_map.items()):
            if chain != "A" and kwargs["one_component"]:
                continue
            s += "set_jump_group GROUP_SUBUNT_%s" % chain
            for jump in sorted(jumps):
                s += "  %s" % jump
            s += "\n\n"

        # #################################################################################
        # # for debugging, make sure subunit jump names are in alphabetical order for Rosetta??
        # #################################################################################
        # for chain,jumps in subunit_group_map.items():
        #   for ijump,jump in enumerate(jumps):
        #       svname = jump.split("__")[1]
        #       s = s.replace(svname,"S%sO%03i_%s"%(chain,ijump,svname))

        if "generic_names" in kwargs and kwargs["generic_names"]:
            # rename some jumps
            s = s.replace("JUMP__%s__to__%s" % celldofjumps[0], "CELL_DOF")
            for icomp, dofs in enumerate(compdofjumps):
                if not dofs:
                    continue
                s = s.replace("JUMP__%s__to__%s" % dofs[0], "COMP_DOF_%i" % (icomp+1))

        return s

class RosettaSymDef(object):
    """Docstring for RosettaSymDef."""
    def __init__(self, virtuals=None, edges=None):
        super(RosettaSymDef, self).__init__()
        if not virtuals:
            virtuals = dict()
        if not edges:
            edges = dict()
        self.virtuals = virtuals
        self.edges = edges

    def add_virt(self, name, X, Y, O):
        self.virtuals[name] = (X.normalized(), Y.normalized(), O)

    def add_edge(self, name, v1name, v2name):
        self.edges[name] = (v1name, v2name)

    def parse(self, s):
        for line in s.split("\n"):
            if verbose:  # type: ignore
                print(line)
            if line.startswith("xyz"):
                dummy, name, X, Y, O = re.split(r"\s+", line.strip())
                X = X.split(",")
                X = Vec(float(X[0]), float(X[1]), float(X[2]))
                Y = Y.split(",")
                Y = Vec(float(Y[0]), float(Y[1]), float(Y[2]))
                O = O.split(",")
                O = Vec(float(O[0]), float(O[1]), float(O[2]))
                self.add_virt(name, X, Y, O)
            elif line.startswith("connect_virtual"):
                line = line.replace("SUBUNIT ", "SUBUNIT")
                splt = re.split(r"\s+", line.strip())
                # if verbose: print "SPLT:", splt
                dummy, name, v1name, v2name = splt[:4]
                self.add_edge(name, v1name, v2name)

    def displaycen(self, name, virt, offset=5.0):
        if re.match(".*DofBeg.*", name):
            return virt[2] + offset * virt[0]  # shift +X only
        elif re.match(".*DofEnd.*", name):
            return virt[2] - offset * virt[0]  # shift -X only
        return virt[2] + offset * virt[0] + offset * virt[1]

    def sanitycheck(self):
        for name, vnames in list(self.edges.items()):
            v1name, v2name = vnames
            v1 = self.virtuals[v1name]
            if not v2name.startswith("SUBUNIT"):
                v2 = self.virtuals[v2name]
                if v1[2].distance(v2[2]) < 0.0001:
                    continue  # overlapping virts, check makes no sense
                jdir = (v2[2] - v1[2]).normalized()
                v1x = v1[0].normalized()
                if jdir.distance(v1x) > 0.0001:
                    if verbose:  # type: ignore
                        print("connect_virtual ERROR", name, v1name, v2name)
                    if verbose:  # type: ignore
                        print("  jdir", jdir)
                    if verbose:  # type: ignore
                        print("  v1x ", v1x)
                    raise ValueError

    def show(self, tag="SYMDEF", XYlen=5.0, rad=3.0, **kwargs):
        self.sanitycheck()
        seenit = []
        CGO = []
        for name, xyo in list(self.virtuals.items()):
            X, Y, O = xyo
            # CGO.extend( cgo_sphere(c=O,r=r) )
            cen = self.displaycen(name, xyo, offset=XYlen)
            if cen in seenit:
                if verbose:  # type: ignore
                    print("ERROR overlapping display center", cen)
                # assert not cen in seenit
            seenit.append(cen)
            CGO.extend(cgo_lineabs(cen, cen + X*XYlen, col=(1, 0, 0)))  # type: ignore
            CGO.extend(cgo_lineabs(cen, cen + Y*XYlen, col=(0, 1, 0)))  # type: ignore
        for name, vnames in list(self.edges.items()):
            v1name, v2name = vnames
            v1 = self.virtuals[v1name]
            v2 = v1
            if not v2name.startswith("SUBUNIT"):
                v2 = self.virtuals[v2name]
            cen1 = self.displaycen(name, v1, offset=XYlen)
            cen2 = self.displaycen(name, v2, offset=XYlen)
            CGO.extend(cgo_lineabs(cen1, cen2, (1, 1, 1)))  # type: ignore
            upmid = (cen1 + 3.0*cen2) / 4.0
            CGO.extend(cgo_sphere(upmid))  # type: ignore
            v = cmd.get_view()  # type: ignore
            axes = [[v[0], v[3], v[6]], [v[1], v[4], v[7]], [v[2], v[5], v[8]]]
            # pymol.cgo.wire_text(CGO,pymol.vfont.plain,[upmid[0],upmid[1],upmid[2]],name)

        cmd.load_cgo(CGO, tag)  # type: ignore
