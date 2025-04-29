import ipd
from ipd.sym.pymol_symgen import PymolSymElem, Vec  # type: ignore
try:
    import pymol  # type: ignore
    from pymol import cmd  # type: ignore
except ImportError:
    pymol = None
    cmd = None

# I213
#    AXS = [Vec(1, 1, 1), Vec(1, 0, 0)]
#    CEN = [cell * Vec(0, 0, 0), cell * Vec(0, 0, 0.25)]

def example_I213(cell=70, **kw):
    "delete all; run ~/pymol3/symgen.py; run ~/pymol3/symgen_test.py; test_I213(depth=8)"
    # AXS = [Vec(1, 1, 1), Vec(1, 1, -1)]
    # CEN = [cell * Vec(0, 0, 0), cell * Vec(0.5, 0, 0.0)]
    G = [
        PymolSymElem("C2", axis=Vec(1, 0, 0), cen=Vec(0, 0, 0.25) * cell, col=[1, 0.7, 0.0]),
        PymolSymElem("C3", axis=Vec(1, 1, 1) * 0.57735, cen=Vec(0, 0, 0) * cell, col=[0.1, 0.5, 1]),
    ]

    hacky_xtal_maker(  # type: ignore
        G,
        cell,
        tag="I213",
        origin=cell * Vec(0.0, 0.0, 0.0),
        showshape=0,
        symdef=0,
        make_symdef=0,
        showlinks=1,
        shownodes=1,
        radius=0.5,
        bbox=[Vec(-1, -1, -1) * -999, Vec(cell + 1, cell + 1, cell + 1) * 100],
        # length=40,
        **kw,
    )
    ipd.viz.pymol_cgo.showcube(Vec(0, 0, 0), Vec(1, 1, 1) * cell)

# def test_pymol_symgen():
# assert 0

if __name__ == "__main__":
    # pymol.finish_launching()

    # test_I213()
    pass
