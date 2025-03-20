import contextlib
import copy
import functools

from ipd.dev import Bunch
from ipd.homog.hgeom import *
from ipd.sym.symframes import *

# from ipd.sym.asufit import *
from ipd.sym.xtal.xtalcls import *
from ipd.sym.xtal.xtalinfo import *

# from ipd.viz import showme

def frames(
    sym,
    axis=None,
    axis0=None,
    bbsym=None,
    asym_of=None,
    asym_index=0,
    sortframes=True,
    com=None,
    ontop=None,
    sgonly=False,
    torch=False,
    helix_radius=None,
    helix_angle=None,
    helix_shift=None,
    **kw,
):
    """Generate symmetrical coordinate frames axis aligns Cx or bbaxis or axis0
    to this bbsym removes redundant building block frames, e.g. TET with c3 bbs
    has 4 frames asym_of removes redundant frames wrt a point group, e.g. turn
    TET into C3 and get asym unit of that C3."""
    ipd.dev.checkpoint(kw, funcbegin=True)

    if sym is None or (not isinstance(sym, int) and sym.upper() == "C1"):
        return np.eye(4).reshape(1, 4, 4)
    sym = map_sym_abbreviation(sym)
    sym = sym.lower()

    okexe = (SystemExit, ) if sgonly else (KeyError, AttributeError)
    with contextlib.suppress(okexe):
        return ipd.sym.xtal.sgframes(sym, ontop=ontop, **kw)
    try:
        if ipd.sym.is_known_xtal(sym):
            return xtal(sym).frames(ontop=ontop, **kw).copy()
        elif sym.startswith('h'):
            helix_nfold = 1 if sym == 'h' else int(sym[1:])
            return helix_frames(helix_radius, helix_angle, helix_shift, helix_nfold)
        else:
            f = sym_frames[sym].copy()
    except KeyError as e:
        raise ValueError(f"unknown symmetry {sym}") from e
    ipd.dev.checkpoint("frames gen")

    if asym_of:
        assert asym_of.startswith("c")
        dupaxis = axes(sym, asym_of)
        dupnfold = int(asym_of[1:])
        arbitrary_step_vector = hnormalized([100, 10, 1, 0])
        arbitrary_delta = np.array([0.0001, 0.0002, 0.0003, 0])
        symx = hrot(dupaxis, 2 * np.pi * asym_index / dupnfold)
        arbitrary_step_vector = hxformvec(symx, arbitrary_step_vector)
        arbitrary_delta = hxformvec(symx, arbitrary_delta)
        angs = np.arange(dupnfold) / dupnfold * 2 * np.pi
        dups = hrot(dupaxis, angs)  # .reshape(-1, 4, 4)
        f2 = dups[None, :] @ f[:, None]
        x = hdot(f2, dupaxis + arbitrary_delta)
        tgtdir = hcross(arbitrary_step_vector, dupaxis)
        dot = hdot(tgtdir, x)
        order = np.argsort(-dot, axis=-1)
        assert np.sum(order[:, 0] == 0) == len(f) / dupnfold
        f = f[order[:, 0] == 0]
        ipd.dev.checkpoint("frames asym_of")

    if bbsym:
        assert asym_of is None or bbsym == asym_of
        if not bbsym.lower().startswith("c"):
            raise ValueError(f"bad bblock sym {bbsym}")
        bbnfold = int(bbsym[1:])
        # bbaxes = axes(sym, bbnfold, all=True)
        bbaxes = symaxes_all[sym][bbnfold].copy()
        partial_ok = asym_of is not None
        f = remove_if_same_axis(f, bbaxes, partial_ok=partial_ok)
        ipd.dev.checkpoint("frames bbsym")

    if axis is not None:
        # assert 0, 'doesnt work right with cyclic...'
        if axis0 is not None:
            startax = axis0
        elif sym.startswith("c"):
            startax = ipd.homog.hvec([0, 0, 1])
        elif bbsym:
            startax = axes(sym, bbnfold)
        elif asym_of:
            startax = axes(sym, asym_of)
        else:
            raise ValueError(f"dont know what to align to axis={axis}")

        # print(startax)
        # print(axis)
        # ipd.showme(f)
        xaln = halign(startax, axis)
        # f = f @ ipd.homog.hrot([0, 1, 0], 90) @ ipd.homog.hinv(f)
        # f = ipd.homog.hinv(f) @ ipd.homog.hrot([0, 1, 0], 90) @ f
        # f = ipd.homog.hrot([0, 1, 0], 90) @ f @ ipd.homog.hrot([0, 1, 0], -90)
        f = ipd.homog.hinv(xaln) @ f @ xaln
        # ipd.showme(f)
        # assert 0, 'this is bugged?'
        ipd.dev.checkpoint("frames axis")

    if sortframes:
        csym = bbsym or asym_of
        if csym:
            if axis is None:
                axis = axes(sym, csym)
            # order = np.argsort(-hdot(axis, hdot(f, axes(sym, csym))))
            # com = None
            ref = axes(sym, csym) if com is None else com
            order = np.argsort(-hdot(axis, hxformvec(f, ref)))
            # print(f.shape, order)
            f = f[order]
            # print(order)
            # assert 0

        if ontop is not None:
            f = put_frames_on_top(f, ontop)
        ipd.dev.checkpoint("frames sortframes")

    if torch:
        import torch as th  # type: ignore
        return th.as_tensor(f, dtype=th.float64)

    ipd.dev.checkpoint(kw)
    return f.round(10)

def put_frames_on_top(frames, ontop, strict=True, allowcellshift=False, cellsize=None, **kw):
    ipd.dev.checkpoint(kw, funcbegin=True)
    # ipd.icv(allowcellshift, cellsize)
    frames2 = list(frames)
    if len(frames) == 0:
        return ontop
    celldeltas = [0]
    if allowcellshift:
        celldeltas = list(itertools.product(*[np.arange(-1, 2) * cellsize] * 3))  # type: ignore
    diff = ipd.homog.hdiff(ontop, np.stack(frames2))
    w = np.nonzero(diff < 0.0001)
    if strict:
        # ipd.icv(w, ontop.shape)
        assert len(w) == 2
        assert set(w[0]) == set(range(len(ontop)))
    for i in reversed(sorted(w[1])):
        del frames2[i]

    # f*ck this code right in the ear
    # for f0 in ontop:
    #    for i, x in enumerate(frames2):
    #       if ipd.homog.hdiff(f0, x) < 0.0001: break
    #       match = False
    #       for delta in celldeltas:
    #          tmp = x.copy()
    #          tmp[:3, 3] += delta
    #          if ipd.homog.hdiff(f0, tmp) < 0.0001:
    #             match = True
    #       if match: break
    #    else:
    #       if strict:
    #          ipd.showme(ipd.hscaled(1, frames), name='frames')
    #          ipd.showme(ipd.hscaled(1, ontop), name='ontop')
    #          raise ValueError(f'ontop frame not found: {f0}')
    #       else:
    #          i = None
    #    if i is not None:
    #       del frames2[i]

    # assert ipd.homog.hnuique(np.stack(frames2))
    if len(frames2) == 0:
        f = ontop
    else:
        f = np.stack(list(ontop) + frames2)
    assert ipd.homog.hnuique(f)  # type: ignore

    ipd.dev.checkpoint(kw)
    return f

def helix_frames(helix_radius, helix_angle, helix_shift, helix_nfold):
    unit = hrot([0, 0, 1], helix_angle, [0, 0, 0], hel=helix_shift)
    xglobal = frames(f'c{helix_nfold}')
    hframes = np.stack([np.eye(4), unit, hinv(unit)])
    xasu = htrans(helix_radius)
    return hxform(xglobal, hframes, xasu).reshape(-1, 4, 4)

def make(sym, x, **kw):
    return ipd.homog.hxform(frames(sym, **kw), x)

def makepts(sym, x, **kw):
    return ipd.homog.hxformpts(frames(sym, **kw), x)

def makex(sym, x, **kw):
    return ipd.homog.hxformx(frames(sym, **kw), x)

def map_sym_abbreviation(sym):
    if sym == "I":
        return "icos"
    if sym == "O":
        return "oct"
    if sym == "T":
        return "tet"
    if sym in "I32 I53 I52".split():
        return "icos"
    if sym in "O32 O42 O43".split():
        return "oct"
    if sym == "T32":
        return "tet"
    if isinstance(sym, int):
        return f"c{sym}"
    return sym

def symaxis_angle(sym, nf1, nf2):
    return ipd.homog.angle(axes(sym, nf1), axes(sym, nf2))

def symaxis_radbias(sym, nf1, nf2):
    return 1 / np.arctan(ipd.homog.angle(axes(sym, nf1), axes(sym, nf2)))

def min_symaxis_angle(sym):
    symaxes = axes(sym)
    minaxsang = 9e9
    for i, iax in symaxes.items():
        for j, jax in symaxes.items():
            if i != j:
                minaxsang = min(minaxsang, line_angle(iax, jax))
                # print(i, j, line_angle_degrees(iax, jax))
    return minaxsang

def axes(sym, nfold=None, all=False, cellsize=1, closest_to=None, **kw):
    sym = sym.lower()
    sym_name_map = dict(t='tet', o='oct', i='icos', i32='icos', i53='icos', i52='icos', i532='icos')
    sym = sym_name_map.get(sym, sym)
    try:
        if ipd.sym.is_known_xtal(sym):
            x = xtal(sym)
            if all:
                elems = copy.deepcopy(x.unitelems)
            else:
                elems = copy.deepcopy(x.symelems.copy())
            for e in elems:
                e.cen = ipd.hscaled(cellsize, e.cen)  # type: ignore
            return elems
        else:
            if sym.startswith(("icos", "oct", "tet")):
                if sym[-1].isdigit() and nfold is None:
                    nfold = int(sym[-1])
                    sym = sym[:-1]
            if all:
                axes = symaxes_all[sym].copy()
            elif closest_to is not None:
                axes = symaxes_all[sym].copy()
                for k, v in axes.items():
                    axes[k] = v[np.argmax(np.abs(np.dot(v, ipd.homog.hvec(closest_to))))]  # type: ignore
            else:
                axes = symaxes[sym].copy()
            if nfold:
                if isinstance(nfold, str):
                    assert nfold.lower().startswith("c")
                    nfold = int(nfold[1:])
                axes = axes[nfold]
            return axes

    except (KeyError, ValueError) as e:
        raise ValueError(f"unknown symmetry {sym}")

@functools.lru_cache()
def symelem_associations(sym=None, symelems=None):
    if not symelems:
        symelems = axes(sym)
    assoc = list()
    n = 1
    for s in symelems:
        nbrs = list()
        for i in range(1, s.nfold):
            nbrs.append(n)
            n += 1
        assoc.append(ipd.dev.Bunch(nbrs=nbrs, symelem=s))
    return assoc

def remove_if_same_axis(frames, bbaxes, onesided=True, partial_ok=False):
    assert onesided
    axes = hxformvec(frames, bbaxes[0])
    dots = hdot(bbaxes, axes, outerprod=True)

    uniq = list()
    whichaxis = list()
    for i, dot in enumerate(dots):
        w = np.where(np.logical_and(0.99999 < np.abs(dot), np.abs(dot) < 1.00001))[0]
        assert len(w) == 1
        w = w[0]
        if not np.any(np.isclose(dots[:i, w], dot[w], atol=0.00001)):
            whichaxis.append(w)
            uniq.append(i)
    whichaxis = np.array(whichaxis)
    # should be same num of bblocks on axis, (1 or 2)
    whichpartial = list()
    for i in range(len(bbaxes)):
        n = np.sum(whichaxis == i)
        # print(i, n)
        if not partial_ok:
            assert n == np.sum(whichaxis == 0)
        if n == 2:
            a, b = np.where(whichaxis == i)[0]
            assert np.allclose(axes[uniq[a]], -axes[uniq[b]], atol=1e-6)
        elif n != 1:
            if not partial_ok:
                assert 0

    uniq = np.array(uniq)
    return frames[uniq]

_ambiguous_axes = Bunch(tet=[], oct=[(2, 4)], icos=[], d2=[], _strict=True)  # type: ignore

def ambiguous_axes(sym):
    return _ambiguous_axes[sym]

_ = -1

tetrahedral_axes = {
    2: hnormalized([0, 0, 1]),
    3: hnormalized([1, 1, 1]),
    "3b": hnormalized([_, 1, 1]),  # other c3
}
octahedral_axes = {2: hnormalized([1, 0, 1]), 3: hnormalized([1, 1, 1]), 4: hnormalized([0, 0, 1])}
icosahedral_axes = {
    2: hnormalized([0, 0, 1]),
    3: hnormalized([0, 0.35682209, 0.93417236]),
    5: hnormalized([0.525731, 0.000000, 0.850651]),
}

tetrahedral_axes_all = {
    2:
    hnormalized([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        # [_, 0, 0],
        # [0, _, 0],
        # [0, 0, _],
    ]),
    3:
    hnormalized([
        [1, 1, 1],
        [1, _, _],
        [_, _, 1],
        [_, 1, _],
        # [_, _, _],
        # [_, 1, 1],
        # [1, 1, _],
        # [1, _, 1],
    ]),
    "3b":
    hnormalized([
        [_, 1, 1],
        [1, _, 1],
        [1, 1, _],
        [_, _, -1],
    ]),
}
octahedral_axes_all = {
    2:
    hnormalized([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [_, 1, 0],
        [0, _, 1],
        [_, 0, 1],
        # [1, _, 0],
        # [0, 1, _],
        # [1, 0, _],
        # [_, _, 0],
        # [0, _, _],
        # [_, 0, _],
    ]),
    3:
    hnormalized([
        [1, 1, 1],
        [_, 1, 1],
        [1, _, 1],
        [1, 1, _],
        # [_, 1, _],
        # [_, _, 1],
        # [1, _, _],
        # [_, _, _],
    ]),
    4:
    hnormalized([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        # [_, 0, 0],
        # [0, _, 0],
        # [0, 0, _],
    ]),
}

def _icosahedral_axes_all():
    a2 = icosahedral_frames @ icosahedral_axes[2]
    a3 = icosahedral_frames @ icosahedral_axes[3]
    a5 = icosahedral_frames @ icosahedral_axes[5]

    # six decimals enough to account for numerical errors
    a2 = a2[np.unique(np.around(a2, decimals=6), axis=0, return_index=True)[1]]
    a3 = a3[np.unique(np.around(a3, decimals=6), axis=0, return_index=True)[1]]
    a5 = a5[np.unique(np.around(a5, decimals=6), axis=0, return_index=True)[1]]
    a2 = np.stack([a for i, a in enumerate(a2) if np.all(np.sum(a * a2[:i], axis=-1) > -0.999)])
    a3 = np.stack([a for i, a in enumerate(a3) if np.all(np.sum(a * a3[:i], axis=-1) > -0.999)])
    a5 = np.stack([a for i, a in enumerate(a5) if np.all(np.sum(a * a5[:i], axis=-1) > -0.999)])

    assert len(a2) == 15  # 30
    assert len(a3) == 10  # 20
    assert len(a5) == 6  # 12
    icosahedral_axes_all = {
        2: hnormalized(a2),
        3: hnormalized(a3),
        5: hnormalized(a5),
    }
    return icosahedral_axes_all

icosahedral_axes_all = _icosahedral_axes_all()

def _d_axes(nfold):
    return {2: hnormalized([1, 0, 0]), nfold: hnormalized([0, 0, 1])}

def _d_frames(nfold):
    cx = hrot([0, 0, 1], np.pi * 2 / nfold)
    c2 = hrot([1, 0, 0], np.pi)
    frames = list()
    for ix in range(nfold):
        rot2 = hrot([0, 0, 1], np.pi * 2 * ix / nfold)
        for i2 in range(2):
            rot1 = [np.eye(4), c2][i2]
            frames.append(rot1 @ rot2)
    return np.array(frames)

def _d_axes_all(nfold):
    ang = 2 * np.pi / nfold
    frames = _d_frames(nfold)
    a2A = frames @ [1, 0, 0, 0]
    anA = frames @ [0, 0, 1, 0]
    if nfold % 2 == 0:
        a2A = np.concatenate([a2A, frames @ [np.cos(ang / 2), np.sin(ang / 2), 0, 0]])

    # six decimals enough to account for numerical errors
    a2B = a2A[np.unique(np.around(a2A, decimals=6), axis=0, return_index=True)[1]]
    anB = anA[np.unique(np.around(anA, decimals=6), axis=0, return_index=True)[1]]
    a2B = np.flip(a2B, axis=0)

    a2 = np.stack([a for i, a in enumerate(a2B) if np.all(np.sum(a * a2B[:i], axis=-1) > -0.999)])
    an = np.stack([a for i, a in enumerate(anB) if np.all(np.sum(a * anB[:i], axis=-1) > -0.999)])

    # if nfold == 4:
    #     print(np.around(a2A, decimals=3))
    #     print()
    #     print(np.around(a2B, decimals=3))
    #     print()
    #     print(np.around(a2, decimals=3))
    #     print()

    assert len(an) == 1, f"nfold {nfold}"
    assert len(a2) == nfold, f"nfold {nfold}"

    axes_all = {
        2: hnormalized(a2),
        nfold: hnormalized(an),
    }
    return axes_all

symaxes = dict(
    tet=tetrahedral_axes,
    oct=octahedral_axes,
    icos=icosahedral_axes,
)

symaxes_all = dict(
    tet=tetrahedral_axes_all,
    oct=octahedral_axes_all,
    icos=icosahedral_axes_all,
)

tetrahedral_angles = {
    (i, j): angle(
        tetrahedral_axes[i],
        tetrahedral_axes[j],
    )
    for i, j in [
        (2, 3),
        (3, "3b"),
    ]
}
octahedral_angles = {
    (i, j): angle(
        octahedral_axes[i],
        octahedral_axes[j],
    )
    for i, j in [
        (2, 3),
        (2, 4),
        (3, 4),
    ]
}
icosahedral_angles = {
    (i, j): angle(
        icosahedral_axes[i],
        icosahedral_axes[j],
    )
    for i, j in [
        (2, 3),
        (2, 5),
        (3, 5),
    ]
}
nfold_axis_angles = dict(
    tet=tetrahedral_angles,
    oct=octahedral_angles,
    icos=icosahedral_angles,
)
sym_point_angles = dict(
    tet={
        2: [np.pi],
        3: [np.pi * 2 / 3]
    },
    oct={
        2: [np.pi],
        3: [np.pi * 2 / 3],
        4: [np.pi / 2]
    },
    icos={
        2: [np.pi],
        3: [np.pi * 2 / 3],
        5: [np.pi * 2 / 5, np.pi * 4 / 5]
    },
    d3={
        2: [np.pi],
        3: [np.pi * 2 / 3],
    },
)

sym_frames = dict(
    tet=tetrahedral_frames,
    oct=octahedral_frames,
    icos=icosahedral_frames,
)
minsymang = dict(
    tet=angle(tetrahedral_axes[2], tetrahedral_axes[3]) / 2,
    oct=angle(octahedral_axes[2], octahedral_axes[3]) / 2,
    icos=angle(icosahedral_axes[2], icosahedral_axes[3]) / 2,
    d2=np.pi / 4,
)
for icyc in range(3, 33):
    sym = "d%i" % icyc
    symaxes[sym] = _d_axes(icyc)
    sym_frames[sym] = _d_frames(icyc)
    ceil = int(np.ceil(icyc / 2))
    sym_point_angles[sym] = {
        2: [np.pi],
        icyc: [np.pi * 2 * j / icyc for j in range(1, ceil)],
    }
    minsymang[sym] = np.pi / icyc / 2
    symaxes_all[sym] = _d_axes_all(icyc)
    _ambiguous_axes[sym] = list() if icyc % 2 else [(2, icyc)]

sym_frames["d2"] = np.stack([
    np.eye(4),
    hrot([1, 0, 0], np.pi),
    hrot([0, 1, 0], np.pi),
    hrot([0, 0, 1], np.pi),
])

symaxes["d2"] = {
    2: np.array([1, 0, 0, 0]),
    "2b": np.array([0, 1, 0, 0]),
    "2c": np.array([0, 0, 1, 0]),
}
symaxes_all["d2"] = {
    2: np.array([
        np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0, 0, 1, 0]),
    ])
}

sym_point_angles["d2"] = {2: [np.pi]}

def sym_nfold_map(nfold):
    if isinstance(nfold, str):
        return int(nfold[:-1])
    return nfold

def get_syminfo(sym):
    sym = sym.lower()
    try:
        ambig = ambiguous_axes(sym)
        nfoldmap = {k: sym_nfold_map(k) for k in symaxes[sym]}
        assert sym_frames[sym].shape[-2:] == (4, 4)
        return Bunch(
            frames=sym_frames[sym],
            axes=symaxes[sym],
            axesall=symaxes_all[sym],
            point_angles=sym_point_angles[sym],
            ambiguous_axes=ambig,
            nfoldmap=nfoldmap,
        )

    except KeyError as e:
        # raise ValueError(f'sim.py: dont know symmetry "{sym}"')
        print(f'sym.py: dont know symmetry "{sym}"')
        raise e

_sym_permute_axes_choices = dict(
    d2=np.array([
        np.eye(4),  #           x y z
        hrot([1, 0, 0], 90),  # x z y
        hrot([0, 0, 1], 90),  # y z x
        hrot([1, 0, 0], 90) @ hrot([0, 0, 1], 90),  # y x z
        hrot([0, 1, 0], 90),  # z y x
        hrot([1, 0, 0], 90) @ hrot([0, 1, 0], 90),  # z y x
    ]),
    d3=np.array([
        np.eye(4),
        hrot([0, 0, 1], 180),
    ]),
)

def sym_permute_axes_choices(sym):
    if sym in _sym_permute_axes_choices:
        return _sym_permute_axes_choices[sym]
    else:
        return np.eye(4).reshape(1, 4, 4)

for icyc in range(2, 33):
    sym = "c%i" % icyc
    symaxes[sym] = {icyc: np.array([0, 0, 1, 0])}
    angles = 2 * np.pi * np.arange(icyc) / icyc
    # print(angles * 180 / np.pi)
    sym_frames[sym] = hrot([0, 0, 1, 0], angles)
    sym_point_angles[sym] = {  # type: ignore
        icyc: [angles],
    }
    minsymang[sym] = np.pi / icyc / 2
    symaxes_all[sym] = symaxes[sym]
    _ambiguous_axes[sym] = list()

def is_closed(sym):
    sym = sym.upper()
    if sym.startswith(("C", "D")):
        return True
    if sym in "I O T ICOS OCT TET":
        return True
    return False

def symunit_bounds(cagesym, cycsym):
    flb, fub, fnum = 1, -1, 2
    if cagesym == "tet" and cycsym == "c3":
        fub, fnum = None, 1
    return flb, fub, fnum

def coords_to_asucen(sym, coords, **kw):
    if ipd.sym.is_known_xtal(sym):
        x = xtal(sym)
        return x.coords_to_asucen(coords, **kw)
    else:
        raise NotImplementedError

def primary_frames(sym, **kw):
    if ipd.sym.is_known_xtal(sym):
        x = xtal(sym)
        return x.primary_frames(**kw)
    else:
        raise NotImplementedError

def ndim(sym):
    try:
        return xtal(sym).dimension
    except KeyError:
        pass

def numpy_or_torch_array(source, example):
    if "torch" in sys.modules:
        import torch  # type: ignore
        if torch.is_tensor(example):
            return torch.as_tensor(source)
    return np.asarray(source)

CoordRMS = collections.namedtuple("CorodRMS", "coords rms")

def average_aligned_coords(coords, nsub=None, repeatfirst=1):
    orig = coords
    coords = ipd.homog.hpoint(coords)
    if nsub is None:
        nsub = len(coords)
    if coords.ndim == 2:
        coords = coords.reshape(nsub, -1, 4)
    assert len(coords) > 1
    assert nsub is None or nsub == len(coords)

    fits = [ipd.hrmsfit(_, coords[0]) for _ in coords[1:]]  # type: ignore
    rms, crds, _ = zip(*fits)
    crds = [coords[0]] * repeatfirst + list(crds)
    crd = numpy_or_torch_array(np.stack(crds).mean(0), orig)
    rms = numpy_or_torch_array(rms, orig)
    return CoordRMS(crd, rms)

def subframes(frames, bbsym, asym):
    assert frames.ndim == 3 and frames.shape[1:] == (4, 4)
    subframes = ipd.sym.frames(bbsym)
    coords = ipd.homog.hxform(frames, ipd.homog.hcom(asym, flat=True))
    ipd.icv(coords)  # type: ignore
    ipd.icv(frames.shape)  # type: ignore
    ipd.icv(subframes.shape)  # type: ignore
    # relframes = frames[1:, None] @ ipd.homog.hinv(frames[None, :-1])
    relframes = frames[:, None] @ ipd.homog.hinv(frames[None, :])
    ipd.icv(relframes.shape)  # type: ignore
    axs, ang, cen, hel = ipd.homog.axis_angle_cen_hel_of(relframes)

    for i in range(len(frames)):
        axdist = ipd.homog.h_point_line_dist(coords, cen[i, :], axs[i, :])  # type: ignore
        ipd.icv(axdist)  # type: ignore
    # what about multiple nfold axes???\
    # can distinguish by axis direction?
    assert 0

    helok = hel == 0
    # priax =
    # closest axis

    axisdist = ipd.hprojperp(axs, cen)  # type: ignore
    ipd.icv(axisdist)  # type: ignore

# computed in ipd.sym.asufit.compute_canonical_asucen
_canon_asucen = dict(
    c2=np.array([1.0, 0., 0.]),
    c3=np.array([1.15470054, 0., 0.]),
    c4=np.array([1.41421357, 0., 0.]),
    c5=np.array([1.70130162, 0., 0.]),
    c6=np.array([2.00000000, 0., 0.]),
    c7=np.array([2.3047649, 0., 0.]),
    c8=np.array([2.61312595, 0., 0.]),
    c9=np.array([2.92380443, 0., 0.]),
    d2=np.array([-0.70690629, 0.7075665, 0.70730722]),
    d3=np.array([5.10486311e-04, 1.15470043e+00, 8.16910008e-01]),
    d4=np.array([1.30706011, 0.54147883, 0.84079882]),
    d5=np.array([1.00216512, 1.37480626, 0.85245505]),
    d6=np.array([0.51697002, 1.93266443, 0.8560035]),
    tet=np.array([9.47438171e-05, 1.00242090e+00, 1.61772847e+00]),
    oct=np.array([0.67599002, 1.2421906, 2.28592391]),
    icos=np.array([1.13567793, 1.28546351, 3.95738551]),
    i2=np.array([0, 0, 1]),
    i3=np.array([0, 0.35, 0.93]),
    i5=np.array([0.52, 0, 0.85]),
    i32=np.array([0, 1, 5.85725386]),
    # i32=np.array([1.13567793, 1.28546351, 3.95738551]),
    i52=np.array([1.13567793, 1.28546351, 3.95738551]),
    i53=np.array([1.13567793, 1.28546351, 3.95738551]),
    i532=np.array([1.13567793, 1.28546351, 3.95738551]),
    icos4=np.array([0, 1, 5.85725386]),
)
_canon_asucen = {k: ipd.homog.hnormalized(_canon_asucen[k]) for k in _canon_asucen}
_canon_asucen['i2'] = _canon_asucen['i2'] * 0.8 + _canon_asucen['i532'] * 0.2
_canon_asucen['i3'] = _canon_asucen['i3'] * 0.8 + _canon_asucen['i532'] * 0.2
_canon_asucen['i5'] = _canon_asucen['i5'] * 0.8 + _canon_asucen['i532'] * 0.2

def canonical_asu_center(sym, cuda=False):
    sym = ipd.sym.map_sym_abbreviation(sym).lower()
    if sym.startswith('h'): return [0, 0, 0]
    try:
        if cuda:
            import torch as th  # type: ignore
            return th.tensor(_canon_asucen[sym], device='cuda')
        return _canon_asucen[sym]
    except KeyError as e:
        if cuda:
            import torch as th  # type: ignore
            return th.tensor(compute_canonical_asucen(sym), device='cuda')
        return compute_canonical_asucen(sym)
        # except KeyError as e:
        #     raise ValueError(f'canonical_asu_center: unknown sym {sym}') from e

def compute_canonical_asucen(sym, neighbors=None):
    import torch as th  # type: ignore
    import ipd.homog.thgeom as h
    sym = ipd.sym.map_sym_abbreviation(sym).lower()
    frames = ipd.sym.frames(sym)
    x = h.randunit(int(5e5))
    symx = h.xform(frames[1:], x)
    d2 = th.sum((x[None] - symx)**2, dim=-1)

    mind2 = d2.min(dim=0)[0]
    if neighbors:
        # ipd.icv(d2.shape)  # type: ignore
        sort = d2.sort(dim=0)[0]
        rank = sort[neighbors] - sort[neighbors - 1]
    else:
        rank = mind2

    ibest = th.argmax(rank)
    best = x[ibest]
    dbest = th.sqrt(mind2[ibest])
    symbest = h.xform(frames, best)
    aln = th.sum(symbest * th.tensor([1, 2, 10]), dim=1)
    best = symbest[th.argmax(aln)] / dbest * 2
    if sym.startswith('c'):
        best = th.tensor([h.norm(best), 0, 0])
    return best.cpu().numpy()

magic_angle_DTOI = ipd.Bunch(
    D=float(np.pi / 2),
    T=0.955316621,
    O=0.615479714,
    I=0.364863837,
)
