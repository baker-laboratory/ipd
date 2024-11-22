import contextlib
import ipd
with contextlib.suppress(ImportError):
    from ipd.sym.xtal.spacegroup_data import *
    from ipd.sym.xtal.spacegroup_deriveddata import *
    from ipd.sym.xtal.spacegroup_util import *

    _memoized_frames = {}

    if "H32" in sg_frames_dict:
        sg_frames_dict["R3"] = sg_frames_dict["H3"]  # type: ignore
        sg_frames_dict["R32"] = sg_frames_dict["H32"]  # type: ignore
    sg_redundant = {"P2": "P121"}
    sg_all = [k for k in sg_pdbname if k not in sg_redundant]
    sg_all_chiral = [k for k in sg_all if sg_is_chiral(k)]

def sgframes(
    spacegroup: str,
    cellgeom=None,
    cells=1,
    sortframes="default",
    roundgeom=10,
    xtalrad=9e9,
    asucen=[0.5, 0.5, 0.5],
    xtalcen=None,
    **kw,
):
    spacegroup = spacegroup_canonical_name(spacegroup)
    if isinstance(cellgeom, np.ndarray) or cellgeom not in ("unit", "nonsingular", None):
        cellgeom = tuple(round(x, roundgeom) for x in cellgeom)
    cells = process_num_cells(cells)
    key = spacegroup, cellgeom, tuple(cells.flat), sortframes
    if key not in _memoized_frames:
        unitframes = sg_frames_dict[spacegroup]  # type: ignore
        if cellgeom == "unit":
            latticevec = np.eye(3)
        else:
            latticevec = lattice_vectors(spacegroup, cellgeom=cellgeom)
        frames = latticeframes(unitframes, latticevec, cells)  # type: ignore
        frames = prune_frames(frames, asucen, xtalrad, xtalcen)
        frames = sort_frames(frames, method=sortframes)

        _memoized_frames[key] = frames.round(10)  # type: ignore
        if len(_memoized_frames) > 10_000:
            ipd.WARNME("sgframes holding >10000 _memoized_frames")

    return _memoized_frames[key]

def sgpermutations(spacegroup: str, cells=4):
    assert cells == 4
    spacegroup = spacegroup_canonical_name(spacegroup)
    return sg_permutations444_dict[spacegroup]  # type: ignore

def symelems(spacegroup: str, psym=None, asdict=False, screws=True, cyclic=True):
    if isinstance(psym, int):
        psym = f"c{psym}"
    spacegroup = spacegroup_canonical_name(spacegroup)
    se = sg_symelem_dict[spacegroup]  # type: ignore
    if not screws:
        se = [e for e in se if e.screw == 0]  # type: ignore
    if not cyclic:
        se = [e for e in se if not e.iscyclic]  # type: ignore
    if psym:
        return [e for e in se if e.label == psym.upper()]  # type: ignore
    if asdict:
        d = defaultdict(list)  # type: ignore
        for e in se:
            d[e.label].append(e)  # type: ignore
        se = d
    return se

def cryst1_line(spacegroup, lattice):
    pdb_sg = ipd.sym.xtal.sg_pymol_name(spacegroup)
    cellgeom = cellgeom_from_lattice(lattice)
    return ipd.sym.cryst1_pattern_full % (*cellgeom, pdb_sg)

def prune_frames(frames, asucen, xtalrad, center=None):
    center = center or asucen
    center = ipd.homog.hpoint(center)
    asucen = ipd.homog.hpoint(asucen)
    pos = ipd.homog.hxformpts(frames, asucen)
    dis = ipd.homog.hnorm(pos - center)
    frames = frames[dis <= xtalrad]
    return frames

def copies_per_cell(spacegroup):
    spacegroup = spacegroup_canonical_name(spacegroup)
    return len(sg_frames_dict[spacegroup])  # type: ignore

def cellgeom_from_lattice(lattice, radians=False):
    u, v, w = lattice.T
    a = ipd.homog.hnorm(u)
    b = ipd.homog.hnorm(v)
    c = ipd.homog.hnorm(w)
    if radians:
        A = ipd.homog.angle(v, w)
        B = ipd.homog.angle(u, w)
        C = ipd.homog.angle(u, v)
    else:
        A = ipd.homog.angle_degrees(v, w)
        B = ipd.homog.angle_degrees(u, w)
        C = ipd.homog.angle_degrees(u, v)
    return [a, b, c, A, B, C]

def sort_frames(frames, method):
    if method == "default":
        return frames
    if method == "dist_to_asucen":
        assert 0

def to_unitcell(spacegroup, cellgeom, coords):
    spacegroup = spacegroup_canonical_name(spacegroup)
    latt = lattice_vectors(spacegroup, cellgeom)
    com = ipd.homog.hcom(coords)
    unitcom = ipd.sym.tounitcellpts(latt, com)
    unitcom[:3] = unitcom[:3] % 1.0
    newcom = ipd.sym.applylatticepts(latt, unitcom)
    return ipd.homog.htrans(newcom - com)

def spacegroups_with_symelem(label, **kw):
    return [sg for sg in ipd.sym.xtal.sg_all_chiral if any([e.label == label for e in ipd.sym.xtal.symelems(sg)])]
