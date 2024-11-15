import itertools

import numpy as np
from opt_einsum import contract as einsum

import ipd
from ipd.sym.xtal.spacegroup_data import *

def spacegroup_canonical_name(spacegroup):
    spacegroup = spacegroup.replace("p", "P").replace("i", "I").replace("f", "F")
    if spacegroup.startswith("c"):
        spacegroup = "C" + spacegroup[1:]
    if spacegroup.startswith("r"):
        spacegroup = "R" + spacegroup[1:]
    # spacegroup = spacegroup.upper()
    if spacegroup not in sg_lattice:
        spacegroup = sg_from_pdbname[spacegroup]
    return spacegroup

def number_of_canonical_cells(spacegroup):
    spacegroup = spacegroup_canonical_name(spacegroup)
    nframes_per_cell = sg_nframes[spacegroup]
    ncell = 4
    # if nframes_per_cell <= 4: ncell = 5
    if spacegroup == "P43212":
        return 5
    if spacegroup == "P21212":
        return 5
    if spacegroup == "P222":
        return 5
    # if spacegroup == 'P312': return 6
    return ncell

def latticetype(spacegroup):
    try:
        return sg_lattice[spacegroup]
    except KeyError:
        return sg_lattice[sg_from_pdbname[spacegroup]]

def applylattice(lattice, unitframes):
    origshape = unitframes.shape
    assert lattice.shape == (3, 3)
    assert unitframes.ndim == 3
    lattice_inv = np.linalg.inv(lattice)
    latticeframes = np.zeros_like(unitframes)
    latticeframes[:, :3, :3] = einsum("ij,fjk,kl->fil", lattice, unitframes[:, :3, :3], lattice_inv)
    latticeframes[:, :3, 3] = einsum("ij,fj->fi", lattice, unitframes[:, :3, 3])
    latticeframes[:, 3, 3] = 1
    return latticeframes

def applylatticepts(lattice, unitpoints):
    origshape = unitpoints.shape
    unitpoints = unitpoints.reshape(-1, 4)
    assert lattice.shape == (3, 3)
    lattice_inv = np.linalg.inv(lattice)
    latticepoints = np.ones_like(unitpoints)
    latticepoints[:, :3] = einsum("ij,fj->fi", lattice, unitpoints[:, :3])
    latticepoints = latticepoints.reshape(origshape)
    return latticepoints

def latticeframes(unitframes, lattice, cells=1):
    cells = process_num_cells(cells)
    xshift = ipd.homog.htrans(cells)
    unitframes = ipd.homog.hxformx(xshift, unitframes, flat=True, improper_ok=True)
    frames = applylattice(lattice, unitframes)
    return frames.round(10)

def tounitcell(lattice, *a, **kw):  # , frames, spacegroup=None):
    return applylattice(np.linalg.inv(lattice), *a, **kw)
    # if not hasattr(lattice, 'shape') or lattice.shape != (3, 3):
    # lattice = lattice_vectors(spacegroup, lattice)
    # unitframes = frames.copy()
    # lattinv = np.linalg.inv(lattice)
    # unitframes[:, :3, :3] = einsum('ij,fjk,kl->fil', lattinv, frames[:, :3, :3], lattice)
    # unitframes[:, :3, 3] = einsum('ij,fj->fi', lattinv, frames[:, :3, 3])
    # return unitframes.round(10)

def tounitcellpts(lattice, *a, **kw):
    return applylatticepts(np.linalg.inv(lattice), *a, **kw)
    # oshape = points.shape
    # points = points.reshape(-1, 4)
    # if not hasattr(lattice, 'shape') or lattice.shape != (3, 3):
    # lattice = lattice_vectors(spacegroup, lattice)
    # unitpoints = points.copy()
    # lattinv = np.linalg.inv(lattice)
    # unitpoints[:, :3] = einsum('ij,fj->fi', lattinv, points[:, :3])
    # unitpoints = unitpoints.reshape(oshape)
    # return unitpoints.round(10)

def process_num_cells(cells):
    if cells is None:
        return np.eye(4)[None]
    if isinstance(cells, np.ndarray) and cells.ndim == 2 and cells.shape[1] == 3:
        return cells
    if isinstance(cells, (int, float)):
        # ub = (cells - 1) // 2
        ub = cells // 2
        lb = ub - cells + 1
        cells = [(a, b, c) for a, b, c in itertools.product(*[range(lb, ub + 1)] * 3)]  # type: ignore
    elif len(cells) == 2:
        lb, ub = cells
        cells = [(a, b, c) for a, b, c in itertools.product(*[range(lb, ub + 1)] * 3)]
    elif len(cells) == 3:
        if isinstance(cells[0], int):
            cells = [(0, cells[0] - 1), (0, cells[1] - 1), (0, cells[2] - 1)]
        cells = [(a, b, c) for a, b, c in itertools.product(
            range(cells[0][0], cells[0][1] + 1),
            range(cells[1][0], cells[1][1] + 1),
            range(cells[2][0], cells[2][1] + 1),
        )]
    else:
        raise ValueError(f"bad cells {cells}")
    cells = np.array(cells)
    # ic(set(cells[:, 0]))

    # order in stages, cell 0 first, cell 0 to 1, cells -1 to 1, cells -1 to 2, etc
    blocked = list()
    mn, mx = np.min(cells, axis=1), np.max(cells, axis=1)
    lb, ub = 0, 0
    prevok = np.zeros(len(cells), dtype=bool)
    for i in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8]:
        # for i in [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8]:
        lb, ub = min(i, lb), max(i, ub)
        ok = np.logical_and(lb <= mn, mx <= ub)
        c = cells[np.logical_and(ok, ~prevok)]
        blocked.append(c)
        prevok |= ok
        if np.all(prevok):
            break
    cells = np.concatenate(blocked)

    return cells

def full_cellgeom(lattice: str, cellgeom, strict=True):
    if isinstance(cellgeom, (int, float)):
        cellgeom = [cellgeom]
    if isinstance(lattice, str) and lattice in sg_lattice:
        lattice = sg_lattice[lattice]

    # assert lattice in 'TETRAGONAL CUBIC'.split()
    assert isinstance(cellgeom, (np.ndarray, list, tuple))
    p = np.array(cellgeom)
    if lattice == "TRICLINIC":
        p = [p[0], p[1], p[2], p[3], p[4], p[5]]
    elif lattice == "MONOCLINIC":
        if strict:
            assert np.allclose(p[3], 90.0), f"invalid cell geometry {p}"
            assert len(p) < 6 or np.allclose(p[5], 90.0), f"invalid cell geometry {p}"
        p = [p[0], p[1], p[2], 90.0, p[4], 90.0]
    elif lattice == "CUBIC":
        if strict:
            assert len(p) < 4 or np.allclose(p[3], 90.0), f"invalid cell geometry {p}"
            assert len(p) < 5 or np.allclose(p[4], 90.0), f"invalid cell geometry {p}"
            assert len(p) < 6 or np.allclose(p[5], 90.0), f"invalid cell geometry {p}"
            assert np.allclose(p[0], p[:3])
        p = [p[0], p[0], p[0], 90.0, 90.0, 90.0]
    elif lattice == "ORTHORHOMBIC":
        if strict:
            assert len(p) < 4 or np.allclose(p[3], 90.0), f"invalid cell geometry {p}"
            assert len(p) < 5 or np.allclose(p[4], 90.0), f"invalid cell geometry {p}"
            assert len(p) < 6 or np.allclose(p[5], 90.0), f"invalid cell geometry {p}"
        p = [p[0], p[1], p[2], 90.0, 90.0, 90.0]
    elif lattice == "TETRAGONAL":
        if strict:
            assert np.allclose(p[0], p[1]), f"invalid cell geometry {p}"
            assert len(p) < 4 or np.allclose(p[3], 90.0), f"invalid cell geometry {p}"
            assert len(p) < 5 or np.allclose(p[4], 90.0), f"invalid cell geometry {p}"
            assert len(p) < 6 or np.allclose(p[5], 90.0), f"invalid cell geometry {p}"
            assert np.allclose(p[0], p[1])
        p = [p[0], p[0], p[2], 90.0, 90.0, 90.0]
    elif lattice == "HEXAGONAL":
        if strict:
            assert np.allclose(p[0], p[1]), f"invalid cell geometry {p}"
            assert len(p) < 4 or np.allclose(p[3], 90.0), f"invalid cell geometry {p}"
            assert len(p) < 5 or np.allclose(p[4], 90.0), f"invalid cell geometry {p}"
            assert len(p) < 6 or np.allclose(p[5], 120.0), f"invalid cell geometry {p}"
            assert np.allclose(p[0], p[1])
        p = [p[0], p[0], p[2], 90.0, 90.0, 120.0]
    else:
        raise ValueError(f"unknown lattice type {lattice} specified with cellgeom {cellgeom}")
    return p

def lattice_vectors(lattice, cellgeom=None, strict=True):
    if lattice in sg_lattice:
        lattice = sg_lattice[lattice]
    if cellgeom is None:
        # if lattice != 'CUBIC':
        raise ValueError(f"no cellgeom specified for lattice type {lattice}")
        # cellgeom = [1.0, 1.0, 1.0, 90.0, 90.0, 90.0]
        # if lattice == 'HEXAGONAL':
        # cellgeom = [1.0, 1.0, 1.0, 90.0, 90.0, 120.0]
    elif isinstance(cellgeom, str) and cellgeom == "nonsingular":
        cellgeom = full_cellgeom(lattice, sg_nonsingular_cellgeom, strict=False)
        # ic('cellgeom nonsingular', cellgeom)

    a, b, c, A, B, C = full_cellgeom(lattice, cellgeom, strict=strict)
    cosA, cosB, cosC = [np.cos(np.radians(_)) for _ in (A, B, C)]
    sinB, sinC = [np.sin(np.radians(_)) for _ in (B, C)]

    # ic(cosB * cosC - cosA)
    # ic(sinB, sinC)
    # ic(1.0 - ((cosB * cosC - cosA) / (sinB * sinC))**2)

    lattice_vectors = np.array([
        [
            a,
            b * cosC,
            c * cosB,
        ],
        [
            0.0,
            b * sinC,
            c * (cosA - cosB*cosC) / sinC,
        ],
        [
            0.0,
            0.0,
            c * sinB * np.sqrt(1.0 - ((cosB*cosC - cosA) / (sinB*sinC))**2),
        ],
    ])
    return lattice_vectors

def cell_volume(spacegroup, cellgeom):
    if isinstance(cellgeom, np.ndarray) and cellgeom.shape == (3, 3):
        cellgeom = ipd.sym.xtal.cellgeom_from_lattice(cellgeom)
    a, b, c, A, B, C = full_cellgeom(spacegroup, cellgeom)
    cosA, cosB, cosC = [np.cos(np.radians(_)) for _ in (A, B, C)]
    sinB, sinC = [np.sin(np.radians(_)) for _ in (B, C)]
    return a * b * c * np.sqrt(1 - cosA**2 - cosB**2 - cosC**2 + 2*cosA*cosB*cosC)

def sg_is_chiral(sg):
    return sg in sg_all_chiral
    # if sg == '231': return False
    # return not any([sg.count(x) for x in 'm-c/n:baHd'])

def sg_pymol_name(spacegroup):
    if spacegroup == "R3":
        return "H3"
    if spacegroup == "R32":
        return "H32"
    return spacegroup

sg_nonsingular_cellgeom = [1.0, 1.3, 1.7, 66.0, 85.0, 104.0]
