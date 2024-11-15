import itertools

import numpy as np

import ipd
from ipd.sym.xtal.xtalinfo import _xtal_asucens

cryst1_pattern = "CRYST1 %8.3f %8.3f %8.3f  90.00  90.00  90.00 %s\n"
cryst1_pattern_full = "CRYST1 %8.3f %8.3f %8.3f%7.2f%7.2f%7.2f %s\n"
# CRYST1  150.000  150.000  150.000  90.00  90.00  90.00 I 21 3
# CRYST1  139.291  139.291  139.291  90.00  90.00  90.00 I 21 3

_xtal_cache = {}

def xtal(sym, **kw):
    global _xtal_cache
    if sym not in _xtal_cache:
        _xtal_cache[sym] = ipd.sym.xtal.Xtal(sym, **kw)
    return _xtal_cache[sym]

class Xtal:
    def __init__(self, name="xtal", symelems=None, **kw):
        self.info = None
        self.symelems = symelems
        if symelems is None:
            self.name, self.info = ipd.sym.xtal.xtalinfo(name)
            if "dimension" not in self.info:
                self.info.dimension = 3
            self.symelems = self.info.symelems
            self.dimension = self.info.dimension
            self.spacegroup = self.info.spacegroup
            self.nsub = self.info.nsub
        self.compute_frames(**kw)
        # self.symelems_to_unitcell()
        self.nprimaryframes = len(self.primary_frames())

    def cellframes(
        self,
        cellsize=1,
        cells=1,
        flat=True,
        center=None,
        asucen=None,
        xtalrad=None,
        ontop="primary",
        **kw,
    ):
        ipd.dev.checkpoint(kw, funcbegin=True)
        if "radius" in kw:
            ipd.WARNME("ipd.sym.xtal.Xtal.cellframes: radius not valid arg")
        if "maxdist" in kw:
            ipd.WARNME("ipd.sym.xtal.Xtal.cellframes: maxdist not valid arg")

        if self.dimension != 3:
            frames = _scaled_frames(cellsize, self.unitframes)
            if ontop == "primary":
                ontop = self.primary_frames(cellsize)
            if ontop is not None and len(ontop) > 0:
                frames = ipd.sym.put_frames_on_top(frames, ontop, cellsize=cellsize, **kw)
            return frames
        if cells is None:
            return np.eye(4)[None]
        if isinstance(cells, (int, float)):
            ub = cells // 2
            lb = ub - cells + 1
            cells = list(itertools.product(*[range(lb, ub + 1)] * 3))

        elif len(cells) == 2:
            lb, ub = cells  # type: ignore
            cells = list(itertools.product(*[range(lb, ub + 1)] * 3))
        elif len(cells) == 3:
            cells = list(
                itertools.product(
                    range(cells[0][0], cells[0][1] + 1),
                    range(cells[1][0], cells[1][1] + 1),
                    range(cells[2][0], cells[2][1] + 1),
                ))
        else:
            raise ValueError(f"bad cells {cells}")

        cells = ipd.homog.hpoint(cells)
        cells = cells.reshape(-1, 4)
        xcellshift = ipd.homog.htrans(cells)
        frames = self.unitframes.copy()
        frames = ipd.homog.hxform(xcellshift, frames)
        frames = _scaled_frames(cellsize, frames)
        if flat:
            frames = frames.reshape(-1, 4, 4)

        if xtalrad is not None:
            if xtalrad <= 3:
                xtalrad = xtalrad * np.linalg.norm(np.array(cellsize))
            if asucen is None:
                asucen = self.asucen(cellsize)
            if center is None:
                center = asucen
            ic(center, asucen)  # type: ignore
            # if xtalrad is None: xtalrad = 0.5 * cellsize # ???
            center = ipd.homog.hpoint(center)
            asucen = ipd.homog.hpoint(asucen)
            pos = ipd.homog.hxform(frames, asucen)
            dis = ipd.homog.hnorm(pos - center)
            # ic(dis.shape)
            # ic(xtalrad)
            # ic(center)
            frames = frames[dis <= xtalrad]

        if ontop == "primary":
            ontop = self.primary_frames(cellsize)
        if ontop is not None and len(ontop) > 0:
            frames = ipd.sym.put_frames_on_top(frames, ontop, cellsize=cellsize, **kw)

        assert ipd.homog.hnuique(frames)
        ipd.dev.checkpoint(kw)

        return frames.round(10)

    def fit_coords(self, *a, **kw):
        return ipd.sym.xtal.xtalfit.fix_coords_to_xtal(self.name, *a, **kw)

    def fit_xtal_to_coords(self, coords, **kw):
        return ipd.sym.xtal.xtalfit.fix_xtal_to_coords(self, coords, **kw)

    def coords_to_asucen(self, coords, cells=5, frames=None, asucen=None, **kw):
        coords = ipd.homog.hpoint(coords)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 4)
        if asucen is None:
            asucen = self.asucen(**kw)
        if frames is None:
            frames = self.cellframes(cells=cells, **kw)
        coordcen = ipd.homog.hcom(coords)
        cens = ipd.homog.hxform(frames, coordcen)
        dist = ipd.homog.hnorm2(asucen - cens)
        bestframe = frames[np.argmin(dist)]
        return ipd.homog.hxform(bestframe, coords)

    def symelems_to_unitcell(self):
        for s in self.symelems:  # type: ignore
            if s.cen[0] < 0:
                s.cen[0] += 1
            if s.cen[1] < 0:
                s.cen[1] += 1
            if s.cen[2] < 0:
                s.cen[2] += 1
            if s.cen[0] > 1:
                s.cen[0] -= 1
            if s.cen[1] > 1:
                s.cen[1] -= 1
            if s.cen[2] > 1:
                s.cen[2] -= 1

    def primary_frames(self, cellsize=1, contacting_only=False, **kw):
        "frames generated by the primary sym elems"
        ipd.dev.checkpoint(kw, funcbegin=True)
        frames = [np.eye(4)]
        # ic(self.name)
        for s in self.symelems:  # type: ignore
            f = s.operators[[1, -1]] if contacting_only else s.operators[1:]
            frames += list(f)
        frames = np.stack(frames)
        scaleframes = _scaled_frames(cellsize, frames)
        ipd.dev.checkpoint(kw)
        return scaleframes.round(10)

    def asucen(
        self,
        cellsize=1,
        use_olig_nbrs=False,
        olig_nbr_wt=0.75,
        xtalasumethod=None,
        **kw,
    ):
        "should be a point roughly in the center of the primary sym elems"

        if xtalasumethod is None:
            if "method" in kw:
                ipd.WARNME('DEPRICATION WARNING Xtal.asucen "method" arg replaced by "xtalasumethod"')
                xtalasumethod = kw["method"]
            else:
                xtalasumethod = "stored" if self.name in _xtal_asucens else "closest_approach"

        if xtalasumethod == "stored":
            if self.name in _xtal_asucens:
                cen = _xtal_asucens[self.name]
            elif self.spacegroup in _xtal_asucens:
                cen = _xtal_asucens[self.spacegroup]
            else:
                raise ValueError(f'no stored asucen info for "{self.name}" or "{self.spacegroup}"')
            cen = _scaled_frames(cellsize, cen)
            # ic(cen)
            return cen
        elif xtalasumethod == "closest_to_cen":
            elems = self.symelems
            # ic([e.cen for e in elems])
            cen0 = np.mean(np.stack([e.cen for e in elems]), axis=0)  # type: ignore
            # ic(cen0)
            # assert 0
        elif xtalasumethod == "closest_approach":
            cens = []
            for i1, elem1 in enumerate(self.symelems):  # type: ignore
                p1, ax1 = elem1.cen, elem1.axis
                for i2, elem2 in enumerate(self.symelems):  # type: ignore
                    if i2 <= i1:
                        continue
                    p2, ax2 = elem2.cen, elem2.axis
                    if ipd.hparallel(ax1, ax2):
                        p, q = elem1.cen, elem2.cen
                    else:
                        p, q = ipd.homog.line_line_closest_points_pa(p1, ax1, p2, ax2)
                    cens.append((p+q) / 2)
            cen0 = np.mean(cens, axis=0)
        else:
            raise ValueError(f"unknown asucen xtalasumethod {xtalasumethod}")
        if use_olig_nbrs:
            opcens = [np.mean(ipd.homog.hxform(e.operators[1:], cen0), axis=0) for e in self.symelems]  # type: ignore
            cen = np.mean(np.stack(opcens), axis=0)
            # this is arbitrary
            cen = olig_nbr_wt*cen + (1-olig_nbr_wt) * cen0
        else:
            cen = cen0
        return _scaled_frames(cellsize, cen)

    def compute_frames(self, **kw):
        """This should be depricated, only needed for 2d stuff currently."""
        if self.dimension == 3:
            self.unitframes = self.info.frames  # type: ignore
            self.genframes = self.cellframes(cells=3, ontop=None)  # type: ignore
            self.coverelems = self.generate_cover_symelems(self.genframes)
            self.unitelems = self.generate_unit_symelems(self.genframes)
        else:
            self.genframes = self.generate_candidate_frames(bound=1.5, **kw)  # expensive-ish
            self.coverelems = self.generate_cover_symelems(self.genframes, bbox=None)
            self.unitframes = self.genframes
            self.unitelems = self.coverelems
        self.nsub = len(self.unitframes)
        if self.info is not None and self.info.nsub is not None:
            assert (self.nsub == self.info.nsub), f'nsub for "{self.name}" should be {self.info.nsub}, not {self.nsub}'

    def symcoords(self, asymcoords, cellsize=1, cells=1, flat=False, center=None, **kw):
        asymcoords = ipd.homog.hpoint(asymcoords)
        if center is None:
            center = ipd.homog.hcom(asymcoords.reshape(-1, asymcoords.shape[-1]))
        if cells == 0:
            frames = self.primary_frames(cellsize)
        else:
            frames = self.cellframes(cellsize=cellsize, cells=cells, center=center, **kw)
        coords = ipd.homog.hxform(frames, asymcoords)
        if flat:
            coords = coords.reshape(-1, 4)
        return coords

    def cryst1(self, cellsize):
        if self.dimension != 3:
            return f"LAYER {self.spacegroup} {cellsize}"
        if isinstance(cellsize, (int, float, np.int32, np.int64, np.float32, np.float64)):  # type: ignore
            cellsize = np.array([cellsize] * 3, dtype=np.float64)
        if len(cellsize) == 3:
            return cryst1_pattern % (*cellsize, self.spacegroup)
        elif len(cellsize) == 6:
            return cryst1_pattern_full % (*cellsize, self.spacegroup)
        else:
            raise ValueError(f"bad cellsize {cellsize}")

    def dump_pdb(self, fname, asymcoords=None, cellsize=1, cells=None, strict=False, **kw):
        cryst1 = self.cryst1(cellsize)
        if asymcoords is None:
            asymcoords = self.asucen(cellsize=cellsize, **kw).reshape(1, 4)
        assert asymcoords.ndim in (2, 3)
        asymcoords = ipd.homog.hpoint(asymcoords)
        if cells is None:
            nchain = kw.get("nchain", 1)
            natom = 1 if asymcoords.ndim == 2 else asymcoords.shape[1]
            if nchain > 1:
                asymcoords = asymcoords.reshape(nchain, -1, natom, 4)
            ipd.pdb.dumppdb(fname, asymcoords, header=cryst1, nchain=nchain)
        else:
            if asymcoords.ndim == 2:
                asymcoords = asymcoords.reshape(-1, 1, asymcoords.shape[-1])
            coords = self.symcoords(asymcoords, cellsize, cells, strict=strict, **kw)
            assert np.allclose(coords[0], asymcoords)

            ipd.pdb.dump_pdb_from_points(fname, coords)

    @property
    def nunit(self):
        return len(self.unitframes)

    def central_symelems(self, cells=3, target=None):
        assert 0, "WARNING central_symelems is buggy, axis direction wrong!?!"
        # assert 0
        # _targets = {'I 41 3 2': [0.3, 0.4, 0.5]}
        # if target is None:
        # if self.spacegroup in _targets:
        # target = _targets[self.spacegroup]
        # else:
        # target = [0.3, 0.4, 0.5]
        target = ipd.homog.hpoint(target)
        cenelems = []
        cells = interp_xtal_cell_list(cells)
        for elems in self.coverelems:
            best = 9e9, None
            for cellshift in cells:
                xcellshift = ipd.homog.htrans(cellshift)
                for elem in elems:
                    elem = elem.xformed(xcellshift)
                    # ic(i, j, elem.cen)
                    d = ipd.homog.hnorm(elem.cen - target)
                    if d < best[0]:
                        best = d, elem
            cenelems.append(best[1])
        return cenelems

    def frames(self, cells=3, xtalrad=None, **kw):
        ipd.dev.checkpoint(kw, funcbegin=True)
        if cells is None or xtalrad == 0:
            frames = self.primary_frames(**kw)
        else:
            frames = self.cellframes(cells=cells, xtalrad=xtalrad, **kw)
        ipd.dev.checkpoint(kw, funcbegin=True)
        return frames

    def generate_candidate_frames(
        self,
        depth=200,
        bound=3.0,
        genradius=9e9,
        trials=20000,
        # cache=True,
        # cache='nosave',
        **kw,
    ):
        cachefile = ipd.dev.package_data_path(f'xtal/lots_of_frames_{self.name.replace(" ","_")}.npy')
        if self.dimension == 2:
            generators = np.concatenate([s.operators for s in self.symelems])  # type: ignore
            x, _ = ipd.homog.hcom.geom.expand_xforms_rand(generators, depth=depth, radius=genradius, trials=trials)
            testpoint = [0.001, 0.002, 0.003]
            cens = ipd.homog.hxform(x, testpoint)
            inboundslow = np.all(cens >= -bound - 0.001, axis=-1)
            inboundshigh = np.all(cens <= bound + 0.001, axis=-1)
            inbounds = np.logical_and(inboundslow, inboundshigh)
            x = x[inbounds]
            assert len(x) < 10000
            # if self.dimension == 3 and cache and cache != "nosave":
            # np.save(cachefile, x)
        else:
            assert 0, 'should only be called for 2D stuff'
        return x  # type: ignore

    def generate_unit_frames(self, candidates, bbox=None, testpoint=None):
        if bbox is None: bbox = [(0, 0, 0), (1, 1, 1)]
        bbox = np.asarray(bbox)
        testpoint = testpoint if testpoint else [0.001, 0.002, 0.003]
        cens = ipd.homog.hxform(candidates, testpoint)
        inboundslow = np.all(cens >= bbox[0] - 0.0001, axis=-1)
        inboundshigh = np.all(cens <= bbox[1] + 0.0001, axis=-1)
        inbounds = np.logical_and(inboundslow, inboundshigh)
        return candidates[inbounds]

    def generate_unit_symelems(self, candidates):
        unitelems = []
        for i, symelem in enumerate(self.symelems):  # type: ignore
            elems = symelem.xformed(self.unitframes)
            for e, f in zip(elems, self.unitframes):
                e.origin = f
            unitelems.append(elems)
        return unitelems

    def generate_cover_symelems(self, candidates, bbox=None):
        if bbox is None: bbox = [(0, 0, 0), (1, 1, 1)]
        coverelems = []
        bbox = np.asarray(bbox) if bbox is not None else None
        for i, symelem in enumerate(self.symelems):  # type: ignore
            testpoint = symelem.cen[:3]
            cens = ipd.homog.hxform(candidates, testpoint)
            frames = candidates
            if bbox is not None:
                inboundslow = np.all(cens >= bbox[0] - 0.0001, axis=-1)
                inboundshigh = np.all(cens <= bbox[1] + 0.0001, axis=-1)
                inbounds = np.logical_and(inboundslow, inboundshigh)
                frames = candidates[inbounds]
            elems = symelem.xformed(frames)
            # ic(len(elems))
            assert len(elems) == len(frames), f"{len(elems)} {len(frames)}"
            for e, f in zip(elems, frames):
                # assert np.allclose(np.eye(4), e.origin)
                e.origin = f
            coverelems.append(elems)
        return coverelems

def interp_xtal_cell_list(cells):
    if cells is None:
        return np.eye(4)[None]
    if isinstance(cells, (int, float)):
        ub = cells // 2
        lb = ub - cells + 1
        cells = list(itertools.product(*[range(lb, ub + 1)] * 3))  # type: ignore
    elif len(cells) == 2:
        lb, ub = cells
        cells = list(itertools.product(*[range(lb, ub + 1)] * 3))
    elif len(cells) == 3:
        cells = list(
            itertools.product(
                range(cells[0][0], cells[0][1] + 1),
                range(cells[1][0], cells[1][1] + 1),
                range(cells[2][0], cells[2][1] + 1),
            ))
    else:
        raise ValueError(f"bad cells {cells}")
    return cells

_warn_asucen_xtalasumethod = True

def _scaled_frames(cellsize, frames):
    if isinstance(cellsize, (int, float, np.int64)):  # type: ignore
        cellsize = np.array([cellsize, cellsize, cellsize])
    if len(cellsize) == 6:
        assert np.all(cellsize[3:] == 90)
    return ipd.hscaled(cellsize[:3], frames)
