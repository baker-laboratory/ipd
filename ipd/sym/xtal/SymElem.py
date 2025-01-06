from collections import defaultdict

import numpy as np
from opt_einsum import contract as einsum

import ipd
from ipd.homog.hgeom import (
    angle,
    halign,
    halign2,
    hdot,
    hinv,
    hnorm,
    hnormalized,
    hpoint,
    htrans,
    hvec,
    hxform,
)
from ipd.sym.symframes import octahedral_frames, tetrahedral_frames
from ipd.sym.xtal.spacegroup_util import applylatticepts, lattice_vectors

class ScrewError(Exception):
    pass

class ComponentIDError(Exception):
    pass

class OutOfUnitCellError(Exception):
    pass

class SymElemAngErr(Exception):
    pass

_HACK_HASH = []

def symelem_of(frame, **kw):
    a, an, c, h = ipd.homog.axis_angle_cen_hel_of(frame)
    if np.isclose(an, 0):
        nfold = 1
    else:
        nfold = np.pi * 2 / an
        ic(nfold, a, an, c, h)
        if not np.allclose(nfold, nfold.round()):
            raise SymElemAngErr(f"angle {an} implies nfold {nfold} which is non-integer")
    # ic(int(nfold), a, c, h, kw)
    return SymElem(int(nfold.round()), axis=a, cen=c, hel=h, **kw)  # type: ignore

def _round(val):
    for v in [
            0,
            0.125,
            1 / 6,
            0.25,
            1 / 3,
            0.375,
            0.5,
            np.sqrt(3) / 3,
            0.625,
            2 / 3,
            np.sqrt(2) / 2,
            0.75,
            5 / 6,
            np.sqrt(3) / 3,
            0.875,
            1,
    ]:
        if isinstance(val, np.ndarray):
            val[np.isclose(val, v, atol=0.0001)] = v
        else:
            if np.isclose(val, v):
                val = v
    return val

class SymElem:
    def __init__(
        self,
        nfold,
        axis,
        cen=[0, 0, 0],
        axis2=None,
        *,
        label=None,
        vizcol=None,
        scale=1,
        parent=None,
        children=None,
        hel=0,
        lattice=None,
        screw=None,
        adjust_cyclic_center=False,
        frame=None,
        isunit=None,
        latticetype=None,
    ):
        self._init_args = ipd.dev.Bunch(vars()).without("self")
        self.vizcol = vizcol
        self.scale = scale

        self._set_geometry(frame, nfold, axis, cen, axis2, hel, lattice, isunit, adjust_cyclic_center)
        self._check_screw(screw, latticetype)
        self._make_label(label)
        self._set_kind()

        self.mobile = False
        if ipd.homog.hgeom.h_point_line_dist([0, 0, 0], cen, axis) > 0.0001:
            self.mobile = True
        if axis2 is not None and ipd.homog.hgeom.h_point_line_dist([0, 0, 0], cen, axis2) > 0.0001:
            self.mobile = True
        self.operators = self.make_operators()
        self.numops = len(self.operators)
        self.parent = parent
        self.children = children or list()

        self.numeric_cleanup()
        self.issues = []

    def __hash__(self):
        global _HACK_HASH
        for i, h in enumerate(_HACK_HASH):
            if self == h:
                return i
        _HACK_HASH.append(self)
        return len(_HACK_HASH) - 1

    def numeric_cleanup(self):
        # self.axis = self.axis.round(9)
        # self.cen = self.cen.round(9)
        # if self.axis2 is not None: self.axis2 = self.axis2.round(9)
        # self.hel = self.hel.round(9)
        # if hdot(self.axis, [3, 2, 1, 0]) < 0: self.axis = -self.axis
        self.axis = _round(self.axis)
        if self.axis2 is not None:
            if hdot(self.axis2, [3, 2, 1, 0]) < 0:
                self.axis2 = -self.axis2
            self.axis2 = _round(self.axis2)
        self.cen = _round(self.cen)
        self.hel = _round(self.hel)
        # self.index = None
        if self.axis2 is not None:
            self.axis2 = ipd.homog.hnormalized(self.axis2)
            self.axis2[self.axis2 == -0] = 0
        # if not self.isscrew:
        if True:
            if angle(self.axis, [1, 1.1, 1.2]) > np.pi / 2:
                self.axis = -self.axis
            if self.axis2 is not None and angle(self.axis2, [1, 1.1, 1.2]) > np.pi / 2:
                self.axis2 = -self.axis2

    def _set_nfold(self, nfold):
        if isinstance(nfold, str):
            self.label = nfold[:-2]  # strip componend nfolds
            if nfold[0] in "CD":
                nfold = int(nfold[1:-2])
            else:
                self._opinfo = int(nfold[-2]), int(nfold[-1])
                assert nfold[0] in "TO"
                nfold = dict(T=12, O=24)[nfold[0]]
        self.nfold = nfold

    def frame_operator_ids(self, frames, sanitycheck=True):
        # ic(self)
        # from ipd.viz.pymol_viz import showme
        # showme(10 * einsum('fij,j->fi', frames, [0, 0, 0, 1]), scale=10)
        # showme(10 * einsum('fij,j->fi', frames, [0.5, 0.5, 0.5, 1]), scale=10)
        # showme(10 * einsum('fij,j->fi', frames, [1, 1, 1, 1]), scale=10)
        # assert 0
        opids = np.arange(len(frames), dtype=np.int32)
        opsframes = einsum("oij,fjk->ofik", self.operators[1:], frames)
        for iop, opframes in enumerate(opsframes):  # type: ignore
            a, b = np.where(np.all(np.isclose(frames[None], opframes[:, None]), axis=(-1, -2)))
            opids[a] = np.minimum(opids[a], opids[b])
        for i, id in enumerate(sorted(set(opids))):
            opids[opids == id] = i
        if sanitycheck and self.iscompound:
            for i in range(np.max(opids)):
                ids = opids == i
                # showme(frames[ids])
                if np.sum(ids) == len(self.operators):
                    assert np.allclose(self.cen, frames[ids, :, 3].mean(axis=0))
        return opids

    def frame_component_ids(self, frames, permutations, sym=None, sanitycheck=True):
        if self.iscompound:
            # compound elements (D2, T, etc) will never overlap their centers
            return self.frame_component_ids_bycenter(frames, sanitycheck)
        assert len(permutations) == len(frames)
        opframes = np.eye(4).reshape(1, 4, 4)
        iframematch0 = self.matching_frames(frames)
        # ic(iframematch0, len(frames), permutations.shape)
        fid = 0
        compid = -np.ones(len(frames), dtype=np.int32)
        for iframe, perm in enumerate(permutations):
            try:
                iframematch = perm[iframematch0]
            except IndexError:
                # from ipd.viz.pymol_viz import showme
                # import ipd.viz.viz_xtal
                # showme(self, scale=10)
                # showme(frames, scale=10)
                # assert 0
                raise ComponentIDError
            iframematch = iframematch[iframematch >= 0]
            centest = einsum("fij,j->fi", frames[iframematch], self.cen)
            axstest = einsum("fij,j->fi", frames[iframematch], self.axis)
            if sanitycheck and self.iscompound:
                assert np.allclose(centest, centest[0])  # type: ignore
                if not (self.istet or self.isoct):
                    assert np.all(  # type: ignore
                        np.logical_or(  # type: ignore
                            np.all(np.isclose(axstest, axstest[0]), axis=1),  # type: ignore
                            np.all(np.isclose(axstest, -axstest[0]), axis=1),  # type: ignore
                        ))
            if np.allclose(compid[iframematch], -1):
                compid[iframematch] = fid
                fid += 1
            else:
                assert min(compid[iframematch]) == max(compid[iframematch])

        if sanitycheck and not self.iscyclic:
            _sanitycheck_compid_cens(self, frames, compid)

        return compid

    def frame_component_ids_bycenter(self, frames, sanitycheck=True):
        assert self.iscompound
        cen = einsum("fij,j->fi", frames, self.cen)
        d = hnorm(cen[:, None] - cen[None])  # type: ignore
        compid = np.ones(len(frames), dtype=np.int32) * -12345
        count = 0
        for i in range(len(d)):
            if compid[i] >= 0:
                continue
            w = np.where(np.isclose(d[i], 0))
            assert len(w) <= len(self.operators)
            compid[w] = count
            count += 1
        if sanitycheck:
            _sanitycheck_compid_cens(self, frames, compid)
        # w = np.isclose(d, 0)[12]
        # ipd.showme(cen[w], scale=10)
        # ipd.showme(frames[w], scale=10)
        # assert 0
        # ic(count)
        # s = set(compid)
        # for i in range(np.max(compid)):
        # if not i in s:
        # ic(i)
        # assert len(set(compid)) == np.max(compid) + 1
        # assert 0
        return compid

    def tolattice(self, latticevec, tounit=False):
        assert self.isunit is None or self.isunit != tounit
        if tounit:
            assert latticevec[0, 0] <= 1
        else:
            assert latticevec[0, 0] >= 1
        newcen = applylatticepts(latticevec, self.cen)
        newhel = applylatticepts(latticevec, self.cen + self.axis * self.hel)
        newhel = hnorm(newhel - newcen)
        newelem = SymElem(self._init_args.nfold,
                          self.axis,
                          newcen,
                          self.axis2,
                          hel=newhel,
                          screw=self.screw,
                          isunit=tounit)
        assert self.operators.shape == newelem.operators.shape
        return newelem

    def tounit(self, latticevec):
        return self.tolattice(np.linalg.inv(latticevec), tounit=True)

    def matching_frames(self, frames):
        "find frames related by self.operators that are closest to cen"
        match = np.isclose(frames[None], self.operators[:, None])
        match = np.any(np.all(match, axis=(2, 3)), axis=0)
        match = np.where(match)[0]
        if len(match) != len(self.operators):
            cperr = ComponentIDError()
            cperr.match = match  # type: ignore
            raise cperr
            ic(frames.shape)
            ic(match)
            from ipd.viz.pymol_viz import showme

            showme(frames, scale=10)
            showme(self, scale=10)
            assert len(match) == len(self.operators)
        return match

        # symaxs = einsum('fij,j->fi', frames, self.axis)
        # symcen = einsum('fij,j->fi', frames, self.cen)
        # match = np.logical_and(
        # np.all(np.isclose(self.axis, symaxs), axis=1),
        # np.all(np.isclose(self.cen, symcen), axis=1),
        # )
        # w = np.where(match)[0]
        # return w

    def _make_label(self, label):
        if hasattr(self, "label"):
            return
        self.label = label
        if self.label is None:
            if self.axis2 is None:
                self.label = f"C{self.nfold}"
                if self.screw != 0:
                    self.label += f"{self.screw}"
            else:
                self.label = f"D{self.nfold}"

    def __eq__(self, other):
        if self.nfold != other.nfold:
            return False
        if not np.allclose(self.axis, other.axis):
            return False
        if self.axis2 is not None and not np.allclose(self.axis2, other.axis2):
            return False
        # if not np.allclose(np.abs(self.axis), np.abs(other.axis)): return False
        # if self.axis2 is not None and not np.allclose(np.abs(self.axis2), np.abs(other.axis2)): return False
        if not np.allclose(self.cen, other.cen):
            return False
        if not np.allclose(self.hel, other.hel):
            return False
        if not np.allclose(self.screw, other.screw):
            return False
        assert np.allclose(self.operators, other.operators)
        return True

    def _check_screw(self, screw, latticetype):
        if screw is not None:
            self.screw = screw
            return
        if self.hel == 0.0:
            self.screw = 0
            return

        assert not self.axis2

        hel0 = self.hel
        # self.hel = float(self.hel)

        s2 = np.sqrt(2)
        s3 = np.sqrt(3)
        axtype = list(sorted(np.abs(self.axis[:3])))

        if np.allclose(axtype, [0, 0, 1]):
            cellextent = 1.0
        elif np.allclose(axtype, [0, s2 / 2, s2 / 2]):
            cellextent = s2
            if latticetype in ["HEXAGONAL"]:
                raise ScrewError()
        elif np.allclose(axtype, [s3 / 3, s3 / 3, s3 / 3]):
            cellextent = s3
            if latticetype in ["HEXAGONAL"]:
                raise ScrewError()
        else:
            raise ScrewError(f"cant understand axis {self.axis}")
        unitcellfrac = self.hel / cellextent

        # raise ScrewError(f'incoherent screw values axis: {self.axis} hel: {self.hel} screw: {self.screw}')
        # ic(unitcellfrac)
        # if not 0 < unitcellfrac < (1 if self.nfold > 1 else 1.001):
        # raise ScrewError(f'screw translation out of unit cell')
        self.screw = unitcellfrac * self.nfold
        # ic(self.nfold, self.screw)
        if not np.isclose(self.screw, round(self.screw)):  # type: ignore
            raise ScrewError(f"screw has non integer value {self.screw}")
        if self.screw >= max(2, self.nfold):  # C11 is ok
            raise ScrewError("screw dosent match nfold")

        self.screw = int(round(self.screw))  # type: ignore
        if self.nfold > 1:
            self.screw = self.screw % self.nfold
            self.hel = self.hel % cellextent
            # if unitcellfrac < -0.9:
            # raise ScrewError(f'unitcellfrac below -0.9')
            # ic(self.nfold, unitcellfrac)
            if self.hel < 0:
                assert 0

        # if self.screw == 3 and self.nfold == 4:
        # self.screw = self.nfold - self.screw
        # self.axis = -self.axis
        # self.hel = -self.hel

        # if self.nfold == 3:
        # if np.min(np.abs(self.axis)) < 0.1:
        # self.hel = self.hel % 1
        # else:
        # self.hel = self.hel % np.sqrt(3)
        # elif self.nfold == 4:
        # self.hel = self.hel % 1.0
        # elif self.nfold == 6:
        # self.hel = self.hel % 1.0
        # assert self.screw <= self.nfold / 2

        # assert 0

    def _set_geometry(self, frame, nfold, axis, cen, axis2, hel, lattice, isunit, adjust_cyclic_center):
        axis = hvec(axis)
        self.isunit = isunit
        cen = hpoint(cen)
        self._set_nfold(nfold)
        self.angle = np.pi * 2 / self.nfold
        self.frame = frame
        if frame is not None:
            assert axis2 is None
            a, an, c, h = ipd.homog.axis_angle_cen_hel_of(frame)
            ic(a, axis, c, cen, nfold, an, h, hel)
            assert np.allclose(axis, a)
            assert np.allclose(an, 0) or np.allclose(nfold, 2 * np.pi / an)
            assert np.allclose(cen, c)
            assert np.allclose(hel, h)

        if lattice is None:
            lattice = np.eye(3)
        invlattice = np.linalg.inv(lattice)
        self.axis = hnormalized(_mul3(invlattice, axis))
        self.cen = hpoint(_mul3(invlattice, hpoint(cen)))
        self.axis2 = None if axis2 is None else hnormalized(_mul3(invlattice, hvec(axis2)))
        heltrans = _mul3(invlattice, cen + hnormalized(axis) * hel) - _mul3(invlattice, cen)
        self.hel = hdot(self.axis, heltrans)

        if hdot(self.axis, [1, 2, 3, 0]) < 0:
            self.axis = -self.axis
        if axis2 is not None and hdot(self.axis2, [1, 2, 3, 0]) < 0:
            self.axis2 = -self.axis2  # type: ignore
        if adjust_cyclic_center and (axis2 is None) and np.isclose(hel, 0):  # cyclic
            assert 0, "this needs an audit"
            dist = ipd.homog.line_line_distance_pa(self.cen, self.axis, _cube_edge_cen * self.scale, _cube_edge_axis)
            w = np.argmin(dist)
            newcen, _ = ipd.homog.line_line_closest_points_pa(self.cen, self.axis, _cube_edge_cen[w] * self.scale,
                                                              _cube_edge_axis[w])
            if not np.any(np.isnan(newcen)):
                self.cen = newcen

    def _set_kind(self):
        self.iscyclic, self.isdihedral, self.istet, self.isoct, self.isscrew, self.iscompound = [False] * 6
        if self.label == "T":
            self.kind, self.istet, self.iscompound = "tet", True, True
        elif self.label == "O":
            self.kind, self.isoct, self.iscompound = "oct", True, True
        elif not np.isclose(self.hel, 0):
            assert self.axis2 is None
            self.kind, self.isscrew = "screw", True
        elif self.axis2 is not None:
            self.kind, self.isdihedral, self.iscompound = "dihedral", True, True
        else:
            self.kind, self.iscyclic = "cyclic", True

    def make_operators_screw(self):
        if not self.isscrew:
            return self.operators
        return np.stack([
            np.eye(4),
            ipd.homog.htrans(self.axis[:3] * -self.hel),
            ipd.homog.htrans(self.axis[:3] * self.hel),
        ])

    def make_operators(self):
        # ic(self)
        if self.label == "T":
            ops = tetrahedral_frames
            assert self._opinfo == (3, 2)
            aln = halign2([1, 1, 1], [0, 0, 1], self.axis, self.axis2)
            ops = aln @ ops @ hinv(aln)
            ops = htrans(self.cen) @ ops @ htrans(-self.cen)
            # ic(self.axis, self.axis2)
        elif self.label == "O":
            assert self._opinfo in [(4, 3), (4, 2), (3, 2)]
            ops = octahedral_frames
            ops = htrans(self.cen) @ ops @ htrans(-self.cen)
        else:
            x = ipd.homog.hgeom.hrot(self.axis, nfold=self.nfold, center=self.cen)
            ops = [ipd.homog.hgeom.hpow(x, p) for p in range(self.nfold)]
            if self.axis2 is not None:
                xd2f = ipd.homog.hgeom.hrot(self.axis2, nfold=2, center=self.cen)
                ops = ops + [xd2f @ x for x in ops]
            if self.hel != 0.0:
                for i, x in enumerate(ops):
                    x[:, 3] += self.axis * self.hel * i
            ops = np.stack(ops)
        assert ipd.homog.hgeom.hvalid(ops)

        if not self.isdihedral:
            self.origin = htrans(self.cen) @ halign([0, 0, 1], self.axis)
        else:
            # ic(self.axis)
            # ic(self.axis2)
            self.origin = htrans(self.cen) @ halign2([0, 0, 1], [1, 0, 0], self.axis, self.axis2)

        return ops

    @property
    def coords(self):
        axis2 = [0, 0, 0, 0] if self.axis2 is None else self.axis2
        return np.stack([self.axis, axis2, self.cen])

    def xformed(self, xform):
        assert xform.shape[-2:] == (4, 4)
        single = False
        if xform.ndim == 2:
            xform = xform.reshape(1, 4, 4)
            single = True
        result = list()
        for x in xform:
            # other = copy.copy(self)
            # other.axis = ipd.homog.hxform(x, self.axis)
            # if self.axis2 is not None: other.axis2 = ipd.homog.hxform(x, self.axis2)
            # other.cen = ipd.homog.hxform(x, self.cen)
            # other.make_operators()
            axis = hxform(x, self.axis)
            axis2 = None if self.axis2 is None else hxform(x, self.axis2)
            cen = hxform(x, self.cen)
            other = SymElem(
                self._init_args.nfold,
                axis,
                cen,
                axis2,
                label=self.label,
                vizcol=self.vizcol,
                scale=1,
                screw=self.screw,
            )
            result.append(other)
        if single:
            result = result[0]
        return result

    def __repr__(self):
        ax = (self.axis / np.max(np.abs(self.axis))).round(6)
        cn = (self.cen / np.max(np.abs(self.cen))).round(6)
        axs = ', '.join([f'{float(_):7.4f}' for _ in ax[:3]])
        cen = ', '.join([f'{float(_):7.4f}' for _ in cn[:3]])
        if np.allclose(ax.round(), ax):
            ax = ax.astype("i")
        if self.istet:
            ax2 = (self.axis2 / np.max(np.abs(self.axis2))).round(6)  # type: ignore
            s = f"SymElem('{self._init_args.nfold}', axis=[{axs}], axis2={[float(_) for _ in ax2[:3]]}, cen={[float(_) for _ in self.cen[:3]]}, label='{self.label}')"
        elif self.isoct:
            ax2 = (self.axis2 / np.max(np.abs(self.axis2))).round(6)  # type: ignore
            s = f"SymElem('{self._init_args.nfold}', axis=[{axs}], axis2={[float(_) for _ in ax2[:3]]}, cen={[float(_) for _ in self.cen[:3]]}, label='{self.label}')"
        elif self.axis2 is None:
            if self.screw == 0:
                s = f"SymElem({self.nfold}, axis=[{axs}], cen={[float(_) for _ in self.cen[:3]]}, label='{self.label}')"
            else:
                s = f"SymElem({self.nfold}, axis=[{axs}], cen={[float(_) for _ in self.cen[:3]]}, hel={self.hel}, label='{self.label}')"
        else:
            ax2 = (self.axis2 / np.max(np.abs(self.axis2))).round(6)
            s = f"SymElem({self.nfold}, axis=[{axs}], axis2={[float(_) for _ in ax2[:3]]}, cen={[float(_) for _ in self.cen[:3]]}, label='{self.label}')"
        # s = s.replace('0.0,', '0,').replace('0.0],', '0]')
        return s

    def __str__(self):
        ax = (self.axis / np.max(np.abs(self.axis))).round(6)
        cn = (self.cen / np.max(np.abs(self.cen))).round(6)
        axs = ' '.join([f'{float(_):7.4f}' for _ in ax[:3]])
        cen = ' '.join([f'{float(_):7.4f}' for _ in cn[:3]])
        if self.axis2 is not None:
            a2 = (self.axis / np.max(np.abs(self.axis2))).round(6)
            ax2 = ' '.join([f'{float(_):7.4f}' for _ in a2[:3]])
        if self.istet:
            s = f"SymElem {self.label:3} {self._init_args.nfold:3}  AXS {axs}  CEN {cen}  AXS2 {axs}"
        elif self.isoct:
            s = f"SymElem {self.label:3} {self._init_args.nfold:3}  AXS {axs}  CEN {cen}  AXS2 {axs}"
        elif self.axis2 is None:
            if self.screw == 0:
                s = f"SymElem {self.label:3} {self.nfold:3}  AXS {axs}  CEN {cen}"
            else:
                s = f"SymElem {self.label:3} {self.nfold:3}  AXS {axs}  CEN {cen}  HEL {self.hel}"
        else:
            s = f"SymElem {self.label:3} {self.nfold:3}  AXS {axs}  CEN {cen}  AXS2 {axs}"
        return s

_cubeedges = [
    [[0, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0]],
    [[0, 0, 1], [1, 0, 0]],
    [[0, 1, 0], [1, 0, 0]],
    [[0, 1, 0], [0, 0, 1]],
    [[1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [0, 1, 0]],
    [[1, 0, 0], [0, 1, 0]],
    [[0, 1, 1], [1, 0, 0]],
    [[1, 1, 0], [0, 0, 1]],
    [[1, 0, 1], [0, 1, 0]],
]
_cube_edge_cen, _cube_edge_axis = np.array(_cubeedges).swapaxes(0, 1)

def showsymelems(
    sym,
    symelems,
    allframes=True,
    colorbyelem=False,
    cells=3,
    framecells=4,
    bounds=[-0.1, 1.1],
    scale=12,
    offset=0,
    weight=2.0,
    scan=0,
    lattice=None,
    # onlyz=False,
    showframes=True,
    screwextraaxis=False,
):
    if isinstance(symelems, list):
        tmp = defaultdict(list)
        for e in symelems:
            tmp[e.label].append(e)
        symelems = tmp

    import pymol  # type: ignore
    elemframes = np.eye(4).reshape(1, 4, 4)
    if lattice is None:
        lattice = lattice_vectors(sym, cellgeom="nonsingular")
    cellgeom = ipd.sym.xtal.cellgeom_from_lattice(lattice)
    frames = ipd.sym.xtal.sgframes(sym, cells=framecells, cellgeom=cellgeom)
    # ipd.showme(frames, scale=scale)
    ipd.showme(frames, bounds=bounds, lattice=lattice, name="frames", scale=scale)
    # assert 0
    if allframes:
        elemframes = ipd.hscaled(scale, ipd.sym.xtal.sgframes(sym, cells=cells, cellgeom=cellgeom))
    if scan in (True, 1):
        scan = scale * 2

    ii = 0
    labelcount, colorcount = defaultdict(lambda: 0), defaultdict(lambda: 0)
    for c in symelems:
        for j, sunit in enumerate(symelems[c]):
            assert sunit.isunit
            s = sunit.tolattice(lattice)
            assert not s.isunit
            # s = sunit
            # if colorbyelem: args.colors = [[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)][ii]]
            f2 = elemframes
            if scan and not s.iscompound:
                f2 = (elemframes[:, None] @ ipd.homog.htrans(
                    s.axis[None] * np.linspace(-scale * np.sqrt(3), scale * np.sqrt(3), scan)[:, None])[None])
                # ic(f2.shape)
                f2 = f2.reshape(-1, 4, 4)
                # ic(f2.shape)

            shift = ipd.homog.htrans(s.cen * scale + offset * ipd.homog.hvec([0.1, 0.2, 0.3]))
            # shift = ipd.homog.htrans(s.cen * scale)
            # shift = np.eye(4)

            if s.istet:
                configs = [
                    ((s.axis, [0, 1, 0]), (None, None), [0.0, 0.8, 0.0]),
                    ((-s.axis, [0, 1, 0]), (None, None), [0.0, 0.8, 0.0]),
                    ((s.axis2, [1, 0, 0]), (None, None), [0.8, 0.0, 0.0]),
                ]
            elif s.isoct:
                configs = [
                    (([0, 1, 1], [1, 0, 0]), (None, None), [0.7, 0.0, 0.0]),
                    (([1, 1, 1], [0, 1, 0]), (None, None), [0.0, 0.7, 0.0]),
                    (([0, 0, 1], [0, 0, 1]), (None, None), [0.0, 0.0, 0.7]),
                ]
            elif s.label == "D2":
                configs = [
                    ((s.axis, [1, 0, 0]), (s.axis2, [0, 1, 0]), [0.7, 0, 0]),
                    ((s.axis, [0, 1, 0]), (s.axis2, [0, 0, 1]), [0.7, 0, 0]),
                    ((s.axis, [0, 0, 1]), (s.axis2, [1, 0, 0]), [0.7, 0, 0]),
                ]
            elif s.label == "D4":
                configs = [
                    ((s.axis2, [1, 0, 0]), (s.axis, [0, 1, 0]), [0.7, 0, 0]),
                    ((ipd.homog.hrot(s.axis, 45, s.cen) @ s.axis2, [1, 0, 0]), (s.axis, [0, 1, 0]), [0.7, 0, 0]),
                    ((s.axis, [0, 0, 1]), (s.axis2, [1, 0, 0]), [0.0, 0, 0.9]),
                ]
            elif s.label == "C11":
                continue
                # ic(configs)
                # assert 0
            elif s.nfold == 2:
                configs = [[(s.axis, [1, 0, 0]), (s.axis2, [0, 0, 1]), [1.0, 0.3, 0.6]]]
            elif s.nfold == 3:
                configs = [[(s.axis, [0, 1, 0]), (s.axis2, [1, 0, 0]), [0.6, 1, 0.3]]]
            elif s.nfold == 4:
                configs = [[(s.axis, [0, 0, 1]), (s.axis2, [1, 0, 0]), [0.6, 0.3, 1]]]
            elif s.nfold == 6:
                configs = [[(s.axis, [1, 0, 0]), (s.axis2, [0, 0, 1]), [1, 0.3, 0.6]]]
            else:
                assert 0
            name = f"{s.label}_" + ("ABCDEFGHIJKLMNOP")[labelcount[s.label]]

            cgo = []

            for (tax, ax), (tax2, ax2), xyzlen in configs:  # type: ignore
                cylweight = weight
                xyzlen = np.array(xyzlen)
                if scan:
                    if s.iscyclic:
                        xyzlen[xyzlen < 0.999] = 0
                        xyzlen[xyzlen == 1] = 2.0
                        cylweight = weight / 4
                    if s.isscrew:
                        cylweight = weight / 4
                        if screwextraaxis:
                            xyzlen[xyzlen == 1] = 0.4
                            if s.label in "C11 C21 C31 C41 C61 C62".split():
                                xyzlen[xyzlen == 0.6] = 0.1
                            else:
                                xyzlen[xyzlen == 0.3] = 0.1
                        else:
                            xyzlen[xyzlen < 0.99] = 0.0
                            xyzlen[xyzlen > 0.99] = 0.15

                if s.isdihedral:
                    origin = ipd.homog.halign2(ax, ax2, tax, tax2)
                    xyzlen[xyzlen == 0.6] = 1
                else:
                    origin = ipd.homog.halign(ax, tax)
                ipd.showme(
                    f2 @ shift @ origin,
                    name=name,
                    bounds=[b * scale for b in bounds],
                    xyzlen=xyzlen,
                    addtocgo=cgo,
                    make_cgo_only=True,
                    weight=cylweight,
                    colorset=colorcount[s.label[:2]],
                    lattice=lattice,
                )
            pymol.cmd.load_cgo(cgo, name)
            labelcount[s.label] += 1
            colorcount[s.label[:2]] += 1
            ii += 1
    from ipd.viz.pymol_viz import showcell

    # ic(sym)
    # ic(lattice)
    # ic(cellgeom)
    showcell(scale * lattice)
    # showcube()

def _sanitycheck_compid_cens(elem, frames, compid):
    seenit = list()
    for i in range(np.max(compid)):
        assert np.sum(compid == i) > 0
        compframes = frames[compid == i]
        # ipd.showme(compframes @ elem.origin @ offset, scale=scale)
        cen = einsum("ij,j->i", compframes[0], elem.origin[:, 3])
        assert np.allclose(cen, einsum("fij,j->fi", compframes, elem.origin[:, 3]))  # type: ignore
        assert not any([np.allclose(cen, s) for s in seenit])  # type: ignore
        seenit.append(cen)

def _make_operator_component_joint_ids(elem1, elem2, frames, fopid, fcompid, sanitycheck=True):
    from ipd.viz.pymol_viz import showme

    opcompid = fcompid.copy()
    for i in range(np.max(fopid)):
        fcids = fcompid[fopid == i]
        idx0 = fcompid == fcids[0]
        for fcid in fcids[1:]:
            idx = fcompid == fcid
            opcompid[idx] = min(min(opcompid[idx]), min(opcompid[idx0]))
    for i, id in enumerate(sorted(set(opcompid))):
        opcompid[opcompid == id] = i

    if sanitycheck and elem2.iscompound:
        seenit = np.empty((0, 4))
        for i in range(np.max(opcompid)):
            compframes = frames[opcompid == i]
            cens = einsum("fij,j->fi", compframes, elem2.origin[:, 3])
            if np.any(np.all(np.isclose(cens[None], seenit[:, None]), axis=2)):  # type: ignore
                raise ComponentIDError
                ic(elem1)
                ic(elem2)
                # for i in range(np.max(fopid)):
                showme(elem1.cen, scale=10, name="ref")
                showme(elem1.operators, scale=10, name="ref1")
                for i in range(100):
                    showme(frames[opcompid == i] @ elem2.origin @ htrans([0.01, 0.02, 0.03]), scale=10)
                assert 0

                showme(cens, scale=10)
                showme(compframes, scale=10)
                showme(seenit, scale=10)

                assert 0
            assert not np.any(np.all(np.isclose(cens[None], seenit[:, None]), axis=2))  # type: ignore
            seenit = np.concatenate([cens, seenit])  # type: ignore
    return opcompid

def _mul3(a, b):
    return (a @ b[:3, None])[:, 0]
