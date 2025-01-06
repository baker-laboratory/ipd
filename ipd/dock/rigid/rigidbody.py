import numpy as np

import ipd

class RigidBodyFollowers:
    def __init__(self, bodies=None, coords=None, frames=None, sym=None, cellsize=1, **kw):
        self.sym = sym.upper() if sym else None
        self.kw = kw
        assert not (frames is None and sym is None)
        if frames is None:
            frames = ipd.sym.frames(sym, cellsize=cellsize, ontop="primary", **kw)
        if bodies is not None:
            self.asym = bodies[0]
            self.symbodies = bodies[1:]
            self.bodies = bodies
        elif frames is not None:
            if coords is None:
                raise ValueError("if no bodies specified, coords and frames/sym must be provided")
            assert ipd.homog.hunique(frames)
            self.asym = RigidBody(coords, **kw)
            self.symbodies = [RigidBody(parent=self.asym, xfromparent=x, **kw) for x in frames[1:]]
            self.bodies = [self.asym] + self.symbodies
        self._cellsize = ipd.homog.to_xyz(cellsize)
        self.orig_cellsize = self._cellsize.copy()
        self.is_point_symmetry = np.sum(ipd.homog.hnorm(ipd.hcart3(self.frames()))) < 0.0001  # type: ignore
        self.rootbody = self.asym
        if not self.asymexists:
            self.bodies[0] = RigidBody(parent=self.asym, xfromparent=frames[0], **kw)
            self.asym = self.bodies[0]
        self.scale_com_with_cellsize = False

        if self.asymexists:
            for i, b in enumerate(self.bodies):
                if i > 0 and np.allclose(b.xfromparent, np.eye(4)):  # type: ignore
                    ic(i)  # type: ignore
                    assert 0

        assert ipd.homog.hunique(self.frames())
        # assert 0

    def set_asym_coords(self, coords):
        newbody = RigidBody(coords, **self.kw)
        self.rootbody = newbody
        if self.asymexists:
            self.bodies[0] = newbody

    @property
    def asymexists(self):
        return self.sym is None or not self.sym.startswith("H")

    def clashes(self, nbrs=None):
        clashes = self.clash_list(nbrs)
        return len(clashes) > 0

    def clash_coms(self, nbrs=None):
        clashes = self.clash_list(nbrs)
        return [self.bodies[i].com() for i in clashes]

    def clash_directions(self, nbrs=None):
        clashes = self.clash_list(nbrs)
        return [self.bodies[i].com() - self.asym.com() for i in clashes]

    def clash_list(self, nbrs=None):
        if isinstance(nbrs, int):
            nbrs = [nbrs]
        indices = range(1, len(self.bodies)) if nbrs is None else nbrs
        clashes = [i for i in indices if self.asym.clashes(self.bodies[i])]
        return clashes

    def contact_fraction(self, nbrs=None):
        if isinstance(nbrs, int):
            nbrs = [nbrs]
        if nbrs is None:
            return [self.asym.contact_fraction(b) for b in self.symbodies]
        else:
            return [self.asym.contact_fraction(self.bodies[i]) for i in nbrs]

    @property
    def cellsize(self):
        return self._cellsize.copy()
        # return self._cellsize * self.asym.scale

    @cellsize.setter
    def cellsize(self, cellsize):
        self.scale_frames(scalefactor=cellsize / self._cellsize, safe=True)

    @property
    def scale(self):
        return self._cellsize / self.orig_cellsize

    @scale.setter
    def scale(self, scale):
        scalefactor = scale / (self._cellsize / self.orig_cellsize)
        self.scale_frames(scalefactor, scalecoords=True, safe=True)

    def scale_frames(self, scalefactor, scalecoords=None, safe=True):
        # ic('CALL scale_frames', scalefactor, scalecoords)
        if scalecoords is None:
            scalecoords = self.scale_com_with_cellsize
        if safe and self.is_point_symmetry:
            raise ValueError("scale_frames only valid for non-point symmetry")

        scalefactor = ipd.homog.to_xyz(scalefactor)
        if self.sym is not None and self.sym.startswith("H"):
            assert np.allclose(scalefactor[0], scalefactor[1])

        self._cellsize *= scalefactor
        self.asym.scale = self.asym.scale * scalefactor  # type: ignore

        if scalecoords:
            # ic(self.asym.xfromparent)
            assert np.allclose(self.asym.xfromparent[:3, :3], np.eye(3))  # type: ignore
            self.asym.moveby(ipd.htrans((scalefactor-1) * self.asym.com()[:3]))  # type: ignore

        return self.cellsize
        # changed = any([b.scale_frame(scalefactor) for b in self.bodies])
        # if not changed:
        #    if safe:
        #       raise ValueError(f'no frames could be scaled, scale_frames only valid for unbounded symmetry')
        # return changed

    # def set_scale(self, scale, safe=True):
    #    self.cellsize *= scalefactor
    #    changed = any([b.set_frame_scale(scale) for b in self.bodies])
    #    if not changed:
    #       if safe:
    #          raise ValueError(f'no frames could be scaled, scale_frames only valid for unbounded symmetry')
    #    return changed

    def get_neighbors_by_axismatch(self, axis, perp=False):
        nbrs = list()
        for i in range(1, len(self.bodies)):
            to_nbr_axs = ipd.axis_of(self.bodies[i].xfromparent)  # type: ignore
            ang = ipd.hangline(to_nbr_axs, axis)  # type: ignore
            # ic(perp, ang, axis, to_nbr_axs)
            if (not perp and ang > 0.001) or (perp and abs(ang - np.pi / 2) < 0.001):  # type: ignore
                nbrs.append(i)
        return nbrs

    def dump_pdb(self, fname, **kw):
        asym = self.asym
        if not self.asym.isroot:
            asym = self.asym.parent
        coords = asym.allcoords.copy()  # type: ignore
        if not self.asym.isroot:
            coords = coords[1:]
        # ic(coords.shape)
        ipd.pdb.dumppdb(fname, coords, nchain=len(self.bodies), **kw)

    def frames(self):
        return np.stack([b.xfromparent for b in self.bodies])  # type: ignore

    def origins(self):
        return np.stack([ipd.hcart3(b.xfromparent) for b in self.bodies])  # type: ignore

    def orientations(self):
        return np.stack([ipd.hori3(b.xfromparent) for b in self.bodies])  # type: ignore

    def coms(self):
        return np.stack([b.com() for b in self.bodies])

    @property
    def coords(self):
        return np.stack([b.coords for b in self.bodies])

    @property
    def bvh_op_count(self):
        return np.sum([b.bvhopcount for b in self.bodies])

    def bvh_op_count_reset(self):
        for b in self.bodies:
            b.bvhopcount = 0

    def __len__(self):
        return len(self.bodies)

class RigidBody:
    def __init__(
        self,
        coords=None,
        contact_coords=None,
        extra=None,
        position=np.eye(4),
        parent=None,
        xfromparent=None,
        contactdis=8,
        clashdis=3,
        usebvh=True,
        scale=1,
        interacting_points=None,
        recenter=False,
        **kw,
    ):
        assert xfromparent is None or not np.allclose(np.eye(4), xfromparent)

        self.extra = extra
        self.parent = parent
        if xfromparent is not None:
            self._xfromparent = xfromparent.copy()
        else:
            self._xfromparent = np.eye(4)
        assert ipd.homog.hvalid(self._xfromparent)

        self._position = position
        self._coords = None
        self.bvh = None
        self.contactbvh = None
        self.clashdis = clashdis
        self.contactdis = contactdis
        self.usebvh = usebvh
        self.tolocal = np.eye(4)
        self.toglobal = np.eye(4)
        assert (parent is None) != (coords is None)
        if coords is not None:
            coords = coords.copy()
            if contact_coords is None:
                contact_coords = coords
            contact_coords = contact_coords.copy()
            if recenter:
                # oldcom =
                self.tolocal = ipd.htrans(-ipd.homog.hcom(coords))  # type: ignore
                self.toglobal = ipd.hinv(self.tolocal)  # type: ignore
                coords = ipd.homog.hxform(self.tolocal, coords)
                contact_coords = ipd.homog.hxform(self.tolocal, contact_coords)
                # position must be set to move coords back to gloabal frame
                self.position = self.toglobal.copy()
            self._coords = ipd.homog.hpoint(coords)
            self._contact_coords = ipd.homog.hpoint(contact_coords)
            self._com = ipd.homog.hcom(self._coords)
            if usebvh:
                self.bvh = willutil_cpp.bvh.BVH(coords[..., :3])  # type: ignore
                self.contactbvh = willutil_cpp.bvh.BVH(contact_coords[..., :3])  # type: ignore
        elif parent is not None:
            self.bvh = parent.bvh
            self.contactbvh = parent.contactbvh
            self._coords = parent._coords
            self._com = ipd.homog.hcom(self._coords)
            parent.children.append(self)
            self.clashdis = parent.clashdis
            self.contactdis = parent.contactdis
            self.usebvh = parent.usebvh
            self._scale = None
        if parent is None:
            scale = ipd.homog.to_xyz(scale)
            self._scale = scale

        self.children = list()
        self.bvhopcount = 0

    def __len__(self):
        return len(self._coords)  # type: ignore

    @property
    def xfromparent(self):
        return ipd.hscaled(self.scale, self._xfromparent)  # type: ignore

    def scale_frame(self, scalefactor):
        self.scale *= scalefactor
        # return
        # if self.xfromparent is not None:
        #    if ipd.homog.hnorm(self.xfromparent[:, 3]) > 0.0001:
        #       self.xfromparent = ipd.hscaled(scalefactor, self.xfromparent)
        #       return True
        # return False

    @property
    def state(self):
        assert self.parent is None
        state = ipd.dev.Bunch(position=self.position, scale=self.scale)
        assert isinstance(state.scale, (int, float, np.ndarray))
        return state

    @state.setter
    def state(self, state):
        assert self.parent is None
        self.position = state.position
        self.set_scale(state.scale)  # type: ignore

    def moveby(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = ipd.htrans(x)  # type: ignore
        self.position = ipd.homog.hxform(x, self.position)
        assert ipd.homog.hvalid(self.position)

    def move_about_com(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = ipd.htrans(x)  # type: ignore
        com = self.com()
        self.moveby(-com)
        self.position = ipd.homog.hxform(x, self.position)
        self.moveby(com)

    @property
    def scale(self):
        if self.parent is None:
            return self._scale
        return self.parent.scale

    @scale.setter
    def scale(self, scale):
        # assert self.parent is None
        if self.parent:
            self.parent.scale = ipd.homog.to_xyz(scale)
        else:
            self._scale = scale

    @property
    def position(self):
        if self.parent is None:
            return self._position
        # x = self.xfromparent.copy()
        # x[:3, 3] *= self.scale
        return self.xfromparent @ self.parent.position

    @position.setter
    def position(self, newposition):
        if newposition.shape[-2:] != (4, 4):
            raise ValueError("RigidBody position is 4,4 matrix (not point)")
        if self.parent is not None:
            # raise ValueError(f'RigidBody with parent cant have position set')
            self.parent.position = ipd.hinv(self.xfromparent) @ newposition  # type: ignore
        self._position = newposition.reshape(4, 4)

    @property
    def globalposition(self):
        assert self.parent is None
        # self.positions has been set to move local coords into intial global frame
        # tolocal moves position so identity doesn't move global frame coords
        # yeah, confusing... self.position 'moves' opposite of intuition
        return ipd.homog.hxform(self.tolocal, self.position)

    @property
    def coords(self):
        return ipd.homog.hxform(self.position, self._coords)

    @property
    def globalcoords(self):
        return ipd.homog.hxform(self.globalposition, self._coords)

    @property
    def allcoords(self):
        crd = [self.coords]
        crd = crd + [c.coords for c in self.children]
        return np.stack(crd)

    def com(self):
        return self.position @ self._com

    def setcom(self, newcom):
        pos = self.position
        pos[:, 3] = newcom - self._com
        pos[3, 3] = 1
        self.position = pos

    def comdirn(self):
        return ipd.homog.hnormalized(self.com())

    def rog(self):
        d = self.coords - self.com()
        return np.sqrt(np.sum(d**2) / len(d))

    def contact_count(self, other, contactdist, usebvh=None):
        self.bvhopcount += 1
        assert isinstance(other, RigidBody)
        if usebvh or (usebvh is None and self.usebvh):
            count = willutil_cpp.bvh.bvh_count_pairs(  # type: ignore
                self.contactbvh,
                other.contactbvh,
                self.position,
                other.position,  # type: ignore
                contactdist)
        else:
            assert 0
            # import scipy.spatial
            # d = scipy.spatial.distance_matrix(self.coords, other.coords)
            d = ipd.homog.hnorm(self.coords[None] - other.coords[:, None])
            count = np.sum(d < contactdist)
        return count

    def contacts(self, other):
        self.bvhopcount += 1
        return self.contact_count(other, self.contactdis)

    def clashes(self, other, clashdis=None):
        self.bvhopcount += 1
        # ic(self.clashdis)
        clashdis = clashdis or self.clashdis
        return self.contact_count(other, self.clashdis)

    def hasclash(self, other, clashdis=None):
        self.bvhopcount += 1
        clashdis = clashdis or self.clashdis
        isect = willutil_cpp.bvh.bvh_isect(self.bvh, other.bvh, self.position, other.position, clashdis)  # type: ignore
        return isect

    def intersects(self, other, otherpos, mindis=10):
        self.bvhopcount += 1
        isect = willutil_cpp.bvh.bvh_isect_vec(self.bvh, other.bvh, self.position, otherpos, mindis)  # type: ignore
        return isect

    def point_contact_count(self, other, contactdist=8):
        self.bvhopcount += 1
        p = self.interactions(other, contactdist=contactdist)
        a = set(p[:, 0])
        b = set(p[:, 1])
        # ic(a)
        # ic(b)
        return len(a), len(b)

    def contact_fraction(self, other, contactdist=None):
        self.bvhopcount += 1
        contactdist = contactdist or self.contactdis

        # ipd.pdb.dump_pdb_from_points('bodyA.pdb', self.coords)
        # ipd.pdb.dump_pdb_from_points('bodyB.pdb', other.coords)

        p = self.interactions(other, contactdist=contactdist)
        a = set(p[:, 0])
        b = set(p[:, 1])
        # ic(len(a), len(self.coords))
        # ic(len(b), len(other.coords))
        cfrac = len(a) / len(self.coords), len(b) / len(self.coords)
        assert cfrac[0] <= 1.0 and cfrac[1] <= 1.0
        if self.parent is not None:
            if cfrac[0] > 0.999 or cfrac[0] > 0.999:
                ic(self.xfromparent)  # type: ignore
                ic(self._xfromparent)  # type: ignore
                ic(self.parent)  # type: ignore
                ic(other.xfromparent)  # type: ignore
                ic(other._xfromparent)  # type: ignore
                ic(other.parent)  # type: ignore
                assert 0

        return cfrac

    def clash_distances(self, other, maxdis=8):
        self.bvhopcount += 1
        crd1 = self.coords
        crd2 = other.coords
        interactions = self.clash_interactions(other, maxdis)
        crd1 = crd1[interactions[:, 0]]
        crd2 = crd2[interactions[:, 1]]
        return ipd.homog.hnorm(crd1 - crd2)

    def interactions(self, other, contactdist=8, buf=None, usebvh=None):
        self.bvhopcount += 1
        assert isinstance(other, RigidBody)
        if usebvh or (usebvh is None and self.usebvh):
            if not buf:
                buf = np.empty((100000, 2), dtype="i4")
            pairs, overflow = willutil_cpp.bvh.bvh_collect_pairs(  # type: ignore
                self.contactbvh,
                other.contactbvh,
                self.position,  # type: ignore
                other.position,
                contactdist,
                buf)
            assert not overflow
        else:
            d = ipd.homog.hnorm(self.contact_coords[None] - other.contact_coords[:, None])  # type: ignore
            pairs = np.stack(np.where(d <= contactdist), axis=1)
        return pairs

    def clash_interactions(self, other, contactdist=8, buf=None, usebvh=None):
        self.bvhopcount += 1
        assert isinstance(other, RigidBody)
        if usebvh or (usebvh is None and self.usebvh):
            if not buf:
                buf = np.empty((100000, 2), dtype="i4")
            pairs, overflow = willutil_cpp.bvh.bvh_collect_pairs(  # type: ignore
                self.bvh,
                other.bvh,
                self.position,
                other.position,  # type: ignore
                contactdist,
                buf)
            assert not overflow
        else:
            d = ipd.homog.hnorm(self.coords[None] - other.coords[:, None])
            pairs = np.stack(np.where(d <= contactdist), axis=1)
        return pairs

    def dumppdb(self, fname, dumpchildren=False, spacegroup=None, **kw):
        if dumpchildren:
            crd = self.allcoords
            ipd.pdb.dumppdb(fname, crd, nchain=len(self.children) + 1, **kw)
        elif spacegroup is not None:
            ipd.pdb.dumppdb(fname, self.coords, spacegroup=spacegroup, cellsize=self.scale, **kw)
        else:
            ipd.pdb.dumppdb(fname, self.coords, **kw)

    @property
    def isroot(self):
        return self.parent is None

    @property
    def isleaf(self):
        return not self.children
