import numpy as np

import ipd
from ipd.homog import *

def tooclose_clash(bodies, nbrs=None, **kw):
    return bodies.clashes(nbrs)

def tooclose_overlap(bodies, nbrs=None, contactfrac=0.1, printme=False, **kw):
    cfrac = bodies.contact_fraction(nbrs)
    # if printme: ic(cfrac)
    maxcfrac = max([np.mean(c) for c in cfrac])
    # ic(maxcfrac, contactfrac)
    # if maxcfrac > 0.999:
    # ipd.showme(bodies)
    # assert 0
    # TRUE means sufficient overlap

    return max(False, maxcfrac - contactfrac)
    # return maxcfrac > contactfrac

def tooclose_primary_overlap(bodies, nbrs=None, contactfrac=0.1, nprimary=None, printme=False, **kw):
    # assert 0
    assert nprimary is not None
    for i in range(nprimary):
        for j in range(nprimary, len(bodies)):
            hasclash = bodies.bodies[i].hasclash(bodies.bodies[j])
            if hasclash:
                "SLIDE EXTRAFRAME CLASH"
                return True
    cfrac = bodies.contact_fraction(nbrs)
    maxcfrac = max([np.mean(c) for c in cfrac])
    # ic(maxcfrac, contactfrac)
    return max(False, maxcfrac - contactfrac)
    # return maxcfrac > contactfrac

class RBLatticeOverlapObjective:
    def __init__(self, *args, **kw):
        self.rbojective = RBOverlapObjective(*args, **kw)

    def __call__(self, state, **kw):
        assert isinstance(state, ipd.dev.Bunch)
        assert isinstance(state.scale, (int, float))
        ic(state.scale)  # type: ignore
        self.rbojective.bodies[0].set_scale(state.scale)  # type: ignore
        return self.rbojective(state.position, **kw)

class RBOverlapObjective:
    def __init__(
        self,
        initial,
        bodies=None,
        contactfrac=None,
        scoreframes=None,
        clashframes=None,
        lever=20,
        biasradial=None,
        biasdir=None,
        contactdist=None,
        clashdist=3,
        driftpenalty=1,
        clashpenalty=1,
        angpenalty=1,
        spreadpenalty=1,
        minradius=0,
        sym=None,
        symaxes=None,
        **kw,
    ):
        self.initial = initial.position.copy()
        self.initialcom = initial.com().copy()
        self.lever = lever or initial.rog()
        self.contactfrac = max(0.001, contactfrac)  # type: ignore
        assert 0 <= contactfrac <= 1  # type: ignore
        self.scoreframes = scoreframes
        self.clashframes = clashframes
        self.bodies = bodies
        self.biasdir = hnormalized(biasdir)
        self.biasradial = biasradial
        self.contactdist = contactdist
        self.clashdist = clashdist
        self.sym = sym
        self.symaxes = symaxes
        self.driftpenalty = driftpenalty
        self.clashpenalty = clashpenalty
        self.angpenalty = angpenalty
        self.spreadpenalty = spreadpenalty
        self.minradius = minradius

    def __call__(self, position, verbose=False):
        asym = self.bodies[0]  # type: ignore
        asym.position = position

        tmp1 = self.initial.copy()
        p = hproj(self.biasdir, hcart3(tmp1))
        pp = hprojperp(self.biasdir, hcart3(tmp1))
        tmp1[:3, 3] = p[:3] / self.biasradial + pp[:3]
        tmp2 = self.bodies[0].position.copy()  # type: ignore
        p = hproj(self.biasdir, hcart3(tmp2))
        pp = hprojperp(self.biasdir, hcart3(tmp2))
        tmp2[:3, 3] = p[:3] / self.biasradial + pp[:3]
        xdiff = hdiff(tmp1, tmp2, lever=self.lever)

        clashfrac, contactfrac = 0, 0
        scores = list()
        clash = 0
        fracs = list()
        # bods = self.bodies[1:] if self.scoreframes is None else [self.bodies[i] for i in self.scoreframes]
        dists = list()
        for ib, b in enumerate(self.bodies):  # type: ignore
            for jb, b2 in enumerate(self.bodies):  # type: ignore
                if (ib, jb) in self.scoreframes:  # type: ignore
                    d = ipd.homog.hnorm(b.com() - b2.com())
                    d = max(0, d - asym.rog() * 3)
                    dists.append(d)
                    f1, f2 = b.contact_fraction(b2, contactdist=self.contactdist)
                    # if verbose: ic(ib, jb, f1, f2)
                    fracs.extend([f1, f2])
                    diff11 = max(0, f1 - self.contactfrac) / (1 - self.contactfrac)
                    diff12 = max(0, self.contactfrac - f1) / self.contactfrac
                    diff21 = max(0, f2 - self.contactfrac) / (1 - self.contactfrac)
                    diff22 = max(0, self.contactfrac - f2) / self.contactfrac

                    # ic(f1, diff11, diff12, f2, diff21, diff22)

                    scores.append((max(diff11, diff12)**2))
                    scores.append((max(diff21, diff22)**2))
                elif (ib, jb) in self.clashframes:  # type: ignore
                    dists = b.clash_distances(b2, self.clashdist)
                    # if len(dists):
                    # ic(dists)
                    # assert 0
                    clash += np.sum((self.clashdist - dists)**2)
                    # clash += (self.clashpenalty / 10 * (b.clashes(b2, self.clashdist) / len(b)))**2

        # ic(dists)
        # ic([int(_) for _ in scores])
        # ic([int(_ * 100) for _ in fracs])
        # ic(max(scores), (self.driftpenalty * xdiff)**2)

        # zxang0 = ipd.homog.dihedral([0, 0, 1], [0, 0, 0], [1, 0, 0], self.initialcom)
        # ax1 = ipd.sym.axes(self.sym)[2]
        # ax2 = ipd.sym.axes(self.sym)[3]
        angdiff1 = angdiff2 = angdiffcen = 0
        if self.symaxes is not None:
            ax1, ax2 = self.symaxes
            nf1rot = ipd.homog.dihedral(ax2, [0, 0, 0], ax1, asym.com())
            nf2rot = ipd.homog.dihedral(ax1, [0, 0, 0], ax2, asym.com())
            angokrange = np.pi / 16
            # angdiff = max(0, abs(zxang0 - zxang) - angokrange)
            angdiff1 = max(0, abs(nf1rot) - angokrange)
            angdiff2 = max(0, abs(nf2rot) - angokrange)
            # axsdist1 = ipd.homog.hnorm(ipd.hprojperp(ax1, asym.com()))
            # axsdist2 = ipd.homog.hnorm(ipd.hprojperp(ax2, asym.com()))
            # angdiff1 = angdiff1 * axsdist1
            # angdiff2 = angdiff2 * axsdist2
            angdiff1 = 10 * angdiff1**2
            # angdiff2 = 10 * angdiff2**2
            angdiffcen = ipd.homog.hnorm(asym.com()) * ipd.hangle(asym.com(), ax1 + ax2)  # type: ignore

        # ic(nf1rot, nf2rot)
        # ic(abs(zxang0 - zxang))
        # ic(angdiff)
        # ic(xdiff, max(scores))

        # scores[0] *= 2
        # scores[1] *= 2
        if verbose:
            # ic(scores)
            ic(fracs)  # type: ignore
            # ic((self.driftpenalty * xdiff)**2)
            # ic((self.angpenalty * 10 * angdiff * ipd.homog.hnorm(ipd.hprojperp([1, 0, 0], asym.com())))**2)
        s = [
            10 * sum(scores),
            (self.spreadpenalty * (max(fracs) - min(fracs)))**2,
            (self.driftpenalty * xdiff)**2,
            (self.angpenalty * angdiff1)**2,
            (self.angpenalty * angdiff2)**2,
            # 0.1 * (axsdist1 + axsdist2)
            (max(0, self.minradius - ipd.homog.hnorm(asym.com())))**2,
            0.0 * angdiffcen**2,
            self.clashpenalty * clash,
            2 * np.sum(np.array(dists)**2),
        ]
        # ic(s)
        return np.sum(s)
