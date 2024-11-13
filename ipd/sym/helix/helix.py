import numpy as np

import ipd

class Helix:
    """Helical symmetry."""
    def __init__(self, turns, phase, nfold=1, turnsB=1):
        self.nfold = nfold
        self.turns = turns
        self.phase = phase
        self.turnsB = turnsB
        # assert nfold == 1
        assert turnsB == 1
        if phase < 0 or phase > 1:
            raise ValueError("helix phase must be 0-1, if you need beyond this range, adjust nturns")

    def dump_pdb(self, fname, coords, **kw):
        frames = self.frames(**kw)
        symcoords = ipd.homog.hxform(frames, coords)
        ipd.pdb.dump_pdb_from_points(fname, symcoords)

    def frames(self, radius, spacing, coils=1, xtalrad=9e9, start=None, closest=0, closest_upper_only=False, **kw):
        """Phase is a little artifical here, as really it just changes
        self.turns "central" frame will be ontop.

        if closest is given, frames will be sorted on dist to cen
        otherwise central frame will be first, then others in order from
        bottom to top
        """
        assert xtalrad is not None
        axis = np.array([0, 0, 1, 0])
        if isinstance(coils, (int, float)):
            coils = (-coils, coils)
        if start is None:
            start = np.eye(4)
            start[0, 3] = radius
        ang = 2 * np.pi / (self.turns + self.phase)
        lb = coils[0] * self.turns - 1
        ub = coils[1] * self.turns + 2
        # ic(coils, self.turns, lb, ub)
        frames = list()
        for icyc in range(self.nfold):
            xcyc = ipd.homog.hrot(axis, (np.pi * 2) / self.nfold * icyc, degrees=False)
            frames += [
                xcyc @ ipd.homog.hrot(axis, i * ang, hel=i * spacing / self.turns, degrees=False) for i in range(lb, ub)
            ]

        frames = np.stack(frames)
        frames = ipd.homog.hxform(frames, start)
        dist = ipd.homog.hnorm(frames[:, :, 3] - start[:, 3])
        frames = frames[np.argsort(dist)]
        frames = frames[dist <= xtalrad]
        if closest > 0:
            if closest_upper_only:
                closest = 2 * (closest-1) + 1
            frames = frames[:closest]
            if closest_upper_only:
                isupper = ipd.homog.hdot(ipd.homog.hnormalized([0, 1, 1]), frames[:, :, 3] - frames[0, :, 3]) >= 0
                isupper[0] = True
                nframes = len(frames)
                frames = frames[isupper]
                # ic(frames.shape, nframes)
                assert len(frames) - 1 == (nframes-1) // 2
        return frames
