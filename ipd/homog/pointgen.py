import numpy as np

import ipd

def prune_radial_outliers(xyz, nprune=10):
    npoints = len(xyz) - nprune
    for _ in range(nprune):
        com = ipd.homog.hcom(xyz)
        r = ipd.homog.hnorm(xyz - com)
        w = np.argsort(r)
        xyz = xyz[w[:-1]]
    return xyz

def point_cloud(npoints=100, std=10, outliers=0):
    xyz = ipd.homog.hrandpoint(npoints + outliers, std=10)
    xyz = prune_radial_outliers(xyz, outliers)
    assert len(xyz) == npoints
    xyz = xyz[np.argsort(xyz[:, 0])]
    xyz -= ipd.homog.hvec(ipd.homog.hcom(xyz))
    return xyz
