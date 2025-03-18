import numpy as np
import ipd.homog.hgeom as h
from ipd.bunch import Bunch

class RelXformInfo(Bunch):
    pass

def rel_xform_info(frame1, frame2, **kw):
    # rel = np.linalg.inv(frame1) @ frame2
    rel = frame2 @ np.linalg.inv(frame1)
    # rot = rel[:3, :3]
    # axs, ang = h.axis_angle_of(rel)
    axs, ang, cen = h.axis_ang_cen_of(rel)

    framecen = (frame2[:, 3] + frame1[:, 3]) / 2
    framecen = framecen - cen
    framecen = h.hproj(axs, framecen)
    framecen = framecen + cen

    inplane = h.hprojperp(axs, cen - frame1[:, 3])
    # inplane2 = h.hprojperp(axs, cen - frame2[:, 3])
    rad = np.sqrt(np.sum(inplane**2))
    if np.isnan(rad):
        print("isnan rad")
        print("xrel")
        print(rel)
        print("det", np.linalg.det(rel))
        print("axs ang", axs, ang)
        print("cen", cen)
        print("inplane", inplane)
        assert 0
    hel = np.sum(axs * rel[:, 3])
    return RelXformInfo(
        xrel=rel,
        axs=axs,
        ang=ang,
        cen=cen,
        rad=rad,
        hel=hel,
        framecen=framecen,
        frames=np.array([frame1, frame2]),
    )
