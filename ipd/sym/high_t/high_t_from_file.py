import sys
import ipd
from ipd import h
import torch as th

def get_quasi_sym_ops(stubs):
    stubgrid = stubs[None] @ h.inv(stubs[:, None])
    stubgrid = h.remove_diagonal_elements(stubgrid)
    ax, ang, cen = h.axis_angle_cen(stubgrid)
    return ax, ang, cen

def get_approx_stubs(ca):
    assert ca.ndim == 3 and ca.shape[-1] == 3
    ca0centered = ca[0] - ca[0].mean(0)
    _, _, pc = th.pca_lowrank(ca0centered)
    stub = h.frame(*pc, cen=ca[0].mean(0))
    a_to_others = th.stack([th.eye(4)] + [h.rmsfit(ca[i], ca[0])[2] for i in range(1, len(ca))])
    stubs = h.xform(a_to_others, stub)
    return stubs

def get_high_t_frames_from_file(fname):
    ca = th.as_tensor(ipd.pdb.readpdb(fname).ca(splitchains=True))
    stubs = get_approx_stubs(ca)
    symframes = ipd.sym.frames('icos', torch=True)
    asymframes = stubs @ h.inv(stubs[0])
    frames = h.xform(symframes, asymframes)
    return frames
    # ops = get_quasi_sym_ops(stubs)
    # ipd.showme(stubs, kind='xform')
    # ipd.showme(ca.reshape(-1, 3))
    # ipd.showme(stubs, weight=20)
    # ipd.showme(h.xform(ipd.sym.frames('I'), stubs), weight=10)
    # ipd.showme(h.xform(frames, stubs[0]))
    # ipd.showme(h.xform(frames, ca[0]).reshape(-1,3))

