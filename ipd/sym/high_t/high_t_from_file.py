import ipd
from ipd import h
import re
import numpy as np

th = ipd.lazyimport('torch')

def get_quasi_sym_ops(stubs):
    stubgrid = stubs[None] @ h.inv(stubs[:, None])  # type: ignore
    stubgrid = h.remove_diagonal_elements(stubgrid)  # type: ignore
    ax, ang, cen = h.axis_angle_cen(stubgrid)  # type: ignore
    return ax, ang, cen

def get_approx_stubs(ca):
    assert ca.ndim == 3 and ca.shape[-1] == 3
    ca0centered = ca[0] - ca[0].mean(0)
    _, _, pc = th.pca_lowrank(ca0centered)
    stub = h.frame(*pc, cen=ca[0].mean(0))  # type: ignore
    a_to_others = th.stack([th.eye(4)] + [h.rmsfit(ca[i], ca[0])[2] for i in range(1, len(ca))])  # type: ignore
    stubs = h.xform(a_to_others, stub)  # type: ignore
    return stubs

def get_high_t_frames_from_file(sym_input_file):
    biomts = get_biomts(sym_input_file)
    ca = th.as_tensor(ipd.pdb.readpdb(sym_input_file).ca(splitchains=True))  #[Tn,Lasu,3]
    _, asu_Rs, asu_COMs = get_ASU_xforms(ca)  # asu xforms wrt main chain
    Rs = []
    for R in biomts:
        for i, a in enumerate(asu_Rs):
            newR = (a @ R).to(th.float32)
            Rs.append(newR)
    Ts = smoothen(th.stack(biomts), th.stack(asu_COMs))
    norms = Ts.norm(p=2, dim=1, keepdim=True)
    Ts = Ts / norms * opt.radius
    Ts = [T - Ts[0] for T in Ts]
    xforms = []
    for i, R in enumerate(Rs):
        X = np.eye(4)
        X[:3, :3] = R[:3, :3]
        X[:3, 3] = Ts[i]
        xforms.append(th.tensor(X))
    return th.stack(xforms)

def smoothen(Rs, Ts, nsteps=10, Tscale=0.1):
    print('Smoothening transforms')
    COM_Ts = Ts.mean(dim=0)
    min_dist = compute_dist(Rs, Ts)
    for n in range(nsteps):
        print(min_dist)
        new_dist = compute_dist(Rs, Ts)
        while new_dist <= min_dist:
            random_perturbations = (2 * th.randn(1) - 1) * (Ts-COM_Ts) * Tscale
            newTs = Ts + random_perturbations
            new_dist = compute_dist(Rs, newTs)
        Ts = newTs
        min_dist = new_dist
    newTs = []
    for R in Rs:
        for T in Ts:
            newTs.append(R @ T)
    print('Smoothed transforms')
    return th.stack(newTs)

def compute_dist(Rs, Ts):
    newTs = []
    for R in Rs:
        for T in Ts:
            newTs.append(R @ T)
    newTs = th.stack(newTs)
    dmap = th.cdist(newTs, newTs, p=2).to(th.float32)
    diag = th.full((1, len(newTs)), 9999).to(th.float32)
    indices = th.arange(len(newTs))
    dmap[indices, indices] = diag
    return dmap.min()

def distribute_points_on_sphere(num_points, r=1):
    """Distribute points on a sphere using the Fibonacci sphere algorithm."""
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    theta = 2 * np.pi * indices / phi
    z = 1 - (2*indices/num_points)
    xy_radius = np.sqrt(1 - z**2)
    x = xy_radius * np.cos(theta)
    y = xy_radius * np.sin(theta)
    points = np.vstack((x, y, z)).T * r
    return th.tensor(points)

def get_ASU_xforms(ca):
    asu_xforms = []
    asu_Ts = []
    for a in ca:
        rms, U, cP, cT = ipd.sym.kabsch(a, ca[0])
        asu_xforms.append(U)
        asu_Ts.append(cP - cT)
    asu_COMs = [ca[i].mean(dim=0) for i in range(ca.shape[0])]
    return asu_Ts, asu_xforms, asu_COMs

def get_biomts(fname):
    biomts = {}
    with open(fname, 'r') as f:
        lines = f.readlines()
    for l in lines:
        if l[:18] == 'REMARK 350   BIOMT':
            fields = re.split(' +', l)
            row = int(fields[2][-1])
            opnum = int(fields[3])
            if (opnum not in biomts):
                biomts[opnum] = (np.zeros((3, 3)), np.zeros(3))
            biomts[opnum][0][row - 1, :] = np.array([float(fields[4]), float(fields[5]), float(fields[6])])
            biomts[opnum][1][row - 1] = float(fields[7])
    Rs = []
    for i, (R, T) in biomts.items():
        Rs.append(th.tensor(R))
    return Rs
