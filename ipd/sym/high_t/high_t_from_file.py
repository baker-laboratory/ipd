import ipd
from ipd import h
import re
import numpy as np

th = ipd.lazyimport('torch')

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

def get_exact_high_t_xforms(opt, cenvec):
    biomts = get_biomts(opt.sym_input_file)
    ca = th.as_tensor(ipd.pdb.readpdb(opt.sym_input_file).ca(splitchains=True)) #[Tn,Lasu,3]
    asu_Ts, asu_xforms, asu_COMs = get_ASU_xforms(ca) # asu xforms wrt main chain
    Rs, Ts = [], []
    for R in biomts:
        for i,a in enumerate(asu_xforms):
            newR = (a @ R).to(th.float32)
            Rs.append(newR)
            newT = R @ asu_COMs[i]
            Ts.append(newT)
    # Ts = smoothen(th.stack(Ts))
    # print(Ts)
    # Ts = Ts * opt.radius*100
    # print(Ts)
    # sys.exit()
    Ts = [T - Ts[0] for T in Ts]
    xforms = []
    for i, R in enumerate(Rs):
        X = np.eye(4)
        X[:3, :3] = R[:3,:3]
        X[:3, 3] = Ts[i]
        xforms.append(th.tensor(X))
    return th.stack(xforms)

# def smoothen(points, num_iterations=10):
#     """
#     Apply Lloyd's relaxation to smooth the points on the sphere.
    
#     Args:
#         points (torch.Tensor): Initial set of points on the sphere, shape (N, 3).
#         num_iterations (int): Number of iterations for the relaxation process.
    
#     Returns:
#         torch.Tensor: Spherically smooth points, shape (N, 3).
#     """
#     for _ in range(num_iterations):
#         new_points = []
#         # For each point, move it to the centroid of its neighbors
#         for i in range(len(points)):
#             # Get the distances between the current point and all other points
#             diff = points - points[i]  # Shape: (N, 3)
#             distances = th.norm(diff, dim=1)  # Shape: (N,)
#             # Compute the centroid of the neighbors (excluding the point itself)
#             neighbor_indices = distances < th.mean(distances)  # Simple thresholding for neighbors
#             neighbor_points = points[neighbor_indices]  # Select neighbors
#             centroid = th.mean(neighbor_points, dim=0) # centroid
#             new_point = centroid
#             new_points.append(new_point)
#         # Stack the new points and project back onto the sphere
#         points = th.stack(new_points)/ th.stack(new_points).norm()
#     return points


# def smoothen(Rs, Ts, high_t_number):
#     # heinous implementation of a spherical approximation of high T transforms
#     Ts -= Ts.mean(dim=0) # center Ts at [0,0,0]
#     norms = th.norm(Ts, dim=1, keepdim=True)
#     Ts /= norms # normalize points
#     ideal_pts = distribute_points_on_sphere(high_t_number*60) # find unit sphere
#     R = rotation_about_origin(ideal_pts[0], Ts[0])
#     ideal_pts = th.stack([R @ i for i in ideal_pts])
#     dmap = th.cdist(ideal_pts, Ts, p=2)
#     min_dists = th.argmin(dmap, dim=1)
#     print(len(th.unique(min_dists)))
#     return

# def rotation_about_origin(p1, p2):
#     axis = th.linalg.cross(p1, p2)
#     sin = th.norm(axis)
#     cos = th.dot(p1,p2)
#     axis /= th.norm(axis)
#     K = th.tensor([
#         [0, -axis[2], axis[1]],
#         [axis[2], 0, -axis[0]],
#         [-axis[1], axis[0], 0]
#     ])
#     R = th.eye(3) +  sin * K + (1 - cos) * (K @ K)
#     return R

# def distribute_points_on_sphere(num_points):
#     """Distribute points on a sphere using the Fibonacci sphere algorithm."""
#     indices = np.arange(0, num_points, dtype=float) + 0.5
#     phi = (1 + np.sqrt(5)) / 2  # Golden ratio
#     theta = 2 * np.pi * indices / phi
#     z = 1 - (2 * indices / num_points)
#     xy_radius = np.sqrt(1 - z**2)
#     x = xy_radius * np.cos(theta)
#     y = xy_radius * np.sin(theta)
#     points = np.vstack((x, y, z)).T
#     return th.tensor(points)

def get_ASU_xforms(ca):
    asu_xforms = []
    asu_Ts = []
    for a in ca:
        rms, U, cP, cT = ipd.sym.kabsch(a, ca[0])
        asu_xforms.append(U)
        asu_Ts.append(cP-cT)
    asu_COMs = [ca[i].mean(dim=0) for i in range(ca.shape[0])]
    return asu_Ts, asu_xforms, asu_COMs

def get_biomts(fname):
    biomts = {}
    with open(fname, 'r') as f:
        lines = f.readlines()
    for l in lines:
        if l[:18] == 'REMARK 350   BIOMT':
            fields = re.split(' +',l)
            row = int(fields[2][-1])
            opnum = int(fields[3])
            if (opnum not in biomts):
                biomts[opnum] = ( np.zeros((3,3)), np.zeros(3) )
            biomts[opnum][0][row-1,:] = np.array(
                [float(fields[4]),float(fields[5]),float(fields[6])])
            biomts[opnum][1][row-1] = float(fields[7])
    Rs = []
    for i, (R,T) in biomts.items():
        Rs.append(th.tensor(R))
    return Rs