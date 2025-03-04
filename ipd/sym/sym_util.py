import math
import numpy as np
import ipd
import ipd.homog.thgeom as h

th = ipd.lazyimport('torch')

SYMA = 1.0

def get_sym_frames(symid, opt, cenvec):
    if opt.H_K is not None: allframes = ipd.sym.high_t.get_pseudo_highT(opt)
    elif symid.lower().startswith('cyclic_vee_'): allframes = cyclic_vee_frames(symid, opt)
    else: allframes = ipd.sym.frames(symid, **opt)
    allframes = th.as_tensor(allframes, dtype=th.float32)
    ic(allframes.shape)
    frames, _ = get_nneigh(allframes, min(len(allframes), opt.max_nsub))
    return allframes, frames

def sym_redock(xyz, Lasu, frames, opt, **_):
    # resolve clashes in placed subunits
    # could probably use this to optimize the radius as well
    def clash_error_comp(R0, T0, xyz, fit_tscale):
        xyz0 = xyz[:Lasu]
        xyz0_corr = xyz0.reshape(-1, 3) @ R0.T
        xyz0_corr = xyz0_corr.reshape(xyz0.shape) + fit_tscale*T0
        # compute clash
        Xsymmall = xyz.clone()
        for n, X in enumerate(frames):
            R = X[:3, :3].float().to(device=xyz.device)
            T = X[:, -1][:3].to(device=xyz.device)
            new_coords = xyz0_corr.reshape(-1, 3) @ R.T
            new_coords = new_coords.reshape(xyz0_corr.shape) + T
            Xsymmall[n * Lasu:(n+1) * Lasu] = new_coords
        # compute clash loss
        Xsymmall = Xsymmall[:, 0, :]
        dsymm = th.cdist(Xsymmall, Xsymmall, p=2)
        dsymm_2 = dsymm.clone()
        # dsymm_2 = dsymm.clone().fill_diagonal_(9999) # avoid in-place operation
        for i in range(0, len(Xsymmall), Lasu):
            dsymm_2[i:i + Lasu, i:i + Lasu] = 9999
        clash = th.clamp(opt.fit_wclash - dsymm_2, min=0)
        loss = th.sum(clash) / Lasu
        return loss

    def Q2R(Q):
        Qs = th.cat((th.ones((1), device=Q.device), Q), dim=-1)
        Qs = normQ(Qs)
        return Qs2Rs(Qs[None, :]).squeeze(0)

    with th.enable_grad():
        T0 = th.zeros(3, device=xyz.device).requires_grad_(True)
        Q0 = th.zeros(3, device=xyz.device).requires_grad_(True)
        lbfgs = th.optim.LBFGS([T0, Q0], history_size=15, max_iter=20, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            loss = clash_error_comp(Q2R(Q0), T0, xyz, opt.fit_tscale)
            loss.backward()  #retain_graph=True)
            return loss

        for e in range(3):
            loss = lbfgs.step(closure)
            ic(loss)

        Q0 = Q0.detach()
        T0 = T0.detach()
        xyz0 = xyz[:Lasu].reshape(-1, 3) @ (Q2R(Q0)).T
        xyz0 = xyz0.reshape(xyz[:Lasu].shape) + opt.fit_tscale * T0

    for n, X in enumerate(frames):
        R = X[:3, :3].float()
        T = X[:, -1][:3]
        new_coords = xyz0.reshape(-1, 3) @ R.T
        new_coords = new_coords.reshape(xyz0.shape) + T
        xyz[n * Lasu:(n+1) * Lasu] = new_coords

    return xyz

def cyclic_vee_frames(symid, opt):
    cx = int(symid[11:])
    frames = ipd.sym.frames(f'c{cx}', torch=True).to(th.float32)
    rotz = h.rot([0, 0, 1], np.radians(opt.cyclic_vee_dihedral))
    rotz180 = h.rot([0, 0, 1], np.pi)
    # roty = h.rot([0, 1, 0], np.radians(90 - opt.cyclic_vee_angle / 2))
    # trans = h.trans([opt.cyclic_vee_separation / 2, 0, 0])
    # flipz = h.rot([0, 0, 1], [0, th.pi])
    # frames = h.xchain(flipz, trans, roty, rotz, frames).reshape(-1, 4, 4)
    roty = h.rot([0, 1, 0], np.radians(180 + opt.cyclic_vee_angle), [opt.cyclic_vee_separation / 2, 0, 0])
    trans = h.trans([opt.cyclic_vee_separation, 0, 0])
    frames1 = h.xform(rotz, frames)
    frames2 = h.xchain(roty, trans, rotz180, rotz, frames)
    frames = th.cat([frames1, frames2])
    frames = h.xform(h.inv(frames[0]), frames)
    return frames.to(th.float32)

    # def calc_Ts(cenvec, radius, Rs):
    # A = radius * cenvec.to(th.float32).cpu()
    # Ts = [R[:3, :3] @ A + R[:3, 3] for R in Rs]
    # Ts = [T - Ts[0] for T in Ts]
    # return Ts
    #
    # Ts = calc_Ts(cenvec, opt.radius, Rs)
    # allframes = []
    # for i, R in enumerate(Rs):
    # X = np.eye(4)
    # X[:3, :3] = R[:3, :3]
    # X[:3, 3] = Ts[i]
    # allframes.append(th.tensor(X))
    # allframes = th.stack(allframes)
    # frames, _ = get_nneigh(allframes, min(len(allframes), opt.max_nsub))
    # return allframes, frames

def get_nneigh(allframes, nsub, w_t=1.0, w_r=1.0):
    '''
    Args:
        allframes (list): list of allframes
    Returns:
        symsub (list): indices corresponding to nearest neighbors around main sub
    '''

    def comb_dist(T1, T2, w_t=1.0, w_r=1.0):
        '''
        Compute the combined distance.
        '''
        t1, t2 = T1[:3, 3], T2[:3, 3]
        t_dist = th.norm(t1 - t2)
        R1, R2 = T1[:3, :3], T2[:3, :3]
        rotation_diff = th.matmul(R1.T, R2)
        trace_value = th.clip((th.trace(rotation_diff) - 1) / 2, -1, 1)
        r_dist = th.acos(trace_value)
        return w_t*t_dist + w_r*r_dist

    dists = th.tensor([comb_dist(allframes[0], x) for x in allframes])
    _, symsub = th.topk(dists, k=nsub, largest=False)
    frames = allframes[symsub]
    return frames, symsub

def get_coords_stack(pdblines):
    """Getting the Ca coords of input "high-T" pdb.

    Grabs coords of ASU. Returns stack of coords [N,L,3], N being n
    chains
    """
    atom_dtype = np.dtype([
        ("chnid", np.unicode_, 1),  # type: ignore
        ("resid", np.int32),
        ("X", np.float64, 3),
    ])
    chains = []
    allatoms = []
    for line in pdblines:
        if line[:4] == 'ATOM' or line[:6] == "HETATM":
            if line[12:16] == " CA ":
                split_line = (line[21], line[22:26], (line[30:38], line[38:46], line[46:54]))
                allatoms.append(split_line)
                if line[21] not in chains: chains.append(line[21])
    allatoms = np.array(allatoms, dtype=atom_dtype)
    L_chain_main = len(allatoms[allatoms['chnid'] == chains[0]])
    xyz_stack = []
    for c in chains:
        assert len(allatoms[allatoms['chnid'] ==
                            c]) == L_chain_main, 'PDB file does not contain symmetrically sized units!'
        xyz_stack.append(allatoms[allatoms['chnid'] == c]['X'])
    return th.tensor(xyz_stack)

def get_sym_frames_from_file(opt):
    """Takes in [N,L,3] Ca xyz stack to map transforms"""
    xyz_stack = get_coords_stack(open(opt.input_pdb))
    assert xyz_stack.shape[
        0] > 1, 'Input PDB file should more than one chain if wanting to mirror input symmetry'
    xforms = []
    for n in range(xyz_stack.shape[0]):
        # centering the main subunit and then calculating xforms from there
        rms, _, X = ipd.h.rmsfit((xyz_stack[n] - xyz_stack[0].mean(dim=0)),
                                 (xyz_stack[0] - xyz_stack[0].mean(dim=0)))  # rms, fitxyz, X
        if rms > 2:
            ic('WARNING: Input PDB contains subunits that have greater than 2A RMSD to main subunit. Consider using a more symmetric input file!'
               )
        xforms.append(X)
    # get reference xform ( in the event of differences in user input )
    asu_xyz = get_coords_stack(open(opt.asu_input_pdb))
    rms, _, X = ipd.h.rmsfit((xyz_stack[0] - xyz_stack[0].mean(dim=0)), (asu_xyz[0] - asu_xyz[0].mean(dim=0)))
    if rms > 2:
        ic('WARNING: ASU input PDB has greater than 2A RMSD to reference input sym file. Check your inputs!')
    xforms = [R @ X for R in xforms]
    xforms = th.stack(xforms)
    subforms, _ = get_nneigh(xforms, min(opt.max_nsub, len(xforms)))
    return xforms.to(th.float32), subforms.to(th.float32)
    # center the complex
    # COM = np.mean(xyz_stack, axis=(0, 1))
    # return th.tensor(xyz_stack - COM, dtype=th.float32)

def generate_ASU_xforms(pdb):
    """Takes in [N,L,3] Ca xyz stack to map transforms of ASU, should return
    homogenous transforms for each subunit in the ASU We use this to generate
    the ASU in the first place."""
    xyz_stack = get_coords_stack(open(pdb))
    allframes = []
    for n in range(xyz_stack.shape[0]):
        _, _, X = ipd.h.rmsfit(xyz_stack[n], xyz_stack[0])  # rms, fitxyz, X  # type: ignore
        allframes.append(X)
    return allframes

def normQ(Q):
    """Normalize a quaternions."""
    return Q / th.linalg.norm(Q, keepdim=True, dim=-1)

def Rs2Qs(Rs):
    Qs = th.zeros((*Rs.shape[:-2], 4), device=Rs.device)

    Qs[..., 0] = 1.0 + Rs[..., 0, 0] + Rs[..., 1, 1] + Rs[..., 2, 2]
    Qs[..., 1] = 1.0 + Rs[..., 0, 0] - Rs[..., 1, 1] - Rs[..., 2, 2]
    Qs[..., 2] = 1.0 - Rs[..., 0, 0] + Rs[..., 1, 1] - Rs[..., 2, 2]
    Qs[..., 3] = 1.0 - Rs[..., 0, 0] - Rs[..., 1, 1] + Rs[..., 2, 2]
    Qs[Qs < 0.0] = 0.0
    Qs = th.sqrt(Qs) / 2.0
    Qs[..., 1] *= th.sign(Rs[..., 2, 1] - Rs[..., 1, 2])
    Qs[..., 2] *= th.sign(Rs[..., 0, 2] - Rs[..., 2, 0])
    Qs[..., 3] *= th.sign(Rs[..., 1, 0] - Rs[..., 0, 1])

    return Qs

def Qs2Rs(Qs):
    Rs = th.zeros((*Qs.shape[:-1], 3, 3), device=Qs.device)

    Rs[..., 0,
       0] = Qs[..., 0] * Qs[..., 0] + Qs[..., 1] * Qs[..., 1] - Qs[..., 2] * Qs[..., 2] - Qs[..., 3] * Qs[...,
                                                                                                          3]
    Rs[..., 0, 1] = 2 * Qs[..., 1] * Qs[..., 2] - 2 * Qs[..., 0] * Qs[..., 3]
    Rs[..., 0, 2] = 2 * Qs[..., 1] * Qs[..., 3] + 2 * Qs[..., 0] * Qs[..., 2]
    Rs[..., 1, 0] = 2 * Qs[..., 1] * Qs[..., 2] + 2 * Qs[..., 0] * Qs[..., 3]
    Rs[..., 1,
       1] = Qs[..., 0] * Qs[..., 0] - Qs[..., 1] * Qs[..., 1] + Qs[..., 2] * Qs[..., 2] - Qs[..., 3] * Qs[...,
                                                                                                          3]
    Rs[..., 1, 2] = 2 * Qs[..., 2] * Qs[..., 3] - 2 * Qs[..., 0] * Qs[..., 1]
    Rs[..., 2, 0] = 2 * Qs[..., 1] * Qs[..., 3] - 2 * Qs[..., 0] * Qs[..., 2]
    Rs[..., 2, 1] = 2 * Qs[..., 2] * Qs[..., 3] + 2 * Qs[..., 0] * Qs[..., 1]
    Rs[..., 2,
       2] = Qs[..., 0] * Qs[..., 0] - Qs[..., 1] * Qs[..., 1] - Qs[..., 2] * Qs[..., 2] + Qs[..., 3] * Qs[...,
                                                                                                          3]

    return Rs

def generateC(angs, eps=1e-4):
    L = angs.shape[0]
    Rs = th.eye(3, device=angs.device).repeat(L, 1, 1)
    Rs[:, 0, 0] = th.cos(angs)
    Rs[:, 0, 1] = -th.sin(angs)
    Rs[:, 1, 0] = th.sin(angs)
    Rs[:, 1, 1] = th.cos(angs)
    return Rs

def generateD(angs, eps=1e-4):
    L = angs.shape[0]
    Rs = th.eye(3, device=angs.device).repeat(2 * L, 1, 1)
    Rs[:L, 0, 0] = th.cos(angs)
    Rs[:L, 0, 1] = -th.sin(angs)
    Rs[:L, 1, 0] = th.sin(angs)
    Rs[:L, 1, 1] = th.cos(angs)
    Rx = th.tensor([[1, 0, 0], [0, -1., 0], [0, 0, -1]], device=angs.device)
    Rs[L:] = th.einsum('ij,bjk->bik', Rx, Rs[:L])
    return Rs

def find_symm_subs(xyz, Rs, metasymm):
    com = xyz[:, :, 1].mean(dim=-2)
    rcoms = th.einsum('sij,bj->si', Rs, com)

    subsymms, nneighs = metasymm

    subs = []
    for i in range(len(subsymms)):
        drcoms = th.linalg.norm(rcoms[0, :] - rcoms[subsymms[i], :], dim=-1)
        _, subs_i = th.topk(drcoms, nneighs[i], largest=False)
        subs_i, _ = th.sort(subsymms[i][subs_i])
        subs.append(subs_i)

    subs = th.cat(subs)
    xyz_new = th.einsum('sij,braj->bsrai', Rs[subs], xyz).reshape(xyz.shape[0], -1, xyz.shape[2], 3)
    return xyz_new, subs

def update_symm_subs(xyz, subs, Rs, metasymm):
    xyz_new = th.einsum('sij,braj->bsrai', Rs[subs], xyz).reshape(xyz.shape[0], -1, xyz.shape[2], 3)
    return xyz_new

def get_symm_map(subs, O):
    symmmask = th.zeros(O, dtype=th.long)
    symmmask[subs] = th.arange(1, subs.shape[0] + 1)
    return symmmask

def rotation_from_matrix(R, eps=1e-4):
    w, W = th.linalg.eig(R.T)
    i = th.where(abs(th.real(w) - 1.0) < eps)[0]
    if (len(i) == 0):
        i = th.tensor([0])
        print('rotation_from_matrix w', th.real(w))
        print('rotation_from_matrix R.T', R.T)
    axis = th.real(W[:, i[-1]]).squeeze()

    cosa = (th.trace(R) - 1.0) / 2.0
    if abs(axis[2]) > eps:
        sina = (R[1, 0] + (cosa-1.0) * axis[0] * axis[1]) / axis[2]
    elif abs(axis[1]) > eps:
        sina = (R[0, 2] + (cosa-1.0) * axis[0] * axis[2]) / axis[1]
    else:
        sina = (R[2, 1] + (cosa-1.0) * axis[1] * axis[2]) / axis[0]
    angle = th.atan2(sina, cosa)

    return angle, axis

def kabsch(pred, true):

    def rmsd(V, W, eps=1e-4):
        L = V.shape[0]
        return th.sqrt(th.sum((V-W) * (V-W)) / L + eps)

    def centroid(X):
        return X.mean(dim=-2, keepdim=True)

    cP = centroid(pred)
    cT = centroid(true)
    pred = pred - cP
    true = true - cT
    C = th.matmul(pred.permute(1, 0), true)
    V, S, W = th.svd(C)
    d = th.ones([3, 3], device=pred.device)
    d[:, -1] = th.sign(th.det(V) * th.det(W))
    U = th.matmul(d * V, W.permute(1, 0))  # (IB, 3, 3)
    rpred = th.matmul(pred, U)  # (IB, L*3, 3)
    rms = rmsd(rpred, true)
    return rms, U, cP, cT

# do lines X0->X and Y0->Y intersect?
def intersect(X0, X, Y0, Y, eps=0.1):
    mtx = th.cat((th.stack((X0, X0 + X, Y0, Y0 + Y)), th.ones((4, 1))), axis=1)
    det = th.linalg.det(mtx)
    return (th.abs(det) <= eps)

def get_angle(X, Y):
    angle = th.acos(th.clamp(th.sum(X * Y), -1., 1.))
    if (angle > np.pi / 2):
        angle = np.pi - angle
    return angle

# given the coordinates of a subunit +
def get_symmetry(xyz, mask, rms_cut=2.5, nfold_cut=0.1, angle_cut=0.05, trans_cut=2.0):
    nops = xyz.shape[0]
    L = xyz.shape[1] // 2

    # PASS 1: find all symm axes
    symmaxes = []
    for i in range(nops):
        # if there are multiple biomt records, this may occur.
        # rather than try to rescue, we will take the 1st (typically author-assigned)
        offset0 = th.linalg.norm(xyz[i, :L, 1] - xyz[0, :L, 1], dim=-1)
        if (th.mean(offset0) > 1e-4):
            continue

        # get alignment
        mask_i = mask[i, :L, 1] * mask[i, L:, 1]
        xyz_i = xyz[i, :L, 1][mask_i, :]
        xyz_j = xyz[i, L:, 1][mask_i, :]
        rms_ij, Uij, cI, cJ = kabsch(xyz_i, xyz_j)
        if (rms_ij > rms_cut):
            #print (i,'rms',rms_ij)
            continue

        # get axis and point symmetry about axis
        angle, axis = rotation_from_matrix(Uij)
        nfold = 2 * np.pi / th.abs(angle)
        # a) ensure integer # of subunits per rotation
        if (th.abs(nfold - th.round(nfold)) > nfold_cut):
            #print ('nfold fail',nfold)
            continue
        nfold = th.round(nfold).long()
        # b) ensure rotation only (no translation)
        delCOM = th.mean(xyz_i, dim=-2) - th.mean(xyz_j, dim=-2)
        trans_dot_symaxis = nfold * th.abs(th.dot(delCOM, axis))
        if (trans_dot_symaxis > trans_cut):
            #print ('trans fail',trans_dot_symaxis)
            continue

        # 3) get a point on the symm axis from CoMs and angle
        cIJ = th.sign(angle) * (cJ - cI).squeeze(0)
        dIJ = th.linalg.norm(cIJ)
        p_mid = (cI + cJ).squeeze(0) / 2
        u = cIJ / dIJ  # unit vector in plane of circle
        v = th.cross(axis, u)  # unit vector from sym axis to p_mid
        r = dIJ / (2 * th.sin(angle / 2))
        d = th.sqrt(r*r - dIJ*dIJ/4)  # distance from mid-chord to center
        point = p_mid - (d) * v

        # check if redundant
        toadd = True
        for j, (nf_j, ax_j, pt_j, err_j) in enumerate(symmaxes):
            if (not intersect(pt_j, ax_j, point, axis)):
                continue
            angle_j = get_angle(ax_j, axis)
            if (angle_j < angle_cut):
                if (nf_j < nfold):  # stored is a subsymmetry of complex, overwrite
                    symmaxes[j] = (nfold, axis, point, i)
                toadd = False

        if (toadd):
            symmaxes.append((nfold, axis, point, i))

    # PASS 2: combine
    symmgroup = 'C1'
    subsymm = []
    if len(symmaxes) == 1:
        symmgroup = 'C%d' % (symmaxes[0][0])
        subsymm = [symmaxes[0][3]]
    elif len(symmaxes) > 1:
        symmaxes = sorted(symmaxes, key=lambda x: x[0], reverse=True)
        angle = get_angle(symmaxes[0][1], symmaxes[1][1])
        subsymm = [symmaxes[0][3], symmaxes[1][3]]

        # 2-fold and n-fold intersect at 90 degress => Dn
        if (symmaxes[1][0] == 2 and th.abs(angle - np.pi / 2) < angle_cut):
            symmgroup = 'D%d' % (symmaxes[0][0])
        else:
            # polyhedral rules:
            #   3-Fold + 2-fold intersecting at acos(-1/sqrt(3)) -> T
            angle_tgt = np.arccos(-1 / np.sqrt(3))
            if (symmaxes[0][0] == 3 and symmaxes[1][0] == 2 and th.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'T'

            #   3-Fold + 2-fold intersecting at asin(1/sqrt(3)) -> O
            angle_tgt = np.arcsin(1 / np.sqrt(3))
            if (symmaxes[0][0] == 3 and symmaxes[1][0] == 2 and th.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'O'

            #   4-Fold + 3-fold intersecting at acos(1/sqrt(3)) -> O
            angle_tgt = np.arccos(1 / np.sqrt(3))
            if (symmaxes[0][0] == 4 and symmaxes[1][0] == 3 and th.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'O'

            #   3-Fold + 2-fold intersecting at 0.5*acos(sqrt(5)/3) -> I
            angle_tgt = 0.5 * np.arccos(np.sqrt(5) / 3)
            if (symmaxes[0][0] == 3 and symmaxes[1][0] == 2 and th.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'I'

            #   5-Fold + 2-fold intersecting at 0.5*acos(1/sqrt(5)) -> I
            angle_tgt = 0.5 * np.arccos(1 / np.sqrt(5))
            if (symmaxes[0][0] == 5 and symmaxes[1][0] == 2 and th.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'I'

            #   5-Fold + 3-fold intersecting at 0.5*acos((4*sqrt(5)-5)/15) -> I
            angle_tgt = 0.5 * np.arccos((4 * np.sqrt(5) - 5) / 15)
            if (symmaxes[0][0] == 5 and symmaxes[1][0] == 3 and th.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'I'
            else:
                pass
                #fd: we could use a single symmetry here instead.
                #    But these cases mostly are bad BIOUNIT annotations...
                #print ('nomatch',angle, [(x,y) for x,_,_,y in symmaxes])

    return symmgroup, subsymm

def symm_subunit_matrix(symmid, symopt=None):
    if (symmid[0] == 'C'):
        nsub = int(symmid[1:])
        symmatrix = (th.arange(nsub)[:, None] - th.arange(nsub)[None, :]) % nsub
        angles = th.linspace(0, 2 * np.pi, nsub + 1)[:nsub]
        Rs = generateC(angles)

        metasymm = ([th.arange(nsub)], [min(nsub, symopt.max_nsub if symopt else min(3, nsub))])

        if (nsub == 1):
            D = 0.0
        else:
            est_radius = 2.0 * SYMA
            theta = 2.0 * np.pi / nsub
            D = est_radius / np.sin(theta / 2)

        offset = th.tensor([float(D), 0.0, 0.0])
    elif (symmid[0] == 'D'):
        nsub = int(symmid[1:])
        cblk = (th.arange(nsub)[:, None] - th.arange(nsub)[None, :]) % nsub
        symmatrix = th.zeros((2 * nsub, 2 * nsub), dtype=th.long)
        symmatrix[:nsub, :nsub] = cblk
        symmatrix[:nsub, nsub:] = cblk + nsub
        symmatrix[nsub:, :nsub] = cblk + nsub
        symmatrix[nsub:, nsub:] = cblk
        angles = th.linspace(0, 2 * np.pi, nsub + 1)[:nsub]
        Rs = generateD(angles)

        metasymm = ([th.arange(nsub),
                     nsub + th.arange(nsub)], [min(nsub, symopt.max_nsub if symopt else min(3, nsub)), 2])
        #metasymm = (
        #    [th.arange(2*nsub)],
        #    [min(2*nsub,5)]
        #)

        est_radius = 2.0 * SYMA
        theta1 = 2.0 * np.pi / nsub
        theta2 = np.pi
        D1 = est_radius / np.sin(theta1 / 2)
        D2 = est_radius / np.sin(theta2 / 2)
        offset = th.tensor([float(D1), float(D2), 0.0])
    elif (symmid == 'T'):
        symmatrix = th.tensor([[0, 1, 2, 3, 8, 11, 9, 10, 4, 6, 7, 5], [1, 0, 3, 2, 9, 10, 8, 11, 5, 7, 6, 4],
                               [2, 3, 0, 1, 10, 9, 11, 8, 6, 4, 5, 7], [3, 2, 1, 0, 11, 8, 10, 9, 7, 5, 4, 6],
                               [4, 6, 7, 5, 0, 1, 2, 3, 8, 11, 9, 10], [5, 7, 6, 4, 1, 0, 3, 2, 9, 10, 8, 11],
                               [6, 4, 5, 7, 2, 3, 0, 1, 10, 9, 11, 8], [7, 5, 4, 6, 3, 2, 1, 0, 11, 8, 10, 9],
                               [8, 11, 9, 10, 4, 6, 7, 5, 0, 1, 2, 3], [9, 10, 8, 11, 5, 7, 6, 4, 1, 0, 3, 2],
                               [10, 9, 11, 8, 6, 4, 5, 7, 2, 3, 0, 1], [11, 8, 10, 9, 7, 5, 4, 6, 3, 2, 1, 0]])
        Rs = th.zeros(12, 3, 3)
        Rs[0] = th.tensor([[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000],
                           [0.000000, 0.000000, 1.000000]])
        Rs[1] = th.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, -1.000000, 0.000000],
                           [0.000000, 0.000000, 1.000000]])
        Rs[2] = th.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000],
                           [0.000000, 0.000000, -1.000000]])
        Rs[3] = th.tensor([[1.000000, 0.000000, 0.000000], [0.000000, -1.000000, 0.000000],
                           [0.000000, 0.000000, -1.000000]])
        Rs[4] = th.tensor([[0.000000, 0.000000, 1.000000], [1.000000, 0.000000, 0.000000],
                           [0.000000, 1.000000, 0.000000]])
        Rs[5] = th.tensor([[0.000000, 0.000000, 1.000000], [-1.000000, 0.000000, 0.000000],
                           [0.000000, -1.000000, 0.000000]])
        Rs[6] = th.tensor([[0.000000, 0.000000, -1.000000], [-1.000000, 0.000000, 0.000000],
                           [0.000000, 1.000000, 0.000000]])
        Rs[7] = th.tensor([[0.000000, 0.000000, -1.000000], [1.000000, 0.000000, 0.000000],
                           [0.000000, -1.000000, 0.000000]])
        Rs[8] = th.tensor([[0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                           [1.000000, 0.000000, 0.000000]])
        Rs[9] = th.tensor([[0.000000, -1.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                           [-1.000000, 0.000000, 0.000000]])
        Rs[10] = th.tensor([[0.000000, 1.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                            [-1.000000, 0.000000, 0.000000]])
        Rs[11] = th.tensor([[0.000000, -1.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                            [1.000000, 0.000000, 0.000000]])

        est_radius = 4.0 * SYMA
        offset = th.tensor([1.0, 0.0, 0.0])
        offset = est_radius * offset / th.linalg.norm(offset)
        metasymm = ([th.arange(12)], [min(len(Rs), symopt.max_nsub if symopt else 6)])

    elif (symmid == 'O'):
        symmatrix = th.tensor(
            [[0, 1, 2, 3, 8, 11, 9, 10, 4, 6, 7, 5, 12, 13, 15, 14, 19, 17, 18, 16, 22, 21, 20, 23],
             [1, 0, 3, 2, 9, 10, 8, 11, 5, 7, 6, 4, 13, 12, 14, 15, 18, 16, 19, 17, 23, 20, 21, 22],
             [2, 3, 0, 1, 10, 9, 11, 8, 6, 4, 5, 7, 14, 15, 13, 12, 17, 19, 16, 18, 20, 23, 22, 21],
             [3, 2, 1, 0, 11, 8, 10, 9, 7, 5, 4, 6, 15, 14, 12, 13, 16, 18, 17, 19, 21, 22, 23, 20],
             [4, 6, 7, 5, 0, 1, 2, 3, 8, 11, 9, 10, 16, 18, 17, 19, 21, 22, 23, 20, 15, 14, 12, 13],
             [5, 7, 6, 4, 1, 0, 3, 2, 9, 10, 8, 11, 17, 19, 16, 18, 20, 23, 22, 21, 14, 15, 13, 12],
             [6, 4, 5, 7, 2, 3, 0, 1, 10, 9, 11, 8, 18, 16, 19, 17, 23, 20, 21, 22, 13, 12, 14, 15],
             [7, 5, 4, 6, 3, 2, 1, 0, 11, 8, 10, 9, 19, 17, 18, 16, 22, 21, 20, 23, 12, 13, 15, 14],
             [8, 11, 9, 10, 4, 6, 7, 5, 0, 1, 2, 3, 20, 23, 22, 21, 14, 15, 13, 12, 17, 19, 16, 18],
             [9, 10, 8, 11, 5, 7, 6, 4, 1, 0, 3, 2, 21, 22, 23, 20, 15, 14, 12, 13, 16, 18, 17, 19],
             [10, 9, 11, 8, 6, 4, 5, 7, 2, 3, 0, 1, 22, 21, 20, 23, 12, 13, 15, 14, 19, 17, 18, 16],
             [11, 8, 10, 9, 7, 5, 4, 6, 3, 2, 1, 0, 23, 20, 21, 22, 13, 12, 14, 15, 18, 16, 19, 17],
             [12, 13, 15, 14, 19, 17, 18, 16, 22, 21, 20, 23, 0, 1, 2, 3, 8, 11, 9, 10, 4, 6, 7, 5],
             [13, 12, 14, 15, 18, 16, 19, 17, 23, 20, 21, 22, 1, 0, 3, 2, 9, 10, 8, 11, 5, 7, 6, 4],
             [14, 15, 13, 12, 17, 19, 16, 18, 20, 23, 22, 21, 2, 3, 0, 1, 10, 9, 11, 8, 6, 4, 5, 7],
             [15, 14, 12, 13, 16, 18, 17, 19, 21, 22, 23, 20, 3, 2, 1, 0, 11, 8, 10, 9, 7, 5, 4, 6],
             [16, 18, 17, 19, 21, 22, 23, 20, 15, 14, 12, 13, 4, 6, 7, 5, 0, 1, 2, 3, 8, 11, 9, 10],
             [17, 19, 16, 18, 20, 23, 22, 21, 14, 15, 13, 12, 5, 7, 6, 4, 1, 0, 3, 2, 9, 10, 8, 11],
             [18, 16, 19, 17, 23, 20, 21, 22, 13, 12, 14, 15, 6, 4, 5, 7, 2, 3, 0, 1, 10, 9, 11, 8],
             [19, 17, 18, 16, 22, 21, 20, 23, 12, 13, 15, 14, 7, 5, 4, 6, 3, 2, 1, 0, 11, 8, 10, 9],
             [20, 23, 22, 21, 14, 15, 13, 12, 17, 19, 16, 18, 8, 11, 9, 10, 4, 6, 7, 5, 0, 1, 2, 3],
             [21, 22, 23, 20, 15, 14, 12, 13, 16, 18, 17, 19, 9, 10, 8, 11, 5, 7, 6, 4, 1, 0, 3, 2],
             [22, 21, 20, 23, 12, 13, 15, 14, 19, 17, 18, 16, 10, 9, 11, 8, 6, 4, 5, 7, 2, 3, 0, 1],
             [23, 20, 21, 22, 13, 12, 14, 15, 18, 16, 19, 17, 11, 8, 10, 9, 7, 5, 4, 6, 3, 2, 1, 0]])
        Rs = th.zeros(24, 3, 3)
        Rs[0] = th.tensor([[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000],
                           [0.000000, 0.000000, 1.000000]])
        Rs[1] = th.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, -1.000000, 0.000000],
                           [0.000000, 0.000000, 1.000000]])
        Rs[2] = th.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000],
                           [0.000000, 0.000000, -1.000000]])
        Rs[3] = th.tensor([[1.000000, 0.000000, 0.000000], [0.000000, -1.000000, 0.000000],
                           [0.000000, 0.000000, -1.000000]])
        Rs[4] = th.tensor([[0.000000, 0.000000, 1.000000], [1.000000, 0.000000, 0.000000],
                           [0.000000, 1.000000, 0.000000]])
        Rs[5] = th.tensor([[0.000000, 0.000000, 1.000000], [-1.000000, 0.000000, 0.000000],
                           [0.000000, -1.000000, 0.000000]])
        Rs[6] = th.tensor([[0.000000, 0.000000, -1.000000], [-1.000000, 0.000000, 0.000000],
                           [0.000000, 1.000000, 0.000000]])
        Rs[7] = th.tensor([[0.000000, 0.000000, -1.000000], [1.000000, 0.000000, 0.000000],
                           [0.000000, -1.000000, 0.000000]])
        Rs[8] = th.tensor([[0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                           [1.000000, 0.000000, 0.000000]])
        Rs[9] = th.tensor([[0.000000, -1.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                           [-1.000000, 0.000000, 0.000000]])
        Rs[10] = th.tensor([[0.000000, 1.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                            [-1.000000, 0.000000, 0.000000]])
        Rs[11] = th.tensor([[0.000000, -1.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                            [1.000000, 0.000000, 0.000000]])
        Rs[12] = th.tensor([[0.000000, 1.000000, 0.000000], [1.000000, 0.000000, 0.000000],
                            [0.000000, 0.000000, -1.000000]])
        Rs[13] = th.tensor([[0.000000, -1.000000, 0.000000], [-1.000000, 0.000000, 0.000000],
                            [0.000000, 0.000000, -1.000000]])
        Rs[14] = th.tensor([[0.000000, 1.000000, 0.000000], [-1.000000, 0.000000, 0.000000],
                            [0.000000, 0.000000, 1.000000]])
        Rs[15] = th.tensor([[0.000000, -1.000000, 0.000000], [1.000000, 0.000000, 0.000000],
                            [0.000000, 0.000000, 1.000000]])
        Rs[16] = th.tensor([[1.000000, 0.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                            [0.000000, -1.000000, 0.000000]])
        Rs[17] = th.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                            [0.000000, 1.000000, 0.000000]])
        Rs[18] = th.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                            [0.000000, -1.000000, 0.000000]])
        Rs[19] = th.tensor([[1.000000, 0.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                            [0.000000, 1.000000, 0.000000]])
        Rs[20] = th.tensor([[0.000000, 0.000000, 1.000000], [0.000000, 1.000000, 0.000000],
                            [-1.000000, 0.000000, 0.000000]])
        Rs[21] = th.tensor([[0.000000, 0.000000, 1.000000], [0.000000, -1.000000, 0.000000],
                            [1.000000, 0.000000, 0.000000]])
        Rs[22] = th.tensor([[0.000000, 0.000000, -1.000000], [0.000000, 1.000000, 0.000000],
                            [1.000000, 0.000000, 0.000000]])
        Rs[23] = th.tensor([[0.000000, 0.000000, -1.000000], [0.000000, -1.000000, 0.000000],
                            [-1.000000, 0.000000, 0.000000]])

        est_radius = 6.0 * SYMA
        offset = th.tensor([1.0, 0.0, 0.0])
        offset = est_radius * offset / th.linalg.norm(offset)
        metasymm = ([th.arange(24)], [min(len(Rs), symopt.max_nsub if symopt else 6)])
    elif (symmid == 'I'):
        symmatrix = th.tensor(
            [[
                0, 4, 3, 2, 1, 5, 33, 49, 41, 22, 10, 27, 51, 59, 38, 15, 16, 17, 18, 19, 40, 21, 9,
                32, 48, 55, 39, 11, 28, 52, 45, 42, 23, 6, 34, 50, 58, 37, 14, 26, 20, 8,
                31, 47, 44, 30, 46, 43, 24, 7, 35, 12, 29, 53, 56, 25, 54, 57, 36, 13
            ],
             [
                 1, 0, 4, 3, 2, 6,
                 34, 45, 42, 23, 11, 28, 52, 55, 39, 16, 17, 18, 19, 15, 41, 22, 5, 33, 49, 56, 35, 12, 29, 53,
                 46, 43, 24, 7, 30, 51, 59, 38, 10, 27, 21, 9, 32, 48, 40, 31, 47, 44, 20, 8, 36, 13, 25, 54,
                 57, 26, 50, 58, 37, 14
             ],
             [
                 2, 1, 0, 4, 3, 7,
                 30, 46, 43, 24, 12, 29, 53, 56, 35, 17, 18, 19, 15, 16, 42, 23, 6, 34, 45, 57, 36, 13, 25, 54,
                 47, 44, 20, 8, 31, 52, 55, 39, 11, 28, 22, 5, 33, 49, 41, 32, 48, 40, 21, 9, 37, 14, 26, 50,
                 58, 27, 51, 59, 38, 10
             ],
             [
                 3, 2, 1, 0, 4, 8,
                 31, 47, 44, 20, 13, 25, 54, 57, 36, 18, 19, 15, 16, 17, 43, 24, 7, 30, 46, 58, 37, 14, 26, 50,
                 48, 40, 21, 9, 32, 53, 56, 35, 12, 29, 23, 6, 34, 45, 42, 33, 49, 41, 22, 5, 38, 10, 27, 51,
                 59, 28, 52, 55, 39, 11
             ],
             [
                 4, 3, 2, 1, 0, 9,
                 32, 48, 40, 21, 14, 26, 50, 58, 37, 19, 15, 16, 17, 18, 44, 20, 8, 31, 47, 59, 38, 10, 27, 51,
                 49, 41, 22, 5, 33, 54, 57, 36, 13, 25, 24, 7, 30, 46, 43, 34, 45, 42, 23, 6, 39, 11, 28, 52,
                 55, 29, 53, 56, 35, 12
             ],
             [
                 5, 33, 49, 41, 22, 0, 4, 3, 2, 1, 15, 16, 17, 18, 19, 10, 27, 51, 59, 38, 45, 42, 23, 6, 34,
                 50, 58, 37, 14, 26, 40, 21, 9, 32, 48, 55, 39, 11, 28, 52, 25, 54, 57, 36, 13, 35, 12, 29, 53,
                 56, 30, 46, 43, 24, 7, 20, 8, 31, 47, 44
             ],
             [
                 6, 34, 45, 42, 23, 1, 0, 4, 3, 2, 16, 17, 18, 19, 15, 11, 28, 52, 55, 39, 46, 43, 24, 7, 30,
                 51, 59, 38, 10, 27, 41, 22, 5, 33, 49, 56, 35, 12, 29, 53, 26, 50, 58, 37, 14, 36, 13, 25, 54,
                 57, 31, 47, 44, 20, 8, 21, 9, 32, 48, 40
             ],
             [
                 7, 30, 46, 43, 24, 2, 1, 0, 4, 3, 17, 18, 19, 15, 16, 12, 29, 53, 56, 35, 47, 44, 20, 8, 31,
                 52, 55, 39, 11, 28, 42, 23, 6, 34, 45, 57, 36, 13, 25, 54, 27, 51, 59, 38, 10, 37, 14, 26, 50,
                 58, 32, 48, 40, 21, 9, 22, 5, 33, 49, 41
             ],
             [
                 8, 31, 47, 44, 20, 3, 2, 1, 0, 4, 18, 19, 15, 16, 17, 13, 25, 54, 57, 36, 48, 40, 21, 9, 32,
                 53, 56, 35, 12, 29, 43, 24, 7, 30, 46, 58, 37, 14, 26, 50, 28, 52, 55, 39, 11, 38, 10, 27, 51,
                 59, 33, 49, 41, 22, 5, 23, 6, 34, 45, 42
             ],
             [
                 9, 32, 48, 40, 21, 4, 3, 2, 1, 0, 19, 15, 16, 17, 18, 14, 26, 50, 58, 37, 49, 41, 22, 5, 33,
                 54, 57, 36, 13, 25, 44, 20, 8, 31, 47, 59, 38, 10, 27, 51, 29, 53, 56, 35, 12, 39, 11, 28, 52,
                 55, 34, 45, 42, 23, 6, 24, 7, 30, 46, 43
             ],
             [
                 10, 27, 51, 59, 38, 15, 16, 17, 18, 19, 0, 4, 3, 2, 1, 5, 33, 49, 41, 22, 50, 58, 37, 14, 26,
                 45, 42, 23, 6, 34, 55, 39, 11, 28, 52, 40, 21, 9, 32, 48, 30, 46, 43, 24, 7, 20, 8, 31, 47,
                 44, 25, 54, 57, 36, 13, 35, 12, 29, 53, 56
             ],
             [
                 11, 28, 52, 55, 39, 16, 17, 18, 19, 15, 1, 0, 4, 3, 2, 6, 34, 45, 42, 23, 51, 59, 38, 10, 27,
                 46, 43, 24, 7, 30, 56, 35, 12, 29, 53, 41, 22, 5, 33, 49, 31, 47, 44, 20, 8, 21, 9, 32, 48,
                 40, 26, 50, 58, 37, 14, 36, 13, 25, 54, 57
             ],
             [
                 12, 29, 53, 56, 35, 17, 18, 19, 15, 16, 2, 1, 0, 4, 3, 7, 30, 46, 43, 24, 52, 55, 39, 11, 28,
                 47, 44, 20, 8, 31, 57, 36, 13, 25, 54, 42, 23, 6, 34, 45, 32, 48, 40, 21, 9, 22, 5, 33, 49,
                 41, 27, 51, 59, 38, 10, 37, 14, 26, 50, 58
             ],
             [
                 13, 25, 54, 57, 36, 18, 19, 15, 16, 17, 3, 2, 1, 0, 4, 8, 31, 47, 44, 20, 53, 56, 35, 12, 29,
                 48, 40, 21, 9, 32, 58, 37, 14, 26, 50, 43, 24, 7, 30, 46, 33, 49, 41, 22, 5, 23, 6, 34, 45,
                 42, 28, 52, 55, 39, 11, 38, 10, 27, 51, 59
             ],
             [
                 14, 26, 50, 58, 37, 19, 15, 16, 17, 18, 4, 3, 2, 1, 0, 9, 32, 48, 40, 21, 54, 57, 36, 13, 25,
                 49, 41, 22, 5, 33, 59, 38, 10, 27, 51, 44, 20, 8, 31, 47, 34, 45, 42, 23, 6, 24, 7, 30, 46,
                 43, 29, 53, 56, 35, 12, 39, 11, 28, 52, 55
             ],
             [
                 15, 16, 17, 18, 19, 10, 27, 51, 59, 38, 5, 33, 49, 41, 22, 0, 4, 3, 2, 1, 55, 39, 11, 28, 52,
                 40, 21, 9, 32, 48, 50, 58, 37, 14, 26, 45, 42, 23, 6, 34, 35, 12, 29, 53, 56, 25, 54, 57, 36,
                 13, 20, 8, 31, 47, 44, 30, 46, 43, 24, 7
             ],
             [
                 16, 17, 18, 19, 15, 11, 28, 52, 55, 39, 6, 34, 45, 42, 23, 1, 0, 4, 3, 2, 56, 35, 12, 29, 53,
                 41, 22, 5, 33, 49, 51, 59, 38, 10, 27, 46, 43, 24, 7, 30, 36, 13, 25, 54, 57, 26, 50, 58, 37,
                 14, 21, 9, 32, 48, 40, 31, 47, 44, 20, 8
             ],
             [
                 17, 18, 19, 15, 16, 12, 29, 53, 56, 35, 7, 30, 46, 43, 24, 2, 1, 0, 4, 3, 57, 36, 13, 25, 54,
                 42, 23, 6, 34, 45, 52, 55, 39, 11, 28, 47, 44, 20, 8, 31, 37, 14, 26, 50, 58, 27, 51, 59, 38,
                 10, 22, 5, 33, 49, 41, 32, 48, 40, 21, 9
             ],
             [
                 18, 19, 15, 16, 17, 13, 25, 54, 57, 36, 8, 31, 47, 44, 20, 3, 2, 1, 0, 4, 58, 37, 14, 26, 50,
                 43, 24, 7, 30, 46, 53, 56, 35, 12, 29, 48, 40, 21, 9, 32, 38, 10, 27, 51, 59, 28, 52, 55, 39,
                 11, 23, 6, 34, 45, 42, 33, 49, 41, 22, 5
             ],
             [
                 19, 15, 16, 17, 18, 14, 26, 50, 58, 37, 9, 32, 48, 40, 21, 4, 3, 2, 1, 0, 59, 38, 10, 27, 51,
                 44, 20, 8, 31, 47, 54, 57, 36, 13, 25, 49, 41, 22, 5, 33, 39, 11, 28, 52, 55, 29, 53, 56, 35,
                 12, 24, 7, 30, 46, 43, 34, 45, 42, 23, 6
             ],
             [
                 20, 8, 31, 47, 44, 30, 46, 43, 24, 7, 35, 12, 29, 53, 56, 25, 54, 57, 36, 13, 0, 4, 3, 2, 1,
                 5, 33, 49, 41, 22, 10, 27, 51, 59, 38, 15, 16, 17, 18, 19, 40, 21, 9, 32, 48, 55, 39, 11, 28,
                 52, 45, 42, 23, 6, 34, 50, 58, 37, 14, 26
             ],
             [
                 21, 9, 32, 48, 40, 31, 47, 44, 20, 8, 36, 13, 25, 54, 57, 26, 50, 58, 37, 14, 1, 0, 4, 3, 2,
                 6, 34, 45, 42, 23, 11, 28, 52, 55, 39, 16, 17, 18, 19, 15, 41, 22, 5, 33, 49, 56, 35, 12, 29,
                 53, 46, 43, 24, 7, 30, 51, 59, 38, 10, 27
             ],
             [
                 22, 5, 33, 49, 41, 32, 48, 40, 21, 9, 37, 14, 26, 50, 58, 27, 51, 59, 38, 10, 2, 1, 0, 4, 3,
                 7, 30, 46, 43, 24, 12, 29, 53, 56, 35, 17, 18, 19, 15, 16, 42, 23, 6, 34, 45, 57, 36, 13, 25,
                 54, 47, 44, 20, 8, 31, 52, 55, 39, 11, 28
             ],
             [
                 23, 6, 34, 45, 42, 33, 49, 41, 22, 5, 38, 10, 27, 51, 59, 28, 52, 55, 39, 11, 3, 2, 1, 0, 4,
                 8, 31, 47, 44, 20, 13, 25, 54, 57, 36, 18, 19, 15, 16, 17, 43, 24, 7, 30, 46, 58, 37, 14, 26,
                 50, 48, 40, 21, 9, 32, 53, 56, 35, 12, 29
             ],
             [
                 24, 7, 30, 46, 43, 34, 45, 42, 23, 6, 39, 11, 28, 52, 55, 29, 53, 56, 35, 12, 4, 3, 2, 1, 0,
                 9, 32, 48, 40, 21, 14, 26, 50, 58, 37, 19, 15, 16, 17, 18, 44, 20, 8, 31, 47, 59, 38, 10, 27,
                 51, 49, 41, 22, 5, 33, 54, 57, 36, 13, 25
             ],
             [
                 25, 54, 57, 36, 13, 35, 12, 29, 53, 56, 30, 46, 43, 24, 7, 20, 8, 31, 47, 44, 5, 33, 49, 41,
                 22, 0, 4, 3, 2, 1, 15, 16, 17, 18, 19, 10, 27, 51, 59, 38, 45, 42, 23, 6, 34, 50, 58, 37, 14,
                 26, 40, 21, 9, 32, 48, 55, 39, 11, 28, 52
             ],
             [
                 26, 50, 58, 37, 14, 36, 13, 25, 54, 57, 31, 47, 44, 20, 8, 21, 9, 32, 48, 40, 6, 34, 45, 42,
                 23, 1, 0, 4, 3, 2, 16, 17, 18, 19, 15, 11, 28, 52, 55, 39, 46, 43, 24, 7, 30, 51, 59, 38, 10,
                 27, 41, 22, 5, 33, 49, 56, 35, 12, 29, 53
             ],
             [
                 27, 51, 59, 38, 10, 37, 14, 26, 50, 58, 32, 48, 40, 21, 9, 22, 5, 33, 49, 41, 7, 30, 46, 43,
                 24, 2, 1, 0, 4, 3, 17, 18, 19, 15, 16, 12, 29, 53, 56, 35, 47, 44, 20, 8, 31, 52, 55, 39, 11,
                 28, 42, 23, 6, 34, 45, 57, 36, 13, 25, 54
             ],
             [
                 28, 52, 55, 39, 11, 38, 10, 27, 51, 59, 33, 49, 41, 22, 5, 23, 6, 34, 45, 42, 8, 31, 47, 44,
                 20, 3, 2, 1, 0, 4, 18, 19, 15, 16, 17, 13, 25, 54, 57, 36, 48, 40, 21, 9, 32, 53, 56, 35, 12,
                 29, 43, 24, 7, 30, 46, 58, 37, 14, 26, 50
             ],
             [
                 29, 53, 56, 35, 12, 39, 11, 28, 52, 55, 34, 45, 42, 23, 6, 24, 7, 30, 46, 43, 9, 32, 48, 40,
                 21, 4, 3, 2, 1, 0, 19, 15, 16, 17, 18, 14, 26, 50, 58, 37, 49, 41, 22, 5, 33, 54, 57, 36, 13,
                 25, 44, 20, 8, 31, 47, 59, 38, 10, 27, 51
             ],
             [
                 30, 46, 43, 24, 7, 20, 8, 31, 47, 44, 25, 54, 57, 36, 13, 35, 12, 29, 53, 56, 10, 27, 51, 59,
                 38, 15, 16, 17, 18, 19, 0, 4, 3, 2, 1, 5, 33, 49, 41, 22, 50, 58, 37, 14, 26, 45, 42, 23, 6,
                 34, 55, 39, 11, 28, 52, 40, 21, 9, 32, 48
             ],
             [
                 31, 47, 44, 20, 8, 21, 9, 32, 48, 40, 26, 50, 58, 37, 14, 36, 13, 25, 54, 57, 11, 28, 52, 55,
                 39, 16, 17, 18, 19, 15, 1, 0, 4, 3, 2, 6, 34, 45, 42, 23, 51, 59, 38, 10, 27, 46, 43, 24, 7,
                 30, 56, 35, 12, 29, 53, 41, 22, 5, 33, 49
             ],
             [
                 32, 48, 40, 21, 9, 22, 5, 33, 49, 41, 27, 51, 59, 38, 10, 37, 14, 26, 50, 58, 12, 29, 53, 56,
                 35, 17, 18, 19, 15, 16, 2, 1, 0, 4, 3, 7, 30, 46, 43, 24, 52, 55, 39, 11, 28, 47, 44, 20, 8,
                 31, 57, 36, 13, 25, 54, 42, 23, 6, 34, 45
             ],
             [
                 33, 49, 41, 22, 5, 23, 6, 34, 45, 42, 28, 52, 55, 39, 11, 38, 10, 27, 51, 59, 13, 25, 54, 57,
                 36, 18, 19, 15, 16, 17, 3, 2, 1, 0, 4, 8, 31, 47, 44, 20, 53, 56, 35, 12, 29, 48, 40, 21, 9,
                 32, 58, 37, 14, 26, 50, 43, 24, 7, 30, 46
             ],
             [
                 34, 45, 42, 23, 6, 24, 7, 30, 46, 43, 29, 53, 56, 35, 12, 39, 11, 28, 52, 55, 14, 26, 50, 58,
                 37, 19, 15, 16, 17, 18, 4, 3, 2, 1, 0, 9, 32, 48, 40, 21, 54, 57, 36, 13, 25, 49, 41, 22, 5,
                 33, 59, 38, 10, 27, 51, 44, 20, 8, 31, 47
             ],
             [
                 35, 12, 29, 53, 56, 25, 54, 57, 36, 13, 20, 8, 31, 47, 44, 30, 46, 43, 24, 7, 15, 16, 17, 18,
                 19, 10, 27, 51, 59, 38, 5, 33, 49, 41, 22, 0, 4, 3, 2, 1, 55, 39, 11, 28, 52, 40, 21, 9, 32,
                 48, 50, 58, 37, 14, 26, 45, 42, 23, 6, 34
             ],
             [
                 36, 13, 25, 54, 57, 26, 50, 58, 37, 14, 21, 9, 32, 48, 40, 31, 47, 44, 20, 8, 16, 17, 18, 19,
                 15, 11, 28, 52, 55, 39, 6, 34, 45, 42, 23, 1, 0, 4, 3, 2, 56, 35, 12, 29, 53, 41, 22, 5, 33,
                 49, 51, 59, 38, 10, 27, 46, 43, 24, 7, 30
             ],
             [
                 37, 14, 26, 50, 58, 27, 51, 59, 38, 10, 22, 5, 33, 49, 41, 32, 48, 40, 21, 9, 17, 18, 19, 15,
                 16, 12, 29, 53, 56, 35, 7, 30, 46, 43, 24, 2, 1, 0, 4, 3, 57, 36, 13, 25, 54, 42, 23, 6, 34,
                 45, 52, 55, 39, 11, 28, 47, 44, 20, 8, 31
             ],
             [
                 38, 10, 27, 51, 59, 28, 52, 55, 39, 11, 23, 6, 34, 45, 42, 33, 49, 41, 22, 5, 18, 19, 15, 16,
                 17, 13, 25, 54, 57, 36, 8, 31, 47, 44, 20, 3, 2, 1, 0, 4, 58, 37, 14, 26, 50, 43, 24, 7, 30,
                 46, 53, 56, 35, 12, 29, 48, 40, 21, 9, 32
             ],
             [
                 39, 11, 28, 52, 55, 29, 53, 56, 35, 12, 24, 7, 30, 46, 43, 34, 45, 42, 23, 6, 19, 15, 16, 17,
                 18, 14, 26, 50, 58, 37, 9, 32, 48, 40, 21, 4, 3, 2, 1, 0, 59, 38, 10, 27, 51, 44, 20, 8, 31,
                 47, 54, 57, 36, 13, 25, 49, 41, 22, 5, 33
             ],
             [
                 40, 21, 9, 32, 48, 55, 39, 11, 28, 52, 45, 42, 23, 6, 34, 50, 58, 37, 14, 26, 20, 8, 31, 47,
                 44, 30, 46, 43, 24, 7, 35, 12, 29, 53, 56, 25, 54, 57, 36, 13, 0, 4, 3, 2, 1, 5, 33, 49, 41,
                 22, 10, 27, 51, 59, 38, 15, 16, 17, 18, 19
             ],
             [
                 41, 22, 5, 33, 49, 56, 35, 12, 29, 53, 46, 43, 24, 7, 30, 51, 59, 38, 10, 27, 21, 9, 32, 48,
                 40, 31, 47, 44, 20, 8, 36, 13, 25, 54, 57, 26, 50, 58, 37, 14, 1, 0, 4, 3, 2, 6, 34, 45, 42,
                 23, 11, 28, 52, 55, 39, 16, 17, 18, 19, 15
             ],
             [
                 42, 23, 6, 34, 45, 57, 36, 13, 25, 54, 47, 44, 20, 8, 31, 52, 55, 39, 11, 28, 22, 5, 33, 49,
                 41, 32, 48, 40, 21, 9, 37, 14, 26, 50, 58, 27, 51, 59, 38, 10, 2, 1, 0, 4, 3, 7, 30, 46, 43,
                 24, 12, 29, 53, 56, 35, 17, 18, 19, 15, 16
             ],
             [
                 43, 24, 7, 30, 46, 58, 37, 14, 26, 50, 48, 40, 21, 9, 32, 53, 56, 35, 12, 29, 23, 6, 34, 45,
                 42, 33, 49, 41, 22, 5, 38, 10, 27, 51, 59, 28, 52, 55, 39, 11, 3, 2, 1, 0, 4, 8, 31, 47, 44,
                 20, 13, 25, 54, 57, 36, 18, 19, 15, 16, 17
             ],
             [
                 44, 20, 8, 31, 47, 59, 38, 10, 27, 51, 49, 41, 22, 5, 33, 54, 57, 36, 13, 25, 24, 7, 30, 46,
                 43, 34, 45, 42, 23, 6, 39, 11, 28, 52, 55, 29, 53, 56, 35, 12, 4, 3, 2, 1, 0, 9, 32, 48, 40,
                 21, 14, 26, 50, 58, 37, 19, 15, 16, 17, 18
             ],
             [
                 45, 42, 23, 6, 34, 50, 58, 37, 14, 26, 40, 21, 9, 32, 48, 55, 39, 11, 28, 52, 25, 54, 57, 36,
                 13, 35, 12, 29, 53, 56, 30, 46, 43, 24, 7, 20, 8, 31, 47, 44, 5, 33, 49, 41, 22, 0, 4, 3, 2,
                 1, 15, 16, 17, 18, 19, 10, 27, 51, 59, 38
             ],
             [
                 46, 43, 24, 7, 30, 51, 59, 38, 10, 27, 41, 22, 5, 33, 49, 56, 35, 12, 29, 53, 26, 50, 58, 37,
                 14, 36, 13, 25, 54, 57, 31, 47, 44, 20, 8, 21, 9, 32, 48, 40, 6, 34, 45, 42, 23, 1, 0, 4, 3,
                 2, 16, 17, 18, 19, 15, 11, 28, 52, 55, 39
             ],
             [
                 47, 44, 20, 8, 31, 52, 55, 39, 11, 28, 42, 23, 6, 34, 45, 57, 36, 13, 25, 54, 27, 51, 59, 38,
                 10, 37, 14, 26, 50, 58, 32, 48, 40, 21, 9, 22, 5, 33, 49, 41, 7, 30, 46, 43, 24, 2, 1, 0, 4,
                 3, 17, 18, 19, 15, 16, 12, 29, 53, 56, 35
             ],
             [
                 48, 40, 21, 9, 32, 53, 56, 35, 12, 29, 43, 24, 7, 30, 46, 58, 37, 14, 26, 50, 28, 52, 55, 39,
                 11, 38, 10, 27, 51, 59, 33, 49, 41, 22, 5, 23, 6, 34, 45, 42, 8, 31, 47, 44, 20, 3, 2, 1, 0,
                 4, 18, 19, 15, 16, 17, 13, 25, 54, 57, 36
             ],
             [
                 49, 41, 22, 5, 33, 54, 57, 36, 13, 25, 44, 20, 8, 31, 47, 59, 38, 10, 27, 51, 29, 53, 56, 35,
                 12, 39, 11, 28, 52, 55, 34, 45, 42, 23, 6, 24, 7, 30, 46, 43, 9, 32, 48, 40, 21, 4, 3, 2, 1,
                 0, 19, 15, 16, 17, 18, 14, 26, 50, 58, 37
             ],
             [
                 50, 58, 37, 14, 26, 45, 42, 23, 6, 34, 55, 39, 11, 28, 52, 40, 21, 9, 32, 48, 30, 46, 43, 24,
                 7, 20, 8, 31, 47, 44, 25, 54, 57, 36, 13, 35, 12, 29, 53, 56, 10, 27, 51, 59, 38, 15, 16, 17,
                 18, 19, 0, 4, 3, 2, 1, 5, 33, 49, 41, 22
             ],
             [
                 51, 59, 38, 10, 27, 46, 43, 24, 7, 30, 56, 35, 12, 29, 53, 41, 22, 5, 33, 49, 31, 47, 44, 20,
                 8, 21, 9, 32, 48, 40, 26, 50, 58, 37, 14, 36, 13, 25, 54, 57, 11, 28, 52, 55, 39, 16, 17, 18,
                 19, 15, 1, 0, 4, 3, 2, 6, 34, 45, 42, 23
             ],
             [
                 52, 55, 39, 11, 28, 47, 44, 20, 8, 31, 57, 36, 13, 25, 54, 42, 23, 6, 34, 45, 32, 48, 40, 21,
                 9, 22, 5, 33, 49, 41, 27, 51, 59, 38, 10, 37, 14, 26, 50, 58, 12, 29, 53, 56, 35, 17, 18, 19,
                 15, 16, 2, 1, 0, 4, 3, 7, 30, 46, 43, 24
             ],
             [
                 53, 56, 35, 12, 29, 48, 40, 21, 9, 32, 58, 37, 14, 26, 50, 43, 24, 7, 30, 46, 33, 49, 41, 22,
                 5, 23, 6, 34, 45, 42, 28, 52, 55, 39, 11, 38, 10, 27, 51, 59, 13, 25, 54, 57, 36, 18, 19, 15,
                 16, 17, 3, 2, 1, 0, 4, 8, 31, 47, 44, 20
             ],
             [
                 54, 57, 36, 13, 25, 49, 41, 22, 5, 33, 59, 38, 10, 27, 51, 44, 20, 8, 31, 47, 34, 45, 42, 23,
                 6, 24, 7, 30, 46, 43, 29, 53, 56, 35, 12, 39, 11, 28, 52, 55, 14, 26, 50, 58, 37, 19, 15, 16,
                 17, 18, 4, 3, 2, 1, 0, 9, 32, 48, 40, 21
             ],
             [
                 55, 39, 11, 28, 52, 40, 21, 9, 32, 48, 50, 58, 37, 14, 26, 45, 42, 23, 6, 34, 35, 12, 29, 53,
                 56, 25, 54, 57, 36, 13, 20, 8, 31, 47, 44, 30, 46, 43, 24, 7, 15, 16, 17, 18, 19, 10, 27, 51,
                 59, 38, 5, 33, 49, 41, 22, 0, 4, 3, 2, 1
             ],
             [
                 56, 35, 12, 29, 53, 41, 22, 5, 33, 49, 51, 59, 38, 10, 27, 46, 43, 24, 7, 30, 36, 13, 25, 54,
                 57, 26, 50, 58, 37, 14, 21, 9, 32, 48, 40, 31, 47, 44, 20, 8, 16, 17, 18, 19, 15, 11, 28, 52,
                 55, 39, 6, 34, 45, 42, 23, 1, 0, 4, 3, 2
             ],
             [
                 57, 36, 13, 25, 54, 42, 23, 6, 34, 45, 52, 55, 39, 11, 28, 47, 44, 20, 8, 31, 37, 14, 26, 50,
                 58, 27, 51, 59, 38, 10, 22, 5, 33, 49, 41, 32, 48, 40, 21, 9, 17, 18, 19, 15, 16, 12, 29, 53,
                 56, 35, 7, 30, 46, 43, 24, 2, 1, 0, 4, 3
             ],
             [
                 58, 37, 14, 26, 50, 43, 24, 7, 30, 46, 53, 56, 35, 12, 29, 48, 40, 21, 9, 32, 38, 10, 27, 51,
                 59, 28, 52, 55, 39, 11, 23, 6, 34, 45, 42, 33, 49, 41, 22, 5, 18, 19, 15, 16, 17, 13, 25, 54,
                 57, 36, 8, 31, 47, 44, 20, 3, 2, 1, 0, 4
             ],
             [
                 59, 38, 10, 27, 51, 44, 20, 8, 31, 47, 54, 57, 36, 13, 25, 49, 41, 22, 5, 33, 39, 11, 28, 52,
                 55, 29, 53, 56, 35, 12, 24, 7, 30, 46, 43, 34, 45, 42, 23, 6, 19, 15, 16, 17, 18, 14, 26, 50,
                 58, 37, 9, 32, 48, 40, 21, 4, 3, 2, 1, 0
             ]])
        Rs = th.zeros(60, 3, 3)
        Rs[0] = th.tensor([[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000],
                           [0.000000, 0.000000, 1.000000]])
        Rs[1] = th.tensor([[0.500000, -0.809017, 0.309017], [0.809017, 0.309017, -0.500000],
                           [0.309017, 0.500000, 0.809017]])
        Rs[2] = th.tensor([[-0.309017, -0.500000, 0.809017], [0.500000, -0.809017, -0.309017],
                           [0.809017, 0.309017, 0.500000]])
        Rs[3] = th.tensor([[-0.309017, 0.500000, 0.809017], [-0.500000, -0.809017, 0.309017],
                           [0.809017, -0.309017, 0.500000]])
        Rs[4] = th.tensor([[0.500000, 0.809017, 0.309017], [-0.809017, 0.309017, 0.500000],
                           [0.309017, -0.500000, 0.809017]])
        Rs[5] = th.tensor([[-0.809017, 0.309017, 0.500000], [0.309017, -0.500000, 0.809017],
                           [0.500000, 0.809017, 0.309017]])
        Rs[6] = th.tensor([[0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                           [1.000000, 0.000000, 0.000000]])
        Rs[7] = th.tensor([[0.809017, 0.309017, -0.500000], [0.309017, 0.500000, 0.809017],
                           [0.500000, -0.809017, 0.309017]])
        Rs[8] = th.tensor([[0.500000, -0.809017, -0.309017], [0.809017, 0.309017, 0.500000],
                           [-0.309017, -0.500000, 0.809017]])
        Rs[9] = th.tensor([[-0.500000, -0.809017, 0.309017], [0.809017, -0.309017, 0.500000],
                           [-0.309017, 0.500000, 0.809017]])
        Rs[10] = th.tensor([[-0.500000, -0.809017, 0.309017], [-0.809017, 0.309017, -0.500000],
                            [0.309017, -0.500000, -0.809017]])
        Rs[11] = th.tensor([[-0.809017, 0.309017, 0.500000], [-0.309017, 0.500000, -0.809017],
                            [-0.500000, -0.809017, -0.309017]])
        Rs[12] = th.tensor([[0.000000, 1.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                            [-1.000000, 0.000000, 0.000000]])
        Rs[13] = th.tensor([[0.809017, 0.309017, -0.500000], [-0.309017, -0.500000, -0.809017],
                            [-0.500000, 0.809017, -0.309017]])
        Rs[14] = th.tensor([[0.500000, -0.809017, -0.309017], [-0.809017, -0.309017, -0.500000],
                            [0.309017, 0.500000, -0.809017]])
        Rs[15] = th.tensor([[0.309017, 0.500000, -0.809017], [0.500000, -0.809017, -0.309017],
                            [-0.809017, -0.309017, -0.500000]])
        Rs[16] = th.tensor([[0.309017, -0.500000, -0.809017], [-0.500000, -0.809017, 0.309017],
                            [-0.809017, 0.309017, -0.500000]])
        Rs[17] = th.tensor([[-0.500000, -0.809017, -0.309017], [-0.809017, 0.309017, 0.500000],
                            [-0.309017, 0.500000, -0.809017]])
        Rs[18] = th.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000],
                            [0.000000, 0.000000, -1.000000]])
        Rs[19] = th.tensor([[-0.500000, 0.809017, -0.309017], [0.809017, 0.309017, -0.500000],
                            [-0.309017, -0.500000, -0.809017]])
        Rs[20] = th.tensor([[-0.500000, -0.809017, -0.309017], [0.809017, -0.309017, -0.500000],
                            [0.309017, -0.500000, 0.809017]])
        Rs[21] = th.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, -1.000000, 0.000000],
                            [0.000000, 0.000000, 1.000000]])
        Rs[22] = th.tensor([[-0.500000, 0.809017, -0.309017], [-0.809017, -0.309017, 0.500000],
                            [0.309017, 0.500000, 0.809017]])
        Rs[23] = th.tensor([[0.309017, 0.500000, -0.809017], [-0.500000, 0.809017, 0.309017],
                            [0.809017, 0.309017, 0.500000]])
        Rs[24] = th.tensor([[0.309017, -0.500000, -0.809017], [0.500000, 0.809017, -0.309017],
                            [0.809017, -0.309017, 0.500000]])
        Rs[25] = th.tensor([[0.000000, 0.000000, -1.000000], [-1.000000, 0.000000, 0.000000],
                            [0.000000, 1.000000, 0.000000]])
        Rs[26] = th.tensor([[-0.309017, -0.500000, -0.809017], [-0.500000, 0.809017, -0.309017],
                            [0.809017, 0.309017, -0.500000]])
        Rs[27] = th.tensor([[-0.809017, -0.309017, -0.500000], [0.309017, 0.500000, -0.809017],
                            [0.500000, -0.809017, -0.309017]])
        Rs[28] = th.tensor([[-0.809017, 0.309017, -0.500000], [0.309017, -0.500000, -0.809017],
                            [-0.500000, -0.809017, 0.309017]])
        Rs[29] = th.tensor([[-0.309017, 0.500000, -0.809017], [-0.500000, -0.809017, -0.309017],
                            [-0.809017, 0.309017, 0.500000]])
        Rs[30] = th.tensor([[0.809017, 0.309017, 0.500000], [-0.309017, -0.500000, 0.809017],
                            [0.500000, -0.809017, -0.309017]])
        Rs[31] = th.tensor([[0.809017, -0.309017, 0.500000], [-0.309017, 0.500000, 0.809017],
                            [-0.500000, -0.809017, 0.309017]])
        Rs[32] = th.tensor([[0.309017, -0.500000, 0.809017], [0.500000, 0.809017, 0.309017],
                            [-0.809017, 0.309017, 0.500000]])
        Rs[33] = th.tensor([[0.000000, 0.000000, 1.000000], [1.000000, 0.000000, 0.000000],
                            [0.000000, 1.000000, 0.000000]])
        Rs[34] = th.tensor([[0.309017, 0.500000, 0.809017], [0.500000, -0.809017, 0.309017],
                            [0.809017, 0.309017, -0.500000]])
        Rs[35] = th.tensor([[-0.309017, 0.500000, 0.809017], [0.500000, 0.809017, -0.309017],
                            [-0.809017, 0.309017, -0.500000]])
        Rs[36] = th.tensor([[0.500000, 0.809017, 0.309017], [0.809017, -0.309017, -0.500000],
                            [-0.309017, 0.500000, -0.809017]])
        Rs[37] = th.tensor([[1.000000, 0.000000, 0.000000], [0.000000, -1.000000, 0.000000],
                            [0.000000, 0.000000, -1.000000]])
        Rs[38] = th.tensor([[0.500000, -0.809017, 0.309017], [-0.809017, -0.309017, 0.500000],
                            [-0.309017, -0.500000, -0.809017]])
        Rs[39] = th.tensor([[-0.309017, -0.500000, 0.809017], [-0.500000, 0.809017, 0.309017],
                            [-0.809017, -0.309017, -0.500000]])
        Rs[40] = th.tensor([[-0.500000, 0.809017, 0.309017], [-0.809017, -0.309017, -0.500000],
                            [-0.309017, -0.500000, 0.809017]])
        Rs[41] = th.tensor([[0.500000, 0.809017, -0.309017], [-0.809017, 0.309017, -0.500000],
                            [-0.309017, 0.500000, 0.809017]])
        Rs[42] = th.tensor([[0.809017, -0.309017, -0.500000], [-0.309017, 0.500000, -0.809017],
                            [0.500000, 0.809017, 0.309017]])
        Rs[43] = th.tensor([[0.000000, -1.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                            [1.000000, 0.000000, 0.000000]])
        Rs[44] = th.tensor([[-0.809017, -0.309017, 0.500000], [-0.309017, -0.500000, -0.809017],
                            [0.500000, -0.809017, 0.309017]])
        Rs[45] = th.tensor([[0.809017, -0.309017, 0.500000], [0.309017, -0.500000, -0.809017],
                            [0.500000, 0.809017, -0.309017]])
        Rs[46] = th.tensor([[0.309017, -0.500000, 0.809017], [-0.500000, -0.809017, -0.309017],
                            [0.809017, -0.309017, -0.500000]])
        Rs[47] = th.tensor([[0.000000, 0.000000, 1.000000], [-1.000000, 0.000000, 0.000000],
                            [0.000000, -1.000000, 0.000000]])
        Rs[48] = th.tensor([[0.309017, 0.500000, 0.809017], [-0.500000, 0.809017, -0.309017],
                            [-0.809017, -0.309017, 0.500000]])
        Rs[49] = th.tensor([[0.809017, 0.309017, 0.500000], [0.309017, 0.500000, -0.809017],
                            [-0.500000, 0.809017, 0.309017]])
        Rs[50] = th.tensor([[-0.309017, 0.500000, -0.809017], [0.500000, 0.809017, 0.309017],
                            [0.809017, -0.309017, -0.500000]])
        Rs[51] = th.tensor([[0.000000, 0.000000, -1.000000], [1.000000, 0.000000, 0.000000],
                            [0.000000, -1.000000, 0.000000]])
        Rs[52] = th.tensor([[-0.309017, -0.500000, -0.809017], [0.500000, -0.809017, 0.309017],
                            [-0.809017, -0.309017, 0.500000]])
        Rs[53] = th.tensor([[-0.809017, -0.309017, -0.500000], [-0.309017, -0.500000, 0.809017],
                            [-0.500000, 0.809017, 0.309017]])
        Rs[54] = th.tensor([[-0.809017, 0.309017, -0.500000], [-0.309017, 0.500000, 0.809017],
                            [0.500000, 0.809017, -0.309017]])
        Rs[55] = th.tensor([[0.000000, -1.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                            [-1.000000, 0.000000, 0.000000]])
        Rs[56] = th.tensor([[-0.809017, -0.309017, 0.500000], [0.309017, 0.500000, 0.809017],
                            [-0.500000, 0.809017, -0.309017]])
        Rs[57] = th.tensor([[-0.500000, 0.809017, 0.309017], [0.809017, 0.309017, 0.500000],
                            [0.309017, 0.500000, -0.809017]])
        Rs[58] = th.tensor([[0.500000, 0.809017, -0.309017], [0.809017, -0.309017, 0.500000],
                            [0.309017, -0.500000, -0.809017]])
        Rs[59] = th.tensor([[0.809017, -0.309017, -0.500000], [0.309017, -0.500000, 0.809017],
                            [-0.500000, -0.809017, -0.309017]])

        est_radius = 10.0 * SYMA
        offset = th.tensor([1.0, 0.0, 0.0])
        offset = est_radius * offset / th.linalg.norm(offset)
        metasymm = ([th.arange(60)], [min(len(Rs), symopt.max_nsub if symopt else 6)])
    elif symmid.startswith('I'):
        ax2, ax3, ax5 = ipd.sym.axes('I', closest_to=[1, 2, 21]).values()
        pi = th.pi
        Rs = [th.eye(3)]
        if '2' in symmid:
            Rs.append(th.as_tensor(ipd.h.rot3(ax2, pi)))  # type: ignore
        if '3' in symmid:
            Rs.append(th.as_tensor(ipd.h.rot3(ax3, 2 / 3 * pi)))  # type: ignore
            Rs.append(th.as_tensor(ipd.h.rot3(ax3, -2 / 3 * pi)))  # type: ignore
        if '5' in symmid:
            Rs.append(th.as_tensor(ipd.h.rot3(ax5, 2 / 5 * pi)))  # type: ignore
            Rs.append(th.as_tensor(ipd.h.rot3(ax5, -2 / 5 * pi)))  # type: ignore
        Rs = th.stack(Rs)
        nsub = len(Rs)
        symmatrix = (th.arange(nsub)[:, None] - th.arange(nsub)[None, :]) % nsub
        metasymm = ([th.arange(nsub)], [min(nsub, symopt.max_nsub if symopt else min(3, nsub))])
        offset = None
    else:
        print("Unknown symmetry", symmid)
        assert False

    return symmatrix, Rs, metasymm, offset

def find_symmsub_pair(Ltot, Lasu, k, pseudo_cycle=False):
    """Creates a symmsub matrix.

    Parameters:
        Ltot (int, required): Total length of all residues

        Lasu (int, required): Length of asymmetric units

        k (int, required): Number of off diagonals to include in symmetrization

        pseudo_cycle (optional): whether to use pseudo-cyclic symmetrization (default: False)
    """
    assert Ltot % Lasu == 0
    nchunk = Ltot // Lasu

    N = 2*k + 1  # total number of diagonals being accessed
    symmsub = th.ones((nchunk, nchunk)) * -1
    C = 0  # a marker for blocks of the same category

    for i in range(N):  # i      = 0, 1,2, 3,4, 5,6...
        offset = int(((i+1) // 2) * (math.pow(-1, i)))  # offset = 0,-1,1,-2,2,-3,3...

        row = th.arange(nchunk)
        col = th.roll(row, offset)

        if offset < 0:
            row = row[:-abs(offset)]
            col = col[:-abs(offset)]
        elif offset > 0:
            row = row[abs(offset):]
            col = col[abs(offset):]
        else:  # i=0
            pass

        symmsub[row, col] = i

    if pseudo_cycle:
        # print('Doing pseudocycle')
        # Last --> First is same as First --> Second
        # First --> Last is same as Second --> First
        top_right = symmsub[1, 0]
        bottom_left = symmsub[0, 1]

        symmsub[0, -1] = top_right
        symmsub[-1, 0] = bottom_left

        # can't have any -1 left if pseudocycle
        assert th.sum(
            symmsub ==
            -1) == 0, 'Current symmsub not compatible with pseudocycle, increase symmsub_k to nrepeat-1'

    return symmsub.long()

def update_symm_Rs(xyz, Lasu, symmsub, allsymmRs, symopt):

    def dist_error_comp(R0, T0, xyz, fit_tscale):
        Ts = xyz  #[:,:,1]
        B = xyz.shape[0]

        Tcom = xyz[:, :Lasu].mean(dim=1, keepdim=True)  # LT center of mass for first ASU
        Tcorr = th.einsum('ij,brj->bri', R0, xyz[:, :Lasu] - Tcom) + Tcom + fit_tscale * T0[
            None, None, :]  # LT Rotated coordinates of first ASU by learned R0, then translated by learned T0

        # distance map loss
        Xsymm = th.einsum('sij,brj->bsri', allsymmRs[symmsub], Tcorr).reshape(B, -1, 3)
        Xtrue = Ts

        # compare dmaps via L1 loss
        delsx = Xsymm[:, :Lasu, None] - Xsymm[:, None, Lasu:]
        deltx = Xtrue[:, :Lasu, None] - Xtrue[:, None, Lasu:]
        dsymm = th.linalg.norm(delsx, dim=-1)
        dtrue = th.linalg.norm(deltx, dim=-1)
        loss1 = th.abs(dsymm - dtrue).mean()

        # clash loss
        Xsymmall = th.einsum('sij,brj->bsri', allsymmRs, Tcorr).reshape(B, -1, 3)
        delsxall = Xsymmall[:, :Lasu, None] - Xsymmall[:, None, Lasu:]
        dsymm = th.linalg.norm(delsxall, dim=-1)

        clash = th.clamp(symopt.fit_wclash - dsymm, min=0)
        loss2 = th.sum(clash) / Lasu

        return loss1, loss2  # 0.0

    def dist_error(R0, T0, xyz, fit_tscale, w_clash=10.0):
        l1, l2 = dist_error_comp(R0, T0, xyz, fit_tscale)
        return l1 + w_clash*l2

    def Q2R(Q):
        Qs = th.cat((th.ones((1), device=Q.device), Q), dim=-1)
        Qs = normQ(Qs)
        return Qs2Rs(Qs[None, :]).squeeze(0)

    B = xyz.shape[0]
    L = xyz.shape[1]
    natoms = xyz.shape[2]

    if symopt.fit:
        with th.enable_grad():
            T0 = th.zeros(3, device=xyz.device).requires_grad_(True)
            Q0 = th.zeros(3, device=xyz.device).requires_grad_(True)
            lbfgs = th.optim.LBFGS([T0, Q0], history_size=10, max_iter=4, line_search_fn="strong_wolfe")

            def closure():
                lbfgs.zero_grad()
                i = 1 if xyz.shape[2] > 1 else 0
                loss = dist_error(Q2R(Q0), T0, xyz[:, :, i], symopt.fit_tscale)
                loss.backward()  #retain_graph=True)
                return loss

            for e in range(4):
                loss = lbfgs.step(closure)

            Tcom = xyz[:, :Lasu].mean(dim=1, keepdim=True).detach()
            Q0 = Q0.detach()
            T0 = T0.detach()
            xyz = th.einsum('ij,braj->brai', Q2R(Q0),
                            xyz[:, :Lasu] - Tcom) + Tcom + symopt.fit_tscale * T0[None, None, :]

    xyz = th.einsum('sij,braj->bsrai', allsymmRs[symmsub], xyz[:, :Lasu])
    xyz = xyz.reshape(B, -1, natoms, 3)  # (B,S,L,3,3) or (B,LASU*S,natoms,3)

    # if symopt.fit:
    # ipd.pdb.dumppdb(f'{symopt.tag}_fit1.pdb', xyz[:,:,:3,:3].reshape(L//Lasu,Lasu,3,3))

    return xyz

def update_symm_subs_track_module(xyz, pair, symmids, symmsub, allsymmRs, metasymm, symopt):
    B, Ls = xyz.shape[0:2]
    Osub = symmsub.shape[0]
    L = Ls // Osub

    com = xyz[:, :L, 1].mean(dim=-2)
    rcoms = th.einsum('sij,bj->si', allsymmRs, com)
    subsymms, nneighs = metasymm
    symmsub_new = []
    for i in range(len(subsymms)):
        drcoms = th.linalg.norm(rcoms[0, :] - rcoms[subsymms[i], :], dim=-1)
        _, subs_i = th.topk(drcoms, nneighs[i], largest=False)
        subs_i, _ = th.sort(subsymms[i][subs_i])
        symmsub_new.append(subs_i)

    symmsub_new = th.cat(symmsub_new)
    # print('symm sub new', symmsub_new)

    s_old = symmids[symmsub[:, None], symmsub[None, :]]
    s_new = symmids[symmsub_new[:, None], symmsub_new[None, :]]

    # remap old->new
    # a) find highest-magnitude patches
    pairsub = dict()
    pairmag = dict()
    for i in range(Osub):
        for j in range(Osub):
            idx_old = s_old[i, j].item()
            sub_ij = pair[:, i * L:(i+1) * L, j * L:(j+1) * L, :].clone()
            mag_ij = th.max(sub_ij.flatten())  #th.norm(sub_ij.flatten())
            if idx_old not in pairsub or mag_ij > pairmag[idx_old]:
                pairmag[idx_old] = mag_ij
                pairsub[idx_old] = (i, j)  #sub_ij

    # b) reindex
    idx = th.zeros((Osub * L, Osub * L), dtype=th.long, device=pair.device)
    idx = (th.arange(Osub * L, device=pair.device)[:, None] * Osub * L +
           th.arange(Osub * L, device=pair.device)[None, :])
    for i in range(Osub):
        for j in range(Osub):
            idx_new = s_new[i, j].item()
            if idx_new in pairsub:
                inew, jnew = pairsub[idx_new]
                idx[i * L:(i+1) * L, j * L:(j+1) * L] = (Osub * L * th.arange(inew * L,
                                                                              (inew+1) * L)[:, None] +
                                                         th.arange(jnew * L, (jnew+1) * L)[None, :])

    pair = pair.reshape(1, -1, pair.shape[-1])[:, idx.flatten(), :].view(1, Osub * L, Osub * L, pair.shape[-1])

    if symmsub is not None and symmsub.shape[0] > 1:
        xyz = update_symm_Rs(xyz, L, symmsub_new, allsymmRs, symopt)

    return xyz, pair, symmsub_new
