import math

import numpy as np

import ipd

torch = ipd.lazyimport('torch')

SYMA = 1.0

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
        assert len(allatoms[allatoms['chnid'] == c]) == L_chain_main, 'PDB file does not contain symmetric units!'
        xyz_stack.append(allatoms[allatoms['chnid'] == c]['X'])
    # center the complex
    COM = np.mean(xyz_stack, axis=(0, 1))
    return torch.tensor(xyz_stack - COM)

def generate_ASU_xforms(pdb):
    """Takes in [N,L,3] Ca xyz stack to map transforms of ASU, should return
    homogenous transforms for each subunit in the ASU We use this to generate
    the ASU in the first place."""
    xyz_stack = get_coords_stack(open(pdb))
    xforms = []
    for n in range(xyz_stack.shape[0]):
        _, _, X = ipd.h.rmsfit(xyz_stack[n], xyz_stack[0])  # rms, fitxyz, X  # type: ignore
        xforms.append(X)
    return xforms

def normQ(Q):
    """Normalize a quaternions."""
    return Q / torch.linalg.norm(Q, keepdim=True, dim=-1)

def Rs2Qs(Rs):
    Qs = torch.zeros((*Rs.shape[:-2], 4), device=Rs.device)

    Qs[..., 0] = 1.0 + Rs[..., 0, 0] + Rs[..., 1, 1] + Rs[..., 2, 2]
    Qs[..., 1] = 1.0 + Rs[..., 0, 0] - Rs[..., 1, 1] - Rs[..., 2, 2]
    Qs[..., 2] = 1.0 - Rs[..., 0, 0] + Rs[..., 1, 1] - Rs[..., 2, 2]
    Qs[..., 3] = 1.0 - Rs[..., 0, 0] - Rs[..., 1, 1] + Rs[..., 2, 2]
    Qs[Qs < 0.0] = 0.0
    Qs = torch.sqrt(Qs) / 2.0
    Qs[..., 1] *= torch.sign(Rs[..., 2, 1] - Rs[..., 1, 2])
    Qs[..., 2] *= torch.sign(Rs[..., 0, 2] - Rs[..., 2, 0])
    Qs[..., 3] *= torch.sign(Rs[..., 1, 0] - Rs[..., 0, 1])

    return Qs

def Qs2Rs(Qs):
    Rs = torch.zeros((*Qs.shape[:-1], 3, 3), device=Qs.device)

    Rs[..., 0, 0] = Qs[..., 0] * Qs[..., 0] + Qs[..., 1] * Qs[..., 1] - Qs[..., 2] * Qs[..., 2] - Qs[..., 3] * Qs[..., 3]
    Rs[..., 0, 1] = 2 * Qs[..., 1] * Qs[..., 2] - 2 * Qs[..., 0] * Qs[..., 3]
    Rs[..., 0, 2] = 2 * Qs[..., 1] * Qs[..., 3] + 2 * Qs[..., 0] * Qs[..., 2]
    Rs[..., 1, 0] = 2 * Qs[..., 1] * Qs[..., 2] + 2 * Qs[..., 0] * Qs[..., 3]
    Rs[..., 1, 1] = Qs[..., 0] * Qs[..., 0] - Qs[..., 1] * Qs[..., 1] + Qs[..., 2] * Qs[..., 2] - Qs[..., 3] * Qs[..., 3]
    Rs[..., 1, 2] = 2 * Qs[..., 2] * Qs[..., 3] - 2 * Qs[..., 0] * Qs[..., 1]
    Rs[..., 2, 0] = 2 * Qs[..., 1] * Qs[..., 3] - 2 * Qs[..., 0] * Qs[..., 2]
    Rs[..., 2, 1] = 2 * Qs[..., 2] * Qs[..., 3] + 2 * Qs[..., 0] * Qs[..., 1]
    Rs[..., 2, 2] = Qs[..., 0] * Qs[..., 0] - Qs[..., 1] * Qs[..., 1] - Qs[..., 2] * Qs[..., 2] + Qs[..., 3] * Qs[..., 3]

    return Rs

def generateC(angs, eps=1e-4):
    L = angs.shape[0]
    Rs = torch.eye(3, device=angs.device).repeat(L, 1, 1)
    Rs[:, 0, 0] = torch.cos(angs)
    Rs[:, 0, 1] = -torch.sin(angs)
    Rs[:, 1, 0] = torch.sin(angs)
    Rs[:, 1, 1] = torch.cos(angs)
    return Rs

def generateD(angs, eps=1e-4):
    L = angs.shape[0]
    Rs = torch.eye(3, device=angs.device).repeat(2 * L, 1, 1)
    Rs[:L, 0, 0] = torch.cos(angs)
    Rs[:L, 0, 1] = -torch.sin(angs)
    Rs[:L, 1, 0] = torch.sin(angs)
    Rs[:L, 1, 1] = torch.cos(angs)
    Rx = torch.tensor([[1, 0, 0], [0, -1., 0], [0, 0, -1]], device=angs.device)
    Rs[L:] = torch.einsum('ij,bjk->bik', Rx, Rs[:L])
    return Rs

def find_symm_subs(xyz, Rs, metasymm):
    com = xyz[:, :, 1].mean(dim=-2)
    rcoms = torch.einsum('sij,bj->si', Rs, com)

    subsymms, nneighs = metasymm

    subs = []
    for i in range(len(subsymms)):
        drcoms = torch.linalg.norm(rcoms[0, :] - rcoms[subsymms[i], :], dim=-1)
        _, subs_i = torch.topk(drcoms, nneighs[i], largest=False)
        subs_i, _ = torch.sort(subsymms[i][subs_i])
        subs.append(subs_i)

    subs = torch.cat(subs)
    xyz_new = torch.einsum('sij,braj->bsrai', Rs[subs], xyz).reshape(xyz.shape[0], -1, xyz.shape[2], 3)
    return xyz_new, subs

def update_symm_subs(xyz, subs, Rs, metasymm):
    xyz_new = torch.einsum('sij,braj->bsrai', Rs[subs], xyz).reshape(xyz.shape[0], -1, xyz.shape[2], 3)
    return xyz_new

def get_symm_map(subs, O):
    symmmask = torch.zeros(O, dtype=torch.long)
    symmmask[subs] = torch.arange(1, subs.shape[0] + 1)
    return symmmask

def rotation_from_matrix(R, eps=1e-4):
    w, W = torch.linalg.eig(R.T)
    i = torch.where(abs(torch.real(w) - 1.0) < eps)[0]
    if (len(i) == 0):
        i = torch.tensor([0])
        print('rotation_from_matrix w', torch.real(w))
        print('rotation_from_matrix R.T', R.T)
    axis = torch.real(W[:, i[-1]]).squeeze()

    cosa = (torch.trace(R) - 1.0) / 2.0
    if abs(axis[2]) > eps:
        sina = (R[1, 0] + (cosa-1.0) * axis[0] * axis[1]) / axis[2]
    elif abs(axis[1]) > eps:
        sina = (R[0, 2] + (cosa-1.0) * axis[0] * axis[2]) / axis[1]
    else:
        sina = (R[2, 1] + (cosa-1.0) * axis[1] * axis[2]) / axis[0]
    angle = torch.atan2(sina, cosa)

    return angle, axis

def kabsch(pred, true):
    def rmsd(V, W, eps=1e-4):
        L = V.shape[0]
        return torch.sqrt(torch.sum((V-W) * (V-W)) / L + eps)

    def centroid(X):
        return X.mean(dim=-2, keepdim=True)

    cP = centroid(pred)
    cT = centroid(true)
    pred = pred - cP
    true = true - cT
    C = torch.matmul(pred.permute(1, 0), true)
    V, S, W = torch.svd(C)
    d = torch.ones([3, 3], device=pred.device)
    d[:, -1] = torch.sign(torch.det(V) * torch.det(W))
    U = torch.matmul(d * V, W.permute(1, 0))  # (IB, 3, 3)
    rpred = torch.matmul(pred, U)  # (IB, L*3, 3)
    rms = rmsd(rpred, true)
    return rms, U, cP, cT

# do lines X0->X and Y0->Y intersect?
def intersect(X0, X, Y0, Y, eps=0.1):
    mtx = torch.cat((torch.stack((X0, X0 + X, Y0, Y0 + Y)), torch.ones((4, 1))), axis=1)
    det = torch.linalg.det(mtx)
    return (torch.abs(det) <= eps)

def get_angle(X, Y):
    angle = torch.acos(torch.clamp(torch.sum(X * Y), -1., 1.))
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
        offset0 = torch.linalg.norm(xyz[i, :L, 1] - xyz[0, :L, 1], dim=-1)
        if (torch.mean(offset0) > 1e-4):
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
        nfold = 2 * np.pi / torch.abs(angle)
        # a) ensure integer # of subunits per rotation
        if (torch.abs(nfold - torch.round(nfold)) > nfold_cut):
            #print ('nfold fail',nfold)
            continue
        nfold = torch.round(nfold).long()
        # b) ensure rotation only (no translation)
        delCOM = torch.mean(xyz_i, dim=-2) - torch.mean(xyz_j, dim=-2)
        trans_dot_symaxis = nfold * torch.abs(torch.dot(delCOM, axis))
        if (trans_dot_symaxis > trans_cut):
            #print ('trans fail',trans_dot_symaxis)
            continue

        # 3) get a point on the symm axis from CoMs and angle
        cIJ = torch.sign(angle) * (cJ - cI).squeeze(0)
        dIJ = torch.linalg.norm(cIJ)
        p_mid = (cI + cJ).squeeze(0) / 2
        u = cIJ / dIJ  # unit vector in plane of circle
        v = torch.cross(axis, u)  # unit vector from sym axis to p_mid
        r = dIJ / (2 * torch.sin(angle / 2))
        d = torch.sqrt(r*r - dIJ*dIJ/4)  # distance from mid-chord to center
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
        if (symmaxes[1][0] == 2 and torch.abs(angle - np.pi / 2) < angle_cut):
            symmgroup = 'D%d' % (symmaxes[0][0])
        else:
            # polyhedral rules:
            #   3-Fold + 2-fold intersecting at acos(-1/sqrt(3)) -> T
            angle_tgt = np.arccos(-1 / np.sqrt(3))
            if (symmaxes[0][0] == 3 and symmaxes[1][0] == 2 and torch.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'T'

            #   3-Fold + 2-fold intersecting at asin(1/sqrt(3)) -> O
            angle_tgt = np.arcsin(1 / np.sqrt(3))
            if (symmaxes[0][0] == 3 and symmaxes[1][0] == 2 and torch.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'O'

            #   4-Fold + 3-fold intersecting at acos(1/sqrt(3)) -> O
            angle_tgt = np.arccos(1 / np.sqrt(3))
            if (symmaxes[0][0] == 4 and symmaxes[1][0] == 3 and torch.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'O'

            #   3-Fold + 2-fold intersecting at 0.5*acos(sqrt(5)/3) -> I
            angle_tgt = 0.5 * np.arccos(np.sqrt(5) / 3)
            if (symmaxes[0][0] == 3 and symmaxes[1][0] == 2 and torch.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'I'

            #   5-Fold + 2-fold intersecting at 0.5*acos(1/sqrt(5)) -> I
            angle_tgt = 0.5 * np.arccos(1 / np.sqrt(5))
            if (symmaxes[0][0] == 5 and symmaxes[1][0] == 2 and torch.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'I'

            #   5-Fold + 3-fold intersecting at 0.5*acos((4*sqrt(5)-5)/15) -> I
            angle_tgt = 0.5 * np.arccos((4 * np.sqrt(5) - 5) / 15)
            if (symmaxes[0][0] == 5 and symmaxes[1][0] == 3 and torch.abs(angle - angle_tgt) < angle_cut):
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
        symmatrix = (torch.arange(nsub)[:, None] - torch.arange(nsub)[None, :]) % nsub
        angles = torch.linspace(0, 2 * np.pi, nsub + 1)[:nsub]
        Rs = generateC(angles)

        metasymm = ([torch.arange(nsub)], [min(nsub, symopt.max_nsub if symopt else min(3, nsub))])

        if (nsub == 1):
            D = 0.0
        else:
            est_radius = 2.0 * SYMA
            theta = 2.0 * np.pi / nsub
            D = est_radius / np.sin(theta / 2)

        offset = torch.tensor([float(D), 0.0, 0.0])
    elif (symmid[0] == 'D'):
        nsub = int(symmid[1:])
        cblk = (torch.arange(nsub)[:, None] - torch.arange(nsub)[None, :]) % nsub
        symmatrix = torch.zeros((2 * nsub, 2 * nsub), dtype=torch.long)
        symmatrix[:nsub, :nsub] = cblk
        symmatrix[:nsub, nsub:] = cblk + nsub
        symmatrix[nsub:, :nsub] = cblk + nsub
        symmatrix[nsub:, nsub:] = cblk
        angles = torch.linspace(0, 2 * np.pi, nsub + 1)[:nsub]
        Rs = generateD(angles)

        metasymm = ([torch.arange(nsub),
                     nsub + torch.arange(nsub)], [min(nsub, symopt.max_nsub if symopt else min(3, nsub)), 2])
        #metasymm = (
        #    [torch.arange(2*nsub)],
        #    [min(2*nsub,5)]
        #)

        est_radius = 2.0 * SYMA
        theta1 = 2.0 * np.pi / nsub
        theta2 = np.pi
        D1 = est_radius / np.sin(theta1 / 2)
        D2 = est_radius / np.sin(theta2 / 2)
        offset = torch.tensor([float(D1), float(D2), 0.0])
    elif (symmid == 'T'):
        symmatrix = torch.tensor([[0, 1, 2, 3, 8, 11, 9, 10, 4, 6, 7, 5], [1, 0, 3, 2, 9, 10, 8, 11, 5, 7, 6, 4],
                                  [2, 3, 0, 1, 10, 9, 11, 8, 6, 4, 5, 7], [3, 2, 1, 0, 11, 8, 10, 9, 7, 5, 4, 6],
                                  [4, 6, 7, 5, 0, 1, 2, 3, 8, 11, 9, 10], [5, 7, 6, 4, 1, 0, 3, 2, 9, 10, 8, 11],
                                  [6, 4, 5, 7, 2, 3, 0, 1, 10, 9, 11, 8], [7, 5, 4, 6, 3, 2, 1, 0, 11, 8, 10, 9],
                                  [8, 11, 9, 10, 4, 6, 7, 5, 0, 1, 2, 3], [9, 10, 8, 11, 5, 7, 6, 4, 1, 0, 3, 2],
                                  [10, 9, 11, 8, 6, 4, 5, 7, 2, 3, 0, 1], [11, 8, 10, 9, 7, 5, 4, 6, 3, 2, 1, 0]])
        Rs = torch.zeros(12, 3, 3)
        Rs[0] = torch.tensor([[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000],
                              [0.000000, 0.000000, 1.000000]])
        Rs[1] = torch.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, -1.000000, 0.000000],
                              [0.000000, 0.000000, 1.000000]])
        Rs[2] = torch.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000],
                              [0.000000, 0.000000, -1.000000]])
        Rs[3] = torch.tensor([[1.000000, 0.000000, 0.000000], [0.000000, -1.000000, 0.000000],
                              [0.000000, 0.000000, -1.000000]])
        Rs[4] = torch.tensor([[0.000000, 0.000000, 1.000000], [1.000000, 0.000000, 0.000000],
                              [0.000000, 1.000000, 0.000000]])
        Rs[5] = torch.tensor([[0.000000, 0.000000, 1.000000], [-1.000000, 0.000000, 0.000000],
                              [0.000000, -1.000000, 0.000000]])
        Rs[6] = torch.tensor([[0.000000, 0.000000, -1.000000], [-1.000000, 0.000000, 0.000000],
                              [0.000000, 1.000000, 0.000000]])
        Rs[7] = torch.tensor([[0.000000, 0.000000, -1.000000], [1.000000, 0.000000, 0.000000],
                              [0.000000, -1.000000, 0.000000]])
        Rs[8] = torch.tensor([[0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                              [1.000000, 0.000000, 0.000000]])
        Rs[9] = torch.tensor([[0.000000, -1.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                              [-1.000000, 0.000000, 0.000000]])
        Rs[10] = torch.tensor([[0.000000, 1.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                               [-1.000000, 0.000000, 0.000000]])
        Rs[11] = torch.tensor([[0.000000, -1.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                               [1.000000, 0.000000, 0.000000]])

        est_radius = 4.0 * SYMA
        offset = torch.tensor([1.0, 0.0, 0.0])
        offset = est_radius * offset / torch.linalg.norm(offset)
        metasymm = ([torch.arange(12)], [min(len(Rs), symopt.max_nsub if symopt else 6)])

    elif (symmid == 'O'):
        symmatrix = torch.tensor([[0, 1, 2, 3, 8, 11, 9, 10, 4, 6, 7, 5, 12, 13, 15, 14, 19, 17, 18, 16, 22, 21, 20, 23],
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
        Rs = torch.zeros(24, 3, 3)
        Rs[0] = torch.tensor([[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000],
                              [0.000000, 0.000000, 1.000000]])
        Rs[1] = torch.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, -1.000000, 0.000000],
                              [0.000000, 0.000000, 1.000000]])
        Rs[2] = torch.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000],
                              [0.000000, 0.000000, -1.000000]])
        Rs[3] = torch.tensor([[1.000000, 0.000000, 0.000000], [0.000000, -1.000000, 0.000000],
                              [0.000000, 0.000000, -1.000000]])
        Rs[4] = torch.tensor([[0.000000, 0.000000, 1.000000], [1.000000, 0.000000, 0.000000],
                              [0.000000, 1.000000, 0.000000]])
        Rs[5] = torch.tensor([[0.000000, 0.000000, 1.000000], [-1.000000, 0.000000, 0.000000],
                              [0.000000, -1.000000, 0.000000]])
        Rs[6] = torch.tensor([[0.000000, 0.000000, -1.000000], [-1.000000, 0.000000, 0.000000],
                              [0.000000, 1.000000, 0.000000]])
        Rs[7] = torch.tensor([[0.000000, 0.000000, -1.000000], [1.000000, 0.000000, 0.000000],
                              [0.000000, -1.000000, 0.000000]])
        Rs[8] = torch.tensor([[0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                              [1.000000, 0.000000, 0.000000]])
        Rs[9] = torch.tensor([[0.000000, -1.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                              [-1.000000, 0.000000, 0.000000]])
        Rs[10] = torch.tensor([[0.000000, 1.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                               [-1.000000, 0.000000, 0.000000]])
        Rs[11] = torch.tensor([[0.000000, -1.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                               [1.000000, 0.000000, 0.000000]])
        Rs[12] = torch.tensor([[0.000000, 1.000000, 0.000000], [1.000000, 0.000000, 0.000000],
                               [0.000000, 0.000000, -1.000000]])
        Rs[13] = torch.tensor([[0.000000, -1.000000, 0.000000], [-1.000000, 0.000000, 0.000000],
                               [0.000000, 0.000000, -1.000000]])
        Rs[14] = torch.tensor([[0.000000, 1.000000, 0.000000], [-1.000000, 0.000000, 0.000000],
                               [0.000000, 0.000000, 1.000000]])
        Rs[15] = torch.tensor([[0.000000, -1.000000, 0.000000], [1.000000, 0.000000, 0.000000],
                               [0.000000, 0.000000, 1.000000]])
        Rs[16] = torch.tensor([[1.000000, 0.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                               [0.000000, -1.000000, 0.000000]])
        Rs[17] = torch.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                               [0.000000, 1.000000, 0.000000]])
        Rs[18] = torch.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                               [0.000000, -1.000000, 0.000000]])
        Rs[19] = torch.tensor([[1.000000, 0.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                               [0.000000, 1.000000, 0.000000]])
        Rs[20] = torch.tensor([[0.000000, 0.000000, 1.000000], [0.000000, 1.000000, 0.000000],
                               [-1.000000, 0.000000, 0.000000]])
        Rs[21] = torch.tensor([[0.000000, 0.000000, 1.000000], [0.000000, -1.000000, 0.000000],
                               [1.000000, 0.000000, 0.000000]])
        Rs[22] = torch.tensor([[0.000000, 0.000000, -1.000000], [0.000000, 1.000000, 0.000000],
                               [1.000000, 0.000000, 0.000000]])
        Rs[23] = torch.tensor([[0.000000, 0.000000, -1.000000], [0.000000, -1.000000, 0.000000],
                               [-1.000000, 0.000000, 0.000000]])

        est_radius = 6.0 * SYMA
        offset = torch.tensor([1.0, 0.0, 0.0])
        offset = est_radius * offset / torch.linalg.norm(offset)
        metasymm = ([torch.arange(24)], [min(len(Rs), symopt.max_nsub if symopt else 6)])
    elif (symmid == 'I'):
        symmatrix = torch.tensor([[
            0, 4, 3, 2, 1, 5, 33, 49, 41, 22, 10, 27, 51, 59, 38, 15, 16, 17, 18, 19, 40, 21, 9, 32, 48, 55, 39, 11, 28,
            52, 45, 42, 23, 6, 34, 50, 58, 37, 14, 26, 20, 8, 31, 47, 44, 30, 46, 43, 24, 7, 35, 12, 29, 53, 56, 25, 54,
            57, 36, 13
        ],
                                  [
                                      1, 0, 4, 3, 2, 6, 34, 45, 42, 23, 11, 28, 52, 55, 39, 16, 17, 18, 19, 15, 41, 22, 5,
                                      33, 49, 56, 35, 12, 29, 53, 46, 43, 24, 7, 30, 51, 59, 38, 10, 27, 21, 9, 32, 48, 40,
                                      31, 47, 44, 20, 8, 36, 13, 25, 54, 57, 26, 50, 58, 37, 14
                                  ],
                                  [
                                      2, 1, 0, 4, 3, 7, 30, 46, 43, 24, 12, 29, 53, 56, 35, 17, 18, 19, 15, 16, 42, 23, 6,
                                      34, 45, 57, 36, 13, 25, 54, 47, 44, 20, 8, 31, 52, 55, 39, 11, 28, 22, 5, 33, 49, 41,
                                      32, 48, 40, 21, 9, 37, 14, 26, 50, 58, 27, 51, 59, 38, 10
                                  ],
                                  [
                                      3, 2, 1, 0, 4, 8, 31, 47, 44, 20, 13, 25, 54, 57, 36, 18, 19, 15, 16, 17, 43, 24, 7,
                                      30, 46, 58, 37, 14, 26, 50, 48, 40, 21, 9, 32, 53, 56, 35, 12, 29, 23, 6, 34, 45, 42,
                                      33, 49, 41, 22, 5, 38, 10, 27, 51, 59, 28, 52, 55, 39, 11
                                  ],
                                  [
                                      4, 3, 2, 1, 0, 9, 32, 48, 40, 21, 14, 26, 50, 58, 37, 19, 15, 16, 17, 18, 44, 20, 8,
                                      31, 47, 59, 38, 10, 27, 51, 49, 41, 22, 5, 33, 54, 57, 36, 13, 25, 24, 7, 30, 46, 43,
                                      34, 45, 42, 23, 6, 39, 11, 28, 52, 55, 29, 53, 56, 35, 12
                                  ],
                                  [
                                      5, 33, 49, 41, 22, 0, 4, 3, 2, 1, 15, 16, 17, 18, 19, 10, 27, 51, 59, 38, 45, 42, 23,
                                      6, 34, 50, 58, 37, 14, 26, 40, 21, 9, 32, 48, 55, 39, 11, 28, 52, 25, 54, 57, 36, 13,
                                      35, 12, 29, 53, 56, 30, 46, 43, 24, 7, 20, 8, 31, 47, 44
                                  ],
                                  [
                                      6, 34, 45, 42, 23, 1, 0, 4, 3, 2, 16, 17, 18, 19, 15, 11, 28, 52, 55, 39, 46, 43, 24,
                                      7, 30, 51, 59, 38, 10, 27, 41, 22, 5, 33, 49, 56, 35, 12, 29, 53, 26, 50, 58, 37, 14,
                                      36, 13, 25, 54, 57, 31, 47, 44, 20, 8, 21, 9, 32, 48, 40
                                  ],
                                  [
                                      7, 30, 46, 43, 24, 2, 1, 0, 4, 3, 17, 18, 19, 15, 16, 12, 29, 53, 56, 35, 47, 44, 20,
                                      8, 31, 52, 55, 39, 11, 28, 42, 23, 6, 34, 45, 57, 36, 13, 25, 54, 27, 51, 59, 38, 10,
                                      37, 14, 26, 50, 58, 32, 48, 40, 21, 9, 22, 5, 33, 49, 41
                                  ],
                                  [
                                      8, 31, 47, 44, 20, 3, 2, 1, 0, 4, 18, 19, 15, 16, 17, 13, 25, 54, 57, 36, 48, 40, 21,
                                      9, 32, 53, 56, 35, 12, 29, 43, 24, 7, 30, 46, 58, 37, 14, 26, 50, 28, 52, 55, 39, 11,
                                      38, 10, 27, 51, 59, 33, 49, 41, 22, 5, 23, 6, 34, 45, 42
                                  ],
                                  [
                                      9, 32, 48, 40, 21, 4, 3, 2, 1, 0, 19, 15, 16, 17, 18, 14, 26, 50, 58, 37, 49, 41, 22,
                                      5, 33, 54, 57, 36, 13, 25, 44, 20, 8, 31, 47, 59, 38, 10, 27, 51, 29, 53, 56, 35, 12,
                                      39, 11, 28, 52, 55, 34, 45, 42, 23, 6, 24, 7, 30, 46, 43
                                  ],
                                  [
                                      10, 27, 51, 59, 38, 15, 16, 17, 18, 19, 0, 4, 3, 2, 1, 5, 33, 49, 41, 22, 50, 58, 37,
                                      14, 26, 45, 42, 23, 6, 34, 55, 39, 11, 28, 52, 40, 21, 9, 32, 48, 30, 46, 43, 24, 7,
                                      20, 8, 31, 47, 44, 25, 54, 57, 36, 13, 35, 12, 29, 53, 56
                                  ],
                                  [
                                      11, 28, 52, 55, 39, 16, 17, 18, 19, 15, 1, 0, 4, 3, 2, 6, 34, 45, 42, 23, 51, 59, 38,
                                      10, 27, 46, 43, 24, 7, 30, 56, 35, 12, 29, 53, 41, 22, 5, 33, 49, 31, 47, 44, 20, 8,
                                      21, 9, 32, 48, 40, 26, 50, 58, 37, 14, 36, 13, 25, 54, 57
                                  ],
                                  [
                                      12, 29, 53, 56, 35, 17, 18, 19, 15, 16, 2, 1, 0, 4, 3, 7, 30, 46, 43, 24, 52, 55, 39,
                                      11, 28, 47, 44, 20, 8, 31, 57, 36, 13, 25, 54, 42, 23, 6, 34, 45, 32, 48, 40, 21, 9,
                                      22, 5, 33, 49, 41, 27, 51, 59, 38, 10, 37, 14, 26, 50, 58
                                  ],
                                  [
                                      13, 25, 54, 57, 36, 18, 19, 15, 16, 17, 3, 2, 1, 0, 4, 8, 31, 47, 44, 20, 53, 56, 35,
                                      12, 29, 48, 40, 21, 9, 32, 58, 37, 14, 26, 50, 43, 24, 7, 30, 46, 33, 49, 41, 22, 5,
                                      23, 6, 34, 45, 42, 28, 52, 55, 39, 11, 38, 10, 27, 51, 59
                                  ],
                                  [
                                      14, 26, 50, 58, 37, 19, 15, 16, 17, 18, 4, 3, 2, 1, 0, 9, 32, 48, 40, 21, 54, 57, 36,
                                      13, 25, 49, 41, 22, 5, 33, 59, 38, 10, 27, 51, 44, 20, 8, 31, 47, 34, 45, 42, 23, 6,
                                      24, 7, 30, 46, 43, 29, 53, 56, 35, 12, 39, 11, 28, 52, 55
                                  ],
                                  [
                                      15, 16, 17, 18, 19, 10, 27, 51, 59, 38, 5, 33, 49, 41, 22, 0, 4, 3, 2, 1, 55, 39, 11,
                                      28, 52, 40, 21, 9, 32, 48, 50, 58, 37, 14, 26, 45, 42, 23, 6, 34, 35, 12, 29, 53, 56,
                                      25, 54, 57, 36, 13, 20, 8, 31, 47, 44, 30, 46, 43, 24, 7
                                  ],
                                  [
                                      16, 17, 18, 19, 15, 11, 28, 52, 55, 39, 6, 34, 45, 42, 23, 1, 0, 4, 3, 2, 56, 35, 12,
                                      29, 53, 41, 22, 5, 33, 49, 51, 59, 38, 10, 27, 46, 43, 24, 7, 30, 36, 13, 25, 54, 57,
                                      26, 50, 58, 37, 14, 21, 9, 32, 48, 40, 31, 47, 44, 20, 8
                                  ],
                                  [
                                      17, 18, 19, 15, 16, 12, 29, 53, 56, 35, 7, 30, 46, 43, 24, 2, 1, 0, 4, 3, 57, 36, 13,
                                      25, 54, 42, 23, 6, 34, 45, 52, 55, 39, 11, 28, 47, 44, 20, 8, 31, 37, 14, 26, 50, 58,
                                      27, 51, 59, 38, 10, 22, 5, 33, 49, 41, 32, 48, 40, 21, 9
                                  ],
                                  [
                                      18, 19, 15, 16, 17, 13, 25, 54, 57, 36, 8, 31, 47, 44, 20, 3, 2, 1, 0, 4, 58, 37, 14,
                                      26, 50, 43, 24, 7, 30, 46, 53, 56, 35, 12, 29, 48, 40, 21, 9, 32, 38, 10, 27, 51, 59,
                                      28, 52, 55, 39, 11, 23, 6, 34, 45, 42, 33, 49, 41, 22, 5
                                  ],
                                  [
                                      19, 15, 16, 17, 18, 14, 26, 50, 58, 37, 9, 32, 48, 40, 21, 4, 3, 2, 1, 0, 59, 38, 10,
                                      27, 51, 44, 20, 8, 31, 47, 54, 57, 36, 13, 25, 49, 41, 22, 5, 33, 39, 11, 28, 52, 55,
                                      29, 53, 56, 35, 12, 24, 7, 30, 46, 43, 34, 45, 42, 23, 6
                                  ],
                                  [
                                      20, 8, 31, 47, 44, 30, 46, 43, 24, 7, 35, 12, 29, 53, 56, 25, 54, 57, 36, 13, 0, 4,
                                      3, 2, 1, 5, 33, 49, 41, 22, 10, 27, 51, 59, 38, 15, 16, 17, 18, 19, 40, 21, 9, 32,
                                      48, 55, 39, 11, 28, 52, 45, 42, 23, 6, 34, 50, 58, 37, 14, 26
                                  ],
                                  [
                                      21, 9, 32, 48, 40, 31, 47, 44, 20, 8, 36, 13, 25, 54, 57, 26, 50, 58, 37, 14, 1, 0,
                                      4, 3, 2, 6, 34, 45, 42, 23, 11, 28, 52, 55, 39, 16, 17, 18, 19, 15, 41, 22, 5, 33,
                                      49, 56, 35, 12, 29, 53, 46, 43, 24, 7, 30, 51, 59, 38, 10, 27
                                  ],
                                  [
                                      22, 5, 33, 49, 41, 32, 48, 40, 21, 9, 37, 14, 26, 50, 58, 27, 51, 59, 38, 10, 2, 1,
                                      0, 4, 3, 7, 30, 46, 43, 24, 12, 29, 53, 56, 35, 17, 18, 19, 15, 16, 42, 23, 6, 34,
                                      45, 57, 36, 13, 25, 54, 47, 44, 20, 8, 31, 52, 55, 39, 11, 28
                                  ],
                                  [
                                      23, 6, 34, 45, 42, 33, 49, 41, 22, 5, 38, 10, 27, 51, 59, 28, 52, 55, 39, 11, 3, 2,
                                      1, 0, 4, 8, 31, 47, 44, 20, 13, 25, 54, 57, 36, 18, 19, 15, 16, 17, 43, 24, 7, 30,
                                      46, 58, 37, 14, 26, 50, 48, 40, 21, 9, 32, 53, 56, 35, 12, 29
                                  ],
                                  [
                                      24, 7, 30, 46, 43, 34, 45, 42, 23, 6, 39, 11, 28, 52, 55, 29, 53, 56, 35, 12, 4, 3,
                                      2, 1, 0, 9, 32, 48, 40, 21, 14, 26, 50, 58, 37, 19, 15, 16, 17, 18, 44, 20, 8, 31,
                                      47, 59, 38, 10, 27, 51, 49, 41, 22, 5, 33, 54, 57, 36, 13, 25
                                  ],
                                  [
                                      25, 54, 57, 36, 13, 35, 12, 29, 53, 56, 30, 46, 43, 24, 7, 20, 8, 31, 47, 44, 5, 33,
                                      49, 41, 22, 0, 4, 3, 2, 1, 15, 16, 17, 18, 19, 10, 27, 51, 59, 38, 45, 42, 23, 6, 34,
                                      50, 58, 37, 14, 26, 40, 21, 9, 32, 48, 55, 39, 11, 28, 52
                                  ],
                                  [
                                      26, 50, 58, 37, 14, 36, 13, 25, 54, 57, 31, 47, 44, 20, 8, 21, 9, 32, 48, 40, 6, 34,
                                      45, 42, 23, 1, 0, 4, 3, 2, 16, 17, 18, 19, 15, 11, 28, 52, 55, 39, 46, 43, 24, 7, 30,
                                      51, 59, 38, 10, 27, 41, 22, 5, 33, 49, 56, 35, 12, 29, 53
                                  ],
                                  [
                                      27, 51, 59, 38, 10, 37, 14, 26, 50, 58, 32, 48, 40, 21, 9, 22, 5, 33, 49, 41, 7, 30,
                                      46, 43, 24, 2, 1, 0, 4, 3, 17, 18, 19, 15, 16, 12, 29, 53, 56, 35, 47, 44, 20, 8, 31,
                                      52, 55, 39, 11, 28, 42, 23, 6, 34, 45, 57, 36, 13, 25, 54
                                  ],
                                  [
                                      28, 52, 55, 39, 11, 38, 10, 27, 51, 59, 33, 49, 41, 22, 5, 23, 6, 34, 45, 42, 8, 31,
                                      47, 44, 20, 3, 2, 1, 0, 4, 18, 19, 15, 16, 17, 13, 25, 54, 57, 36, 48, 40, 21, 9, 32,
                                      53, 56, 35, 12, 29, 43, 24, 7, 30, 46, 58, 37, 14, 26, 50
                                  ],
                                  [
                                      29, 53, 56, 35, 12, 39, 11, 28, 52, 55, 34, 45, 42, 23, 6, 24, 7, 30, 46, 43, 9, 32,
                                      48, 40, 21, 4, 3, 2, 1, 0, 19, 15, 16, 17, 18, 14, 26, 50, 58, 37, 49, 41, 22, 5, 33,
                                      54, 57, 36, 13, 25, 44, 20, 8, 31, 47, 59, 38, 10, 27, 51
                                  ],
                                  [
                                      30, 46, 43, 24, 7, 20, 8, 31, 47, 44, 25, 54, 57, 36, 13, 35, 12, 29, 53, 56, 10, 27,
                                      51, 59, 38, 15, 16, 17, 18, 19, 0, 4, 3, 2, 1, 5, 33, 49, 41, 22, 50, 58, 37, 14, 26,
                                      45, 42, 23, 6, 34, 55, 39, 11, 28, 52, 40, 21, 9, 32, 48
                                  ],
                                  [
                                      31, 47, 44, 20, 8, 21, 9, 32, 48, 40, 26, 50, 58, 37, 14, 36, 13, 25, 54, 57, 11, 28,
                                      52, 55, 39, 16, 17, 18, 19, 15, 1, 0, 4, 3, 2, 6, 34, 45, 42, 23, 51, 59, 38, 10, 27,
                                      46, 43, 24, 7, 30, 56, 35, 12, 29, 53, 41, 22, 5, 33, 49
                                  ],
                                  [
                                      32, 48, 40, 21, 9, 22, 5, 33, 49, 41, 27, 51, 59, 38, 10, 37, 14, 26, 50, 58, 12, 29,
                                      53, 56, 35, 17, 18, 19, 15, 16, 2, 1, 0, 4, 3, 7, 30, 46, 43, 24, 52, 55, 39, 11, 28,
                                      47, 44, 20, 8, 31, 57, 36, 13, 25, 54, 42, 23, 6, 34, 45
                                  ],
                                  [
                                      33, 49, 41, 22, 5, 23, 6, 34, 45, 42, 28, 52, 55, 39, 11, 38, 10, 27, 51, 59, 13, 25,
                                      54, 57, 36, 18, 19, 15, 16, 17, 3, 2, 1, 0, 4, 8, 31, 47, 44, 20, 53, 56, 35, 12, 29,
                                      48, 40, 21, 9, 32, 58, 37, 14, 26, 50, 43, 24, 7, 30, 46
                                  ],
                                  [
                                      34, 45, 42, 23, 6, 24, 7, 30, 46, 43, 29, 53, 56, 35, 12, 39, 11, 28, 52, 55, 14, 26,
                                      50, 58, 37, 19, 15, 16, 17, 18, 4, 3, 2, 1, 0, 9, 32, 48, 40, 21, 54, 57, 36, 13, 25,
                                      49, 41, 22, 5, 33, 59, 38, 10, 27, 51, 44, 20, 8, 31, 47
                                  ],
                                  [
                                      35, 12, 29, 53, 56, 25, 54, 57, 36, 13, 20, 8, 31, 47, 44, 30, 46, 43, 24, 7, 15, 16,
                                      17, 18, 19, 10, 27, 51, 59, 38, 5, 33, 49, 41, 22, 0, 4, 3, 2, 1, 55, 39, 11, 28, 52,
                                      40, 21, 9, 32, 48, 50, 58, 37, 14, 26, 45, 42, 23, 6, 34
                                  ],
                                  [
                                      36, 13, 25, 54, 57, 26, 50, 58, 37, 14, 21, 9, 32, 48, 40, 31, 47, 44, 20, 8, 16, 17,
                                      18, 19, 15, 11, 28, 52, 55, 39, 6, 34, 45, 42, 23, 1, 0, 4, 3, 2, 56, 35, 12, 29, 53,
                                      41, 22, 5, 33, 49, 51, 59, 38, 10, 27, 46, 43, 24, 7, 30
                                  ],
                                  [
                                      37, 14, 26, 50, 58, 27, 51, 59, 38, 10, 22, 5, 33, 49, 41, 32, 48, 40, 21, 9, 17, 18,
                                      19, 15, 16, 12, 29, 53, 56, 35, 7, 30, 46, 43, 24, 2, 1, 0, 4, 3, 57, 36, 13, 25, 54,
                                      42, 23, 6, 34, 45, 52, 55, 39, 11, 28, 47, 44, 20, 8, 31
                                  ],
                                  [
                                      38, 10, 27, 51, 59, 28, 52, 55, 39, 11, 23, 6, 34, 45, 42, 33, 49, 41, 22, 5, 18, 19,
                                      15, 16, 17, 13, 25, 54, 57, 36, 8, 31, 47, 44, 20, 3, 2, 1, 0, 4, 58, 37, 14, 26, 50,
                                      43, 24, 7, 30, 46, 53, 56, 35, 12, 29, 48, 40, 21, 9, 32
                                  ],
                                  [
                                      39, 11, 28, 52, 55, 29, 53, 56, 35, 12, 24, 7, 30, 46, 43, 34, 45, 42, 23, 6, 19, 15,
                                      16, 17, 18, 14, 26, 50, 58, 37, 9, 32, 48, 40, 21, 4, 3, 2, 1, 0, 59, 38, 10, 27, 51,
                                      44, 20, 8, 31, 47, 54, 57, 36, 13, 25, 49, 41, 22, 5, 33
                                  ],
                                  [
                                      40, 21, 9, 32, 48, 55, 39, 11, 28, 52, 45, 42, 23, 6, 34, 50, 58, 37, 14, 26, 20, 8,
                                      31, 47, 44, 30, 46, 43, 24, 7, 35, 12, 29, 53, 56, 25, 54, 57, 36, 13, 0, 4, 3, 2, 1,
                                      5, 33, 49, 41, 22, 10, 27, 51, 59, 38, 15, 16, 17, 18, 19
                                  ],
                                  [
                                      41, 22, 5, 33, 49, 56, 35, 12, 29, 53, 46, 43, 24, 7, 30, 51, 59, 38, 10, 27, 21, 9,
                                      32, 48, 40, 31, 47, 44, 20, 8, 36, 13, 25, 54, 57, 26, 50, 58, 37, 14, 1, 0, 4, 3, 2,
                                      6, 34, 45, 42, 23, 11, 28, 52, 55, 39, 16, 17, 18, 19, 15
                                  ],
                                  [
                                      42, 23, 6, 34, 45, 57, 36, 13, 25, 54, 47, 44, 20, 8, 31, 52, 55, 39, 11, 28, 22, 5,
                                      33, 49, 41, 32, 48, 40, 21, 9, 37, 14, 26, 50, 58, 27, 51, 59, 38, 10, 2, 1, 0, 4, 3,
                                      7, 30, 46, 43, 24, 12, 29, 53, 56, 35, 17, 18, 19, 15, 16
                                  ],
                                  [
                                      43, 24, 7, 30, 46, 58, 37, 14, 26, 50, 48, 40, 21, 9, 32, 53, 56, 35, 12, 29, 23, 6,
                                      34, 45, 42, 33, 49, 41, 22, 5, 38, 10, 27, 51, 59, 28, 52, 55, 39, 11, 3, 2, 1, 0, 4,
                                      8, 31, 47, 44, 20, 13, 25, 54, 57, 36, 18, 19, 15, 16, 17
                                  ],
                                  [
                                      44, 20, 8, 31, 47, 59, 38, 10, 27, 51, 49, 41, 22, 5, 33, 54, 57, 36, 13, 25, 24, 7,
                                      30, 46, 43, 34, 45, 42, 23, 6, 39, 11, 28, 52, 55, 29, 53, 56, 35, 12, 4, 3, 2, 1, 0,
                                      9, 32, 48, 40, 21, 14, 26, 50, 58, 37, 19, 15, 16, 17, 18
                                  ],
                                  [
                                      45, 42, 23, 6, 34, 50, 58, 37, 14, 26, 40, 21, 9, 32, 48, 55, 39, 11, 28, 52, 25, 54,
                                      57, 36, 13, 35, 12, 29, 53, 56, 30, 46, 43, 24, 7, 20, 8, 31, 47, 44, 5, 33, 49, 41,
                                      22, 0, 4, 3, 2, 1, 15, 16, 17, 18, 19, 10, 27, 51, 59, 38
                                  ],
                                  [
                                      46, 43, 24, 7, 30, 51, 59, 38, 10, 27, 41, 22, 5, 33, 49, 56, 35, 12, 29, 53, 26, 50,
                                      58, 37, 14, 36, 13, 25, 54, 57, 31, 47, 44, 20, 8, 21, 9, 32, 48, 40, 6, 34, 45, 42,
                                      23, 1, 0, 4, 3, 2, 16, 17, 18, 19, 15, 11, 28, 52, 55, 39
                                  ],
                                  [
                                      47, 44, 20, 8, 31, 52, 55, 39, 11, 28, 42, 23, 6, 34, 45, 57, 36, 13, 25, 54, 27, 51,
                                      59, 38, 10, 37, 14, 26, 50, 58, 32, 48, 40, 21, 9, 22, 5, 33, 49, 41, 7, 30, 46, 43,
                                      24, 2, 1, 0, 4, 3, 17, 18, 19, 15, 16, 12, 29, 53, 56, 35
                                  ],
                                  [
                                      48, 40, 21, 9, 32, 53, 56, 35, 12, 29, 43, 24, 7, 30, 46, 58, 37, 14, 26, 50, 28, 52,
                                      55, 39, 11, 38, 10, 27, 51, 59, 33, 49, 41, 22, 5, 23, 6, 34, 45, 42, 8, 31, 47, 44,
                                      20, 3, 2, 1, 0, 4, 18, 19, 15, 16, 17, 13, 25, 54, 57, 36
                                  ],
                                  [
                                      49, 41, 22, 5, 33, 54, 57, 36, 13, 25, 44, 20, 8, 31, 47, 59, 38, 10, 27, 51, 29, 53,
                                      56, 35, 12, 39, 11, 28, 52, 55, 34, 45, 42, 23, 6, 24, 7, 30, 46, 43, 9, 32, 48, 40,
                                      21, 4, 3, 2, 1, 0, 19, 15, 16, 17, 18, 14, 26, 50, 58, 37
                                  ],
                                  [
                                      50, 58, 37, 14, 26, 45, 42, 23, 6, 34, 55, 39, 11, 28, 52, 40, 21, 9, 32, 48, 30, 46,
                                      43, 24, 7, 20, 8, 31, 47, 44, 25, 54, 57, 36, 13, 35, 12, 29, 53, 56, 10, 27, 51, 59,
                                      38, 15, 16, 17, 18, 19, 0, 4, 3, 2, 1, 5, 33, 49, 41, 22
                                  ],
                                  [
                                      51, 59, 38, 10, 27, 46, 43, 24, 7, 30, 56, 35, 12, 29, 53, 41, 22, 5, 33, 49, 31, 47,
                                      44, 20, 8, 21, 9, 32, 48, 40, 26, 50, 58, 37, 14, 36, 13, 25, 54, 57, 11, 28, 52, 55,
                                      39, 16, 17, 18, 19, 15, 1, 0, 4, 3, 2, 6, 34, 45, 42, 23
                                  ],
                                  [
                                      52, 55, 39, 11, 28, 47, 44, 20, 8, 31, 57, 36, 13, 25, 54, 42, 23, 6, 34, 45, 32, 48,
                                      40, 21, 9, 22, 5, 33, 49, 41, 27, 51, 59, 38, 10, 37, 14, 26, 50, 58, 12, 29, 53, 56,
                                      35, 17, 18, 19, 15, 16, 2, 1, 0, 4, 3, 7, 30, 46, 43, 24
                                  ],
                                  [
                                      53, 56, 35, 12, 29, 48, 40, 21, 9, 32, 58, 37, 14, 26, 50, 43, 24, 7, 30, 46, 33, 49,
                                      41, 22, 5, 23, 6, 34, 45, 42, 28, 52, 55, 39, 11, 38, 10, 27, 51, 59, 13, 25, 54, 57,
                                      36, 18, 19, 15, 16, 17, 3, 2, 1, 0, 4, 8, 31, 47, 44, 20
                                  ],
                                  [
                                      54, 57, 36, 13, 25, 49, 41, 22, 5, 33, 59, 38, 10, 27, 51, 44, 20, 8, 31, 47, 34, 45,
                                      42, 23, 6, 24, 7, 30, 46, 43, 29, 53, 56, 35, 12, 39, 11, 28, 52, 55, 14, 26, 50, 58,
                                      37, 19, 15, 16, 17, 18, 4, 3, 2, 1, 0, 9, 32, 48, 40, 21
                                  ],
                                  [
                                      55, 39, 11, 28, 52, 40, 21, 9, 32, 48, 50, 58, 37, 14, 26, 45, 42, 23, 6, 34, 35, 12,
                                      29, 53, 56, 25, 54, 57, 36, 13, 20, 8, 31, 47, 44, 30, 46, 43, 24, 7, 15, 16, 17, 18,
                                      19, 10, 27, 51, 59, 38, 5, 33, 49, 41, 22, 0, 4, 3, 2, 1
                                  ],
                                  [
                                      56, 35, 12, 29, 53, 41, 22, 5, 33, 49, 51, 59, 38, 10, 27, 46, 43, 24, 7, 30, 36, 13,
                                      25, 54, 57, 26, 50, 58, 37, 14, 21, 9, 32, 48, 40, 31, 47, 44, 20, 8, 16, 17, 18, 19,
                                      15, 11, 28, 52, 55, 39, 6, 34, 45, 42, 23, 1, 0, 4, 3, 2
                                  ],
                                  [
                                      57, 36, 13, 25, 54, 42, 23, 6, 34, 45, 52, 55, 39, 11, 28, 47, 44, 20, 8, 31, 37, 14,
                                      26, 50, 58, 27, 51, 59, 38, 10, 22, 5, 33, 49, 41, 32, 48, 40, 21, 9, 17, 18, 19, 15,
                                      16, 12, 29, 53, 56, 35, 7, 30, 46, 43, 24, 2, 1, 0, 4, 3
                                  ],
                                  [
                                      58, 37, 14, 26, 50, 43, 24, 7, 30, 46, 53, 56, 35, 12, 29, 48, 40, 21, 9, 32, 38, 10,
                                      27, 51, 59, 28, 52, 55, 39, 11, 23, 6, 34, 45, 42, 33, 49, 41, 22, 5, 18, 19, 15, 16,
                                      17, 13, 25, 54, 57, 36, 8, 31, 47, 44, 20, 3, 2, 1, 0, 4
                                  ],
                                  [
                                      59, 38, 10, 27, 51, 44, 20, 8, 31, 47, 54, 57, 36, 13, 25, 49, 41, 22, 5, 33, 39, 11,
                                      28, 52, 55, 29, 53, 56, 35, 12, 24, 7, 30, 46, 43, 34, 45, 42, 23, 6, 19, 15, 16, 17,
                                      18, 14, 26, 50, 58, 37, 9, 32, 48, 40, 21, 4, 3, 2, 1, 0
                                  ]])
        Rs = torch.zeros(60, 3, 3)
        Rs[0] = torch.tensor([[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000],
                              [0.000000, 0.000000, 1.000000]])
        Rs[1] = torch.tensor([[0.500000, -0.809017, 0.309017], [0.809017, 0.309017, -0.500000],
                              [0.309017, 0.500000, 0.809017]])
        Rs[2] = torch.tensor([[-0.309017, -0.500000, 0.809017], [0.500000, -0.809017, -0.309017],
                              [0.809017, 0.309017, 0.500000]])
        Rs[3] = torch.tensor([[-0.309017, 0.500000, 0.809017], [-0.500000, -0.809017, 0.309017],
                              [0.809017, -0.309017, 0.500000]])
        Rs[4] = torch.tensor([[0.500000, 0.809017, 0.309017], [-0.809017, 0.309017, 0.500000],
                              [0.309017, -0.500000, 0.809017]])
        Rs[5] = torch.tensor([[-0.809017, 0.309017, 0.500000], [0.309017, -0.500000, 0.809017],
                              [0.500000, 0.809017, 0.309017]])
        Rs[6] = torch.tensor([[0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                              [1.000000, 0.000000, 0.000000]])
        Rs[7] = torch.tensor([[0.809017, 0.309017, -0.500000], [0.309017, 0.500000, 0.809017],
                              [0.500000, -0.809017, 0.309017]])
        Rs[8] = torch.tensor([[0.500000, -0.809017, -0.309017], [0.809017, 0.309017, 0.500000],
                              [-0.309017, -0.500000, 0.809017]])
        Rs[9] = torch.tensor([[-0.500000, -0.809017, 0.309017], [0.809017, -0.309017, 0.500000],
                              [-0.309017, 0.500000, 0.809017]])
        Rs[10] = torch.tensor([[-0.500000, -0.809017, 0.309017], [-0.809017, 0.309017, -0.500000],
                               [0.309017, -0.500000, -0.809017]])
        Rs[11] = torch.tensor([[-0.809017, 0.309017, 0.500000], [-0.309017, 0.500000, -0.809017],
                               [-0.500000, -0.809017, -0.309017]])
        Rs[12] = torch.tensor([[0.000000, 1.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                               [-1.000000, 0.000000, 0.000000]])
        Rs[13] = torch.tensor([[0.809017, 0.309017, -0.500000], [-0.309017, -0.500000, -0.809017],
                               [-0.500000, 0.809017, -0.309017]])
        Rs[14] = torch.tensor([[0.500000, -0.809017, -0.309017], [-0.809017, -0.309017, -0.500000],
                               [0.309017, 0.500000, -0.809017]])
        Rs[15] = torch.tensor([[0.309017, 0.500000, -0.809017], [0.500000, -0.809017, -0.309017],
                               [-0.809017, -0.309017, -0.500000]])
        Rs[16] = torch.tensor([[0.309017, -0.500000, -0.809017], [-0.500000, -0.809017, 0.309017],
                               [-0.809017, 0.309017, -0.500000]])
        Rs[17] = torch.tensor([[-0.500000, -0.809017, -0.309017], [-0.809017, 0.309017, 0.500000],
                               [-0.309017, 0.500000, -0.809017]])
        Rs[18] = torch.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000],
                               [0.000000, 0.000000, -1.000000]])
        Rs[19] = torch.tensor([[-0.500000, 0.809017, -0.309017], [0.809017, 0.309017, -0.500000],
                               [-0.309017, -0.500000, -0.809017]])
        Rs[20] = torch.tensor([[-0.500000, -0.809017, -0.309017], [0.809017, -0.309017, -0.500000],
                               [0.309017, -0.500000, 0.809017]])
        Rs[21] = torch.tensor([[-1.000000, 0.000000, 0.000000], [0.000000, -1.000000, 0.000000],
                               [0.000000, 0.000000, 1.000000]])
        Rs[22] = torch.tensor([[-0.500000, 0.809017, -0.309017], [-0.809017, -0.309017, 0.500000],
                               [0.309017, 0.500000, 0.809017]])
        Rs[23] = torch.tensor([[0.309017, 0.500000, -0.809017], [-0.500000, 0.809017, 0.309017],
                               [0.809017, 0.309017, 0.500000]])
        Rs[24] = torch.tensor([[0.309017, -0.500000, -0.809017], [0.500000, 0.809017, -0.309017],
                               [0.809017, -0.309017, 0.500000]])
        Rs[25] = torch.tensor([[0.000000, 0.000000, -1.000000], [-1.000000, 0.000000, 0.000000],
                               [0.000000, 1.000000, 0.000000]])
        Rs[26] = torch.tensor([[-0.309017, -0.500000, -0.809017], [-0.500000, 0.809017, -0.309017],
                               [0.809017, 0.309017, -0.500000]])
        Rs[27] = torch.tensor([[-0.809017, -0.309017, -0.500000], [0.309017, 0.500000, -0.809017],
                               [0.500000, -0.809017, -0.309017]])
        Rs[28] = torch.tensor([[-0.809017, 0.309017, -0.500000], [0.309017, -0.500000, -0.809017],
                               [-0.500000, -0.809017, 0.309017]])
        Rs[29] = torch.tensor([[-0.309017, 0.500000, -0.809017], [-0.500000, -0.809017, -0.309017],
                               [-0.809017, 0.309017, 0.500000]])
        Rs[30] = torch.tensor([[0.809017, 0.309017, 0.500000], [-0.309017, -0.500000, 0.809017],
                               [0.500000, -0.809017, -0.309017]])
        Rs[31] = torch.tensor([[0.809017, -0.309017, 0.500000], [-0.309017, 0.500000, 0.809017],
                               [-0.500000, -0.809017, 0.309017]])
        Rs[32] = torch.tensor([[0.309017, -0.500000, 0.809017], [0.500000, 0.809017, 0.309017],
                               [-0.809017, 0.309017, 0.500000]])
        Rs[33] = torch.tensor([[0.000000, 0.000000, 1.000000], [1.000000, 0.000000, 0.000000],
                               [0.000000, 1.000000, 0.000000]])
        Rs[34] = torch.tensor([[0.309017, 0.500000, 0.809017], [0.500000, -0.809017, 0.309017],
                               [0.809017, 0.309017, -0.500000]])
        Rs[35] = torch.tensor([[-0.309017, 0.500000, 0.809017], [0.500000, 0.809017, -0.309017],
                               [-0.809017, 0.309017, -0.500000]])
        Rs[36] = torch.tensor([[0.500000, 0.809017, 0.309017], [0.809017, -0.309017, -0.500000],
                               [-0.309017, 0.500000, -0.809017]])
        Rs[37] = torch.tensor([[1.000000, 0.000000, 0.000000], [0.000000, -1.000000, 0.000000],
                               [0.000000, 0.000000, -1.000000]])
        Rs[38] = torch.tensor([[0.500000, -0.809017, 0.309017], [-0.809017, -0.309017, 0.500000],
                               [-0.309017, -0.500000, -0.809017]])
        Rs[39] = torch.tensor([[-0.309017, -0.500000, 0.809017], [-0.500000, 0.809017, 0.309017],
                               [-0.809017, -0.309017, -0.500000]])
        Rs[40] = torch.tensor([[-0.500000, 0.809017, 0.309017], [-0.809017, -0.309017, -0.500000],
                               [-0.309017, -0.500000, 0.809017]])
        Rs[41] = torch.tensor([[0.500000, 0.809017, -0.309017], [-0.809017, 0.309017, -0.500000],
                               [-0.309017, 0.500000, 0.809017]])
        Rs[42] = torch.tensor([[0.809017, -0.309017, -0.500000], [-0.309017, 0.500000, -0.809017],
                               [0.500000, 0.809017, 0.309017]])
        Rs[43] = torch.tensor([[0.000000, -1.000000, 0.000000], [0.000000, 0.000000, -1.000000],
                               [1.000000, 0.000000, 0.000000]])
        Rs[44] = torch.tensor([[-0.809017, -0.309017, 0.500000], [-0.309017, -0.500000, -0.809017],
                               [0.500000, -0.809017, 0.309017]])
        Rs[45] = torch.tensor([[0.809017, -0.309017, 0.500000], [0.309017, -0.500000, -0.809017],
                               [0.500000, 0.809017, -0.309017]])
        Rs[46] = torch.tensor([[0.309017, -0.500000, 0.809017], [-0.500000, -0.809017, -0.309017],
                               [0.809017, -0.309017, -0.500000]])
        Rs[47] = torch.tensor([[0.000000, 0.000000, 1.000000], [-1.000000, 0.000000, 0.000000],
                               [0.000000, -1.000000, 0.000000]])
        Rs[48] = torch.tensor([[0.309017, 0.500000, 0.809017], [-0.500000, 0.809017, -0.309017],
                               [-0.809017, -0.309017, 0.500000]])
        Rs[49] = torch.tensor([[0.809017, 0.309017, 0.500000], [0.309017, 0.500000, -0.809017],
                               [-0.500000, 0.809017, 0.309017]])
        Rs[50] = torch.tensor([[-0.309017, 0.500000, -0.809017], [0.500000, 0.809017, 0.309017],
                               [0.809017, -0.309017, -0.500000]])
        Rs[51] = torch.tensor([[0.000000, 0.000000, -1.000000], [1.000000, 0.000000, 0.000000],
                               [0.000000, -1.000000, 0.000000]])
        Rs[52] = torch.tensor([[-0.309017, -0.500000, -0.809017], [0.500000, -0.809017, 0.309017],
                               [-0.809017, -0.309017, 0.500000]])
        Rs[53] = torch.tensor([[-0.809017, -0.309017, -0.500000], [-0.309017, -0.500000, 0.809017],
                               [-0.500000, 0.809017, 0.309017]])
        Rs[54] = torch.tensor([[-0.809017, 0.309017, -0.500000], [-0.309017, 0.500000, 0.809017],
                               [0.500000, 0.809017, -0.309017]])
        Rs[55] = torch.tensor([[0.000000, -1.000000, 0.000000], [0.000000, 0.000000, 1.000000],
                               [-1.000000, 0.000000, 0.000000]])
        Rs[56] = torch.tensor([[-0.809017, -0.309017, 0.500000], [0.309017, 0.500000, 0.809017],
                               [-0.500000, 0.809017, -0.309017]])
        Rs[57] = torch.tensor([[-0.500000, 0.809017, 0.309017], [0.809017, 0.309017, 0.500000],
                               [0.309017, 0.500000, -0.809017]])
        Rs[58] = torch.tensor([[0.500000, 0.809017, -0.309017], [0.809017, -0.309017, 0.500000],
                               [0.309017, -0.500000, -0.809017]])
        Rs[59] = torch.tensor([[0.809017, -0.309017, -0.500000], [0.309017, -0.500000, 0.809017],
                               [-0.500000, -0.809017, -0.309017]])

        est_radius = 10.0 * SYMA
        offset = torch.tensor([1.0, 0.0, 0.0])
        offset = est_radius * offset / torch.linalg.norm(offset)
        metasymm = ([torch.arange(60)], [min(len(Rs), symopt.max_nsub if symopt else 6)])
    elif symmid.startswith('I'):
        ax2, ax3, ax5 = ipd.sym.axes('I', closest_to=[1, 2, 21]).values()
        pi = torch.pi
        Rs = [torch.eye(3)]
        if '2' in symmid:
            Rs.append(torch.as_tensor(ipd.h.rot3(ax2, pi)))  # type: ignore
        if '3' in symmid:
            Rs.append(torch.as_tensor(ipd.h.rot3(ax3, 2 / 3 * pi)))  # type: ignore
            Rs.append(torch.as_tensor(ipd.h.rot3(ax3, -2 / 3 * pi)))  # type: ignore
        if '5' in symmid:
            Rs.append(torch.as_tensor(ipd.h.rot3(ax5, 2 / 5 * pi)))  # type: ignore
            Rs.append(torch.as_tensor(ipd.h.rot3(ax5, -2 / 5 * pi)))  # type: ignore
        Rs = torch.stack(Rs)
        nsub = len(Rs)
        symmatrix = (torch.arange(nsub)[:, None] - torch.arange(nsub)[None, :]) % nsub
        metasymm = ([torch.arange(nsub)], [min(nsub, symopt.max_nsub if symopt else min(3, nsub))])
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
    symmsub = torch.ones((nchunk, nchunk)) * -1
    C = 0  # a marker for blocks of the same category

    for i in range(N):  # i      = 0, 1,2, 3,4, 5,6...
        offset = int(((i+1) // 2) * (math.pow(-1, i)))  # offset = 0,-1,1,-2,2,-3,3...

        row = torch.arange(nchunk)
        col = torch.roll(row, offset)

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
        assert torch.sum(
            symmsub == -1) == 0, 'Current symmsub not compatible with pseudocycle, increase symmsub_k to nrepeat-1'

    return symmsub.long()

def update_symm_Rs(xyz, Lasu, symmsub, allsymmRs, symopt):
    def dist_error_comp(R0, T0, xyz, fittscale):
        Ts = xyz  #[:,:,1]
        B = xyz.shape[0]

        Tcom = xyz[:, :Lasu].mean(dim=1, keepdim=True)  # LT center of mass for first ASU
        Tcorr = torch.einsum('ij,brj->bri', R0, xyz[:, :Lasu] - Tcom) + Tcom + fittscale * T0[
            None, None, :]  # LT Rotated coordinates of first ASU by learned R0, then translated by learned T0

        # distance map loss
        Xsymm = torch.einsum('sij,brj->bsri', allsymmRs[symmsub], Tcorr).reshape(B, -1, 3)
        Xtrue = Ts

        # compare dmaps via L1 loss
        delsx = Xsymm[:, :Lasu, None] - Xsymm[:, None, Lasu:]
        deltx = Xtrue[:, :Lasu, None] - Xtrue[:, None, Lasu:]
        dsymm = torch.linalg.norm(delsx, dim=-1)
        dtrue = torch.linalg.norm(deltx, dim=-1)
        loss1 = torch.abs(dsymm - dtrue).mean()

        # clash loss
        Xsymmall = torch.einsum('sij,brj->bsri', allsymmRs, Tcorr).reshape(B, -1, 3)
        delsxall = Xsymmall[:, :Lasu, None] - Xsymmall[:, None, Lasu:]
        dsymm = torch.linalg.norm(delsxall, dim=-1)

        clash = torch.clamp(symopt.fitwclash - dsymm, min=0)
        loss2 = torch.sum(clash) / Lasu

        return loss1, loss2  # 0.0

    def dist_error(R0, T0, xyz, fittscale, w_clash=10.0):
        l1, l2 = dist_error_comp(R0, T0, xyz, fittscale)
        return l1 + w_clash*l2

    def Q2R(Q):
        Qs = torch.cat((torch.ones((1), device=Q.device), Q), dim=-1)
        Qs = normQ(Qs)
        return Qs2Rs(Qs[None, :]).squeeze(0)

    B = xyz.shape[0]
    L = xyz.shape[1]
    natoms = xyz.shape[2]

    if symopt.fit:
        with torch.enable_grad():
            T0 = torch.zeros(3, device=xyz.device).requires_grad_(True)
            Q0 = torch.zeros(3, device=xyz.device).requires_grad_(True)
            lbfgs = torch.optim.LBFGS([T0, Q0], history_size=10, max_iter=4, line_search_fn="strong_wolfe")

            def closure():
                lbfgs.zero_grad()
                i = 1 if xyz.shape[2] > 1 else 0
                loss = dist_error(Q2R(Q0), T0, xyz[:, :, i], symopt.fittscale)
                loss.backward()  #retain_graph=True)
                return loss

            for e in range(4):
                loss = lbfgs.step(closure)

            Tcom = xyz[:, :Lasu].mean(dim=1, keepdim=True).detach()
            Q0 = Q0.detach()
            T0 = T0.detach()
            xyz = torch.einsum('ij,braj->brai', Q2R(Q0),
                               xyz[:, :Lasu] - Tcom) + Tcom + symopt.fittscale * T0[None, None, :]

    xyz = torch.einsum('sij,braj->bsrai', allsymmRs[symmsub], xyz[:, :Lasu])
    xyz = xyz.reshape(B, -1, natoms, 3)  # (B,S,L,3,3) or (B,LASU*S,natoms,3)

    # if symopt.fit:
    # ipd.pdb.dumppdb(f'{symopt.tag}_fit1.pdb', xyz[:,:,:3,:3].reshape(L//Lasu,Lasu,3,3))

    return xyz

def update_symm_subs_track_module(xyz, pair, symmids, symmsub, allsymmRs, metasymm, symopt):
    B, Ls = xyz.shape[0:2]
    Osub = symmsub.shape[0]
    L = Ls // Osub

    com = xyz[:, :L, 1].mean(dim=-2)
    rcoms = torch.einsum('sij,bj->si', allsymmRs, com)
    subsymms, nneighs = metasymm
    symmsub_new = []
    for i in range(len(subsymms)):
        drcoms = torch.linalg.norm(rcoms[0, :] - rcoms[subsymms[i], :], dim=-1)
        _, subs_i = torch.topk(drcoms, nneighs[i], largest=False)
        subs_i, _ = torch.sort(subsymms[i][subs_i])
        symmsub_new.append(subs_i)

    symmsub_new = torch.cat(symmsub_new)
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
            mag_ij = torch.max(sub_ij.flatten())  #torch.norm(sub_ij.flatten())
            if idx_old not in pairsub or mag_ij > pairmag[idx_old]:
                pairmag[idx_old] = mag_ij
                pairsub[idx_old] = (i, j)  #sub_ij

    # b) reindex
    idx = torch.zeros((Osub * L, Osub * L), dtype=torch.long, device=pair.device)
    idx = (torch.arange(Osub * L, device=pair.device)[:, None] * Osub * L +
           torch.arange(Osub * L, device=pair.device)[None, :])
    for i in range(Osub):
        for j in range(Osub):
            idx_new = s_new[i, j].item()
            if idx_new in pairsub:
                inew, jnew = pairsub[idx_new]
                idx[i * L:(i+1) * L, j * L:(j+1) * L] = (Osub * L * torch.arange(inew * L, (inew+1) * L)[:, None] +
                                                         torch.arange(jnew * L, (jnew+1) * L)[None, :])

    pair = pair.reshape(1, -1, pair.shape[-1])[:, idx.flatten(), :].view(1, Osub * L, Osub * L, pair.shape[-1])

    if symmsub is not None and symmsub.shape[0] > 1:
        xyz = update_symm_Rs(xyz, L, symmsub_new, allsymmRs, symopt)

    return xyz, pair, symmsub_new
