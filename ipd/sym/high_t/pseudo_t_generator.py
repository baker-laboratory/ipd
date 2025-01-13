import numpy as np

from ipd.homog import *

import torch

def get_pseudo_highT(opt):
    pseudo_gen = PseudoSymmGenerator(h=opt.H_K[0], k=opt.H_K[1], spherical_frac=opt.spherical_frac)
    xforms = torch.tensor(pseudo_gen.HTs)
    Rs = [X[:3, :3] for X in xforms]
    Ts = torch.stack([X[:3, 3] for X in xforms])
    Ts *= opt.radius
    Ts = [T - Ts[0] for T in Ts]
    xforms = []
    # convert the first rotation back to the identity operation
    R_inv = torch.linalg.inv(Rs[0])
    for i, R in enumerate(Rs):
        R = R @ R_inv
        X = torch.eye(4)
        X[:3, :3] = R
        X[:3, 3] = Ts[i]
        xforms.append(X)
    return torch.stack(xforms)

class PseudoSymmGenerator:
    # from Frank DiMaio
    def __init__(self, h, k, spherical_frac=0.0):
        # 0) h=1,k=0 ref
        ico_samples_I = self.gen_ico_samples(1, 0, 0.0)
        edges_I = self.get_edges(ico_samples_I)
        HTs_I = self.get_transforms(ico_samples_I, edges_I, 0.0)
        HTs_I = np.einsum('jk,bij->bik', np.linalg.inv(HTs_I[0]), HTs_I)  # HTs relative to xform 0

        # 1) get points
        ico_samples = self.gen_ico_samples(h, k, spherical_frac)

        # 2) get edges
        edges = self.get_edges(ico_samples)
        assert edges.shape[0] // 60 == (h*h + k*k + h*k)

        # 2b) select asymmetric subset
        edges_ASU = self.extract_ASU_edges(edges, ico_samples, HTs_I)

        # 3) get homogeneous transforms
        HTs = self.get_transforms(ico_samples, edges, 0.0)
        HTs_ASU = self.get_transforms(ico_samples, edges_ASU, 0.0)

        oris = self.get_origin(HTs, ico_samples, edges)
        HTs[:, :3, 3] = oris

        # 4) save
        self.Tnum = h*h + k*k + h*k
        self.points = ico_samples
        self.edges = edges
        self.edges_ASU = edges_ASU
        self.HTs = HTs
        self.HTs_ASU = HTs_ASU
        self.HTs_I = HTs_I

    # generates points and edges for a pseudosymmetric polygon with the given Caspar-Klug coefficients H and K
    def gen_ico_samples(self, H, K, spherical_frac):
        theta = 26.56505117707799 * np.pi / 180.0
        stheta, ctheta = np.sin(theta), np.cos(theta)

        icoBase, ico = [], []
        icoBase.append(np.array([0.0, 0.0, -1.0]))
        phi = np.pi / 5.0
        for i in range(5):
            icoBase.append(np.array([ctheta * np.cos(phi), ctheta * np.sin(phi), -stheta]))
            phi += 2.0 * np.pi / 5.0
        phi = 0.0
        for i in range(5):
            icoBase.append(np.array([ctheta * np.cos(phi), ctheta * np.sin(phi), stheta]))
            phi += 2.0 * np.pi / 5.0
        icoBase.append(np.array([0.0, 0.0, 1.0]))
        TRIS = ([[0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 5, 4], [0, 1, 5], [1, 2, 7], [2, 3, 8], [3, 4, 9], [4, 5, 10],
                 [5, 1, 6], [1, 7, 6], [2, 8, 7], [3, 9, 8], [4, 10, 9], [5, 6, 10], [6, 7, 11], [7, 8, 11], [8, 9, 11],
                 [9, 10, 11], [10, 6, 11]])
        EDGES = ([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 5], [1, 6], [1, 7], [2, 3], [2, 7], [2, 8], [3, 4],
                  [3, 8], [3, 9], [4, 5], [4, 9], [4, 10], [5, 6], [5, 10], [6, 7], [6, 10], [6, 11], [7, 8], [7, 11],
                  [8, 9], [8, 11], [9, 10], [9, 11], [10, 11]])

        for ii, fi in enumerate(TRIS):
            a, b, c = icoBase[fi[0]], icoBase[fi[1]], icoBase[fi[2]]
            ab, ac = b - a, c - a
            if (K != 0):
                basis1 = (H*ab + K*ac) / (H*H + K*K + H*K)
                basis2 = (ab - H*basis1) / (K)
            else:
                basis1 = (ab) / (H)
                basis2 = (ac) / (H)

            #fd not sure about these bounds...
            #fd  .. and there has to be a better way to iterate?
            for h in range(-(H + K), (H + K + 1)):
                for k in range(-(H + K), (H + K + 1)):
                    newpoint = a + h*basis1 + k*basis2
                    ad = newpoint - a
                    dot00 = np.dot(ac, ac)
                    dot01 = np.dot(ac, ab)
                    dot02 = np.dot(ac, ad)
                    dot11 = np.dot(ab, ab)
                    dot12 = np.dot(ab, ad)
                    invDenom = 1 / (dot00*dot11 - dot01*dot01)
                    # Compute barycentric coordinates
                    u = (dot11*dot02 - dot01*dot12) * invDenom
                    v = (dot00*dot12 - dot01*dot02) * invDenom
                    w = 1 - u - v
                    if ((u > -1e-6 and v > -1e-6 and w > -1e-6)):  # in the triangle
                        if spherical_frac != 0.0:
                            newpointS = newpoint / np.linalg.norm(newpoint)
                            newpoint = spherical_frac*newpointS + (1-spherical_frac) * newpoint
                        ico.append(newpoint)

        ico = np.stack(ico)

        # delete duplicated points
        Ds = np.linalg.norm(ico[:, None] - ico[None, :], axis=-1)
        x, y = np.tril_indices(Ds.shape[0], k=0, m=None)
        Ds[x, y] = 999
        _, todel = (Ds < 1e-8).nonzero()
        mask = np.ones(Ds.shape[0], dtype=bool)
        mask[todel] = False
        ico = ico[mask]

        return ico

    def get_edges(self, ico_samples):
        Ds = np.linalg.norm(ico_samples[:, None] - ico_samples[None, :], axis=-1)
        Ds[Ds < 1e-8] = 999.0
        minDist = np.min(Ds)
        edges_i, edges_j = np.nonzero(Ds < (minDist * 1.5))  #fd is this too permissive? not permissive enough?
        edges = np.stack((edges_i, edges_j), axis=1)
        return edges

    def extract_ASU_edges(self, edges, ico_samples, HTs_I):
        edges_ASU = []
        for ei, ej in edges:
            dij = ico_samples[ej] - ico_samples[ei]
            di = ico_samples[ei]
            s_ij_rot = np.einsum('bji,i->bj', HTs_I[:, :3, :3], dij)
            s_i_rot = np.einsum('bji,i->bj', HTs_I[:, :3, :3], di)
            toAdd = True
            for ek, el in edges_ASU:
                dkl = ico_samples[el] - ico_samples[ek]
                dk = ico_samples[ek]
                ddkl = np.linalg.norm(s_ij_rot - dkl[None], axis=-1) + np.linalg.norm(s_i_rot - dk[None], axis=-1)
                if (np.min(ddkl) < 1e-4):
                    toAdd = False
                    break
            if toAdd:
                edges_ASU.append([ei, ej])

        return np.array(edges_ASU)

    def get_transforms(self, ico_samples, edges, offset):
        # each edge corresponds to a transform
        # ori = point location
        # +y = along edge
        # +x = away from origin
        # +z = cross(+x,+y)w
        transforms = []
        for ei, ej in edges:
            RT_i = np.eye(4)
            ori = ico_samples[ei] + offset * (ico_samples[ej] - ico_samples[ei])
            x = ico_samples[ei]  #- np.dot(self.ico_samples[ei],y)*y
            x = x / np.linalg.norm(x)
            y = ico_samples[ej] - ico_samples[ei]  #
            y = y - np.dot(x, y) * x
            y = y / np.linalg.norm(y)
            z = np.cross(x, y)
            RT_i[:3, 0] = x
            RT_i[:3, 1] = y
            RT_i[:3, 2] = z
            RT_i[:3, 3] = ori
            transforms.append(RT_i)
        return np.stack(transforms)

    def _get_cost(self, ori0, HTs):
        oris = torch.einsum('bij,j->bi', HTs, ori0)[:, :3]
        MAG = torch.linalg.norm(oris, dim=-1).max()
        oris = oris / MAG
        allDs = torch.linalg.norm(oris[:, None] - oris[None, :], dim=-1)
        ii, jj = torch.triu_indices(*allDs.shape, 1)
        return -torch.min(allDs[ii, jj]), MAG

    def get_origin(self, HTs, ico_samples, edges):
        HTs = torch.tensor(HTs).float()

        ori0 = torch.tensor([0.0, 0.1, 0.1, 1], requires_grad=True)
        optimizer = torch.optim.Adam([ori0], lr=0.01)
        for i in range(20):
            optimizer.zero_grad()
            loss, MAG = self._get_cost(ori0, HTs)
            loss.backward()
            optimizer.step()

        oris = torch.einsum('bij,j->bi', HTs, ori0)[:, :3]
        MAG = torch.linalg.norm(oris, dim=-1).max()
        oris = oris / MAG
        return oris.detach().numpy()
