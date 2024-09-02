import ipd
import torch as th

class VoxRB(ipd.voxel.Voxel):
    '''Represents a rigid body with an associated Voxel score'''
    def __init__(self, *a, func=ipd.cuda.ContactFunc(), **kw):
        super().__init__(*a, func=func, **kw)
        self._vizpos = th.eye(4)

    def score(self, other, pos=th.eye(4), otherpos=th.eye(4), **kw):
        if isinstance(other, VoxRB):
            other = other.xyz
        return super().score(other, otherpos, pos, **kw)



class VoxRBSym:
    '''Represents a symmetric rigid body with an associated Voxel score'''
    def __init__(self, rb, frames):
        super().__init__()
        self.rb = rb
        self.frames = frames
