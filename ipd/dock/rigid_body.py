import abc

class RigidBody(abc.ABC):
    """base class for docking related rigid bodies"""
    def __init__(self, xyz):
        self.xyz = xyz

class InvRotBody(RigidBody):
    """rigid body with inverse rotamers"""
    def __init__(self, xyz, rots, tips):
        super().__init__(xyz)
        self.rots = rots
        self.tips = tips
