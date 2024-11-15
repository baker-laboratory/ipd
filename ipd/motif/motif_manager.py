from abc import ABC, abstractmethod
from copy import deepcopy
import ipd
from ipd.motif.motif_factory import MetaMotifManager, set_default_motif_manager

class MotifManager(ABC, metaclass=MetaMotifManager):
    """All motif logic should go through a MotifManager."""
    kind = None

    def __init__(self, opts=None, device='cpu'):
        super().__init__()
        self.opts = opts or ipd.Bunch()
        self.device = device

    def post_init(self, *a, **kw):
        pass

    @property
    def have_motif(self):
        return True

    @abstractmethod
    def __call__(self, *a, **kw):
        self.apply_motifs(*a, **kw)  # type: ignore

    def setup_for_motifs(self, thing):
        return thing

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

class NullMotifManager(MotifManager):
    """No-op motif manager."""
    kind = 'nomotif'

    def __call__(self, xyz, **kw):
        return xyz

    @property
    def have_motif(self):  # type: ignore
        return False

set_default_motif_manager('nomotif')
