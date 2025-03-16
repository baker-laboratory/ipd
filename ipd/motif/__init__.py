from ipd.motif.motif_options import *
from ipd.motif.motif_factory import *
from ipd.motif.motif_manager import *

_global_motif_manager = None

def set_global_motif_manager(motif_manager):
    global _global_motif_manager
    _global_motif_manager = motif_manager

def get_global_motif_manager():
    global _global_motif_manager
    if _global_motif_manager is None:
        _global_motif_manager = create_motif_manager(kind='nomotif')
    return _global_motif_manager

def __getattr__(name):
    if name == 'motif_applier':
        mmgr = get_global_motif_manager()
        assert mmgr is not None
        return mmgr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
