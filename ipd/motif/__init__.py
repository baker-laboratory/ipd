from ipd.motif.motif_options import *
from ipd.motif.motif_factory import *
from ipd.motif.motif_manager import *

_global_motif_manager = None

def set_global_motif_manager(motif_manager):
    global _global_motif_manager
    _global_motif_manager = motif_manager

def get_global_motif_manager():
    return _global_motif_manager

set_global_motif_manager(create_motif_manager(kind='nomotif'))
