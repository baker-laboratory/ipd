import abc

import ipd

_motif_managers = {}
_default_motif_manager = 'nomotif'

class MetaMotifManager(abc.ABCMeta):
    """Metaclass for MotifManager, ensures all subclasses are registered here
    even if in other modules."""
    def __init__(cls, cls_name, cls_bases, cls_dict):
        # sourcery skip: instance-method-first-arg-name
        """Register the MotifManager subclass."""
        super(MetaMotifManager, cls).__init__(cls_name, cls_bases, cls_dict)
        kind = cls.kind or cls_name  # type: ignore
        if kind in _motif_managers:
            raise TypeError(f'multiple MotifManagers with same kind!'
                            f'trying to add {kind}:{cls_name} to:\n{_motif_managers}')
        _motif_managers[kind] = cls

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.post_init()
        return instance

def set_default_motif_manager(kind):
    """Set the default motifmetry manager."""
    global _default_motif_manager
    _default_motif_manager = kind
    # ic('set_default_motif_manager', kind, _default_motif_manager)

def create_motif_manager(conf=None, extra_params=None, kind=None, device=None, **kw):
    """Create a motif manager based on the configuration.

    Args:
        conf (dict, optional): Hydra conf
        extra_params (dict, optional): extra parameters
        kind (str, optional): motifmetry manager kind
    Returns:
        MotifManager: a motif manager
    """
    global _default_motif_manager
    opt = ipd.motif.get_motif_options(conf, extra_params=extra_params)
    opt._add_params(**kw)
    opt = ipd.motif.process_motif_options(opt)
    kind = kind or opt.get(kind, None) or _default_motif_manager
    motif = _motif_managers[kind](opt, device=device)
    ipd.motif.set_global_motif_manager(motif)
    assert ipd.motif_applier is motif
    return motif
