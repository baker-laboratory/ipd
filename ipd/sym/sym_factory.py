import abc

import ipd

_sym_managers = {}
_default_sym_manager = 'base'

class MetaSymManager(abc.ABCMeta):
    """Metaclass for SymmetryManager, ensures all subclasses are registered
    here even if in other modules."""
    def __init__(cls, cls_name, cls_bases, cls_dict):
        # sourcery skip: instance-method-first-arg-name
        """Register the SymmetryManager subclass."""
        super(MetaSymManager, cls).__init__(cls_name, cls_bases, cls_dict)
        kind = cls.kind or cls_name  # type: ignore
        from ipd.sym.sym_factory import _sym_managers
        if kind in _sym_managers:
            raise TypeError(f'multiple SymmetryManagers with same kind!'
                            f'trying to add {kind}:{cls_name} to:\n{_sym_managers}')
        _sym_managers[kind] = cls

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.post_init()
        return instance

def set_default_sym_manager(kind):
    """Set the default symmetry manager."""
    global _default_sym_manager
    _default_sym_manager = kind
    # ic('set_default_sym_manager', kind, _default_sym_manager)

def create_sym_manager(conf=None, extra_params=None, kind=None, device=None, **kw):
    """Create a symmetry manager based on the configuration.

    Args:
        conf (dict, optional): Hydra conf
        extra_params (dict, optional): extra parameters
        kind (str, optional): symmetry manager kind
    Returns:
        SymmetryManager: a symmetry manager
    """
    global _default_sym_manager
    opt = ipd.sym.get_sym_options(conf, extra_params=extra_params)
    opt._add_params(**kw)
    opt = ipd.sym.process_symmetry_options(opt)
    kind = kind or opt.get(kind, None) or _default_sym_manager
    if kind == 'input_defined': opt.symid = 'input_defined'
    elif opt.symid == 'C1': kind = 'C1'
    sym = _sym_managers[kind](opt, device=device)
    ipd.sym.set_global_symmetry(sym)
    assert ipd.symmetrize is sym
    return sym
