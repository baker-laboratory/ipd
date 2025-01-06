import contextlib
from collections.abc import Sequence

import ipd
from ipd.lazy_import import lazyimport
from ipd.sym.sym_options import resolve_option

hydra = lazyimport('hydra')
omegaconf = lazyimport('omegaconf')

default_params = dict(motif_pdb='motif.pdb', )

def parse(s):
    if s.isdigit(): return int(s)
    if s.isnumeric(): return float(s)
    if s.lower() == 'false': return False
    if s.lower() == 'true': return True
    return s

def get_motif_options(conf=None, opt=None, extra_params=None, **kw):
    """Reads all options in conf.motif, and anything in extra_params."""
    kw = ipd.dev.Bunch(kw)
    with contextlib.suppress(FileNotFoundError):
        path = '../../../../rf_diffusion/config/inference/motif.yaml'
        conf = conf or omegaconf.OmegaConf.load(path)

    extra_params = extra_params or {}
    if isinstance(extra_params, Sequence):
        extra_params = {v.split('=')[0].lstrip('+'): parse(v.split('=')[1]) for v in extra_params}

    opt = opt or ipd.dev.DynamicParameters(
        ndesign=resolve_option('inference.num_designs', kw, conf, 1),
        ndiffuse=resolve_option('diffuser.T', kw, conf, 1),
        nrfold=40,
    )
    if conf and 'motif' in conf:
        for key, val in conf.motif.items():
            opt.parse_dynamic_param(key, val)
    # ic(extra_params)
    for name, val in default_params.items():
        key = name.split('.')[-1]
        if key in opt: continue
        opt.parse_dynamic_param(key, val, overwrite=True)
    for name, val in extra_params.items():
        key = name.split('.')[-1]
        # ic(key, val)
        opt.parse_dynamic_param(key, val, overwrite=True)
    opt = process_motif_options(opt, **kw)
    if opt.has('kind'):
        ipd.motif.set_default_motif_manager(opt.kind)
    return opt

def process_motif_options(opt, **kw):
    """Does some basic logic on opt."""
    ...
    return opt
