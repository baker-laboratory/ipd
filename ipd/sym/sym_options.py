from collections.abc import Sequence

import ipd
from ipd.lazy_import import lazyimport

hydra = lazyimport('hydra')
omegaconf = lazyimport('omegaconf')

# these defaults mainly needed for testing
default_params = dict(
    L=None,
    Lasu=None,
    asu_to_best_frame=None,
    center_cyclic=False,
    copy_main_block_template=None,
    contig_is_symmetric=False,
    contig_relabel_chains=False,
    fit=None,
    fittscale=1.0,
    fitwclash=4.0,
    guideposts_are_symmetric=False,
    high_t_number=1,
    ligand_is_symmetric=None,
    max_nsub=99,
    motif_copy_position_from_px0=False,
    motif_position='fixed',
    move_unsym_with_asu=True,
    nsub=None,
    pseudo_cycle=None,
    recenter_for_diffusion=None,
    recenter_xt_chains_on_px0=None,
    rfsym_enabled=None,
    subsymid=None,
    start_radius=0,
    sym_enabled=True,
    symid='C1',
    symmetrize_repeats=None,
    sym_input_file=None,
    sympair_enabled=None,
    sympair_method=None,
    sympair_protein_only=None,
)

def parse(s):
    if s.isdigit(): return int(s)
    if s.isnumeric(): return float(s)
    if s.lower() == 'false': return False
    if s.lower() == 'true': return True
    return s

def get_sym_options(conf=None, opt=None, extra_params=None, **kw):
    """Reads all options in conf.sym, and anything in extra_params."""
    kw = ipd.dev.Bunch(kw)
    # if conf is None:
    # with contextlib.suppress(FileNotFoundError):
    # path = '../../../../rf_diffusion/config/inference/sym.yaml'
    # cfg = omegaconf.OmegaConf.load(path)
    if extra_params is None: extra_params = {}
    if isinstance(extra_params, Sequence):
        extra_params = {v.split('=')[0].lstrip('+'): parse(v.split('=')[1]) for v in extra_params}

    opt = opt or ipd.dev.DynamicParameters(
        ndesign=resolve_option('inference.num_designs', kw, conf, 1),
        ndiffuse=resolve_option('diffuser.T', kw, conf, 1),
        nrfold=40,
    )
    if conf and 'sym' in conf:
        for key, val in conf.sym.items():
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
    opt = process_symmetry_options(opt, **kw)
    if opt.has('kind'):
        ipd.sym.set_default_sym_manager(opt.kind)
    if 'nsub' not in opt or not opt.nsub:
        if opt.symid[0] == 'C': opt.nsub = int(opt.symid[1:])  # type: ignore
        if opt.symid[0] == 'D': opt.nsub = 2 * int(opt.symid[1:])  # type: ignore
        if opt.symid == 'I':
            opt.nsub = 60
        elif opt.symid == 'O':
            opt.nsub = 24
        elif opt.symid == 'T':
            opt.nsub = 12
    return opt

def process_symmetry_options(opt, **kw):
    """Does some basic logic on opt."""
    if opt.has('symid'):
        opt.symid = opt.symid.upper()

    if opt.istrue('symmetrize_repeats'):
        assert opt.symid[0] == 'C'
        opt.nsub = opt.n_repeats

    if opt.istrue('repeat_length'):
        opt.Lasu = opt.repeat_length
        if opt.n_repeats:
            opt.L = opt.n_repeats * opt.repeat_length

    if opt.has('sym_input_pdb'):
        opt.T_xforms = ipd.sym.generate_ASU_xforms(opt.sym_input_pdb)
        opt.high_t_number = len(opt.T_xforms)
        log.info(f'HIGH T - processed T{opt.high_t_number} symmetry')  # type: ignore
    return opt

def resolve_option(name, kw, conf, default, strict=False):
    """Take from kwargs first, then conf, then use default."""
    *path, name = name.split('.')
    if name in kw:
        return kw[name]
    try:
        c = conf
        for p in path:
            c = conf[p]
        return c[name]
    except (KeyError, TypeError) as e:
        if strict and conf:
            raise e
        return default
