from collections.abc import Sequence

import ipd
from ipd import lazyimport

hydra = lazyimport('hydra')
omegaconf = lazyimport('omegaconf')

# these defaults mainly needed for testing
default_params = dict(
    H_K=None,
    L=None,
    Lasu=None,
    asu_input_pdb=None,
    asu_to_best_frame=None,
    asucen=[0, 0, 0],
    asurotzdeg=0,
    center_cyclic=False,
    center_lig_gp=False,
    contig_is_symmetric=False,
    contig_relabel_chains=False,
    copy_main_block_template=None,
    fit=None,
    fit_tscale=1.0,
    fit_wclash=4.0,
    helix_angle=0,
    helix_radius=0,
    helix_shift=13,
    high_t_number=1,
    input_pdb=None,
    live_center='None',
    make_guideposts_symmetric=True,
    make_ligand_symmetric=True,
    max_nsub=99,
    motif_copy_position_from_px0=False,
    motif_position='fixed',
    move_unsym_with_asu=True,
    nsub=None,
    pseudo_cycle=None,
    radius=0,
    rand_rot_lig_gp=False,
    recenter_for_diffusion=None,
    recenter_xt_chains_on_px0=None,
    rfsym_enabled=None,
    subsymid=None,
    sym_enabled=True,
    sym_redock=False,
    symcen=[0, 0, 0],
    symcen_lig_gp=[0, 0, 0],
    symid='C1',
    symmetrize_repeats=None,
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
    if opt.has('kind'):
        ipd.sym.set_default_sym_manager(opt.kind)
    if conf.sym.symid != '_training':
        opt.asu_input_pdb = conf.inference.input_pdb  # storing this in the sym manager as well for easy ref if needed
    for name, val in default_params.items():
        key = name.split('.')[-1]
        if key in opt: continue
        opt.parse_dynamic_param(key, val, overwrite=True)
    for name, val in extra_params.items():
        key = name.split('.')[-1]
        # ic(key, val)
        opt.parse_dynamic_param(key, val, overwrite=True)
    opt = process_symmetry_options(opt, **kw)
    if 'nsub' not in opt or not opt.nsub:
        if opt.symid.startswith('CYCLIC_VEE_'): opt.nsub = 2 * int(opt.symid[11:])
        elif opt.symid[0] == 'C': opt.nsub = int(opt.symid[1:])
        elif opt.symid[0] == 'D': opt.nsub = 2 * int(opt.symid[1:])
        elif opt.symid == 'I':
            opt.nsub = 60
            if opt.high_t_number > 1:
                opt.nsub = opt.nsub * opt.high_t_number
            if 'H_K' in opt and opt.H_K is not None:
                h = opt.H_K[0]
                k = opt.H_K[1]
                opt.nsub = opt.nsub * (h*h + k*k + h*k)
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
