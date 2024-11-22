import functools

import hypothesis
from hypothesis import strategies as st

import ipd
from ipd.lazy_import import lazyimport

hydra = lazyimport('hydra')

def hydra_sandbox(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        hydra_instance = None
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra_instance = hydra.core.global_hydra.GlobalHydra().instance()
        hydra.core.global_hydra.GlobalHydra().clear()
        result = func(*args, **kwargs)
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()
        if hydra_instance:
            hydra.core.global_hydra.GlobalHydra.set_instance(hydra_instance)
        return result

    return wrapper

@hydra_sandbox
def construct_conf_symtest(overrides=[]):
    hydra.core.global_hydra.GlobalHydra().clear()
    hydra.initialize(version_base=None, config_path='../../config', job_name='test_app')
    return hydra.compose(
        config_name='sym_test.yaml',
        overrides=overrides,
        return_hydra_config=True,
    )

@functools.lru_cache
def _create_test_sym_manager(extras=(), conf=None, **kw):
    conf = conf or construct_conf_symtest(extras)
    sym = ipd.sym.create_sym_manager(None, extras, device='cpu', **kw)
    assert sym.device == 'cpu'
    return sym

def create_test_sym_manager(extras=(), conf=None, idx=None, **kw):
    if isinstance(extras, dict): extras = tuple(f'{k}={v}' for k, v in extras.items())
    sym = _create_test_sym_manager(extras=tuple(extras), **kw)
    sym.reset()
    if idx: sym.idx = idx
    return sym

@st.composite
def symslices(draw, L=None, Lmin=20, Lmax=100, maxslice=None, bad=False, raw=False):
    L = L or draw(st.integers(Lmin, Lmax))
    maxslice = maxslice or draw(st.integers(1, 10)) * 2
    ranges = draw(st.sets(st.integers(0, L), min_size=2, max_size=maxslice).filter(lambda x: len(x) % 2 == 0))
    nsub = draw(st.integers(1, 5))
    good = list()
    overlaps = list()
    badsym = list()
    itr = iter(sorted(ranges))
    for lb, ub in zip(itr, itr):
        if good and good[-1][-1] > lb:
            overlaps.append((L, lb, ub))
            continue
        while (ub-lb) % nsub:
            if len(badsym) == len(good): badsym.append((L, lb, ub))
            ub += 1
        if ub > L: break
        good.append((L, lb, ub))
        overlaps.append((L, lb, ub))
        if len(badsym) < len(good): badsym.append((L, lb, ub))
    if not good: hypothesis.assume(False)

    if bad:
        olapok = len(overlaps) > len(good)
        symok = any((ub-lb) % nsub for _, lb, ub in badsym)
        if olapok and symok:
            if draw(st.integers(0, 1)): return nsub, overlaps
            else: return nsub, badsym
        elif olapok: return nsub, overlaps
        elif symok: return nsub, badsym
        else: hypothesis.assume(False)

    if raw: return nsub, good
    return ipd.sym.SymIndex(nsub, good)

@st.composite
def sym_manager(draw, *a, **kw):
    slices = draw(symslices(*a, **kw))
    sym = create_test_sym_manager([f'sym.symid=C{slices.nsub}'])
    sym.idx = slices
    # assert sym.device=='cuda'
    # assert sym._symmRs.device.type=='cuda'
    return sym
