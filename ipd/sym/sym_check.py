'''Symmetry checks'''
import ipd

th = ipd.lazyimport('torch')
wu = ipd.lazyimport('willutil')

import numpy as np
import assertpy
from ipd.sym import SymKind, ShapeKind, ValueKind

def symcheck(sym, thing, kind=None, **kw):
    thing, kind, adaptor = get_kind_and_adaptor(sym, thing, kind)
    kw = ipd.Bunch(kw)
    kw.idx = sym.idx
    kw.sym = sym
    kw.thing = thing
    kw.kind = kind
    match kind:
        case SymKind(ShapeKind.SEQUENCE, _):
            return [symcheck(**kw.sub(thing=x, kind=None)) for x in adaptor.adapted]
        case SymKind(ShapeKind.MAPPING, _):
            return {k: symcheck(key=k, **kw.sub(thing=x, kind=None)) for k, x in adaptor.adapted.items()}
        case SymKind(_, ValueKind.XYZ):
            return symcheck_XYZ(**kw)
        case SymKind(_, ValueKind.INDEX):
            return symcheck_INDEX(**kw)
        case SymKind(_, ValueKind.BASIC | ValueKind.PAIR):
            return symcheck_BASIC(**kw)
        case _:
            assert 0, f'unknown sym kind {kind.valuekind}'

def get_kind_and_adaptor(sym, thing, kind):
    if kind is not None:
        adaptor = sym.sym_adapt(thing)
    elif isinstance(thing, ipd.sym.SimpleSparseTensor):
        adaptor = None
        kind = thing.kind
    elif isinstance(thing, ipd.sym.SymAdapt):
        adaptor = thing
        thing = adaptor.orig
        kind = adaptor.kind
    else:
        adaptor = sym.sym_adapt(thing)
        kind = adaptor.kind
    if isinstance(thing, (th.Tensor, np.ndarray)):
        while len(thing) == 1:
            thing = thing[0]
    if isinstance(thing, ipd.sym.SimpleSparseTensor):
        thing.idx = thing.idx.to(sym.device)
        thing.val = thing.val.to(sym.device)
    elif isinstance(thing, th.Tensor):
        thing = thing.to(sym.device)
    return thing, kind, adaptor

def symcheck_XYZ(*args, kind, **kw):
    'verify symmetry type XYZ'
    match kind.shapekind:
        case ipd.sym.ShapeKind.ONEDIM:
            symcheck_XYZ_1D(*args, kind=kind, **kw)
        case ipd.sym.ShapeKind.TWODIM:
            symcheck_XYZ_2D(*args, kind=kind, **kw)
        case ipd.sym.ShapeKind.SPARSE:
            symcheck_XYZ_SPARSE(*args, kind=kind, **kw)
        case _:
            assert 0, f'bad ShapeKind {kind.shapekind}'

def symcheck_INDEX(*args, kind, **kw):
    'verify symmetry type INDEX'
    assert isinstance(kw['idx'], ipd.sym.SymIndex)

    match kind.shapekind:
        case ipd.sym.ShapeKind.ONEDIM:
            symcheck_INDEX_1D(*args, kind=kind, **kw)
        case ipd.sym.ShapeKind.TWODIM:
            symcheck_INDEX_2D(*args, kind=kind, **kw)
        case ipd.sym.ShapeKind.SPARSE:
            symcheck_INDEX_SPARSE(*args, kind=kind, **kw)
        case _:
            assert 0, f'bad ShapeKind {kind.shapekind}'

def symcheck_BASIC(*args, kind, **kw):
    'verify symmetry type BASIC'
    match kind.shapekind:
        case ipd.sym.ShapeKind.ONEDIM:
            symcheck_BASIC_1D(*args, kind=kind, **kw)
        case ipd.sym.ShapeKind.TWODIM:
            symcheck_BASIC_2D(*args, kind=kind, **kw)
        case ipd.sym.ShapeKind.SPARSE:
            symcheck_BASIC_SPARSE(*args, kind=kind, **kw)
        case _:
            assert 0, f'bad ShapeKind {kind.shapekind}'

def symcheck_INDEX_common(idx, thing, **kw):
    assert th.all(thing >= 0)
    assert th.all(thing < idx.L)
    th.testing.assert_close(thing, thing.to(int).to(thing.dtype), atol=0, rtol=0)

def symcheck_XYZ_1D(sym, idx, thing, **kw):
    for s in idx:
        # ic(s)
        # ic(s.asu)
        # ic(s.beg, s.asuend)
        # ic(s.Lasu)
        for i in range(idx.nsub):
            # ic(i, thing.shape, sym.symmRs.shape, sym.symid, sym.nsub,idx.nsub)
            tmp1 = thing[s.beg + i * s.Lasu:s.asuend + i * s.Lasu]
            tmp2 = wu.h.xform(wu.h.homog(sym.symmRs[i]), thing[s.asu])
            # ic(tmp1.shape,tmp2.shape)
            th.testing.assert_close(tmp1, tmp2.to(tmp1.dtype), atol=1e-3, rtol=1e-5, equal_nan=True)

def symcheck_XYZ_2D(idx, thing, **kw):
    raise NotImplementedError('symcheck_XYZ_2D')

def symcheck_XYZ_SPARSE(idx, thing, **kw):
    assert isinstance(thing, ipd.sym.SimpleSparseTensor)
    thing, idx, isidx = thing.val, thing.idx, thing.isidx
    assert not isidx
    raise NotImplementedError('symcheck_XYZ_SPARSE')

def symcheck_INDEX_1D(idx, thing, **kw):
    symcheck_INDEX_common(idx, thing, **kw)
    assert 0
    for i in range(idx.nsub):
        s = slice(i * Lasu, (i + 1) * Lasu)
        assert th.all(i == idx.subnum[thing[s]])
        th.testing.assert_close(idx.idx_sub_to_asu[thing[s]], thing[:Lasu], rtol=1e-5)

def symcheck_INDEX_2D(idx, thing, **kw):
    raise NotImplementedError('symcheck_INDEX_2D')

def symcheck_INDEX_SPARSE(idx, thing, **kw):
    assert isinstance(idx, ipd.sym.SymIndex)
    assert isinstance(thing, ipd.sym.SimpleSparseTensor)
    thing.val = thing.val.to(idx.sub.device)
    thing.idx = thing.idx.to(idx.sub.device)
    x, _, isidx = thing.val, thing.idx, thing.isidx
    if isidx is not None and isidx is not True:
        x = x[:, thing.isidx]
    x = x.to(int)
    symcheck_INDEX_common(idx, x)
    sub = idx.subnum[x]
    asu = idx.idx_sym_to_sub[0, x[sub == 0]]
    # ic(x, sub, asu)
    for i in range(idx.nsub):
        # print(i, idx.idx_sym_to_sub[i, x[sub == i]])
        th.testing.assert_close(asu, idx.idx_sym_to_sub[i, x[sub == i]], atol=0, rtol=0)

def symcheck_BASIC_1D(idx, thing, **kw):
    assertpy.assert_that(len(thing)).is_equal_to(idx.L)
    try:
        thing = th.as_tensor(thing)
        for s in idx:
            for i in range(idx.nsub):
                assert th.all(thing[s.beg + i * s.Lasu:s.asuend + i * s.Lasu] == thing[s.asu])
    except TypeError:
        for s in idx:
            for i in range(idx.nsub):
                a = thing[int(s.beg):int(s.asuend)]
                b = thing[int(s.beg + i * s.Lasu):int(s.asuend + s.Lasu)]
                try:
                    assert all(u == v for u, v in zip(a, b))
                except TypeError:
                    assert a == b

def symcheck_BASIC_2D(idx, thing, kind, sympair_protein_only=None, **kw):
    stopslice = None
    if kind.valuekind == ValueKind.PAIR and sympair_protein_only:
        stopslice = 1
    if thing.ndim < 3:
        thing = thing[..., None]
    assertpy.assert_that(thing.shape[:2]).is_equal_to((idx.L, idx.L))
    assert not th.any(thing.isnan())
    for s in idx[:stopslice]:
        for i in range(1, idx.nsub):
            for k in range(thing.shape[-1]):
                lb, ub = s.beg + i * s.Lasu, s.asuend + i * s.Lasu
                sym = thing[lb:ub, lb:ub, ..., k]
                asu = thing[s.beg:s.asuend, s.beg:s.asuend, ..., k]
                if not th.allclose(sym, asu, atol=1e-3, rtol=1e-5):
                    ic(sym.device, sym.shape, asu.shape, s, i, k)
                    # import torchshow
                    # torchshow.show([sym, asu])
                    th.testing.assert_close(sym, asu, atol=1e-3, rtol=1e-5)

def symcheck_BASIC_SPARSE(idx, thing, **kw):
    assert isinstance(thing, ipd.sym.SimpleSparseTensor)
    thing, idx, isidx = thing.val, thing.idx, thing.isidx
    assert not isidx
    raise NotImplementedError('symcheck_BASIC_SPARSE')

def check_sym_asu(sym, xyz, symxyz, perm_ok=False, atol=1e-4):
    masks = sym.idx.sub.to(xyz.device)
    symxyz = symxyz.to(xyz.device)
    frames = th.as_tensor(wu.sym.frames(sym.symid), dtype=xyz.dtype, device=xyz.device)
    s = sym.idx
    if sym.fit or sym.asu_to_best_frame:
        rms, _, _ = wu.h.rmsfit(xyz[s.asu], symxyz[s.asu])
        if rms > 1e-3:
            print(f'ASU mismatck {rms}')
            return False
        if s.Nunsym >= 3:
            rms, _, _ = wu.h.rmsfit(xyz[s.unsym], symxyz[s.unsym])
            if rms > 1e-3:
                print(f'UNSYM mismatck {rms}')
                return False
        rms, _, _ = wu.h.rmsfit(xyz[s.asym], symxyz[s.asym])
        if rms > 1e-3:
            print(f'ASYM mismatch {rms}')
            return False
    else:
        th.testing.assert_close(xyz[sym.idx.asym], symxyz[sym.idx.asym], atol=1e-3, rtol=1e-5)
    wusym = wu.h.xform(frames, symxyz[masks[0]])
    for i in range(sym.nsub):
        a = symxyz[masks[i]]
        b = wusym[i]
        if perm_ok: b = wusym[th.argmin(wu.h.norm((wusym - a).reshape(len(wusym), -1)))]
        if not th.allclose(a, b, atol=1e-3, rtol=1e-5):
            print(masks[i].to(int))
            print(i, 'a', a)
    return True
