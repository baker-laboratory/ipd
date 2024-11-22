"""Symmetry checks."""
import contextlib
from typing import TYPE_CHECKING
import ipd

th = ipd.lazyimport('torch')

with contextlib.suppress(ImportError):
    import assertpy

if TYPE_CHECKING:
    import assertpy

import numpy as np

from ipd.sym import ShapeKind, ValueKind

def symcheck(sym, thing, kind=None, **kw):
    thing, kind, adaptor = get_kind_and_adaptor(sym, thing, kind)
    kw = ipd.dev.Bunch(kw)
    kw.idx = sym.idx
    kw.sym = sym
    kw.thing = thing
    kw.kind = kind
    if kind.shapekind == ShapeKind.SEQUENCE:
        return [symcheck(**kw.sub(thing=x, kind=None)) for x in adaptor.adapted]  # type: ignore
    if kind.shapekind == ShapeKind.MAPPING:
        return {k: symcheck(key=k, **kw.sub(thing=x, kind=None)) for k, x in adaptor.adapted.items()}  # type: ignore
    if kind.valuekind == ValueKind.XYZ:
        return symcheck_XYZ(**kw)
    if kind.valuekind == ValueKind.INDEX:
        return symcheck_INDEX(**kw)
    if kind.valuekind in (ValueKind.BASIC, ValueKind.PAIR):
        return symcheck_BASIC(**kw)
    assert 0, f'unknown sym kind {kind.valuekind}'

def get_kind_and_adaptor(sym, thing, kind):
    if kind is not None:
        adaptor = sym.sym_adapt(thing)
    elif isinstance(thing, ipd.sym.SimpleSparseTensor):
        adaptor = None
        kind = thing.kind
    elif isinstance(thing, ipd.sym.SymAdapt):
        adaptor = thing
        thing = adaptor.orig  # type: ignore
        kind = adaptor.kind  # type: ignore
    else:
        adaptor = sym.sym_adapt(thing)
        kind = adaptor.kind
    if isinstance(thing, (th.Tensor, np.ndarray)):
        while len(thing) == 1:  # type: ignore
            thing = thing[0]  # type: ignore
    if isinstance(thing, ipd.sym.SimpleSparseTensor):
        thing.idx = thing.idx.to(sym.device)
        thing.val = thing.val.to(sym.device)
    elif isinstance(thing, th.Tensor):
        thing = thing.to(sym.device)
    return thing, kind, adaptor

def symcheck_XYZ(*args, kind, **kw):
    'verify symmetry type XYZ'
    if kind.shapekind == ipd.sym.ShapeKind.ONEDIM:  # type: ignore
        symcheck_XYZ_1D(*args, kind=kind, **kw)
    elif kind.shapekind == ipd.sym.ShapeKind.TWODIM:  # type: ignore
        symcheck_XYZ_2D(*args, kind=kind, **kw)
    elif kind.shapekind == ipd.sym.ShapeKind.SPARSE:  # type: ignore
        symcheck_XYZ_SPARSE(*args, kind=kind, **kw)
    else:
        assert 0, f'bad ShapeKind {kind.shapekind}'

def symcheck_INDEX(*args, kind, **kw):
    'verify symmetry type INDEX'
    assert isinstance(kw['idx'], ipd.sym.SymIndex)

    if kind.shapekind == ipd.sym.ShapeKind.ONEDIM:  # type: ignore
        symcheck_INDEX_1D(*args, kind=kind, **kw)
    elif kind.shapekind == ipd.sym.ShapeKind.TWODIM:  # type: ignore
        symcheck_INDEX_2D(*args, kind=kind, **kw)
    elif kind.shapekind == ipd.sym.ShapeKind.SPARSE:  # type: ignore
        symcheck_INDEX_SPARSE(*args, kind=kind, **kw)
    else:
        assert 0, f'bad ShapeKind {kind.shapekind}'

def symcheck_BASIC(*args, kind, **kw):
    'verify symmetry type BASIC'
    if kind.shapekind == ipd.sym.ShapeKind.ONEDIM:  # type: ignore
        symcheck_BASIC_1D(*args, kind=kind, **kw)
    elif kind.shapekind == ipd.sym.ShapeKind.TWODIM:  # type: ignore
        symcheck_BASIC_2D(*args, kind=kind, **kw)
    elif kind.shapekind == ipd.sym.ShapeKind.SPARSE:  # type: ignore
        symcheck_BASIC_SPARSE(*args, kind=kind, **kw)
    else:
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
            tmp2 = ipd.h.xform(ipd.h.homog(sym.symmRs[i]), thing[s.asu])  # type: ignore
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
        s = slice(i * Lasu, (i+1) * Lasu)  # type: ignore
        assert th.all(i == idx.subnum[thing[s]])
        th.testing.assert_close(idx.idx_sub_to_asu[thing[s]], thing[:Lasu], rtol=1e-5)  # type: ignore

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
                    ic(sym.device, sym.shape, asu.shape, s, i, k)  # type: ignore
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
    frames = th.as_tensor(ipd.sym.frames(sym.symid), dtype=xyz.dtype, device=xyz.device)
    s = sym.idx
    if sym.fit or sym.asu_to_best_frame:
        rms, _, _ = ipd.h.rmsfit(xyz[s.asu], symxyz[s.asu])  # type: ignore
        if rms > 1e-3:
            print(f'ASU mismatck {rms}')
            return False
        if s.Nunsym >= 3:
            rms, _, _ = ipd.h.rmsfit(xyz[s.unsym], symxyz[s.unsym])  # type: ignore
            if rms > 1e-3:
                print(f'UNSYM mismatck {rms}')
                return False
        rms, _, _ = ipd.h.rmsfit(xyz[s.asym], symxyz[s.asym])  # type: ignore
        if rms > 1e-3:
            print(f'ASYM mismatch {rms}')
            return False
    else:
        th.testing.assert_close(xyz[sym.idx.asym], symxyz[sym.idx.asym], atol=1e-3, rtol=1e-5)
    wusym = ipd.h.xform(frames, symxyz[masks[0]])  # type: ignore
    for i in range(sym.nsub):
        a = symxyz[masks[i]]
        b = wusym[i]
        if perm_ok: b = wusym[th.argmin(ipd.h.norm((wusym - a).reshape(len(wusym), -1)))]  # type: ignore
        if not th.allclose(a, b, atol=1e-3, rtol=1e-5):
            print(masks[i].to(int))
            print(i, 'a', a)
    return True
