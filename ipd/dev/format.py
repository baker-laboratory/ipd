from collections.abc import Mapping, Iterable
import re
import sys

import numpy as np
import rich
from rich.table import Table
from rich.console import Console

import ipd

console = Console()

def print(*args, **kw):
    rich.print(*args, **kw)

def make_table(thing, precision=3, **kw):
    npopt = np.get_printoptions()
    np.set_printoptions(precision=precision, suppress=True)
    try:
        if ipd.homog.is_tensor(thing): return make_table_list(thing, **kw)
        if isinstance(thing, ipd.Bunch): return make_table_bunch(thing, **kw)
        if isinstance(thing, dict): return make_table_dict(thing, **kw)
        if isinstance(thing, (list, tuple)): return make_table_list(thing, **kw)
        xr = ipd.maybeimport('xarray')
        if xr and isinstance(thing, xr.Dataset): return make_table_dataset(thing, **kw)
        raise TypeError(f'cant make table for {type(thing)}')
    finally:
        np.set_printoptions(npopt['precision'], suppress=npopt['suppress'])

def print_table(table, **kw):
    if not isinstance(table, Table):
        if table is None or not len(table): return '<empty table>'
        table = make_table(table, **kw)
    console.print(table)

def make_table_list(lst, title=None, header=[], **kw):
    t = ipd.kwcall(kw, Table, title=title, show_header=bool(header))
    for k in header:
        ipd.kwcall(kw, t.add_column, k)
    for v in lst:
        row = [to_renderable(f, **kw) for f in v]
        t.add_row(*row)
    return t

def make_table_bunch(bunch, **kw):
    return make_table_dict(bunch, **kw)

def make_table_dict(mapping, **kw):
    assert isinstance(mapping, Mapping)
    vals = list(mapping.values())
    # assert all(type(v)==type(vals[0]) for v in vals)
    try:
        if isinstance(vals[0], Mapping):
            return make_table_dict_of_dict(mapping, **kw)
        if isinstance(vals[0], Iterable) and not isinstance(vals[0], str):
            return make_table_dict_of_iter(mapping, **kw)
    except AssertionError:
        return make_table_dict_of_any(mapping, **kw)

def _keys(mapping, exclude=(), **kw):
    return [k for k in mapping if k[0] != '_' and k[-1] != '_' and k not in exclude]

def _items(mapping, exclude=(), **kw):
    return [(k, v) for k, v in mapping.items() if k[0] != '_' and k[-1] != '_' and k not in exclude]

def make_table_dict_of_dict(mapping, title=None, key='key', **kw):
    assert all(isinstance(m, Mapping) for m in mapping.values())
    vals = list(mapping.values())
    assert all(_keys(v, **kw) == _keys(vals[0], **kw) for v in vals)
    t = ipd.kwcall(kw, Table, title=title)
    if key: ipd.kwcall(kw, t.add_column, to_renderable(key, **kw))
    for k in _keys(vals[0], **kw):
        ipd.kwcall(kw, t.add_column, to_renderable(k, **kw))
    for k, submap in _items(mapping):
        row = [k] * bool(key) + [to_renderable(f, **kw) for f in submap.values()]
        t.add_row(*row)
    return t

def make_table_dict_of_iter(mapping, title=None, **kw):
    vals = list(mapping.values())
    assert all(len(v) == len(vals[0]) for v in vals)
    t = ipd.kwcall(kw, Table, title=title)
    for k in _keys(mapping, **kw):
        ipd.kwcall(kw, t.add_column, to_renderable(k, **kw))
    for i in range(len(vals[0])):
        row = [to_renderable(v[i], **kw) for k, v in _items(mapping)]
        t.add_row(*row)
    return t

def make_table_dict_of_any(mapping, title=None, **kw):
    vals = list(mapping.values())
    t = ipd.kwcall(kw, Table, title=title)
    for k in _keys(mapping, **kw):
        ipd.kwcall(kw, t.add_column, to_renderable(k, **kw))
    row = [to_renderable(v, **kw) for k, v in _items(mapping)]
    t.add_row(*row)
    return t

def make_table_dataset(dataset, title=None, **kw):
    t = ipd.kwcall(kw, Table, title=title)
    cols = list(dataset.coords) + list(dataset.keys())
    for c in cols:
        ipd.kwcall(kw, table.add_column, to_renderable(c, **kw))
    for nf in np.unique(dataset['nfold']):
        ds = dataset.sel(index=dataset['nfold'] == nf)
        for i in ds.index:
            row = list()
            for c in cols:
                d = ds[c].data[i].round(1 if c == 'cen' else 4)
                if d.shape and d.shape[-1] == 4: d = d[..., :3]
                row.append(to_renderable(d, **kw))
            table.add_row(*row)
    return table

def to_renderable(obj, textmap=None, strip=True, nohomog=False, **kw):
    textmap = textmap or {}
    if isinstance(obj, float): return f'{obj:7.3f}'
    if isinstance(obj, bool): return str(obj)
    if isinstance(obj, int): return f'{obj:4}'
    if isinstance(obj, Table): return obj
    if nohomog and ipd.homog.is_tensor(obj): obj = obj[..., :3]
    s = str(summary(obj))
    for pattern, replace in textmap.items():
        if '__REGEX__' in textmap and textmap['__REGEX__']: s = re.sub(pattern, replace, s)
        else: s = s.replace(pattern, str(replace))
    if strip: s = s.strip()
    return s

# @iterize_on_first_param(allowmap=True)
def summary(obj) -> str:
    if hasattr(obj, 'summary'): return obj.summary()
    if ipd.homog.is_tensor(obj): return ipd.homog.tensor_summary(obj)
    if (bs := sys.modules.get('biotite.structure')) and isinstance(obj, bs.AtomArray):
        return f'AtomArray({len(obj)})'
    if isinstance(obj, (list, tuple)): return [summary(o) for o in obj]
    return str(obj)
