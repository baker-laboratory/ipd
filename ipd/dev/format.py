from collections.abc import Mapping, Iterable
import re

import numpy as np
from rich.table import Table
from rich.console import Console

import ipd

console = Console()

def make_table(thing, **kw):
    npopt = np.get_printoptions()
    np.set_printoptions(precision=4, suppress=True)
    try:
        if ipd.homog.is_tensor(thing): return make_table_list(thing, **kw)
        if isinstance(thing, ipd.Bunch): return make_table_bunch(thing, **kw)
        if isinstance(thing, dict): return make_table_dict(thing, **kw)
        if isinstance(thing, (list, tuple)): return make_table_list(thing, **kw)
        xr = ipd.importornone('xarray')
        if xr and isinstance(thing, xr.Dataset): return make_table_dataset(thing, **kw)
        raise TypeError(f'cant make table for {type(thing)}')
    finally:
        np.set_printoptions(npopt['precision'], suppress=npopt['suppress'])

def print_table(thing, **kw):
    if thing is None or not len(thing): return '<empty table>'
    table = make_table(thing, **kw)
    console.print(table)

def make_table_list(lst, title=None, header=[], **kw):
    t = Table(title=title, show_header=bool(header))
    for k in header:
        ipd.kwcall(t.add_column, kw, k)
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
    if isinstance(vals[0], Mapping):
        return make_table_dict_of_dict(mapping, **kw)
    if isinstance(vals[0], Iterable) and not isinstance(vals[0], str):
        return make_table_dict_of_iter(mapping, **kw)
    return make_table_dict_of_any(mapping, **kw)

def make_table_dict_of_dict(mapping, title=None, key='key', **kw):
    vals = list(mapping.values())
    assert all(v.keys() == vals[0].keys() for v in vals)
    t = Table(title=title)
    if key: ipd.kwcall(t.add_column, kw, to_renderable(key, **kw))
    for k in vals[0].keys():
        ipd.kwcall(t.add_column, kw, k)
    for k, submap in mapping.items():
        row = [k] * bool(key) + [to_renderable(f, **kw) for f in submap.values()]
        t.add_row(*row)
    return t

def make_table_dict_of_iter(mapping, title=None, **kw):
    vals = list(mapping.values())
    assert all(len(v) == len(vals[0]) for v in vals)
    t = Table(title=title)
    for k in mapping.keys():
        ipd.kwcall(t.add_column, kw, k)
    for i in range(len(vals[0])):
        row = [to_renderable(mapping[k][i], **kw) for k in mapping]
        t.add_row(*row)
    return t

def make_table_dict_of_any(mapping, title=None, **kw):
    vals = list(mapping.values())
    t = Table(title=title)
    for k in mapping.keys():
        ipd.kwcall(t.add_column, kw, k)
    row = [to_renderable(mapping[k], **kw) for k in mapping]
    t.add_row(*row)
    return t

def make_table_dataset(dataset, title=None, **kw):
    table = Table()
    cols = list(dataset.coords) + list(dataset.keys())
    for c in cols:
        ipd.kwcall(table.add_column, kw, to_renderable(c, **kw))
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

def to_renderable(thing, textmap=None, strip=True, nohomog=False, **kw):
    textmap = textmap or {}
    if isinstance(thing, float): return f'{thing:7.3f}'
    if isinstance(thing, bool): return str(thing)
    if isinstance(thing, int): return f'{thing:4}'
    if isinstance(thing, Table): return thing
    if nohomog and ipd.homog.is_tensor(thing): thing = thing[..., :3]
    s = str(thing)
    for pattern, replace in textmap.items():
        if '__REGEX__' in textmap and textmap['__REGEX__']: s = re.sub(pattern, replace, s)
        else: s = s.replace(pattern, str(replace))
    if strip: s = s.strip()
    return s
