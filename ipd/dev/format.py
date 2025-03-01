import re

import numpy as np
from rich.table import Table
from rich.console import Console

import ipd

console = Console()

def make_table(thing, **kw):
    if isinstance(thing, ipd.Bunch): return make_table_bunch(thing, **kw)
    if isinstance(thing, dict): return make_table_dict(thing, **kw)
    if isinstance(thing, (list, tuple)): return make_table_list(thing, **kw)
    raise TypeError(f'cant make table for {type(thing)}')

def print_table(thing, **kw):
    if isinstance(thing, ipd.Bunch): table = make_table_bunch(thing, **kw)
    elif isinstance(thing, dict): table = make_table_dict(thing, **kw)
    elif isinstance(thing, (list, tuple)): table = make_table_list(thing, **kw)
    else: raise TypeError(f'cant print table for {type(thing)}')
    console.print(table)

def make_table_list(lst, title=None, header=[], **kw):
    t = Table(title=title)
    for k in header:
        t.add_column(k)
    for v in lst:
        row = [to_renderable(f, **kw) for f in v]
        t.add_row(*row)
    return t

def make_table_bunch(bunch, title=None, **kw):
    return make_table_dict(bunch, title, **kw)

def make_table_dict(dic, title=None, key='key', **kw):
    t = Table(title=title)
    if key: t.add_column(to_renderable(key, **kw))
    for k in ipd.first(dic.values()).keys():
        t.add_column(k)
    for k, v in dic.items():
        row = [k] * bool(key) + [to_renderable(f, **kw) for f in v.values()]
        t.add_row(*row)
    return t

def make_table_dataset(dataset, title=None, **kw):
    table = Table()
    cols = list(dataset.coords) + list(dataset.keys())
    [table.add_column(to_renderable(c, **kw), justify='left') for c in cols]
    cols = [*cols]
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

def to_renderable(thing, textmap=None, **_):
    textmap = textmap or {}
    if isinstance(thing, float): return f'{thing:7.3f}'
    if isinstance(thing, bool): return str(thing)
    if isinstance(thing, int): return f'{thing:4}'
    if isinstance(thing, Table): return thing
    s = str(thing)
    for pattern, replace in textmap.items():
        if '__REGEX__' in textmap and textmap['__REGEX__']:
            s = re.sub(pattern, replace, s)
        else:
            s = s.replace(pattern, str(replace))
    return s
