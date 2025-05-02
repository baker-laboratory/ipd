from collections.abc import Mapping, Iterable
import difflib
import re

import rich
from rich.table import Table
from rich.console import Console

import evn

np = evn.lazyimport('numpy')

console = Console()


def print(*args, **kw):
    rich.print(*args, **kw)


def make_table(thing, precision=3, expand=False, **kw):
    kw['precision'] = precision
    kw['expand'] = expand
    with evn.np_printopts(precision=precision, suppress=True):
        # if evn.homog.is_tensor(thing): return make_table_list(thing, **kw)
        if isinstance(thing, evn.Bunch):
            return make_table_bunch(thing, **kw)
        if isinstance(thing, dict):
            return make_table_dict(thing, **kw)
        # if isinstance(thing, (list, tuple)): return make_table_list(thing, **kw)
        xr = evn.maybeimport('xarray')
        if xr and isinstance(thing, xr.Dataset):
            return make_table_dataset(thing, **kw)
        raise TypeError(f'cant make table for {type(thing)}')


def print_table(table, **kw):
    if not isinstance(table, Table):
        if table is None or not len(table):
            return '<empty table>'
        table = make_table(table, **kw)
    console.print(table)


def make_table_list(lst, title=None, header=[], **kw):
    t = evn.kwcall(kw, Table, title=title, show_header=bool(header))
    for k in header:
        evn.kwcall(kw, t.add_column, k)
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
    return [
        k for k in mapping if k[0] != '_' and k[-1] != '_' and k not in exclude
    ]


def _items(mapping, exclude=(), **kw):
    return [(k, v) for k, v in mapping.items()
            if k[0] != '_' and k[-1] != '_' and k not in exclude]


def make_table_dict_of_dict(mapping, title=None, key='key', **kw):
    assert all(isinstance(m, Mapping) for m in mapping.values())
    vals = list(mapping.values())
    assert all(_keys(v, **kw) == _keys(vals[0], **kw) for v in vals)
    t = evn.kwcall(kw, Table, title=title)
    if key:
        evn.kwcall(kw, t.add_column, to_renderable(key, **kw))
    for k in _keys(vals[0], **kw):
        evn.kwcall(kw, t.add_column, to_renderable(k, **kw))
    for k, submap in _items(mapping):
        row = [k] * bool(key) + [
            to_renderable(f, **kw) for f in submap.values()
        ]
        t.add_row(*row)
    return t


def make_table_dict_of_iter(mapping, title=None, **kw):
    vals = list(mapping.values())
    assert all(len(v) == len(vals[0]) for v in vals)
    t = evn.kwcall(kw, Table, title=title)
    for k in _keys(mapping, **kw):
        evn.kwcall(kw, t.add_column, to_renderable(k, **kw))
    for i in range(len(vals[0])):
        row = [to_renderable(v[i], **kw) for k, v in _items(mapping)]
        t.add_row(*row)
    return t


def make_table_dict_of_any(mapping, title=None, **kw):
    # vals = list(mapping.values())
    table = evn.kwcall(kw, Table, title=title)
    for k in _keys(mapping, **kw):
        evn.kwcall(kw, table.add_column, to_renderable(k, **kw))
    row = [to_renderable(v, **kw) for k, v in _items(mapping)]
    table.add_row(*row)
    return table


def make_table_dataset(dataset, title=None, **kw):
    table = evn.kwcall(kw, Table, title=title)
    cols = list(dataset.coords) + list(dataset.keys())
    for c in cols:
        evn.kwcall(kw, table.add_column, to_renderable(c, **kw))
    for nf in np.unique(dataset['nfold']):
        ds = dataset.sel(index=dataset['nfold'] == nf)
        for i in ds.index:
            row = []
            for c in cols:
                d = ds[c].data[i].round(1 if c == 'cen' else 4)
                if d.shape and d.shape[-1] == 4:
                    d = d[..., :3]
                row.append(to_renderable(d, **kw))
            table.add_row(*row)
    return table


def to_renderable(obj,
                  textmap=None,
                  strip=True,
                  nohomog=False,
                  precision=3,
                  **kw):
    textmap = textmap or {}
    if isinstance(obj, float):
        return f'{obj:7.{precision}f}'
    if isinstance(obj, bool):
        return str(obj)
    if isinstance(obj, int):
        return f'{obj:4}'
    if isinstance(obj, Table):
        return obj
    # if nohomog and evn.homog.is_tensor(obj): obj = obj[..., :3]
    s = str(evn.summary(obj))
    assert "'" not in s, s
    for pattern, replace in textmap.items():
        if '__REGEX__' in textmap and textmap['__REGEX__']:
            s = re.sub(pattern, replace, s)
        else:
            s = s.replace(pattern, str(replace))
    if strip:
        s = s.strip()
    return s


def diff(ref: str, new: str) -> None:
    # Use difflib to create a unified diff
    diff = difflib.unified_diff(ref.splitlines(),
                                new.splitlines(),
                                fromfile='Original',
                                tofile='Got',
                                lineterm='')
    return '\n'.join(diff)


def compare_multiline_strings(ref, got):
    """
    Compare two multi-line strings line by line.
    For lines that are different, show a character-level diff using SequenceMatcher.

    Parameters:
        ref (str): The ref multi-line string.
        got (str): The got multi-line string.
    """
    ref_lines = ref.splitlines()
    got_lines = got.splitlines()
    total_lines = max(len(ref_lines), len(got_lines))

    for i in range(total_lines):
        # Retrieve each line or mark as empty if one string is shorter.
        e_line = ref_lines[i] if i < len(ref_lines) else ''
        g_line = got_lines[i] if i < len(got_lines) else ''

        # If the lines are exactly equal, print that they match.
        if e_line == g_line:
            # print(f"Line {i+1}: OK")
            pass
        else:
            # print(f"Line {i+1}: DIFFERENCE")
            # Use SequenceMatcher to get a detailed diff.
            matcher = difflib.SequenceMatcher(None, e_line, g_line)
            diff_line = []
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    diff_line.append(e_line[i1:i2])
                elif tag == 'insert':
                    diff_line.append(f'[+{g_line[j1:j2]}+]')
                elif tag == 'delete':
                    diff_line.append(f'[-{e_line[i1:i2]}-]')
                elif tag == 'replace':
                    diff_line.append(f'[-{e_line[i1:i2]}-]')
                    diff_line.append(f'[+{g_line[j1:j2]}+]')
            print(f'{i:3}', ''.join(diff_line))
