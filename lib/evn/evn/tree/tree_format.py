import itertools
import shutil
import os

from enum import Enum
import typing as t
import evn

import enum
import evn.tree
from pprint import pprint

class Format(str, enum.Enum):
    TREE = 'tree'
    FOREST = 'forest'
    FANCY = 'fancy'
    FANCYHORIZ = 'fancyhoriz'
    SPIDER = 'spider'
    PPRINT = 'pprint'
    JSON = 'json'
    YAML = 'yaml'
    RICH = 'rich'


@evn.dispatch(dict)
def show_impl(dict_, format='forest', **kw):
    dict_ = evn.tree.sanitize(dict_)

    if not any(isinstance(d, (dict, list)) for d in dict_.values()):
        pprint(dict_)
        return

    try:
        fmt = Format(format)
    except ValueError as e:
        print(f"⚠️ Invalid format '{format}': {e}", flush=True)
        return

    # Define a mapping from Format to callable
    format_handlers = {
        Format.SPIDER: lambda: evn.tree.print_spider_tree(dict_, **kw),
        Format.FOREST: lambda: evn.tree.print_tree(dict_, **kw),
        Format.FANCY: lambda: evn.tree.print_spider_tree(dict_, mirror=False, **kw),
        Format.FANCYHORIZ: lambda: evn.tree.print_spider_tree(dict_, mirror=False, orient='horiz', **kw),
        Format.TREE: lambda: evn.tree.print_tree(dict_, max_width=1, **kw),
        Format.JSON: lambda: print(__import__('json5').dumps(dict_, indent=2)),
        Format.YAML: lambda: print(__import__('yaml').dump(evn.unbunchify(dict_))),
        Format.RICH: lambda: __import__('rich').print(evn.tree.rich_tree(dict_, **kw)),
        Format.PPRINT: lambda: pprint(dict_),
    }

    # Try the requested format first, then fall back to others
    formats_to_try = [fmt] + [f for f in Format if f != fmt]

    for f in formats_to_try:
        handler = format_handlers.get(f)
        try:
            handler()
            break
        except evn.tree.TreeFormatError:
            print(f'tree_format failed {f}, continuing...', flush=True)
            continue
        except Exception as e:
            print(f"⚠️ Error in format '{f.value}': {e}", flush=True)
            break


# class Format(str, enum.Enum):
#     TREE = 'tree'
#     FOREST = 'forest'
#     FANCY = 'fancy'
#     FANCYHORIZ = 'fancyhoriz'
#     SPIDER = 'spider'
#     PPRINT = 'pprint'
#     JSON = 'json'
#     YAML = 'yaml'
#     RICH = 'rich'


# @evn.dispatch(dict)
# def show_impl(dict_, format='forest', **kw):
#     dict_ = evn.tree.sanitize(dict_)
#     if not any(isinstance(d, (dict, list)) for d in dict_.values()):
#         import pprint
#         pprint.pprint(dict_)
#         return
#     try:
#         fmt = Format(format)
#     except ValueError as e:
#         print(f"⚠️ Invalid format '{format}': {e}", flush=True)
#         return
#     if fmt is Format.SPIDER:
#         evn.tree.print_spider_tree(dict_, **kw)
#     elif fmt is Format.FOREST:
#         evn.tree.print_tree(dict_, **kw)
#     elif fmt is Format.FANCY:
#         evn.tree.print_spider_tree(dict_, mirror=False, **kw)
#     elif fmt is Format.FANCYHORIZ:
#         evn.tree.print_spider_tree(dict_, mirror=False, orient='horiz', **kw)
#     elif fmt is Format.TREE:
#         evn.tree.print_tree(dict_, max_width=1, **kw)
#     elif fmt is Format.JSON:
#         import json5 as json
#         print(json.dumps(dict_, indent=2))
#     elif fmt is Format.YAML:
#         import yaml
#         print(yaml.dump(evn.unbunchify(dict_)))
#     elif fmt is Format.RICH:
#         import rich
#         tree = evn.tree.rich_tree(dict_, **kw)
#         rich.print(tree)
#     elif fmt is Format.PPRINT:
#         import pprint
#         pprint.pprint(dict_)


class DiffMode(str, Enum):
    COMPACT = ('compact', )
    SYMMETRIC = ('symmetric', )
    EXPLICIT = ('explicit', )


def treediff(tree1: dict[str, t.Any],
             tree2: dict[str, t.Any],
             mode=DiffMode.SYMMETRIC):
    """
    Walks two trees in parallel, applying a function to each pair of nodes.
    """
    result = evn.diff(tree1, tree2, syntax=mode)

    return result


def str_replace_multiple(text, replacements):
    translation_table = str.maketrans(replacements)
    return text.translate(translation_table)


def rich_tree(data, name='root', compact=True, style='bold green', **kw):
    """
    Create and display a Rich Tree from a nested dictionary.

    Parameters
    ----------
    data : dict or value
        The nested dictionary (or scalar value).
    name : str
        Label for the root node.
    compact : bool
        Collapse chains of single-key dicts into path strings.
    style : str
        Rich style to apply to tree labels (default: "bold green").
    """
    from rich.console import Console
    from rich.tree import Tree as RichTree
    from rich.text import Text

    console = Console()
    tree = RichTree(Text(name, style=style))

    def add_node(rich_node, obj, path=None):
        path = path or []

        if isinstance(obj, dict):
            keys = list(obj.keys())
            i = 0
            while compact and i < len(keys):
                k = keys[i]
                child = obj[k]
                if isinstance(child, dict) and len(child) == 1:
                    path.append(k)
                    obj = child
                    keys = list(obj.keys())
                    i = 0
                else:
                    break

            if path:
                collapsed = '.'.join(path + [keys[i]]) if i < len(
                    keys) else '.'.join(path)
                child_obj = obj[keys[i]] if i < len(keys) else obj
                new_node = rich_node.add(Text(collapsed, style=style))
                add_node(new_node, child_obj)
                for k in keys[i + 1:]:
                    add_node(rich_node, obj[k], path=[k])
            else:
                for k in keys:
                    add_node(rich_node, obj[k], path=[k])
        else:
            label = '.'.join(path) + f': {obj}' if path else str(obj)
            rich_node.add(Text(label, style=style))

    add_node(tree, data)
    console.print(tree)


def print_spider_tree(
    input: dict,
    name='dict',
    orient='vertical',
    width=80,
    mirror=True,
    out=print,
    **kw,
):
    from PrettyPrint import PrettyPrintTree

    if not input:
        return
    pt = PrettyPrintTree()
    if orient.startswith('vert'):
        orient = PrettyPrintTree.Vertical
    elif orient.startswith('hori'):
        orient = PrettyPrintTree.Horizontal
    else:
        raise ValueError(f'Unknown orientation {orient}')
    if orient == PrettyPrintTree.Horizontal:
        pt.print_json(input,
                      name=f'{name} at {id(input)}',
                      color='',
                      orientation=orient)
        return

    trees = []
    head, tail = [], input
    while len(tail.keys()) == 1:
        head += list(tail.keys())
        tail = tail[head[-1]]
        if not isinstance(tail, dict):
            pt.print_json(input,
                          name=f'{name} at {id(input)}',
                          color='',
                          orientation=orient)
            return
    keysleft = list(tail.keys())
    assert len(keysleft) > 1
    kw = dict(name=f'{name} at {id(input)}',
              color='',
              orientation=orient,
              return_instead_of_print=True) | kw
    while keysleft:
        last = None
        for i in range(1, len(keysleft)):
            subinput = input
            subtree_tail = subtree = dict()
            for key in head:
                subtree_tail[key] = subtree_tail = {}
                subinput = subinput[key]
            subtree_tail |= {k: subinput[k] for k in keysleft[:i]}
            tree = pt.print_json(subtree, **kw)
            j, w = None, max(len(line) for line in tree.splitlines())
            if w > width and last:
                j, tree = i, last
            elif w >= width or i + 1 == len(keysleft):
                j = i + 1
            if j is not None:
                trees.append(tree)
                keysleft = keysleft[j:]
                # print(i, j, w, keysleft)
                break
    if mirror and len(trees) > 1:
        flip_top = {
            '┌': '└',
            '┐': '┘',
            '└': '┌',
            '┘': '┐',
            '┴': '┬',
            '┬': '┴',
        }
        bar = '|'.center(width)
        for i in range(0, len(trees), 2):
            trees[i] = str_replace_multiple(trees[i], flip_top)
            top = list(reversed(trees[i].splitlines()))[:-1]
            bottom = trees[i + 1].splitlines()
            if head:
                root = bottom[0].strip().center(width)
            else:
                root = os.linesep.join(
                    [bar, bottom[0].strip().center(width), bar])
            bottom = bottom[1:]
            for k in head:
                top = top[:-2]
                bottom = bottom[2:]
                k = k.center(width)
                root = os.linesep.join([bar, k, root, k, bar])
            top[-1] = top[-1].replace('┬', '─')
            top[-1] = top[-1][:width // 2 - 1] + '┬' + top[-1][width // 2:]
            bottom[0] = bottom[0].replace('┴', '─')
            bottom[0] = bottom[0][:width // 2 - 1] + '┴' + bottom[0][width //
                                                                     2:]
            print(os.linesep.join(top + [root] + bottom))
        if len(trees) % 2:
            print('continued below>')
            print(trees[-1], flush=True)
    else:
        for tree in trees[int(bool(mirror)):-1]:
            print(tree)
            print('<continued below>')
        print(trees[-1], flush=True)


def print_tree(d,
               name='root',
               compact=True,
               max_width=None,
               style='unicode',
               **kw):
    """
    Print a nested dictionary starting at the first branching point.
    Each key under that point is rendered as a tree block in columns.

    Parameters
    ----------
    d : dict
        The nested dictionary to print.
    compact : bool
        Collapse chains of single-key dicts into 'a/b/c' form.
    max_width : int or None
        Maximum output width in characters (default: terminal width).
    style : str
        Use 'unicode' or 'ascii' line characters.
    """
    width = max_width or shutil.get_terminal_size((80, 20)).columns
    prefix_path, blocks = build_tree_blocks_from_branching_aligned(
        d, compact=compact, style=style)
    label = '.'.join(prefix_path) if prefix_path else name
    lines, colwidth = _columnize_blocks(blocks, width)
    if len(lines) == 1:
        print(lines[0])
        return
    header, wtot = label, 0
    for i, w in enumerate(colwidth[:-1]):
        wtot += w
        s = wtot - len(header)
        if s > 0:
            header += '─' * s + ('┐' if i + 2 == len(colwidth) else '┬')

    print(header)
    for line in lines:
        print(line)


def find_first_branching_path(d, path=None):
    if path is None:
        path = []
    if len(d) != 1:
        return path
    key = evn.first(d)
    if not isinstance(d[key], dict):
        return []
    return find_first_branching_path(d[key], path + [key])


def descend_to_path(d, path):
    for key in path:
        d = d[key]
    return d


def collect_blocks_at_branching_point(d):
    branching_path = find_first_branching_path(d)
    subdict = descend_to_path(d, branching_path)
    return branching_path, subdict


def _collapse_path(d, path):
    while isinstance(d, dict) and len(d) == 1:
        k, v = next(iter(d.items()))
        path.append(k)
        d = v
    return d


def build_tree_blocks_from_branching_aligned(d, compact=True, style='unicode'):
    BOX = {
        'tee': '├─ ' if style == 'unicode' else '|-- ',
        'corner': '└─ ' if style == 'unicode' else '`-- ',
        'vert': '│   ' if style == 'unicode' else '|   ',
        'space': '    ',
    }

    def recurse(node, prefix, key, islast):
        connector = BOX['corner'] if islast else BOX['tee']
        branch = prefix + connector if prefix else ''
        lines = []

        path = [key]
        value = _collapse_path(node, path) if compact else node
        path_str = '.'.join(path)

        if isinstance(value, dict):
            lines.append((branch + path_str, None))
            children = list(value.items())
            for i, (k, v) in enumerate(children):
                child_lines = recurse(
                    v, prefix + (BOX['space' if islast else 'vert']), k,
                    i == len(children) - 1)
                lines.extend(child_lines)
        else:
            lines.append((branch + path_str, str(value)))
        return lines

    path, subdict = collect_blocks_at_branching_point(d)
    # keys = list(subdict.keys())
    blocks = []

    def padkey(k):
        if len(k) > 4:
            return k
        k = k.ljust(4, '─') + '┐'
        return k

    for i, (k, v) in enumerate(subdict.items()):
        islast = False  # (i == len(keys) - 1)
        raw_lines = recurse(v, '', k, islast)
        # key_width = max(len(key) for key, val in raw_lines)
        aligned = [
            f'{key}: {val}' if val is not None else padkey(key)
            # f"{key.ljust(key_width)}: {val}" if val is not None else key
            for key, val in raw_lines
        ]
        blocks.append((aligned, islast))

    return path, blocks


def _columnize_blocks(blocks, max_width, spacing=2):
    widths = [max(len(line) for line in block) for block, _ in blocks]
    if len(blocks) == 1:
        return blocks, widths
    lens = [len(block) for block, _ in blocks]
    ncol = max_width // max(widths)
    parts = partition_balanced(lens, ncol, reorder=True)
    # print(len(parts))
    # assert 0, f'parts: {parts}'
    colwidth = [max(widths[i] for i in part) + 2 for part in parts]
    cols = [evn.addreduce([blocks[i][0] for i in part]) for part in parts]
    cols = [[line.rstrip().ljust(colwidth[i]) for line in cols[i]]
            for i in range(len(cols))]
    for c, w in zip(cols, colwidth):
        for _ in range(max(map(len, cols)) - len(c)):
            c.append(' ' * w)
    assert len({len(c) for c in cols}) == 1
    alllines = [''.join(lines) for lines in zip(*cols)]
    return alllines, colwidth


def partition_balanced(nums: list[int],
                       n_parts: int,
                       reorder: bool = True) -> list[list[int]]:
    """
    Partition a list of integers into `n_parts` sublists to balance the sums as evenly as possible.

    If `reorder` is False, the partitioning must preserve the input order and return
    contiguous, non-overlapping partitions. The output is a list of lists of indices into `nums`.

    Parameters:
        nums (list[int]): The list of integers to partition.
        n_parts (int): Number of partitions.
        reorder (bool): If True, reorder nums to optimize balance. If False, preserve original order.

    Returns:
        list[list[int]]: A list of `n_parts` sublists of indices into `nums`.
    """
    if n_parts < 2:
        return [list(range(len(nums)))]
    if n_parts > len(nums):
        return [[i] for i in range(len(nums))
                ] + [[] for _ in range(n_parts - len(nums))]

    if reorder:
        # Use greedy load balancing (same as before)
        import heapq

        indexed = sorted(enumerate(nums), key=lambda x: -x[1])
        partitions: list[list[int]] = [[] for _ in range(n_parts)]
        heap = [(0, i) for i in range(n_parts)]
        heapq.heapify(heap)

        for idx, val in indexed:
            current_sum, i = heapq.heappop(heap)
            partitions[i].append(idx)
            heapq.heappush(heap, (current_sum + val, i))

        return [list(sorted(p)) for p in partitions]

    # --- ORDER-PRESERVING PARTITIONING (Linear Partition DP) ---
    n = len(nums)
    prefix_sums = list(itertools.accumulate(nums))

    # DP tables
    cost = [[0] * n for _ in range(n_parts)]
    dividers = [[0] * n for _ in range(n_parts)]

    # Base case: 1 partition
    for i in range(n):
        cost[0][i] = prefix_sums[i]

    # Fill DP table
    for k in range(1, n_parts):
        for i in range(n):
            best_cost = float('inf')
            best_j = -1
            for j in range(k - 1, i):
                left = cost[k - 1][j]
                right = prefix_sums[i] - prefix_sums[j]
                this_cost = max(left, right)
                if this_cost < best_cost:
                    best_cost = this_cost
                    best_j = j
            cost[k][i] = best_cost
            dividers[k][i] = best_j

    # Reconstruct partition indices
    def reconstruct(dividers, n, k):
        result = []
        end = n - 1
        for part in reversed(range(1, k)):
            start = dividers[part][end] + 1
            result.append(list(range(start, end + 1)))
            end = dividers[part][end]
        result.append(list(range(0, end + 1)))
        return list(reversed(result))

    return reconstruct(dividers, n, n_parts)
