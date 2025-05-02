import os
import fnmatch
import json
from pathlib import Path

from typing import Union
from functools import cached_property

import click
from rich.console import Console
from rich.tree import Tree as RichTree
from rich.table import Table as RichTable

import evn

class FilePath(Path):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = self

    @cached_property
    def size(self) -> int:
        # try:
        return self.stat().st_size

    # except OSError:
    # return 0

class FileTree(dict):

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    @cached_property
    def size(self) -> int:
        return sum(v.size for v in self.values())

    def to_dict(self):
        return {
            "path": str(self.path),
            "size": self.size,
            "children": {
                k: v.to_dict() if isinstance(v, FileTree) else {
                    "path": str(v),
                    "size": v.size
                }
                for k, v in self.items()
            }
        }

def scan_dir(path: Path, opaque: list[str], follow_symlinks=False) -> Union[FileTree, FilePath]:
    name = path.name
    if any(fnmatch.fnmatch(name, pat) for pat in opaque):
        return FilePath(path)

    if not path.is_dir():
        return FilePath(path)

    tree = FileTree(path)
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    p = Path(entry.path)
                    if entry.is_symlink() and not follow_symlinks:
                        continue
                    node = scan_dir(p, opaque, follow_symlinks)
                    tree[entry.name] = node
                except Exception:
                    continue
    except Exception:
        return FilePath(path)
    return tree

def find_big(tree: Union[FileTree, FilePath], threshold: int,
             max_children: int) -> list[Union[FileTree, FilePath]]:
    if tree.size < threshold: return []
    if isinstance(tree, FilePath): return [tree]

    children = sorted(tree.items(), key=lambda x: x[1].size, reverse=True)
    if sum(c[1].size for c in children[max_children:]) < threshold:
        return [v for _, v in children[:max_children]]

    big: list[Union[FileTree, FilePath]] = [tree]
    for _, child in children:
        big.extend(find_big(child, threshold, max_children))
    return big

def print_output(results, format):
    if format == 'json':
        print(
            json.dumps([
                r.to_dict() if isinstance(r, FileTree) else {
                    "path": str(r),
                    "size": r.size
                } for r in results
            ],
                       indent=2))
    elif format == 'flat':
        for r in results:
            print(f"{r.path} ({r.size} bytes)")
    elif format == 'tree':
        console = Console()
        for r in results:
            t = RichTree(f"{r.path} ({r.size} bytes)")
            if isinstance(r, FileTree):

                def add(tree, node):
                    for k, v in node.items():
                        label = f"{k} ({v.size} bytes)"
                        if isinstance(v, FileTree):
                            subtree = tree.add(label)
                            add(subtree, v)
                        else:
                            tree.add(label)

                add(t, r)
            console.print(t)
    elif format == 'table':
        table = RichTable(title="Big Stuff")
        table.add_column("Path")
        table.add_column("Size", justify="right")
        for r in results:
            table.add_row(str(r.path), f"{r.size} bytes")
        Console().print(table)

class SizeTypeHandler(evn.cli.ClickTypeHandler):
    __supported_types__ = {str: evn.cli.MetadataPolicy.REQUIRED}

    @classmethod
    def typehint_to_click_paramtype(cls, typ, meta=None):
        if typ is str and meta == ('size', ): return SizeParamType()
        raise evn.cli.HandlerNotFoundError(f"Unsupported type: {typ} with meta: {meta}")

def parse_size(size_str: str) -> int:
    size_str = size_str.strip().upper()
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if size_str[-1] in units:
        return int(float(size_str[:-1]) * units[size_str[-1]])
    return int(size_str)

class SizeParamType:
    __name__ = "size"

    # def convert(self, value, param, ctx):
    def __call__(self, value):
        try:
            return parse_size(value)
        except Exception as e:
            raise click.BadParameter(f"Invalid size value: {value}") from e

class BigStuff(evn.cli.CLI):
    __test__ = False
    __type_handlers__ = SizeTypeHandler

    def _callback(self, debug: bool = False, follow_symlinks: bool = False):
        """
        Shared options for all commands.

        :param debug: Enable debug output.
        :param follow_symlinks: Follow symlinks instead of skipping them.
        """
        self._debug = debug
        self._follow_symlinks = follow_symlinks

    def scan(self,
             root: str,
             threshold: evn.annotype(str, 'size') = '1m',
             max_children: int = 3,
             opaque: list[str] = ['.git', '__pycache__'],
             output: str = 'flat'):
        """
        Scan a directory and report large files or folders.

        :param root: Path to scan.
        :param threshold: Minimum size to consider "big", e.g. '10M', '2G'.
        :param max_children: Max children to report instead of entire dir.
        :param opaque: Patterns (fnmatch) to treat as opaque directories.
        :param output: Output format: json, tree, table, flat.
        """
        root_path = Path(root)
        full_tree = scan_dir(root_path, opaque, self._follow_symlinks)
        big = find_big(full_tree, threshold, max_children)
        print_output(big, output)

if __name__ == '__main__':
    BigStuff._run()
