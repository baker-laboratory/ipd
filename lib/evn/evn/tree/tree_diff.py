from enum import Enum
from typing import Any, Callable, Optional, Tuple, Dict, List, Set
from collections.abc import Mapping
import evn


@evn.dispatch(dict, dict)
def diff_impl(tree1, tree2, out=print, **kw):
    differ = evn.kwcall(kw, TreeDiffer)
    diff = evn.kwcall(kw, differ.diff, tree1, tree2)
    return diff


class ShallowHandling(Enum):
    DEEP = 'deep'
    SHALLOW = 'shallow'
    SUMMARY = 'summary'


class TreeDiffStrategy:

    def is_shallow(self, path: Tuple[str, ...], val1: Any,
                   val2: Any) -> ShallowHandling:
        return ShallowHandling.DEEP

    def values_equal(self, val1: Any, val2: Any) -> bool:
        return val1 == val2

    def summarize(self, val: Any, maxlen: int = 80) -> str:
        return repr(val)[:maxlen] + ('...' if len(repr(val)) > maxlen else '')

    def should_ignore_key(self, path: Tuple[str, ...], key: str) -> bool:
        return False

    def classify_lists(self, l1: list,
                       l2: list) -> Dict[str, List[Tuple[int, int]]]:

        def is_nested(x):
            return isinstance(x, (Mapping, list))

        flat, nested = [], []
        for i, (x1, x2) in enumerate(zip(l1, l2)):
            if is_nested(x1) or is_nested(x2):
                nested.append((i, i))
            else:
                flat.append((i, i))
        return {'flat': flat, 'nested': nested}

    def compare_flat_lists(self, l1: list, l2: list) -> Tuple[list, list]:
        return sorted(set(l1) - set(l2)), sorted(set(l2) - set(l1))

    def match_lists_by_type(self, path: Tuple[str, ...], l1: list,
                            l2: list) -> Optional[str]:
        if all(isinstance(x, Mapping) for x in l1 + l2) and len(l1) == len(l2):
            return 'zip'
        return None


class TreeDiffer:

    def __init__(
        self,
        *,
        strategy: Optional[TreeDiffStrategy] = None,
        flatpaths: bool = False,
        summarize_subtrees: bool = False,
        summary_len: int = 80,
        max_depth: Optional[int] = None,
        path_fmt: Optional[Callable[[Tuple[str, ...]], str]] = None,
    ):
        self.strategy = strategy or TreeDiffStrategy()
        self.flatpaths = flatpaths
        self.summarize_subtrees = summarize_subtrees
        self.summary_len = summary_len
        self.max_depth = max_depth
        self.path_fmt = path_fmt or (lambda p: '.'.join(p))
        self._seen_pairs: Set[Tuple[int, int]] = set()

    def diff(self, cfg1: Any, cfg2: Any) -> dict:
        self._seen_pairs.clear()
        result = self._walk(cfg1, cfg2, path=())
        return self._flatten(result) if self.flatpaths else result

    def _walk(self, val1: Any, val2: Any, path: Tuple[str, ...]) -> Any:
        if self.max_depth is not None and len(path) > self.max_depth:
            return None

        id_pair = (id(val1), id(val2))
        if id_pair in self._seen_pairs:
            return None
        self._seen_pairs.add(id_pair)

        mode = self.strategy.is_shallow(path, val1, val2)
        if mode == ShallowHandling.SUMMARY:
            return (self._summarize(val1), self._summarize(val2))
        elif mode == ShallowHandling.SHALLOW:
            if not self.strategy.values_equal(val1, val2):
                return (val1, val2)
            return None

        if isinstance(val1, Mapping) and isinstance(val2, Mapping):
            all_keys = set(val1) | set(val2)
            diff = {}
            subtree1 = {}
            subtree2 = {}
            for key in sorted(all_keys):
                if self.strategy.should_ignore_key(path, key):
                    continue
                in1 = key in val1
                in2 = key in val2
                if in1 and in2:
                    sub = self._walk(val1[key], val2[key], path + (key, ))
                    if sub is not None:
                        diff[key] = sub
                elif in1:
                    subtree1[key] = val1[key]
                elif in2:
                    subtree2[key] = val2[key]
            if not diff and not subtree1 and not subtree2:
                return None
            elif not diff:
                return (self._summarize(subtree1), self._summarize(subtree2))
            else:
                node = {**diff}
                if subtree1 or subtree2:
                    node['_diff'] = (self._summarize(subtree1),
                                     self._summarize(subtree2))
                return node

        elif isinstance(val1, list) and isinstance(val2, list):
            match_mode = self.strategy.match_lists_by_type(path, val1, val2)
            out = {}
            if match_mode == 'zip':
                for i, (x1, x2) in enumerate(zip(val1, val2)):
                    d = self._walk(x1, x2, path + (f'[{i}]', ))
                    if d is not None:
                        out.setdefault('_nested', {})[i] = d
            else:
                idxs = self.strategy.classify_lists(val1, val2)

                if idxs['flat']:
                    l1 = [val1[i] for i, _ in idxs['flat']]
                    l2 = [val2[j] for _, j in idxs['flat']]
                    flat_diff = self.strategy.compare_flat_lists(l1, l2)
                    if flat_diff[0] or flat_diff[1]:
                        out['_flat'] = flat_diff

                nested_diffs = []
                for i, j in idxs['nested']:
                    d = self._walk(val1[i], val2[j], path + (f'[{i}]', ))
                    if d is not None:
                        nested_diffs.append((i, d))

                if nested_diffs:
                    out['_nested'] = dict(nested_diffs)

            return out if out else None

        if not self.strategy.values_equal(val1, val2):
            return (val1, val2)

        return None

    def _summarize(self, val: Any) -> Any:
        if self.summarize_subtrees:
            return self.strategy.summarize(val, self.summary_len)
        return val

    def _flatten(self, nested: dict) -> dict:
        flat = {}

        def walk(node: Any, path: Tuple[str, ...]):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in ('_diff', '_flat') and not isinstance(v, dict):
                        flat[self.path_fmt(path) + f':{k}'] = v
                    elif k == '_nested':
                        for idx, sub in v.items():
                            walk(sub, path + (f'[{idx}]', ))
                    else:
                        walk(v, path + (k, ))
            else:
                flat[self.path_fmt(path)] = node

        walk(nested, ())
        return flat


class SummaryStrategy(TreeDiffStrategy):

    def is_shallow(self, path, v1, v2):
        return ShallowHandling.SUMMARY


class IgnoreKeyStrategy(TreeDiffStrategy):

    def should_ignore_key(self, path, key):
        return key.startswith('_')


class IgnoreUnderscoreKeysStrategy(TreeDiffStrategy):

    def should_ignore_key(self, path, key):
        return key.startswith('_')


class FloatToleranceStrategy(TreeDiffStrategy):

    def values_equal(self, val1, val2):
        if isinstance(val1, float) and isinstance(val2, float):
            return abs(val1 - val2) < 1e-6
        return val1 == val2

    def compare_flat_lists(self, l1, l2):
        only1, only2 = [], []
        only1.extend(v for v in l1
                     if not any(self.values_equal(v, w) for w in l2))
        only2.extend(v for v in l2
                     if not any(self.values_equal(v, w) for w in l1))
        return only1, only2


class ShallowOnPathStrategy(TreeDiffStrategy):

    def is_shallow(self, path, v1, v2):
        return ShallowHandling.SHALLOW if 'meta' in path else ShallowHandling.DEEP


class SummaryOnPathStrategy(TreeDiffStrategy):

    def is_shallow(self, path, v1, v2):
        return ShallowHandling.SUMMARY if path and path[
            -1] == 'data' else ShallowHandling.DEEP
